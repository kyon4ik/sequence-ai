use super::*;
use sequence_agent::{
    Agent, AgentConfig, AgentDescriptor, AgentFactory, AgentSpec, InferenceDevice, mc::McAgent,
};
use sequence_runner::Replay;
use sequence_runner::{MatchRunner, Participant};

#[test]
fn extended_budget_preserves_exploration_and_rejects_learning_changes() {
    let saved = TrainingConfig {
        iterations: 200,
        ..Default::default()
    };
    let mut extended = TrainingConfig {
        iterations: 1000,
        epsilon_decay_iterations: Some(200),
        ..saved.clone()
    };
    assert!(resume_compatible(&saved, &extended));
    for i in 0..200 {
        assert_eq!(saved.epsilon_at(i), extended.epsilon_at(i));
    }
    assert!((extended.epsilon_at(999) - saved.epsilon_end).abs() < 1e-12);
    extended.epsilon_decay_iterations = None;
    assert!(!resume_compatible(&saved, &extended));
    extended.epsilon_decay_iterations = Some(200);
    extended.learning_rate *= 2.;
    assert!(!resume_compatible(&saved, &extended));
    extended = saved.clone();
    extended.iterations = 100;
    assert!(!resume_compatible(&saved, &extended));
    extended.epsilon_decay_iterations = Some(0);
    assert!(extended.validate().is_err());
}

#[test]
fn resumed_training_matches_uninterrupted_updates() {
    assert_resumed_updates(Device::flex());
}

#[test]
#[ignore = "requires an accessible NVIDIA GPU; run outside the sandbox"]
fn cuda_resumed_training_matches_uninterrupted_updates() {
    assert_resumed_updates(Device::cuda(0));
}

fn assert_resumed_updates(device: Device) {
    device.seed(91);
    let config = TrainingConfig {
        iterations: 2,
        games_per_iteration: 3,
        inference_batch_games: 2,
        batch_size: 128,
        epsilon_start: 1.,
        epsilon_end: 1.,
        evaluation_games: 2,
        ..Default::default()
    };
    let directory = tempfile::tempdir().unwrap();
    let saved = directory.path().join("iteration-1");
    let continuous = train(
        ValueNet::new(&device),
        device.clone(),
        &config,
        None,
        |checkpoint| {
            if checkpoint.report().iterations.len() == 1 {
                checkpoint.save(&saved)?;
            }
            Ok(())
        },
    )
    .unwrap();
    let mut incomplete = serde_json::to_value(&continuous.report.iterations[0]).unwrap();
    incomplete
        .as_object_mut()
        .unwrap()
        .remove("rollout_inference");
    assert!(serde_json::from_value::<IterationMetrics>(incomplete).is_err());
    // A fresh model has fresh parameter IDs; restoring must reconnect Adam state too.
    let mut resume_config = config.clone();
    resume_config.inference_batch_games = 3;
    let resumed = train(
        ValueNet::new(&device),
        device.clone(),
        &resume_config,
        Some(&saved),
        |_| Ok(()),
    )
    .unwrap();
    let a = &continuous.report.iterations[1];
    let b = &resumed.report.iterations[1];
    assert_eq!(a.samples, b.samples);
    assert_eq!(a.outcomes, b.outcomes);
    assert_eq!(a.epsilon, b.epsilon);
    assert!((a.mse - b.mse).abs() < 1e-6, "{} != {}", a.mse, b.mse);
    let game = Simple::from_seed(777);
    let mut actions = Vec::new();
    game.legal_actions(&mut actions);
    let inputs: Vec<_> = actions
        .iter()
        .map(|&a| EncodedAfterstate::new(game.observation(), a).unwrap())
        .collect();
    let predict = |model: &ValueNet| {
        let (board, context) = batch(&inputs, &device);
        model
            .forward(board, context)
            .try_into_data()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap()
    };
    for (a, b) in predict(&continuous.model)
        .iter()
        .zip(predict(&resumed.model))
    {
        assert!((a - b).abs() < 1e-6, "{a} != {b}");
    }
    let mut changed = config.clone();
    changed.learning_rate *= 2.;
    assert!(
        train(
            ValueNet::new(&device),
            device,
            &changed,
            Some(&saved),
            |_| Ok(())
        )
        .is_err()
    );
}

#[test]
fn targets_use_the_actor_perspective() {
    for winner in [Color::Red, Color::Blue] {
        assert_eq!(terminal_target(GameOutcome::Won(winner), winner), 1.);
        assert_eq!(
            terminal_target(GameOutcome::Won(winner), winner.next()),
            -1.
        );
        assert_eq!(terminal_target(GameOutcome::Draw, winner), 0.);
    }
}

#[test]
fn config_defaults_and_invalid_batching_is_rejected() {
    let mut config: TrainingConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(config, TrainingConfig::default());
    config.inference_batch_games = 0;
    assert!(config.validate().is_err());
}

#[test]
fn batching_preserves_episode_rng_order_rewards_and_replays() {
    let device = Device::flex();
    let net = ValueNet::new(&device);
    let configs: Vec<_> = (0..7)
        .map(|seed| MatchConfig {
            game_seed: seed,
            agent_seeds: [seed + 100, seed + 200],
            ..Default::default()
        })
        .collect();
    let sequential = rollout::play(&net, &device, &configs, 1, 1., None).unwrap();
    let batched = rollout::play(&net, &device, &configs, 3, 1., None).unwrap();
    assert_eq!(batched.inference.batches, 0);
    for ((expected, actual), config) in sequential
        .records
        .iter()
        .zip(&batched.records)
        .zip(&configs)
    {
        let mut red = McAgent::new(net.clone(), device.clone(), 0, 1.);
        let mut blue = McAgent::new(net.clone(), device.clone(), 0, 1.);
        let scalar = MatchRunner::<Simple>::default()
            .run(
                config.clone(),
                [
                    Participant::new(AgentDescriptor::new("self-play", "MC"), &mut red),
                    Participant::new(AgentDescriptor::new("self-play", "MC"), &mut blue),
                ],
            )
            .unwrap();
        for reference in [expected, &scalar] {
            assert_eq!(actual.header, reference.header);
            assert_eq!(actual.termination, reference.termination);
            assert_eq!(
                actual
                    .steps
                    .iter()
                    .map(|s| (s.action, s.rewards, &s.state))
                    .collect::<Vec<_>>(),
                reference
                    .steps
                    .iter()
                    .map(|s| (s.action, s.rewards, &s.state))
                    .collect::<Vec<_>>()
            );
        }
        Replay::<Simple>::new(actual).unwrap();
    }
}

#[test]
fn batched_baselines_preserve_seats_seeds_and_action_limits() {
    let device = Device::flex();
    let net = ValueNet::new(&device);
    let configs: Vec<_> = (0..5)
        .map(|seed| MatchConfig {
            game_seed: seed / 2,
            agent_seeds: [seed + 50, seed + 100],
            max_actions: 3,
        })
        .collect();
    for baseline in [AgentConfig::Random, AgentConfig::greedy()] {
        let output = rollout::play(&net, &device, &configs, 3, 1., Some(&baseline)).unwrap();
        for (index, (config, actual)) in configs.iter().zip(output.records).enumerate() {
            let mut policy = McAgent::new(net.clone(), device.clone(), 0, 1.);
            let mut opponent = baseline.create(0, InferenceDevice::Cpu).unwrap();
            let candidate =
                Participant::new(AgentDescriptor::new("candidate", "Candidate"), &mut policy);
            let other = Participant::new(
                AgentDescriptor::new("baseline", "Baseline"),
                opponent.as_mut(),
            );
            let players = if index % 2 == 0 {
                [candidate, other]
            } else {
                [other, candidate]
            };
            let expected = MatchRunner::<Simple>::default()
                .run(config.clone(), players)
                .unwrap();
            assert_eq!(actual.header, expected.header);
            assert_eq!(actual.termination, expected.termination);
            assert_eq!(
                actual
                    .steps
                    .iter()
                    .map(|s| (s.actor, s.action, s.rewards, &s.state))
                    .collect::<Vec<_>>(),
                expected
                    .steps
                    .iter()
                    .map(|s| (s.actor, s.action, s.rewards, &s.state))
                    .collect::<Vec<_>>()
            );
            Replay::<Simple>::new(&actual).unwrap();
        }
    }
}

#[test]
#[ignore = "GPU throughput benchmark; run outside the sandbox with --nocapture"]
fn cuda_rollout_throughput() {
    let device = Device::cuda(0);
    device.seed(42);
    let mut net = ValueNet::new(&device);
    if let Ok(path) = std::env::var("SEQUENCE_BENCH_CHECKPOINT") {
        net = net.try_load_file(path).unwrap();
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(771);
    let configs: Vec<_> = (0..128)
        .map(|_| MatchConfig {
            game_seed: rng.random(),
            agent_seeds: [rng.random(), rng.random()],
            ..Default::default()
        })
        .collect();
    let batch_games: Vec<usize> = std::env::var("SEQUENCE_BENCH_BATCH_GAMES")
        .unwrap_or_else(|_| "1,8,16,32,64,128".into())
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse()
                .expect("benchmark batch sizes must be integers")
        })
        .collect();
    assert!(batch_games.iter().all(|&size| size > 0));
    for &size in &batch_games {
        // Use the full workload so batches larger than 32 are warmed too.
        rollout::play(&net, &device, &configs, size, 0.175, None).unwrap();
        evaluate(&net, &device, 100, size).unwrap();
    }
    // Alternate batch sizes between repetitions to reduce ordering bias.
    for repetition in 0..5 {
        for &batch_games in &batch_games {
            let started = Instant::now();
            let output = rollout::play(&net, &device, &configs, batch_games, 0.175, None).unwrap();
            let elapsed = started.elapsed().as_secs_f64();
            for record in &output.records {
                Replay::<Simple>::new(record).unwrap();
            }
            println!(
                "{}",
                serde_json::json!({"phase": "rollout", "batch_games": batch_games, "repetition": repetition, "seconds": elapsed, "games_per_second": 128. / elapsed, "actions": output.records.iter().map(|r| r.steps.len()).sum::<usize>(), "inference": output.inference})
            );
            let started = Instant::now();
            let (score, inference) = evaluate(&net, &device, 100, batch_games).unwrap();
            println!(
                "{}",
                serde_json::json!({"phase": "evaluation", "batch_games": batch_games, "repetition": repetition, "seconds": started.elapsed().as_secs_f64(), "score": score, "inference": inference})
            );
        }
    }
}

#[test]
fn completed_game_updates_weights_and_checkpoint_roundtrips() {
    let device = Device::flex();
    device.seed(7);
    let net = ValueNet::new(&device);
    let game = Simple::from_seed(7);
    let mut actions = Vec::new();
    game.legal_actions(&mut actions);
    let input = EncodedAfterstate::new(game.observation(), actions[0]).unwrap();
    let predict = |net: &ValueNet| {
        let (board, context) = batch(std::slice::from_ref(&input), &device);
        net.forward(board, context).into_scalar::<f32>()
    };
    let before = predict(&net);
    let output = train(
        net,
        device.clone(),
        &TrainingConfig {
            iterations: 1,
            evaluation_games: 2,
            games_per_iteration: 1,
            batch_size: 128,
            epsilon_start: 1.,
            epsilon_end: 1.,
            seed: 7,
            ..Default::default()
        },
        None,
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(output.report.iterations.len(), 1);
    assert_eq!(output.report.iterations[0].episodes, 1);
    assert!(output.report.iterations[0].mse.is_finite());
    let net = output.model;
    let after = predict(&net);
    assert!(after.is_finite() && (-1.0..=1.0).contains(&after));
    assert!((before - after).abs() > 1e-6);
    let path = std::env::temp_dir().join(format!("sequence-mc-test-{}.bpk", std::process::id()));
    net.clone().save_file(&path).unwrap();
    let loaded = ValueNet::new(&device).load_file(&path);
    assert!((predict(&loaded) - after).abs() < 1e-6);
    let mut bot = McAgent::new(loaded, device.clone(), 7, 0.0);
    let selected = bot.choose_action(game.observation(), &actions).unwrap();
    assert!(actions.contains(&selected));
    let spec = AgentSpec {
        descriptor: AgentDescriptor::new("trained", "Trained MC"),
        config: AgentConfig::MonteCarlo {
            checkpoint: path.clone(),
        },
    };
    let mut factory_bot = spec.create(7).unwrap();
    assert_eq!(
        factory_bot
            .choose_action(game.observation(), &actions)
            .unwrap(),
        selected
    );
    // Exercise the same factory path used by a future tournament scheduler.
    let mut runner = MatchRunner::<Simple>::default();
    let random = AgentSpec {
        descriptor: AgentDescriptor::new("random", "Random"),
        config: AgentConfig::Random,
    };
    let record = runner
        .run_factories(
            MatchConfig {
                max_actions: 2,
                ..Default::default()
            },
            [&spec.on_device(InferenceDevice::Cpu), &random],
        )
        .unwrap();
    assert_eq!(record.steps.len(), 2);
    Replay::<Simple>::new(&record).unwrap();
    std::fs::remove_file(path).unwrap();
}

#[test]
#[ignore = "requires an accessible NVIDIA GPU; run outside the sandbox"]
fn cuda_training_checkpoint_and_match() {
    let device = Device::cuda(0);
    device.seed(17);
    let output = train(
        ValueNet::new(&device),
        device.clone(),
        &TrainingConfig {
            iterations: 1,
            evaluation_games: 2,
            games_per_iteration: 1,
            batch_size: 32,
            epsilon_start: 1.,
            epsilon_end: 1.,
            seed: 17,
            ..Default::default()
        },
        None,
        |_| Ok(()),
    )
    .unwrap();
    assert!(output.report.iterations[0].mse.is_finite());
    let path = std::env::temp_dir().join(format!("sequence-cuda-test-{}.bpk", std::process::id()));
    output.model.save_file(&path).unwrap();
    let spec = AgentSpec {
        descriptor: AgentDescriptor::new("cuda", "CUDA MC"),
        config: AgentConfig::MonteCarlo {
            checkpoint: path.clone(),
        },
    };
    let random = AgentSpec {
        descriptor: AgentDescriptor::new("random", "Random"),
        config: AgentConfig::Random,
    };
    let record = MatchRunner::<Simple>::default()
        .run_factories(
            MatchConfig {
                max_actions: 2,
                ..Default::default()
            },
            [&spec.on_device(InferenceDevice::Cuda(0)), &random],
        )
        .unwrap();
    Replay::<Simple>::new(&record).unwrap();
    std::fs::remove_file(path).unwrap();
}
