use super::*;
use opponents::OpponentPlan;
use sequence_agent::{AgentConfig, InferenceDevice, mc::McAgent};
use sequence_runner::{MatchRunner, Participant, Replay};

#[test]
fn mixed_rollout_matches_scalar_policies_and_restores_order_across_waves() {
    assert_mixed_rollout(Device::flex());
}

#[test]
#[ignore = "requires host NVIDIA GPU access; run outside the sandbox"]
fn cuda_mixed_rollout_matches_scalar_policies() {
    assert_mixed_rollout(Device::cuda(0));
}

fn assert_mixed_rollout(device: Device) {
    device.seed(719);
    let net = ValueNet::new(&device);
    let mut pool = CheckpointPool::new().unwrap();
    for iteration in 0..6 {
        pool.insert(iteration, &ValueNet::new(&device)).unwrap();
    }
    // Six distinct checkpoints force more than one four-model wave. Both seats
    // are exercised with learner epsilon=1 and frozen-opponent epsilon=0.
    let policies = [OpponentPolicy::SelfPlay, OpponentPolicy::Greedy]
        .into_iter()
        .chain((0..6).map(OpponentPolicy::Checkpoint));
    let plans: Vec<_> = policies
        .flat_map(|policy| {
            (0..2).map(move |learner_seat| OpponentPlan {
                historical_sample: None,
                source: OpponentKind::RandomCheckpoint,
                policy,
                learner_seat,
            })
        })
        .collect();
    let configs: Vec<_> = (0..plans.len())
        .map(|index| MatchConfig {
            game_seed: index as u64,
            agent_seeds: [100 + index as u64, 200 + index as u64],
            max_actions: 4,
        })
        .collect();
    let started = Instant::now();
    let output =
        rollout::play_opponents(&net, &device, &configs, 3, 1., &plans, Some(&pool)).unwrap();
    let elapsed = started.elapsed();
    assert!(output.inference.batches > 0);
    for ((plan, config), record) in plans.iter().zip(&configs).zip(&output.records) {
        let mut learner = McAgent::new(net.clone(), device.clone(), 0, 1.);
        let mut opponent: Box<dyn sequence_agent::Agent> = match plan.policy {
            OpponentPolicy::SelfPlay => Box::new(McAgent::new(net.clone(), device.clone(), 0, 1.)),
            OpponentPolicy::Greedy => AgentConfig::greedy()
                .create(0, InferenceDevice::Cpu)
                .unwrap(),
            OpponentPolicy::Checkpoint(iteration) => Box::new(McAgent::new(
                net.clone()
                    .valid()
                    .try_load_file(pool.path(iteration))
                    .unwrap(),
                device.clone(),
                0,
                0.,
            )),
        };
        let learner_seat = plan.learner_seat;
        let learner = Participant::new(record.header.players[learner_seat].clone(), &mut learner);
        let opponent = Participant::new(
            record.header.players[1 - learner_seat].clone(),
            opponent.as_mut(),
        );
        let seats = if learner_seat == 0 {
            [learner, opponent]
        } else {
            [opponent, learner]
        };
        let expected = MatchRunner::<Simple>::default()
            .run(config.clone(), seats)
            .unwrap();
        assert_eq!(record.header, expected.header);
        assert_eq!(record.termination, expected.termination);
        assert_eq!(
            record
                .steps
                .iter()
                .map(|s| (s.actor, s.action))
                .collect::<Vec<_>>(),
            expected
                .steps
                .iter()
                .map(|s| (s.actor, s.action))
                .collect::<Vec<_>>()
        );
        Replay::<Simple>::new(record).unwrap();
        let learned = record
            .steps
            .iter()
            .filter(|step| plan.learns_from(step.actor as usize))
            .count();
        assert_eq!(
            learned,
            if plan.policy == OpponentPolicy::SelfPlay {
                4
            } else {
                2
            }
        );
    }
    println!(
        "mixed rollout: {} games, {:.3}s, {} batches, {} candidates",
        configs.len(),
        elapsed.as_secs_f64(),
        output.inference.batches,
        output.inference.candidates
    );
}

#[test]
#[ignore = "GPU throughput benchmark; run outside the sandbox with --nocapture"]
fn cuda_mixed_rollout_throughput() {
    let device = Device::cuda(0);
    device.seed(177);
    let net = ValueNet::new(&device);
    let mut pool = CheckpointPool::new().unwrap();
    for iteration in 0..6 {
        pool.insert(iteration, &ValueNet::new(&device)).unwrap();
    }
    let mix = OpponentMix {
        self_play: 40.,
        best_checkpoint: 20.,
        previous_checkpoint: 15.,
        random_checkpoint: 15.,
        greedy: 10.,
    };
    let mut state = OpponentState::new(177);
    state.checkpoints = (0..6).collect();
    state.best_iteration = Some(3);
    let mixed = mix.schedule(128, 6, &mut state);
    let self_play = OpponentMix::default().schedule(128, 6, &mut state);
    let configs: Vec<_> = (0..128)
        .map(|index| MatchConfig {
            game_seed: 9000 + index,
            agent_seeds: [10_000 + index, 20_000 + index],
            ..Default::default()
        })
        .collect();
    for plans in [&self_play, &mixed] {
        rollout::play_opponents(&net, &device, &configs, 128, 0.175, plans, Some(&pool)).unwrap();
    }
    for repetition in 0..3 {
        for (name, plans) in [("self_play", &self_play), ("mixed", &mixed)] {
            let started = Instant::now();
            let output =
                rollout::play_opponents(&net, &device, &configs, 128, 0.175, plans, Some(&pool))
                    .unwrap();
            let seconds = started.elapsed().as_secs_f64();
            for record in &output.records {
                assert!(matches!(record.termination, Termination::Completed(_)));
                Replay::<Simple>::new(record).unwrap();
            }
            println!(
                "{}",
                serde_json::json!({"opponents": name, "repetition": repetition, "games": 128, "seconds": seconds, "games_per_second": 128. / seconds, "inference": output.inference})
            );
        }
    }
}

#[test]
fn mixed_training_resumes_from_a_moved_self_contained_snapshot() {
    assert_mixed_resume(Device::flex(), false);
}

#[test]
fn fixed_budget_opponent_sampling_resumes_with_exact_update_counts() {
    assert_mixed_resume(Device::flex(), true);
}

#[test]
#[ignore = "requires host NVIDIA GPU access; run outside the sandbox"]
fn cuda_mixed_training_resumes() {
    assert_mixed_resume(Device::cuda(0), false);
}

#[test]
#[ignore = "requires host NVIDIA GPU access; run outside the sandbox"]
fn cuda_fixed_budget_opponent_sampling_resumes() {
    assert_mixed_resume(Device::cuda(0), true);
}

#[test]
#[ignore = "requires host NVIDIA GPU access; run outside the sandbox"]
fn cuda_adaptive_training_resumes() {
    assert_mixed_resume_with(Device::cuda(0), false, true);
}

#[test]
fn adaptive_training_resumes() {
    assert_mixed_resume_with(Device::flex(), false, true);
}

fn assert_mixed_resume(device: Device, balanced: bool) {
    assert_mixed_resume_with(device, balanced, false);
}

fn assert_mixed_resume_with(device: Device, balanced: bool, adaptive: bool) {
    device.seed(813);
    let config = TrainingConfig {
        historical_sampling: HistoricalSampling {
            adaptive,
            ..Default::default()
        },
        iterations: 3,
        games_per_iteration: 5,
        inference_batch_games: 3,
        batch_size: 256,
        evaluation_interval: 1,
        evaluation_games: 2,
        epsilon_start: 1.,
        epsilon_end: 1.,
        sampling: if balanced {
            SamplingStrategy::OpponentMix
        } else {
            SamplingStrategy::Natural
        },
        updates_per_iteration: if balanced { Some(3) } else { None },
        opponents: OpponentMix {
            self_play: 1.,
            best_checkpoint: 1.,
            previous_checkpoint: 1.,
            random_checkpoint: 1.,
            greedy: 1.,
        },
        ..Default::default()
    };
    let original = tempfile::tempdir().unwrap();
    let snapshot = original.path().join("iteration-2");
    let output = train(
        ValueNet::new(&device),
        device.clone(),
        &config,
        None,
        |state| {
            if state.report().iterations.len() == 2 {
                state.save(&snapshot)?;
            }
            Ok(())
        },
    )
    .unwrap();
    let moved = tempfile::tempdir().unwrap();
    let snapshot_moved = moved.path().join("snapshot");
    std::fs::rename(&snapshot, &snapshot_moved).unwrap();
    drop(original);
    let resumed_snapshot = moved.path().join("resumed-3");
    let resumed = train(
        ValueNet::new(&device),
        device.clone(),
        &config,
        Some(&snapshot_moved),
        |state| state.save(&resumed_snapshot),
    )
    .unwrap();
    for (expected, actual) in output
        .report
        .iterations
        .iter()
        .zip(&resumed.report.iterations)
    {
        assert_eq!(expected.samples, actual.samples);
        assert_eq!(expected.optimization_samples, actual.optimization_samples);
        assert_eq!(expected.updates, actual.updates);
        assert_eq!(expected.rollout_seed, actual.rollout_seed);
        assert_eq!(
            actual.optimization_samples,
            actual
                .opponents
                .iter()
                .map(|o| o.optimization_samples)
                .sum::<usize>()
        );
        if balanced {
            assert_eq!(actual.updates, 3);
            assert_eq!(actual.optimization_samples, 768);
            assert!(
                actual
                    .opponents
                    .iter()
                    .all(|o| (153..=154).contains(&o.optimization_samples))
            );
        }
        assert_eq!(expected.outcomes, actual.outcomes);
        assert_eq!(expected.opponents, actual.opponents);
        assert_eq!(
            expected.historical_distribution,
            actual.historical_distribution
        );
        assert_eq!(expected.evaluation_opponents, actual.evaluation_opponents);
        assert!((expected.mse - actual.mse).abs() < 1e-6);
        assert_eq!(actual.opponents.iter().map(|o| o.games).sum::<usize>(), 5);
        assert_eq!(
            actual.opponents.iter().map(|o| o.samples).sum::<usize>(),
            actual.samples
        );
        assert!(
            actual
                .opponents
                .iter()
                .all(|o| o.outcomes.iter().sum::<usize>() == o.games)
        );
        let baselines = &actual.evaluation_opponents[..2];
        assert!(
            (actual.evaluation_score.unwrap()
                - baselines.iter().map(|o| o.score).sum::<f64>() / 2.)
                .abs()
                < 1e-12
        );
    }
    assert!(
        output.report.iterations[0]
            .opponents
            .iter()
            .filter(|o| o.fallback_games > 0)
            .count()
            == 3
    );
    assert!(
        output.report.iterations[2]
            .opponents
            .iter()
            .all(|o| o.fallback_games == 0)
    );
    for iteration in 0..=3 {
        assert!(
            resumed_snapshot
                .join(format!("opponents/iteration-{iteration:06}.bpk"))
                .is_file()
        );
    }
    let game = Simple::from_seed(33);
    let mut actions = Vec::new();
    game.legal_actions(&mut actions);
    let inputs: Vec<_> = actions
        .iter()
        .map(|&action| EncodedAfterstate::new(game.observation(), action).unwrap())
        .collect();
    let predict = |net: &ValueNet| {
        let (board, context) = batch(&inputs, &device);
        net.forward(board, context)
            .try_into_data()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap()
    };
    for (a, b) in predict(&output.model).iter().zip(predict(&resumed.model)) {
        assert!((a - b).abs() < 1e-6);
    }
    let mut changed = config.clone();
    changed.opponents.greedy = 2.;
    assert!(
        train(
            ValueNet::new(&device),
            device.clone(),
            &changed,
            Some(&snapshot_moved),
            |_| Ok(())
        )
        .is_err()
    );
    std::fs::remove_file(snapshot_moved.join("opponents/iteration-000000.bpk")).unwrap();
    let error = train(
        ValueNet::new(&device),
        device,
        &config,
        Some(&snapshot_moved),
        |_| Ok(()),
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("opponent checkpoint"));
}

#[test]
fn extending_completed_training_matches_fixed_horizon_run() {
    let device = Device::flex();
    device.seed(911);
    let net = ValueNet::new(&device);
    let short = TrainingConfig {
        iterations: 2,
        games_per_iteration: 4,
        batch_size: 128,
        evaluation_interval: 1,
        evaluation_games: 2,
        opponents: OpponentMix {
            self_play: 80.,
            random_checkpoint: 20.,
            ..Default::default()
        },
        historical_sampling: HistoricalSampling {
            adaptive: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let extended = TrainingConfig {
        iterations: 3,
        epsilon_decay_iterations: Some(2),
        ..short.clone()
    };
    let root = tempfile::tempdir().unwrap();
    let snapshot = root.path().join("completed");
    train(net.clone(), device.clone(), &short, None, |checkpoint| {
        if checkpoint.report().iterations.len() == 2 {
            checkpoint.save(&snapshot)?;
        }
        Ok(())
    })
    .unwrap();
    let resumed = train(
        net.clone(),
        device.clone(),
        &extended,
        Some(&snapshot),
        |_| Ok(()),
    )
    .unwrap();
    let full = train(net, device, &extended, None, |_| Ok(())).unwrap();
    for (a, b) in resumed
        .report
        .iterations
        .iter()
        .zip(&full.report.iterations)
    {
        assert_eq!(a.epsilon, b.epsilon);
        assert_eq!(a.samples, b.samples);
        assert_eq!(a.outcomes, b.outcomes);
        assert_eq!(a.historical_distribution, b.historical_distribution);
        assert!((a.mse - b.mse).abs() < 1e-6);
    }
}

#[test]
fn retained_samples_resume_with_identical_buffer_and_updates() {
    let device = Device::flex();
    device.seed(901);
    let config = TrainingConfig {
        iterations: 3,
        games_per_iteration: 5,
        batch_size: 32,
        evaluation_games: 2,
        evaluation_interval: 1,
        epsilon_start: 1.,
        epsilon_end: 1.,
        replay: Some(ReplayConfig {
            capacity: 128,
            fraction: 0.25,
        }),
        ..Default::default()
    };
    let root = tempfile::tempdir().unwrap();
    let snapshot = root.path().join("snapshot");
    let net = ValueNet::new(&device);
    let full = train(net.clone(), device.clone(), &config, None, |state| {
        if state.report().iterations.len() == 2 {
            state.save(&snapshot)?;
        }
        Ok(())
    })
    .unwrap();
    assert!(snapshot.join("replay.bin").exists());
    let resumed = train(net, device, &config, Some(&snapshot), |_| Ok(())).unwrap();
    for (a, b) in full
        .report
        .iterations
        .iter()
        .zip(&resumed.report.iterations)
    {
        assert_eq!(a.samples, b.samples);
        assert_eq!(a.replay_samples, b.replay_samples);
        assert_eq!(a.opponents, b.opponents);
        assert!((a.mse - b.mse).abs() < 1e-6);
        assert_eq!(
            a.optimization_samples,
            a.opponents
                .iter()
                .map(|o| o.optimization_samples)
                .sum::<usize>()
                + a.replay_samples
        );
    }
    assert_eq!(full.report.iterations[0].replay_samples, 0);
    assert!(full.report.iterations[1].replay_samples > 0);
    assert_eq!(full.report.iterations[2].replay_size, 128);
}
