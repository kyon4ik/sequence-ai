mod optimizer;
mod rollout;

use burn::{
    optim::{GradientsParams, ModuleOptimizer},
    prelude::*,
};
use rand::{RngExt, SeedableRng, seq::SliceRandom};
use rand_xoshiro::Xoshiro256PlusPlus;
use sequence_agent::mc::{EncodedAfterstate, McInferenceMetrics, ValueNet, batch};
use sequence_core::{Color, Game, GameOutcome, Simple};
use sequence_runner::{MatchConfig, Termination};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainingConfig {
    pub iterations: usize,
    pub games_per_iteration: usize,
    pub batch_size: usize,
    /// Concurrent games whose candidate moves share an inference batch.
    /// This execution setting may change when resuming a checkpoint.
    pub inference_batch_games: usize,
    pub learning_rate: f64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub seed: u64,
    pub evaluation_interval: usize,
    pub evaluation_games: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            games_per_iteration: 128,
            batch_size: 256,
            inference_batch_games: 16,
            learning_rate: 3e-4,
            epsilon_start: 0.2,
            epsilon_end: 0.05,
            seed: 42,
            evaluation_interval: 10,
            evaluation_games: 20,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationMetrics {
    pub iteration: usize,
    pub episodes: usize,
    pub samples: usize,
    pub mse: f64,
    pub evaluation_score: Option<f64>,
    pub epsilon: f64,
    /// Red wins, Blue wins, draws.
    pub outcomes: [usize; 3],
    pub reward: [f64; 2],
    pub decision_time: [Duration; 2],
    pub rollout_time: Duration,
    #[serde(default)]
    pub rollout_inference: McInferenceMetrics,
    pub optimization_time: Duration,
    #[serde(default)]
    pub evaluation_time: Duration,
    #[serde(default)]
    pub evaluation_inference: McInferenceMetrics,
    pub elapsed: Duration,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrainingReport {
    pub iterations: Vec<IterationMetrics>,
}

pub struct TrainingOutput {
    pub model: ValueNet,
    pub best_model: Option<ValueNet>,
    pub report: TrainingReport,
}

/// Collect frozen-policy self-play through sequence-runner, then regress on
/// actor-relative terminal outcomes. The returned report replaces stdout logging.
/// The caller owns model/report persistence and device selection.
pub fn train(
    net: ValueNet,
    device: Device,
    config: &TrainingConfig,
) -> Result<TrainingOutput, sequence_agent::AgentError> {
    train_with_progress(net, device, config, |_| {})
}

impl TrainingConfig {
    pub fn validate(&self) -> Result<(), sequence_agent::AgentError> {
        if self.iterations == 0
            || self.games_per_iteration == 0
            || self.batch_size == 0
            || self.inference_batch_games == 0
        {
            return Err("iterations, games_per_iteration, batch_size and inference_batch_games must be positive".into());
        }
        if !self.learning_rate.is_finite()
            || self.learning_rate <= 0.
            || !(0.0..=1.0).contains(&self.epsilon_start)
            || !(0.0..=1.0).contains(&self.epsilon_end)
        {
            return Err("invalid learning rate or exploration probability".into());
        }
        if self.evaluation_interval == 0
            || self.evaluation_games == 0
            || !self.evaluation_games.is_multiple_of(2)
        {
            return Err(
                "evaluation interval must be positive and games must be positive and even".into(),
            );
        }
        Ok(())
    }
}

/// Like `train`, with a callback after each completed optimization iteration.
/// The callback borrows the same metrics retained in the final report.
pub fn train_with_progress(
    net: ValueNet,
    device: Device,
    config: &TrainingConfig,
    mut progress: impl FnMut(&IterationMetrics),
) -> Result<TrainingOutput, sequence_agent::AgentError> {
    train_with_checkpoint(net, device, config, |_, report| {
        progress(report.iterations.last().unwrap());
        Ok(())
    })
}

/// Compatibility callback for callers only interested in inference weights.
pub fn train_with_checkpoint(
    net: ValueNet,
    device: Device,
    config: &TrainingConfig,
    mut checkpoint: impl FnMut(&ValueNet, &TrainingReport) -> Result<(), sequence_agent::AgentError>,
) -> Result<TrainingOutput, sequence_agent::AgentError> {
    train_resumable(net, device, config, None, |state| {
        checkpoint(state.model, &state.metadata.report)
    })
}

#[derive(Serialize, Deserialize)]
struct ResumeMetadata {
    version: u32,
    config: TrainingConfig,
    rng: Xoshiro256PlusPlus,
    report: TrainingReport,
    best_score: Option<f64>,
}

pub struct TrainingCheckpoint<'a> {
    model: &'a ValueNet,
    optimizer: &'a ModuleOptimizer,
    metadata: &'a ResumeMetadata,
    best_model: Option<&'a ValueNet>,
}

impl TrainingCheckpoint<'_> {
    pub fn report(&self) -> &TrainingReport {
        &self.metadata.report
    }

    /// Publish a complete checkpoint directory; incomplete writes stay invisible.
    pub fn save(&self, destination: &Path) -> Result<(), sequence_agent::AgentError> {
        if destination.exists() {
            return Err("checkpoint already exists".into());
        }
        let parent = destination.parent().unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent)?;
        let staging = tempfile::tempdir_in(parent)?;
        self.model
            .clone()
            .valid()
            .save_file(staging.path().join("model.bpk"))?;
        self.optimizer.save(staging.path().join("optimizer.bpk"))?;
        if let Some(best) = self.best_model {
            best.clone()
                .valid()
                .save_file(staging.path().join("best.bpk"))?;
        }
        std::fs::write(
            staging.path().join("state.json"),
            serde_json::to_vec_pretty(self.metadata)?,
        )?;
        for entry in std::fs::read_dir(staging.path())? {
            std::fs::File::open(entry?.path())?.sync_all()?;
        }
        std::fs::rename(staging.path(), destination)?;
        Ok(())
    }
}

/// Resume at an iteration boundary with the original exploration schedule.
/// The current model has no stochastic tensor layers; all continuing randomness
/// comes from the serialized rollout/shuffle RNG and the episode seeds it emits.
pub fn train_resumable(
    mut net: ValueNet,
    device: Device,
    config: &TrainingConfig,
    resume: Option<&Path>,
    mut checkpoint: impl FnMut(&TrainingCheckpoint<'_>) -> Result<(), sequence_agent::AgentError>,
) -> Result<TrainingOutput, sequence_agent::AgentError> {
    config.validate()?;
    let device = device.autodiff();
    let mut optimizer = optimizer::adam();
    let mut metadata = ResumeMetadata {
        version: 1,
        config: config.clone(),
        rng: Xoshiro256PlusPlus::seed_from_u64(config.seed),
        report: TrainingReport::default(),
        best_score: None,
    };
    let mut best_model = None;
    if let Some(path) = resume {
        metadata = serde_json::from_reader(std::fs::File::open(path.join("state.json"))?)?;
        // Scheduling is an execution choice. All learning settings, including
        // total iterations and the exploration schedule, remain immutable.
        metadata.config.inference_batch_games = config.inference_batch_games;
        if metadata.version != 1 || metadata.config != *config {
            return Err("checkpoint version or training configuration differs".into());
        }
        if metadata.report.iterations.len() > config.iterations
            || metadata
                .report
                .iterations
                .iter()
                .enumerate()
                .any(|(i, m)| m.iteration != i + 1)
        {
            return Err("invalid checkpoint iteration history".into());
        }
        net = net.try_load_file(path.join("model.bpk"))?;
        optimizer = optimizer.load(path.join("optimizer.bpk"))?;
        if metadata.best_score.is_some() {
            best_model = Some(net.clone().try_load_file(path.join("best.bpk"))?);
        }
    }
    net = net.train();
    let ResumeMetadata {
        rng,
        report,
        best_score,
        ..
    } = &mut metadata;
    for iteration in report.iterations.len()..config.iterations {
        let iteration_started = Instant::now();
        let fraction = iteration as f64 / config.iterations.saturating_sub(1).max(1) as f64;
        let epsilon = config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start);

        let mut samples = Vec::new();
        let mut results = [0usize; 3];
        let mut reward = [0.; 2];
        let mut decision_time = [Duration::ZERO; 2];
        let rollout_started = Instant::now();

        let matches: Vec<_> = (0..config.games_per_iteration)
            .map(|_| MatchConfig {
                game_seed: rng.random(),
                agent_seeds: [rng.random(), rng.random()],
                ..Default::default()
            })
            .collect();
        let rollout = rollout::play(
            &net,
            &device,
            &matches,
            config.inference_batch_games,
            epsilon,
            None,
        )?;
        for record in rollout.records {
            let Termination::Completed(outcome) = record.termination else {
                return Err("Monte Carlo targets require a completed episode".into());
            };
            results[match outcome {
                GameOutcome::Won(c) => c as usize,
                GameOutcome::Draw => 2,
                _ => unreachable!(),
            }] += 1;

            for i in 0..2 {
                reward[i] += record.metrics.players[i].reward;
                decision_time[i] += record.metrics.players[i].decision_time;
            }
            let mut game = Simple::from_seed(record.header.config.game_seed);
            for step in &record.steps {
                let encoded = EncodedAfterstate::new(game.observation(), step.action)?;
                samples.push((encoded, terminal_target(outcome, step.actor)));
                game.apply(step.action)?;
            }
        }
        let rollout_time = rollout_started.elapsed();

        samples.shuffle(rng);

        let mut total_loss = Tensor::<1>::zeros([1], &device.clone().without_autodiff());
        let optimization_started = Instant::now();
        for minibatch in samples.chunks(config.batch_size) {
            let inputs: Vec<_> = minibatch.iter().map(|(x, _)| x.clone()).collect();
            let targets: Vec<_> = minibatch.iter().map(|(_, y)| *y).collect();
            let (board, context) = batch(&inputs, &device);
            let targets =
                Tensor::<2>::from_data(TensorData::new(targets, [inputs.len(), 1]), &device);
            let error = net.forward(board, context) - targets;
            let loss = (error.clone() * error).mean();
            total_loss = total_loss + loss.clone().without_autodiff() * inputs.len() as f32;
            let gradients = GradientsParams::from_grads(loss.backward(), &net);
            net = optimizer.step(config.learning_rate, net, gradients);
        }

        let mse = total_loss
            .try_into_scalar::<f32>()
            .map_err(|e| format!("reading training loss: {e}"))? as f64
            / samples.len() as f64;
        let optimization_time = optimization_started.elapsed();
        let evaluation_started = Instant::now();
        let mut evaluation_inference = McInferenceMetrics::default();
        let evaluation_score = if (iteration + 1).is_multiple_of(config.evaluation_interval)
            || iteration + 1 == config.iterations
        {
            let (score, inference) = evaluate(
                &net,
                &device,
                config.evaluation_games,
                config.inference_batch_games,
            )?;
            evaluation_inference = inference;
            Some(score)
        } else {
            None
        };
        report.iterations.push(IterationMetrics {
            iteration: iteration + 1,
            episodes: config.games_per_iteration,
            samples: samples.len(),
            mse,
            evaluation_score,
            epsilon,
            outcomes: results,
            reward,
            decision_time,
            rollout_time,
            rollout_inference: rollout.inference,
            optimization_time,
            evaluation_time: if evaluation_score.is_some() {
                evaluation_started.elapsed()
            } else {
                Duration::ZERO
            },
            evaluation_inference,
            elapsed: iteration_started.elapsed(),
        });
        if let Some(score) = evaluation_score
            && best_score.is_none_or(|best| score > best)
        {
            *best_score = Some(score);
            best_model = Some(net.clone().valid());
        }
        let snapshot_metadata = ResumeMetadata {
            version: 1,
            config: config.clone(),
            rng: rng.clone(),
            report: report.clone(),
            best_score: *best_score,
        };
        checkpoint(&TrainingCheckpoint {
            model: &net,
            optimizer: &optimizer,
            metadata: &snapshot_metadata,
            best_model: best_model.as_ref(),
        })?;
    }

    Ok(TrainingOutput {
        model: net.valid(),
        best_model,
        report: metadata.report,
    })
}

fn evaluate(
    net: &ValueNet,
    device: &Device,
    games: usize,
    batch_games: usize,
) -> Result<(f64, McInferenceMetrics), sequence_agent::AgentError> {
    use sequence_agent::AgentConfig;
    let mut score = 0.;
    let mut metrics = McInferenceMetrics::default();
    for baseline in [AgentConfig::Random, AgentConfig::greedy()] {
        let configs: Vec<_> = (0..games)
            .map(|index| MatchConfig {
                game_seed: 0xEA10_0000 + (index / 2) as u64,
                agent_seeds: [0xEA11_0000 + index as u64, 0xEA12_0000 + index as u64],
                ..Default::default()
            })
            .collect();
        let rollout = rollout::play(net, device, &configs, batch_games, 0., Some(&baseline))?;
        metrics.decisions += rollout.inference.decisions;
        metrics.batches += rollout.inference.batches;
        metrics.candidates += rollout.inference.candidates;
        metrics.encoding_time += rollout.inference.encoding_time;
        metrics.prediction_time += rollout.inference.prediction_time;
        for (index, record) in rollout.records.iter().enumerate() {
            score += match record.termination {
                Termination::Completed(GameOutcome::Won(color)) if color as usize == index % 2 => {
                    1.
                }
                Termination::Completed(GameOutcome::Draw) => 0.5,
                Termination::Completed(_) => 0.,
                _ => return Err("evaluation did not complete".into()),
            };
        }
    }
    Ok((score / (2 * games) as f64, metrics))
}

fn terminal_target(outcome: GameOutcome, actor: Color) -> f32 {
    match outcome {
        GameOutcome::Won(winner) if winner == actor => 1.,
        GameOutcome::Won(_) => -1.,
        GameOutcome::Draw => 0.,
        GameOutcome::InProgress => panic!("Monte Carlo targets require a completed game"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sequence_agent::{
        Agent, AgentConfig, AgentDescriptor, AgentFactory, AgentSpec, InferenceDevice, mc::McAgent,
    };
    use sequence_runner::Replay;
    use sequence_runner::{MatchRunner, Participant};

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
        let continuous = train_resumable(
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
        // A fresh model has fresh parameter IDs; restoring must reconnect Adam state too.
        let mut resume_config = config.clone();
        resume_config.inference_batch_games = 3;
        let resumed = train_resumable(
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
            train_resumable(
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
    fn legacy_config_and_metrics_load_and_invalid_batching_is_rejected() {
        let mut legacy = serde_json::to_value(TrainingConfig::default()).unwrap();
        legacy
            .as_object_mut()
            .unwrap()
            .remove("inference_batch_games");
        let mut config: TrainingConfig = serde_json::from_value(legacy).unwrap();
        assert_eq!(config.inference_batch_games, 16);
        config.inference_batch_games = 0;
        assert!(config.validate().is_err());
        let mut metric = serde_json::json!({"iteration": 1, "episodes": 1, "samples": 1, "mse": 1., "evaluation_score": null, "epsilon": 1., "outcomes": [1,0,0], "reward": [1.,-1.]});
        for field in ["rollout_time", "optimization_time", "elapsed"] {
            metric[field] = serde_json::to_value(Duration::ZERO).unwrap();
        }
        metric["decision_time"] = serde_json::to_value([Duration::ZERO; 2]).unwrap();
        let loaded: IterationMetrics = serde_json::from_value(metric).unwrap();
        assert_eq!(loaded.rollout_inference.batches, 0);
        assert_eq!(loaded.evaluation_time, Duration::ZERO);
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
                let output =
                    rollout::play(&net, &device, &configs, batch_games, 0.175, None).unwrap();
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
        )
        .unwrap();
        assert_eq!(output.report.iterations.len(), 1);
        assert_eq!(output.report.iterations[0].episodes, 1);
        assert!(output.report.iterations[0].mse.is_finite());
        let net = output.model;
        let after = predict(&net);
        assert!(after.is_finite() && (-1.0..=1.0).contains(&after));
        assert!((before - after).abs() > 1e-6);
        let path =
            std::env::temp_dir().join(format!("sequence-mc-test-{}.bpk", std::process::id()));
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
        )
        .unwrap();
        assert!(output.report.iterations[0].mse.is_finite());
        let path =
            std::env::temp_dir().join(format!("sequence-cuda-test-{}.bpk", std::process::id()));
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
}
