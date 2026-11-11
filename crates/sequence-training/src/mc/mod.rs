mod optimizer;

use burn::{
    optim::{GradientsParams, ModuleOptimizer},
    prelude::*,
};
use rand::{RngExt, SeedableRng, seq::SliceRandom};
use rand_xoshiro::Xoshiro256PlusPlus;
use sequence_agent::AgentDescriptor;
use sequence_agent::mc::{EncodedAfterstate, McAgent, ValueNet, batch};
use sequence_core::{Color, Game, GameOutcome, Simple};
use sequence_runner::{MatchConfig, MatchRunner, Participant, Termination};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainingConfig {
    pub iterations: usize,
    pub games_per_iteration: usize,
    pub batch_size: usize,
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
    pub optimization_time: Duration,
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
        if self.iterations == 0 || self.games_per_iteration == 0 || self.batch_size == 0 {
            return Err("iterations, games_per_iteration and batch_size must be positive".into());
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
    let mut runner = MatchRunner::<Simple>::default();
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

        for _ in 0..config.games_per_iteration {
            let config = MatchConfig {
                game_seed: rng.random(),
                agent_seeds: [rng.random(), rng.random()],
                ..Default::default()
            };
            let mut red = McAgent::new(net.clone(), device.clone(), config.agent_seeds[0], epsilon);
            let mut blue =
                McAgent::new(net.clone(), device.clone(), config.agent_seeds[1], epsilon);
            let record = runner.run(
                config,
                [
                    Participant::new(AgentDescriptor::new("self-play", "MC"), &mut red),
                    Participant::new(AgentDescriptor::new("self-play", "MC"), &mut blue),
                ],
            )?;
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
        let evaluation_score = if (iteration + 1).is_multiple_of(config.evaluation_interval)
            || iteration + 1 == config.iterations
        {
            Some(evaluate(&net, &device, config.evaluation_games)?)
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
            optimization_time,
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
) -> Result<f64, sequence_agent::AgentError> {
    use sequence_agent::AgentConfig;
    let mut score = 0.;
    for baseline in [AgentConfig::Random, AgentConfig::greedy()] {
        let mut opponent = baseline.create(0, sequence_agent::InferenceDevice::Cpu)?;
        let mut policy = McAgent::new(net.clone(), device.clone(), 0, 0.);
        for index in 0..games {
            let seat = index % 2;
            let candidate =
                Participant::new(AgentDescriptor::new("candidate", "Candidate"), &mut policy);
            let other = Participant::new(
                AgentDescriptor::new("baseline", "Baseline"),
                opponent.as_mut(),
            );
            let players = if seat == 0 {
                [candidate, other]
            } else {
                [other, candidate]
            };
            let record = MatchRunner::<Simple>::default().run(
                MatchConfig {
                    game_seed: 0xEA10_0000 + (index / 2) as u64,
                    agent_seeds: [0xEA11_0000 + index as u64, 0xEA12_0000 + index as u64],
                    ..Default::default()
                },
                players,
            )?;
            score += match record.termination {
                Termination::Completed(GameOutcome::Won(color)) if color as usize == seat => 1.,
                Termination::Completed(GameOutcome::Draw) => 0.5,
                Termination::Completed(_) => 0.,
                _ => return Err("evaluation did not complete".into()),
            };
        }
    }
    Ok(score / (2 * games) as f64)
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
    use sequence_agent::{Agent, AgentConfig, AgentFactory, AgentSpec, InferenceDevice};
    use sequence_runner::Replay;

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
            games_per_iteration: 1,
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
        let resumed = train_resumable(
            ValueNet::new(&device),
            device.clone(),
            &config,
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
