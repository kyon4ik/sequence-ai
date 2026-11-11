use burn::{
    optim::{AdamConfig, GradientsParams},
    prelude::*,
};
use rand::{RngExt, SeedableRng, rngs::SmallRng, seq::SliceRandom};
use sequence_agent::AgentDescriptor;
use sequence_agent::mc::{EncodedAfterstate, McAgent, ValueNet, batch};
use sequence_core::{Color, Game, GameOutcome, Simple};
use sequence_runner::{MatchConfig, MatchRunner, Participant, Replay, Termination};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub iterations: usize,
    pub games_per_iteration: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub seed: u64,
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
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationMetrics {
    pub iteration: usize,
    pub episodes: usize,
    pub samples: usize,
    pub mse: f64,
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
    pub report: TrainingReport,
}

/// Collect frozen-policy self-play through sequence-runner, then regress on
/// actor-relative terminal outcomes. The returned report replaces stdout logging.
/// The caller owns model/report persistence and device selection.
pub fn train(
    mut net: ValueNet,
    device: Device,
    config: &TrainingConfig,
) -> Result<TrainingOutput, sequence_agent::AgentError> {
    if config.games_per_iteration == 0 || config.batch_size == 0 {
        return Err("games_per_iteration and batch_size must be positive".into());
    }
    if !config.learning_rate.is_finite()
        || config.learning_rate <= 0.
        || !(0.0..=1.0).contains(&config.epsilon_start)
        || !(0.0..=1.0).contains(&config.epsilon_end)
    {
        return Err("invalid learning rate or exploration probability".into());
    }

    let device = device.autodiff();
    net = net.train();
    let mut optimizer = AdamConfig::new().init();
    let mut rng = SmallRng::seed_from_u64(config.seed);
    let mut runner = MatchRunner::<Simple>::default();
    let mut report = TrainingReport::default();

    for iteration in 0..config.iterations {
        let iteration_started = Instant::now();
        let progress = iteration as f64 / config.iterations.saturating_sub(1).max(1) as f64;
        let epsilon = config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start);

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
            let mut replay = Replay::<Simple>::new(&record)?;
            for step in &record.steps {
                let game = replay.seek(step.index)?;
                let encoded = EncodedAfterstate::new(game.observation(), step.action)?;
                samples.push((encoded, terminal_target(outcome, step.actor)));
            }
        }
        let rollout_time = rollout_started.elapsed();

        samples.shuffle(&mut rng);

        let mut total_loss = 0.;
        let optimization_started = Instant::now();
        for minibatch in samples.chunks(config.batch_size) {
            let inputs: Vec<_> = minibatch.iter().map(|(x, _)| x.clone()).collect();
            let targets: Vec<_> = minibatch.iter().map(|(_, y)| *y).collect();
            let (board, context) = batch(&inputs, &device);
            let targets =
                Tensor::<2>::from_data(TensorData::new(targets, [inputs.len(), 1]), &device);
            let error = net.forward(board, context) - targets;
            let loss = (error.clone() * error).mean();
            total_loss += loss.clone().into_scalar::<f32>() as f64 * inputs.len() as f64;
            let gradients = GradientsParams::from_grads(loss.backward(), &net);
            net = optimizer.step(config.learning_rate, net, gradients);
        }

        report.iterations.push(IterationMetrics {
            iteration: iteration + 1,
            episodes: config.games_per_iteration,
            samples: samples.len(),
            mse: total_loss / samples.len() as f64,
            epsilon,
            outcomes: results,
            reward,
            decision_time,
            rollout_time,
            optimization_time: optimization_started.elapsed(),
            elapsed: iteration_started.elapsed(),
        });
    }

    Ok(TrainingOutput {
        model: net.valid(),
        report,
    })
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
