mod checkpoint;
mod config;
mod metrics;
#[cfg(test)]
mod tests;
pub use checkpoint::TrainingCheckpoint;
use checkpoint::{CHECKPOINT_VERSION, ResumeMetadata};
pub use config::TrainingConfig;
use config::resume_compatible;
pub use metrics::{IterationMetrics, TrainingOutput, TrainingReport};
mod replay_buffer;
use replay_buffer::ReplayBuffer;
pub use replay_buffer::ReplayConfig;
mod diagnostics;
pub use diagnostics::PolicyDiagnostics;
mod evaluation;
pub use evaluation::{
    BaselineEvaluation, EvaluationConfig, EvaluationGame, evaluate_seeded_baselines,
    evaluate_seeded_panel,
};
#[cfg(test)]
mod mixed_tests;
mod opponents;
mod optimizer;
mod rollout;
mod sampling;
pub use sampling::SamplingStrategy;

use opponents::{CheckpointPool, OpponentPolicy, OpponentState};
pub use opponents::{
    HistoricalProbability, HistoricalSampling, OpponentEvaluation, OpponentKind, OpponentMetrics,
    OpponentMix,
};

use burn::{optim::GradientsParams, prelude::*};
use rand::{RngExt, SeedableRng, seq::SliceRandom};
use rand_xoshiro::Xoshiro256PlusPlus;
use sequence_agent::mc::{EncodedAfterstate, McInferenceMetrics, ValueNet, batch};
use sequence_core::{Color, Game, GameOutcome, Simple};
use sequence_runner::{MatchConfig, Termination};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

/// Train from scratch or resume a complete checkpoint at an iteration boundary.
/// The callback receives each completed iteration and may save its checkpoint.
/// Pass `None` and `|_| Ok(())` to train without resuming or saving checkpoints.
/// The current model has no stochastic tensor layers; all continuing randomness
/// comes from the serialized rollout/shuffle RNG and the episode seeds it emits.
pub fn train(
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
        version: CHECKPOINT_VERSION,
        config: config.clone(),
        rng: Xoshiro256PlusPlus::seed_from_u64(config.seed),
        report: TrainingReport::default(),
        best_score: None,
        opponents: OpponentState::new(config.seed),
    };
    let mut replay_buffer = if let (Some(replay), Some(path)) = (&config.replay, resume) {
        ReplayBuffer::load(&path.join("replay.bin"), replay.capacity)?
    } else {
        ReplayBuffer::default()
    };
    let mut best_model = None;
    if let Some(path) = resume {
        metadata = ResumeMetadata::load(path)?;
        // Only extend the training budget when the effective exploration horizon
        // stays identical. Other learning settings remain immutable.
        if !resume_compatible(&metadata.config, config) {
            return Err("checkpoint training configuration differs".into());
        }
        metadata.config = config.clone();
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
    let opponent_state = &mut metadata.opponents;
    let mut pool = if config.opponents.uses_history() {
        Some(CheckpointPool::new()?)
    } else {
        None
    };
    if let Some(pool) = &mut pool {
        if let Some(path) = resume {
            let completed = metadata.report.iterations.len();
            if opponent_state.checkpoints != (0..=completed).collect::<Vec<_>>()
                || opponent_state.best_iteration.is_some_and(|i| i > completed)
                || opponent_state.best_iteration.is_some() != metadata.best_score.is_some()
            {
                return Err("invalid opponent checkpoint manifest".into());
            }
            if opponent_state
                .qualities
                .iter()
                .any(|(i, q)| !q.is_finite() || !opponent_state.checkpoints.contains(i))
            {
                return Err("invalid historical opponent qualities".into());
            }
            pool.restore(path, &opponent_state.checkpoints)?;
        } else {
            pool.insert(0, &net)?;
            opponent_state.checkpoints.push(0);
        }
    }
    net = net.train();
    let ResumeMetadata {
        rng,
        report,
        best_score,
        opponents,
        ..
    } = &mut metadata;
    let opponent_state = opponents;
    for iteration in report.iterations.len()..config.iterations {
        let iteration_started = Instant::now();
        let epsilon = config.epsilon_at(iteration);

        let mut samples = Vec::new();
        let mut results = [0usize; 3];
        let mut reward = [0.; 2];
        let mut decision_time = [Duration::ZERO; 2];
        let rollout_started = Instant::now();

        let rollout_seed = config
            .updates_per_iteration
            .map(|_| iteration_seed(config.seed, iteration, 0xE915_0DE5));
        let mut episode_rng = rollout_seed.map(Xoshiro256PlusPlus::seed_from_u64);
        let match_rng = episode_rng.as_mut().unwrap_or(rng);
        let matches: Vec<_> = (0..config.games_per_iteration)
            .map(|_| MatchConfig {
                game_seed: match_rng.random(),
                agent_seeds: [match_rng.random(), match_rng.random()],
                ..Default::default()
            })
            .collect();
        let historical_distribution =
            opponent_state.distribution(iteration, &config.historical_sampling);
        let plans = config.opponents.schedule_with(
            config.games_per_iteration,
            iteration,
            opponent_state,
            &config.historical_sampling,
        );
        let mut opponent_metrics: BTreeMap<_, OpponentMetrics> = BTreeMap::new();
        let rollout = rollout::play_opponents(
            &net,
            &device,
            &matches,
            config.inference_batch_games,
            epsilon,
            &plans,
            pool.as_ref(),
        )?;
        for (record, plan) in rollout.records.into_iter().zip(&plans) {
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
            let metrics = opponent_metrics
                .entry((plan.source, plan.policy))
                .or_insert_with(|| OpponentMetrics {
                    source: plan.source,
                    opponent: plan.policy.id(),
                    ..Default::default()
                });
            metrics.games += 1;
            metrics.learner_seats[plan.learner_seat] += 1;
            metrics.fallback_games += usize::from(
                plan.source != OpponentKind::SelfPlay && plan.policy == OpponentPolicy::SelfPlay,
            );
            metrics.outcomes[outcome_index(outcome, plan.learner_seat)] += 1;
            opponent_state.observe(
                plan,
                outcome_index(outcome, plan.learner_seat) == 0,
                &config.historical_sampling,
            )?;
            let mut game = Simple::from_seed(record.header.config.game_seed);
            for step in &record.steps {
                if plan.learns_from(step.actor as usize) {
                    let encoded = EncodedAfterstate::new(game.observation(), step.action)?;
                    samples.push((
                        encoded,
                        terminal_target(outcome, step.actor),
                        (plan.source, plan.policy),
                    ));
                    metrics.samples += 1;
                }
                game.apply(step.action)?;
            }
        }
        let rollout_time = rollout_started.elapsed();

        samples.shuffle(rng);

        let sources: Vec<_> = samples.iter().map(|sample| sample.2.0).collect();
        let mut selection_rng =
            Xoshiro256PlusPlus::seed_from_u64(iteration_seed(config.seed, iteration, 0x5A4F_1E50));
        let selected = sampling::select(
            &sources,
            &config.opponents,
            config.sampling,
            config
                .updates_per_iteration
                .map(|updates| updates * config.batch_size),
            iteration,
            &mut selection_rng,
        );
        let retained = if let Some(replay) = &config.replay {
            replay_buffer.draws(
                selected.len(),
                replay.fraction,
                iteration_seed(config.seed, iteration, 0xAE91_A750),
            )
        } else {
            BTreeMap::new()
        };
        for (slot, &index) in selected.iter().enumerate() {
            if retained.contains_key(&slot) {
                continue;
            }
            opponent_metrics
                .get_mut(&samples[index].2)
                .unwrap()
                .optimization_samples += 1;
        }
        let mut total_loss = Tensor::<1>::zeros([1], &device.clone().without_autodiff());
        let optimization_started = Instant::now();
        for (batch_index, minibatch) in selected.chunks(config.batch_size).enumerate() {
            let examples: Vec<_> = minibatch
                .iter()
                .enumerate()
                .map(|(offset, &index)| {
                    let slot = batch_index * config.batch_size + offset;
                    if let Some(&old) = retained.get(&slot) {
                        let s = replay_buffer.get(old);
                        (&s.0, s.1)
                    } else {
                        (&samples[index].0, samples[index].1)
                    }
                })
                .collect();
            let inputs: Vec<_> = examples
                .iter()
                .map(|(encoded, _)| (*encoded).clone())
                .collect();
            let targets: Vec<_> = examples.iter().map(|(_, target)| *target).collect();
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
            / selected.len() as f64;
        let optimization_time = optimization_started.elapsed();
        if let Some(replay) = &config.replay {
            replay_buffer.extend(samples.iter().map(|s| (&s.0, s.1)), replay.capacity);
        }
        let evaluation_started = Instant::now();
        let mut evaluation_inference = McInferenceMetrics::default();
        let mut evaluation_opponents = Vec::new();
        let evaluation_score = if (iteration + 1).is_multiple_of(config.evaluation_interval)
            || iteration + 1 == config.iterations
        {
            let (score, inference, details) = evaluate_baselines(
                &net,
                &device,
                config.evaluation_games,
                config.inference_batch_games,
            )?;
            evaluation_inference = inference;
            evaluation_opponents = details;
            if let Some(pool) = &pool {
                let (details, inference) =
                    evaluate_history(&net, &device, config, iteration, opponent_state, pool)?;
                evaluation_opponents.extend(details);
                rollout::add_metrics(&mut evaluation_inference, &inference);
            }
            Some(score)
        } else {
            None
        };
        report.iterations.push(IterationMetrics {
            iteration: iteration + 1,
            episodes: config.games_per_iteration,
            samples: samples.len(),
            optimization_samples: selected.len(),
            updates: selected.len().div_ceil(config.batch_size),
            replay_samples: retained.len(),
            replay_size: replay_buffer.len(),
            rollout_seed,
            mse,
            evaluation_score,
            evaluation_opponents,
            opponents: opponent_metrics.into_values().collect(),
            historical_distribution,
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
            opponent_state.best_iteration = Some(iteration + 1);
        }
        if let Some(pool) = &mut pool {
            pool.insert(iteration + 1, &net)?;
            opponent_state.checkpoints.push(iteration + 1);
        }
        let snapshot_metadata = ResumeMetadata {
            version: CHECKPOINT_VERSION,
            config: config.clone(),
            rng: rng.clone(),
            report: report.clone(),
            best_score: *best_score,
            opponents: opponent_state.clone(),
        };
        checkpoint(&TrainingCheckpoint {
            model: &net,
            optimizer: &optimizer,
            metadata: &snapshot_metadata,
            best_model: best_model.as_ref(),
            pool: pool.as_ref(),
            replay: config.replay.as_ref().map(|_| &replay_buffer),
        })?;
    }

    Ok(TrainingOutput {
        model: net.valid(),
        best_model,
        report: metadata.report,
    })
}

#[cfg(test)]
fn evaluate(
    net: &ValueNet,
    device: &Device,
    games: usize,
    batch_games: usize,
) -> Result<(f64, McInferenceMetrics), sequence_agent::AgentError> {
    let (score, metrics, _) = evaluate_baselines(net, device, games, batch_games)?;
    Ok((score, metrics))
}

fn evaluate_baselines(
    net: &ValueNet,
    device: &Device,
    games: usize,
    batch_games: usize,
) -> Result<(f64, McInferenceMetrics, Vec<OpponentEvaluation>), sequence_agent::AgentError> {
    use sequence_agent::AgentConfig;
    let mut details = Vec::new();
    let mut metrics = McInferenceMetrics::default();
    for (name, baseline) in [
        ("random", AgentConfig::Random),
        ("greedy", AgentConfig::greedy()),
    ] {
        let configs = evaluation_configs(games, 0xEA10_0000);
        let rollout = rollout::play(net, device, &configs, batch_games, 0., Some(&baseline))?;
        rollout::add_metrics(&mut metrics, &rollout.inference);
        details.push(evaluation_summary(name.into(), &rollout.records)?);
    }
    Ok((
        details
            .iter()
            .map(|d| d.outcomes[0] as f64 + 0.5 * d.outcomes[2] as f64)
            .sum::<f64>()
            / (2 * games) as f64,
        metrics,
        details,
    ))
}

fn evaluate_history(
    net: &ValueNet,
    device: &Device,
    config: &TrainingConfig,
    completed: usize,
    state: &OpponentState,
    pool: &CheckpointPool,
) -> Result<(Vec<OpponentEvaluation>, McInferenceMetrics), sequence_agent::AgentError> {
    use opponents::OpponentPlan;
    // Evaluate the initial policy, a lagged policy and the incumbent best. The
    // moving historical panel is diagnostic and never changes best selection.
    let mut iterations = BTreeSet::from([0]);
    iterations.extend(completed.checked_sub(1));
    iterations.extend(state.best_iteration);
    let mut details = Vec::new();
    let mut inference = McInferenceMetrics::default();
    for iteration in iterations
        .into_iter()
        .filter(|i| state.checkpoints.contains(i))
    {
        let policy = OpponentPolicy::Checkpoint(iteration);
        let configs = evaluation_configs(config.evaluation_games, 0xEB10_0000);
        let plans: Vec<_> = (0..configs.len())
            .map(|index| OpponentPlan {
                historical_sample: None,
                source: OpponentKind::RandomCheckpoint,
                policy,
                learner_seat: index % 2,
            })
            .collect();
        let output = rollout::play_opponents(
            net,
            device,
            &configs,
            config.inference_batch_games,
            0.,
            &plans,
            Some(pool),
        )?;
        rollout::add_metrics(&mut inference, &output.inference);
        details.push(evaluation_summary(policy.id(), &output.records)?);
    }
    Ok((details, inference))
}

fn evaluation_configs(games: usize, base: u64) -> Vec<MatchConfig> {
    (0..games)
        .map(|index| MatchConfig {
            game_seed: base + (index / 2) as u64,
            agent_seeds: [
                base + 0x1_0000 + index as u64,
                base + 0x2_0000 + index as u64,
            ],
            ..Default::default()
        })
        .collect()
}

fn evaluation_summary(
    opponent: String,
    records: &[sequence_runner::MatchRecord],
) -> Result<OpponentEvaluation, sequence_agent::AgentError> {
    let mut outcomes = [0; 3];
    for (index, record) in records.iter().enumerate() {
        let Termination::Completed(outcome) = record.termination else {
            return Err("evaluation did not complete".into());
        };
        outcomes[outcome_index(outcome, index % 2)] += 1;
    }
    Ok(OpponentEvaluation {
        opponent,
        games: records.len(),
        outcomes,
        score: (outcomes[0] as f64 + 0.5 * outcomes[2] as f64) / records.len() as f64,
    })
}

fn outcome_index(outcome: GameOutcome, learner: usize) -> usize {
    match outcome {
        GameOutcome::Won(color) if color as usize == learner => 0,
        GameOutcome::Won(_) => 1,
        GameOutcome::Draw => 2,
        GameOutcome::InProgress => unreachable!("completed episodes only"),
    }
}

fn iteration_seed(seed: u64, iteration: usize, domain: u64) -> u64 {
    seed ^ (iteration as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ domain
}

fn terminal_target(outcome: GameOutcome, actor: Color) -> f32 {
    match outcome {
        GameOutcome::Won(winner) if winner == actor => 1.,
        GameOutcome::Won(_) => -1.,
        GameOutcome::Draw => 0.,
        GameOutcome::InProgress => panic!("Monte Carlo targets require a completed game"),
    }
}
