use super::{OpponentEvaluation, evaluation_summary, outcome_index, rollout};
use burn::prelude::*;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use sequence_agent::{
    AgentConfig, AgentError,
    mc::{McInferenceMetrics, ValueNet},
};
use sequence_runner::{MatchConfig, Termination};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationConfig {
    /// Games per baseline, including both seats for each sampled game seed.
    pub games: usize,
    pub inference_batch_games: usize,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationGame {
    pub config: MatchConfig,
    pub learner_seat: usize,
    /// Win 1, draw 0.5, loss 0 from the learner's perspective.
    pub score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BaselineEvaluation {
    pub baseline: AgentConfig,
    pub summary: OpponentEvaluation,
    pub games: Vec<EvaluationGame>,
    pub inference: McInferenceMetrics,
    pub elapsed: Duration,
}

/// Batched, zero-exploration evaluation on an explicit seed schedule independent
/// of training. Returns per-game scores for paired comparisons without retaining
/// full replays. Each pair shares a game seed and swaps the learner's seat.
pub fn evaluate_seeded_baselines(
    net: &ValueNet,
    device: &Device,
    config: &EvaluationConfig,
) -> Result<Vec<BaselineEvaluation>, AgentError> {
    if config.games == 0 || !config.games.is_multiple_of(2) || config.inference_batch_games == 0 {
        return Err(
            "evaluation needs positive even games and positive inference_batch_games".into(),
        );
    }
    let configs = seeded_configs(config);
    let mut results = Vec::new();
    for (name, baseline) in [
        ("random", AgentConfig::Random),
        ("greedy", AgentConfig::greedy()),
    ] {
        let started = Instant::now();
        let output = rollout::play(
            net,
            device,
            &configs,
            config.inference_batch_games,
            0.,
            Some(&baseline),
        )?;
        let summary = evaluation_summary(name.into(), &output.records)?;
        let games = output
            .records
            .iter()
            .enumerate()
            .map(|(index, record)| {
                let Termination::Completed(outcome) = record.termination else {
                    unreachable!("summary requires complete games")
                };
                EvaluationGame {
                    config: record.header.config.clone(),
                    learner_seat: index % 2,
                    score: [1., 0., 0.5][outcome_index(outcome, index % 2)],
                }
            })
            .collect();
        results.push(BaselineEvaluation {
            baseline,
            summary,
            games,
            inference: output.inference,
            elapsed: started.elapsed(),
        });
    }
    Ok(results)
}

fn seeded_configs(config: &EvaluationConfig) -> Vec<MatchConfig> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(config.seed);
    let mut configs = Vec::with_capacity(config.games);
    for _ in 0..config.games / 2 {
        let game_seed = rng.random();
        let seeds = [rng.random(), rng.random()];
        for seat in 0..2 {
            configs.push(MatchConfig {
                game_seed,
                agent_seeds: if seat == 0 {
                    seeds
                } else {
                    [seeds[1], seeds[0]]
                },
                ..Default::default()
            });
        }
    }
    configs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_evaluation_seeds_are_paired_and_reproducible() {
        let config = EvaluationConfig {
            games: 6,
            inference_batch_games: 3,
            seed: 2026111401,
        };
        let configs = seeded_configs(&config);
        assert_eq!(configs, seeded_configs(&config));
        for pair in configs.as_chunks::<2>().0 {
            assert_eq!(pair[0].game_seed, pair[1].game_seed);
            assert_eq!(
                pair[0].agent_seeds,
                [pair[1].agent_seeds[1], pair[1].agent_seeds[0]]
            );
        }
        let device = Device::flex();
        let net = ValueNet::new(&device);
        let output = evaluate_seeded_baselines(&net, &device, &config).unwrap();
        for baseline in output {
            assert_eq!(baseline.games.len(), 6);
            assert_eq!(
                baseline.summary.score,
                baseline.games.iter().map(|g| g.score).sum::<f64>() / 6.
            );
            assert!(
                baseline
                    .games
                    .iter()
                    .enumerate()
                    .all(|(i, g)| g.learner_seat == i % 2)
            );
        }
        let invalid = EvaluationConfig { games: 3, ..config };
        assert!(evaluate_seeded_baselines(&net, &device, &invalid).is_err());
    }
}
