use std::path::Path;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sequence_game::{Board, Outcome};
use sequence_match::{EvaluationReport, MatchEnd};
use serde::Serialize;

use crate::experiment::{AgentKind, BootstrapConfig, ExperimentConfig};

#[derive(Serialize)]
pub struct Summary {
    pub board_layout: &'static str,
    pub policies: [AgentKind; 2],
    pub deals: usize,
    pub overall: Score,
    pub starts: Score,
    pub moves_second: Score,
    pub median_actions: f64,
    pub paired_ci_95: Option<[f64; 2]>,
}

#[derive(Default, Serialize)]
pub struct Counts {
    wins: usize,
    losses: usize,
    draws: usize,
    pub truncations: usize,
}

impl Counts {
    fn record(&mut self, end: MatchEnd, policy_a_seat: usize) -> f64 {
        match end {
            MatchEnd::Completed(Outcome::Win(winner)) if winner.index() == policy_a_seat => {
                self.wins += 1;
                1.0
            }
            MatchEnd::Completed(Outcome::Win(_)) => {
                self.losses += 1;
                0.0
            }
            MatchEnd::Completed(Outcome::Draw) => {
                self.draws += 1;
                0.5
            }
            MatchEnd::ActionLimit => {
                self.truncations += 1;
                0.0
            }
        }
    }

    fn score(self) -> Score {
        let games = self.wins + self.losses + self.draws + self.truncations;
        let earned = self.wins as f64 + 0.5 * self.draws as f64;
        let lower = earned / games as f64;
        let upper = (earned + self.truncations as f64) / games as f64;
        Score {
            score: (self.truncations == 0).then_some(lower),
            score_bounds: [lower, upper],
            counts: self,
        }
    }
}

#[derive(Serialize)]
pub struct Score {
    #[serde(flatten)]
    pub counts: Counts,
    pub score: Option<f64>,
    pub score_bounds: [f64; 2],
}

impl Summary {
    pub fn from_report(config: &ExperimentConfig, report: &EvaluationReport) -> Self {
        let mut starts = Counts::default();
        let mut moves_second = Counts::default();
        let mut lengths = Vec::with_capacity(report.pairs.len() * 2);
        let mut pair_scores = Vec::with_capacity(report.pairs.len());
        for pair in &report.pairs {
            let mut pair_points = 0.0;
            for (policy_a_seat, game) in pair.iter().enumerate() {
                let counts = if policy_a_seat == report.config.starting_player.index() {
                    &mut starts
                } else {
                    &mut moves_second
                };
                pair_points += counts.record(game.end, policy_a_seat);
                lengths.push(game.action_count);
            }
            pair_scores.push(pair_points / 2.0);
        }

        let totals = report.totals;
        let overall = Counts {
            wins: totals.wins[0],
            losses: totals.wins[1],
            draws: totals.draws,
            truncations: totals.truncations,
        }
        .score();
        let paired_ci_95 = (totals.truncations == 0 && pair_scores.len() >= 2)
            .then(|| bootstrap_interval(&pair_scores, &config.bootstrap));
        lengths.sort_unstable();
        let middle = lengths.len() / 2;
        Self {
            board_layout: Board::LAYOUT_ID,
            policies: config.policies,
            deals: report.pairs.len(),
            overall,
            starts: starts.score(),
            moves_second: moves_second.score(),
            median_actions: (lengths[middle - 1] as f64 + lengths[middle] as f64) / 2.0,
            paired_ci_95,
        }
    }

    pub fn print(&self, output: &Path) {
        let [a, b] = self.policies;
        let score = &self.overall;
        println!(
            "{a} (A) vs {b} (B), {} paired deals\nA won {}, lost {}, drew {} and reached the limit in {} games",
            self.deals,
            score.counts.wins,
            score.counts.losses,
            score.counts.draws,
            score.counts.truncations
        );
        if let Some(value) = score.score {
            println!("A scored {:.2}%", value * 100.0);
            if let Some([lower, upper]) = self.paired_ci_95 {
                println!(
                    "Paired 95% bootstrap interval {:.2}% to {:.2}%",
                    lower * 100.0,
                    upper * 100.0
                );
            }
        } else {
            let [lower, upper] = score.score_bounds;
            println!(
                "A's score bounds with unresolved games {:.2}% to {:.2}%",
                lower * 100.0,
                upper * 100.0
            );
        }
        println!("Results written to {}", output.display());
    }
}

fn bootstrap_interval(pair_scores: &[f64], config: &BootstrapConfig) -> [f64; 2] {
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut distribution = Vec::with_capacity(config.samples.get());
    let count = pair_scores.len() as u64;
    for _ in 0..config.samples.get() {
        let total: f64 = (0..count)
            .map(|_| pair_scores[rng.random_range(0..count) as usize])
            .sum();
        distribution.push(total / count as f64);
    }
    distribution.sort_unstable_by(f64::total_cmp);
    [0.025, 0.975].map(|probability| {
        let rank = probability * (distribution.len() - 1) as f64;
        let lower = distribution[rank.floor() as usize];
        let upper = distribution[rank.ceil() as usize];
        lower + rank.fract() * (upper - lower)
    })
}
