use std::path::Path;

use sequence_game::{Board, Outcome};
use sequence_match::{EvaluationReport, MatchEnd};
use serde::Serialize;
use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::experiment::{ExperimentConfig, PolicyKind};

#[derive(Serialize)]
pub struct Summary {
    pub board_layout: &'static str,
    pub policies: [PolicyKind; 2],
    pub deals: usize,
    pub overall: Score,
    pub starts: Score,
    pub moves_second: Score,
    pub median_actions: f64,
    pub paired_ci_95: Option<[f64; 2]>,
    pub starts_ci_95: Option<[f64; 2]>,
    pub moves_second_ci_95: Option<[f64; 2]>,
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
        let starter_index = report.config.starting_player.index();
        for pair in &report.pairs {
            let mut scores = [0.0; 2];
            for (policy_a_seat, game) in pair.iter().enumerate() {
                let counts = if policy_a_seat == starter_index {
                    &mut starts
                } else {
                    &mut moves_second
                };
                scores[policy_a_seat] = counts.record(game.end, policy_a_seat);
                lengths.push(game.action_count);
            }
            pair_scores.push(scores);
        }

        let totals = report.totals;
        let overall = Counts {
            wins: totals.wins[0],
            losses: totals.wins[1],
            draws: totals.draws,
            truncations: totals.truncations,
        }
        .score();
        let starts = starts.score();
        let moves_second = moves_second.score();
        let paired_ci_95 = confidence_interval(
            overall.score,
            pair_scores.iter().map(|&[one, two]| (one + two) / 2.0),
        );
        let starts_ci_95 = confidence_interval(
            starts.score,
            pair_scores.iter().map(|scores| scores[starter_index]),
        );
        let moves_second_ci_95 = confidence_interval(
            moves_second.score,
            pair_scores.iter().map(|scores| scores[1 - starter_index]),
        );

        lengths.sort_unstable();
        let middle = lengths.len() / 2;
        Self {
            board_layout: Board::LAYOUT_ID,
            policies: config.policies,
            deals: report.pairs.len(),
            overall,
            starts,
            moves_second,
            median_actions: (lengths[middle - 1] as f64 + lengths[middle] as f64) / 2.0,
            paired_ci_95,
            starts_ci_95,
            moves_second_ci_95,
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

        for (label, score, interval) in [
            ("Overall", &self.overall, self.paired_ci_95),
            ("Starts", &self.starts, self.starts_ci_95),
            ("Moves second", &self.moves_second, self.moves_second_ci_95),
        ] {
            if let Some(value) = score.score {
                print!("{label} {:.2}%", value * 100.0);
                if let Some([lower, upper]) = interval {
                    print!(
                        " (95% t-interval {:.2}% to {:.2}%)",
                        lower * 100.0,
                        upper * 100.0
                    );
                }
                println!();
            } else {
                let [lower, upper] = score.score_bounds;
                println!(
                    "{label} score bounds with unresolved games {:.2}% to {:.2}%",
                    lower * 100.0,
                    upper * 100.0
                );
            }
        }
        println!("Results written to {}", output.display());
    }
}

fn confidence_interval(
    mean: Option<f64>,
    scores: impl ExactSizeIterator<Item = f64>,
) -> Option<[f64; 2]> {
    let mean = mean?;
    let count = scores.len() as f64;
    if count < 2.0 {
        return None;
    }

    let degrees_of_freedom = count - 1.0;
    let squared_deviations: f64 = scores.map(|score| (score - mean).powi(2)).sum();
    let standard_error = (squared_deviations / degrees_of_freedom / count).sqrt();
    let critical_value = StudentsT::new(0.0, 1.0, degrees_of_freedom)
        .expect("at least two observations")
        .inverse_cdf(0.975);
    let margin = critical_value * standard_error;

    Some([(mean - margin).max(0.0), (mean + margin).min(1.0)])
}

#[cfg(test)]
mod tests {
    use super::confidence_interval;

    #[test]
    fn confidence_interval_uses_sample_variance_and_student_t() {
        let scores = [0.25, 0.5, 0.5, 0.75, 0.75];
        let [lower, upper] = confidence_interval(Some(0.55), scores.into_iter()).unwrap();

        assert!((lower - 0.290_287_341_579).abs() < 1e-9);
        assert!((upper - 0.809_712_658_421).abs() < 1e-9);
        assert_eq!(confidence_interval(Some(0.5), [0.5].into_iter()), None);
    }
}
