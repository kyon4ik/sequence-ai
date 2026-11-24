use std::path::Path;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::{ThreadPool, prelude::*};
use serde::Serialize;

use crate::{
    Result,
    experiment::{AgentKind, GameEnd},
};

use super::{Config, Evaluations, Matches, OPPONENTS, Run, Score, probe::Tactic};

#[derive(Serialize)]
pub(super) struct TacticalCounts {
    tactic: Tactic,
    pub positions: usize,
    pub ranked_first: usize,
}

impl TacticalCounts {
    pub(super) fn rate(&self) -> Option<f64> {
        (self.positions != 0).then(|| self.ranked_first as f64 / self.positions as f64)
    }
}

pub(super) fn tactical_counts(scores: &[Score], tactic: Tactic) -> TacticalCounts {
    let mut counts = TacticalCounts {
        tactic,
        positions: 0,
        ranked_first: 0,
    };
    for score in scores.iter().filter(|score| score.tactic == tactic) {
        counts.positions += 1;
        counts.ranked_first += usize::from(score.best_tactical_rank == 1);
    }
    counts
}

#[derive(Clone, Copy, Default)]
struct Counts {
    wins: usize,
    completed: usize,
}

impl std::ops::AddAssign for Counts {
    fn add_assign(&mut self, other: Self) {
        self.wins += other.wins;
        self.completed += other.completed;
    }
}

impl Counts {
    fn rate(self) -> Option<f64> {
        (self.completed > 0).then(|| self.wins as f64 / self.completed as f64)
    }
}

#[derive(Serialize)]
pub(super) struct Comparison {
    arm: String,
    opponent: AgentKind,
    pub seeds: Vec<SeedRates>,
    pub best_mean: Option<f64>,
    pub last_mean: Option<f64>,
    pub best_minus_last: Option<f64>,
    pub deal_paired_ci_95: Option<[f64; 2]>,
}

#[derive(Serialize)]
pub(super) struct SeedRates {
    seed: u64,
    pub best: Option<f64>,
    pub last: Option<f64>,
}

pub(super) fn write_comparisons(path: &Path, comparisons: &[Comparison]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "arm",
        "opponent",
        "best_mean",
        "last_mean",
        "best_minus_last",
        "deal_paired_ci_95_low",
        "deal_paired_ci_95_high",
    ])?;
    for row in comparisons {
        let [low, high] = row
            .deal_paired_ci_95
            .map_or([None; 2], |limits| limits.map(Some));
        writer.serialize((
            &row.arm,
            row.opponent,
            row.best_mean,
            row.last_mean,
            row.best_minus_last,
            low,
            high,
        ))?;
    }
    writer.flush()?;
    Ok(())
}

pub(super) fn compare_all(
    pool: &ThreadPool,
    config: &Config,
    runs: &[Run],
    best: &[usize],
    holdout: &Evaluations<Matches>,
) -> Vec<Comparison> {
    let tasks: Vec<_> = (0..config.arms.len())
        .flat_map(|arm| (0..OPPONENTS.len()).map(move |opponent| (arm, opponent)))
        .collect();
    pool.install(|| {
        tasks
            .par_iter()
            .map(|&(arm, opponent)| compare(config, runs, best, holdout, arm, opponent))
            .collect()
    })
}

fn compare(
    config: &Config,
    runs: &[Run],
    best: &[usize],
    holdout: &Evaluations<Matches>,
    arm: usize,
    opponent: usize,
) -> Comparison {
    let deals = config.holdout.deal_count.get();
    let seed_count = config.seeds.len();
    let mut clusters = vec![Counts::default(); deals * seed_count * 2];
    let mut seeds = Vec::new();
    for (index, run) in runs.iter().enumerate().filter(|(_, run)| run.arm == arm) {
        let results = [
            &holdout[&(index, best[index])],
            &holdout[&(index, run.last())],
        ];
        let seed = seeds.len();
        for (endpoint, result) in results.into_iter().enumerate() {
            for game in result
                .games
                .iter()
                .filter(|game| game.opponent == OPPONENTS[opponent].0)
            {
                let deal = (game.shuffle_seed - config.holdout.first_shuffle_seed) as usize;
                let counts = &mut clusters[(deal * seed_count + seed) * 2 + endpoint];
                counts.wins += usize::from(matches!(
                    (game.outcome, game.learner_seat),
                    (GameEnd::PlayerOneWin, 1) | (GameEnd::PlayerTwoWin, 2)
                ));
                counts.completed += usize::from(game.outcome != GameEnd::ActionLimit);
            }
        }
        seeds.push(SeedRates {
            seed: run.seed,
            best: results[0].totals[opponent].win_rate,
            last: results[1].totals[opponent].win_rate,
        });
    }

    let best_mean = complete_mean(seeds.iter().map(|seed| seed.best));
    let last_mean = complete_mean(seeds.iter().map(|seed| seed.last));
    let best_minus_last = best_mean.zip(last_mean).map(|(best, last)| best - last);
    let deal_paired_ci_95 = best_minus_last.and_then(|_| bootstrap(&clusters, config));
    Comparison {
        arm: config.arms[arm].directory.clone(),
        opponent: OPPONENTS[opponent].0,
        seeds,
        best_mean,
        last_mean,
        best_minus_last,
        deal_paired_ci_95,
    }
}

pub(super) fn complete_mean(rates: impl ExactSizeIterator<Item = Option<f64>>) -> Option<f64> {
    let count = rates.len();
    (count > 0).then_some(())?;
    let sum = rates.sum::<Option<f64>>()?;
    Some(sum / count as f64)
}

fn bootstrap(clusters: &[Counts], config: &Config) -> Option<[f64; 2]> {
    let deals = config.holdout.deal_count.get();
    let seeds = config.seeds.len();
    if deals < 2 {
        return None;
    }

    let replicates = config.bootstrap_replicates.get();
    let mut samples = Vec::with_capacity(replicates);
    let mut totals = vec![Counts::default(); seeds * 2];
    let mut rng = ChaCha8Rng::seed_from_u64(config.bootstrap_seed);
    for _ in 0..replicates {
        totals.fill(Counts::default());
        for _ in 0..deals {
            let offset = rng.random_range(0..deals) * totals.len();
            for (total, &count) in totals.iter_mut().zip(&clusters[offset..]) {
                *total += count;
            }
        }
        if let Some(mean) = complete_mean(
            totals
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| Some(pair[0].rate()? - pair[1].rate()?)),
        ) {
            samples.push(mean);
        }
    }
    if samples.len() != replicates {
        return None;
    }
    samples.sort_unstable_by(f64::total_cmp);
    Some([0.025, 0.975].map(|q| samples[(q * (samples.len() - 1) as f64).round() as usize]))
}
