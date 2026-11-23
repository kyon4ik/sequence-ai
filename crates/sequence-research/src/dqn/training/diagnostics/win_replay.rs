use burn::tensor::Device;
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::{collections::VecDeque, num::NonZeroUsize, path::Path};

use super::{Coverage, Experience};
use crate::{
    Result,
    dqn::{model::Network, observation::ACTIONS, probe::rank_actions},
};

pub(super) struct WinReplay {
    pub(super) minimum: usize,
    rng: ChaCha8Rng,
    pub(super) batches: Vec<Vec<Experience>>,
    pub(super) sampled: Coverage,
    pub(super) shortfall_batches: usize,
}

fn selected_win(e: &Experience) -> bool {
    e.state.winning_actions.contains(&e.action)
}

impl WinReplay {
    pub(super) fn new(minimum: NonZeroUsize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        rng.set_stream(2);
        Self {
            minimum: minimum.get(),
            rng,
            batches: Vec::new(),
            sampled: Coverage::default(),
            shortfall_batches: 0,
        }
    }

    pub(super) fn record(&mut self, uniform: &[&Experience], replay: &VecDeque<Experience>) {
        let mut batch = uniform.to_vec();
        let mut replace: Vec<_> = batch
            .iter()
            .enumerate()
            .filter(|(_, e)| !selected_win(e))
            .map(|(i, _)| i)
            .collect();
        let present = batch.len() - replace.len();
        let needed = self.minimum.saturating_sub(present);
        if needed > 0 {
            let mut available: Vec<_> = replay
                .iter()
                .filter(|e| {
                    selected_win(e) && !uniform.iter().any(|&other| std::ptr::eq(*e, other))
                })
                .collect();
            let count = needed.min(available.len());
            self.shortfall_batches += usize::from(count < needed);
            let (slots, _) = replace.partial_shuffle(&mut self.rng, count);
            let (wins, _) = available.partial_shuffle(&mut self.rng, count);
            for (&slot, &mut win) in slots.iter().zip(wins) {
                batch[slot] = win;
            }
        }
        for e in &batch {
            self.sampled.observe(e);
        }
        self.batches.push(batch.into_iter().cloned().collect());
    }
}

#[derive(Serialize)]
struct ReplayRank {
    replay_index: usize,
    selected_win: bool,
    legal_actions: usize,
    winning_actions: usize,
    best_win_rank: usize,
    best_win_q: f32,
    best_alternative_q: f32,
}

pub(super) fn score_replay(
    network: &Network,
    device: &Device,
    replay: &VecDeque<Experience>,
    path: &Path,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    let opportunities: Vec<_> = replay
        .iter()
        .enumerate()
        .filter(|(_, e)| !e.state.winning_actions.is_empty())
        .collect();
    for batch in opportunities.chunks(64) {
        let data = network
            .forward(batch.iter().map(|(_, e)| &e.state.observation), device)
            .into_data();
        for (&(replay_index, e), scores) in batch
            .iter()
            .zip(data.as_slice::<f32>()?.as_chunks::<ACTIONS>().0)
        {
            let wins = &e.state.winning_actions;
            let mut candidates: Vec<_> = (0..ACTIONS)
                .filter(|&i| e.state.legal[i])
                .map(|i| (i, wins.contains(&i)))
                .collect();
            let Some(ranking) = rank_actions(scores, &mut candidates) else {
                continue;
            };
            writer.serialize(ReplayRank {
                replay_index,
                selected_win: selected_win(e),
                legal_actions: candidates.len(),
                winning_actions: wins.len(),
                best_win_rank: ranking.rank,
                best_win_q: ranking.best_tactical_q,
                best_alternative_q: ranking.best_alternative_q,
            })?;
        }
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, rc::Rc};

    use super::*;
    use crate::dqn::{
        observation::{BOARD_CHANNELS, CONTEXT_FEATURES, Observation, SQUARES},
        training::Decision,
    };

    #[test]
    fn fills_only_missing_wins_and_keeps_replay_rows_distinct() {
        let state = Rc::new(Decision {
            observation: Observation {
                board: [0.0; BOARD_CHANNELS * SQUARES],
                features: [0.0; CONTEXT_FEATURES],
            },
            legal: [true; ACTIONS],
            winning_actions: vec![0, 1, 2],
            potential: 0.0,
        });
        let replay: VecDeque<_> = (0..6)
            .map(|action| Experience::new(state.clone(), action, 0.0, None, 0.99))
            .collect();
        let uniform = [&replay[0], &replay[3], &replay[4], &replay[5]];

        for (minimum, expected_wins, shortfalls) in [(1, 1, 0), (2, 2, 0), (4, 3, 1)] {
            let mut sampler = WinReplay::new(NonZeroUsize::new(minimum).unwrap(), 42);
            sampler.record(&uniform, &replay);
            let batch = &sampler.batches[0];
            assert_eq!(batch.len(), uniform.len());
            assert_eq!(batch[0].action, 0);
            assert_eq!(
                batch.iter().map(|e| e.action).collect::<HashSet<_>>().len(),
                batch.len()
            );
            assert_eq!(sampler.sampled.win_selected, expected_wins);
            assert_eq!(sampler.shortfall_batches, shortfalls);
            for (original, sampled) in uniform.iter().zip(batch) {
                if !selected_win(sampled) {
                    assert_eq!(sampled.action, original.action);
                }
            }
        }
    }
}
