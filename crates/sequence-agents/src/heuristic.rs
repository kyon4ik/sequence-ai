use std::{cmp::Ordering, sync::LazyLock};

use rand::{SeedableRng, seq::IndexedRandom};
use rand_chacha::ChaCha8Rng;
use sequence_game::{
    Action, BOARD_SIZE, PlayerView, Position, Rank, SEQUENCE_LENGTH, Sequence, Space,
};

use crate::Policy;

const OPEN_LINE_WEIGHTS: [i32; SEQUENCE_LENGTH + 1] = [0, 64, 16, 4, 1, 0];
const _: () = assert!(BOARD_SIZE * BOARD_SIZE <= u128::BITS as usize);

static LINE_MASKS: LazyLock<Vec<u128>> =
    LazyLock::new(|| Sequence::all().map(|line| line_mask(&line)).collect());

pub struct HeuristicAgent {
    rng: ChaCha8Rng,
}

impl HeuristicAgent {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

#[derive(Eq, Ord, PartialEq, PartialOrd)]
struct PlayScore {
    wins: bool,
    new_claims: usize,
    open_line_advantage: i32,
    saves_jack: bool,
}

impl Policy for HeuristicAgent {
    fn choose_action(&mut self, view: &PlayerView<'_>) -> Option<Action> {
        match view.legal_actions.as_slice() {
            [] => return None,
            [action] => return Some(*action),
            _ => {}
        }

        if let Some(&exchange) = view
            .legal_actions
            .iter()
            .find(|action| matches!(action, Action::Exchange(_)))
        {
            return Some(exchange);
        }

        let mut chips = [0; 2];
        for position in Position::all() {
            if let Space::Labeled {
                chip: Some(player), ..
            } = view.board.space(position)
            {
                chips[player.index()] |= position_mask(position);
            }
        }
        let windows = view.claimed_sequences.map(|claimed| {
            let claims: Vec<_> = claimed.iter().map(line_mask).collect();
            LINE_MASKS
                .iter()
                .copied()
                .filter(|&window| compatible(window, &claims))
                .collect()
        });
        let mut best_score = None;
        let mut best_actions = Vec::new();

        for &action in &view.legal_actions {
            let Some(score) = score_play(view, action, &windows, chips) else {
                continue;
            };

            match Some(&score).cmp(&best_score.as_ref()) {
                Ordering::Greater => {
                    best_score = Some(score);
                    best_actions.clear();
                    best_actions.push(action);
                }
                Ordering::Equal => best_actions.push(action),
                Ordering::Less => {}
            }
        }

        best_actions.choose(&mut self.rng).copied()
    }
}

fn score_play(
    view: &PlayerView<'_>,
    action: Action,
    windows: &[Vec<u128>; 2],
    mut chips: [u128; 2],
) -> Option<PlayScore> {
    let Action::Play { card, target } = action else {
        return None;
    };
    let preview = view.preview_play(action)?;

    // A play changes occupancy only at its target.
    let bit = position_mask(target);
    chips = chips.map(|mask| mask & !bit);
    if let Space::Labeled {
        chip: Some(player), ..
    } = preview.board.space(target)
    {
        chips[player.index()] |= bit;
    }

    let own = view.player.index();
    let other = view.player.opponent().index();
    let new_claims: Vec<_> = preview.new_claims.iter().map(line_mask).collect();
    let own_score = open_line_score(&windows[own], chips[own], chips[other], &new_claims);
    let opponent_score = open_line_score(&windows[other], chips[other], chips[own], &[]);

    Some(PlayScore {
        wins: preview.wins,
        new_claims: preview.new_claims.len(),
        open_line_advantage: own_score - opponent_score,
        saves_jack: card.rank != Rank::Jack,
    })
}

fn position_mask(position: Position) -> u128 {
    1 << (position.row() * BOARD_SIZE + position.column())
}

fn line_mask(line: &Sequence) -> u128 {
    // A corner is already filled for both players and never counts toward overlap.
    line.spaces()
        .iter()
        .copied()
        .filter(|position| !position.is_corner())
        .map(position_mask)
        .fold(0, |mask, bit| mask | bit)
}

fn compatible(window: u128, claims: &[u128]) -> bool {
    claims
        .iter()
        .all(|&claim| (window & claim).count_ones() <= 1)
}

fn open_line_score(windows: &[u128], own: u128, other: u128, new_claims: &[u128]) -> i32 {
    windows
        .iter()
        .copied()
        .filter(|&window| window & other == 0 && compatible(window, new_claims))
        .map(|window| OPEN_LINE_WEIGHTS[(window & !own).count_ones() as usize])
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_masks_match_the_game_overlap_rule() {
        let lines: Vec<_> = Sequence::all().collect();
        for line in &lines {
            for claim in &lines {
                assert_eq!(
                    compatible(line_mask(line), &[line_mask(claim)]),
                    line.compatible_with(claim),
                );
            }
        }
    }
}
