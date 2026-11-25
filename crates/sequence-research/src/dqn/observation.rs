use sequence_game::{
    Action, BOARD_SIZE, Card, PLAY_HISTORY_LENGTH, PlayerView, Position, Rank, Space,
};
use serde::{Deserialize, Serialize};

pub const BOARD_CHANNELS: usize = 5;
pub const SQUARES: usize = BOARD_SIZE * BOARD_SIZE;
const PLAYABLE_SQUARES: usize = SQUARES - 4;
const CARDS: usize = 52;
const NON_JACK_RANKS: usize = 12;
const SUITS: usize = 4;

pub const CONTEXT_FEATURES: usize = 3 * CARDS + 6;
pub const HISTORY_FEATURES: usize = PLAY_HISTORY_LENGTH * (CARDS + SQUARES);
pub const ACTIONS: usize = (1 + SUITS) * PLAYABLE_SQUARES + SUITS * NON_JACK_RANKS;

pub fn action_index(action: Action) -> Option<usize> {
    match action {
        Action::Play { card, target } if !target.is_corner() => {
            let plane = if card.rank == Rank::Jack {
                1 + card.suit as usize
            } else {
                0
            };
            let square = square_index(target);
            let earlier_corners = [0, BOARD_SIZE - 1, SQUARES - BOARD_SIZE, SQUARES - 1]
                .into_iter()
                .filter(|&corner| corner < square)
                .count();
            Some(plane * PLAYABLE_SQUARES + square - earlier_corners)
        }
        Action::Exchange(card) if card.rank != Rank::Jack => Some(
            (1 + SUITS) * PLAYABLE_SQUARES
                + card.suit as usize * NON_JACK_RANKS
                + card.rank as usize,
        ),
        _ => None,
    }
}

fn square_index(position: Position) -> usize {
    position.row() * BOARD_SIZE + position.column()
}

fn card_index(card: Card) -> usize {
    card.suit as usize * (NON_JACK_RANKS + 1) + card.rank as usize
}

fn card_counts(cards: &[Card]) -> [f32; CARDS] {
    let mut counts = [0.0; CARDS];
    for &card in cards {
        counts[card_index(card)] += 0.5;
    }
    counts
}

#[derive(Serialize, Deserialize)]
pub struct Observation {
    #[serde(with = "super::snapshot::array")]
    pub board: [f32; BOARD_CHANNELS * SQUARES],
    #[serde(with = "super::snapshot::array")]
    pub features: [f32; CONTEXT_FEATURES],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    recent_plays: Option<[Option<(usize, usize)>; PLAY_HISTORY_LENGTH]>,
}

impl Observation {
    pub fn new(view: &PlayerView<'_>, include_history: bool) -> Self {
        let player = view.player;
        let own = player.index();
        let other = player.opponent().index();
        let hand = card_counts(view.hand);
        let mut planes = [[0.0; SQUARES]; BOARD_CHANNELS];

        for position in Position::all() {
            let square = square_index(position);
            match view.board.space(position) {
                Space::Corner => {
                    planes[0][square] = 1.0;
                    planes[1][square] = 1.0;
                }
                Space::Labeled { card, chip } => {
                    planes[0][square] = f32::from(chip == Some(player));
                    planes[1][square] = f32::from(chip == Some(player.opponent()));
                    planes[4][square] = hand[card_index(card)];
                }
            }
        }

        for (seat, plane) in [(own, 2), (other, 3)] {
            for sequence in view.claimed_sequences[seat] {
                for &position in sequence.spaces() {
                    if !position.is_corner() {
                        planes[plane][square_index(position)] = 1.0;
                    }
                }
            }
        }

        let mut features = [0.0; CONTEXT_FEATURES];
        features[..CARDS].copy_from_slice(&hand);
        features[CARDS..2 * CARDS].copy_from_slice(&card_counts(view.discard_piles[own]));
        features[2 * CARDS..3 * CARDS].copy_from_slice(&card_counts(view.discard_piles[other]));
        features[3 * CARDS..].copy_from_slice(&[
            view.claimed_sequences[own].len() as f32 / 2.0,
            view.claimed_sequences[other].len() as f32 / 2.0,
            view.hand_sizes[other] as f32 / 7.0,
            view.draw_pile_size as f32 / 104.0,
            f32::from(view.consecutive_passes) / 3.0,
            f32::from(
                view.legal_actions
                    .iter()
                    .any(|action| matches!(action, Action::Exchange(_))),
            ),
        ]);

        Self {
            board: std::array::from_fn(|index| planes[index / SQUARES][index % SQUARES]),
            features,
            recent_plays: include_history.then(|| {
                view.opponent_plays
                    .map(|play| play.map(|play| (card_index(play.card), square_index(play.target))))
            }),
        }
    }

    pub fn write_history(&self, features: &mut [f32]) {
        if let Some(plays) = &self.recent_plays {
            let (slots, _) = features.as_chunks_mut::<{ CARDS + SQUARES }>();
            for (slot, play) in slots.iter_mut().zip(plays) {
                if let Some((card, target)) = *play {
                    slot[card] = 1.0;
                    slot[CARDS + target] = 1.0;
                }
            }
        }
    }
}

pub fn legal_mask(view: &PlayerView<'_>) -> [bool; ACTIONS] {
    let mut mask = [false; ACTIONS];
    for index in view
        .legal_actions
        .iter()
        .filter_map(|&action| action_index(action))
    {
        mask[index] = true;
    }
    mask
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use sequence_game::{Game, GameConfig, Player};

    use super::*;

    #[test]
    fn each_action_has_its_own_output_and_pass_has_none() {
        let game = Game::new(GameConfig {
            shuffle_seed: 0,
            starting_player: Player::One,
            policies: Default::default(),
        });
        let view = game.view(Player::One);
        let mut indices = HashSet::new();

        for target in Position::all() {
            let Space::Labeled { card, .. } = view.board.space(target) else {
                continue;
            };
            for card in
                std::iter::once(card).chain(Card::all().filter(|card| card.rank == Rank::Jack))
            {
                assert!(indices.insert(action_index(Action::Play { card, target }).unwrap()));
            }
        }
        for card in Card::all().filter(|card| card.rank != Rank::Jack) {
            assert!(indices.insert(action_index(Action::Exchange(card)).unwrap()));
        }

        assert_eq!(ACTIONS, 528);
        assert_eq!(indices, (0..528).collect());
        assert_eq!(action_index(Action::Pass), None);
    }
}
