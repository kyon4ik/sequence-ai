use std::fmt;

use arrayvec::ArrayVec;
use rand::{SeedableRng, seq::SliceRandom};
use rand_xoshiro::Xoroshiro128PlusPlus;
use strum::VariantArray;

use crate::{Board, Card, CardRank, CardSuit, Color, Position};

mod classic;
mod simple;
pub use classic::{Classic, ClassicObservation};
pub use simple::{Simple, SimpleObservation};

/// Common game operations; variants own their state, observations, and rules.
/// Import this trait to use these methods on `Classic` and `Simple`.
pub trait Game {
    type Observation<'a>
    where
        Self: 'a;

    fn from_seed(seed: u64) -> Self
    where
        Self: Sized;
    fn new() -> Self
    where
        Self: Sized,
    {
        Self::from_seed(rand::random())
    }
    fn board(&self) -> &Board;
    fn turns(&self) -> u16;
    /// Advances after every completed turn, including a terminal turn.
    fn current_player(&self) -> Color;
    fn outcome(&self) -> GameOutcome;
    fn is_game_over(&self) -> bool {
        self.outcome() != GameOutcome::InProgress
    }
    fn observation(&self) -> Self::Observation<'_>;
    /// Replaces the buffer contents; terminal states return no actions.
    fn legal_actions(&self, output: &mut Vec<Action>);
    fn validate(&self, action: Action) -> Result<(), ActionError>;
    /// Rejected actions leave all state unchanged.
    fn apply(&mut self, action: Action) -> Result<(), ActionError>;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Action {
    Play {
        card_index: usize,
        position: Position,
    },
    Exchange {
        card_index: usize,
    },
    Pass,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GameOutcome {
    InProgress,
    Won(Color),
    Draw,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActionError {
    GameOver,
    InvalidCardIndex(usize),
    InvalidPlayPosition(Position),
    ExchangeUnavailable,
    CardNotDead,
    UnexpectedPass,
}

// Shared storage and mechanical operations, without variant-dependent behavior.
#[derive(Clone, Debug)]
struct State {
    board: Board,
    turns: u16,
    current_player: Color,
    hands: [ArrayVec<Card, 7>; 2],
    draw_pile: Vec<Card>,
    discard_pile: Vec<Card>,
    outcome: GameOutcome,
}

impl State {
    fn deal(
        seed: u64,
        ranks: impl Iterator<Item = CardRank> + Clone,
    ) -> (Self, Xoroshiro128PlusPlus) {
        let mut rng = Xoroshiro128PlusPlus::seed_from_u64(seed);
        let mut draw_pile = Vec::with_capacity(2 * Card::COUNT);
        for &suit in CardSuit::VARIANTS {
            for rank in ranks.clone() {
                draw_pile.extend([Card::new(rank, suit); 2]);
            }
        }
        draw_pile.shuffle(&mut rng);
        let mut hands: [ArrayVec<Card, 7>; 2] = Default::default();
        for _ in 0..7 {
            for hand in &mut hands {
                hand.push(draw_pile.pop().unwrap());
            }
        }
        (
            Self {
                board: Board::new(),
                turns: 0,
                current_player: Color::Red,
                hands,
                draw_pile,
                discard_pile: Vec::with_capacity(2 * Card::COUNT),
                outcome: GameOutcome::InProgress,
            },
            rng,
        )
    }

    fn hand_mut(&mut self) -> &mut ArrayVec<Card, 7> {
        &mut self.hands[self.current_player as usize]
    }
    fn discard(&mut self, card_index: usize) {
        let card = self.hand_mut().remove(card_index);
        self.discard_pile.push(card);
    }
    fn draw(&mut self) {
        if let Some(card) = self.draw_pile.pop() {
            self.hand_mut().push(card);
        }
    }
    fn finish_turn(&mut self, draw_limit: bool) {
        self.turns += 1;
        self.outcome = if self.board.sequences(self.current_player).len() >= 2 {
            GameOutcome::Won(self.current_player)
        } else if draw_limit || self.hands.iter().all(|hand| hand.is_empty()) {
            GameOutcome::Draw
        } else {
            GameOutcome::InProgress
        };
        self.current_player = self.current_player.next();
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Play {
                card_index,
                position,
            } => write!(f, "play card #{card_index} on {position}"),
            Self::Exchange { card_index } => write!(f, "exchange card #{card_index}"),
            Self::Pass => f.write_str("pass"),
        }
    }
}

impl fmt::Display for GameOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InProgress => f.write_str("in progress"),
            Self::Won(c) => write!(f, "{c} won"),
            Self::Draw => f.write_str("draw"),
        }
    }
}

impl fmt::Display for ActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GameOver => f.write_str("game is over"),
            Self::InvalidCardIndex(i) => write!(f, "invalid card index {i}"),
            Self::InvalidPlayPosition(p) => write!(f, "cannot play at {p}"),
            Self::ExchangeUnavailable => f.write_str("dead-card exchange is unavailable"),
            Self::CardNotDead => f.write_str("only dead ordinary cards may be exchanged"),
            Self::UnexpectedPass => f.write_str("a play or exchange is available"),
        }
    }
}

impl std::error::Error for ActionError {}

#[cfg(test)]
mod tests;
