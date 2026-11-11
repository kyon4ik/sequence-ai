#![deny(unsafe_code)]
//! Two-player Sequence: Classic with jacks and Simple without jacks.

mod board;
mod card;
mod game;

pub use board::{
    BOARD_SIZE, Board, Cell, Color, OccupiedPosition, Position, PositionError, Sequence, get_card,
    positions_for_card,
};
pub use card::{Card, CardRank, CardSuit};
pub use game::{
    Action, ActionError, Classic, ClassicObservation, Game, GameOutcome, Simple, SimpleObservation,
};
