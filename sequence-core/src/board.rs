use core::fmt;

use crate::Card;

// Accept compact card tokens (Ac, 2d, 10h, Ks) and _ for free spaces.
// Decode at compile time so invalid values or suits cannot enter the layout.
macro_rules! board_layout {
    ($($card:tt),* $(,)?) => {{
        [$(Card::from_str(stringify!($card))),*]
    }};
}

// Classic 10 x 10 board, stored row by row from the top-left corner.
#[rustfmt::skip]
const CARD_BY_POSITION: [Option<Card>; BOARD_SIZE] = board_layout![
      _, 10s,  Qs,  Ks,  As,  2d,  3d,  4d,  5d,   _,
     9s, 10h,  9h,  8h,  7h,  6h,  5h,  4h,  3h,  6d,
     8s,  Qh,  7d,  8d,  9d, 10d,  Qd,  Kd,  2h,  7d,
     7s,  Kh,  6d,  2s,  Ah,  Kh,  Qh,  Ad,  2c,  8d,
     6s,  Ah,  5d,  3s,  4h,  3h, 10h,  Ac,  3c,  9d,
     5s,  2c,  4d,  4s,  5h,  2h,  9h,  Kc,  4c, 10d,
     4s,  3c,  3d,  5s,  6h,  7h,  8h,  Qc,  5c,  Qd,
     3s,  4c,  2d,  6c,  7c,  8c,  9c, 10c,  6s,  Kd,
     2s,  5c,  As,  Ks,  Qs, 10s,  9s,  8s,  7s,  Ad,
      _,  6c,  7c,  8c,  9c, 10c,  Qc,  Kc,  Ac,   _,
];

pub const BOARD_SIZE: usize = Position::COUNT;

pub const fn get_card(position: Position) -> Option<Card> {
    CARD_BY_POSITION[position.id()]
}

/// A square's contents. Free corners count toward either player's sequences.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cell {
    Empty,
    Corner,
    Occupied(Color),
}

/// A claimed, ordered line of five squares.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Sequence([Position; 5]);

impl Sequence {
    pub fn positions(&self) -> &[Position; 5] {
        &self.0
    }
}

/// Immutable public board state, including the identities of claimed sequences.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Board {
    cells: [u8; BOARD_SIZE / 2],
    sequences: [Vec<Sequence>; 2],
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Position(u8);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Color {
    Red = 0,
    Blue = 1,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PositionError {
    InvalidRank(u8),
    InvalidFile(u8),
    InvalidIndex(usize),
}

impl Color {
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::Red => Self::Blue,
            Self::Blue => Self::Red,
        }
    }
}

impl Position {
    pub const MAX_RANK: u8 = 9;
    pub const MAX_FILE: u8 = 9;
    pub const COUNT: usize = 100;
    pub const CORNERS: [Self; 4] = [Self(0), Self(9), Self(90), Self(99)];

    pub const fn new(rank: u8, file: u8) -> Result<Self, PositionError> {
        if rank > Self::MAX_RANK {
            return Err(PositionError::InvalidRank(rank));
        }
        if file > Self::MAX_FILE {
            return Err(PositionError::InvalidFile(file));
        }
        Ok(Self(rank * 10 + file))
    }

    pub const fn from_index(index: usize) -> Result<Self, PositionError> {
        if index >= Self::COUNT {
            return Err(PositionError::InvalidIndex(index));
        }
        Ok(Self(index as u8))
    }

    pub fn all() -> impl ExactSizeIterator<Item = Self> + DoubleEndedIterator + Clone {
        (0..Self::COUNT as u8).map(Self)
    }

    pub const fn rank(self) -> u8 {
        self.0 / 10
    }

    pub const fn file(self) -> u8 {
        self.0 % 10
    }

    pub const fn id(self) -> usize {
        self.0 as usize
    }

    pub const fn offset(self, dy: i8, dx: i8) -> Result<Self, PositionError> {
        Self::new(
            self.rank().wrapping_add_signed(dy),
            self.file().wrapping_add_signed(dx),
        )
    }
}

pub(crate) const DIRS: [(i8, i8); 4] = [(0, 1), (1, 0), (1, 1), (-1, 1)];

// Jack entries have no positions; all other cards have exactly two.
const POSITIONS_BY_CARD: [[Option<Position>; 2]; Card::COUNT] = {
    let mut result = [[None; 2]; Card::COUNT];
    let mut index = 0;
    while index < BOARD_SIZE {
        if let Some(card) = CARD_BY_POSITION[index] {
            let slots = &mut result[card.id()];
            if slots[0].is_none() {
                slots[0] = Some(Position(index as u8));
            } else {
                assert!(slots[1].is_none(), "card occurs more than twice");
                slots[1] = Some(Position(index as u8));
            }
        }
        index += 1;
    }
    result
};

/// Board squares corresponding to this card (empty for jacks).
pub fn positions_for_card(card: Card) -> impl Iterator<Item = Position> {
    POSITIONS_BY_CARD[card.id()].into_iter().flatten()
}

/// A chip cannot be placed on an occupied square or a free corner.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OccupiedPosition(pub Position);

impl fmt::Display for OccupiedPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "square {} is occupied", self.0)
    }
}
impl std::error::Error for OccupiedPosition {}

impl Board {
    pub(crate) fn new() -> Self {
        let mut board = Self {
            cells: [0; BOARD_SIZE / 2],
            sequences: Default::default(),
        };
        for corner in Position::CORNERS {
            board.write(corner, 3);
        }
        board
    }

    pub fn cell(&self, position: Position) -> Cell {
        match (self.cells[position.id() / 2] >> (4 * (position.id() % 2))) & 0xf {
            0 => Cell::Empty,
            1 => Cell::Occupied(Color::Red),
            2 => Cell::Occupied(Color::Blue),
            3 => Cell::Corner,
            _ => unreachable!("private cell representation"),
        }
    }

    /// Includes free corners, which count for both colors.
    pub fn is_owned_by(&self, position: Position, color: Color) -> bool {
        matches!(self.cell(position), Cell::Corner) || self.cell(position) == Cell::Occupied(color)
    }

    pub fn sequences(&self, color: Color) -> &[Sequence] {
        &self.sequences[color as usize]
    }

    pub fn is_in_sequence(&self, position: Position) -> bool {
        self.sequences
            .iter()
            .flatten()
            .any(|seq| seq.0.contains(&position))
    }

    pub fn is_protected(&self, position: Position) -> bool {
        self.cell(position) == Cell::Corner || self.is_in_sequence(position)
    }

    fn write(&mut self, position: Position, value: u8) {
        let offset = 4 * (position.id() % 2);
        let slot = &mut self.cells[position.id() / 2];
        *slot = (*slot & !(0xf << offset)) | (value << offset);
    }

    pub(crate) fn remove(&mut self, position: Position) {
        self.write(position, 0);
    }

    /// Places a chip and claims newly completed sequences.
    /// Returns an error without mutation if the square is occupied.
    pub fn place_chip(&mut self, position: Position, color: Color) -> Result<(), OccupiedPosition> {
        if self.cell(position) != Cell::Empty {
            return Err(OccupiedPosition(position));
        }
        self.place(position, color);
        Ok(())
    }

    pub(crate) fn place(&mut self, position: Position, color: Color) {
        self.write(position, 1 << color as u8);
        self.claim_sequences(position, color);
    }

    fn claim_sequences(&mut self, position: Position, color: Color) {
        // Only windows containing the new chip can become newly claimable.
        for (dy, dx) in DIRS {
            for offset in (0..5).rev() {
                let Ok(start) = position.offset(-dy * offset, -dx * offset) else {
                    continue;
                };
                let mut positions = [start; 5];
                let mut valid = true;
                for (i, slot) in positions.iter_mut().enumerate() {
                    match start.offset(dy * i as i8, dx * i as i8) {
                        Ok(p) if self.is_owned_by(p, color) => *slot = p,
                        _ => {
                            valid = false;
                            break;
                        }
                    }
                }
                if valid
                    && self.sequences(color).iter().all(|claimed| {
                        positions.iter().filter(|p| claimed.0.contains(p)).count() <= 1
                    })
                {
                    self.sequences[color as usize].push(Sequence(positions));
                }
            }
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Red => "Red",
            Self::Blue => "Blue",
        })
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.rank(), self.file())
    }
}

impl fmt::Display for PositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRank(v) => write!(f, "invalid rank {v}"),
            Self::InvalidFile(v) => write!(f, "invalid file {v}"),
            Self::InvalidIndex(v) => write!(f, "invalid board index {v}"),
        }
    }
}

impl std::error::Error for PositionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CardRank, CardSuit};
    use strum::VariantArray;

    fn p(row: u8, col: u8) -> Position {
        Position::new(row, col).unwrap()
    }

    fn chips(board: &mut Board, positions: impl IntoIterator<Item = Position>, color: Color) {
        for position in positions {
            board.write(position, 1 << color as u8);
        }
    }

    #[test]
    fn positions_are_checked_and_roundtrip() {
        assert_eq!(Position::all().len(), BOARD_SIZE);
        for position in Position::all() {
            assert_eq!(Position::from_index(position.id()).unwrap(), position);
            assert_eq!(
                Position::new(position.rank(), position.file()).unwrap(),
                position
            );
        }
        assert!(Position::new(0, 10).is_err());
        assert!(Position::new(10, 0).is_err());
        assert!(Position::from_index(100).is_err());
        assert!(Position::from_index(usize::MAX).is_err());
        assert!(p(0, 0).offset(-1, 0).is_err());
        assert!(p(9, 9).offset(1, 0).is_err());
    }

    #[test]
    fn cells_are_isolated_and_corners_are_shared() {
        let mut board = Board::new();
        for corner in Position::CORNERS {
            assert_eq!(board.cell(corner), Cell::Corner);
            assert!(board.is_owned_by(corner, Color::Red));
            assert!(board.is_owned_by(corner, Color::Blue));
            assert!(board.is_protected(corner));
            assert!(!board.is_in_sequence(corner));
        }
        for position in Position::all().filter(|p| !Position::CORNERS.contains(p)) {
            let before = board.clone();
            board.write(position, 1);
            for other in Position::all().filter(|&other| other != position) {
                assert_eq!(board.cell(other), before.cell(other));
            }
            board.remove(position);
            assert_eq!(board, before);
        }
    }

    #[test]
    fn board_cards_have_exactly_two_positions_and_jacks_have_none() {
        for &rank in CardRank::VARIANTS {
            for &suit in CardSuit::VARIANTS {
                let card = Card::new(rank, suit);
                let positions: Vec<_> = positions_for_card(card).collect();
                assert_eq!(positions.len(), if rank == CardRank::Jack { 0 } else { 2 });
                for position in positions {
                    assert_eq!(get_card(position), Some(card));
                }
                assert_eq!(
                    card.is_one_eyed_jack(),
                    rank == CardRank::Jack && matches!(suit, CardSuit::Hearts | CardSuit::Spades)
                );
                assert_eq!(
                    card.is_two_eyed_jack(),
                    rank == CardRank::Jack && matches!(suit, CardSuit::Clubs | CardSuit::Diamonds)
                );
            }
        }
    }

    #[test]
    fn claims_all_directions_and_free_corners() {
        for (start, dy, dx) in [
            (p(0, 0), 0, 1),
            (p(0, 0), 1, 0),
            (p(0, 0), 1, 1),
            (p(9, 0), -1, 1),
        ] {
            let mut board = Board::new();
            chips(
                &mut board,
                (1..4).map(|i| start.offset(dy * i, dx * i).unwrap()),
                Color::Red,
            );
            board.place(start.offset(dy * 4, dx * 4).unwrap(), Color::Red);
            assert_eq!(board.sequences(Color::Red).len(), 1);
            assert_eq!(board.sequences(Color::Blue).len(), 0);
            assert!(board.is_in_sequence(start));
        }
    }

    #[test]
    fn bridge_claims_two_sequences_and_overlap_is_not_recounted() {
        let mut board = Board::new();
        chips(
            &mut board,
            (0..9).filter(|&col| col != 4).map(|col| p(4, col)),
            Color::Red,
        );
        board.place(p(4, 4), Color::Red);
        let seqs = board.sequences(Color::Red);
        assert_eq!(seqs.len(), 2);
        assert_eq!(
            seqs[0].positions(),
            &[p(4, 0), p(4, 1), p(4, 2), p(4, 3), p(4, 4)]
        );
        assert_eq!(
            seqs[1].positions(),
            &[p(4, 4), p(4, 5), p(4, 6), p(4, 7), p(4, 8)]
        );
        board.claim_sequences(p(4, 4), Color::Red);
        assert_eq!(board.sequences(Color::Red).len(), 2);
    }

    #[test]
    fn slides_past_overlapping_windows_and_claims_crossings() {
        let mut board = Board::new();
        chips(&mut board, (0..4).map(|col| p(4, col)), Color::Red);
        board.place(p(4, 4), Color::Red);
        chips(&mut board, (5..8).map(|col| p(4, col)), Color::Red);
        board.place(p(4, 8), Color::Red);
        assert_eq!(board.sequences(Color::Red).len(), 2);
        chips(&mut board, (0..4).map(|row| p(row, 8)), Color::Red);
        board.claim_sequences(p(4, 8), Color::Red);
        assert_eq!(board.sequences(Color::Red).len(), 3);
    }

    #[test]
    fn distinct_claims_may_be_crossed_on_adjacent_squares() {
        let mut board = Board::new();
        for col in [1, 2] {
            chips(&mut board, (0..4).map(|row| p(row, col)), Color::Red);
            board.place(p(4, col), Color::Red);
        }
        chips(&mut board, [p(4, 0), p(4, 3)], Color::Red);
        board.place(p(4, 4), Color::Red);
        assert_eq!(board.sequences(Color::Red).len(), 3);
    }
}
