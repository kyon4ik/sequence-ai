use crate::card::BOARD_LABEL_COUNT;
use crate::{Card, Player};

pub const BOARD_SIZE: usize = 10;
const SPACE_COUNT: usize = BOARD_SIZE * BOARD_SIZE;

/// A zero-based board coordinate, ordered from top to bottom and left to right.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Position(usize);

impl Position {
    const ALL: [Self; SPACE_COUNT] = {
        let mut positions = [Self(0); SPACE_COUNT];
        let mut index = 0;
        while index < SPACE_COUNT {
            positions[index] = Self(index);
            index += 1;
        }
        positions
    };

    pub const fn new(row: usize, column: usize) -> Option<Self> {
        if row < BOARD_SIZE && column < BOARD_SIZE {
            Some(Self(row * BOARD_SIZE + column))
        } else {
            None
        }
    }

    pub fn all() -> impl Iterator<Item = Self> {
        Self::ALL.iter().copied()
    }

    pub const fn row(self) -> usize {
        self.0 / BOARD_SIZE
    }

    pub const fn column(self) -> usize {
        self.0 % BOARD_SIZE
    }

    pub const fn is_corner(self) -> bool {
        (self.row() == 0 || self.row() == BOARD_SIZE - 1)
            && (self.column() == 0 || self.column() == BOARD_SIZE - 1)
    }

    pub(crate) fn offset(self, rows: isize, columns: isize) -> Option<Self> {
        Self::new(
            self.row().checked_add_signed(rows)?,
            self.column().checked_add_signed(columns)?,
        )
    }
}

// Lay out the first copy in row order and rotate it 180 degrees for the second copy.
const CARD_POSITIONS: [[Position; 2]; BOARD_LABEL_COUNT] = {
    let mut pairs = [[Position(0); 2]; BOARD_LABEL_COUNT];
    let mut index = 0;
    let mut label_index = 0;

    while index < SPACE_COUNT / 2 {
        if !Position(index).is_corner() {
            pairs[label_index] = [Position(index), Position(SPACE_COUNT - 1 - index)];
            label_index += 1;
        }
        index += 1;
    }

    pairs
};

const LABELS: [Option<Card>; SPACE_COUNT] = {
    let mut labels = [None; SPACE_COUNT];
    let mut index = 0;
    while index < CARD_POSITIONS.len() {
        let [first, second] = CARD_POSITIONS[index];
        let card = Card::from_label_index(index);
        labels[first.0] = Some(card);
        labels[second.0] = Some(card);
        index += 1;
    }
    labels
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Space {
    Corner,
    Labeled { card: Card, chip: Option<Player> },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Board {
    chips: [Option<Player>; SPACE_COUNT],
}

impl Board {
    pub(crate) fn empty() -> Self {
        Self {
            chips: [None; SPACE_COUNT],
        }
    }

    pub fn space(&self, position: Position) -> Space {
        match LABELS[position.0] {
            Some(card) => Space::Labeled {
                card,
                chip: self.chip(position),
            },
            None => Space::Corner,
        }
    }

    pub(crate) fn matching_positions(card: Card) -> Option<&'static [Position; 2]> {
        card.label_index().map(|index| &CARD_POSITIONS[index])
    }

    pub(crate) fn candidate_positions(card: Card) -> &'static [Position] {
        match Self::matching_positions(card) {
            Some(positions) => positions,
            None => &Position::ALL,
        }
    }

    pub(crate) fn chip(&self, position: Position) -> Option<Player> {
        self.chips[position.0]
    }

    pub(crate) fn set_chip(&mut self, position: Position, chip: Option<Player>) {
        debug_assert!(!position.is_corner());
        self.chips[position.0] = chip;
    }
}
