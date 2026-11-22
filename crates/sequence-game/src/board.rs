use crate::card::BOARD_LABEL_COUNT;
use crate::{Card, Player, Rank, Suit};

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

#[rustfmt::skip]
const TABLETOP_ROWS: [[&str; BOARD_SIZE]; BOARD_SIZE] = [
    [ "_", "10s",  "Qs",  "Ks",  "As",  "2d",  "3d",  "4d",  "5d",   "_"],
    ["9s", "10h",  "9h",  "8h",  "7h",  "6h",  "5h",  "4h",  "3h",  "6d"],
    ["8s",  "Qh",  "7d",  "8d",  "9d", "10d",  "Qd",  "Kd",  "2h",  "7d"],
    ["7s",  "Kh",  "6d",  "2s",  "Ah",  "Kh",  "Qh",  "Ad",  "2c",  "8d"],
    ["6s",  "Ah",  "5d",  "3s",  "4h",  "3h", "10h",  "Ac",  "3c",  "9d"],
    ["5s",  "2c",  "4d",  "4s",  "5h",  "2h",  "9h",  "Kc",  "4c", "10d"],
    ["4s",  "3c",  "3d",  "5s",  "6h",  "7h",  "8h",  "Qc",  "5c",  "Qd"],
    ["3s",  "4c",  "2d",  "6c",  "7c",  "8c",  "9c", "10c",  "6s",  "Kd"],
    ["2s",  "5c",  "As",  "Ks",  "Qs", "10s",  "9s",  "8s",  "7s",  "Ad"],
    [ "_",  "6c",  "7c",  "8c",  "9c", "10c",  "Qc",  "Kc",  "Ac",   "_"],
];

struct BoardLayout {
    labels: [Option<Card>; SPACE_COUNT],
    card_positions: [[Position; 2]; BOARD_LABEL_COUNT],
}

const TABLETOP: BoardLayout = {
    let mut labels = [None; SPACE_COUNT];
    let mut card_positions = [[Position(0); 2]; BOARD_LABEL_COUNT];
    let mut counts = [0; BOARD_LABEL_COUNT];
    let mut index = 0;

    while index < SPACE_COUNT {
        let position = Position(index);
        let label = parse_label(TABLETOP_ROWS[position.row()][position.column()]);
        assert!(
            label.is_none() == position.is_corner(),
            "invalid free corner"
        );

        if let Some(card) = label {
            let card_index = card.label_index().expect("invalid board card");
            let count = counts[card_index];
            assert!(count < 2, "card appears more than twice");
            card_positions[card_index][count] = position;
            counts[card_index] += 1;
        }
        labels[index] = label;
        index += 1;
    }

    let mut card_index = 0;
    while card_index < BOARD_LABEL_COUNT {
        assert!(counts[card_index] == 2, "card appears fewer than twice");
        card_index += 1;
    }

    BoardLayout {
        labels,
        card_positions,
    }
};

const fn parse_label(label: &str) -> Option<Card> {
    let (rank, suit) = match label.as_bytes() {
        [b'_'] => return None,
        [b'1', b'0', suit] => (Rank::Ten, *suit),
        [rank, suit] => {
            let rank = match *rank {
                b'A' => Rank::Ace,
                b'2' => Rank::Two,
                b'3' => Rank::Three,
                b'4' => Rank::Four,
                b'5' => Rank::Five,
                b'6' => Rank::Six,
                b'7' => Rank::Seven,
                b'8' => Rank::Eight,
                b'9' => Rank::Nine,
                b'Q' => Rank::Queen,
                b'K' => Rank::King,
                _ => panic!("invalid board rank"),
            };
            (rank, *suit)
        }
        _ => panic!("invalid board label"),
    };
    let suit = match suit {
        b'c' => Suit::Clubs,
        b'd' => Suit::Diamonds,
        b'h' => Suit::Hearts,
        b's' => Suit::Spades,
        _ => panic!("invalid board suit"),
    };
    Some(Card::new(rank, suit))
}

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
    pub const LAYOUT_ID: &str = "tabletop";

    pub(crate) fn empty() -> Self {
        Self {
            chips: [None; SPACE_COUNT],
        }
    }

    pub fn space(&self, position: Position) -> Space {
        match TABLETOP.labels[position.0] {
            Some(card) => Space::Labeled {
                card,
                chip: self.chip(position),
            },
            None => Space::Corner,
        }
    }

    pub fn is_dead_card(&self, card: Card) -> bool {
        Self::matching_positions(card).is_some_and(|positions| {
            positions
                .iter()
                .all(|&position| self.chip(position).is_some())
        })
    }

    pub(crate) fn matching_positions(card: Card) -> Option<&'static [Position; 2]> {
        card.label_index()
            .map(|index| &TABLETOP.card_positions[index])
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
