use core::fmt;

use arrayvec::ArrayVec;

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

pub const BOARD_SIZE: usize = Position::MAX_INDEX;

pub const fn get_card(position: Position) -> Option<Card> {
    CARD_BY_POSITION[position.id()]
}

// Compressed masks
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Board {
    masks: [u8; BOARD_SIZE / 2],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BoardMask {
    mask: [bool; BOARD_SIZE],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Position(u8);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color {
    Red = 0,
    Blue = 1,
}

#[derive(Clone)]
pub struct PositionIter {
    index: u8,
}

#[derive(Debug)]
pub enum PositionError {
    InvalidRank(u8),
    InvalidFile(u8),
}

impl Color {
    #[must_use]
    pub fn next(&self) -> Self {
        match self {
            Self::Red => Self::Blue,
            Self::Blue => Self::Red,
        }
    }
}

impl PositionIter {
    pub fn new() -> Self {
        Self { index: 0 }
    }
}

impl Iterator for PositionIter {
    type Item = Position;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < Position::MAX_INDEX as u8 {
            let position = Position(self.index);
            self.index += 1;
            Some(position)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = Position::MAX_INDEX - self.index as usize;
        (len, Some(len))
    }
}

impl ExactSizeIterator for PositionIter {
    fn len(&self) -> usize {
        Position::MAX_INDEX - self.index as usize
    }
}

impl Position {
    pub const MAX_RANK: u8 = 9;
    pub const MAX_FILE: u8 = 9;

    pub const CORNERS: [Position; 4] = [
        Position::new_unchecked(0, 0),
        Position::new_unchecked(0, Self::MAX_FILE),
        Position::new_unchecked(Self::MAX_RANK, 0),
        Position::new_unchecked(Self::MAX_RANK, Self::MAX_FILE),
    ];

    // TODO: rename (it is one greater than max index)
    const MAX_INDEX: usize = ((Self::MAX_RANK + 1) * (Self::MAX_FILE + 1)) as usize;

    pub const fn new(rank: u8, file: u8) -> Result<Self, PositionError> {
        if rank > Self::MAX_RANK {
            return Err(PositionError::InvalidRank(rank));
        }

        if file > Self::MAX_FILE {
            return Err(PositionError::InvalidFile(file));
        }

        Ok(Self::new_unchecked(rank, file))
    }

    pub const fn new_unchecked(rank: u8, file: u8) -> Self {
        let inner = rank * (Self::MAX_FILE + 1) + file;
        Self(inner)
    }

    pub const fn rank(&self) -> u8 {
        self.0 / (Self::MAX_FILE + 1)
    }

    pub const fn file(&self) -> u8 {
        self.0 % (Self::MAX_FILE + 1)
    }

    pub const fn id(&self) -> usize {
        self.0 as usize
    }

    pub const fn offset(&self, dy: i8, dx: i8) -> Result<Self, PositionError> {
        let new_rank = self.rank().wrapping_add_signed(dy);
        let new_file = self.file().wrapping_add_signed(dx);
        Self::new(new_rank, new_file)
    }
}

impl Board {
    pub const RED_MASK: u8 = 1 << 0;
    pub const BLUE_MASK: u8 = 1 << 1;
    pub const SEQ_MASK: u8 = 1 << 2;
    pub const PROT_MASK: u8 = 1 << 3;

    const DIRS: [(i8, i8); 4] = [(0, 1), (1, 0), (1, 1), (-1, 1)];

    pub fn empty() -> Self {
        let masks = [0u8; 50];
        Self { masks }
    }

    pub fn get(&self, pos: Position, mask: u8) -> u8 {
        let slot = pos.id() >> 1;
        let offset = 4 * (pos.id() & 1);
        (self.masks[slot] >> offset) & mask
    }

    pub fn set(&mut self, pos: Position, mask: u8) {
        let slot = pos.id() >> 1;
        let offset = 4 * (pos.id() & 1);
        self.masks[slot] |= mask << offset;
    }

    pub fn unset(&mut self, pos: Position, mask: u8) {
        let slot = pos.id() >> 1;
        let offset = 4 * (pos.id() & 1);
        self.masks[slot] &= !(mask << offset);
    }

    pub fn mask(&self, mask: u8) -> BoardMask {
        let mut board = BoardMask::none();
        for position in PositionIter::new() {
            board.set(position, self.get(position, mask) == mask);
        }
        board
    }

    pub fn update_sequences(&mut self, position: Position, color: Color) -> u8 {
        let (rank, file) = (position.rank() as i8, position.file() as i8);
        let offsets = [
            file,
            rank,
            i8::min(rank, file),
            i8::min(Position::MAX_RANK as i8 - rank, file),
        ];
        let our = 1 << color as u8;

        let mut sequences = 0;
        for ((dy, dx), offset) in Self::DIRS.into_iter().zip(offsets) {
            let mut pos = position.offset(-dy * offset, -dx * offset);
            let mut run = ArrayVec::<Position, 5>::new();
            let mut in_seq = false;

            while let Ok(p) = pos {
                let bits = self.get(p, our | Self::SEQ_MASK);
                if bits == our {
                    run.push(p);
                    in_seq = false;
                } else if bits == our | Self::SEQ_MASK {
                    if in_seq {
                        run.clear();
                    }
                    run.push(p);
                    in_seq = true;
                } else {
                    run.clear();
                    in_seq = false;
                }

                if run.is_full() {
                    for pp in run.drain(..) {
                        self.set(pp, Self::SEQ_MASK | Self::PROT_MASK);
                    }
                    sequences += 1;
                    run.push(p);
                }

                pos = p.offset(dy, dx);
            }
        }
        sequences
    }

    pub fn count_sequences(&self, position: Position, color: Color) -> u8 {
        let (rank, file) = (position.rank() as i8, position.file() as i8);
        let offsets = [
            file,
            rank,
            i8::min(rank, file),
            i8::min(Position::MAX_RANK as i8 - rank, file),
        ];
        let our = 1 << color as u8;

        let mut sequences = 0;
        for ((dy, dx), offset) in Self::DIRS.into_iter().zip(offsets) {
            let mut pos = position.offset(-dy * offset, -dx * offset);
            let mut run = 0;
            let mut in_seq = false;

            while let Ok(p) = pos {
                let bits = self.get(p, our | Self::SEQ_MASK);
                if bits == our {
                    run += 1;
                    in_seq = false;
                } else if bits == our | Self::SEQ_MASK {
                    if in_seq {
                        run = 1;
                    } else {
                        run += 1;
                    }
                    in_seq = true;
                } else {
                    run = 0;
                    in_seq = false;
                }

                if run == 5 {
                    sequences += 1;
                    run = 1;
                }

                pos = p.offset(dy, dx);
            }
        }
        sequences
    }
}

impl BoardMask {
    pub fn none() -> Self {
        Self {
            mask: [false; BOARD_SIZE],
        }
    }

    pub fn all() -> Self {
        Self {
            mask: [true; BOARD_SIZE],
        }
    }

    pub fn get(&self, pos: Position) -> bool {
        self.mask[pos.id()]
    }

    pub fn set(&mut self, pos: Position, value: bool) {
        self.mask[pos.id()] = value;
    }
}

impl fmt::Display for BoardMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in 0..=Position::MAX_RANK {
            for file in 0..=Position::MAX_FILE {
                let position = Position::new(rank, file).unwrap();
                write!(f, "{}", self.mask[position.id()] as u8)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Color::Red => "Red",
            Color::Blue => "Blue",
        };
        f.write_str(s)
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
            PositionError::InvalidRank(rank) => write!(f, "Invalid rank {rank}"),
            PositionError::InvalidFile(file) => write!(f, "Invalid file {file}"),
        }
    }
}
