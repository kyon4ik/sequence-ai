use crate::{Board, Player, Position};

pub const SEQUENCE_LENGTH: usize = 5;
const DIRECTIONS: [(isize, isize); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Sequence([Position; SEQUENCE_LENGTH]);

impl Sequence {
    /// Enumerates all five-space windows on the board.
    pub fn all() -> impl Iterator<Item = Self> {
        Position::all().flat_map(|start| {
            DIRECTIONS
                .into_iter()
                .filter_map(move |(row_step, column_step)| {
                    Self::through(start, 0, row_step, column_step)
                })
        })
    }

    pub fn spaces(&self) -> &[Position; SEQUENCE_LENGTH] {
        &self.0
    }

    fn through(
        placed: Position,
        offset: isize,
        row_step: isize,
        column_step: isize,
    ) -> Option<Self> {
        let start = placed.offset(-offset * row_step, -offset * column_step)?;
        let mut spaces = [start; SEQUENCE_LENGTH];
        for index in 1..SEQUENCE_LENGTH {
            spaces[index] = spaces[index - 1].offset(row_step, column_step)?;
        }
        Some(Self(spaces))
    }

    /// Checks whether both windows can be claimed by the same player under the overlap rule.
    pub fn compatible_with(&self, other: &Self) -> bool {
        self.0
            .iter()
            .filter(|position| !position.is_corner() && other.0.contains(position))
            .nth(1)
            .is_none()
    }
}

pub(crate) fn new_claims(
    board: &Board,
    player: Player,
    placed: Position,
    claimed: &[Sequence],
) -> Vec<Sequence> {
    let mut candidates = Vec::new();
    for (row_step, column_step) in DIRECTIONS {
        for offset in 0..SEQUENCE_LENGTH as isize {
            if let Some(sequence) = Sequence::through(placed, offset, row_step, column_step)
                && sequence
                    .0
                    .iter()
                    .all(|&position| position.is_corner() || board.chip(position) == Some(player))
                && claimed.iter().all(|other| sequence.compatible_with(other))
            {
                candidates.push(sequence);
            }
        }
    }

    // Every direction lists its spaces in board order. Visiting sorted subsets in
    // order means the first group found at the maximum size wins the tie.
    candidates.sort_unstable();
    let mut best = Vec::new();
    find_largest_group(&candidates, &mut Vec::new(), &mut best);
    best
}

fn find_largest_group(
    candidates: &[Sequence],
    group: &mut Vec<Sequence>,
    best: &mut Vec<Sequence>,
) {
    if group.len() > best.len() {
        best.clone_from(group);
    }

    for (index, candidate) in candidates.iter().enumerate() {
        if group.len() + candidates.len() - index <= best.len() {
            break;
        }
        if group.iter().all(|other| candidate.compatible_with(other)) {
            group.push(*candidate);
            find_largest_group(&candidates[index + 1..], group, best);
            group.pop();
        }
    }
}
