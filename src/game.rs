use core::array;
use std::fmt;

use arrayvec::ArrayVec;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_xoshiro::Xoroshiro128PlusPlus;
use strum::VariantArray;

use crate::board::PositionIter;
use crate::{Board, Card, CardRank, CardSuit, Color, Position, board};

// 4 two-eyed jacks and 3 other cards on empty board
pub const MAX_LEGAL_ACTIONS: usize = 4 * 96 + 3 * 2;

#[derive(Clone, Debug)]
pub struct NoJacksGame {
    board: Board,
    turns: u16,
    color: Color,
    sequences: [u8; 2],
    hands: [ArrayVec<Card, 7>; 2],
    draw_pile: Vec<Card>,
}

impl Default for NoJacksGame {
    fn default() -> Self {
        let seed = rand::random();
        Self::from_seed(seed)
    }
}

impl NoJacksGame {
    pub fn from_seed(seed: u64) -> Self {
        let mut rng = Xoroshiro128PlusPlus::seed_from_u64(seed);

        let mut board = Board::empty();
        for corner in Position::CORNERS {
            board.set(
                corner,
                Board::RED_MASK | Board::BLUE_MASK | Board::PROT_MASK,
            );
        }

        let mut draw_pile = Vec::with_capacity(2 * Card::COUNT);
        for &suit in CardSuit::VARIANTS {
            for &rank in CardRank::VARIANTS
                .iter()
                .filter(|&&rank| rank != CardRank::Jack)
            {
                draw_pile.push(Card::new(rank, suit));
                draw_pile.push(Card::new(rank, suit));
            }
        }
        draw_pile.shuffle(&mut rng);

        let mut hands = array::repeat(ArrayVec::new());
        for _ in 0..7 {
            for hand in &mut hands {
                hand.push(draw_pile.pop().unwrap());
            }
        }

        Self {
            board,
            turns: 0,
            color: Color::Red,
            sequences: [0; 2],
            hands,
            draw_pile,
        }
    }

    pub fn turns(&self) -> u16 {
        self.turns
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn us(&self) -> Color {
        self.color
    }

    pub fn they(&self) -> Color {
        self.color.next()
    }

    pub fn our_mask(&self) -> u8 {
        1 << self.us() as u8
    }

    pub fn their_mask(&self) -> u8 {
        1 << self.they() as u8
    }

    pub fn our_hand(&self) -> &[Card] {
        &self.hands[self.color as usize]
    }

    pub fn our_sequences(&self) -> u8 {
        self.sequences[self.color as usize]
    }

    pub fn is_draw(&self) -> bool {
        self.hands[0].is_empty() && self.hands[1].is_empty()
    }

    pub fn outcome(&self) -> GameOutcome {
        if self.sequences[0] >= 2 {
            GameOutcome::Won(Color::Red)
        } else if self.sequences[1] >= 2 {
            GameOutcome::Won(Color::Blue)
        } else if self.is_draw() {
            GameOutcome::Draw
        } else {
            GameOutcome::Unknown
        }
    }

    pub fn is_game_over(&self) -> bool {
        self.outcome() != GameOutcome::Unknown
    }

    // If no legal actions, then game is over
    pub fn legal_actions(&self, actions: &mut Vec<Action>) {
        if self.is_game_over() {
            return;
        }

        for (card_index, &card) in self.our_hand().iter().enumerate() {
            for position in PositionIter::new() {
                if self.can_play(position, card) {
                    actions.push(Action::Play {
                        card_index,
                        position,
                    });
                }
            }
        }
    }

    pub fn validate(&self, action: Action) -> bool {
        let mut legal_actions = Vec::with_capacity(MAX_LEGAL_ACTIONS);
        self.legal_actions(&mut legal_actions);
        legal_actions.contains(&action)
    }

    pub fn apply_unchecked(&mut self, action: Action) {
        if let Action::Play {
            card_index,
            position,
        } = action
        {
            self.play_card(card_index, position)
        }
    }

    #[must_use]
    pub fn apply(&mut self, action: Action) -> bool {
        if self.validate(action) {
            self.apply_unchecked(action);
            return true;
        }
        false
    }

    fn can_play(&self, position: Position, card: Card) -> bool {
        board::get_card(position) == Some(card)
            && self.board.get(position, Board::RED_MASK | Board::BLUE_MASK) == 0
    }

    fn play_card(&mut self, card_index: usize, position: Position) {
        self.place_chip(position);
        self.discard_card(card_index);
        self.advance_turn();
    }

    fn place_chip(&mut self, position: Position) {
        self.board.set(position, self.our_mask());
        let new_sequences = self.board.update_sequences(position, self.color);
        self.sequences[self.color as usize] += new_sequences;
    }

    fn discard_card(&mut self, card_index: usize) {
        self.hands[self.color as usize].remove(card_index);
        if let Some(card) = self.draw_pile.pop() {
            self.hands[self.color as usize].push(card);
        }
    }

    fn advance_turn(&mut self) {
        self.turns += 1;
        self.color = self.color.next();
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
pub enum GameOutcome {
    Unknown,
    Won(Color),
    Draw,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::Play {
                card_index,
                position,
            } => write!(f, "play card #{card_index} on {position}"),
            Action::Exchange { card_index } => write!(f, "exchange card #{card_index}"),
            Action::Pass => f.write_str("pass"),
        }
    }
}

impl fmt::Display for GameOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GameOutcome::Unknown => f.write_str("unknown"),
            GameOutcome::Won(color) => write!(f, "{color} won"),
            GameOutcome::Draw => f.write_str("draw"),
        }
    }
}
