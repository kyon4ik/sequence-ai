use core::fmt;

use arrayvec::ArrayVec;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_xoshiro::Xoroshiro128PlusPlus;

#[derive(Clone, Debug)]
pub struct Game {
    player_board_masks: [BoardMask; 2],
    player_hands: [PlayerHand; 2],
    // All squars that can't be removed (complete sequences and corners)
    protected_mask: BoardMask,
    draw_pile: Vec<Card>,
    discard_pile: Vec<Card>,
    current_player: PlayerOrder,
    status: GameStatus,
    // Did current player already exchanged dead card on this move?
    exchanged_dead_card: bool,
    rand_gen: Xoroshiro128PlusPlus,
    turn_count: usize,
    consecutive_pass_count: usize,
    player_sequences: [Vec<ArrayVec<Position, 5>>; 2],
}

// 4 two-eyed jacks and 3 other cards on empty board
pub const MAX_LEGAL_ACTIONS: usize = 4 * 96 + 3 * 2;
pub type LegalActions = ArrayVec<Action, MAX_LEGAL_ACTIONS>;

impl Game {
    pub const PLAYER_HAND_SIZE: usize = 7;

    pub const MAX_TURN_COUNT: usize = 300;

    pub const MAX_CONSECUTIVE_PASS_COUNT: usize = 6;

    pub fn new() -> Self {
        let seed = rand::random();
        Self::from_seed(seed)
    }

    pub fn from_seed(seed: u64) -> Self {
        let mut rand_gen = Xoroshiro128PlusPlus::seed_from_u64(seed);

        let mut corners_mask = BoardMask::none();
        corners_mask.set(Position::new(0, 0).unwrap(), true);
        corners_mask.set(Position::new(Position::MAX_RANK, 0).unwrap(), true);
        corners_mask.set(Position::new(0, Position::MAX_FILE).unwrap(), true);
        corners_mask.set(
            Position::new(Position::MAX_RANK, Position::MAX_FILE).unwrap(),
            true,
        );

        let mut draw_pile = Vec::with_capacity(2 * Card::COUNT);
        for suit in CardSuit::ALL {
            for rank in CardRank::ALL {
                draw_pile.push(Card::new(rank, suit));
                draw_pile.push(Card::new(rank, suit));
            }
        }
        draw_pile.shuffle(&mut rand_gen);

        let mut player_hands = core::array::repeat(PlayerHand::empty(Self::PLAYER_HAND_SIZE));
        for _ in 0..Self::PLAYER_HAND_SIZE {
            for player_hand in &mut player_hands {
                assert!(player_hand.add(draw_pile.pop().unwrap()));
            }
        }

        Self {
            player_board_masks: [corners_mask; 2],
            player_hands,
            protected_mask: corners_mask,
            draw_pile,
            discard_pile: vec![],
            current_player: PlayerOrder::First,
            status: GameStatus::InProgress,
            exchanged_dead_card: false,
            rand_gen,
            turn_count: 0,
            consecutive_pass_count: 0,
            player_sequences: core::array::repeat(vec![]),
        }
    }

    pub fn get_status(&self) -> GameStatus {
        self.status
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.status, GameStatus::Draw | GameStatus::Won(_))
    }

    pub fn get_turn_count(&self) -> usize {
        self.turn_count
    }

    pub fn get_current_player(&self) -> PlayerOrder {
        self.current_player
    }

    pub fn apply_action(&mut self, action: Action) -> Result<(), GameError> {
        match action {
            Action::Play {
                card_index,
                position,
            } => self.play_card(card_index, position),
            Action::Exchange { card_index } => self.exchange_dead_card(card_index),
            Action::Pass => self.pass_turn(),
        }
    }

    // This doesn't include Action::Pass, player should apply it when no legal actions are available
    pub fn legal_actions(&self) -> LegalActions {
        if self.is_finished() {
            return ArrayVec::new();
        }

        let mut actions = ArrayVec::new();

        let current_hand = &self.player_hands[self.current_player as usize];

        for (card_index, card) in current_hand.get_cards().iter().enumerate() {
            if card.is_one_eyed_jack() {
                for rank in 0..=Position::MAX_RANK {
                    for file in 0..=Position::MAX_FILE {
                        let position = unsafe { Position::new_unchecked(rank, file) };
                        if self.can_remove_from_spot(position) {
                            actions.push(Action::Play {
                                card_index,
                                position,
                            });
                        }
                    }
                }
            } else if card.is_two_eyed_jack() {
                for rank in 0..=Position::MAX_RANK {
                    for file in 0..=Position::MAX_FILE {
                        let position = unsafe { Position::new_unchecked(rank, file) };
                        if !self.is_spot_occupied(position) {
                            actions.push(Action::Play {
                                card_index,
                                position,
                            });
                        }
                    }
                }
            } else {
                let mut is_card_dead = true;

                for position in POSITIONS_BY_CARD[card.as_index()] {
                    if !self.is_spot_occupied(position) {
                        actions.push(Action::Play {
                            card_index,
                            position,
                        });

                        is_card_dead = false;
                    }
                }

                if is_card_dead && !self.exchanged_dead_card {
                    actions.push(Action::Exchange { card_index });
                }
            }
        }

        actions
    }

    pub fn is_spot_occupied(&self, position: Position) -> bool {
        self.player_board_masks[0].get(position) || self.player_board_masks[1].get(position)
    }

    pub fn is_spot_protected(&self, position: Position) -> bool {
        self.protected_mask.get(position)
    }

    pub fn can_remove_from_spot(&self, position: Position) -> bool {
        self.player_board_masks[self.current_player.next() as usize].get(position)
    }

    pub fn play_card(&mut self, card_index: usize, position: Position) -> Result<(), GameError> {
        let card = self
            .current_player_hand()
            .get_cards()
            .get(card_index)
            .ok_or(GameError::InvalidCardIndex(card_index))?;

        if card.is_one_eyed_jack() {
            self.remove_chip(position)?;
        } else {
            self.place_chip(position)?;
        }

        self.consecutive_pass_count = 0;
        self.discard_card(card_index)
            .expect("BUG: All checks passed before");
        self.advance_turn();
        Ok(())
    }

    pub fn exchange_dead_card(&mut self, card_index: usize) -> Result<(), GameError> {
        let card = self
            .current_player_hand()
            .get_cards()
            .get(card_index)
            .ok_or(GameError::InvalidCardIndex(card_index))?;

        if !self.is_card_dead(*card) || self.exchanged_dead_card {
            return Err(GameError::ExchangeOfAliveCard(card_index));
        }

        self.discard_card(card_index)
            .expect("BUG: All checks passed before");
        self.exchanged_dead_card = true;
        Ok(())
    }

    pub fn pass_turn(&mut self) -> Result<(), GameError> {
        if !self.legal_actions().is_empty() {
            return Err(GameError::UnexpectedPass);
        }

        self.consecutive_pass_count += 1;
        self.advance_turn();
        Ok(())
    }

    pub fn is_card_dead(&self, card: Card) -> bool {
        for position in POSITIONS_BY_CARD[card.as_index()] {
            if !self.is_spot_occupied(position) {
                return false;
            }
        }

        true
    }

    pub fn current_player_hand(&self) -> &PlayerHand {
        &self.player_hands[self.current_player as usize]
    }

    pub fn current_player_mask(&self) -> &BoardMask {
        &self.player_board_masks[self.current_player as usize]
    }

    fn current_player_hand_mut(&mut self) -> &mut PlayerHand {
        &mut self.player_hands[self.current_player as usize]
    }

    fn discard_card(&mut self, card_index: usize) -> Result<(), GameError> {
        let card = self
            .current_player_hand_mut()
            .remove(card_index)
            .ok_or(GameError::InvalidCardIndex(card_index))?;
        self.discard_pile.push(card);

        if let Some(next_card) = self.draw_card() {
            assert!(self.current_player_hand_mut().add(next_card));
        }

        Ok(())
    }

    fn place_chip(&mut self, position: Position) -> Result<(), GameError> {
        if self.is_spot_occupied(position) {
            return Err(GameError::InvalidPlayPosition(position));
        }

        let player_mask = &mut self.player_board_masks[self.current_player as usize];
        player_mask.set(position, true);

        // Calculate sequences
        const DIRS: [(i8, i8); 4] = [(0, 1), (1, 0), (1, 1), (-1, 1)];
        let (rank, file) = (position.get_rank() as i8, position.get_file() as i8);
        let offsets = [
            file,
            rank,
            i8::min(rank, file),
            i8::min(Position::MAX_RANK as i8 - rank, file),
        ];

        for ((dy, dx), offset) in DIRS.into_iter().zip(offsets) {
            let mut current_pos = position.offset(-dy * offset, -dx * offset);
            let mut sequence = ArrayVec::new();

            while let Ok(pos) = current_pos {
                if self.current_player_mask().get(pos) {
                    sequence.push(pos);
                } else {
                    sequence.clear();
                }

                if sequence.is_full() {
                    self.claim_sequence(sequence.clone());
                    sequence.clear();
                    sequence.push(pos);
                }

                current_pos = pos.offset(dy, dx);
            }
        }

        Ok(())
    }

    pub fn claim_sequence(&mut self, sequence: ArrayVec<Position, 5>) {
        for position in sequence.iter() {
            self.protected_mask.set(*position, true);
        }

        self.player_sequences[self.current_player as usize].push(sequence);
    }

    fn remove_chip(&mut self, position: Position) -> Result<(), GameError> {
        if !self.can_remove_from_spot(position) {
            return Err(GameError::InvalidPlayPosition(position));
        }

        let opponent_mask = &mut self.player_board_masks[self.current_player.next() as usize];
        opponent_mask.set(position, false);
        Ok(())
    }

    fn draw_card(&mut self) -> Option<Card> {
        if self.draw_pile.is_empty() {
            self.draw_pile.append(&mut self.discard_pile);
            self.draw_pile.shuffle(&mut self.rand_gen);
        }

        self.draw_pile.pop()
    }

    fn update_game_state(&mut self) {
        if (self.player_hands[0].is_empty() && self.player_hands[1].is_empty())
            || self.turn_count >= Self::MAX_TURN_COUNT
            || self.consecutive_pass_count >= Self::MAX_CONSECUTIVE_PASS_COUNT
        {
            self.status = GameStatus::Draw;
        } else if self.player_sequences[self.current_player as usize].len() >= 2 {
            self.status = GameStatus::Won(self.current_player);
        }
    }

    fn advance_turn(&mut self) {
        self.turn_count += 1;
        self.update_game_state();
        self.current_player = self.current_player.next();
        self.exchanged_dead_card = false;
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum GameError {
    InvalidCardIndex(usize),
    InvalidPlayPosition(Position),
    ExchangeOfAliveCard(usize),
    UnexpectedPass,
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

// Accept compact card tokens (Ac, 2d, 10h, Ks) and _ for free spaces.
// Decode at compile time so invalid values or suits cannot enter the layout.
macro_rules! board_layout {
    ($($card:tt),* $(,)?) => {{
        [$(Card::from_str(stringify!($card))),*]
    }};
}

// Classic 10 x 10 board, stored row by row from the top-left corner.
// Layout reference: https://www.sequencerivals.com/how-to-play#board-layout
#[rustfmt::skip]
pub const CARD_BY_POSITION: [Option<Card>; BoardMask::SIZE] = board_layout![
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

pub const POSITIONS_BY_CARD: [[Position; 2]; Card::COUNT] = build_positions_by_card();

const fn build_positions_by_card() -> [[Position; 2]; Card::COUNT] {
    let mut positions_by_card = [[Position { rank: 0, file: 0 }; 2]; Card::COUNT];

    // FIXME: for loop on u8 ranges are not allowed yet in const
    let mut rank = 0;
    while rank <= Position::MAX_RANK {
        let mut file = 0;
        while file <= Position::MAX_FILE {
            let position = unsafe { Position::new_unchecked(rank, file) };

            if let Some(card) = CARD_BY_POSITION[position.as_index()] {
                let first_position = positions_by_card[card.as_index()][0];
                if first_position.rank == 0 && first_position.file == 0 {
                    positions_by_card[card.as_index()][0] = position;
                } else {
                    positions_by_card[card.as_index()][1] = position;
                }
            }

            file += 1;
        }

        rank += 1;
    }

    positions_by_card
}

#[derive(Clone, Copy, Debug)]
pub enum GameStatus {
    InProgress,
    Won(PlayerOrder),
    Draw,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BoardMask {
    mask: [bool; Self::SIZE],
}

impl fmt::Display for BoardMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in 0..=Position::MAX_RANK {
            for file in 0..=Position::MAX_FILE {
                let position = Position::new(rank, file).unwrap();
                write!(f, "{}", self.mask[position.as_index()] as u8)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl BoardMask {
    const SIZE: usize = ((Position::MAX_RANK + 1) * (Position::MAX_FILE + 1)) as usize;

    pub fn none() -> Self {
        Self {
            mask: [false; Self::SIZE],
        }
    }

    pub fn all() -> Self {
        Self {
            mask: [true; Self::SIZE],
        }
    }

    #[must_use]
    pub fn get(&self, pos: Position) -> bool {
        self.mask[pos.as_index()]
    }

    pub fn set(&mut self, pos: Position, value: bool) {
        self.mask[pos.as_index()] = value;
    }

    pub fn toggle(&mut self, pos: Position) -> bool {
        let tmp = self.mask[pos.as_index()];
        self.mask[pos.as_index()] = !tmp;
        tmp
    }
}

#[derive(Debug)]
pub struct PlayerHand {
    cards: Vec<Card>,
}

// NOTE: Vec doesn't clone capacity :(
impl Clone for PlayerHand {
    fn clone(&self) -> Self {
        let mut cards = Vec::with_capacity(self.cards.capacity());
        cards.extend_from_slice(&self.cards);
        Self { cards }
    }
}

impl PlayerHand {
    pub fn empty(max_size: usize) -> Self {
        let cards = Vec::with_capacity(max_size);
        Self { cards }
    }

    #[must_use]
    pub fn add(&mut self, card: Card) -> bool {
        if self.cards.len() + 1 > self.cards.capacity() {
            return false;
        }

        self.cards.push(card);
        true
    }

    pub fn remove(&mut self, index: usize) -> Option<Card> {
        if index >= self.cards.len() {
            return None;
        }

        Some(self.cards.remove(index))
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.cards.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn get_cards(&self) -> &[Card] {
        &self.cards
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Position {
    rank: u8,
    file: u8,
}

#[derive(Debug)]
pub enum PositionError {
    InvalidRank(u8),
    InvalidFile(u8),
}

impl Position {
    pub const MAX_RANK: u8 = 9;
    pub const MAX_FILE: u8 = 9;

    pub const fn new(rank: u8, file: u8) -> Result<Self, PositionError> {
        if rank > Self::MAX_RANK {
            return Err(PositionError::InvalidRank(rank));
        }

        if file > Self::MAX_FILE {
            return Err(PositionError::InvalidFile(file));
        }

        Ok(unsafe { Self::new_unchecked(rank, file) })
    }

    #[must_use]
    pub fn get_rank(&self) -> u8 {
        self.rank
    }

    #[must_use]
    pub fn get_file(&self) -> u8 {
        self.file
    }

    const fn as_index(&self) -> usize {
        (self.rank * (Self::MAX_FILE + 1) + self.file) as usize
    }

    const unsafe fn new_unchecked(rank: u8, file: u8) -> Self {
        Self { rank, file }
    }

    fn offset(&self, dy: i8, dx: i8) -> Result<Self, PositionError> {
        let new_rank = self.rank.wrapping_add_signed(dy);
        let new_file = self.file.wrapping_add_signed(dx);
        Self::new(new_rank, new_file)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CardSuit {
    Clubs = 0,
    Spades = 1,
    Hearts = 2,
    Diamonds = 3,
}

impl CardSuit {
    pub const COUNT: usize = 4;

    pub const ALL: [Self; Self::COUNT] = [Self::Clubs, Self::Spades, Self::Hearts, Self::Diamonds];
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CardRank {
    Ace = 0,
    Two = 1,
    Three = 2,
    Four = 3,
    Five = 4,
    Six = 5,
    Seven = 6,
    Eight = 7,
    Nine = 8,
    Ten = 9,
    Jack = 10,
    Queen = 11,
    King = 12,
}

impl CardRank {
    pub const COUNT: usize = 13;

    pub const ALL: [Self; Self::COUNT] = [
        Self::Ace,
        Self::Two,
        Self::Three,
        Self::Four,
        Self::Five,
        Self::Six,
        Self::Seven,
        Self::Eight,
        Self::Nine,
        Self::Ten,
        Self::Jack,
        Self::Queen,
        Self::King,
    ];
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Card {
    suit: CardSuit,
    rank: CardRank,
}

impl Card {
    pub const COUNT: usize = CardSuit::COUNT * CardRank::COUNT;

    pub const fn new(rank: CardRank, suit: CardSuit) -> Self {
        Self { suit, rank }
    }

    const fn from_str(token: &str) -> Option<Card> {
        let (rank, suit) = match token.as_bytes() {
            [b'_'] => return None,
            [rank, suit] => {
                let rank = match rank {
                    b'A' => CardRank::Ace,
                    b'2' => CardRank::Two,
                    b'3' => CardRank::Three,
                    b'4' => CardRank::Four,
                    b'5' => CardRank::Five,
                    b'6' => CardRank::Six,
                    b'7' => CardRank::Seven,
                    b'8' => CardRank::Eight,
                    b'9' => CardRank::Nine,
                    b'Q' => CardRank::Queen,
                    b'K' => CardRank::King,
                    _ => panic!("invalid board card value"),
                };
                (rank, *suit)
            }
            [b'1', b'0', suit] => (CardRank::Ten, *suit),
            _ => panic!("invalid board card token"),
        };

        let suit = match suit {
            b'c' => CardSuit::Clubs,
            b'd' => CardSuit::Diamonds,
            b'h' => CardSuit::Hearts,
            b's' => CardSuit::Spades,
            _ => panic!("invalid board card suit"),
        };

        Some(Card::new(rank, suit))
    }

    #[must_use]
    pub fn get_rank(&self) -> CardRank {
        self.rank
    }

    #[must_use]
    pub fn get_suit(&self) -> CardSuit {
        self.suit
    }

    #[must_use]
    pub const fn as_index(&self) -> usize {
        self.suit as usize * CardRank::COUNT + self.rank as usize
    }

    pub fn is_one_eyed_jack(&self) -> bool {
        self.rank == CardRank::Jack && matches!(self.suit, CardSuit::Hearts | CardSuit::Spades)
    }

    pub fn is_two_eyed_jack(&self) -> bool {
        self.rank == CardRank::Jack && matches!(self.suit, CardSuit::Clubs | CardSuit::Diamonds)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlayerOrder {
    First = 0,
    Second = 1,
}

impl PlayerOrder {
    #[must_use]
    pub fn next(&self) -> Self {
        match self {
            Self::First => Self::Second,
            Self::Second => Self::First,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_of_card_option() {
        assert_eq!(core::mem::size_of::<Option<Card>>(), 2);
    }
}
