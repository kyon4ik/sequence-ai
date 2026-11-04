#![deny(unsafe_code)]

mod board;
mod card;
mod game;

pub use board::{Board, BoardMask, Color, Position};
pub use card::{Card, CardRank, CardSuit};
pub use game::{Action, GameOutcome, MAX_LEGAL_ACTIONS, NoJacksGame, Observation};

// impl Game {
// pub const MAX_TURN_COUNT: usize = 300;

// pub const MAX_CONSECUTIVE_PASS_COUNT: usize = 6;

// pub fn new() -> Self {
//     let seed = rand::random();
//     Self::from_seed(seed)
// }

// pub fn from_seed(seed: u64) -> Self {
//     let mut rand_gen = Xoroshiro128PlusPlus::seed_from_u64(seed);

//     let mut corners_mask = BoardMask::none();
//     corners_mask.set(Position::new(0, 0).unwrap(), true);
//     corners_mask.set(Position::new(Position::MAX_RANK, 0).unwrap(), true);
//     corners_mask.set(Position::new(0, Position::MAX_FILE).unwrap(), true);
//     corners_mask.set(
//         Position::new(Position::MAX_RANK, Position::MAX_FILE).unwrap(),
//         true,
//     );

//     let mut draw_pile = Vec::with_capacity(2 * Card::COUNT);
//     for &suit in CardSuit::VARIANTS {
//         for &rank in CardRank::VARIANTS {
//             draw_pile.push(Card::new(rank, suit));
//             draw_pile.push(Card::new(rank, suit));
//         }
//     }
//     draw_pile.shuffle(&mut rand_gen);

//     let mut player_hands = core::array::repeat(ArrayVec::new());
//     for _ in 0..7 {
//         for player_hand in &mut player_hands {
//             player_hand.push(draw_pile.pop().unwrap());
//         }
//     }

//     Self {
//         player_board_masks: [corners_mask; 2],
//         player_hands,
//         protected_mask: corners_mask,
//         draw_pile,
//         discard_pile: vec![],
//         current_player: Color::Red,
//         status: GameStatus::InProgress,
//         exchanged_dead_card: false,
//         rand_gen,
//         turn_count: 0,
//         consecutive_pass_count: 0,
//         player_sequences: core::array::repeat(vec![]),
//     }
// }

// pub fn get_status(&self) -> GameStatus {
//     self.status
// }

// pub fn is_finished(&self) -> bool {
//     matches!(self.status, GameStatus::Draw | GameStatus::Won(_))
// }

// pub fn get_turn_count(&self) -> usize {
//     self.turn_count
// }

// pub fn get_current_player(&self) -> Color {
//     self.current_player
// }

// pub fn apply_action(&mut self, action: Action) -> Result<(), GameError> {
//     match action {
//         Action::Play {
//             card_index,
//             position,
//         } => self.play_card(card_index, position),
//         Action::Exchange { card_index } => self.exchange_dead_card(card_index),
//         Action::Pass => self.pass_turn(),
//     }
// }

// This doesn't include Action::Pass, player should apply it when no legal actions are available
// pub fn legal_actions(&self) -> LegalActions {
//     if self.is_finished() {
//         return ArrayVec::new();
//     }

//     let mut actions = ArrayVec::new();

//     let current_hand = &self.player_hands[self.current_player as usize];

//     for (card_index, card) in current_hand.iter().enumerate() {
//         if card.is_one_eyed_jack() {
//             for rank in 0..=Position::MAX_RANK {
//                 for file in 0..=Position::MAX_FILE {
//                     let position = unsafe { Position::new_unchecked(rank, file) };
//                     if self.can_remove_from_spot(position) {
//                         actions.push(Action::Play {
//                             card_index,
//                             position,
//                         });
//                     }
//                 }
//             }
//         } else if card.is_two_eyed_jack() {
//             for rank in 0..=Position::MAX_RANK {
//                 for file in 0..=Position::MAX_FILE {
//                     let position = unsafe { Position::new_unchecked(rank, file) };
//                     if !self.is_spot_occupied(position) {
//                         actions.push(Action::Play {
//                             card_index,
//                             position,
//                         });
//                     }
//                 }
//             }
//         } else {
//             let mut is_card_dead = true;

//             for position in POSITIONS_BY_CARD[card.id()] {
//                 if !self.is_spot_occupied(position) {
//                     actions.push(Action::Play {
//                         card_index,
//                         position,
//                     });

//                     is_card_dead = false;
//                 }
//             }

//             if is_card_dead && !self.exchanged_dead_card {
//                 actions.push(Action::Exchange { card_index });
//             }
//         }
//     }

//     actions
// }

// pub fn is_spot_occupied(&self, position: Position) -> bool {
//     self.player_board_masks[0].get(position) || self.player_board_masks[1].get(position)
// }

// pub fn is_spot_protected(&self, position: Position) -> bool {
//     self.protected_mask.get(position)
// }

// pub fn can_remove_from_spot(&self, position: Position) -> bool {
//     self.player_board_masks[self.current_player.next() as usize].get(position)
//         && !self.is_spot_protected(position)
// }

// pub fn play_card(&mut self, card_index: usize, position: Position) -> Result<(), GameError> {
//     let card = self
//         .current_player_hand()
//         .get(card_index)
//         .ok_or(GameError::InvalidCardIndex(card_index))?;

//     if card.is_one_eyed_jack() {
//         self.remove_chip(position)?;
//     } else {
//         if !card.is_two_eyed_jack() && board::get_card(position) != Some(*card) {
//             return Err(GameError::InvalidPlayPosition(position));
//         }
//         self.place_chip(position)?;
//     }

//     self.consecutive_pass_count = 0;
//     self.discard_card(card_index)
//         .expect("BUG: All checks passed before");
//     self.advance_turn();
//     Ok(())
// }

// pub fn exchange_dead_card(&mut self, card_index: usize) -> Result<(), GameError> {
//     let card = self
//         .current_player_hand()
//         .get(card_index)
//         .ok_or(GameError::InvalidCardIndex(card_index))?;

//     if !self.is_card_dead(*card) || self.exchanged_dead_card {
//         return Err(GameError::ExchangeOfAliveCard(card_index));
//     }

//     self.discard_card(card_index)
//         .expect("BUG: All checks passed before");
//     self.exchanged_dead_card = true;
//     Ok(())
// }

// pub fn pass_turn(&mut self) -> Result<(), GameError> {
//     if !self.legal_actions().is_empty() {
//         return Err(GameError::UnexpectedPass);
//     }

//     self.consecutive_pass_count += 1;
//     self.advance_turn();
//     Ok(())
// }

// pub fn is_card_dead(&self, card: Card) -> bool {
//     if card.rank() == CardRank::Jack {
//         return false;
//     }

//     for position in POSITIONS_BY_CARD[card.id()] {
//         if !self.is_spot_occupied(position) {
//             return false;
//         }
//     }

//     true
// }

// pub fn current_player_hand(&self) -> &ArrayVec<Card, 7> {
//     &self.player_hands[self.current_player as usize]
// }

// pub fn current_player_mask(&self) -> &BoardMask {
//     &self.player_board_masks[self.current_player as usize]
// }

// fn current_player_hand_mut(&mut self) -> &mut ArrayVec<Card, 7> {
//     &mut self.player_hands[self.current_player as usize]
// }

// fn discard_card(&mut self, card_index: usize) -> Result<(), GameError> {
//     let card = self
//         .current_player_hand_mut()
//         .pop_at(card_index)
//         .ok_or(GameError::InvalidCardIndex(card_index))?;
//     self.discard_pile.push(card);

//     if let Some(next_card) = self.draw_card() {
//         self.current_player_hand_mut().push(next_card);
//     }

//     Ok(())
// }

// fn place_chip(&mut self, position: Position) -> Result<(), GameError> {
//     if self.is_spot_occupied(position) {
//         return Err(GameError::InvalidPlayPosition(position));
//     }

//     let player_mask = &mut self.player_board_masks[self.current_player as usize];
//     player_mask.set(position, true);

//     // Calculate sequences
//     const DIRS: [(i8, i8); 4] = [(0, 1), (1, 0), (1, 1), (-1, 1)];
//     let (rank, file) = (position.rank() as i8, position.file() as i8);
//     let offsets = [
//         file,
//         rank,
//         i8::min(rank, file),
//         i8::min(Position::MAX_RANK as i8 - rank, file),
//     ];

//     for ((dy, dx), offset) in DIRS.into_iter().zip(offsets) {
//         let mut current_pos = position.offset(-dy * offset, -dx * offset);
//         let mut sequence = ArrayVec::new();

//         while let Ok(pos) = current_pos {
//             if self.current_player_mask().get(pos) {
//                 sequence.push(pos);
//             } else {
//                 sequence.clear();
//             }

//             if sequence.is_full() {
//                 // Slide one square even after rejection, so an overlapping
//                 // window cannot hide a later valid sequence.
//                 let compatible = self.player_sequences[self.current_player as usize]
//                     .iter()
//                     .all(|claimed| {
//                         sequence.iter().filter(|pos| claimed.contains(pos)).count() <= 1
//                     });
//                 if compatible {
//                     self.claim_sequence(sequence.clone());
//                 }
//                 sequence.remove(0);
//             }

//             current_pos = pos.offset(dy, dx);
//         }
//     }

//     Ok(())
// }

// fn claim_sequence(&mut self, sequence: ArrayVec<Position, 5>) {
//     for position in sequence.iter() {
//         self.protected_mask.set(*position, true);
//     }

//     self.player_sequences[self.current_player as usize].push(sequence);
// }

// fn remove_chip(&mut self, position: Position) -> Result<(), GameError> {
//     if !self.can_remove_from_spot(position) {
//         return Err(GameError::InvalidPlayPosition(position));
//     }

//     let opponent_mask = &mut self.player_board_masks[self.current_player.next() as usize];
//     opponent_mask.set(position, false);
//     Ok(())
// }

// fn draw_card(&mut self) -> Option<Card> {
//     if self.draw_pile.is_empty() {
//         self.draw_pile.append(&mut self.discard_pile);
//         self.draw_pile.shuffle(&mut self.rand_gen);
//     }

//     self.draw_pile.pop()
// }

// fn update_game_state(&mut self) {
//     if (self.player_hands[0].is_empty() && self.player_hands[1].is_empty())
//         || self.turn_count >= Self::MAX_TURN_COUNT
//         || self.consecutive_pass_count >= Self::MAX_CONSECUTIVE_PASS_COUNT
//     {
//         self.status = GameStatus::Draw;
//     } else if self.player_sequences[self.current_player as usize].len() >= 2 {
//         self.status = GameStatus::Won(self.current_player);
//     }
// }

// fn advance_turn(&mut self) {
//     self.turn_count += 1;
//     self.update_game_state();
//     self.current_player = self.current_player.next();
//     self.exchanged_dead_card = false;
// }
// }
