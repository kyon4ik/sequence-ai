use super::*;
use crate::{Cell, get_card, positions_for_card};

/// Full 104-card game with jacks, dead-card exchanges, and discard recycling.
#[derive(Clone, Debug)]
pub struct Classic {
    pub(super) state: State,
    pub(super) rng: Xoroshiro128PlusPlus,
    pub(super) exchanged_dead_card: bool,
    pub(super) consecutive_passes: u8,
}

/// Public information and the current player's hand in a Classic game.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ClassicObservation<'a> {
    pub board: &'a Board,
    pub turns: u16,
    pub current_player: Color,
    pub hand: &'a [Card],
    pub opponent_hand_size: usize,
    pub draw_pile_size: usize,
    pub discard_pile: &'a [Card],
    pub exchanged_dead_card: bool,
    pub consecutive_passes: u8,
    pub outcome: GameOutcome,
}

impl Classic {
    pub const MAX_TURNS: u16 = 300;
    pub const MAX_CONSECUTIVE_PASSES: u8 = 6;
}
impl Default for Classic {
    fn default() -> Self {
        Self::new()
    }
}

impl Game for Classic {
    type Observation<'a> = ClassicObservation<'a>;

    fn from_seed(seed: u64) -> Self {
        let (state, rng) = State::deal(seed, CardRank::VARIANTS.iter().copied());
        Self {
            state,
            rng,
            exchanged_dead_card: false,
            consecutive_passes: 0,
        }
    }
    fn board(&self) -> &Board {
        &self.state.board
    }
    fn turns(&self) -> u16 {
        self.state.turns
    }
    fn current_player(&self) -> Color {
        self.state.current_player
    }
    fn outcome(&self) -> GameOutcome {
        self.state.outcome
    }
    fn observation(&self) -> ClassicObservation<'_> {
        let state = &self.state;
        ClassicObservation {
            board: &state.board,
            turns: state.turns,
            current_player: state.current_player,
            hand: &state.hands[state.current_player as usize],
            opponent_hand_size: state.hands[state.current_player.next() as usize].len(),
            draw_pile_size: state.draw_pile.len(),
            discard_pile: &state.discard_pile,
            exchanged_dead_card: self.exchanged_dead_card,
            consecutive_passes: self.consecutive_passes,
            outcome: state.outcome,
        }
    }
    fn legal_actions(&self, output: &mut Vec<Action>) {
        self.observation().legal_actions(output);
    }
    fn validate(&self, action: Action) -> Result<(), ActionError> {
        self.observation().validate(action)
    }
    fn apply(&mut self, action: Action) -> Result<(), ActionError> {
        self.validate(action)?;
        if let Action::Play {
            card_index,
            position,
        } = action
        {
            let card = self.state.hands[self.state.current_player as usize][card_index];
            if card.is_one_eyed_jack() {
                self.state.board.remove(position);
            } else {
                self.state.board.place(position, self.state.current_player);
            }
        }
        match action {
            Action::Play { card_index, .. } | Action::Exchange { card_index } => {
                self.state.discard(card_index);
                if self.state.draw_pile.is_empty() {
                    std::mem::swap(&mut self.state.draw_pile, &mut self.state.discard_pile);
                    self.state.draw_pile.shuffle(&mut self.rng);
                }
                self.state.draw();
            }
            Action::Pass => {}
        }
        if matches!(action, Action::Exchange { .. }) {
            self.exchanged_dead_card = true;
        } else {
            if action == Action::Pass {
                self.consecutive_passes += 1;
            } else {
                self.consecutive_passes = 0;
            }
            let draw_limit = self.state.turns + 1 >= Self::MAX_TURNS
                || self.consecutive_passes >= Self::MAX_CONSECUTIVE_PASSES;
            self.state.finish_turn(draw_limit);
            self.exchanged_dead_card = false;
        }
        Ok(())
    }
}

impl ClassicObservation<'_> {
    fn card(&self, index: usize) -> Result<Card, ActionError> {
        self.hand
            .get(index)
            .copied()
            .ok_or(ActionError::InvalidCardIndex(index))
    }
    fn can_play(&self, card: Card, position: Position) -> bool {
        if card.is_one_eyed_jack() {
            self.board.cell(position) == Cell::Occupied(self.current_player.next())
                && !self.board.is_protected(position)
        } else if card.is_two_eyed_jack() {
            self.board.cell(position) == Cell::Empty
        } else {
            get_card(position) == Some(card) && self.board.cell(position) == Cell::Empty
        }
    }
    fn is_dead(&self, card: Card) -> bool {
        card.rank() != CardRank::Jack
            && positions_for_card(card).all(|p| self.board.cell(p) != Cell::Empty)
    }
    fn can_exchange(&self, card: Card) -> bool {
        !self.exchanged_dead_card && self.is_dead(card)
    }
    fn has_non_pass_action(&self) -> bool {
        self.hand.iter().copied().any(|card| {
            self.can_exchange(card)
                || if card.rank() == CardRank::Jack {
                    Position::all().any(|p| self.can_play(card, p))
                } else {
                    positions_for_card(card).any(|p| self.can_play(card, p))
                }
        })
    }
    pub fn legal_actions(&self, output: &mut Vec<Action>) {
        output.clear();
        if self.outcome != GameOutcome::InProgress {
            return;
        }
        for (card_index, &card) in self.hand.iter().enumerate() {
            let mut add = |position| {
                if self.can_play(card, position) {
                    output.push(Action::Play {
                        card_index,
                        position,
                    });
                }
            };
            if card.rank() == CardRank::Jack {
                Position::all().for_each(&mut add);
            } else {
                positions_for_card(card).for_each(&mut add);
            }
            if self.can_exchange(card) {
                output.push(Action::Exchange { card_index });
            }
        }
        if output.is_empty() {
            output.push(Action::Pass);
        }
    }
    pub fn validate(&self, action: Action) -> Result<(), ActionError> {
        if self.outcome != GameOutcome::InProgress {
            return Err(ActionError::GameOver);
        }
        match action {
            Action::Play {
                card_index,
                position,
            } => {
                if !self.can_play(self.card(card_index)?, position) {
                    return Err(ActionError::InvalidPlayPosition(position));
                }
            }
            Action::Exchange { card_index } => {
                let card = self.card(card_index)?;
                if self.exchanged_dead_card {
                    return Err(ActionError::ExchangeUnavailable);
                }
                if !self.is_dead(card) {
                    return Err(ActionError::CardNotDead);
                }
            }
            Action::Pass if self.has_non_pass_action() => return Err(ActionError::UnexpectedPass),
            Action::Pass => {}
        }
        Ok(())
    }
}
