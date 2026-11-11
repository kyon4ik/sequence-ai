use super::*;
use crate::{Cell, get_card, positions_for_card};

/// Finite 96-card game without jacks, exchanges, or discard recycling.
#[derive(Clone, Debug)]
pub struct Simple {
    pub(super) state: State,
}

/// Information available to the current player of a Simple game.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct SimpleObservation<'a> {
    pub board: &'a Board,
    pub turns: u16,
    pub current_player: Color,
    pub hand: &'a [Card],
    pub opponent_hand_size: usize,
    pub draw_pile_size: usize,
    pub discard_pile: &'a [Card],
    pub outcome: GameOutcome,
}

impl Default for Simple {
    fn default() -> Self {
        Self::new()
    }
}

impl Game for Simple {
    type Observation<'a> = SimpleObservation<'a>;

    fn from_seed(seed: u64) -> Self {
        let (state, _) = State::deal(
            seed,
            CardRank::VARIANTS
                .iter()
                .copied()
                .filter(|&rank| rank != CardRank::Jack),
        );
        Self { state }
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
    fn observation(&self) -> SimpleObservation<'_> {
        let state = &self.state;
        SimpleObservation {
            board: &state.board,
            turns: state.turns,
            current_player: state.current_player,
            hand: &state.hands[state.current_player as usize],
            opponent_hand_size: state.hands[state.current_player.next() as usize].len(),
            draw_pile_size: state.draw_pile.len(),
            discard_pile: &state.discard_pile,
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
            self.state.board.place(position, self.state.current_player);
            self.state.discard(card_index);
            self.state.draw();
        }
        self.state.finish_turn(false);
        Ok(())
    }
}

impl SimpleObservation<'_> {
    fn can_play(&self, card: Card, position: Position) -> bool {
        get_card(position) == Some(card) && self.board.cell(position) == Cell::Empty
    }
    fn has_play(&self) -> bool {
        self.hand
            .iter()
            .any(|&card| positions_for_card(card).any(|p| self.can_play(card, p)))
    }
    pub fn legal_actions(&self, output: &mut Vec<Action>) {
        output.clear();
        if self.outcome != GameOutcome::InProgress {
            return;
        }
        for (card_index, &card) in self.hand.iter().enumerate() {
            for position in positions_for_card(card) {
                if self.can_play(card, position) {
                    output.push(Action::Play {
                        card_index,
                        position,
                    });
                }
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
                let card = self
                    .hand
                    .get(card_index)
                    .ok_or(ActionError::InvalidCardIndex(card_index))?;
                if !self.can_play(*card, position) {
                    return Err(ActionError::InvalidPlayPosition(position));
                }
            }
            Action::Exchange { .. } => return Err(ActionError::ExchangeUnavailable),
            Action::Pass if self.has_play() => return Err(ActionError::UnexpectedPass),
            Action::Pass => {}
        }
        Ok(())
    }
}
