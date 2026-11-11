use arrayvec::ArrayVec;
use sequence_core::{Action, ActionError, Board, Card, Color, SimpleObservation};

/// A Simple move's deterministic effects, before replacement drawing or turn advancement.
/// The remaining hand belongs to `actor`; this is not a playable game state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Afterstate {
    pub board: Board,
    pub actor: Color,
    pub consumed_card: Option<Card>,
    pub ends_turn: bool,
    pub requires_draw: bool,
    hand: ArrayVec<Card, 7>,
}

impl Afterstate {
    pub fn new(obs: SimpleObservation<'_>, action: Action) -> Result<Self, ActionError> {
        obs.validate(action)?;
        let mut board = obs.board.clone();
        let mut hand = ArrayVec::new();
        hand.extend(obs.hand.iter().copied());
        let consumed_card = match action {
            Action::Play {
                card_index,
                position,
            } => {
                board
                    .place_chip(position, obs.current_player)
                    .expect("validated empty square");
                Some(hand.remove(card_index))
            }
            Action::Pass => None,
            Action::Exchange { .. } => unreachable!("Simple does not permit exchanges"),
        };
        Ok(Self {
            board,
            hand,
            actor: obs.current_player,
            consumed_card,
            ends_turn: true,
            requires_draw: consumed_card.is_some() && obs.draw_pile_size > 0,
        })
    }

    pub fn hand(&self) -> &[Card] {
        &self.hand
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sequence_core::{Game, Simple};

    #[test]
    fn afterstates_match_moves_without_drawing_or_mutating_the_game() {
        for seed in 0..8 {
            let mut game = Simple::from_seed(seed);
            let mut actions = vec![];
            while !game.is_game_over() {
                game.legal_actions(&mut actions);
                let before = format!("{game:?}");
                for &action in &actions {
                    let after = Afterstate::new(game.observation(), action).unwrap();
                    assert_eq!(format!("{game:?}"), before);
                    let mut actual = game.clone();
                    actual.apply(action).unwrap();
                    assert_eq!(after.board, *actual.board());
                    let mut remaining: Vec<_> = game.observation().hand.to_vec();
                    if let Action::Play { card_index, .. } = action {
                        assert_eq!(after.consumed_card, Some(remaining.remove(card_index)));
                    }
                    assert_eq!(after.hand(), remaining);
                    assert_eq!(after.actor, game.current_player());
                    assert_eq!(actual.turns(), game.turns() + 1);
                }
                game.apply(actions[(seed as usize + game.turns() as usize) % actions.len()])
                    .unwrap();
            }
        }
    }

    #[test]
    fn invalid_actions_return_errors() {
        let game = Simple::from_seed(7);
        assert_eq!(
            Afterstate::new(game.observation(), Action::Pass),
            Err(ActionError::UnexpectedPass)
        );
        assert_eq!(
            Afterstate::new(game.observation(), Action::Exchange { card_index: 0 }),
            Err(ActionError::ExchangeUnavailable)
        );
    }
}
