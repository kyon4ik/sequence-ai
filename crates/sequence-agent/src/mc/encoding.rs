use sequence_core::{Action, ActionError, Card, CardRank, Position, SimpleObservation, get_card};

#[derive(Clone, Debug)]
pub struct EncodedAfterstate {
    pub board: [f32; 400],
    pub context: [f32; 50],
}

// Preserve the existing 48-card checkpoint encoding independently of card IDs.
fn card_index(card: Card) -> usize {
    let rank = card.rank() as usize;
    assert_ne!(card.rank(), CardRank::Jack, "no-jacks encoder");
    (rank - usize::from(rank > CardRank::Jack as usize)) * 4 + card.suit() as usize
}

impl EncodedAfterstate {
    /// Encodes a Simple afterstate, returning an error for an illegal action.
    pub fn new(obs: SimpleObservation<'_>, action: Action) -> Result<Self, ActionError> {
        let after = crate::Afterstate::new(obs, action)?;
        let board = &after.board;
        let mut encoded = Self {
            board: [0.; 400],
            context: [0.; 50],
        };
        for &card in after.hand() {
            encoded.context[card_index(card)] += 0.5;
        }
        encoded.context[48] = board.sequences(after.actor).len() as f32 / 2.;
        encoded.context[49] = board.sequences(after.actor.next()).len() as f32 / 2.;
        for rank in 0..10 {
            for file in 0..10 {
                let p = Position::new(rank, file).unwrap();
                let i = p.id();
                // Channel-first layout; free corners belong to both players.
                encoded.board[i] = f32::from(board.is_owned_by(p, after.actor));
                encoded.board[100 + i] = f32::from(board.is_owned_by(p, after.actor.next()));
                encoded.board[200 + i] = f32::from(board.is_in_sequence(p));
                if let Some(card) = get_card(p) {
                    encoded.board[300 + i] = encoded.context[card_index(card)];
                }
            }
        }
        Ok(encoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sequence_core::{Game, Simple};

    #[test]
    fn preview_matches_actual_board_but_excludes_draw() {
        let mut game = Simple::from_seed(42);
        let mut actions = Vec::new();
        game.legal_actions(&mut actions);
        let actor = game.current_player();
        let before = game.board().clone();
        let encoded = EncodedAfterstate::new(game.observation(), actions[0]).unwrap();
        assert_eq!(*game.board(), before);
        assert_eq!(encoded.context[..48].iter().sum::<f32>(), 3.);
        assert_eq!(encoded.board[0], 1.);
        assert_eq!(encoded.board[100], 1.);
        game.apply(actions[0]).unwrap();
        for rank in 0..10 {
            for file in 0..10 {
                let p = Position::new(rank, file).unwrap();
                assert_eq!(
                    encoded.board[p.id()],
                    f32::from(game.board().is_owned_by(p, actor))
                );
                assert_eq!(
                    encoded.board[200 + p.id()],
                    f32::from(game.board().is_in_sequence(p))
                );
            }
        }
    }
}

#[cfg(test)]
mod compatibility_tests {
    use super::*;
    use sequence_core::{CardSuit, Game, Simple};

    #[test]
    fn invalid_actions_return_errors() {
        let game = Simple::from_seed(42);
        assert!(matches!(
            EncodedAfterstate::new(game.observation(), Action::Pass),
            Err(ActionError::UnexpectedPass)
        ));
    }

    #[test]
    fn card_indices_preserve_checkpoint_order() {
        let ranks = [
            CardRank::Ace,
            CardRank::Two,
            CardRank::Three,
            CardRank::Four,
            CardRank::Five,
            CardRank::Six,
            CardRank::Seven,
            CardRank::Eight,
            CardRank::Nine,
            CardRank::Ten,
            CardRank::Queen,
            CardRank::King,
        ];
        let suits = [
            CardSuit::Clubs,
            CardSuit::Spades,
            CardSuit::Hearts,
            CardSuit::Diamonds,
        ];
        for (rank_index, rank) in ranks.into_iter().enumerate() {
            for (suit_index, suit) in suits.into_iter().enumerate() {
                assert_eq!(
                    card_index(Card::new(rank, suit)),
                    4 * rank_index + suit_index
                );
            }
        }
    }
}
