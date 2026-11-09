use sequence_core::{Action, Board, Card, CardRank, Observation, Position, get_card};

#[derive(Clone, Debug)]
pub struct EncodedAfterstate {
    pub board: [f32; 400],
    pub context: [f32; 50],
}

// TODO: Make Jack last rank to remove this encoding
fn card_index(card: Card) -> usize {
    let rank = card.rank() as usize;
    assert_ne!(card.rank(), CardRank::Jack, "no-jacks encoder");
    (rank - usize::from(rank > CardRank::Jack as usize)) * 4 + card.suit() as usize
}

impl EncodedAfterstate {
    /// `action` must be a legal placement from this observation.
    pub fn new(obs: Observation<'_>, action: Action) -> Self {
        let Action::Play {
            card_index: played,
            position,
        } = action
        else {
            panic!("no-jacks agent requires a placement")
        };

        // Preview only deterministic effects: never draw or inspect the opponent's hand.
        let mut board = *obs.board;
        let ours = 1 << obs.color as u8;
        let theirs = 1 << obs.color.next() as u8;
        board.set(position, ours);
        let added = board.update_sequences(position, obs.color);
        let mut encoded = Self {
            board: [0.; 400],
            context: [0.; 50],
        };
        assert!(played < obs.hand.len());
        for (i, &card) in obs.hand.iter().enumerate() {
            if i != played {
                encoded.context[card_index(card)] += 0.5;
            }
        }
        encoded.context[48] = (obs.sequences[obs.color as usize] + added) as f32 / 2.;
        encoded.context[49] = obs.sequences[obs.color.next() as usize] as f32 / 2.;
        for rank in 0..10 {
            for file in 0..10 {
                let p = Position::new(rank, file).unwrap();
                let i = p.id();
                // Channel-first layout; free corners belong to both players.
                encoded.board[i] = f32::from(board.get(p, ours) != 0);
                encoded.board[100 + i] = f32::from(board.get(p, theirs) != 0);
                encoded.board[200 + i] = f32::from(board.get(p, Board::SEQ_MASK) != 0);
                if let Some(card) = get_card(p) {
                    encoded.board[300 + i] = encoded.context[card_index(card)];
                }
            }
        }
        encoded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sequence_core::NoJacksGame;

    #[test]
    fn preview_matches_actual_board_but_excludes_draw() {
        let mut game = NoJacksGame::from_seed(42);
        let mut actions = Vec::new();
        game.legal_actions(&mut actions);
        let actor = game.us();
        let before = *game.board();
        let encoded = EncodedAfterstate::new(game.observation(), actions[0]);
        assert_eq!(*game.board(), before);
        assert_eq!(encoded.context[..48].iter().sum::<f32>(), 3.);
        assert_eq!(encoded.board[0], 1.);
        assert_eq!(encoded.board[100], 1.);
        game.apply_unchecked(actions[0]);
        for rank in 0..10 {
            for file in 0..10 {
                let p = Position::new(rank, file).unwrap();
                assert_eq!(
                    encoded.board[p.id()],
                    f32::from(game.board().get(p, 1 << actor as u8) != 0)
                );
                assert_eq!(
                    encoded.board[200 + p.id()],
                    f32::from(game.board().get(p, Board::SEQ_MASK) != 0)
                );
            }
        }
    }
}
