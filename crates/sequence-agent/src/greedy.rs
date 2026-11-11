use sequence_core::{Action, Board, Color, Position, SimpleObservation};

use crate::{Agent, AgentError};

pub struct GreedyAgent {
    weights: [i32; 6],
}

impl Default for GreedyAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl GreedyAgent {
    pub const DEFAULT_WEIGHTS: [i32; 6] = [0, 0, 1, 5, 10, 100];

    pub fn new() -> Self {
        Self::with_weights(Self::DEFAULT_WEIGHTS)
    }
    pub fn with_weights(weights: [i32; 6]) -> Self {
        Self { weights }
    }

    pub fn score_board(&self, board: &Board, color: Color) -> i64 {
        let mut num_runs = [0i64; 6];
        for (dy, dx) in [(0, 1), (1, 0), (1, 1), (-1, 1)] {
            for start in Position::all() {
                if start.offset(-dy, -dx).is_ok() {
                    continue;
                }
                let mut position = Ok(start);
                let mut run: usize = 0;
                while let Ok(p) = position {
                    if board.is_owned_by(p, color) {
                        run += 1;
                    } else if run > 0 {
                        num_runs[run.min(5)] += 1;
                        run = 0;
                    }
                    position = p.offset(dy, dx);
                }
                if run > 0 {
                    num_runs[run.min(5)] += 1;
                }
            }
        }
        core::iter::zip(self.weights, num_runs)
            .map(|(weight, count)| i64::from(weight) * count)
            .sum()
    }
}

impl Agent<sequence_core::Simple> for GreedyAgent {
    fn choose_action(
        &mut self,
        obs: SimpleObservation<'_>,
        actions: &[Action],
    ) -> Result<Action, AgentError> {
        let mut best_action = *actions.first().ok_or("no legal actions")?;
        let mut best_score = i64::MIN;
        for &action in actions {
            let score = match action {
                Action::Play { position, .. } => {
                    let mut board = obs.board.clone();
                    board
                        .place_chip(position, obs.current_player)
                        .expect("bot requires legal actions");
                    self.score_board(&board, obs.current_player)
                }
                Action::Exchange { .. } => 10000,
                Action::Pass => -1,
            };
            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }
        Ok(best_action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sequence_core::{Game, Simple, get_card};

    #[test]
    fn scores_own_runs_without_opponent_penalties_or_win_bonuses() {
        let game = Simple::from_seed(42);
        let mut board = game.board().clone();
        for col in 2..5 {
            board
                .place_chip(Position::new(5, col).unwrap(), Color::Red)
                .unwrap();
        }
        for col in 0..4 {
            board
                .place_chip(Position::new(2, col).unwrap(), Color::Blue)
                .unwrap();
        }
        let extend = Position::new(5, 5).unwrap();
        let block = Position::new(2, 4).unwrap();
        let cards = [get_card(extend).unwrap(), get_card(block).unwrap()];
        let mut obs = game.observation();
        obs.board = &board;
        obs.hand = &cards;
        let actions = [
            Action::Play {
                card_index: 0,
                position: extend,
            },
            Action::Play {
                card_index: 1,
                position: block,
            },
        ];
        for action in actions {
            obs.validate(action).unwrap();
        }
        let mut bot = GreedyAgent::new();
        assert_eq!(bot.score_board(&board, Color::Red), 5);
        assert_eq!(bot.choose_action(obs, &actions).unwrap(), actions[0]);
    }

    #[test]
    fn tied_actions_keep_the_first_candidate() {
        let game = Simple::from_seed(42);
        let position = Position::new(5, 5).unwrap();
        let cards = [get_card(position).unwrap(); 2];
        let mut obs = game.observation();
        obs.hand = &cards;
        let actions = [
            Action::Play {
                card_index: 1,
                position,
            },
            Action::Play {
                card_index: 0,
                position,
            },
        ];
        assert_eq!(
            GreedyAgent::new().choose_action(obs, &actions).unwrap(),
            actions[0]
        );
    }
}
