use sequence_core::{Action, Board, Observation};

use crate::Bot;

pub struct GreedyBot {
    weights: [i32; 6],
}

impl GreedyBot {
    pub fn new() -> Self {
        let weights = [0, 0, 1, 5, 10, 100];
        Self { weights }
    }

    pub fn score_board(&self, board: &Board, our_mask: u8) -> i32 {
        let mut score = 0;

        let mut num_runs = [0; 6];
        for (dy, dx) in Board::DIRS {
            board.scan_direction(dy, dx, our_mask, &mut num_runs);
        }

        for (weight, num_run) in core::iter::zip(self.weights, num_runs) {
            score += weight * num_run;
        }

        score
    }
}

impl Bot for GreedyBot {
    fn choose_action<'a>(&mut self, obs: Observation<'a>, actions: &[Action]) -> Action {
        let mut best_action = Action::Pass;
        let mut best_score = -1;

        let our_mask = 1 << obs.color as u8;

        for &action in actions {
            let mut board = *obs.board;
            let score = match action {
                Action::Play { position, .. } => {
                    board.set(position, our_mask);
                    self.score_board(&board, our_mask)
                }
                Action::Exchange { .. } => 10000,
                Action::Pass => -1,
            };

            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }

        best_action
    }
}
