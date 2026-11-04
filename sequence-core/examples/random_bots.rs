use std::time::Instant;

use arrayvec::ArrayVec;
use rand::rngs::SmallRng;
use rand::seq::IndexedRandom;
use rand::{RngExt, SeedableRng};
use sequence_core::{Action, Board, GameOutcome, NoJacksGame, Observation};

trait Bot {
    fn choose_action<'a>(&mut self, obs: Observation<'a>, actions: &[Action]) -> Action;
}

#[derive(Clone, Copy, Debug)]
enum BotKind {
    Random,
    Greedy,
}

impl BotKind {
    fn create(self, seed: u64) -> Box<dyn Bot> {
        match self {
            BotKind::Random => Box::new(RandomBot::from_seed(seed)),
            BotKind::Greedy => Box::new(GreedyBot::new()),
        }
    }
}

struct RandomBot {
    rng: SmallRng,
}

struct GreedyBot {
    weights: [i32; 6],
}

impl GreedyBot {
    fn new() -> Self {
        let weights = [0, 0, 1, 5, 10, 100];
        Self { weights }
    }

    fn score_board(&self, board: &Board, our_mask: u8) -> i32 {
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

impl RandomBot {
    fn from_seed(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl Bot for RandomBot {
    fn choose_action<'a>(&mut self, _obs: Observation<'a>, actions: &[Action]) -> Action {
        *actions.choose(&mut self.rng).unwrap()
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

fn main() {
    run_games(67, 10000, [BotKind::Random, BotKind::Random], [43, 42]);
    run_games(67, 10000, [BotKind::Random, BotKind::Greedy], [43, 42]);
    run_games(67, 10000, [BotKind::Greedy, BotKind::Greedy], [43, 42]);
}

fn run_games(seed: u64, n_games: usize, bot_kinds: [BotKind; 2], bot_seeds: [u64; 2]) {
    let mut game_rng = SmallRng::seed_from_u64(seed);
    let mut legal_actions = Vec::with_capacity(sequence_core::MAX_LEGAL_ACTIONS);

    let timer = Instant::now();

    let mut wins = [0; 2];
    let mut draws = 0;

    for _ in 0..n_games {
        let mut game = NoJacksGame::from_seed(game_rng.random());
        let mut bots: ArrayVec<_, 2> = core::iter::zip(bot_kinds, bot_seeds)
            .map(|(kind, seed)| kind.create(seed))
            .collect();

        while !game.is_game_over() {
            let bot_index = game.us() as usize;
            let obs = game.observation();
            game.legal_actions(&mut legal_actions);
            let action = bots[bot_index].choose_action(obs, &legal_actions);
            legal_actions.clear();
            game.apply_unchecked(action);
        }

        match game.outcome() {
            GameOutcome::Unknown => unreachable!(),
            GameOutcome::Won(color) => wins[color as usize] += 1,
            GameOutcome::Draw => draws += 1,
        }
    }

    let execution_time = timer.elapsed();
    println!("{} games in {}s", n_games, execution_time.as_secs_f64());
    println!(
        "Red won {} | Blue won {} | Draws {}",
        wins[0], wins[1], draws,
    );
}
