use std::time::Instant;

use rand::rngs::SmallRng;
use rand::seq::IndexedRandom;
use rand::{RngExt, SeedableRng};
use sequence::{Action, NoJacksGame};

struct RandomBot {
    rng: SmallRng,
}

impl RandomBot {
    fn from_seed(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    fn choose_action(&mut self, actions: &[Action]) -> Action {
        *actions.choose(&mut self.rng).unwrap()
    }
}

fn main() {
    run_games(67, 100000, [43, 42]);
}

fn run_games(seed: u64, n_games: usize, bot_seeds: [u64; 2]) {
    let mut game_rng = SmallRng::seed_from_u64(seed);
    let mut legal_actions = Vec::with_capacity(sequence::MAX_LEGAL_ACTIONS);

    let timer = Instant::now();

    let mut wins = [0; 2];
    let mut draws = 0;

    for _ in 0..n_games {
        let mut game = NoJacksGame::from_seed(game_rng.random());
        let mut bots = bot_seeds.map(RandomBot::from_seed);

        while !game.is_game_over() {
            let bot_index = game.us() as usize;
            game.legal_actions(&mut legal_actions);
            let action = bots[bot_index].choose_action(&legal_actions);
            legal_actions.clear();
            game.apply_unchecked(action);
        }

        match game.outcome() {
            sequence::GameOutcome::Unknown => unreachable!(),
            sequence::GameOutcome::Won(color) => wins[color as usize] += 1,
            sequence::GameOutcome::Draw => draws += 1,
        }
    }

    let execution_time = timer.elapsed();
    println!("{} games in {}s", n_games, execution_time.as_secs_f64());
    println!(
        "Red won {} | Blue won {} | Draws {}",
        wins[0], wins[1], draws,
    );
}
