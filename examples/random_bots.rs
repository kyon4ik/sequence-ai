use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::IndexedRandom;
use sequence::{Action, Game, LegalActions};

struct RandomBot {
    rng: SmallRng,
}

impl RandomBot {
    fn from_seed(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    fn choose_action(&mut self, actions: LegalActions) -> Action {
        // println!("{:?}", actions);
        actions
            .choose(&mut self.rng)
            .copied()
            .unwrap_or(Action::Pass)
    }
}

fn main() {
    let mut game = Game::from_seed(0);
    let mut bots = [RandomBot::from_seed(42), RandomBot::from_seed(43)];

    while !game.is_finished() {
        let bot_index = game.get_current_player() as usize;
        let action = bots[bot_index].choose_action(game.legal_actions());
        match game.apply_action(action) {
            Ok(_) => println!("Player {bot_index}: {action:?}"),
            Err(err) => panic!("ERROR: {:?}", err),
        }
    }

    let turn_count = game.get_turn_count();
    let game_status = game.get_status();
    println!("Game finished in {turn_count} turns with {game_status:?} status")
}
