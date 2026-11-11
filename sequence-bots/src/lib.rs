use sequence_core::{Action, Game, Simple};

use crate::greedy::GreedyBot;
use crate::random::RandomBot;

mod afterstate;
pub use afterstate::Afterstate;

mod greedy;
mod random;

#[cfg(feature = "burn")]
pub mod mc;

/// A policy for one game variant. Current implementations support only Simple.
pub trait Bot<G: Game = Simple> {
    fn choose_action<'a>(&mut self, obs: G::Observation<'a>, actions: &[Action]) -> Action;
}

#[derive(Clone, Debug)]
pub enum BotKind {
    Random,
    Greedy,
    #[cfg(feature = "burn")]
    MonteCarlo {
        checkpoint: std::path::PathBuf,
    },
}

impl BotKind {
    pub fn create(&self, seed: u64) -> Box<dyn Bot<Simple>> {
        match self {
            BotKind::Random => Box::new(RandomBot::from_seed(seed)),
            BotKind::Greedy => Box::new(GreedyBot::new()),
            #[cfg(feature = "burn")]
            BotKind::MonteCarlo { checkpoint } => {
                use burn::prelude::*;
                let device = Device::cuda(0);
                let model = mc::ValueNet::new(&device)
                    .try_load_file(checkpoint)
                    .unwrap_or_else(|error| {
                        panic!("cannot load {}: {error}", checkpoint.display())
                    });
                // Checkpoint agents evaluate greedily; exploration belongs to training.
                Box::new(mc::McBot::new(model, device, seed, 0.0))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_and_greedy_play_complete_simple_games() {
        for seed in 0..8 {
            let mut game = Simple::from_seed(seed);
            let mut bots = [
                BotKind::Random.create(seed),
                BotKind::Greedy.create(seed + 1),
            ];
            let mut actions = vec![];
            while !game.is_game_over() {
                game.legal_actions(&mut actions);
                let action = bots[game.current_player() as usize]
                    .choose_action(game.observation(), &actions);
                assert!(actions.contains(&action));
                game.apply(action).unwrap();
                assert!(game.turns() <= 96);
            }
        }
    }
}
