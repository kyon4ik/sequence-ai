use sequence_core::{Action, Observation};

use crate::greedy::GreedyBot;
use crate::random::RandomBot;

mod greedy;
mod random;

#[cfg(feature = "burn")]
pub mod mc;

pub trait Bot {
    fn choose_action<'a>(&mut self, obs: Observation<'a>, actions: &[Action]) -> Action;
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
    pub fn create(&self, seed: u64) -> Box<dyn Bot> {
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
                // Arena agents evaluate greedily; exploration belongs to training.
                Box::new(mc::McBot::new(model, device, seed, 0.0))
            }
        }
    }
}
