use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::IndexedRandom;
use sequence_core::{Action, Observation};

use crate::Bot;

pub struct RandomBot {
    rng: SmallRng,
}

impl RandomBot {
    pub fn from_seed(seed: u64) -> Self {
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
