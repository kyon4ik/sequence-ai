use rand::{SeedableRng, seq::IndexedRandom};
use rand_chacha::ChaCha8Rng;
use sequence_game::{Action, PlayerView};

use crate::Policy;

pub struct RandomAgent {
    rng: ChaCha8Rng,
}

impl RandomAgent {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl Policy for RandomAgent {
    fn choose_action(&mut self, view: &PlayerView<'_>) -> Option<Action> {
        view.legal_actions.choose(&mut self.rng).copied()
    }
}
