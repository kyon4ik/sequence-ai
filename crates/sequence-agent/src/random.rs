use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::IndexedRandom;
use sequence_core::{Action, Game};

use crate::{Agent, AgentError, EpisodeContext};

pub struct RandomAgent {
    rng: SmallRng,
}

impl RandomAgent {
    pub fn from_seed(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl<G: Game> Agent<G> for RandomAgent {
    fn choose_action(
        &mut self,
        _obs: G::Observation<'_>,
        actions: &[Action],
    ) -> Result<Action, AgentError> {
        actions
            .choose(&mut self.rng)
            .copied()
            .ok_or_else(|| "no legal actions".into())
    }
    fn on_episode_start(&mut self, context: EpisodeContext) {
        self.rng = SmallRng::seed_from_u64(context.agent_seed);
    }
}
