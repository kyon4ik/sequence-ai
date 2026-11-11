mod encoding;
mod model;
pub use encoding::EncodedAfterstate;
pub use model::{ValueNet, batch};

use crate::{Agent, AgentError, EpisodeContext};
use burn::prelude::*;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use sequence_core::{Action, SimpleObservation};

pub struct McAgent {
    model: ValueNet,
    device: Device,
    rng: SmallRng,
    epsilon: f64,
}

impl McAgent {
    pub fn new(model: ValueNet, device: Device, seed: u64, epsilon: f64) -> Self {
        assert!((0.0..=1.0).contains(&epsilon));
        Self {
            model: model.valid(),
            device: device.without_autodiff(),
            rng: SmallRng::seed_from_u64(seed),
            epsilon,
        }
    }

    fn select(
        &mut self,
        obs: SimpleObservation<'_>,
        actions: &[Action],
    ) -> Result<Action, AgentError> {
        if self.rng.random::<f64>() < self.epsilon {
            let action = actions[self.rng.random_range(0..actions.len())];
            return Ok(action);
        }
        let candidates: Vec<_> = actions
            .iter()
            .map(|&a| EncodedAfterstate::new(obs, a))
            .collect::<Result<_, _>>()?;
        // One inference batch scores all legal moves from the same player's perspective.
        let (board, context) = model::batch(&candidates, &self.device);
        let values = self
            .model
            .forward(board, context)
            .into_data()
            .try_to_vec::<f32>()?;
        if values.len() != actions.len() || values.iter().any(|value| !value.is_finite()) {
            return Err("model returned invalid action values".into());
        }
        let best = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        Ok(actions[best])
    }
}

impl Agent<sequence_core::Simple> for McAgent {
    fn choose_action(
        &mut self,
        obs: SimpleObservation<'_>,
        actions: &[Action],
    ) -> Result<Action, AgentError> {
        if actions.is_empty() {
            return Err("no legal actions".into());
        }
        self.select(obs, actions)
    }
    fn on_episode_start(&mut self, context: EpisodeContext) {
        self.rng = SmallRng::seed_from_u64(context.agent_seed);
    }
}
