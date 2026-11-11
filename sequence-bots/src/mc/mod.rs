mod encoding;
mod model;
mod training;
pub use encoding::EncodedAfterstate;
pub use model::ValueNet;
pub use training::{TrainingConfig, train};

use crate::Bot;
use burn::prelude::*;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use sequence_core::{Action, SimpleObservation};

pub struct McBot {
    model: ValueNet,
    device: Device,
    rng: SmallRng,
    epsilon: f64,
}

impl McBot {
    pub fn new(model: ValueNet, device: Device, seed: u64, epsilon: f64) -> Self {
        assert!((0.0..=1.0).contains(&epsilon));
        Self {
            model: model.valid(),
            device: device.without_autodiff(),
            rng: SmallRng::seed_from_u64(seed),
            epsilon,
        }
    }

    pub(crate) fn select(
        &mut self,
        obs: SimpleObservation<'_>,
        actions: &[Action],
    ) -> (Action, EncodedAfterstate) {
        assert!(
            !actions.is_empty(),
            "cannot choose an action in a terminal game"
        );
        if self.rng.random::<f64>() < self.epsilon {
            let action = actions[self.rng.random_range(0..actions.len())];
            return (
                action,
                EncodedAfterstate::new(obs, action)
                    .expect("Monte Carlo bot requires legal Simple actions"),
            );
        }
        let candidates: Vec<_> = actions
            .iter()
            .map(|&a| {
                EncodedAfterstate::new(obs, a)
                    .expect("Monte Carlo bot requires legal Simple actions")
            })
            .collect();
        // One inference batch scores all legal moves from the same player's perspective.
        let (board, context) = model::batch(&candidates, &self.device);
        let values = self
            .model
            .forward(board, context)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let best = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        (actions[best], candidates[best].clone())
    }
}

impl Bot<sequence_core::Simple> for McBot {
    fn choose_action<'a>(&mut self, obs: SimpleObservation<'a>, actions: &[Action]) -> Action {
        self.select(obs, actions).0
    }
}
