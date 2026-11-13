mod encoding;
mod model;
pub use encoding::EncodedAfterstate;
pub use model::{ValueNet, batch};

use crate::{Agent, AgentError, EpisodeContext};
use burn::prelude::*;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use sequence_core::{Action, SimpleObservation};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Exploration belongs to a seat in one game, independent of batch scheduling.
pub struct McPolicyState {
    rng: SmallRng,
    epsilon: f64,
}

impl McPolicyState {
    pub fn new(seed: u64, epsilon: f64) -> Self {
        assert!((0.0..=1.0).contains(&epsilon));
        Self {
            rng: SmallRng::seed_from_u64(seed),
            epsilon,
        }
    }

    fn explore(&mut self, actions: &[Action]) -> Option<Action> {
        if self.rng.random::<f64>() < self.epsilon {
            Some(actions[self.rng.random_range(0..actions.len())])
        } else {
            None
        }
    }
}

pub struct McDecision<'a> {
    pub observation: SimpleObservation<'a>,
    pub actions: &'a [Action],
    pub state: &'a mut McPolicyState,
}

/// Host wall times. Prediction includes packing/upload, GPU execution and the
/// synchronized readback; it is not a CUDA kernel-only timer.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct McInferenceMetrics {
    pub decisions: usize,
    pub batches: usize,
    pub candidates: usize,
    pub encoding_time: Duration,
    pub prediction_time: Duration,
}

/// One frozen policy shared across games. All non-exploratory candidates in a
/// scheduling round are evaluated in one forward pass and one readback.
pub struct McBatchPolicy {
    model: ValueNet,
    device: Device,
    candidates: Vec<EncodedAfterstate>,
    metrics: McInferenceMetrics,
}

impl McBatchPolicy {
    pub fn new(model: ValueNet, device: Device) -> Self {
        Self {
            model: model.valid(),
            device: device.without_autodiff(),
            candidates: Vec::new(),
            metrics: McInferenceMetrics::default(),
        }
    }

    pub fn metrics(&self) -> &McInferenceMetrics {
        &self.metrics
    }

    pub fn choose_actions(
        &mut self,
        requests: &mut [McDecision<'_>],
    ) -> Result<Vec<Action>, AgentError> {
        if requests.iter().any(|r| r.actions.is_empty()) {
            return Err("no legal actions".into());
        }
        self.candidates.clear();
        let started = Instant::now();
        let mut choices: Vec<_> = requests.iter().map(|r| r.actions[0]).collect();
        let mut pending = Vec::new();
        for (index, request) in requests.iter_mut().enumerate() {
            if let Some(action) = request.state.explore(request.actions) {
                choices[index] = action;
                continue;
            }
            let start = self.candidates.len();
            for &action in request.actions {
                self.candidates
                    .push(EncodedAfterstate::new(request.observation, action)?);
            }
            pending.push((index, start..self.candidates.len()));
        }
        self.metrics.encoding_time += started.elapsed();
        self.metrics.decisions += requests.len();
        if self.candidates.is_empty() {
            return Ok(choices);
        }

        let started = Instant::now();
        let (board, context) = model::batch(&self.candidates, &self.device);
        let values = self
            .model
            .forward(board, context)
            .try_into_data()
            .map_err(|e| format!("reading model predictions: {e}"))?
            .try_to_vec::<f32>()?;
        self.metrics.prediction_time += started.elapsed();
        self.metrics.batches += 1;
        self.metrics.candidates += self.candidates.len();
        if values.len() != self.candidates.len() || values.iter().any(|value| !value.is_finite()) {
            return Err("model returned invalid action values".into());
        }
        for (index, range) in pending {
            // Match scalar inference's last-maximum tie breaking.
            let best = values[range]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            choices[index] = requests[index].actions[best];
        }
        Ok(choices)
    }
}

pub struct McAgent {
    policy: McBatchPolicy,
    state: McPolicyState,
}

impl McAgent {
    pub fn new(model: ValueNet, device: Device, seed: u64, epsilon: f64) -> Self {
        Self {
            policy: McBatchPolicy::new(model, device),
            state: McPolicyState::new(seed, epsilon),
        }
    }
}

impl Agent<sequence_core::Simple> for McAgent {
    fn choose_action(
        &mut self,
        obs: SimpleObservation<'_>,
        actions: &[Action],
    ) -> Result<Action, AgentError> {
        Ok(self.policy.choose_actions(&mut [McDecision {
            observation: obs,
            actions,
            state: &mut self.state,
        }])?[0])
    }
    fn on_episode_start(&mut self, context: EpisodeContext) {
        self.state.rng = SmallRng::seed_from_u64(context.agent_seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sequence_core::{Game, Simple};

    #[test]
    fn batching_preserves_predictions_and_exploration() {
        assert_batched_predictions(Device::flex());
    }

    #[test]
    #[ignore = "requires host NVIDIA GPU access"]
    fn cuda_batching_preserves_predictions_and_exploration() {
        assert_batched_predictions(Device::cuda(0));
    }

    fn assert_batched_predictions(device: Device) {
        device.seed(431);
        let net = ValueNet::new(&device).valid();
        let games: Vec<_> = (0..4)
            .map(|seed| {
                let mut game = Simple::from_seed(seed);
                let mut actions = Vec::new();
                for _ in 0..seed * 4 {
                    game.legal_actions(&mut actions);
                    game.apply(actions[0]).unwrap();
                }
                game
            })
            .collect();
        let actions: Vec<_> = games
            .iter()
            .map(|game| {
                let mut actions = Vec::new();
                game.legal_actions(&mut actions);
                actions
            })
            .collect();
        let encoded: Vec<Vec<_>> = games
            .iter()
            .zip(&actions)
            .map(|(game, actions)| {
                actions
                    .iter()
                    .map(|&a| EncodedAfterstate::new(game.observation(), a).unwrap())
                    .collect()
            })
            .collect();
        let predict = |inputs: &[EncodedAfterstate]| {
            let (board, context) = batch(inputs, &device);
            net.forward(board, context)
                .try_into_data()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap()
        };
        let scalar: Vec<Vec<_>> = encoded.iter().map(|inputs| predict(inputs)).collect();
        let combined: Vec<_> = encoded.iter().flatten().cloned().collect();
        for (expected, actual) in scalar.iter().flatten().zip(predict(&combined)) {
            assert!(
                (expected - actual).abs() < 1e-4,
                "scalar={expected} batched={actual}"
            );
        }
        // Check the old RNG consumption independently, including mixed random
        // and greedy decisions and several successive calls per seat.
        let epsilons = [0., 0.3, 1., 0.7];
        let mut states: Vec<_> = epsilons
            .iter()
            .enumerate()
            .map(|(i, &epsilon)| McPolicyState::new(i as u64, epsilon))
            .collect();
        let mut reference: Vec<_> = (0..4).map(SmallRng::seed_from_u64).collect();
        let mut policy = McBatchPolicy::new(net, device);
        for _ in 0..8 {
            let mut requests: Vec<_> = games
                .iter()
                .zip(&actions)
                .zip(&mut states)
                .map(|((game, actions), state)| McDecision {
                    observation: game.observation(),
                    actions,
                    state,
                })
                .collect();
            let selected = policy.choose_actions(&mut requests).unwrap();
            for i in 0..4 {
                if reference[i].random::<f64>() < epsilons[i] {
                    assert_eq!(
                        selected[i],
                        actions[i][reference[i].random_range(0..actions[i].len())]
                    );
                } else {
                    let index = actions[i].iter().position(|a| *a == selected[i]).unwrap();
                    let max = scalar[i].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    assert!(max - scalar[i][index] < 2e-4);
                }
            }
        }
        assert_eq!(policy.metrics().decisions, 32);
        assert!(policy.choose_actions(&mut []).unwrap().is_empty());
        assert!(
            policy
                .choose_actions(&mut [McDecision {
                    observation: games[0].observation(),
                    actions: &[],
                    state: &mut states[0]
                }])
                .is_err()
        );
    }
}
