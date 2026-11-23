use std::path::Path;

use burn::{
    module::Module,
    nn::{
        Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Device, Tensor, TensorData, activation::relu},
};
use sequence_agents::Policy;
use sequence_game::{Action, BOARD_SIZE, PlayerView};
use serde::Serialize;

use super::observation::{
    ACTIONS, BOARD_CHANNELS, CONTEXT_FEATURES, Observation, SQUARES, action_index,
};

const CHANNELS: usize = 32;
const HIDDEN: usize = 128;

#[derive(Module, Debug)]
pub struct Network {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    hidden: Linear,
    output: Linear,
}

#[derive(Clone, Copy, Serialize)]
pub struct ParameterCounts {
    pub trainable_parameters: usize,
    pub weights: usize,
    pub biases: usize,
}

impl Network {
    pub fn new(device: &Device) -> Self {
        let conv = |inputs| {
            Conv2dConfig::new([inputs, CHANNELS], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device)
        };
        Self {
            conv1: conv(BOARD_CHANNELS),
            conv2: conv(CHANNELS),
            conv3: conv(CHANNELS),
            hidden: LinearConfig::new(CHANNELS * SQUARES + CONTEXT_FEATURES, HIDDEN).init(device),
            output: LinearConfig::new(HIDDEN, ACTIONS).init(device),
        }
    }

    pub fn load(path: &Path, device: &Device) -> crate::Result<Self> {
        Ok(Self::new(device).try_load_file(path)?.valid())
    }

    pub fn parameter_counts(&self) -> ParameterCounts {
        let trainable_parameters = self.num_params();
        let biases = self.conv1.bias.num_params()
            + self.conv2.bias.num_params()
            + self.conv3.bias.num_params()
            + self.hidden.bias.num_params()
            + self.output.bias.num_params();
        ParameterCounts {
            trainable_parameters,
            weights: trainable_parameters - biases,
            biases,
        }
    }

    pub fn forward<'a>(
        &self,
        observations: impl ExactSizeIterator<Item = &'a Observation>,
        device: &Device,
    ) -> Tensor<2> {
        let count = observations.len();
        let mut boards = Vec::with_capacity(count * BOARD_CHANNELS * SQUARES);
        let mut features = Vec::with_capacity(count * CONTEXT_FEATURES);
        for observation in observations {
            boards.extend_from_slice(&observation.board);
            features.extend_from_slice(&observation.features);
        }

        let boards = Tensor::from_data(
            TensorData::new(boards, [count, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE]),
            device,
        );
        let features =
            Tensor::from_data(TensorData::new(features, [count, CONTEXT_FEATURES]), device);
        let board = relu(self.conv1.forward(boards));
        let board = relu(self.conv2.forward(board));
        let board = relu(self.conv3.forward(board)).flatten(1, 3);
        let hidden = relu(self.hidden.forward(Tensor::cat(vec![board, features], 1)));
        self.output.forward(hidden)
    }

    pub fn greedy_action(
        &self,
        view: &PlayerView<'_>,
        observation: &Observation,
        device: &Device,
    ) -> Action {
        let scores = self
            .forward(std::iter::once(observation), device)
            .into_data();
        let scores = scores.as_slice::<f32>().expect("DQN scores are f32");

        view.legal_actions
            .iter()
            .filter_map(|&action| action_index(action).map(|index| (action, index)))
            .max_by(|&(_, first), &(_, second)| {
                scores[first]
                    .total_cmp(&scores[second])
                    .then_with(|| second.cmp(&first))
            })
            .map(|(action, _)| action)
            .expect("a DQN decision has a scored legal action")
    }
}

pub struct GreedyPolicy<'a> {
    pub network: &'a Network,
    pub device: &'a Device,
}

impl Policy for GreedyPolicy<'_> {
    fn choose_action(&mut self, view: &PlayerView<'_>) -> Option<Action> {
        match view.legal_actions.as_slice() {
            [] => None,
            [Action::Pass] => Some(Action::Pass),
            _ => Some(
                self.network
                    .greedy_action(view, &Observation::new(view), self.device),
            ),
        }
    }
}
