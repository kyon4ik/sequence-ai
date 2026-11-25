use std::path::Path;

use burn::{
    module::Module,
    nn::{
        Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Bytes, Device, Tensor, TensorData, activation::relu},
};
use burn_pack::{Reader, Writer};
use rand::{
    Rng,
    distr::{Distribution, weighted::WeightedIndex},
};
use sequence_agents::Policy;
use sequence_game::{Action, BOARD_SIZE, PlayerView};
use serde::{Deserialize, Serialize};

use crate::Result;

use super::observation::{
    ACTIONS, BOARD_CHANNELS, CONTEXT_FEATURES, HISTORY_FEATURES, Observation, SQUARES, action_index,
};

const CHANNELS: usize = 32;
const HIDDEN: usize = 128;
const HISTORY_METADATA: &str = "sequence.move_history.v1";

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum MoveHistory {
    #[default]
    Disabled,
    Zeroed,
    Opponent,
}

impl MoveHistory {
    pub(super) fn is_disabled(&self) -> bool {
        *self == Self::Disabled
    }

    fn feature_count(self) -> usize {
        CONTEXT_FEATURES + HISTORY_FEATURES * usize::from(!self.is_disabled())
    }
}

#[derive(Module, Debug)]
pub struct Network {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    hidden: Linear,
    output: Linear,
    #[module(skip)]
    move_history: MoveHistory,
}

#[derive(Clone, Copy, Serialize)]
pub struct ParameterCounts {
    pub trainable_parameters: usize,
    pub weights: usize,
    pub biases: usize,
}

impl Network {
    pub(super) fn new(device: &Device, move_history: MoveHistory) -> Self {
        let conv = |inputs| {
            Conv2dConfig::new([inputs, CHANNELS], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device)
        };
        Self {
            conv1: conv(BOARD_CHANNELS),
            conv2: conv(CHANNELS),
            conv3: conv(CHANNELS),
            hidden: LinearConfig::new(CHANNELS * SQUARES + move_history.feature_count(), HIDDEN)
                .init(device),
            output: LinearConfig::new(HIDDEN, ACTIONS).init(device),
            move_history,
        }
    }

    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let reader = Reader::from_file(path)?;
        let move_history = match reader.metadata().get(HISTORY_METADATA).map(String::as_str) {
            None => MoveHistory::Disabled,
            Some("zeroed") => MoveHistory::Zeroed,
            Some("opponent") => MoveHistory::Opponent,
            Some(mode) => {
                return Err(format!("unknown checkpoint move history mode {mode:?}").into());
            }
        };
        Ok(Self::new(device, move_history).try_load_file(path)?.valid())
    }

    pub(super) fn checkpoint(&self) -> Result<Bytes> {
        let bytes = self.valid().into_record().into_bytes()?;
        let mode = match self.move_history {
            MoveHistory::Disabled => return Ok(bytes),
            MoveHistory::Zeroed => "zeroed",
            MoveHistory::Opponent => "opponent",
        };
        let tensors = Reader::from_bytes(bytes)?.into_tensors()?;
        Ok(Writer::new(tensors)
            .with_metadata(HISTORY_METADATA, mode)
            .into_bytes()?)
    }

    pub(super) fn observe(&self, view: &PlayerView<'_>) -> Observation {
        Observation::new(view, self.move_history == MoveHistory::Opponent)
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
        let feature_count = self.move_history.feature_count();
        let mut boards = Vec::with_capacity(count * BOARD_CHANNELS * SQUARES);
        let mut features = Vec::with_capacity(count * feature_count);
        for observation in observations {
            boards.extend_from_slice(&observation.board);
            features.extend_from_slice(&observation.features);
            let start = features.len();
            features.resize(start + feature_count - CONTEXT_FEATURES, 0.0);
            if self.move_history == MoveHistory::Opponent {
                observation.write_history(&mut features[start..]);
            }
        }

        let boards = Tensor::from_data(
            TensorData::new(boards, [count, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE]),
            device,
        );
        let features = Tensor::from_data(TensorData::new(features, [count, feature_count]), device);
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
        let scores = self.action_scores(observation, device);
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

    pub fn sample_action(
        &self,
        view: &PlayerView<'_>,
        observation: &Observation,
        device: &Device,
        temperature: f64,
        rng: &mut impl Rng,
    ) -> Result<Action> {
        let scores = self.action_scores(observation, device);
        let scores = scores.as_slice::<f32>().expect("DQN scores are f32");
        sample_legal_action(&view.legal_actions, scores, temperature, rng)
    }

    pub(super) fn action_scores(&self, observation: &Observation, device: &Device) -> TensorData {
        self.forward(std::iter::once(observation), device)
            .into_data()
    }
}

fn sample_legal_action(
    actions: &[Action],
    scores: &[f32],
    temperature: f64,
    rng: &mut impl Rng,
) -> Result<Action> {
    let candidates: Vec<_> = actions
        .iter()
        .filter_map(|&action| action_index(action).map(|index| (action, f64::from(scores[index]))))
        .collect();
    if candidates.iter().any(|&(_, score)| !score.is_finite()) {
        return Err("DQN produced a non-finite score for a legal action".into());
    }

    let best = candidates
        .iter()
        .map(|&(_, score)| score)
        .fold(f64::NEG_INFINITY, f64::max);
    let weights = WeightedIndex::new(
        candidates
            .iter()
            .map(|&(_, score)| ((score - best) / temperature).exp()),
    )
    .map_err(|error| format!("could not sample a legal DQN action: {error}"))?;

    Ok(candidates[weights.sample(rng)].0)
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
                    .greedy_action(view, &self.network.observe(view), self.device),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use sequence_game::{Game, GameConfig, Player};

    use super::*;

    #[test]
    fn saved_history_models_score_public_plays_or_mask_them() {
        let device = Device::flex();
        let network = Network::new(&device, MoveHistory::Opponent);
        let mut game = Game::new(GameConfig {
            shuffle_seed: 0,
            starting_player: Player::One,
            policies: Default::default(),
        });
        let action = game.view(Player::One).legal_actions[0];
        game.apply(Player::One, action).unwrap();
        let view = game.view(Player::Two);
        let with_history = Observation::new(&view, true);
        let without_history = Observation::new(&view, false);
        let directory = tempfile::tempdir().unwrap();
        let scores = |network: &Network, observation| {
            network
                .action_scores(observation, &device)
                .try_into_vec::<f32>()
                .unwrap()
        };

        for mode in [MoveHistory::Opponent, MoveHistory::Zeroed] {
            let model = Network {
                move_history: mode,
                ..network.clone()
            };
            let path = directory.path().join("model.bpk");
            std::fs::write(&path, &*model.checkpoint().unwrap()).unwrap();
            let loaded = Network::load(&path, &device).unwrap();

            assert_eq!(loaded.move_history, mode);
            assert_eq!(
                scores(&loaded, &with_history) == scores(&loaded, &without_history),
                mode == MoveHistory::Zeroed,
            );
        }
    }

    #[test]
    fn temperature_weights_only_legal_scores_and_stays_stable_at_low_temperature() {
        let game = Game::new(GameConfig {
            shuffle_seed: 0,
            starting_player: Player::One,
            policies: Default::default(),
        });
        let view = game.view(Player::One);
        let actions = &view.legal_actions[..2];
        let [best, other] = [actions[0], actions[1]];
        let mut scores = [f32::MAX; ACTIONS];
        scores[action_index(best).unwrap()] = 1.0;
        scores[action_index(other).unwrap()] = 0.0;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let selected = sample_legal_action(actions, &scores, f64::MIN_POSITIVE, &mut rng).unwrap();
        assert_eq!(selected, best);

        let best_count = (0..1000)
            .filter(|_| sample_legal_action(actions, &scores, 1.0, &mut rng).unwrap() == best)
            .count();
        assert!((680..780).contains(&best_count));
    }
}
