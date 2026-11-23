use std::{collections::VecDeque, rc::Rc};

use burn::{
    module::Module,
    nn::loss::{HuberLossConfig, Reduction},
    optim::{AdamConfig, GradientsParams, ModuleOptimizer},
    tensor::{Bool, Device, Int, Tensor, TensorData},
};
use rand::{Rng, SeedableRng, seq::index};
use rand_chacha::ChaCha8Rng;
use sequence_game::{Action, GameConfig, Outcome, Player};
use sequence_match::{Match, MatchConfig, MatchEnd};
use serde::Serialize;

use crate::{Result, experiment::GameEnd};

use super::{
    Config,
    model::Network,
    observation::{ACTIONS, Observation, action_index, legal_mask},
};

struct Decision {
    observation: Observation,
    legal: [bool; ACTIONS],
}

struct Experience {
    state: Rc<Decision>,
    action: usize,
    reward: f32,
    next: Option<Rc<Decision>>,
    decisions_to_end: Option<usize>,
}

impl Experience {
    fn is_within_depth(&self, depth: usize) -> bool {
        self.decisions_to_end
            .is_some_and(|distance| distance <= depth)
    }
}

#[derive(Serialize)]
pub struct Episode {
    episode: usize,
    shuffle_seed: u64,
    learner_seat: usize,
    opponent_episode: usize,
    outcome: GameEnd,
    actions: usize,
    decisions: usize,
    updates: usize,
    epsilon: f64,
    reward: Option<f32>,
    replay_depth: Option<usize>,
    replay_eligible: usize,
}

pub struct Trainer<'a> {
    config: &'a Config,
    device: &'a Device,
    training_device: Device,
    pub online: Network,
    target: Network,
    opponent: Network,
    optimizer: ModuleOptimizer,
    replay: VecDeque<Experience>,
    exploration: ChaCha8Rng,
    sampling: ChaCha8Rng,
    opponent_episode: usize,
    pub decisions: usize,
    pub updates: usize,
    replay_insertions: usize,
}

impl<'a> Trainer<'a> {
    pub fn new(config: &'a Config, device: &'a Device) -> Self {
        device.seed(config.seed);
        let training_device = device.clone().autodiff();
        let online = Network::new(&training_device);
        let mut sampling = ChaCha8Rng::seed_from_u64(config.seed);
        sampling.set_stream(1);

        Self {
            config,
            device,
            training_device,
            target: online.valid(),
            opponent: online.valid(),
            online,
            optimizer: AdamConfig::new().init(),
            replay: VecDeque::with_capacity(config.replay_capacity.get()),
            exploration: ChaCha8Rng::seed_from_u64(config.seed),
            sampling,
            opponent_episode: 0,
            decisions: 0,
            updates: 0,
            replay_insertions: 0,
        }
    }

    pub fn play_episode(&mut self, episode: usize) -> Result<Episode> {
        let learner = if episode.is_multiple_of(2) {
            Player::Two
        } else {
            Player::One
        };
        let shuffle_seed = self.config.first_shuffle_seed + (episode - 1) as u64;
        let mut policies = ["dqn".into(), format!("dqn@{}", self.opponent_episode)];
        if learner == Player::Two {
            policies.swap(0, 1);
        }
        let mut game = Match::new(
            MatchConfig {
                game: GameConfig {
                    shuffle_seed,
                    starting_player: Player::One,
                    policies,
                },
                action_limit: self.config.action_limit,
                policy_seeds: [None, None],
            },
            false,
        );

        let replay_depth = self.config.replay_depth(episode);
        let decisions_before_episode = self.decisions;
        let mut pending = None;
        while let Some(view) = game.active_view() {
            let actor = view.player;
            let action = if view.legal_actions == [Action::Pass] {
                Action::Pass
            } else if actor != learner {
                self.opponent
                    .greedy_action(&view, &Observation::new(&view), self.device)
            } else {
                let decision = Rc::new(Decision {
                    observation: Observation::new(&view),
                    legal: legal_mask(&view),
                });
                if let Some((state, action)) = pending.take() {
                    self.record(
                        Experience {
                            state,
                            action,
                            reward: 0.0,
                            next: Some(Rc::clone(&decision)),
                            decisions_to_end: None,
                        },
                        replay_depth,
                    )?;
                }

                let action = if self.exploration.random::<f64>() < self.epsilon() {
                    view.legal_actions[self.exploration.random_range(0..view.legal_actions.len())]
                } else {
                    self.online
                        .valid()
                        .greedy_action(&view, &decision.observation, self.device)
                };
                pending = Some((decision, action_index(action).expect("scored action")));
                self.decisions += 1;
                action
            };
            game.step(actor, action)?;
        }

        let end = game.end().expect("match has finished");
        let reward = match end {
            MatchEnd::Completed(Outcome::Win(winner)) if winner == learner => Some(1.0),
            MatchEnd::Completed(Outcome::Win(_)) => Some(-1.0),
            MatchEnd::Completed(Outcome::Draw) => Some(0.0),
            MatchEnd::ActionLimit => None,
        };
        if let (Some((state, action)), Some(reward)) = (pending, reward) {
            if replay_depth.is_some() {
                // The terminal decision is still pending. All earlier decisions
                // from this game form the newest suffix of replay.
                let earlier_decisions = self.decisions - decisions_before_episode - 1;
                for (index, experience) in self
                    .replay
                    .iter_mut()
                    .rev()
                    .take(earlier_decisions)
                    .enumerate()
                {
                    experience.decisions_to_end = Some(index + 2);
                }
            }
            self.record(
                Experience {
                    state,
                    action,
                    reward,
                    next: None,
                    decisions_to_end: Some(1),
                },
                replay_depth,
            )?;
        }

        let row = Episode {
            episode,
            shuffle_seed,
            learner_seat: learner.index() + 1,
            opponent_episode: self.opponent_episode,
            outcome: end.into(),
            actions: game.action_count(),
            decisions: self.decisions,
            updates: self.updates,
            epsilon: self.epsilon(),
            reward,
            replay_depth,
            replay_eligible: replay_depth.map_or(self.replay.len(), |depth| {
                self.replay
                    .iter()
                    .filter(|experience| experience.is_within_depth(depth))
                    .count()
            }),
        };
        if episode.is_multiple_of(self.config.opponent_update_every.get()) {
            self.opponent = self.online.valid();
            self.opponent_episode = episode;
        }
        Ok(row)
    }

    fn epsilon(&self) -> f64 {
        let duration = self.config.epsilon_decay_decisions.get();
        let progress = self.decisions.min(duration) as f64 / duration as f64;
        self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)
    }

    fn record(&mut self, experience: Experience, replay_depth: Option<usize>) -> Result<()> {
        if self.replay.len() == self.config.replay_capacity.get() {
            self.replay.pop_front();
        }
        self.replay.push_back(experience);
        self.replay_insertions += 1;

        if self.replay.len() >= self.config.replay_warmup.get()
            && self
                .replay_insertions
                .is_multiple_of(self.config.train_every.get())
        {
            self.optimize(replay_depth)?;
        }
        Ok(())
    }

    fn optimize(&mut self, replay_depth: Option<usize>) -> Result<()> {
        let size = self.config.batch_size.get();
        let batch: Vec<_> = if let Some(depth) = replay_depth {
            let eligible: Vec<_> = self
                .replay
                .iter()
                .filter(|experience| experience.is_within_depth(depth))
                .collect();
            if eligible.len() < size {
                return Ok(());
            }
            index::sample(&mut self.sampling, eligible.len(), size)
                .iter()
                .map(|index| eligible[index])
                .collect()
        } else {
            index::sample(&mut self.sampling, self.replay.len(), size)
                .iter()
                .map(|index| &self.replay[index])
                .collect()
        };
        let mut legal = Vec::with_capacity(size * ACTIONS);
        let mut terminals = Vec::with_capacity(size);
        let mut rewards = Vec::with_capacity(size);
        let mut actions = Vec::with_capacity(size);
        for experience in &batch {
            let next = experience.next.as_ref().unwrap_or(&experience.state);
            legal.extend_from_slice(&next.legal);
            terminals.push(experience.next.is_none());
            rewards.push(experience.reward);
            actions.push(experience.action as i32);
        }

        let next_scores = self.target.forward(
            batch.iter().map(|experience| {
                &experience
                    .next
                    .as_ref()
                    .unwrap_or(&experience.state)
                    .observation
            }),
            self.device,
        );
        let legal = Tensor::from_data(TensorData::new(legal, [size, ACTIONS]), self.device);
        let terminals = Tensor::from_data(TensorData::new(terminals, [size, 1]), self.device);
        let rewards = Tensor::from_data(TensorData::new(rewards, [size, 1]), self.device);
        let targets = bellman_targets(next_scores, legal, terminals, rewards, self.config.gamma);

        let actions =
            Tensor::<2, Int>::from_data(TensorData::new(actions, [size, 1]), &self.training_device);
        let values = self
            .online
            .forward(
                batch.iter().map(|experience| &experience.state.observation),
                &self.training_device,
            )
            .gather(1, actions);
        let loss = HuberLossConfig::new(1.0)
            .init()
            .forward(values, targets, Reduction::Mean);
        if !loss.clone().into_scalar::<f32>().is_finite() {
            return Err(
                format!("DQN loss became non-finite after {} updates", self.updates).into(),
            );
        }

        let gradients = GradientsParams::from_grads(loss.backward(), &self.online);
        self.online =
            self.optimizer
                .step(self.config.learning_rate, self.online.clone(), gradients);
        self.updates += 1;
        if self
            .updates
            .is_multiple_of(self.config.target_update_every.get())
        {
            self.target = self.online.valid();
        }
        Ok(())
    }
}

fn bellman_targets(
    next_scores: Tensor<2>,
    legal: Tensor<2, Bool>,
    terminals: Tensor<2, Bool>,
    rewards: Tensor<2>,
    gamma: f32,
) -> Tensor<2> {
    let next = next_scores
        .mask_fill(legal.bool_not(), f32::NEG_INFINITY)
        .max_dim(1)
        .mask_fill(terminals, 0.0);
    rewards + next * gamma
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_curriculum_widens_existing_tails_then_uses_all_decisions() {
        let config: Config =
            serde_json::from_str(include_str!("../../../../experiments/dqn-curriculum.json"))
                .unwrap();
        let device = Device::flex();
        let mut trainer = Trainer::new(&config, &device);

        let first = trainer.play_episode(200).unwrap();
        assert!(first.reward.is_some());
        assert_eq!(first.replay_depth, Some(8));
        assert_eq!(first.replay_eligible, first.decisions.min(8));
        assert!(
            trainer
                .replay
                .iter()
                .rev()
                .enumerate()
                .all(|(index, experience)| experience.decisions_to_end == Some(index + 1))
        );

        let second = trainer.play_episode(201).unwrap();
        assert!(second.reward.is_some());
        assert_eq!(second.replay_depth, Some(16));
        assert_eq!(
            second.replay_eligible,
            first.decisions.min(16) + (second.decisions - first.decisions).min(16)
        );

        let full = trainer.play_episode(801).unwrap();
        assert_eq!(full.replay_depth, None);
        assert_eq!(full.replay_eligible, trainer.replay.len());
    }

    #[test]
    fn target_uses_only_legal_actions_and_does_not_bootstrap_at_game_end() {
        let device = Device::flex();
        let scores = Tensor::from_data([[20.0f32, 2.0, 1.0], [20.0, 2.0, 1.0]], &device);
        let legal = Tensor::from_data([[false, true, true], [false, false, false]], &device);
        let terminals = Tensor::from_data([[false], [true]], &device);
        let rewards = Tensor::from_data([[0.0f32], [-1.0]], &device);

        let result = bellman_targets(scores, legal, terminals, rewards, 0.5).into_data();
        assert_eq!(result.as_slice::<f32>().unwrap(), &[1.0, -1.0]);
    }
}
