use std::{collections::VecDeque, rc::Rc};

use burn::{
    module::Module,
    nn::loss::{HuberLossConfig, Reduction},
    optim::{AdamConfig, GradientsParams, ModuleOptimizer},
    tensor::{Bool, Device, Int, Tensor, TensorData},
};
use rand::{Rng, SeedableRng, seq::index};
use rand_chacha::ChaCha8Rng;
use sequence_agents::{HeuristicAgent, Policy};
use sequence_game::{Action, GameConfig, Outcome, Player};
use sequence_match::{Match, MatchConfig, MatchEnd};
use serde::Serialize;

use crate::{Result, experiment::GameEnd};

use super::{
    Config, Target,
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
}

#[derive(Serialize)]
pub struct Episode {
    episode: usize,
    shuffle_seed: u64,
    learner_seat: usize,
    opponent_episode: Option<usize>,
    outcome: GameEnd,
    actions: usize,
    decisions: usize,
    updates: usize,
    epsilon: f64,
    reward: Option<f32>,
    opponent: &'static str,
    opponent_seed: Option<u64>,
}

pub struct Learner {
    pub online: Network,
    target: Network,
    optimizer: ModuleOptimizer,
    pub updates: usize,
}

pub struct Trainer<'a> {
    config: &'a Config,
    device: &'a Device,
    training_device: Device,
    pub learner: Learner,
    opponent: Network,
    replay: VecDeque<Experience>,
    exploration: ChaCha8Rng,
    sampling: ChaCha8Rng,
    opponent_rng: ChaCha8Rng,
    heuristic_slot: usize,
    opponent_episode: usize,
    pub decisions: usize,
    replay_insertions: usize,
}

impl<'a> Trainer<'a> {
    pub fn new(config: &'a Config, device: &'a Device) -> Self {
        device.seed(config.seed);
        let training_device = device.clone().autodiff();
        let online = Network::new(&training_device);
        let mut sampling = ChaCha8Rng::seed_from_u64(config.seed);
        sampling.set_stream(1);
        let mut opponent_rng = ChaCha8Rng::seed_from_u64(config.seed);
        opponent_rng.set_stream(2);

        Self {
            config,
            device,
            training_device,
            opponent: online.valid(),
            learner: Learner {
                target: online.valid(),
                online,
                optimizer: AdamConfig::new().init(),
                updates: 0,
            },
            replay: VecDeque::with_capacity(config.replay_capacity.get()),
            exploration: ChaCha8Rng::seed_from_u64(config.seed),
            sampling,
            opponent_rng,
            heuristic_slot: 0,
            opponent_episode: 0,
            decisions: 0,
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
        let opponent_seed = self.config.heuristic_opponent_every.and_then(|interval| {
            let slot = (episode - 1) % interval.get();
            if slot == 0 {
                self.heuristic_slot = self.opponent_rng.random_range(0..interval.get());
            }
            (slot == self.heuristic_slot).then(|| self.opponent_rng.random())
        });
        let mut heuristic = opponent_seed.map(HeuristicAgent::new);
        let (opponent, opponent_episode, opponent_id) = if heuristic.is_some() {
            ("heuristic", None, "heuristic".into())
        } else {
            (
                "self_play",
                Some(self.opponent_episode),
                format!("dqn@{}", self.opponent_episode),
            )
        };
        let mut policies = ["dqn".into(), opponent_id];
        let mut policy_seeds = [None, opponent_seed];
        if learner == Player::Two {
            policies.swap(0, 1);
            policy_seeds.swap(0, 1);
        }
        let mut game = Match::new(
            MatchConfig {
                game: GameConfig {
                    shuffle_seed,
                    starting_player: Player::One,
                    policies,
                },
                action_limit: self.config.action_limit,
                policy_seeds,
            },
            false,
        );

        let mut pending = None;
        while let Some(view) = game.active_view() {
            let actor = view.player;
            let action = if view.legal_actions == [Action::Pass] {
                Action::Pass
            } else if actor != learner {
                match &mut heuristic {
                    Some(policy) => policy
                        .choose_action(&view)
                        .ok_or("heuristic did not choose an action in an active match")?,
                    None => {
                        self.opponent
                            .greedy_action(&view, &Observation::new(&view), self.device)
                    }
                }
            } else {
                let decision = Rc::new(Decision {
                    observation: Observation::new(&view),
                    legal: legal_mask(&view),
                });
                if let Some((state, action)) = pending.take() {
                    self.record(Experience {
                        state,
                        action,
                        reward: 0.0,
                        next: Some(Rc::clone(&decision)),
                    })?;
                }

                let action = if self.exploration.random::<f64>() < self.epsilon() {
                    view.legal_actions[self.exploration.random_range(0..view.legal_actions.len())]
                } else {
                    self.learner.online.valid().greedy_action(
                        &view,
                        &decision.observation,
                        self.device,
                    )
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
            self.record(Experience {
                state,
                action,
                reward,
                next: None,
            })?;
        }

        let row = Episode {
            episode,
            shuffle_seed,
            learner_seat: learner.index() + 1,
            opponent_episode,
            outcome: end.into(),
            actions: game.action_count(),
            decisions: self.decisions,
            updates: self.learner.updates,
            epsilon: self.epsilon(),
            reward,
            opponent,
            opponent_seed,
        };
        if episode.is_multiple_of(self.config.opponent_update_every.get()) {
            self.opponent = self.learner.online.valid();
            self.opponent_episode = episode;
        }
        Ok(row)
    }

    fn epsilon(&self) -> f64 {
        let duration = self.config.epsilon_decay_decisions.get();
        let progress = self.decisions.min(duration) as f64 / duration as f64;
        self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)
    }

    fn record(&mut self, experience: Experience) -> Result<()> {
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
            self.optimize()?;
        }
        Ok(())
    }

    fn optimize(&mut self) -> Result<()> {
        let batch: Vec<_> = index::sample(
            &mut self.sampling,
            self.replay.len(),
            self.config.batch_size.get(),
        )
        .iter()
        .map(|index| &self.replay[index])
        .collect();
        self.learner
            .update_batch(self.config, self.device, &self.training_device, &batch)
    }
}

impl Learner {
    fn update_batch(
        &mut self,
        config: &Config,
        device: &Device,
        training_device: &Device,
        batch: &[&Experience],
    ) -> Result<()> {
        let size = batch.len();
        let mut legal = Vec::with_capacity(size * ACTIONS);
        let mut terminals = Vec::with_capacity(size);
        let mut rewards = Vec::with_capacity(size);
        let mut actions = Vec::with_capacity(size);
        for experience in batch {
            let next = experience.next.as_ref().unwrap_or(&experience.state);
            legal.extend_from_slice(&next.legal);
            terminals.push(experience.next.is_none());
            rewards.push(experience.reward);
            actions.push(experience.action as i32);
        }

        let next_states = || {
            batch.iter().map(|experience| {
                &experience
                    .next
                    .as_ref()
                    .unwrap_or(&experience.state)
                    .observation
            })
        };
        let next_scores = self.target.forward(next_states(), device);
        let legal = Tensor::from_data(TensorData::new(legal, [size, ACTIONS]), device);
        let terminals = Tensor::from_data(TensorData::new(terminals, [size, 1]), device);
        let rewards = Tensor::from_data(TensorData::new(rewards, [size, 1]), device);
        let selector = match config.target {
            Target::Dqn => None,
            Target::DoubleDqn => Some(self.online.valid().forward(next_states(), device)),
        };
        let targets = bellman_targets(
            next_scores,
            selector,
            legal,
            terminals,
            rewards,
            config.gamma,
        );

        let actions =
            Tensor::<2, Int>::from_data(TensorData::new(actions, [size, 1]), training_device);
        let values = self
            .online
            .forward(
                batch.iter().map(|experience| &experience.state.observation),
                training_device,
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
        self.online = self
            .optimizer
            .step(config.learning_rate, self.online.clone(), gradients);
        self.updates += 1;
        if self
            .updates
            .is_multiple_of(config.target_update_every.get())
        {
            self.target = self.online.valid();
        }
        Ok(())
    }
}

fn bellman_targets(
    next_scores: Tensor<2>,
    selector: Option<Tensor<2>>,
    legal: Tensor<2, Bool>,
    terminals: Tensor<2, Bool>,
    rewards: Tensor<2>,
    gamma: f32,
) -> Tensor<2> {
    let illegal = legal.bool_not();
    let next = match selector {
        Some(online) => {
            let selected = online.mask_fill(illegal, f32::NEG_INFINITY).argmax(1);
            next_scores.gather(1, selected)
        }
        None => next_scores.mask_fill(illegal, f32::NEG_INFINITY).max_dim(1),
    };
    rewards + next.mask_fill(terminals, 0.0) * gamma
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_opponents_schedule_one_seeded_heuristic_per_block() {
        let mut config: Config = serde_json::from_str(include_str!(
            "../../../../experiments/double-dqn-mixed-full.json"
        ))
        .unwrap();
        config.validate().unwrap();
        config.action_limit = 8.try_into().unwrap();
        let device = Device::flex();
        let mut trainer = Trainer::new(&config, &device);

        for block in 0..2 {
            let mut heuristic_games = 0;
            for episode in (block * 5 + 1)..=(block * 5 + 5) {
                let row = trainer.play_episode(episode).unwrap();
                if row.opponent == "heuristic" {
                    heuristic_games += 1;
                    assert!(row.opponent_seed.is_some());
                    assert!(row.opponent_episode.is_none());
                } else {
                    assert_eq!(row.opponent, "self_play");
                    assert_eq!(row.opponent_episode, Some(0));
                    assert!(row.opponent_seed.is_none());
                }
            }
            assert_eq!(heuristic_games, 1);
        }
    }

    #[test]
    fn target_uses_only_legal_actions_and_does_not_bootstrap_at_game_end() {
        let device = Device::flex();
        let scores = Tensor::from_data([[20.0f32, 2.0, 1.0], [20.0, 2.0, 1.0]], &device);
        let legal = Tensor::from_data([[false, true, true], [false, false, false]], &device);
        let terminals = Tensor::from_data([[false], [true]], &device);
        let rewards = Tensor::from_data([[0.0f32], [-1.0]], &device);

        let selector = Tensor::from_data([[99.0f32, 3.0, 4.0], [99.0, 3.0, 4.0]], &device);
        let double = bellman_targets(
            scores.clone(),
            Some(selector),
            legal.clone(),
            terminals.clone(),
            rewards.clone(),
            0.5,
        )
        .into_data();
        assert_eq!(double.as_slice::<f32>().unwrap(), &[0.5, -1.0]);

        let result = bellman_targets(scores, None, legal, terminals, rewards, 0.5).into_data();
        assert_eq!(result.as_slice::<f32>().unwrap(), &[1.0, -1.0]);
    }
}
