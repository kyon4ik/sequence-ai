use std::{
    collections::VecDeque,
    io::{Read, Write},
    rc::Rc,
};

use burn::{
    module::Module,
    nn::loss::{HuberLossConfig, Reduction},
    optim::{AdamConfig, GradientsParams, ModuleOptimizer},
    store::ModuleRecord,
    tensor::{Bool, Bytes, Device, Int, Tensor, TensorData, activation::relu},
};
use rand::{Rng, SeedableRng, seq::index};
use rand_chacha::ChaCha8Rng;
use sequence_agents::{HeuristicAgent, Policy};
use sequence_game::{Action, GameConfig, Outcome, Player, PlayerView};
use sequence_match::{Match, MatchConfig, MatchEnd};
use serde::{Deserialize, Serialize};

use crate::{Result, experiment::GameEnd};

use super::{
    Config, Exploration, Target,
    model::Network,
    observation::{ACTIONS, Observation, action_index, legal_mask},
};

#[derive(Serialize, Deserialize)]
struct Decision {
    observation: Observation,
    #[serde(with = "super::snapshot::array")]
    legal: [bool; ACTIONS],
    winning_actions: Vec<usize>,
}

impl Decision {
    fn new(view: &PlayerView<'_>, observation: Observation, label_wins: bool) -> Self {
        let mut winning_actions = Vec::new();
        if label_wins {
            winning_actions.extend(view.legal_actions.iter().filter_map(|&action| {
                view.preview_play(action)
                    .filter(|preview| preview.wins)
                    .and_then(|_| action_index(action))
            }));
        }

        Self {
            observation,
            legal: legal_mask(view),
            winning_actions,
        }
    }
}

#[derive(Serialize, Deserialize)]
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
    epsilon: Option<f64>,
    reward: Option<f32>,
    opponent: &'static str,
    opponent_seed: Option<u64>,
    temperature: Option<f64>,
    immediate_win_opportunities: Option<usize>,
    immediate_wins_taken: Option<usize>,
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
    inference_device: &'a Device,
    training_device: Device,
    exploration_schedule: Exploration,
    pub learner: Learner,
    learner_policy: Option<Network>,
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

#[derive(Serialize, Deserialize)]
struct SavedTrainer<R> {
    online: Bytes,
    target: Bytes,
    opponent: Bytes,
    optimizer: Bytes,
    replay: R,
    exploration: ChaCha8Rng,
    sampling: ChaCha8Rng,
    opponent_rng: ChaCha8Rng,
    heuristic_slot: usize,
    opponent_episode: usize,
    decisions: usize,
    replay_insertions: usize,
    updates: usize,
}

impl<'a> Trainer<'a> {
    pub fn new(
        config: &'a Config,
        exploration_schedule: Exploration,
        device: &'a Device,
        inference_device: &'a Device,
    ) -> Self {
        device.seed(config.seed);
        let training_device = device.clone().autodiff();
        let online = Network::new(&training_device, config.move_history);
        let mut sampling = ChaCha8Rng::seed_from_u64(config.seed);
        sampling.set_stream(1);
        let mut opponent_rng = ChaCha8Rng::seed_from_u64(config.seed);
        opponent_rng.set_stream(2);

        Self {
            config,
            device,
            inference_device,
            training_device,
            exploration_schedule,
            opponent: online.valid().to_device(inference_device),
            learner_policy: None,
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

    pub fn save_state(&self, writer: impl Write, online: &Bytes) -> Result<()> {
        let state = SavedTrainer {
            online: online.clone(),
            target: self.learner.target.clone().into_record().into_bytes()?,
            opponent: self.opponent.clone().into_record().into_bytes()?,
            optimizer: self.learner.optimizer.into_bytes()?,
            replay: &self.replay,
            exploration: self.exploration.clone(),
            sampling: self.sampling.clone(),
            opponent_rng: self.opponent_rng.clone(),
            heuristic_slot: self.heuristic_slot,
            opponent_episode: self.opponent_episode,
            decisions: self.decisions,
            replay_insertions: self.replay_insertions,
            updates: self.learner.updates,
        };
        ciborium::into_writer(&state, writer)?;
        Ok(())
    }

    pub fn restore_state(&mut self, reader: impl Read) -> Result<Bytes> {
        let state: SavedTrainer<VecDeque<Experience>> = ciborium::from_reader(reader)?;
        if state.replay.len()
            != state
                .replay_insertions
                .min(self.config.replay_capacity.get())
        {
            return Err("checkpoint replay length disagrees with the insertion count".into());
        }

        let online = state.online.shared();
        self.learner.online = self
            .learner
            .online
            .clone()
            .try_load_record(ModuleRecord::from_bytes(online.clone())?)?;
        self.learner.target = self
            .learner
            .target
            .clone()
            .try_load_record(ModuleRecord::from_bytes(state.target)?)?;
        self.opponent = self
            .opponent
            .clone()
            .try_load_record(ModuleRecord::from_bytes(state.opponent)?)?;
        self.learner.optimizer = AdamConfig::new().init().from_bytes(state.optimizer)?;

        self.learner_policy = None;
        self.replay = state.replay;
        self.exploration = state.exploration;
        self.sampling = state.sampling;
        self.opponent_rng = state.opponent_rng;
        self.heuristic_slot = state.heuristic_slot;
        self.opponent_episode = state.opponent_episode;
        self.decisions = state.decisions;
        self.replay_insertions = state.replay_insertions;
        self.learner.updates = state.updates;
        Ok(online)
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

        let label_wins = self.config.immediate_win_loss_weight.unwrap_or_default() > 0.0;
        let mut immediate_win_opportunities = 0;
        let mut immediate_wins_taken = 0;
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
                    None => self.opponent.greedy_action(
                        &view,
                        &self.opponent.observe(&view),
                        self.inference_device,
                    ),
                }
            } else {
                let observation = self.learner.online.observe(&view);
                let decision = Rc::new(Decision::new(&view, observation, label_wins));
                if let Some((state, action)) = pending.take() {
                    self.record(Experience {
                        state,
                        action,
                        reward: 0.0,
                        next: Some(Rc::clone(&decision)),
                    })?;
                }

                let action = self.learner_action(&view, &decision.observation)?;
                let index = action_index(action).expect("scored action");
                immediate_win_opportunities += usize::from(!decision.winning_actions.is_empty());
                immediate_wins_taken += usize::from(decision.winning_actions.contains(&index));
                pending = Some((decision, index));
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

        let (epsilon, temperature) = match self.exploration_schedule {
            Exploration::Epsilon(schedule) => (Some(schedule.at(self.decisions)), None),
            Exploration::Temperature(schedule) => {
                (None, Some(schedule.temperature_at(self.decisions)))
            }
        };
        let row = Episode {
            episode,
            shuffle_seed,
            learner_seat: learner.index() + 1,
            opponent_episode,
            outcome: end.into(),
            actions: game.action_count(),
            decisions: self.decisions,
            updates: self.learner.updates,
            epsilon,
            reward,
            opponent,
            opponent_seed,
            temperature,
            immediate_win_opportunities: label_wins.then_some(immediate_win_opportunities),
            immediate_wins_taken: label_wins.then_some(immediate_wins_taken),
        };
        if episode.is_multiple_of(self.config.opponent_update_every.get()) {
            self.opponent = self.learner.online.valid().to_device(self.inference_device);
            self.opponent_episode = episode;
        }
        Ok(row)
    }

    fn learner_action(
        &mut self,
        view: &PlayerView<'_>,
        observation: &Observation,
    ) -> Result<Action> {
        let temperature = match self.exploration_schedule {
            Exploration::Epsilon(schedule) => {
                if self.exploration.random::<f64>() < schedule.at(self.decisions) {
                    return Ok(view.legal_actions
                        [self.exploration.random_range(0..view.legal_actions.len())]);
                }
                None
            }
            Exploration::Temperature(schedule) => {
                if let [action] = view.legal_actions.as_slice() {
                    return Ok(*action);
                }
                Some(schedule.temperature_at(self.decisions))
            }
        };

        let online;
        let policy = if self.device == self.inference_device {
            online = self.learner.online.valid();
            &online
        } else {
            // Copy the updated weights only when the learner next needs network scores.
            self.learner_policy
                .get_or_insert_with(|| self.learner.online.valid().to_device(self.inference_device))
        };
        match temperature {
            Some(temperature) => policy.sample_action(
                view,
                observation,
                self.inference_device,
                temperature,
                &mut self.exploration,
            ),
            None => Ok(policy.greedy_action(view, observation, self.inference_device)),
        }
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
            .update_batch(self.config, self.device, &self.training_device, &batch)?;
        self.learner_policy = None;
        Ok(())
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
        let scores = self.online.forward(
            batch.iter().map(|experience| &experience.state.observation),
            training_device,
        );
        let auxiliary = config
            .immediate_win_loss_weight
            .filter(|&weight| weight > 0.0)
            .and_then(|weight| {
                immediate_win_loss(
                    scores.clone(),
                    batch.iter().map(|experience| experience.state.as_ref()),
                    training_device,
                )
                .map(|loss| loss * weight)
            });
        let values = scores.gather(1, actions);
        let mut loss = HuberLossConfig::new(1.0)
            .init()
            .forward(values, targets, Reduction::Mean);
        if let Some(auxiliary) = auxiliary {
            loss = loss + auxiliary;
        }
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

fn immediate_win_loss<'a>(
    scores: Tensor<2>,
    decisions: impl Iterator<Item = &'a Decision>,
    device: &Device,
) -> Option<Tensor<1>> {
    const MARGIN: f32 = 0.1;

    let mut rows = Vec::new();
    let mut winning = Vec::new();
    let mut alternatives = Vec::new();
    for (row, state) in decisions.enumerate() {
        if state.winning_actions.is_empty() {
            continue;
        }
        let mut wins = [false; ACTIONS];
        for &action in &state.winning_actions {
            wins[action] = true;
        }
        let other = state
            .legal
            .iter()
            .zip(&wins)
            .map(|(&legal, &wins)| legal && !wins);
        if !other.clone().any(|alternative| alternative) {
            continue;
        }
        rows.push(row as i32);
        winning.extend_from_slice(&wins);
        alternatives.extend(other);
    }
    if rows.is_empty() {
        return None;
    }

    let count = rows.len();
    let rows = Tensor::<1, Int>::from_data(TensorData::new(rows, [count]), device);
    let scores = scores.select(0, rows);
    let winning = Tensor::<2, Bool>::from_data(TensorData::new(winning, [count, ACTIONS]), device);
    let alternatives =
        Tensor::<2, Bool>::from_data(TensorData::new(alternatives, [count, ACTIONS]), device);
    let best_win = scores
        .clone()
        .mask_fill(winning.bool_not(), f32::NEG_INFINITY)
        .max_dim(1);
    let best_alternative = scores
        .mask_fill(alternatives.bool_not(), f32::NEG_INFINITY)
        .max_dim(1);

    Some(relu(best_alternative - best_win + MARGIN).mean())
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
    use sequence_game::Game;

    use super::*;

    #[test]
    fn mixed_opponents_schedule_one_seeded_heuristic_per_block() {
        let mut config: Config = serde_json::from_str(include_str!(
            "../../../../experiments/double-dqn-mixed-full.json"
        ))
        .unwrap();
        let exploration = config.validate().unwrap();
        config.action_limit = 8.try_into().unwrap();
        let device = Device::flex();
        let mut trainer = Trainer::new(&config, exploration, &device, &device);

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

    #[test]
    fn immediate_win_loss_ranks_the_best_win_against_legal_alternatives() {
        let game = Game::new(GameConfig {
            shuffle_seed: 0,
            starting_player: Player::One,
            policies: Default::default(),
        });
        let decision = |winning_actions: Vec<usize>| {
            let view = game.view(Player::One);
            let mut state = Decision::new(&view, Observation::new(&view, false), false);
            state.legal = std::array::from_fn(|i| i < 3);
            state.winning_actions = winning_actions;
            state
        };
        let states = [decision(vec![0, 1]), decision(vec![]), decision(vec![2])];
        let mut values = [[100.0f32; ACTIONS]; 3];
        values[0][..3].copy_from_slice(&[-0.3, 0.1, 0.4]);
        values[2][..3].copy_from_slice(&[0.1, 0.3, 0.8]);

        let device = Device::flex();
        let scores = Tensor::from_data(values, &device);
        let loss = immediate_win_loss(scores, states.iter(), &device)
            .unwrap()
            .into_scalar::<f32>();
        assert!((loss - 0.2).abs() < 1e-6);
    }
}
