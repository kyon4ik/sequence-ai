use std::{
    cell::RefCell,
    fmt, fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Instant,
};

use burn::tensor::Device;
use sequence_agents::{HeuristicAgent, Policy, RandomAgent};
use sequence_game::{Action, GameConfig, Outcome, Player, PlayerView};
use sequence_match::{
    EvaluationConfig, EvaluationReport, Match, MatchConfig, MatchEnd, Replay, evaluate,
};
use serde::{Deserialize, Serialize};

use crate::{
    Result,
    dqn::{
        model::{GreedyPolicy, Network},
        search::{SearchPolicy, SearchSettings},
    },
};

const STARTING_PLAYER: Player = Player::One;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentConfig {
    pub policies: [PolicyKind; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    checkpoint: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    search: Option<SearchSettings>,
    first_shuffle_seed: u64,
    deal_count: NonZeroUsize,
    policy_seed: u64,
    action_limit: NonZeroUsize,
}

impl ExperimentConfig {
    pub fn save_checkpoint(
        &mut self,
        config_directory: &Path,
        run_directory: &Path,
    ) -> Result<bool> {
        self.validate_checkpoint()?;
        let Some(path) = &self.checkpoint else {
            return Ok(false);
        };
        let source = config_directory.join(path);
        let name = path.file_name().ok_or("expected a checkpoint filename")?;
        let relative = Path::new("checkpoints").join(name);
        fs::create_dir(run_directory.join("checkpoints"))?;
        fs::copy(&source, run_directory.join(&relative))
            .map_err(|error| format!("could not copy checkpoint {}: {error}", source.display()))?;
        self.checkpoint = Some(relative);
        Ok(true)
    }

    fn validate_checkpoint(&self) -> Result<()> {
        if self.policies.contains(&PolicyKind::Ismcts) != self.search.is_some() {
            return Err("search settings are required exactly when using an ismcts policy".into());
        }

        match (
            self.policies.contains(&PolicyKind::Dqn) || self.policies.contains(&PolicyKind::Ismcts),
            self.checkpoint.is_some(),
        ) {
            (true, false) => Err("a dqn or ismcts policy requires a checkpoint".into()),
            (false, true) => Err("a checkpoint requires a dqn or ismcts policy".into()),
            _ => Ok(()),
        }
    }

    pub fn load_checkpoint(&self, directory: &Path) -> Result<Option<CheckpointPolicy>> {
        self.validate_checkpoint()?;
        self.checkpoint
            .as_ref()
            .map(|path| CheckpointPolicy::load(&directory.join(path)))
            .transpose()
    }

    pub fn evaluate(
        &self,
        checkpoint: Option<&CheckpointPolicy>,
    ) -> Result<(EvaluationReport, Option<[MoveTiming; 2]>)> {
        let [first, second] = self.policies;
        let first = first.factory(checkpoint, self.search)?;
        let second = second.factory(checkpoint, self.search)?;
        let last_seed = self
            .first_shuffle_seed
            .checked_add(u64::try_from(self.deal_count.get() - 1)?)
            .ok_or("shuffle seed range exceeds u64::MAX")?;
        let config = EvaluationConfig {
            policy_ids: self.policies.map(|policy| policy.to_string()),
            shuffle_seeds: (self.first_shuffle_seed..=last_seed).collect(),
            starting_player: STARTING_PLAYER,
            policy_seed: self.policy_seed,
            action_limit: self.action_limit,
            capture_replays: false,
        };
        let timings = self
            .search
            .map(|_| [RefCell::default(), RefCell::default()]);
        let report = evaluate(
            &config,
            |seed| MeasuredAgent {
                agent: first.build(seed),
                times: timings.as_ref().map(|times| &times[0]),
            },
            |seed| MeasuredAgent {
                agent: second.build(seed),
                times: timings.as_ref().map(|times| &times[1]),
            },
        )?;
        let timings = timings.map(|timings| {
            let [first, second] = timings.map(RefCell::into_inner);
            [
                MoveTiming::new(&self.policies[0], first),
                MoveTiming::new(&self.policies[1], second),
            ]
        });

        Ok((report, timings))
    }
}

struct MeasuredAgent<'a> {
    agent: Agent<'a>,
    times: Option<&'a RefCell<Vec<f64>>>,
}

impl Policy for MeasuredAgent<'_> {
    fn choose_action(&mut self, view: &PlayerView<'_>) -> Option<Action> {
        let Some(times) = self.times else {
            return self.agent.choose_action(view);
        };
        let start = Instant::now();
        let action = self.agent.choose_action(view);
        times
            .borrow_mut()
            .push(start.elapsed().as_secs_f64() * 1000.0);
        action
    }
}

#[derive(Serialize)]
pub struct MoveTiming {
    policy: String,
    decisions: usize,
    median_ms: Option<f64>,
    p95_ms: Option<f64>,
}

impl MoveTiming {
    fn new(policy: &PolicyKind, mut times: Vec<f64>) -> Self {
        times.sort_unstable_by(f64::total_cmp);
        let percentile = |percent: usize| {
            (times.len() * percent)
                .div_ceil(100)
                .checked_sub(1)
                .map(|index| times[index])
        };
        Self {
            policy: policy.to_string(),
            decisions: times.len(),
            median_ms: percentile(50),
            p95_ms: percentile(95),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyKind {
    Random,
    Heuristic,
    Dqn,
    Ismcts,
}

impl fmt::Display for PolicyKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Random => AgentKind::Random.fmt(f),
            Self::Heuristic => AgentKind::Heuristic.fmt(f),
            Self::Dqn => f.write_str("dqn"),
            Self::Ismcts => f.write_str("ismcts"),
        }
    }
}

pub struct CheckpointPolicy {
    network: Network,
    device: Device,
}

impl CheckpointPolicy {
    fn load(path: &Path) -> Result<Self> {
        let device = Device::flex();
        let network = Network::load(path, &device)
            .map_err(|error| format!("could not load checkpoint {}: {error}", path.display()))?;
        Ok(Self { network, device })
    }
}

enum PolicyFactory<'a> {
    Baseline(AgentKind),
    Dqn(&'a CheckpointPolicy),
    Ismcts(&'a CheckpointPolicy, SearchSettings),
}

impl PolicyKind {
    fn factory(
        self,
        checkpoint: Option<&CheckpointPolicy>,
        search: Option<SearchSettings>,
    ) -> Result<PolicyFactory<'_>> {
        match self {
            Self::Random => Ok(PolicyFactory::Baseline(AgentKind::Random)),
            Self::Heuristic => Ok(PolicyFactory::Baseline(AgentKind::Heuristic)),
            Self::Dqn => Ok(PolicyFactory::Dqn(
                checkpoint.ok_or("missing DQN checkpoint")?,
            )),
            Self::Ismcts => Ok(PolicyFactory::Ismcts(
                checkpoint.ok_or("missing ISMCTS checkpoint")?,
                search.ok_or("missing ISMCTS search settings")?,
            )),
        }
    }
}

impl<'a> PolicyFactory<'a> {
    fn build(&self, seed: u64) -> Agent<'a> {
        match self {
            Self::Baseline(kind) => kind.build(seed),
            Self::Dqn(policy) => Agent::Dqn(GreedyPolicy {
                network: &policy.network,
                device: &policy.device,
            }),
            Self::Ismcts(policy, settings) => Agent::Ismcts(SearchPolicy::new(
                &policy.network,
                &policy.device,
                *settings,
                seed,
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentKind {
    Random,
    Heuristic,
}

impl fmt::Display for AgentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Random => "random",
            Self::Heuristic => "heuristic",
        })
    }
}

pub(crate) enum Agent<'a> {
    Random(RandomAgent),
    Heuristic(HeuristicAgent),
    Dqn(GreedyPolicy<'a>),
    Ismcts(SearchPolicy<'a>),
}

impl AgentKind {
    pub(crate) fn build(self, seed: u64) -> Agent<'static> {
        match self {
            Self::Random => Agent::Random(RandomAgent::new(seed)),
            Self::Heuristic => Agent::Heuristic(HeuristicAgent::new(seed)),
        }
    }
}

impl Policy for Agent<'_> {
    fn choose_action(&mut self, view: &PlayerView<'_>) -> Option<Action> {
        match self {
            Self::Random(agent) => agent.choose_action(view),
            Self::Heuristic(agent) => agent.choose_action(view),
            Self::Dqn(agent) => agent.choose_action(view),
            Self::Ismcts(agent) => agent.choose_action(view),
        }
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Seat {
    One,
    Two,
}

impl From<Seat> for Player {
    fn from(seat: Seat) -> Self {
        match seat {
            Seat::One => Player::One,
            Seat::Two => Player::Two,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GameEnd {
    PlayerOneWin,
    PlayerTwoWin,
    Draw,
    ActionLimit,
}

impl From<MatchEnd> for GameEnd {
    fn from(end: MatchEnd) -> Self {
        match end {
            MatchEnd::Completed(Outcome::Win(Player::One)) => Self::PlayerOneWin,
            MatchEnd::Completed(Outcome::Win(Player::Two)) => Self::PlayerTwoWin,
            MatchEnd::Completed(Outcome::Draw) => Self::Draw,
            MatchEnd::ActionLimit => Self::ActionLimit,
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct GameRow {
    pub game_id: usize,
    pub pair_id: usize,
    pub shuffle_seed: u64,
    pub policy_a_seat: Seat,
    player_one_policy: PolicyKind,
    player_two_policy: PolicyKind,
    player_one_seed: u64,
    player_two_seed: u64,
    pub outcome: GameEnd,
    pub actions: usize,
}

pub fn write_games(
    path: &Path,
    config: &ExperimentConfig,
    report: &EvaluationReport,
) -> Result<()> {
    let [policy_a, policy_b] = config.policies;
    let mut writer = csv::Writer::from_path(path)?;
    for (index, pair) in report.pairs.iter().enumerate() {
        for (swap, game) in pair.iter().enumerate() {
            let (policy_a_seat, player_one_policy, player_two_policy) = if swap == 0 {
                (Seat::One, policy_a, policy_b)
            } else {
                (Seat::Two, policy_b, policy_a)
            };
            let [player_one_seed, player_two_seed] = game
                .config
                .policy_seeds
                .map(|seed| seed.expect("the evaluator seeds both policies"));
            writer.serialize(GameRow {
                game_id: index * 2 + swap + 1,
                pair_id: index + 1,
                shuffle_seed: game.config.game.shuffle_seed,
                policy_a_seat,
                player_one_policy,
                player_two_policy,
                player_one_seed,
                player_two_seed,
                outcome: game.end.into(),
                actions: game.action_count,
            })?;
        }
    }
    writer.flush()?;
    Ok(())
}

impl GameRow {
    pub fn reconstruct(
        &self,
        config: &ExperimentConfig,
        checkpoint: Option<&CheckpointPolicy>,
    ) -> Result<Replay> {
        let mut player_one = self
            .player_one_policy
            .factory(checkpoint, config.search)?
            .build(self.player_one_seed);
        let mut player_two = self
            .player_two_policy
            .factory(checkpoint, config.search)?
            .build(self.player_two_seed);
        let mut game = Match::new(
            MatchConfig {
                game: GameConfig {
                    shuffle_seed: self.shuffle_seed,
                    starting_player: STARTING_PLAYER,
                    policies: [
                        self.player_one_policy.to_string(),
                        self.player_two_policy.to_string(),
                    ],
                },
                policy_seeds: [Some(self.player_one_seed), Some(self.player_two_seed)],
                action_limit: config.action_limit,
            },
            true,
        );
        let end = game.run([&mut player_one, &mut player_two], |_, _| {})?;
        if GameEnd::from(end) != self.outcome || game.action_count() != self.actions {
            return Err(format!(
                "reconstructed game {} differs from games.csv. Use the original checkout.",
                self.game_id
            )
            .into());
        }

        Ok(game
            .into_report()
            .and_then(|report| report.replay)
            .expect("capture is enabled"))
    }
}
