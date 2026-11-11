use crate::{
    ActionRecord, EventSink, FORMAT_VERSION, MatchConfig, MatchEvent, MatchHeader, MatchMetrics,
    MatchRecord, RecordedGame, Termination,
};
use sequence_agent::{
    Agent, AgentDescriptor, AgentFactory, EpisodeContext, EpisodeResult, EpisodeStatus,
};
use sequence_core::{Action, Color, Game, GameOutcome};
use std::{marker::PhantomData, time::Instant};

/// Replaceable reward shaping. Implementations may keep per-episode state;
/// reset is called before every match. No reward is invented on truncation.
pub trait Reward<G: Game> {
    fn id(&self) -> &str;
    fn reset(&mut self) {}
    fn rewards(&mut self, before: &G, action: Action, after: &G) -> [f64; 2];
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TerminalReward;

impl<G: Game> Reward<G> for TerminalReward {
    fn id(&self) -> &str {
        "terminal-win-loss-v1"
    }
    fn rewards(&mut self, _before: &G, _action: Action, after: &G) -> [f64; 2] {
        match after.outcome() {
            GameOutcome::Won(Color::Red) => [1., -1.],
            GameOutcome::Won(Color::Blue) => [-1., 1.],
            _ => [0., 0.],
        }
    }
}

pub struct Participant<'a, G: Game> {
    pub descriptor: AgentDescriptor,
    pub agent: &'a mut dyn Agent<G>,
}

impl<'a, G: Game> Participant<'a, G> {
    pub fn new(descriptor: AgentDescriptor, agent: &'a mut dyn Agent<G>) -> Self {
        Self { descriptor, agent }
    }
}

/// Setup failures have no record; execution failures retain all applied actions.
#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("cannot start match: {0}")]
    Setup(String),
    #[error("match aborted: {message}")]
    Execution {
        message: String,
        record: Box<MatchRecord>,
    },
}

impl RunError {
    pub fn partial_record(&self) -> Option<&MatchRecord> {
        match self {
            Self::Execution { record, .. } => Some(record),
            _ => None,
        }
    }
}

pub struct MatchRunner<G, R = TerminalReward> {
    reward: R,
    game: PhantomData<fn() -> G>,
}

impl<G: RecordedGame> Default for MatchRunner<G> {
    fn default() -> Self {
        Self::with_reward(TerminalReward)
    }
}

impl<G: RecordedGame, R: Reward<G>> MatchRunner<G, R> {
    pub fn with_reward(reward: R) -> Self {
        Self {
            reward,
            game: PhantomData,
        }
    }

    pub fn run_factories(
        &mut self,
        config: MatchConfig,
        factories: [&dyn AgentFactory<G>; 2],
    ) -> Result<MatchRecord, RunError> {
        self.run_factories_with_sink(config, factories, &mut ())
    }

    pub fn run_factories_with_sink(
        &mut self,
        config: MatchConfig,
        factories: [&dyn AgentFactory<G>; 2],
        sink: &mut dyn EventSink,
    ) -> Result<MatchRecord, RunError> {
        Self::validate_config(&config)?;
        let [red, blue] = factories;
        let mut red_agent = red
            .create(config.agent_seeds[0])
            .map_err(|e| RunError::Setup(format!("Red agent: {e}")))?;
        let mut blue_agent = blue
            .create(config.agent_seeds[1])
            .map_err(|e| RunError::Setup(format!("Blue agent: {e}")))?;
        self.run_with_sink(
            config,
            [
                Participant::new(red.descriptor(), &mut red_agent),
                Participant::new(blue.descriptor(), &mut blue_agent),
            ],
            sink,
        )
    }

    pub fn run(
        &mut self,
        config: MatchConfig,
        players: [Participant<'_, G>; 2],
    ) -> Result<MatchRecord, RunError> {
        self.run_with_sink(config, players, &mut ())
    }

    pub fn run_with_sink(
        &mut self,
        config: MatchConfig,
        mut players: [Participant<'_, G>; 2],
        sink: &mut dyn EventSink,
    ) -> Result<MatchRecord, RunError> {
        Self::validate_config(&config)?;
        if players.iter().any(|p| p.descriptor.id.trim().is_empty()) {
            return Err(RunError::Setup("agent IDs must not be empty".into()));
        }
        let mut game = G::from_seed(config.game_seed);
        let mut record = MatchRecord {
            header: MatchHeader {
                format_version: FORMAT_VERSION,
                game_id: G::ID.into(),
                reward_id: self.reward.id().into(),
                config: config.clone(),
                players: players.each_ref().map(|p| p.descriptor.clone()),
                initial_state: game.snapshot(),
            },
            steps: Vec::new(),
            termination: Termination::ActionLimit,
            metrics: MatchMetrics::default(),
        };
        let started = Instant::now();
        self.reward.reset();
        for (i, player) in players.iter_mut().enumerate() {
            player.agent.on_episode_start(EpisodeContext {
                game_seed: config.game_seed,
                agent_seed: config.agent_seeds[i],
                color: [Color::Red, Color::Blue][i],
            });
        }

        let mut failure = sink
            .record(&MatchEvent::Started(record.header.clone()))
            .err()
            .map(|e| format!("recording start: {e}"));
        let mut sink_failed = failure.is_some();
        let mut actions = Vec::new();
        while failure.is_none() && !game.is_game_over() && record.steps.len() < config.max_actions {
            let actor = game.current_player();
            let index = actor as usize;
            game.legal_actions(&mut actions);
            if actions.is_empty() {
                failure = Some("nonterminal game has no legal actions".into());
                break;
            }
            let decision_started = Instant::now();
            let selected = players[index]
                .agent
                .choose_action(game.observation(), &actions);
            let decision_time = decision_started.elapsed();
            let metrics = &mut record.metrics.players[index];
            metrics.decisions += 1;
            metrics.decision_time += decision_time;
            metrics.max_decision_time = metrics.max_decision_time.max(decision_time);
            let action = match selected {
                Ok(action) => action,
                Err(error) => {
                    failure = Some(format!("{actor} inference: {error}"));
                    break;
                }
            };
            let before = game.clone();
            if let Err(error) = game.apply(action) {
                failure = Some(format!("{actor} returned illegal action {action}: {error}"));
                break;
            }
            let rewards = self.reward.rewards(&before, action, &game);
            // Keep the successful move even if reward calculation or recording fails.
            let valid_reward = rewards.iter().enumerate().all(|(i, reward)| {
                reward.is_finite() && (record.metrics.players[i].reward + reward).is_finite()
            });
            let rewards = if valid_reward { rewards } else { [0., 0.] };
            for (i, reward) in rewards.iter().enumerate() {
                record.metrics.players[i].reward += reward;
            }
            let step = ActionRecord {
                index: record.steps.len(),
                actor,
                action,
                decision_time,
                rewards,
                state: game.snapshot(),
            };
            record.steps.push(step.clone());
            if let Err(error) = sink.record(&MatchEvent::Action(step)) {
                failure = Some(format!("recording action: {error}"));
                sink_failed = true;
            } else if !valid_reward {
                failure = Some("reward or cumulative return is not finite".into());
            }
        }

        record.termination = if let Some(reason) = &failure {
            Termination::Aborted {
                reason: reason.clone(),
            }
        } else if game.is_game_over() {
            Termination::Completed(game.outcome())
        } else {
            Termination::ActionLimit
        };
        record.metrics.actions = record.steps.len();
        record.metrics.turns = game.turns();
        for (i, player) in players.iter_mut().enumerate() {
            player.agent.on_episode_end(EpisodeResult {
                status: match &record.termination {
                    Termination::Completed(_) => EpisodeStatus::Completed,
                    Termination::ActionLimit => EpisodeStatus::Truncated,
                    Termination::Aborted { .. } => EpisodeStatus::Aborted,
                },
                outcome: game.outcome(),
                reward: record.metrics.players[i].reward,
                actions: record
                    .steps
                    .iter()
                    .filter(|step| step.actor as usize == i)
                    .count(),
                turns: game.turns(),
            });
        }
        record.metrics.elapsed = started.elapsed();
        // A failing sink is not called again. The partial record is the recovery path.
        if !sink_failed
            && let Err(error) = sink.record(&MatchEvent::Finished {
                termination: record.termination.clone(),
                metrics: record.metrics.clone(),
            })
        {
            let reason = format!("recording finish: {error}");
            record.termination = Termination::Aborted {
                reason: reason.clone(),
            };
            failure = Some(reason);
        }
        match failure {
            Some(message) => Err(RunError::Execution {
                message,
                record: Box::new(record),
            }),
            None => Ok(record),
        }
    }

    fn validate_config(config: &MatchConfig) -> Result<(), RunError> {
        if config.max_actions == 0 {
            Err(RunError::Setup("max_actions must be positive".into()))
        } else {
            Ok(())
        }
    }
}
