use crate::{
    ActionRecord, EventSink, FORMAT_VERSION, MatchConfig, MatchEvent, MatchHeader, MatchMetrics,
    MatchRecord, RecordedGame, Termination,
};
use sequence_agent::{
    Agent, AgentDescriptor, AgentFactory, EpisodeContext, EpisodeResult, EpisodeStatus,
};
use sequence_core::{Action, Color, Game, GameOutcome};
use std::{
    marker::PhantomData,
    time::{Duration, Instant},
};

/// Replaceable reward shaping. Implementations may keep per-episode state;
/// reset is called before every match. No reward is invented on truncation.
pub trait Reward<G: Game> {
    fn id(&self) -> &str;
    fn reset(&mut self) {}
    fn rewards(&mut self, before: &G, action: Action, after: &G) -> [f64; 2];
}

impl<G: Game, R: Reward<G> + ?Sized> Reward<G> for &mut R {
    fn id(&self) -> &str {
        (**self).id()
    }
    fn reset(&mut self) {
        (**self).reset();
    }
    fn rewards(&mut self, before: &G, action: Action, after: &G) -> [f64; 2] {
        (**self).rewards(before, action, after)
    }
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

/// A match paused between decisions. The caller owns policy scheduling and hooks;
/// the session owns rule execution, rewards, action limits and replay recording.
pub struct MatchSession<G, R = TerminalReward> {
    game: G,
    reward: R,
    record: MatchRecord,
    started: Instant,
}

impl<G: RecordedGame> MatchSession<G> {
    pub fn new(config: MatchConfig, players: [AgentDescriptor; 2]) -> Result<Self, RunError> {
        Self::with_reward(config, players, TerminalReward)
    }
}

impl<G: RecordedGame, R: Reward<G>> MatchSession<G, R> {
    pub fn with_reward(
        config: MatchConfig,
        players: [AgentDescriptor; 2],
        mut reward: R,
    ) -> Result<Self, RunError> {
        MatchRunner::<G>::validate_config(&config)?;
        if players.iter().any(|p| p.id.trim().is_empty()) {
            return Err(RunError::Setup("agent IDs must not be empty".into()));
        }
        let game = G::from_seed(config.game_seed);
        let record = MatchRecord {
            header: MatchHeader {
                format_version: FORMAT_VERSION,
                game_id: G::ID.into(),
                reward_id: reward.id().into(),
                config,
                players,
                initial_state: game.snapshot(),
            },
            steps: Vec::new(),
            termination: Termination::ActionLimit,
            metrics: MatchMetrics::default(),
        };
        let started = Instant::now();
        reward.reset();
        Ok(Self {
            game,
            reward,
            record,
            started,
        })
    }

    pub fn game(&self) -> &G {
        &self.game
    }

    pub fn record(&self) -> &MatchRecord {
        &self.record
    }

    pub fn is_finished(&self) -> bool {
        self.game.is_game_over()
            || self.record.steps.len() >= self.record.header.config.max_actions
            || matches!(self.record.termination, Termination::Aborted { .. })
    }

    /// Decision time is attributed by the scheduler. For shared inference batches,
    /// callers can apportion batch elapsed time among the participating decisions.
    pub fn step(&mut self, action: Action, decision_time: Duration) -> Result<(), RunError> {
        if let Err(message) = self.advance(action, decision_time) {
            self.finalize(Some(&message));
            return Err(RunError::Execution {
                message,
                record: Box::new(self.record.clone()),
            });
        }
        Ok(())
    }

    /// Only completed, action-limited or aborted sessions can be finished.
    pub fn finish(mut self) -> Result<MatchRecord, RunError> {
        if !self.is_finished() {
            let message = "match still has pending decisions".to_string();
            self.finalize(Some(&message));
            return Err(RunError::Execution {
                message,
                record: Box::new(self.record),
            });
        }
        self.finalize(None);
        if let Termination::Aborted { reason } = &self.record.termination {
            return Err(RunError::Execution {
                message: reason.clone(),
                record: Box::new(self.record),
            });
        }
        Ok(self.record)
    }

    fn record_decision(&mut self, decision_time: Duration) {
        let metrics = &mut self.record.metrics.players[self.game.current_player() as usize];
        metrics.decisions += 1;
        metrics.decision_time += decision_time;
        metrics.max_decision_time = metrics.max_decision_time.max(decision_time);
    }

    fn advance(&mut self, action: Action, decision_time: Duration) -> Result<(), String> {
        if self.is_finished() {
            return Err("cannot advance a finished match".into());
        }
        self.record_decision(decision_time);
        let actor = self.game.current_player();
        let before = self.game.clone();
        self.game
            .apply(action)
            .map_err(|error| format!("{actor} returned illegal action {action}: {error}"))?;
        let rewards = self.reward.rewards(&before, action, &self.game);
        let valid_reward = rewards.iter().enumerate().all(|(i, reward)| {
            reward.is_finite() && (self.record.metrics.players[i].reward + reward).is_finite()
        });
        let rewards = if valid_reward { rewards } else { [0., 0.] };
        for (i, reward) in rewards.iter().enumerate() {
            self.record.metrics.players[i].reward += reward;
        }
        self.record.steps.push(ActionRecord {
            index: self.record.steps.len(),
            actor,
            action,
            decision_time,
            rewards,
            state: self.game.snapshot(),
        });
        if !valid_reward {
            return Err("reward or cumulative return is not finite".into());
        }
        Ok(())
    }

    fn finalize(&mut self, failure: Option<&str>) {
        if let Some(reason) = failure {
            self.record.termination = Termination::Aborted {
                reason: reason.into(),
            };
        } else if !matches!(self.record.termination, Termination::Aborted { .. }) {
            self.record.termination = if self.game.is_game_over() {
                Termination::Completed(self.game.outcome())
            } else {
                Termination::ActionLimit
            };
        }
        self.record.metrics.actions = self.record.steps.len();
        self.record.metrics.turns = self.game.turns();
        self.record.metrics.elapsed = self.started.elapsed();
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
        let mut session = MatchSession::with_reward(
            config.clone(),
            players.each_ref().map(|p| p.descriptor.clone()),
            &mut self.reward,
        )?;
        for (i, player) in players.iter_mut().enumerate() {
            player.agent.on_episode_start(EpisodeContext {
                game_seed: config.game_seed,
                agent_seed: config.agent_seeds[i],
                color: [Color::Red, Color::Blue][i],
            });
        }

        let mut failure = sink
            .record(&MatchEvent::Started(session.record.header.clone()))
            .err()
            .map(|e| format!("recording start: {e}"));
        let mut sink_failed = failure.is_some();
        let mut actions = Vec::new();
        while failure.is_none() && !session.is_finished() {
            let actor = session.game.current_player();
            let index = actor as usize;
            session.game.legal_actions(&mut actions);
            if actions.is_empty() {
                failure = Some("nonterminal game has no legal actions".into());
                break;
            }
            let decision_started = Instant::now();
            let selected = players[index]
                .agent
                .choose_action(session.game.observation(), &actions);
            let decision_time = decision_started.elapsed();
            let action = match selected {
                Ok(action) => action,
                Err(error) => {
                    session.record_decision(decision_time);
                    failure = Some(format!("{actor} inference: {error}"));
                    break;
                }
            };
            let count = session.record.steps.len();
            failure = session.advance(action, decision_time).err();
            // A successfully applied move is emitted even if reward validation failed.
            if session.record.steps.len() > count
                && let Err(error) = sink.record(&MatchEvent::Action(
                    session.record.steps.last().unwrap().clone(),
                ))
            {
                failure = Some(format!("recording action: {error}"));
                sink_failed = true;
            }
        }
        session.finalize(failure.as_deref());
        if !sink_failed
            && let Err(error) = sink.record(&MatchEvent::Finished {
                termination: session.record.termination.clone(),
                metrics: session.record.metrics.clone(),
            })
        {
            let reason = format!("recording finish: {error}");
            session.record.termination = Termination::Aborted {
                reason: reason.clone(),
            };
            failure = Some(reason);
        }
        let game = session.game;
        let record = session.record;
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
