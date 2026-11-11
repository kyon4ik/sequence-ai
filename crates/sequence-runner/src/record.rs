use crate::GameSnapshot;
use sequence_agent::AgentDescriptor;
use sequence_core::{Action, Color, GameOutcome};
use serde::{Deserialize, Serialize};
use std::{error::Error, time::Duration};

pub const FORMAT_VERSION: u32 = 1;
pub type RecordError = Box<dyn Error + Send + Sync + 'static>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchConfig {
    pub game_seed: u64,
    /// Red, then Blue. Independent of the game RNG and seat assignment.
    pub agent_seeds: [u64; 2],
    /// Actions, not completed turns: Classic exchanges also count.
    pub max_actions: usize,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            game_seed: 42,
            agent_seeds: [0, 1],
            max_actions: 1_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MatchHeader {
    pub format_version: u32,
    pub game_id: String,
    pub reward_id: String,
    pub config: MatchConfig,
    /// Red, then Blue; duplicate IDs are allowed for self-play.
    pub players: [AgentDescriptor; 2],
    pub initial_state: GameSnapshot,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlayerMetrics {
    /// Includes a failed inference attempt; successful actions live in steps.
    pub decisions: usize,
    pub decision_time: Duration,
    pub max_decision_time: Duration,
    pub reward: f64,
}

impl PlayerMetrics {
    pub fn mean_decision_time(&self) -> Duration {
        if self.decisions == 0 {
            Duration::ZERO
        } else {
            self.decision_time.div_f64(self.decisions as f64)
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MatchMetrics {
    /// From episode start to finish, including hooks and event recording.
    /// Excludes agent/model creation and the final event sink call.
    pub elapsed: Duration,
    pub actions: usize,
    pub turns: u16,
    pub players: [PlayerMetrics; 2],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionRecord {
    /// Zero-based action index, independent of completed turn count.
    pub index: usize,
    pub actor: Color,
    pub action: Action,
    pub decision_time: Duration,
    /// Rewards in Red/Blue order, including rewards to the nonacting player.
    pub rewards: [f64; 2],
    pub state: GameSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Termination {
    Completed(GameOutcome),
    /// An unfinished episode, never an implicit draw.
    ActionLimit,
    Aborted {
        reason: String,
    },
}

impl Termination {
    pub fn outcome(&self) -> Option<GameOutcome> {
        match self {
            Self::Completed(outcome) if *outcome != GameOutcome::InProgress => Some(*outcome),
            _ => None,
        }
    }
}

/// Full replay, self-contained for viewing. Seed/action reconstruction additionally
/// checks every stored observation against the current engine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MatchRecord {
    pub header: MatchHeader,
    pub steps: Vec<ActionRecord>,
    pub termination: Termination,
    pub metrics: MatchMetrics,
}

impl MatchRecord {
    pub fn frame_count(&self) -> usize {
        self.steps.len() + 1
    }

    /// Frame 0 is the dealt state; frame n follows action n-1. This requires
    /// neither a checkpoint, an inference backend, nor the original game engine.
    pub fn frame(&self, index: usize) -> Option<&GameSnapshot> {
        if index == 0 {
            Some(&self.header.initial_state)
        } else {
            self.steps.get(index - 1).map(|s| &s.state)
        }
    }
}

/// Owned events are suitable for a file/DB recorder or future WebSocket API.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MatchEvent {
    Started(MatchHeader),
    Action(ActionRecord),
    Finished {
        termination: Termination,
        metrics: MatchMetrics,
    },
}

/// Persistence and live consumers implement this boundary. A sink error stops
/// the match and returns the partial replay; the runner never ignores data loss.
pub trait EventSink {
    fn record(&mut self, event: &MatchEvent) -> Result<(), RecordError>;
}

impl EventSink for () {
    fn record(&mut self, _event: &MatchEvent) -> Result<(), RecordError> {
        Ok(())
    }
}

impl EventSink for Vec<MatchEvent> {
    fn record(&mut self, event: &MatchEvent) -> Result<(), RecordError> {
        self.push(event.clone());
        Ok(())
    }
}
