#![deny(unsafe_code)]
//! Execute two-player matches and expose serializable events, replays and metrics.
//! Scheduling, rankings, persistence, learning and presentation are separate layers.

mod record;
mod replay;
mod runner;
mod snapshot;

pub use record::{
    ActionRecord, EventSink, FORMAT_VERSION, MatchConfig, MatchEvent, MatchHeader, MatchMetrics,
    MatchRecord, PlayerMetrics, RecordError, Termination,
};
pub use replay::{Replay, ReplayError};
pub use runner::{MatchRunner, MatchSession, Participant, Reward, RunError, TerminalReward};
pub use snapshot::{BoardSnapshot, ClassicSnapshot, GameSnapshot, RecordedGame};
