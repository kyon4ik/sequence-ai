#![deny(unsafe_code)]
//! Policies and inference. Execution belongs to `sequence-runner`, learning to
//! `sequence-training`.

use sequence_core::{Action, Color, Game, GameOutcome, Simple};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, error::Error};

mod afterstate;
mod config;
mod greedy;
mod random;
pub use afterstate::Afterstate;
pub use config::{AgentConfig, AgentSpec, BuildError, InferenceDevice};
pub use greedy::GreedyAgent;
pub use random::RandomAgent;

#[cfg(feature = "burn")]
pub mod mc;

pub type AgentError = Box<dyn Error + Send + Sync + 'static>;

/// Stable experiment identity. Give distinct checkpoints distinct IDs for rankings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentDescriptor {
    pub id: String,
    pub name: String,
    pub metadata: BTreeMap<String, String>,
}

impl AgentDescriptor {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            metadata: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EpisodeContext {
    pub game_seed: u64,
    pub agent_seed: u64,
    pub color: Color,
}

#[derive(Clone, Copy, Debug)]
pub enum EpisodeStatus {
    Completed,
    Truncated,
    Aborted,
}

/// A result from this agent's perspective. Check status before using outcome;
/// unfinished or aborted episodes must not be counted as draws or losses.
#[derive(Clone, Copy, Debug)]
pub struct EpisodeResult {
    pub status: EpisodeStatus,
    pub outcome: GameOutcome,
    pub reward: f64,
    /// Successfully applied actions by this seat, excluding failed attempts.
    pub actions: usize,
    pub turns: u16,
}

/// Object-safe interface shared by heuristic and neural policies. Only the
/// actor's observation and legal actions reach a policy; inference may fail.
pub trait Agent<G: Game = Simple>: Send {
    fn choose_action(
        &mut self,
        observation: G::Observation<'_>,
        actions: &[Action],
    ) -> Result<Action, AgentError>;
    fn on_episode_start(&mut self, _context: EpisodeContext) {}
    fn on_episode_end(&mut self, _result: EpisodeResult) {}
}

impl<G: Game, A: Agent<G> + ?Sized> Agent<G> for Box<A> {
    fn choose_action(
        &mut self,
        observation: G::Observation<'_>,
        actions: &[Action],
    ) -> Result<Action, AgentError> {
        (**self).choose_action(observation, actions)
    }
    fn on_episode_start(&mut self, context: EpisodeContext) {
        (**self).on_episode_start(context);
    }
    fn on_episode_end(&mut self, result: EpisodeResult) {
        (**self).on_episode_end(result);
    }
}

/// Creates a fresh policy per seat and match with reproducible exploration.
pub trait AgentFactory<G: Game = Simple>: Send + Sync {
    fn descriptor(&self) -> AgentDescriptor;
    fn create(&self, seed: u64) -> Result<Box<dyn Agent<G>>, AgentError>;
}
