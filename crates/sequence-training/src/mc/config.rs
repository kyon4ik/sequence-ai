//! Learning settings and resume validation.
use super::{HistoricalSampling, OpponentMix, ReplayConfig, SamplingStrategy};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainingConfig {
    pub iterations: usize,
    pub games_per_iteration: usize,
    pub batch_size: usize,
    /// Concurrent games whose candidate moves share an inference batch.
    /// This execution setting may change when resuming a checkpoint.
    pub inference_batch_games: usize,
    pub learning_rate: f64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    /// None decays over the configured total iterations. Set explicitly
    /// to extend training without slowing or restarting exploration decay.
    pub epsilon_decay_iterations: Option<usize>,
    pub seed: u64,
    pub evaluation_interval: usize,
    pub evaluation_games: usize,
    pub opponents: OpponentMix,
    pub historical_sampling: HistoricalSampling,
    pub sampling: SamplingStrategy,
    /// Fixed full gradient batches per iteration; None uses the fresh sample count.
    pub updates_per_iteration: Option<usize>,
    pub replay: Option<ReplayConfig>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            games_per_iteration: 128,
            batch_size: 256,
            inference_batch_games: 16,
            learning_rate: 1e-4,
            epsilon_start: 0.2,
            epsilon_end: 0.05,
            epsilon_decay_iterations: None,
            seed: 42,
            evaluation_interval: 10,
            evaluation_games: 20,
            opponents: OpponentMix::default(),
            historical_sampling: HistoricalSampling::default(),
            sampling: SamplingStrategy::Natural,
            updates_per_iteration: None,
            replay: None,
        }
    }
}

impl TrainingConfig {
    pub(super) fn epsilon_at(&self, iteration: usize) -> f64 {
        let horizon = self.epsilon_decay_iterations.unwrap_or(self.iterations);
        let fraction = (iteration as f64 / horizon.saturating_sub(1).max(1) as f64).min(1.0);
        self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
    }

    pub fn validate(&self) -> Result<(), sequence_agent::AgentError> {
        if self.iterations == 0
            || self.games_per_iteration == 0
            || self.batch_size == 0
            || self.inference_batch_games == 0
        {
            return Err("iterations, games_per_iteration, batch_size and inference_batch_games must be positive".into());
        }
        if !self.learning_rate.is_finite()
            || self.learning_rate <= 0.
            || !(0.0..=1.0).contains(&self.epsilon_start)
            || !(0.0..=1.0).contains(&self.epsilon_end)
        {
            return Err("invalid learning rate or exploration probability".into());
        }
        if self.evaluation_interval == 0
            || self.evaluation_games == 0
            || !self.evaluation_games.is_multiple_of(2)
        {
            return Err(
                "evaluation interval must be positive and games must be positive and even".into(),
            );
        }
        if let Some(updates) = self.updates_per_iteration
            && (updates == 0 || updates.checked_mul(self.batch_size).is_none())
        {
            return Err(
                "updates_per_iteration must be positive and its sample budget must fit usize"
                    .into(),
            );
        }
        if self.epsilon_decay_iterations == Some(0) {
            return Err("epsilon_decay_iterations must be positive".into());
        }
        if let Some(replay) = &self.replay {
            replay.validate()?;
        }
        self.opponents.validate()?;
        self.historical_sampling.validate()?;
        Ok(())
    }
}

pub(super) fn resume_compatible(saved: &TrainingConfig, requested: &TrainingConfig) -> bool {
    if requested.iterations < saved.iterations {
        return false;
    }
    let mut normalized = saved.clone();
    normalized.epsilon_decay_iterations =
        Some(saved.epsilon_decay_iterations.unwrap_or(saved.iterations));
    normalized.iterations = requested.iterations;
    normalized.inference_batch_games = requested.inference_batch_games;
    let mut requested = requested.clone();
    requested.epsilon_decay_iterations = Some(
        requested
            .epsilon_decay_iterations
            .unwrap_or(requested.iterations),
    );
    normalized == requested
}
