//! Training reports and returned models.
use super::{HistoricalProbability, OpponentEvaluation, OpponentMetrics};
use sequence_agent::mc::{McInferenceMetrics, ValueNet};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationMetrics {
    pub iteration: usize,
    pub episodes: usize,
    pub samples: usize,
    pub optimization_samples: usize,
    pub updates: usize,
    pub replay_samples: usize,
    pub replay_size: usize,
    /// Fixed-budget runs derive episode seeds independently of sample selection.
    pub rollout_seed: Option<u64>,
    pub mse: f64,
    pub evaluation_score: Option<f64>,
    pub evaluation_opponents: Vec<OpponentEvaluation>,
    pub opponents: Vec<OpponentMetrics>,
    pub historical_distribution: Vec<HistoricalProbability>,
    pub epsilon: f64,
    /// Red wins, Blue wins, draws.
    pub outcomes: [usize; 3],
    pub reward: [f64; 2],
    pub decision_time: [Duration; 2],
    pub rollout_time: Duration,
    pub rollout_inference: McInferenceMetrics,
    pub optimization_time: Duration,
    pub evaluation_time: Duration,
    pub evaluation_inference: McInferenceMetrics,
    pub elapsed: Duration,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrainingReport {
    pub iterations: Vec<IterationMetrics>,
}

pub struct TrainingOutput {
    pub model: ValueNet,
    pub best_model: Option<ValueNet>,
    pub report: TrainingReport,
}
