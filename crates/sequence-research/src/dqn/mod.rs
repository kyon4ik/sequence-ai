mod evaluation;
mod model;
mod observation;
pub mod probe;
mod training;

use std::{
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use burn::{module::Module, tensor::Device};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::Result;

use self::{evaluation::evaluate, model::Network, training::Trainer};

pub use evaluation::Options as EvaluationOptions;
pub use training::diagnostics;

#[derive(Clone, Copy, Deserialize, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Cpu,
    Cuda,
}

impl Backend {
    fn device(self) -> Result<Device> {
        match self {
            Self::Cpu => Ok(Device::flex()),
            #[cfg(feature = "cuda")]
            Self::Cuda => Ok(Device::cuda(0)),
            #[cfg(not(feature = "cuda"))]
            Self::Cuda => {
                Err("CUDA requires building sequence-research with --features cuda".into())
            }
        }
    }
}

#[derive(Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Target {
    #[default]
    Dqn,
    DoubleDqn,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    device: Backend,
    #[serde(default)]
    target: Target,
    seed: u64,
    first_shuffle_seed: u64,
    episodes: NonZeroUsize,
    action_limit: NonZeroUsize,
    gamma: f32,
    learning_rate: f64,
    epsilon_start: f64,
    epsilon_end: f64,
    epsilon_decay_decisions: NonZeroUsize,
    batch_size: NonZeroUsize,
    replay_capacity: NonZeroUsize,
    replay_warmup: NonZeroUsize,
    train_every: NonZeroUsize,
    target_update_every: NonZeroUsize,
    opponent_update_every: NonZeroUsize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    heuristic_opponent_every: Option<NonZeroUsize>,
    checkpoint_every: NonZeroUsize,
    #[serde(default)]
    sequence_shaping_weight: f32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    replay_curriculum: Vec<ReplayStage>,
    evaluation: Evaluation,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ReplayStage {
    through_episode: NonZeroUsize,
    max_decisions_to_end: NonZeroUsize,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Evaluation {
    first_shuffle_seed: u64,
    deal_count: NonZeroUsize,
    policy_seed: u64,
}

impl Config {
    fn read(path: &Path, backend: Option<Backend>) -> Result<Self> {
        let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let mut config: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        if let Some(backend) = backend {
            config.device = backend;
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err("gamma must be between 0 and 1".into());
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err("learning_rate must be positive and finite".into());
        }
        if !self.sequence_shaping_weight.is_finite() || self.sequence_shaping_weight < 0.0 {
            return Err("sequence_shaping_weight must be nonnegative and finite".into());
        }
        if !(0.0..=1.0).contains(&self.epsilon_start)
            || !(0.0..=self.epsilon_start).contains(&self.epsilon_end)
        {
            return Err("epsilon must satisfy 0 <= epsilon_end <= epsilon_start <= 1".into());
        }
        if self.batch_size > self.replay_warmup || self.replay_warmup > self.replay_capacity {
            return Err(
                "replay must satisfy batch_size <= replay_warmup <= replay_capacity".into(),
            );
        }
        if self.replay_curriculum.windows(2).any(|pair| {
            pair[0].through_episode >= pair[1].through_episode
                || pair[0].max_decisions_to_end >= pair[1].max_decisions_to_end
        }) {
            return Err(
                "replay curriculum episode bounds and depths must strictly increase".into(),
            );
        }

        let train_last = last_seed(self.first_shuffle_seed, self.episodes)?;
        let eval_last = last_seed(
            self.evaluation.first_shuffle_seed,
            self.evaluation.deal_count,
        )?;
        if self.first_shuffle_seed <= eval_last && self.evaluation.first_shuffle_seed <= train_last
        {
            return Err("training and evaluation shuffle seeds must be disjoint".into());
        }
        Ok(())
    }

    fn replay_depth(&self, episode: usize) -> Option<usize> {
        self.replay_curriculum
            .iter()
            .find(|stage| episode <= stage.through_episode.get())
            .map(|stage| stage.max_decisions_to_end.get())
    }
}

fn last_seed(first: u64, count: NonZeroUsize) -> Result<u64> {
    first
        .checked_add(u64::try_from(count.get() - 1)?)
        .ok_or_else(|| "shuffle seed range exceeds u64::MAX".into())
}

pub fn train(
    path: &Path,
    output: Option<PathBuf>,
    backend: Option<Backend>,
    seed: Option<u64>,
) -> Result<()> {
    let mut config = Config::read(path, backend)?;
    if let Some(seed) = seed {
        config.seed = seed;
    }
    let device = config.device.device()?;
    let output = match output {
        Some(path) => path,
        None => Path::new("results").join(path.file_stem().ok_or("expected a config filename")?),
    };
    let parent = output.parent().filter(|path| !path.as_os_str().is_empty());
    if let Some(parent) = parent {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&output).map_err(|error| format!("{}: {error}", output.display()))?;

    let checkpoints = output.join("checkpoints");
    fs::create_dir(&checkpoints)?;
    fs::write(
        output.join("config.json"),
        serde_json::to_string_pretty(&config)? + "\n",
    )?;
    let mut episodes = csv::Writer::from_path(output.join("training.csv"))?;
    let mut evaluations = csv::Writer::from_path(output.join("evaluation.csv"))?;
    let mut rows = Vec::new();
    let mut trainer = Trainer::new(&config, &device);
    let model = trainer.learner.online.parameter_counts();
    let mut checkpoint = |episode, trainer: &Trainer<'_>| -> Result<()> {
        let path = checkpoints.join(format!("episode-{episode:06}.bpk"));
        trainer.learner.online.valid().save_file(&path)?;
        let network = Network::load(&path, &device)?;

        for totals in evaluate(&network, &device, &config, EvaluationOptions::default())? {
            let row = evaluation::Checkpoint {
                episode,
                decisions: trainer.decisions,
                updates: trainer.learner.updates,
                opponent: totals.opponent,
                wins: totals.wins,
                draws: totals.draws,
                losses: totals.losses,
                cutoffs: totals.cutoffs,
                win_rate: totals.win_rate,
            };
            evaluations.serialize(&row)?;
            rows.push(row);
        }
        evaluations.flush()?;
        evaluation::write_report(&output, model, &rows)?;
        eprintln!(
            "Evaluated checkpoint {episode} after {} learner decisions",
            trainer.decisions
        );
        Ok(())
    };

    checkpoint(0, &trainer)?;
    for episode in 1..=config.episodes.get() {
        episodes.serialize(trainer.play_episode(episode)?)?;
        episodes.flush()?;
        if episode.is_multiple_of(config.checkpoint_every.get()) || episode == config.episodes.get()
        {
            checkpoint(episode, &trainer)?;
        }
    }
    println!("Results saved to {}", output.display());
    Ok(())
}

pub fn evaluate_checkpoint(
    path: &Path,
    config: &Path,
    backend: Option<Backend>,
    options: EvaluationOptions,
) -> Result<()> {
    let config = Config::read(config, backend)?;
    let device = config.device.device()?;
    let network = Network::load(path, &device)?;
    let results = evaluate(&network, &device, &config, options)?;
    println!("{}", serde_json::to_string_pretty(&results)?);
    Ok(())
}
