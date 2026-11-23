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
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::Result;

use self::{evaluation::evaluate, model::Network, training::Trainer};

pub use evaluation::Options as EvaluationOptions;

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    inference_device: Option<Backend>,
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
    evaluation: Evaluation,
}

#[derive(Deserialize)]
struct SavedEvaluation {
    device: Backend,
    inference_device: Option<Backend>,
    first_shuffle_seed: u64,
    episodes: NonZeroUsize,
    action_limit: NonZeroUsize,
    evaluation: Evaluation,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Evaluation {
    first_shuffle_seed: u64,
    deal_count: NonZeroUsize,
    policy_seed: u64,
}

impl Config {
    fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err("gamma must be between 0 and 1".into());
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err("learning_rate must be positive and finite".into());
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

        self.evaluation
            .validate_deals(self.first_shuffle_seed, self.episodes)
    }
}

impl Evaluation {
    fn validate_deals(&self, train_first: u64, episodes: NonZeroUsize) -> Result<()> {
        let train_last = last_seed(train_first, episodes)?;
        let eval_last = last_seed(self.first_shuffle_seed, self.deal_count)?;
        if train_first <= eval_last && self.first_shuffle_seed <= train_last {
            return Err("training and evaluation shuffle seeds must be disjoint".into());
        }
        Ok(())
    }
}

fn read_config<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()).into())
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
    let mut config: Config = read_config(path)?;
    config.validate()?;
    if let Some(backend) = backend {
        config.device = backend;
    }
    if let Some(seed) = seed {
        config.seed = seed;
    }
    let device = config.device.device()?;
    let inference_device = config.inference_device.unwrap_or(config.device).device()?;
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
    let mut trainer = Trainer::new(&config, &device, &inference_device);
    let model = trainer.learner.online.parameter_counts();
    let mut checkpoint = |episode, trainer: &Trainer<'_>| -> Result<()> {
        let path = checkpoints.join(format!("episode-{episode:06}.bpk"));
        trainer.learner.online.valid().save_file(&path)?;
        let network = Network::load(&path, &inference_device)?;

        for totals in evaluate(
            &network,
            &inference_device,
            &config.evaluation,
            config.action_limit,
            EvaluationOptions::default(),
        )? {
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
    let config: SavedEvaluation = read_config(config)?;
    config
        .evaluation
        .validate_deals(config.first_shuffle_seed, config.episodes)?;
    let device = backend
        .or(config.inference_device)
        .unwrap_or(config.device)
        .device()?;
    let network = Network::load(path, &device)?;
    let results = evaluate(
        &network,
        &device,
        &config.evaluation,
        config.action_limit,
        options,
    )?;
    println!("{}", serde_json::to_string_pretty(&results)?);
    Ok(())
}
