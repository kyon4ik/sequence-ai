mod evaluation;
mod model;
mod observation;
pub mod probe;
mod snapshot;
pub mod suite;
mod training;

use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use burn::{module::Module, store::ModuleRecord, tensor::Device};
use clap::ValueEnum;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::Result;

use self::{
    evaluation::{Checkpoint, evaluate},
    model::Network,
    snapshot::{LogOffsets, Metadata},
    training::Trainer,
};

pub use evaluation::Options as EvaluationOptions;

#[derive(Clone, Copy, Deserialize, PartialEq, Serialize, ValueEnum)]
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

#[derive(Clone, Copy, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Target {
    #[default]
    Dqn,
    DoubleDqn,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Schedule {
    start: f64,
    end: f64,
    decay_decisions: NonZeroUsize,
}

impl Schedule {
    fn progress(self, decisions: usize) -> f64 {
        let duration = self.decay_decisions.get();
        decisions.min(duration) as f64 / duration as f64
    }

    fn at(self, decisions: usize) -> f64 {
        self.start + self.progress(decisions) * (self.end - self.start)
    }

    fn temperature_at(self, decisions: usize) -> f64 {
        self.end + (1.0 - self.progress(decisions)) * (self.start - self.end)
    }
}

#[derive(Clone, Copy)]
enum Exploration {
    Epsilon(Schedule),
    Temperature(Schedule),
}

#[derive(Clone, Deserialize, PartialEq, Serialize)]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    epsilon_start: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    epsilon_end: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    epsilon_decay_decisions: Option<NonZeroUsize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<Schedule>,
    batch_size: NonZeroUsize,
    replay_capacity: NonZeroUsize,
    replay_warmup: NonZeroUsize,
    train_every: NonZeroUsize,
    target_update_every: NonZeroUsize,
    opponent_update_every: NonZeroUsize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    heuristic_opponent_every: Option<NonZeroUsize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    immediate_win_loss_weight: Option<f32>,
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

#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Evaluation {
    first_shuffle_seed: u64,
    deal_count: NonZeroUsize,
    policy_seed: u64,
}

impl Config {
    fn validate(&self) -> Result<Exploration> {
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err("gamma must be between 0 and 1".into());
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err("learning_rate must be positive and finite".into());
        }
        if self.batch_size > self.replay_warmup || self.replay_warmup > self.replay_capacity {
            return Err(
                "replay must satisfy batch_size <= replay_warmup <= replay_capacity".into(),
            );
        }
        if self
            .immediate_win_loss_weight
            .is_some_and(|weight| !weight.is_finite() || weight < 0.0)
        {
            return Err("immediate_win_loss_weight must be finite and nonnegative".into());
        }

        self.evaluation
            .validate_deals(self.first_shuffle_seed, self.episodes)?;

        match (
            self.epsilon_start,
            self.epsilon_end,
            self.epsilon_decay_decisions,
            self.temperature,
        ) {
            (Some(start), Some(end), Some(decay_decisions), None) => {
                if !(0.0..=1.0).contains(&start) || !(0.0..=start).contains(&end) {
                    return Err(
                        "epsilon must satisfy 0 <= epsilon_end <= epsilon_start <= 1".into(),
                    );
                }
                Ok(Exploration::Epsilon(Schedule {
                    start,
                    end,
                    decay_decisions,
                }))
            }
            (None, None, None, Some(schedule)) => {
                if !schedule.start.is_finite()
                    || !(schedule.end > 0.0 && schedule.end <= schedule.start)
                {
                    return Err(
                        "temperature must satisfy 0 < end <= start, with finite endpoints".into(),
                    );
                }
                Ok(Exploration::Temperature(schedule))
            }
            _ => Err("configure either all three epsilon fields or a temperature schedule".into()),
        }
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
    let exploration = config.validate()?;
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

    fs::create_dir(output.join("checkpoints"))?;
    write_config(&output, &config)?;
    let logs = TrainingLogs::create(&output)?;
    let trainer = Trainer::new(&config, exploration, &device, &inference_device);
    train_run(&output, &config, &inference_device, trainer, logs, None)
}

pub fn resume(output: &Path, episodes: Option<NonZeroUsize>) -> Result<()> {
    let training = OpenOptions::new()
        .read(true)
        .write(true)
        .open(output.join("training.csv"))?;
    lock_training(&training, output)?;
    let (mut metadata, reader) = snapshot::open(output)?;

    let mut config: Config = read_config(&output.join("config.json"))?;
    metadata.config.episodes = config.episodes;
    if metadata.config != config {
        return Err("only the episode budget can change when resuming training".into());
    }
    if let Some(episodes) = episodes {
        config.episodes = episodes;
    }
    if config.episodes.get() < metadata.episode {
        return Err(format!(
            "the latest checkpoint already completed {} episodes",
            metadata.episode
        )
        .into());
    }

    let exploration = config.validate()?;
    let device = config.device.device()?;
    let inference_device = config.inference_device.unwrap_or(config.device).device()?;
    let mut trainer = Trainer::new(&config, exploration, &device, &inference_device);
    let online = trainer.restore_state(reader)?;
    let logs = TrainingLogs::resume(output, training, &metadata, &trainer)?;

    let path = snapshot::policy_path(output, metadata.episode);
    if !path.try_exists()? {
        snapshot::atomic_write(&path, |writer| Ok(writer.write_all(&online)?))?;
    }
    write_config(output, &config)?;
    eprintln!(
        "Resuming from checkpoint {} after {} learner decisions",
        metadata.episode, trainer.decisions
    );
    train_run(
        output,
        &config,
        &inference_device,
        trainer,
        logs,
        Some(metadata.episode),
    )
}

fn write_config(output: &Path, config: &Config) -> Result<()> {
    snapshot::atomic_write(&output.join("config.json"), |writer| {
        serde_json::to_writer_pretty(&mut *writer, config)?;
        Ok(writer.write_all(b"\n")?)
    })
}

fn lock_training(file: &File, output: &Path) -> Result<()> {
    file.try_lock().map_err(|error| {
        format!(
            "could not acquire the training lock for {}: {error}",
            output.display()
        )
        .into()
    })
}

struct TrainingLogs {
    training: csv::Writer<File>,
    evaluations: csv::Writer<File>,
    rows: Vec<Checkpoint>,
}

impl TrainingLogs {
    fn create(output: &Path) -> Result<Self> {
        let training = File::create_new(output.join("training.csv"))?;
        lock_training(&training, output)?;
        Ok(Self {
            training: csv::Writer::from_writer(training),
            evaluations: csv::Writer::from_path(output.join("evaluation.csv"))?,
            rows: Vec::new(),
        })
    }

    fn resume(
        output: &Path,
        mut training: File,
        metadata: &Metadata,
        trainer: &Trainer<'_>,
    ) -> Result<Self> {
        let mut evaluation = OpenOptions::new()
            .read(true)
            .write(true)
            .open(output.join("evaluation.csv"))?;
        let offsets = metadata.logs;
        if training.metadata()?.len() < offsets.training
            || evaluation.metadata()?.len() < offsets.evaluation
        {
            return Err("training or evaluation CSV is shorter than the latest snapshot".into());
        }
        let rows: Vec<Checkpoint> =
            csv::Reader::from_reader((&evaluation).take(offsets.evaluation))
                .deserialize()
                .collect::<std::result::Result<_, _>>()?;
        if !rows.last().is_some_and(|row| {
            row.episode == metadata.episode
                && row.decisions == trainer.decisions
                && row.updates == trainer.learner.updates
        }) {
            return Err("evaluation CSV disagrees with the latest training snapshot".into());
        }

        // Discard rows written after the last committed snapshot before replaying those episodes.
        training.set_len(offsets.training)?;
        training.seek(SeekFrom::Start(offsets.training))?;
        evaluation.set_len(offsets.evaluation)?;
        evaluation.seek(SeekFrom::Start(offsets.evaluation))?;

        Ok(Self {
            training: csv::WriterBuilder::new()
                .has_headers(offsets.training == 0)
                .from_writer(training),
            evaluations: csv::WriterBuilder::new()
                .has_headers(false)
                .from_writer(evaluation),
            rows,
        })
    }

    fn sync(&mut self) -> Result<LogOffsets> {
        self.training.flush()?;
        self.evaluations.flush()?;
        self.training.get_ref().sync_all()?;
        self.evaluations.get_ref().sync_all()?;
        Ok(LogOffsets {
            training: self.training.get_ref().metadata()?.len(),
            evaluation: self.evaluations.get_ref().metadata()?.len(),
        })
    }
}

fn train_run(
    output: &Path,
    config: &Config,
    inference_device: &Device,
    mut trainer: Trainer<'_>,
    mut logs: TrainingLogs,
    completed: Option<usize>,
) -> Result<()> {
    let model = trainer.learner.online.parameter_counts();
    let checkpoint = |episode, trainer: &Trainer<'_>, logs: &mut TrainingLogs| -> Result<()> {
        let online = trainer
            .learner
            .online
            .valid()
            .into_record()
            .into_bytes()?
            .shared();
        let network = Network::new(inference_device)
            .try_load_record(ModuleRecord::from_bytes(online.clone())?)?
            .valid();

        for totals in evaluate(
            &network,
            inference_device,
            &config.evaluation,
            config.action_limit,
            EvaluationOptions::default(),
        )? {
            let row = Checkpoint {
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
            logs.evaluations.serialize(&row)?;
            logs.rows.push(row);
        }

        let metadata = Metadata {
            config: config.clone(),
            episode,
            logs: logs.sync()?,
        };
        snapshot::save(output, &metadata, trainer, &online)?;

        // The snapshot can restore the policy artifact if interrupted after committing it.
        snapshot::atomic_write(&snapshot::policy_path(output, episode), |writer| {
            Ok(writer.write_all(&online)?)
        })?;
        evaluation::write_report(output, model, &logs.rows)?;
        eprintln!(
            "Evaluated checkpoint {episode} after {} learner decisions",
            trainer.decisions
        );
        Ok(())
    };

    if completed.is_some() {
        evaluation::write_report(output, model, &logs.rows)?;
    } else {
        checkpoint(0, &trainer, &mut logs)?;
    }
    for episode in completed.unwrap_or_default()..config.episodes.get() {
        let episode = episode + 1;
        logs.training.serialize(trainer.play_episode(episode)?)?;
        logs.training.flush()?;
        if episode.is_multiple_of(config.checkpoint_every.get()) || episode == config.episodes.get()
        {
            checkpoint(episode, &trainer, &mut logs)?;
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
