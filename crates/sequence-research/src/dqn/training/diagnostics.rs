mod win_replay;

use std::{
    collections::VecDeque,
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use clap::Args;
use serde_json::json;
use win_replay::WinReplay;

use super::{Experience, Learner, Trainer};
use crate::dqn::{Backend, EvaluationOptions, evaluate};
use crate::dqn::{Config, Target, model::Network, observation::ACTIONS};
use crate::{Result, experiment::GameEnd};
use burn::{module::Module, optim::AdamConfig, tensor::Device};
use serde::Serialize;

#[derive(Args)]
pub struct Options {
    config: PathBuf,
    output: PathBuf,
    #[arg(long, value_enum)]
    device: Option<Backend>,
    #[arg(long)]
    seed: Option<u64>,
    /// Stop all learners after this many updates
    #[arg(long, default_value = "4000")]
    updates: NonZeroUsize,
    /// Compare uniform Double DQN with a Double DQN given at least this many selected immediate wins per batch
    #[arg(long)]
    min_wins_per_batch: Option<NonZeroUsize>,
}

#[derive(Default, Serialize)]
pub(super) struct Coverage {
    transitions: usize,
    win_available: usize,
    win_selected: usize,
    terminal: usize,
}

impl Coverage {
    pub(super) fn observe(&mut self, e: &Experience) {
        self.transitions += 1;
        self.win_available += usize::from(!e.state.winning_actions.is_empty());
        self.win_selected += usize::from(e.state.winning_actions.contains(&e.action));
        self.terminal += usize::from(e.next.is_none());
    }
}

#[derive(Default)]
pub(super) struct Trace {
    pub(super) inserted: Coverage,
    pub(super) sampled: Coverage,
    pub(super) batches: Vec<Vec<Experience>>,
    pub(super) update_limit: usize,
    enrichment: Option<WinReplay>,
}

impl Trace {
    pub(super) fn record_batch(&mut self, uniform: &[&Experience], replay: &VecDeque<Experience>) {
        if let Some(enrichment) = &mut self.enrichment {
            enrichment.record(uniform, replay);
        }
        for experience in uniform {
            self.sampled.observe(experience);
        }
        self.batches
            .push(uniform.iter().copied().cloned().collect());
    }
}

fn selected_values(
    network: &Network,
    device: &Device,
    rows: &VecDeque<Experience>,
) -> Result<Vec<f32>> {
    let data = network
        .forward(rows.iter().map(|e| &e.state.observation), device)
        .into_data();
    Ok(data
        .as_slice::<f32>()?
        .as_chunks::<ACTIONS>()
        .0
        .iter()
        .zip(rows)
        .map(|(q, e)| q[e.action])
        .collect())
}

fn fixed_batch(config: &Config, device: &Device, directory: &Path) -> Result<()> {
    let mut trainer = Trainer::new(config, device);
    trainer.trace = Some(Trace::default()); // Collect real transitions with optimization disabled.
    let mut counts = [[0; 2]; 2];
    let mut examples = VecDeque::new();
    let mut records = Vec::new();
    for episode in 1..=config.episodes.get() {
        let row = trainer.play_episode(episode)?;
        let winner = match row.outcome {
            GameEnd::PlayerOneWin => 1,
            GameEnd::PlayerTwoWin => 2,
            _ => continue,
        };
        let win = winner == row.learner_seat;
        let count = &mut counts[row.learner_seat - 1][usize::from(win)];
        if *count == 2 {
            continue;
        }
        let e = trainer.replay.back().ok_or("missing terminal experience")?;
        let reward = if win { 1.0 } else { -1.0 };
        if e.next.is_some()
            || !e.state.legal[e.action]
            || e.reward != reward
            || e.state.winning_actions.contains(&e.action) != win
        {
            return Err(
                format!("terminal transition disagrees with rules at episode {episode}").into(),
            );
        }
        *count += 1;
        examples.push_back(e.clone());
        records.push(json!({
            "episode": episode,
            "shuffle_seed": row.shuffle_seed,
            "learner_seat": row.learner_seat,
            "action_index": e.action,
            "target": reward,
        }));
        if examples.len() == 8 {
            break;
        }
    }
    if examples.len() != 8 {
        return Err("insufficient terminal examples from both seats".into());
    }
    let before = selected_values(&trainer.learner.online.valid(), device, &examples)?;
    let batch = examples.iter().collect::<Vec<_>>();
    for _ in 0..200 {
        trainer.learner.update_batch(
            config,
            device,
            &trainer.training_device,
            &batch,
            Target::Dqn,
        )?;
    }
    let after = selected_values(&trainer.learner.online.valid(), device, &examples)?;
    let tmp = tempfile::tempdir_in(directory)?;
    let path = tmp.path().join("fit.bpk");
    trainer.learner.online.valid().save_file(&path)?;
    let restored = selected_values(&Network::load(&path, device)?, device, &examples)?;
    let mut passed = true;
    for (i, e) in examples.iter().enumerate() {
        records[i]["before"] = json!(before[i]);
        records[i]["after"] = json!(after[i]);
        records[i]["restored"] = json!(restored[i]);
        passed &= (after[i] - e.reward).abs() < 0.05 && (restored[i] - after[i]).abs() < 1e-5;
    }
    fs::write(
        directory.join("fixed-batch.json"),
        serde_json::to_vec_pretty(&json!({
            "updates": 200,
            "max_allowed_absolute_error": 0.05,
            "reload_tolerance": 1e-5,
            "passed": passed,
            "examples": records,
        }))?,
    )?;
    if !passed {
        return Err("fixed batch failed, see fixed-batch.json".into());
    }
    Ok(())
}

pub fn run(options: Options) -> Result<()> {
    let mut config = Config::read(&options.config, options.device)?;
    config.target = Target::Dqn;
    if let Some(seed) = options.seed {
        config.seed = seed;
    }
    if config.sequence_shaping_weight != 0.0 || !config.replay_curriculum.is_empty() {
        return Err("the terminal attribution experiment requires unshaped, uniform replay".into());
    }
    if options
        .min_wins_per_batch
        .is_some_and(|n| n > config.batch_size)
    {
        return Err("min-wins-per-batch must fit in the configured batch".into());
    }
    let device = config.device.device()?;
    fs::create_dir(&options.output)?;
    fs::write(
        options.output.join("config.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    fixed_batch(&config, &device, &options.output)?;
    let mut trainer = Trainer::new(&config, &device);
    let new_learner = || Learner {
        online: trainer.learner.online.clone(),
        target: trainer.learner.target.clone(),
        optimizer: AdamConfig::new().init(),
        updates: 0,
    };
    let mut double = new_learner();
    let mut enriched = options.min_wins_per_batch.map(|_| new_learner());
    trainer.trace = Some(Trace {
        enrichment: options
            .min_wins_per_batch
            .map(|n| WinReplay::new(n, config.seed)),
        update_limit: options.updates.get(),
        ..Trace::default()
    });
    let mut episodes = csv::Writer::from_path(options.output.join("training.csv"))?;
    let mut stop = 0;
    for episode in 1..=config.episodes.get() {
        episodes.serialize(trainer.play_episode(episode)?)?;
        let trace = trainer.trace.as_mut().expect("diagnostic trace enabled");
        let update_double =
            |learner: &mut Learner, batches: &mut Vec<Vec<Experience>>| -> Result<()> {
                for batch in batches.drain(..) {
                    learner.update_batch(
                        &config,
                        &device,
                        &trainer.training_device,
                        &batch.iter().collect::<Vec<_>>(),
                        Target::DoubleDqn,
                    )?;
                }
                Ok(())
            };
        update_double(&mut double, &mut trace.batches)?;
        if let (Some(learner), Some(replay)) = (&mut enriched, &mut trace.enrichment) {
            update_double(learner, &mut replay.batches)?;
        }
        if trainer.learner.updates == options.updates.get() {
            stop = episode;
            break;
        }
    }
    episodes.flush()?;
    if stop == 0 {
        return Err("increase episodes to reach the requested update budget".into());
    }
    let trace = trainer.trace.take().expect("diagnostic trace enabled");
    fs::write(
        options.output.join("coverage.json"),
        serde_json::to_vec_pretty(&json!({
            "episode": stop,
            "updates": double.updates,
            "inserted": trace.inserted,
            "sampled": trace.sampled,
        }))?,
    )?;
    let learners = if let (Some(learner), Some(replay)) = (enriched, trace.enrichment) {
        fs::write(
            options.output.join("win-replay.json"),
            serde_json::to_vec_pretty(&json!({
                "min_wins_per_batch": replay.minimum,
                "shortfall_batches": replay.shortfall_batches,
                "updates": learner.updates,
                "sampled": replay.sampled,
            }))?,
        )?;
        [("double-dqn", double), ("double-dqn-win-replay", learner)]
    } else {
        [("dqn", trainer.learner), ("double-dqn", double)]
    };
    for (name, learner) in learners {
        let run = options.output.join(name);
        fs::create_dir_all(run.join("checkpoints"))?;
        let path = run
            .join("checkpoints")
            .join(format!("episode-{stop:06}.bpk"));
        learner.online.valid().save_file(&path)?;
        let network = Network::load(&path, &device)?;
        if options.min_wins_per_batch.is_some() {
            win_replay::score_replay(
                &network,
                &device,
                &trainer.replay,
                &run.join("replay-ranks.csv"),
            )?;
        }
        let mut evaluation = EvaluationOptions::default();
        evaluation.games = Some(run.join("games.csv"));
        let totals = evaluate(&network, &device, &config, evaluation)?;
        fs::write(
            run.join("evaluation.json"),
            serde_json::to_vec_pretty(&totals)?,
        )?;
    }
    Ok(())
}
