mod reuse;
mod statistics;

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Read,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    process::Command,
};

use clap::Args;
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::{Result, experiment::AgentKind};

use super::{
    Backend, Evaluation, SavedEvaluation,
    evaluation::{self, Checkpoint, GameRow, MatchOptions, OPPONENTS, Totals},
    figures::{self, Axis, Curve, Series},
    last_seed,
    model::Network,
    read_config, snapshot,
    tactics::{self, Score},
};

#[derive(Args)]
pub struct Options {
    /// JSON config naming the runs and shared evaluation panels
    config: PathBuf,
    /// New report directory
    #[arg(short, long, default_value = "results/evaluation")]
    output: PathBuf,
    /// Reuse matching checkpoint results and decision targets from a prior report
    #[arg(long)]
    reuse: Option<PathBuf>,
    /// CPU worker threads. CUDA evaluation uses one worker by default.
    #[arg(short, long)]
    jobs: Option<NonZeroUsize>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    runs_directory: PathBuf,
    arms: Vec<Arm>,
    seeds: Vec<u64>,
    checkpoint_percent: NonZeroUsize,
    device: Backend,
    action_limit: NonZeroUsize,
    validation: Evaluation,
    holdout: Evaluation,
    tactics: Evaluation,
    bootstrap_seed: u64,
    bootstrap_replicates: NonZeroUsize,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Arm {
    directory: String,
    label: String,
}

#[derive(Deserialize)]
struct RunSettings {
    seed: u64,
    #[serde(flatten)]
    settings: SavedEvaluation,
}

#[derive(Serialize)]
struct Available {
    episode: usize,
    decisions: usize,
    path: PathBuf,
}

struct Run {
    arm: usize,
    seed: u64,
    directory: PathBuf,
    checkpoints: Vec<Available>,
    // Each target within this run's budget maps to a checkpoint index.
    sampled: Vec<Option<usize>>,
}

impl Run {
    fn last(&self) -> usize {
        self.checkpoints.len() - 1
    }
}

struct Matches {
    totals: [Totals; 2],
    games: Vec<GameRow>,
}

struct Validated {
    matches: Matches,
    tactics: Vec<Score>,
    sha256: String,
}

type CheckpointId = (usize, usize);
type CheckpointSet = BTreeSet<CheckpointId>;
type Evaluations<T> = BTreeMap<CheckpointId, T>;

// Bump when a change affects match outcomes or probe scores for the same inputs.
const EVALUATION_VERSION: u32 = 1;

pub fn run(options: Options) -> Result<()> {
    if options.output.try_exists()? {
        return Err(format!(
            "{} already exists. Choose a new --output directory.",
            options.output.display()
        )
        .into());
    }

    let parent = options
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let staging = tempfile::tempdir_in(parent)?;

    let config: Config = read_config(&options.config)?;
    let mut runs = discover(&config)?;
    let prior = options
        .reuse
        .as_deref()
        .map(|path| reuse::PriorReport::open(path, &config))
        .transpose()?;
    let targets = match &prior {
        Some(report) => report.targets().to_vec(),
        None => sample_targets(&runs, config.checkpoint_percent)?,
    };
    sample(&mut runs, &targets);
    let positions = tactics::positions(&config.tactics, config.action_limit)?;
    if positions.is_empty() {
        return Err("no tactical positions on the requested deals".into());
    }
    let jobs = options.jobs.map_or_else(
        || match config.device {
            Backend::Cpu => std::thread::available_parallelism().map_or(1, NonZeroUsize::get),
            Backend::Cuda => 1,
        },
        NonZeroUsize::get,
    );
    let pool = ThreadPoolBuilder::new().num_threads(jobs).build()?;
    let candidates: CheckpointSet = runs
        .iter()
        .enumerate()
        .flat_map(|(index, run)| {
            run.sampled
                .iter()
                .flatten()
                .copied()
                .chain([run.last()])
                .map(move |id| (index, id))
        })
        .collect();
    let hashes = for_checkpoints(&pool, &runs, &candidates, |_, _, checkpoint| {
        sha256(&checkpoint.path)
    })?;
    let (mut validation, mut holdout) = match &prior {
        Some(report) => report.load(&config, &runs, &hashes, &positions)?,
        None => (Evaluations::new(), Evaluations::new()),
    };

    let pending: CheckpointSet = candidates
        .into_iter()
        .filter(|id| !validation.contains_key(id))
        .collect();
    let reused_validation = validation.len();
    eprintln!(
        "Validating {} checkpoints on {} paired deals ({reused_validation} reused)",
        pending.len(),
        config.validation.deal_count,
    );
    validation.extend(for_checkpoints(
        &pool,
        &runs,
        &pending,
        |id, run, checkpoint| {
            let device = config.device.device()?;
            let network = Network::load(&checkpoint.path, &device)?;
            Ok(Validated {
                matches: play(&network, &device, &config.validation, config.action_limit)?,
                tactics: tactics::score(
                    &network,
                    &device,
                    &positions,
                    &run.directory,
                    Some(checkpoint.episode),
                )?,
                sha256: hashes[&id].clone(),
            })
        },
    )?);

    let best: Vec<_> = runs
        .iter()
        .enumerate()
        .map(|(index, run)| {
            validation
                .range((index, 0)..=(index, run.last()))
                .filter_map(|(&(_, checkpoint), validated)| {
                    Some((validated.matches.totals[1].win_rate?, checkpoint))
                })
                .max_by(
                    |(left_rate, left_checkpoint), (right_rate, right_checkpoint)| {
                        left_rate
                            .total_cmp(right_rate)
                            .then(right_checkpoint.cmp(left_checkpoint))
                    },
                )
                .map(|(_, checkpoint)| checkpoint)
                .ok_or_else(|| {
                    format!(
                        "{} has no completed validation games against Heuristic",
                        run.directory.display()
                    )
                })
        })
        .collect::<std::result::Result<_, _>>()?;
    let finalists: CheckpointSet = runs
        .iter()
        .zip(&best)
        .enumerate()
        .flat_map(|(index, (run, &best))| [(index, best), (index, run.last())])
        .collect();
    holdout.retain(|id, _| finalists.contains(id));
    let pending: CheckpointSet = finalists
        .into_iter()
        .filter(|id| !holdout.contains_key(id))
        .collect();
    let reused_holdout = holdout.len();

    eprintln!(
        "Evaluating {} finalists on {} held-out paired deals ({reused_holdout} reused)",
        pending.len(),
        config.holdout.deal_count,
    );
    holdout.extend(for_checkpoints(
        &pool,
        &runs,
        &pending,
        |_, _, checkpoint| {
            let device = config.device.device()?;
            let network = Network::load(&checkpoint.path, &device)?;
            play(&network, &device, &config.holdout, config.action_limit)
        },
    )?);
    let comparisons = statistics::compare_all(&pool, &config, &runs, &best, &holdout);

    let report = Report {
        config: &config,
        runs: &runs,
        targets: &targets,
        validation: &validation,
        holdout: &holdout,
        best: &best,
        comparisons: &comparisons,
    };
    let reuse = prior
        .as_ref()
        .map(|report| report.provenance(reused_validation, reused_holdout));
    report.write(staging.path(), jobs, reuse)?;
    fs::rename(staging.path(), &options.output)?;
    println!("Results saved to {}", options.output.display());
    Ok(())
}

fn discover(config: &Config) -> Result<Vec<Run>> {
    if config.arms.is_empty() || config.seeds.is_empty() || config.checkpoint_percent.get() > 100 {
        return Err(
            "configure at least one arm and seed and a checkpoint percent from 1 to 100".into(),
        );
    }
    if config.seeds.iter().collect::<BTreeSet<_>>().len() != config.seeds.len()
        || config
            .arms
            .iter()
            .map(|arm| &arm.directory)
            .collect::<BTreeSet<_>>()
            .len()
            != config.arms.len()
    {
        return Err("arm directories and training seeds must be unique".into());
    }

    let panels = [&config.validation, &config.holdout, &config.tactics];
    for (index, panel) in panels.into_iter().enumerate() {
        last_seed(panel.first_shuffle_seed, panel.deal_count)?;
        for other in &panels[..index] {
            other
                .validate_deals(panel.first_shuffle_seed, panel.deal_count)
                .map_err(|_| "validation, holdout and tactics shuffle seeds must be disjoint")?;
        }
    }

    let mut runs = Vec::new();
    for (arm, definition) in config.arms.iter().enumerate() {
        for &seed in &config.seeds {
            let directory = config
                .runs_directory
                .join(&definition.directory)
                .join(format!("seed-{seed}"));
            let saved: RunSettings = read_config(&directory.join("config.json"))?;
            if seed != saved.seed {
                return Err(format!(
                    "{} records training seed {}, expected {seed}",
                    directory.display(),
                    saved.seed
                )
                .into());
            }
            let settings = saved.settings;
            for panel in panels {
                panel.validate_deals(settings.first_shuffle_seed, settings.episodes)?;
                panel.validate_deals(settings.evaluation.first_shuffle_seed, settings.evaluation.deal_count)
                    .map_err(|_| "evaluation suite and training monitoring shuffle seeds must be disjoint")?;
            }

            let mut checkpoints = Vec::new();
            let mut rows = csv::Reader::from_path(directory.join("evaluation.csv"))?;
            for row in rows.deserialize::<Checkpoint>() {
                let row = row?;
                if row.opponent == AgentKind::Random {
                    let path = snapshot::policy_path(&directory, row.episode);
                    if path.is_file() {
                        checkpoints.push(Available {
                            episode: row.episode,
                            decisions: row.decisions,
                            path,
                        });
                    }
                }
            }
            checkpoints.sort_unstable_by_key(|checkpoint| checkpoint.episode);
            if checkpoints.is_empty()
                || checkpoints.windows(2).any(|two| {
                    two[0].episode == two[1].episode || two[0].decisions > two[1].decisions
                })
            {
                return Err(format!(
                    "{} requires saved checkpoints with ordered evaluation records",
                    directory.display()
                )
                .into());
            }
            runs.push(Run {
                arm,
                seed,
                directory,
                checkpoints,
                sampled: Vec::new(),
            });
        }
    }
    Ok(runs)
}

fn sample_targets(runs: &[Run], percent: NonZeroUsize) -> Result<Vec<usize>> {
    let first = runs
        .iter()
        .map(|run| run.checkpoints[0].decisions)
        .max()
        .unwrap();
    let last = runs
        .iter()
        .map(|run| run.checkpoints[run.last()].decisions)
        .min()
        .unwrap();
    let available = runs.iter().map(|run| run.checkpoints.len()).min().unwrap();
    if last < first {
        return Err("training runs share no learner-decision range".into());
    }
    let count = (available * percent.get())
        .div_ceil(100)
        .max(2)
        .min(available);
    let intervals = count.saturating_sub(1).min(last - first);
    Ok(if intervals == 0 {
        vec![first]
    } else {
        (0..=intervals)
            .map(|i| {
                let offset = ((last - first) as u128 * i as u128 + (intervals / 2) as u128)
                    / intervals as u128;
                first + offset as usize
            })
            .collect()
    })
}

fn sample(runs: &mut [Run], targets: &[usize]) {
    for run in runs {
        let range = run.checkpoints[0].decisions..=run.checkpoints[run.last()].decisions;
        run.sampled = targets
            .iter()
            .map(|&target| {
                range.contains(&target).then(|| {
                    run.checkpoints
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, checkpoint)| {
                            (checkpoint.decisions.abs_diff(target), checkpoint.decisions)
                        })
                        .unwrap()
                        .0
                })
            })
            .collect();
    }
}

fn for_checkpoints<'a, T: Send>(
    pool: &ThreadPool,
    runs: &'a [Run],
    checkpoints: &CheckpointSet,
    evaluate: impl Fn(CheckpointId, &'a Run, &'a Available) -> Result<T> + Sync,
) -> Result<Evaluations<T>> {
    pool.install(|| {
        checkpoints
            .par_iter()
            .map(|&(run, id)| {
                let checkpoint = &runs[run].checkpoints[id];
                evaluate((run, id), &runs[run], checkpoint)
                    .map(|data| ((run, id), data))
                    .map_err(|error| format!("{}: {error}", checkpoint.path.display()))
            })
            .collect::<std::result::Result<_, _>>()
    })
    .map_err(Into::into)
}

fn play(
    network: &Network,
    device: &burn::tensor::Device,
    panel: &Evaluation,
    action_limit: NonZeroUsize,
) -> Result<Matches> {
    let mut games = Vec::new();
    let totals = evaluation::evaluate_with_games(
        network,
        device,
        panel,
        action_limit,
        &MatchOptions {
            tactics: true,
            ..Default::default()
        },
        |game| {
            games.push(game);
            Ok(())
        },
    )?;
    Ok(Matches { totals, games })
}

fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 32768];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            return Ok(format!("{:x}", digest.finalize()));
        }
        digest.update(&buffer[..count]);
    }
}

struct Report<'a> {
    config: &'a Config,
    runs: &'a [Run],
    targets: &'a [usize],
    validation: &'a Evaluations<Validated>,
    holdout: &'a Evaluations<Matches>,
    best: &'a [usize],
    comparisons: &'a [statistics::Comparison],
}

impl Report<'_> {
    fn write_figures(&self, directory: &Path) -> Result<()> {
        figures::write_curves(
            &directory.join("win-rates.svg"),
            "Win rates across training",
            &format!(
                "{} shared validation deals · both seats · {} training seeds · small dots show individual seeds",
                self.config.validation.deal_count,
                self.config.seeds.len(),
            ),
            OPPONENTS.map(|(_, label)| label.to_owned()),
            Axis::Decisions,
            "Win rate (completed games)",
            &self.curves(|panel, result| result.matches.totals[panel].win_rate),
        )?;

        let scores = &self.validation.values().next().unwrap().tactics;
        figures::write_curves(
            &directory.join("tactics.svg"),
            "Tactical choices across training",
            "Fixed positions from Random vs Random games · small dots show individual seeds",
            tactics::TACTICS.map(|tactic| {
                let count = tactics::counts(scores, tactic).positions;
                format!("{} · {count} positions", tactic.label())
            }),
            Axis::Decisions,
            "Tactical action ranked first",
            &self.curves(|panel, result| {
                tactics::counts(&result.tactics, tactics::TACTICS[panel]).rate()
            }),
        )
    }

    fn curves(&self, rate: impl Fn(usize, &Validated) -> Option<f64>) -> Vec<Series<'_>> {
        self.config
            .arms
            .iter()
            .enumerate()
            .map(|(arm, definition)| {
                let seeds: Vec<_> = self
                    .runs
                    .iter()
                    .enumerate()
                    .filter(|(_, run)| run.arm == arm)
                    .collect();
                let panels = std::array::from_fn(|panel| {
                    let mut curve = Curve::default();
                    for (point, &target) in self.targets.iter().enumerate() {
                        let rates: Vec<_> = seeds
                            .iter()
                            .map(|&(index, run)| {
                                let id = run.sampled[point]?;
                                let value = rate(panel, &self.validation[&(index, id)])?;
                                Some((run.checkpoints[id].decisions, value))
                            })
                            .collect();
                        curve.samples.extend(rates.iter().flatten().copied());
                        let mean = statistics::complete_mean(
                            rates.iter().map(|rate| rate.map(|(_, value)| value)),
                        );
                        curve.line.push((target, mean));
                    }
                    curve
                });
                Series {
                    label: &definition.label,
                    panels,
                }
            })
            .collect()
    }

    fn write(
        &self,
        directory: &Path,
        jobs: usize,
        reuse: Option<reuse::Provenance<'_>>,
    ) -> Result<()> {
        fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
            fs::write(path, serde_json::to_string_pretty(value)? + "\n")?;
            Ok(())
        }

        let mut games = csv::Writer::from_path(directory.join("games.csv"))?;
        games.write_record([
            "panel",
            "run",
            "episode",
            "shuffle_seed",
            "learner_seat",
            "opponent",
            "outcome",
        ])?;
        let mut tactics = tactics::writer(&directory.join("tactical-positions.csv"))?;

        let mut reports = Vec::new();
        for (index, run) in self.runs.iter().enumerate() {
            let best = self.best[index];
            let last = run.last();
            let mut checkpoints = Vec::new();
            for (&(_, id), validated) in self.validation.range((index, 0)..=(index, last)) {
                let checkpoint = &run.checkpoints[id];
                let holdout = self.holdout.get(&(index, id));
                for (panel, evaluation) in [
                    ("validation", Some(&validated.matches)),
                    ("holdout", holdout),
                ] {
                    if let Some(evaluation) = evaluation {
                        for game in &evaluation.games {
                            games.serialize((
                                panel,
                                &run.directory,
                                checkpoint.episode,
                                game.shuffle_seed,
                                game.learner_seat,
                                game.opponent,
                                game.outcome,
                            ))?;
                        }
                    }
                }
                for score in &validated.tactics {
                    tactics.serialize(score)?;
                }
                checkpoints.push(json!({
                    "checkpoint": checkpoint,
                    "sha256": validated.sha256,
                    "target_decisions": self.targets.iter().zip(&run.sampled).filter_map(|(&target, &sampled)| (sampled == Some(id)).then_some(target)).collect::<Vec<_>>(),
                    "validation": validated.matches.totals,
                    "holdout": holdout.map(|result| result.totals),
                    "tactics": tactics::TACTICS.map(|name| tactics::counts(&validated.tactics, name)),
                }));
            }
            reports.push(json!({
                "arm": self.config.arms[run.arm].directory,
                "seed": run.seed,
                "directory": run.directory,
                "saved_checkpoints": run.checkpoints.len(),
                "best_episode": run.checkpoints[best].episode,
                "last_episode": run.checkpoints[last].episode,
                "checkpoints": checkpoints,
            }));
        }
        games.flush()?;
        tactics.flush()?;
        write_json(&directory.join("config.json"), self.config)?;
        statistics::write_comparisons(&directory.join("summary.csv"), self.comparisons)?;
        let revision = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned());
        write_json(
            &directory.join("summary.json"),
            &json!({
            "command": std::env::args_os().map(|arg| arg.to_string_lossy().into_owned()).collect::<Vec<_>>(),
                "revision": revision,
                "binary_sha256": sha256(&std::env::current_exe()?)?,
                "evaluation_version": EVALUATION_VERSION,
                "reuse": reuse,
                "worker_threads": jobs,
                "target_decisions": self.targets,
                "runs": reports,
                "holdout_comparisons": self.comparisons,
            }),
        )?;
        self.write_figures(directory)?;
        Ok(())
    }
}
