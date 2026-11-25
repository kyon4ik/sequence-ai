use std::{
    collections::BTreeSet,
    fs::{self, File},
    io,
    path::{Path, PathBuf},
};

use clap::Args;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{Checkpoint, MatchOptions, OPPONENTS, Totals, evaluate_with_games};
use crate::{Result, experiment::AgentKind};

use crate::dqn::{
    Backend, Evaluation, SavedEvaluation,
    figures::{self, Axis, Curve, Series},
    model::Network,
    read_config, sampling, tactics,
};

#[derive(Args)]
pub struct Options {
    /// New report directory
    #[arg(short, long, default_value = "results/dqn-evaluation")]
    output: PathBuf,
    /// Write per-game results to a new CSV
    #[arg(long)]
    games: Option<PathBuf>,
    /// Percent of a run's checkpoints to sample (1–100, default 100). Keeps both endpoints.
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=100))]
    checkpoint_percent: Option<u8>,
    #[command(flatten)]
    matches: MatchOptions,
}

#[derive(Deserialize)]
struct Settings {
    #[serde(flatten)]
    matches: SavedEvaluation,
    tactics: Option<Evaluation>,
    checkpoint_percent: Option<u8>,
}

#[derive(Serialize)]
struct Weights {
    path: PathBuf,
    episode: Option<usize>,
    decisions: Option<usize>,
}

impl Weights {
    fn new(path: PathBuf) -> Self {
        let episode = path
            .file_stem()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_prefix("episode-"))
            .and_then(|number| number.parse().ok());
        Self {
            path,
            episode,
            decisions: None,
        }
    }
}

#[derive(Serialize)]
struct Evaluated {
    checkpoint: Weights,
    evaluation: [Totals; 2],
    tactics: [tactics::Counts; 2],
}

pub fn run(input: &Path, config: &Path, backend: Option<Backend>, options: Options) -> Result<()> {
    let Options {
        output,
        games,
        checkpoint_percent,
        matches,
    } = options;
    if output.try_exists()? {
        return Err(format!(
            "{} already exists. Choose a new --output directory.",
            output.display()
        )
        .into());
    }
    let config: Settings = read_config(config)?;
    let checkpoint_percent = checkpoint_percent
        .or(config.checkpoint_percent)
        .unwrap_or(100);
    if !(1..=100).contains(&checkpoint_percent) {
        return Err("checkpoint_percent must be from 1 to 100".into());
    }
    let settings = config.matches;
    let tactical_panel = config.tactics.as_ref().unwrap_or(&settings.evaluation);
    for panel in [&settings.evaluation, tactical_panel] {
        panel.validate_deals(settings.first_shuffle_seed, settings.episodes)?;
    }
    let is_run = input.is_dir();
    let run = if is_run {
        Some(input)
    } else {
        input
            .parent()
            .filter(|parent| parent.ends_with("checkpoints"))
            .and_then(Path::parent)
    };
    let checkpoints = checkpoints(input, run, is_run)?;
    let checkpoints = sample_checkpoints(checkpoints, checkpoint_percent);
    let backend = backend
        .or(settings.inference_device)
        .unwrap_or(settings.device);
    let device = backend.device()?;
    let positions = tactics::positions(tactical_panel, settings.action_limit)?;

    let parent = parent_directory(&output);
    fs::create_dir_all(parent)?;
    let staging = tempfile::tempdir_in(parent)?;
    let directory = staging.path();
    let mut tactical_csv = tactics::writer(&directory.join("tactical-positions.csv"))?;
    let mut game_file = games
        .as_deref()
        .map(|path| -> Result<_> {
            if path.try_exists()? {
                return Err(format!(
                    "{} already exists. Choose a new --games file.",
                    path.display()
                )
                .into());
            }
            Ok(tempfile::NamedTempFile::new_in(parent_directory(path))?)
        })
        .transpose()?;
    let mut game_csv = game_file.as_mut().map(|file| {
        csv::WriterBuilder::new()
            .has_headers(!is_run)
            .from_writer(file.as_file_mut())
    });
    if is_run && let Some(writer) = &mut game_csv {
        writer.write_record([
            "episode",
            "shuffle_seed",
            "learner_seat",
            "opponent",
            "outcome",
        ])?;
    }

    let mut results = Vec::with_capacity(checkpoints.len());
    for checkpoint in checkpoints {
        let network = Network::load(&checkpoint.path, &device)?;
        let evaluation = evaluate_with_games(
            &network,
            &device,
            &settings.evaluation,
            settings.action_limit,
            &matches,
            |game| {
                if let Some(writer) = &mut game_csv {
                    if is_run {
                        writer.serialize((checkpoint.episode, game))?;
                    } else {
                        writer.serialize(game)?;
                    }
                }
                Ok(())
            },
        )?;
        let scores = tactics::score(
            &network,
            &device,
            &positions,
            run.unwrap_or(input),
            checkpoint.episode,
        )?;
        for score in &scores {
            tactical_csv.serialize(score)?;
        }
        results.push(Evaluated {
            checkpoint,
            evaluation,
            tactics: tactics::TACTICS.map(|tactic| tactics::counts(&scores, tactic)),
        });
    }
    if let Some(writer) = &mut game_csv {
        writer.flush()?;
    }
    drop(game_csv);
    tactical_csv.flush()?;

    let report = serde_json::to_string_pretty(&json!({
        "input": input,
        "device": backend,
        "action_limit": settings.action_limit,
        "evaluation_panel": settings.evaluation,
        "tactics_panel": tactical_panel,
        "checkpoint_percent": checkpoint_percent,
        "checkpoints": results,
    }))?;
    fs::write(directory.join("summary.json"), format!("{report}\n"))?;
    write_figures(
        directory,
        input,
        settings.evaluation.deal_count.get(),
        &results,
    )?;
    if let Some((file, path)) = game_file.zip(games) {
        file.persist_noclobber(&path)
            .map_err(|error| format!("could not publish games to {}: {error}", path.display()))?;
    }
    fs::rename(directory, &output)
        .map_err(|error| format!("could not publish results to {}: {error}", output.display()))?;

    eprintln!("Results saved to {}", output.display());
    if is_run {
        println!("{report}");
    } else {
        println!("{}", serde_json::to_string_pretty(&results[0].evaluation)?);
    }
    Ok(())
}

fn parent_directory(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

fn checkpoints(input: &Path, run: Option<&Path>, is_run: bool) -> Result<Vec<Weights>> {
    let mut weights = Vec::new();
    if is_run {
        for entry in fs::read_dir(input.join("checkpoints"))? {
            let path = entry?.path();
            if path.extension().is_some_and(|extension| extension == "bpk") {
                let checkpoint = Weights::new(path);
                if checkpoint.episode.is_none() {
                    return Err(
                        format!("invalid checkpoint name {}", checkpoint.path.display()).into(),
                    );
                }
                weights.push(checkpoint);
            }
        }
        weights.sort_unstable_by_key(|checkpoint| checkpoint.episode);
        if weights.is_empty()
            || weights
                .windows(2)
                .any(|pair| pair[0].episode == pair[1].episode)
        {
            return Err(format!(
                "{} requires saved checkpoints with unique episodes",
                input.display()
            )
            .into());
        }
    } else {
        weights.push(Weights::new(input.to_owned()));
    }

    if let Some(run) = run {
        let log = run.join("evaluation.csv");
        match File::open(&log) {
            Ok(file) => {
                let mut reader = csv::Reader::from_reader(file);
                for row in reader.deserialize::<Checkpoint>() {
                    let row = row.map_err(|error| format!("{}: {error}", log.display()))?;
                    if row.opponent == AgentKind::Random
                        && let Ok(index) = weights
                            .binary_search_by_key(&Some(row.episode), |checkpoint| {
                                checkpoint.episode
                            })
                    {
                        weights[index].decisions = Some(row.decisions);
                    }
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("{}: {error}", log.display()).into()),
        }
    }
    Ok(weights)
}

fn sample_checkpoints(weights: Vec<Weights>, percent: u8) -> Vec<Weights> {
    if percent == 100 || weights.len() < 2 {
        return weights;
    }

    let decisions: Option<Vec<_>> = weights
        .iter()
        .map(|checkpoint| checkpoint.decisions)
        .collect();
    let positions = decisions
        .filter(|decisions| decisions.is_sorted())
        .unwrap_or_else(|| {
            weights
                .iter()
                .map(|checkpoint| checkpoint.episode.unwrap())
                .collect()
        });

    let last = weights.len() - 1;
    let targets = sampling::targets(
        positions[0],
        positions[last],
        weights.len(),
        usize::from(percent),
    );
    let mut selected = BTreeSet::from([0, last]);
    if targets.len() > 2 {
        selected.extend(
            targets[1..targets.len() - 1]
                .iter()
                .filter_map(|&target| sampling::nearest_index(positions.iter().copied(), target)),
        );
    }

    weights
        .into_iter()
        .enumerate()
        .filter_map(|(index, checkpoint)| selected.contains(&index).then_some(checkpoint))
        .collect()
}

fn write_figures(directory: &Path, input: &Path, deals: usize, rows: &[Evaluated]) -> Result<()> {
    let axis = if rows.iter().all(|row| row.checkpoint.decisions.is_some()) {
        Axis::Decisions
    } else if rows.iter().all(|row| row.checkpoint.episode.is_some()) {
        Axis::Episode
    } else {
        Axis::Checkpoint
    };
    let label = input
        .file_name()
        .unwrap_or(input.as_os_str())
        .to_string_lossy();
    let series = |rate: fn(&Evaluated, usize) -> Option<f64>| Series {
        label: &label,
        panels: std::array::from_fn(|panel| Curve {
            line: rows
                .iter()
                .enumerate()
                .map(|(index, row)| {
                    let x = match axis {
                        Axis::Decisions => row.checkpoint.decisions.unwrap(),
                        Axis::Episode => row.checkpoint.episode.unwrap(),
                        Axis::Checkpoint => index + 1,
                    };
                    (x, rate(row, panel))
                })
                .collect(),
            samples: Vec::new(),
        }),
    };
    figures::write_curves(
        &directory.join("win-rates.svg"),
        "Held-out win rates",
        &format!("{deals} deals · both seats"),
        OPPONENTS.map(|(_, label)| label.to_owned()),
        axis,
        "Win rate (completed games)",
        &[series(|row, panel| row.evaluation[panel].win_rate)],
    )?;
    figures::write_curves(
        &directory.join("tactics.svg"),
        "Tactical choices",
        "Fixed positions from Random vs Random games",
        std::array::from_fn(|panel| {
            let label = tactics::TACTICS[panel].label();
            let count = rows[0].tactics[panel].positions;
            format!("{label} · {count} positions")
        }),
        axis,
        "Tactical action ranked first",
        &[series(|row, panel| row.tactics[panel].rate())],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_by_decisions_or_falls_back_to_episodes() {
        let selected_episodes = |decisions: [Option<usize>; 5], percent| {
            let weights = decisions
                .into_iter()
                .enumerate()
                .map(|(episode, decisions)| Weights {
                    path: PathBuf::new(),
                    episode: Some(episode),
                    decisions,
                })
                .collect();
            sample_checkpoints(weights, percent)
                .into_iter()
                .map(|checkpoint| checkpoint.episode.unwrap())
                .collect::<Vec<_>>()
        };

        let decisions = [Some(0), Some(40), Some(60), Some(62), Some(100)];
        assert_eq!(selected_episodes(decisions, 60), [0, 1, 4]);
        assert_eq!(selected_episodes(decisions, 1), [0, 4]);
        assert_eq!(selected_episodes(decisions, 100), [0, 1, 2, 3, 4]);
        assert_eq!(selected_episodes([None; 5], 60), [0, 2, 4]);
        assert_eq!(selected_episodes([Some(0); 5], 60), [0, 4]);
    }
}
