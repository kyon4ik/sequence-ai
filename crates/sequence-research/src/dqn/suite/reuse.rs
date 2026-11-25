use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    Result,
    experiment::{AgentKind, GameEnd},
};

use super::super::Evaluation;
use super::{
    Config, EVALUATION_VERSION, Evaluations, GameRow, Matches, OPPONENTS, Run, Score, Totals,
    Validated, read_config, tactics::Position,
};

pub(super) struct PriorReport<'a> {
    directory: &'a Path,
    summary: SavedSummary,
}

#[derive(Deserialize)]
struct SavedSummary {
    evaluation_version: Option<u32>,
    binary_sha256: String,
    target_decisions: Vec<usize>,
    runs: Vec<SavedRun>,
}

#[derive(Deserialize)]
struct SavedRun {
    arm: String,
    seed: u64,
    directory: PathBuf,
    checkpoints: Vec<SavedCheckpoint>,
}

#[derive(Deserialize)]
struct SavedCheckpoint {
    checkpoint: SavedEpisode,
    sha256: String,
    validation: [Totals; 2],
    holdout: Option<[Totals; 2]>,
}

#[derive(Deserialize)]
struct SavedEpisode {
    episode: usize,
}

#[derive(Serialize)]
pub(super) struct Provenance<'a> {
    directory: &'a Path,
    binary_sha256: &'a str,
    validation_checkpoints: usize,
    holdout_checkpoints: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Panel {
    Validation,
    Holdout,
}

#[derive(Deserialize)]
struct SavedGame {
    panel: Panel,
    run: PathBuf,
    episode: usize,
    shuffle_seed: u64,
    learner_seat: usize,
    opponent: AgentKind,
    outcome: GameEnd,
}

impl<'a> PriorReport<'a> {
    pub(super) fn open(directory: &'a Path, config: &Config) -> Result<Self> {
        let saved: Config = read_config(&directory.join("config.json"))?;
        if saved.device != config.device
            || saved.action_limit != config.action_limit
            || saved.validation != config.validation
            || saved.holdout != config.holdout
            || saved.tactics != config.tactics
        {
            return Err(format!(
                "{} uses different evaluation panels, device, or action limit. Match the saved config to reuse results.",
                directory.display(),
            ).into());
        }

        let summary: SavedSummary = read_config(&directory.join("summary.json"))?;
        if summary.evaluation_version.unwrap_or(1) != EVALUATION_VERSION {
            return Err(format!(
                "{} uses an incompatible evaluation version",
                directory.display()
            )
            .into());
        }
        if summary.target_decisions.is_empty()
            || !summary.target_decisions.is_sorted_by(|a, b| a < b)
        {
            return Err(format!(
                "{} requires increasing learner-decision targets",
                directory.display()
            )
            .into());
        }
        Ok(Self { directory, summary })
    }

    pub(super) fn targets(&self) -> &[usize] {
        &self.summary.target_decisions
    }

    pub(super) fn provenance(
        &self,
        validation_checkpoints: usize,
        holdout_checkpoints: usize,
    ) -> Provenance<'_> {
        Provenance {
            directory: self.directory,
            binary_sha256: &self.summary.binary_sha256,
            validation_checkpoints,
            holdout_checkpoints,
        }
    }

    pub(super) fn load(
        &self,
        config: &Config,
        runs: &[Run],
        hashes: &Evaluations<String>,
        positions: &[Position],
    ) -> Result<(Evaluations<Validated>, Evaluations<Matches>)> {
        let current: BTreeMap<_, _> = runs
            .iter()
            .enumerate()
            .map(|(index, run)| ((config.arms[run.arm].directory.as_str(), run.seed), index))
            .collect();
        let mut lookup = BTreeMap::new();
        let mut validation = Evaluations::new();
        let mut holdout = Evaluations::new();
        for run in &self.summary.runs {
            let Some(&index) = current.get(&(run.arm.as_str(), run.seed)) else {
                continue;
            };
            for saved in &run.checkpoints {
                let episode = saved.checkpoint.episode;
                let Ok(id) = runs[index]
                    .checkpoints
                    .binary_search_by_key(&episode, |cp| cp.episode)
                else {
                    continue;
                };
                let key = (index, id);
                if hashes.get(&key) != Some(&saved.sha256) {
                    continue;
                }
                let previous = validation.insert(
                    key,
                    Validated {
                        matches: Matches {
                            totals: saved.validation,
                            games: Vec::new(),
                        },
                        tactics: Vec::new(),
                        sha256: saved.sha256.clone(),
                    },
                );
                if previous.is_some()
                    || lookup
                        .insert((run.directory.clone(), episode), key)
                        .is_some()
                {
                    return Err(format!(
                        "{} contains duplicate cached checkpoints",
                        self.directory.display()
                    )
                    .into());
                }
                if let Some(totals) = saved.holdout {
                    holdout.insert(
                        key,
                        Matches {
                            totals,
                            games: Vec::new(),
                        },
                    );
                }
            }
        }

        if lookup.is_empty() {
            return Ok((validation, holdout));
        }
        let path = self.directory.join("games.csv");
        let mut games = csv::Reader::from_path(&path)?;
        for row in games.deserialize::<SavedGame>() {
            let row = row.map_err(|error| format!("{}: {error}", path.display()))?;
            let Some(key) = lookup.get(&(row.run, row.episode)) else {
                continue;
            };
            let matches = match row.panel {
                Panel::Validation => Some(&mut validation.get_mut(key).unwrap().matches),
                Panel::Holdout => holdout.get_mut(key),
            };
            if let Some(matches) = matches {
                matches.games.push(GameRow {
                    shuffle_seed: row.shuffle_seed,
                    learner_seat: row.learner_seat,
                    opponent: row.opponent,
                    outcome: row.outcome,
                });
            }
        }

        let path = self.directory.join("tactical-positions.csv");
        let mut tactics = csv::Reader::from_path(&path)?;
        for row in tactics.deserialize::<Score>() {
            let mut score = row.map_err(|error| format!("{}: {error}", path.display()))?;
            let episode = score
                .episode
                .ok_or("cached tactic is missing its checkpoint episode")?;
            let key = (score.run, episode);
            if let Some(&(index, id)) = lookup.get(&key) {
                score.run = runs[index].directory.clone();
                validation
                    .get_mut(&(index, id))
                    .unwrap()
                    .tactics
                    .push(score);
            }
        }

        for (&(index, id), result) in &validation {
            let checkpoint = &runs[index].checkpoints[id];
            if !valid_games(&result.matches, &config.validation)
                || result.tactics.len() != positions.len()
                || !result
                    .tactics
                    .iter()
                    .zip(positions)
                    .all(|(score, position)| score.matches_position(position))
            {
                return Err(format!(
                    "{}: incomplete or inconsistent cached validation for {}",
                    self.directory.display(),
                    checkpoint.path.display()
                )
                .into());
            }
        }
        for (&(index, id), result) in &holdout {
            if !valid_games(result, &config.holdout) {
                let checkpoint = &runs[index].checkpoints[id];
                return Err(format!(
                    "{}: incomplete or inconsistent cached holdout for {}",
                    self.directory.display(),
                    checkpoint.path.display()
                )
                .into());
            }
        }
        Ok((validation, holdout))
    }
}

fn valid_games(matches: &Matches, panel: &Evaluation) -> bool {
    let deals = panel.deal_count.get();
    if matches.games.len() != 4 * deals {
        return false;
    }

    matches
        .totals
        .iter()
        .zip(OPPONENTS)
        .zip(matches.games.chunks_exact(2 * deals))
        .all(|((total, (opponent, _)), games)| {
            let mut outcomes = [0usize; 4];
            for (index, game) in games.iter().enumerate() {
                if game.opponent != opponent
                    || game.shuffle_seed != panel.first_shuffle_seed + (index / 2) as u64
                    || game.learner_seat != index % 2 + 1
                {
                    return false;
                }
                let outcome = match (game.outcome, game.learner_seat) {
                    (GameEnd::PlayerOneWin, 1) | (GameEnd::PlayerTwoWin, 2) => 0,
                    (GameEnd::Draw, _) => 1,
                    (GameEnd::ActionLimit, _) => 3,
                    _ => 2,
                };
                outcomes[outcome] += 1;
            }
            let [wins, draws, losses, cutoffs] = outcomes;
            let completed = wins + draws + losses;
            total.opponent == opponent
                && [total.wins, total.draws, total.losses, total.cutoffs] == outcomes
                && total.win_rate == (completed > 0).then(|| wins as f64 / completed as f64)
                && games.len() == completed + cutoffs
        })
}
