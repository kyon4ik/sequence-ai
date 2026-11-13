//! Pairwise summaries derived from saved reports; no models or replays required.
use std::{fmt::Write as _, path::Path, time::Duration};

use anyhow::{Result, bail};
use clap::ValueEnum;
use sequence_agent::AgentDescriptor;
use sequence_core::GameOutcome;
use sequence_runner::{MatchMetrics, Termination};
use serde::{Deserialize, Serialize};

use crate::{
    artifacts::{Store, read_json},
    validate_id,
};

#[derive(Clone, Copy, ValueEnum)]
pub enum Format {
    Table,
    Json,
    Csv,
}

#[derive(Deserialize)]
struct AgentEntry {
    descriptor: AgentDescriptor,
}

#[derive(Deserialize)]
struct GameEntry {
    players: [usize; 2],
    termination: Termination,
    metrics: MatchMetrics,
}

#[derive(Deserialize)]
struct Run {
    id: String,
    agents: Vec<AgentEntry>,
    games: Vec<GameEntry>,
}

#[derive(Serialize)]
pub struct Agent {
    pub index: usize,
    pub id: String,
    pub name: String,
}

#[derive(Default, Serialize)]
struct Counts {
    games: usize,
    wins: usize,
    losses: usize,
    draws: usize,
    truncated: usize,
    aborted: usize,
}

impl Counts {
    fn record(&mut self, termination: &Termination, seat: usize) {
        self.games += 1;
        match termination {
            Termination::Completed(GameOutcome::Won(color)) if *color as usize == seat => {
                self.wins += 1
            }
            Termination::Completed(GameOutcome::Won(_)) => self.losses += 1,
            Termination::Completed(GameOutcome::Draw) => self.draws += 1,
            Termination::ActionLimit => self.truncated += 1,
            Termination::Aborted { .. } => self.aborted += 1,
            _ => unreachable!("outcomes are validated before aggregation"),
        }
    }

    fn finish(self) -> Results {
        let completed = self.wins + self.losses + self.draws;
        let win_rate = (completed > 0).then(|| self.wins as f64 / completed as f64);
        let score_rate = (completed > 0)
            .then(|| (self.wins as f64 + 0.5 * self.draws as f64) / completed as f64);
        let win_rate_ci95 = win_rate.map(|p| {
            let n = completed as f64;
            let z = 1.959963984540054;
            let scale = 1. + z * z / n;
            let center = (p + z * z / (2. * n)) / scale;
            let half = z * (p * (1. - p) / n + z * z / (4. * n * n)).sqrt() / scale;
            [(center - half).max(0.), (center + half).min(1.)]
        });
        Results {
            counts: self,
            completed,
            win_rate,
            score_rate,
            win_rate_ci95,
        }
    }
}

#[derive(Serialize)]
pub struct Results {
    #[serde(flatten)]
    counts: Counts,
    completed: usize,
    win_rate: Option<f64>,
    score_rate: Option<f64>,
    /// Wilson interval for win probability; draws count as non-wins.
    win_rate_ci95: Option<[f64; 2]>,
}

#[derive(Serialize)]
pub struct Pair {
    #[serde(flatten)]
    results: Results,
    as_red: Results,
    as_blue: Results,
    reward: f64,
    decisions: usize,
    mean_move_ms: Option<f64>,
    max_move_ms: f64,
    mean_actions: f64,
    mean_turns: f64,
}

#[derive(Default)]
struct Accumulator {
    counts: Counts,
    seats: [Counts; 2],
    reward: f64,
    decisions: usize,
    decision_time: Duration,
    max_decision_time: Duration,
    actions: usize,
    turns: usize,
}

impl Accumulator {
    fn record(&mut self, game: &GameEntry, seat: usize) {
        self.counts.record(&game.termination, seat);
        self.seats[seat].record(&game.termination, seat);
        let player = &game.metrics.players[seat];
        self.reward += player.reward;
        self.decisions += player.decisions;
        self.decision_time += player.decision_time;
        self.max_decision_time = self.max_decision_time.max(player.max_decision_time);
        self.actions += game.metrics.actions;
        self.turns += game.metrics.turns as usize;
    }

    fn finish(self) -> Pair {
        let games = self.counts.games as f64;
        let [red, blue] = self.seats;
        Pair {
            results: self.counts.finish(),
            as_red: red.finish(),
            as_blue: blue.finish(),
            reward: self.reward,
            decisions: self.decisions,
            mean_move_ms: (self.decisions > 0)
                .then(|| self.decision_time.as_secs_f64() * 1000. / self.decisions as f64),
            max_move_ms: self.max_decision_time.as_secs_f64() * 1000.,
            mean_actions: self.actions as f64 / games,
            mean_turns: self.turns as f64 / games,
        }
    }
}

#[derive(Serialize)]
pub struct PairwiseReport {
    pub run_id: String,
    pub total_games: usize,
    pub agents: Vec<Agent>,
    /// matrix[row][opponent], always from the row agent's perspective.
    /// Unplayed pairs and the diagonal are null, never a fabricated 0% result.
    pub matrix: Vec<Vec<Option<Pair>>>,
}

fn aggregate(run: Run) -> Result<PairwiseReport> {
    let n = run.agents.len();
    let mut totals: Vec<Vec<Option<Accumulator>>> =
        (0..n).map(|_| (0..n).map(|_| None).collect()).collect();
    for (index, game) in run.games.iter().enumerate() {
        let [red, blue] = game.players;
        if red >= n || blue >= n || red == blue {
            bail!(
                "game {} has invalid player indices {:?}",
                index + 1,
                game.players
            );
        }
        if !matches!(
            game.termination,
            Termination::Completed(GameOutcome::Won(_) | GameOutcome::Draw)
                | Termination::ActionLimit
                | Termination::Aborted { .. }
        ) {
            bail!("game {} has an invalid completed outcome", index + 1);
        }
        for seat in 0..2 {
            totals[game.players[seat]][game.players[1 - seat]]
                .get_or_insert_with(Accumulator::default)
                .record(game, seat);
        }
    }
    Ok(PairwiseReport {
        run_id: run.id,
        total_games: run.games.len(),
        agents: run
            .agents
            .into_iter()
            .enumerate()
            .map(|(index, a)| Agent {
                index,
                id: a.descriptor.id,
                name: a.descriptor.name,
            })
            .collect(),
        matrix: totals
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|a| a.map(Accumulator::finish))
                    .collect()
            })
            .collect(),
    })
}

pub fn load(path: &Path) -> Result<PairwiseReport> {
    aggregate(read_json(path)?)
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.into()
    }
}

fn number(value: Option<f64>) -> String {
    value.map(|x| format!("{x:.6}")).unwrap_or_default()
}

pub fn csv(report: &PairwiseReport) -> String {
    let mut output = "agent_index,agent,opponent_index,opponent,games,completed,wins,losses,draws,truncated,aborted,win_rate,score_rate,win_ci95_lower,win_ci95_upper,red_games,red_completed,red_wins,red_losses,red_draws,red_win_rate,blue_games,blue_completed,blue_wins,blue_losses,blue_draws,blue_win_rate,reward,decisions,mean_move_ms,max_move_ms,mean_actions,mean_turns\n".to_string();
    for row in &report.agents {
        for opponent in &report.agents {
            let Some(pair) = &report.matrix[row.index][opponent.index] else {
                continue;
            };
            let r = &pair.results;
            let c = &r.counts;
            let red = &pair.as_red;
            let blue = &pair.as_blue;
            let fields = vec![
                row.index.to_string(),
                csv_field(&row.id),
                opponent.index.to_string(),
                csv_field(&opponent.id),
                c.games.to_string(),
                r.completed.to_string(),
                c.wins.to_string(),
                c.losses.to_string(),
                c.draws.to_string(),
                c.truncated.to_string(),
                c.aborted.to_string(),
                number(r.win_rate),
                number(r.score_rate),
                number(r.win_rate_ci95.map(|ci| ci[0])),
                number(r.win_rate_ci95.map(|ci| ci[1])),
                red.counts.games.to_string(),
                red.completed.to_string(),
                red.counts.wins.to_string(),
                red.counts.losses.to_string(),
                red.counts.draws.to_string(),
                number(red.win_rate),
                blue.counts.games.to_string(),
                blue.completed.to_string(),
                blue.counts.wins.to_string(),
                blue.counts.losses.to_string(),
                blue.counts.draws.to_string(),
                number(blue.win_rate),
                number(Some(pair.reward)),
                pair.decisions.to_string(),
                number(pair.mean_move_ms),
                number(Some(pair.max_move_ms)),
                number(Some(pair.mean_actions)),
                number(Some(pair.mean_turns)),
            ];
            writeln!(output, "{}", fields.join(",")).unwrap();
        }
    }
    output
}

fn table(report: &PairwiseReport) -> String {
    let width = report
        .agents
        .iter()
        .map(|a| format!("{} [{}]", a.id, a.index).len())
        .max()
        .unwrap_or(5)
        .max(5);
    let cell_width = width.max(18);
    let mut output = format!(
        "{}: {} games\nRow vs column: win rate and W/L/D; unfinished games excluded from rates.\n\n",
        report.run_id, report.total_games
    );
    write!(output, "{:width$}", "Agent").unwrap();
    for a in &report.agents {
        write!(
            output,
            " {:>cell_width$}",
            format!("{} [{}]", a.id, a.index)
        )
        .unwrap();
    }
    output.push('\n');
    for a in &report.agents {
        write!(output, "{:width$}", format!("{} [{}]", a.id, a.index)).unwrap();
        for b in &report.agents {
            let value = match &report.matrix[a.index][b.index] {
                Some(pair) => {
                    let r = &pair.results;
                    let rate = r
                        .win_rate
                        .map(|p| format!("{:.1}%", p * 100.))
                        .unwrap_or_else(|| "n/a".into());
                    format!(
                        "{} {}/{}/{}",
                        rate, r.counts.wins, r.counts.losses, r.counts.draws
                    )
                }
                None => "—".into(),
            };
            write!(output, " {value:>cell_width$}").unwrap();
        }
        output.push('\n');
    }
    output.push_str("\nScore rate = (wins + 0.5 × draws) / completed. Use --format csv or json for counts, seat splits and timings.\n");
    output
}

pub fn run(store: &Store, reference: &str, format: Format) -> Result<()> {
    let supplied = Path::new(reference);
    let path = if supplied.is_file() {
        supplied.to_path_buf()
    } else {
        validate_id(reference)?;
        store.root.join("runs").join(reference).join("report.json")
    };
    let report = load(&path)?;
    match format {
        Format::Table => print!("{}", table(&report)),
        Format::Json => println!("{}", serde_json::to_string_pretty(&report)?),
        Format::Csv => print!("{}", csv(&report)),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sequence_core::Color;

    fn fixture() -> Run {
        let agents = ["same", "same", "unplayed"]
            .map(|id| AgentEntry {
                descriptor: AgentDescriptor::new(id, id),
            })
            .into();
        let outcomes = [
            ([0, 1], Termination::Completed(GameOutcome::Won(Color::Red))),
            ([1, 0], Termination::Completed(GameOutcome::Won(Color::Red))),
            (
                [1, 0],
                Termination::Completed(GameOutcome::Won(Color::Blue)),
            ),
            ([0, 1], Termination::Completed(GameOutcome::Draw)),
            ([0, 1], Termination::ActionLimit),
            (
                [1, 0],
                Termination::Aborted {
                    reason: "test".into(),
                },
            ),
        ];
        let games = outcomes
            .into_iter()
            .map(|(players, termination)| {
                let mut metrics = MatchMetrics {
                    actions: 5,
                    turns: 5,
                    ..Default::default()
                };
                for (i, player) in metrics.players.iter_mut().enumerate() {
                    player.decisions = i + 2;
                    player.decision_time = Duration::from_millis(if i == 0 { 4 } else { 9 });
                    player.max_decision_time = Duration::from_millis(if i == 0 { 3 } else { 5 });
                    player.reward = if i == 0 { 1. } else { -1. };
                }
                GameEntry {
                    players,
                    termination,
                    metrics,
                }
            })
            .collect();
        Run {
            id: "test".into(),
            agents,
            games,
        }
    }

    #[test]
    fn seat_swaps_draws_unfinished_games_and_duplicate_ids_are_distinguished() {
        let report = aggregate(fixture()).unwrap();
        let a = report.matrix[0][1].as_ref().unwrap();
        let b = report.matrix[1][0].as_ref().unwrap();
        assert_eq!(a.results.counts.games, 6);
        assert_eq!(
            (
                a.results.counts.wins,
                a.results.counts.losses,
                a.results.counts.draws
            ),
            (2, 1, 1)
        );
        assert_eq!(
            (a.results.counts.truncated, a.results.counts.aborted),
            (1, 1)
        );
        assert_eq!(a.results.completed, 4);
        assert_eq!(a.results.win_rate, Some(0.5));
        assert_eq!(a.results.score_rate, Some(0.625));
        assert_eq!(b.results.score_rate, Some(0.375));
        assert_eq!(b.results.counts.wins, a.results.counts.losses);
        assert_eq!(a.as_red.counts.games, 3);
        assert_eq!(a.as_red.score_rate, Some(0.75));
        assert_eq!(a.as_blue.score_rate, Some(0.5));
        assert_eq!(a.decisions, 15);
        assert_eq!(a.mean_move_ms, Some(2.6));
        assert_eq!(a.max_move_ms, 5.);
        assert!(a.results.win_rate_ci95.unwrap()[0] < 0.5);
        assert!(a.results.win_rate_ci95.unwrap()[1] > 0.5);
        assert!(report.matrix[0][0].is_none());
        assert!(report.matrix[0][2].is_none());
        assert_eq!(csv(&report).lines().count(), 3);
    }

    #[test]
    fn unfinished_pairs_have_no_rate_and_malformed_indices_are_errors() {
        let mut run = fixture();
        run.games
            .retain(|g| !matches!(g.termination, Termination::Completed(_)));
        let report = aggregate(run).unwrap();
        let pair = report.matrix[0][1].as_ref().unwrap();
        assert_eq!(pair.results.completed, 0);
        assert_eq!(pair.results.win_rate, None);
        assert_eq!(pair.results.score_rate, None);
        assert_eq!(pair.results.win_rate_ci95, None);
        for players in [[0, 0], [0, 3]] {
            let mut run = fixture();
            run.games[0].players = players;
            assert!(aggregate(run).is_err());
        }
        let mut run = fixture();
        run.games[0].termination = Termination::Completed(GameOutcome::InProgress);
        assert!(aggregate(run).is_err());
    }
}
