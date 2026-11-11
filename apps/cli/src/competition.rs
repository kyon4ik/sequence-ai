use std::{
    collections::HashSet,
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{Result, bail};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use sequence_agent::{AgentSpec, InferenceDevice};
use sequence_core::{GameOutcome, Simple};
use sequence_runner::{MatchConfig, MatchRecord, MatchRunner};
use serde::{Deserialize, Serialize};

use crate::{
    artifacts::{Store, read_toml, write_json},
    parse_device,
};

#[derive(Debug, Serialize)]
pub struct Standing {
    pub agent: String,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub truncated: usize,
    pub reward: f64,
    pub decisions: usize,
    pub decision_time: Duration,
    pub max_decision_time: Duration,
    /// Fresh rating for this run; only completed games update Elo.
    pub elo: f64,
}

impl Standing {
    fn new(agent: String, elo: f64) -> Self {
        Self {
            agent,
            wins: 0,
            losses: 0,
            draws: 0,
            truncated: 0,
            reward: 0.,
            decisions: 0,
            decision_time: Duration::ZERO,
            max_decision_time: Duration::ZERO,
            elo,
        }
    }

    fn record(&mut self, record: &MatchRecord, seat: usize) {
        let metrics = &record.metrics.players[seat];
        self.reward += metrics.reward;
        self.decisions += metrics.decisions;
        self.decision_time += metrics.decision_time;
        self.max_decision_time = self.max_decision_time.max(metrics.max_decision_time);
        match record.termination.outcome() {
            Some(GameOutcome::Won(color)) if color as usize == seat => self.wins += 1,
            Some(GameOutcome::Won(_)) => self.losses += 1,
            Some(GameOutcome::Draw) => self.draws += 1,
            _ => self.truncated += 1,
        }
    }
}

#[derive(Serialize)]
struct Report {
    id: String,
    kind: String,
    seed: u64,
    max_actions: usize,
    device: String,
    agents: Vec<AgentSpec>,
    initial_elo: f64,
    elo_k: f64,
    games: Vec<GameEntry>,
    standings: Vec<Standing>,
    /// Includes loading policies and writing replays.
    elapsed: Duration,
}

#[derive(Serialize)]
struct GameEntry {
    replay: String,
    /// Red then Blue, as indices into agents/standings.
    players: [usize; 2],
    termination: sequence_runner::Termination,
    metrics: sequence_runner::MatchMetrics,
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct LeagueConfig {
    agents: Vec<String>,
    games_per_pair: usize,
    seed: u64,
    max_actions: usize,
    device: String,
    initial_elo: f64,
    elo_k: f64,
}

impl Default for LeagueConfig {
    fn default() -> Self {
        Self {
            agents: vec![],
            games_per_pair: 100,
            seed: 42,
            max_actions: 1000,
            device: "cpu".into(),
            initial_elo: 1000.,
            elo_k: 32.,
        }
    }
}

pub fn evaluate(
    store: &Store,
    agents: [&str; 2],
    games: usize,
    seed: u64,
    max_actions: usize,
    device: InferenceDevice,
) -> Result<()> {
    let specs = agents
        .into_iter()
        .map(|name| store.load_agent(name, Path::new(".")))
        .collect::<Result<Vec<_>>>()?;
    // Self-evaluation is valid; indices distinguish the two copies in this run.
    run(
        store,
        "eval",
        specs,
        LeagueConfig {
            games_per_pair: games,
            seed,
            max_actions,
            ..Default::default()
        },
        device,
    )
}

pub fn tournament(store: &Store, path: &Path, device: Option<InferenceDevice>) -> Result<()> {
    let config: LeagueConfig = read_toml(path)?;
    let device = device
        .map(Ok)
        .unwrap_or_else(|| parse_device(&config.device))
        .map_err(anyhow::Error::msg)?;
    let specs = config
        .agents
        .iter()
        .map(|name| store.load_agent(name, path.parent().unwrap_or(Path::new("."))))
        .collect::<Result<Vec<_>>>()?;
    let ids: HashSet<_> = specs.iter().map(|spec| &spec.descriptor.id).collect();
    if ids.len() != specs.len() {
        bail!(
            "tournament agents must have distinct descriptor IDs (greedy and heuristic are aliases)"
        );
    }
    run(store, "tournament", specs, config, device)
}

fn run(
    store: &Store,
    kind: &str,
    specs: Vec<AgentSpec>,
    config: LeagueConfig,
    device: InferenceDevice,
) -> Result<()> {
    if specs.len() < 2 || config.games_per_pair == 0 || config.max_actions == 0 {
        bail!("need at least two agents, positive games_per_pair and positive max_actions");
    }
    if !config.initial_elo.is_finite() || !config.elo_k.is_finite() || config.elo_k <= 0. {
        bail!("initial_elo must be finite and elo_k must be finite and positive");
    }
    let started = Instant::now();
    // Fail before starting the run if any spec cannot construct a policy.
    for spec in &specs {
        spec.config
            .create(config.seed, device)
            .map_err(|e| anyhow::anyhow!("agent {}: {e}", spec.descriptor.id))?;
    }
    let (id, dir) = store.new_run(kind)?;
    let mut report = Report {
        id,
        kind: kind.into(),
        seed: config.seed,
        max_actions: config.max_actions,
        device: match device {
            InferenceDevice::Cpu => "cpu".into(),
            InferenceDevice::Cuda(n) => format!("cuda:{n}"),
        },
        standings: specs
            .iter()
            .map(|spec| Standing::new(spec.descriptor.id.clone(), config.initial_elo))
            .collect(),
        agents: specs,
        initial_elo: config.initial_elo,
        elo_k: config.elo_k,
        games: Vec::new(),
        elapsed: Duration::ZERO,
    };
    // Input manifest survives an interrupted run along with every completed replay.
    write_json(&dir.join("agents.json"), &report.agents)?;
    let mut rng = SmallRng::seed_from_u64(config.seed);
    let mut runner = MatchRunner::<Simple>::default();
    for a in 0..report.agents.len() {
        for b in a + 1..report.agents.len() {
            for game in 0..config.games_per_pair {
                let players = if game % 2 == 0 { [a, b] } else { [b, a] };
                let factories = players.map(|i| report.agents[i].on_device(device));
                let match_config = MatchConfig {
                    game_seed: rng.random(),
                    agent_seeds: [rng.random(), rng.random()],
                    max_actions: config.max_actions,
                };
                let replay = format!("{}-{:06}", report.id, report.games.len() + 1);
                let record =
                    match runner.run_factories(match_config, [&factories[0], &factories[1]]) {
                        Ok(record) => record,
                        Err(error) => {
                            if let Some(partial) = error.partial_record() {
                                store.save_replay(&replay, partial)?;
                            }
                            report.elapsed = started.elapsed();
                            write_json(&dir.join("partial-report.json"), &report)?;
                            return Err(error.into());
                        }
                    };
                store.save_replay(&replay, &record)?;
                for (seat, index) in players.into_iter().enumerate() {
                    report.standings[index].record(&record, seat);
                }
                if let Some(outcome) = record.termination.outcome() {
                    let score = match outcome {
                        GameOutcome::Won(color) if color as usize == 0 => 1.,
                        GameOutcome::Won(_) => 0.,
                        GameOutcome::Draw => 0.5,
                        GameOutcome::InProgress => unreachable!(),
                    };
                    let [red, blue] = players.map(|i| report.standings[i].elo);
                    let delta = elo_delta(red, blue, score, config.elo_k);
                    report.standings[players[0]].elo += delta;
                    report.standings[players[1]].elo -= delta;
                }
                report.games.push(GameEntry {
                    replay,
                    players,
                    termination: record.termination,
                    metrics: record.metrics,
                });
                let n = report.games.len();
                if n == 1 || n.is_multiple_of(10) {
                    eprintln!("{kind}: {n} games recorded");
                }
            }
        }
    }
    report.elapsed = started.elapsed();
    write_json(&dir.join("report.json"), &report)?;
    println!(
        "{}: {} games in {:.2}s",
        report.id,
        report.games.len(),
        report.elapsed.as_secs_f64()
    );
    println!(
        "{:<24} {:>6} {:>6} {:>6} {:>9} {:>9} {:>10} {:>12}",
        "Agent", "Wins", "Losses", "Draws", "Truncated", "Elo", "Reward", "Mean move ms"
    );
    let mut ranked: Vec<_> = report.standings.iter().collect();
    ranked.sort_by(|a, b| b.elo.total_cmp(&a.elo));
    for standing in ranked {
        let mean_ms = if standing.decisions == 0 {
            0.
        } else {
            standing.decision_time.as_secs_f64() * 1000. / standing.decisions as f64
        };
        println!(
            "{:<24} {:>6} {:>6} {:>6} {:>9} {:>9.1} {:>10.1} {:>12.3}",
            standing.agent,
            standing.wins,
            standing.losses,
            standing.draws,
            standing.truncated,
            standing.elo,
            standing.reward,
            mean_ms
        );
    }
    println!("Report: {}", dir.join("report.json").display());
    if let Some(game) = report.games.first() {
        println!(
            "Replay: sequence --data-dir {:?} replay {}",
            store.root, game.replay
        );
    }
    Ok(())
}

fn elo_delta(a: f64, b: f64, score_a: f64, k: f64) -> f64 {
    k * (score_a - 1. / (1. + 10_f64.powf((b - a) / 400.)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn elo_expected_results_and_upsets() {
        assert_eq!(elo_delta(1000., 1000., 1., 32.), 16.);
        assert_eq!(elo_delta(1000., 1000., 0.5, 32.), 0.);
        assert!(elo_delta(800., 1200., 1., 32.) > 16.);
        assert!((elo_delta(1200., 800., 1., 32.) + elo_delta(800., 1200., 0., 32.)).abs() < 1e-12);
    }
}
