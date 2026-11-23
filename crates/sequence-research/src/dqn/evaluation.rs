use std::{fs, path::Path};

use burn::tensor::Device;
use plotters::prelude::*;
use sequence_game::Player;
use sequence_match::{EvaluationConfig, evaluate as evaluate_matches};
use serde::Serialize;

use crate::{Result, experiment::AgentKind};

use super::{
    Config, last_seed,
    model::{GreedyPolicy, Network, ParameterCounts},
};

#[derive(Clone, Copy, Serialize)]
pub struct Totals {
    pub opponent: AgentKind,
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
    pub cutoffs: usize,
    pub win_rate: Option<f64>,
}

#[derive(Serialize)]
pub struct Checkpoint {
    pub episode: usize,
    pub decisions: usize,
    pub updates: usize,
    pub opponent: AgentKind,
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
    pub cutoffs: usize,
    pub win_rate: Option<f64>,
}

#[derive(Serialize)]
struct Summary<'a> {
    model: ParameterCounts,
    latest_evaluation: &'a [Checkpoint],
}

pub fn evaluate(network: &Network, device: &Device, config: &Config) -> Result<[Totals; 2]> {
    let settings = &config.evaluation;
    let last = last_seed(settings.first_shuffle_seed, settings.deal_count)?;
    let mut evaluation = EvaluationConfig {
        policy_ids: ["dqn".into(), String::new()],
        shuffle_seeds: (settings.first_shuffle_seed..=last).collect(),
        starting_player: Player::One,
        policy_seed: settings.policy_seed,
        action_limit: config.action_limit,
        capture_replays: false,
    };
    let mut versus = |opponent: AgentKind| -> Result<_> {
        evaluation.policy_ids[1] = opponent.to_string();
        let report = evaluate_matches(
            &evaluation,
            |_| GreedyPolicy { network, device },
            |seed| opponent.build(seed),
        )?;
        let [wins, losses] = report.totals.wins;
        let draws = report.totals.draws;
        let completed = wins + losses + draws;
        Ok(Totals {
            opponent,
            wins,
            draws,
            losses,
            cutoffs: report.totals.truncations,
            win_rate: (completed > 0).then(|| wins as f64 / completed as f64),
        })
    };
    Ok([versus(AgentKind::Random)?, versus(AgentKind::Heuristic)?])
}

pub fn write_report(directory: &Path, model: ParameterCounts, rows: &[Checkpoint]) -> Result<()> {
    let summary = Summary {
        model,
        latest_evaluation: rows
            .chunk_by(|left, right| left.episode == right.episode)
            .next_back()
            .unwrap_or_default(),
    };
    fs::write(
        directory.join("summary.json"),
        serde_json::to_string_pretty(&summary)? + "\n",
    )?;
    plot(
        &directory.join("win-rates.svg"),
        rows,
        model.trainable_parameters,
    )
}

fn plot(path: &Path, rows: &[Checkpoint], trainable_parameters: usize) -> Result<()> {
    let root = SVGBackend::new(path, (720, 420)).into_drawing_area();
    root.fill(&WHITE)?;
    let last = rows.last().map_or(1, |row| row.decisions.max(1));
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("DQN checkpoint evaluation ({trainable_parameters} trainable parameters)"),
            ("sans-serif", 20),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(62)
        .build_cartesian_2d(0..last, 0.0..1.0)?;
    chart
        .configure_mesh()
        .x_desc("Learner decisions")
        .y_desc("Win rate (completed games)")
        .y_label_formatter(&|value| format!("{:.0}%", value * 100.0))
        .draw()?;

    for (opponent, label, color) in [
        (AgentKind::Random, "Random", BLUE),
        (AgentKind::Heuristic, "Heuristic", RED),
    ] {
        let points: Vec<_> = rows
            .iter()
            .filter(|row| row.opponent == opponent)
            .filter_map(|row| Some((row.decisions, row.win_rate?)))
            .collect();
        chart
            .draw_series(std::iter::once(PathElement::new(
                points.clone(),
                color.stroke_width(2),
            )))?
            .label(label)
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 18, y)], color.stroke_width(2)));
        chart.draw_series(
            points
                .into_iter()
                .map(|point| Circle::new(point, 3, color.filled())),
        )?;
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.filled())
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    Ok(())
}
