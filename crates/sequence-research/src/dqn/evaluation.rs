pub mod report;

use std::{cell::Cell, fs, num::NonZeroUsize, path::Path};

use burn::tensor::Device;
use clap::Args;
use plotters::prelude::*;
use sequence_agents::Policy;
use sequence_game::{Action, Player, PlayerView};
use sequence_match::{EvaluationConfig, evaluate as evaluate_matches};
use serde::{Deserialize, Serialize};

use crate::{
    Result,
    experiment::{AgentKind, GameEnd},
};

use super::{
    Evaluation, last_seed,
    model::{GreedyPolicy, Network, ParameterCounts},
};

pub(super) const OPPONENTS: [(AgentKind, &str); 2] = [
    (AgentKind::Random, "Random"),
    (AgentKind::Heuristic, "Heuristic"),
];

#[derive(Args, Default)]
pub(super) struct MatchOptions {
    /// Count turns with a legal first-sequence or immediate winning move and how often it is taken
    #[arg(long)]
    pub(super) tactics: bool,
    /// Take an immediate win when available and report tactical counts
    #[arg(long)]
    pub(super) take_immediate_wins: bool,
}

#[derive(Clone, Copy, Default, Deserialize, Serialize)]
struct Tactics {
    take_immediate_wins: bool,
    first_sequence_opportunities: usize,
    first_sequences_taken: usize,
    immediate_win_opportunities: usize,
    immediate_wins_taken: usize,
}

struct EvaluationPolicy<'a> {
    greedy: GreedyPolicy<'a>,
    tactics: Option<&'a Cell<Tactics>>,
}

impl Policy for EvaluationPolicy<'_> {
    fn choose_action(&mut self, view: &PlayerView<'_>) -> Option<Action> {
        let mut selected = self.greedy.choose_action(view)?;
        let Some(tactics) = self.tactics else {
            return Some(selected);
        };

        let mut counts = tactics.get();
        let needs_first_sequence = view.claimed_sequences[view.player.index()].is_empty();
        let mut first_sequence_available = false;
        let mut first_sequence_taken = false;
        let mut winning_move = None;
        let mut immediate_win_taken = false;

        for &action in &view.legal_actions {
            let Some(preview) = view.preview_play(action) else {
                continue;
            };
            if needs_first_sequence && !preview.new_claims.is_empty() {
                first_sequence_available = true;
                first_sequence_taken |= action == selected;
            }
            if preview.wins {
                winning_move.get_or_insert(action);
                immediate_win_taken |= action == selected;
            }
        }

        if counts.take_immediate_wins
            && let Some(action) = winning_move
        {
            selected = action;
            immediate_win_taken = true;
            first_sequence_taken = needs_first_sequence;
        }

        counts.first_sequence_opportunities += usize::from(first_sequence_available);
        counts.first_sequences_taken += usize::from(first_sequence_taken);
        counts.immediate_win_opportunities += usize::from(winning_move.is_some());
        counts.immediate_wins_taken += usize::from(immediate_win_taken);
        tactics.set(counts);
        Some(selected)
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct Totals {
    pub opponent: AgentKind,
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
    pub cutoffs: usize,
    pub win_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tactics: Option<Tactics>,
}

#[derive(Deserialize, Serialize)]
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

#[derive(Clone, Copy, Serialize)]
pub(super) struct GameRow {
    pub shuffle_seed: u64,
    pub learner_seat: usize,
    pub opponent: AgentKind,
    pub outcome: GameEnd,
}

pub(super) fn evaluate(
    network: &Network,
    device: &Device,
    settings: &Evaluation,
    action_limit: NonZeroUsize,
) -> Result<[Totals; 2]> {
    evaluate_with_games(
        network,
        device,
        settings,
        action_limit,
        &MatchOptions::default(),
        |_| Ok(()),
    )
}

pub(super) fn evaluate_with_games(
    network: &Network,
    device: &Device,
    settings: &Evaluation,
    action_limit: NonZeroUsize,
    options: &MatchOptions,
    mut game_row: impl FnMut(GameRow) -> Result<()>,
) -> Result<[Totals; 2]> {
    let last = last_seed(settings.first_shuffle_seed, settings.deal_count)?;
    let mut evaluation = EvaluationConfig {
        policy_ids: ["dqn".into(), String::new()],
        shuffle_seeds: (settings.first_shuffle_seed..=last).collect(),
        starting_player: Player::One,
        policy_seed: settings.policy_seed,
        action_limit,
        capture_replays: false,
    };
    let mut versus = |opponent: AgentKind| -> Result<_> {
        evaluation.policy_ids[1] = opponent.to_string();
        let tactics = (options.tactics || options.take_immediate_wins).then(|| {
            Cell::new(Tactics {
                take_immediate_wins: options.take_immediate_wins,
                ..Tactics::default()
            })
        });
        let report = evaluate_matches(
            &evaluation,
            |_| EvaluationPolicy {
                greedy: GreedyPolicy { network, device },
                tactics: tactics.as_ref(),
            },
            |seed| opponent.build(seed),
        )?;
        for pair in &report.pairs {
            for (seat, game) in pair.iter().enumerate() {
                game_row(GameRow {
                    shuffle_seed: game.config.game.shuffle_seed,
                    learner_seat: seat + 1,
                    opponent,
                    outcome: game.end.into(),
                })?;
            }
        }
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
            tactics: tactics.map(Cell::into_inner),
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
