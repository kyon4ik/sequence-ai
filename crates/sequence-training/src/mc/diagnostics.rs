//! Diagnostics use legal learner observations only. Outcome prediction errors
//! describe the evaluated matchup, not calibration against the training mixture.
use burn::prelude::*;
use sequence_agent::{
    Afterstate, AgentError,
    mc::{EncodedAfterstate, ValueNet, batch},
};
use sequence_core::{Action, Game, Simple};
use sequence_runner::MatchRecord;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ValueBin {
    pub decisions: usize,
    pub prediction_sum: f64,
    pub outcome_sum: f64,
    pub squared_error_sum: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MissedWin {
    pub game_seed: u64,
    pub learner_seat: usize,
    pub turn: u16,
    pub chosen: Action,
    pub winning: Action,
    pub chosen_value: f32,
    pub winning_value: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PolicyDiagnostics {
    pub games: usize,
    pub decisions: usize,
    pub immediate_win_opportunities: usize,
    pub immediate_wins_taken: usize,
    /// Ten bins of chosen predictions in [-1, 1]. Entries are correlated within games.
    pub value_bins: [ValueBin; 10],
    /// Early (<20), middle (20..40), late (>=40) move-number bins.
    pub phase_bins: [ValueBin; 3],
    /// Up to 20 reproducible examples, not an exhaustive error listing.
    pub missed_wins: Vec<MissedWin>,
}

pub(super) fn inspect(
    net: &ValueNet,
    device: &Device,
    records: &[MatchRecord],
) -> Result<PolicyDiagnostics, AgentError> {
    let mut result = PolicyDiagnostics {
        games: records.len(),
        ..Default::default()
    };
    let mut encoded = Vec::new();
    let mut chosen = Vec::new();
    let mut misses = Vec::new();
    for (index, record) in records.iter().enumerate() {
        let seat = index % 2;
        let outcome = record
            .termination
            .outcome()
            .ok_or("diagnostics require completed games")?;
        let target = [1., -1., 0.][super::outcome_index(outcome, seat)];
        let mut game = Simple::from_seed(record.header.config.game_seed);
        let mut actions = Vec::new();
        for step in &record.steps {
            if step.actor as usize == seat {
                game.legal_actions(&mut actions);
                let wins: Vec<_> = actions
                    .iter()
                    .copied()
                    .filter(|&action| {
                        Afterstate::new(game.observation(), action)
                            .unwrap()
                            .board
                            .sequences(step.actor)
                            .len()
                            >= 2
                    })
                    .collect();
                let chosen_index = encoded.len();
                encoded.push(EncodedAfterstate::new(game.observation(), step.action)?);
                chosen.push((chosen_index, target, (game.turns() as usize / 20).min(2)));
                if !wins.is_empty() {
                    result.immediate_win_opportunities += 1;
                    if wins.contains(&step.action) {
                        result.immediate_wins_taken += 1;
                    } else if misses.len() < 20 {
                        let winning_index = encoded.len();
                        encoded.push(EncodedAfterstate::new(game.observation(), wins[0])?);
                        misses.push((
                            chosen_index,
                            winning_index,
                            MissedWin {
                                game_seed: record.header.config.game_seed,
                                learner_seat: seat,
                                turn: game.turns(),
                                chosen: step.action,
                                winning: wins[0],
                                chosen_value: 0.,
                                winning_value: 0.,
                            },
                        ));
                    }
                }
            }
            game.apply(step.action)?;
        }
    }
    let model = net.clone().valid();
    let device = device.clone().without_autodiff();
    let mut values = Vec::with_capacity(encoded.len());
    for chunk in encoded.chunks(1024) {
        let (board, context) = batch(chunk, &device);
        values.extend(
            model
                .forward(board, context)
                .try_into_data()?
                .try_to_vec::<f32>()?,
        );
    }
    if values.len() != encoded.len() || values.iter().any(|v| !v.is_finite()) {
        return Err("invalid diagnostic predictions".into());
    }
    for (index, target, phase) in chosen {
        let value = values[index] as f64;
        result.decisions += 1;
        let bin = (((value + 1.) * 5.).floor() as usize).min(9);
        for entry in [&mut result.value_bins[bin], &mut result.phase_bins[phase]] {
            entry.decisions += 1;
            entry.prediction_sum += value;
            entry.outcome_sum += target;
            entry.squared_error_sum += (value - target).powi(2);
        }
    }
    for (chosen, winning, mut example) in misses {
        example.chosen_value = values[chosen];
        example.winning_value = values[winning];
        result.missed_wins.push(example);
    }
    Ok(result)
}
