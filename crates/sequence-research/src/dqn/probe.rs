use std::{
    fs, io,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use burn::tensor::Device;
use clap::Args;
use sequence_agents::{Policy, RandomAgent};
use sequence_game::{Game, GameConfig, Player};
use serde::{Deserialize, Serialize};

use super::{
    Backend, Evaluation, last_seed,
    model::Network,
    observation::{ACTIONS, Observation, action_index},
};
use crate::Result;

#[derive(Args)]
pub struct Options {
    /// Training run directories containing checkpoints
    #[arg(required = true)]
    runs: Vec<PathBuf>,
    #[arg(long, default_value_t = 4_000_000)]
    first_shuffle_seed: u64,
    #[arg(long, default_value = "128")]
    deals: NonZeroUsize,
    /// Seat RNG seeds are policy_seed + 2 * deal offset + seat index (wrapping)
    #[arg(long, default_value_t = 2029)]
    policy_seed: u64,
    #[arg(long, value_enum, default_value = "cpu")]
    device: Backend,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Tactic {
    ImmediateWin,
    FirstSequence,
}

pub(super) const TACTICS: [Tactic; 2] = [Tactic::ImmediateWin, Tactic::FirstSequence];

pub(super) struct Position {
    deal: u64,
    turn: usize,
    seat: usize,
    tactic: Tactic,
    observation: Observation,
    candidates: Vec<(usize, bool)>,
}

pub(super) fn positions(
    settings: &Evaluation,
    action_limit: NonZeroUsize,
) -> Result<Vec<Position>> {
    let mut positions = Vec::new();
    for deal in
        settings.first_shuffle_seed..=last_seed(settings.first_shuffle_seed, settings.deal_count)?
    {
        let mut game = Game::new(GameConfig {
            shuffle_seed: deal,
            starting_player: Player::One,
            policies: ["random".into(), "random".into()],
        });
        let mut agents: [RandomAgent; 2] = std::array::from_fn(|seat| {
            let offset = (deal - settings.first_shuffle_seed).wrapping_mul(2);
            RandomAgent::new(
                settings
                    .policy_seed
                    .wrapping_add(offset)
                    .wrapping_add(seat as u64),
            )
        });
        // Give each deal at most one vote per seat and tactic.
        let mut seen = [[false; 2]; 2];
        for turn in 1..=action_limit.get() {
            let Some(actor) = game.current_player() else {
                break;
            };
            let seat = actor.index();
            let view = game.view(actor);
            if seen[seat] != [true; 2] && view.legal_actions.len() > 1 {
                let actions: Vec<_> = view
                    .legal_actions
                    .iter()
                    .filter_map(|&action| {
                        let index = action_index(action)?;
                        let preview = view.preview_play(action);
                        Some((
                            index,
                            [
                                preview.as_ref().is_some_and(|p| p.wins),
                                view.claimed_sequences[seat].is_empty()
                                    && preview.is_some_and(|p| !p.new_claims.is_empty()),
                            ],
                        ))
                    })
                    .collect();
                for (i, tactic) in TACTICS.into_iter().enumerate() {
                    if !seen[seat][i]
                        && actions.iter().any(|(_, label)| label[i])
                        && actions.iter().any(|(_, label)| !label[i])
                    {
                        positions.push(Position {
                            deal,
                            turn,
                            seat: seat + 1,
                            tactic,
                            observation: Observation::new(&view),
                            candidates: actions.iter().map(|(a, label)| (*a, label[i])).collect(),
                        });
                        seen[seat][i] = true;
                    }
                }
            }
            let action = agents[seat].choose_action(&view).expect("active player");
            game.apply(actor, action)?;
        }
    }
    if positions.is_empty() {
        return Err("no tactical positions on the requested deals".into());
    }
    Ok(positions)
}

#[derive(Deserialize, Serialize)]
pub(super) struct Score {
    pub run: PathBuf,
    pub episode: usize,
    shuffle_seed: u64,
    action_number: usize,
    seat: usize,
    pub tactic: Tactic,
    legal_actions: usize,
    tactical_actions: usize,
    pub best_tactical_rank: usize,
    best_tactical_q: f32,
    best_alternative_q: f32,
    gap: f32,
}

impl Score {
    pub(super) fn matches_position(&self, position: &Position) -> bool {
        self.shuffle_seed == position.deal
            && self.action_number == position.turn
            && self.seat == position.seat
            && self.tactic == position.tactic
            && self.legal_actions == position.candidates.len()
            && self.tactical_actions == position.candidates.iter().filter(|(_, t)| *t).count()
            && (1..=self.legal_actions).contains(&self.best_tactical_rank)
            && self.best_tactical_q.is_finite()
            && self.best_alternative_q.is_finite()
            && self.gap == self.best_tactical_q - self.best_alternative_q
    }
}

struct Ranking {
    rank: usize,
    best_tactical_q: f32,
    best_alternative_q: f32,
}

fn rank_actions(scores: &[f32], candidates: &mut [(usize, bool)]) -> Option<Ranking> {
    candidates.sort_by(|(a, _), (b, _)| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b)));
    let rank = candidates.iter().position(|(_, tactical)| *tactical)?;
    let &(alternative, _) = candidates.iter().find(|(_, tactical)| !tactical)?;
    Some(Ranking {
        rank: rank + 1,
        best_tactical_q: scores[candidates[rank].0],
        best_alternative_q: scores[alternative],
    })
}

pub(super) fn score(
    network: &Network,
    device: &Device,
    positions: &[Position],
    run: &Path,
    episode: usize,
) -> Result<Vec<Score>> {
    let mut results = Vec::with_capacity(positions.len());
    for batch in positions.chunks(64) {
        let data = network
            .forward(batch.iter().map(|p| &p.observation), device)
            .into_data();
        for (p, scores) in batch
            .iter()
            .zip(data.as_slice::<f32>()?.as_chunks::<ACTIONS>().0.iter())
        {
            if !scores.iter().all(|x| x.is_finite()) {
                return Err("non-finite probe score".into());
            }
            let mut ranked = p.candidates.clone();
            let Ranking {
                rank,
                best_tactical_q,
                best_alternative_q,
            } = rank_actions(scores, &mut ranked)
                .expect("probe includes a tactic and an alternative");
            results.push(Score {
                run: run.to_owned(),
                episode,
                shuffle_seed: p.deal,
                action_number: p.turn,
                seat: p.seat,
                tactic: p.tactic,
                legal_actions: ranked.len(),
                tactical_actions: ranked.iter().filter(|(_, t)| *t).count(),
                best_tactical_rank: rank,
                best_tactical_q,
                best_alternative_q,
                gap: best_tactical_q - best_alternative_q,
            });
        }
    }
    Ok(results)
}

pub fn run(options: Options) -> Result<()> {
    let settings = Evaluation {
        first_shuffle_seed: options.first_shuffle_seed,
        deal_count: options.deals,
        policy_seed: options.policy_seed,
    };
    let positions = positions(&settings, NonZeroUsize::new(512).unwrap())?;
    let device = options.device.device()?;
    let mut writer = csv::Writer::from_writer(io::stdout().lock());
    for run in &options.runs {
        let mut checkpoints = Vec::new();
        for entry in fs::read_dir(run.join("checkpoints"))? {
            let path = entry?.path();
            if path.extension().is_some_and(|e| e == "bpk") {
                let episode = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.strip_prefix("episode-"))
                    .ok_or("invalid checkpoint name")?
                    .parse::<usize>()?;
                checkpoints.push((episode, path));
            }
        }
        if checkpoints.is_empty() {
            return Err(format!("no checkpoints in {}", run.display()).into());
        }
        checkpoints.sort();
        for (episode, path) in checkpoints {
            for result in score(
                &Network::load(&path, &device)?,
                &device,
                &positions,
                run,
                episode,
            )? {
                writer.serialize(result)?;
            }
        }
        writer.flush()?;
    }
    Ok(())
}
