use std::{fs, io, num::NonZeroUsize, path::PathBuf};

use burn::tensor::Device;
use clap::Args;
use sequence_agents::{Policy, RandomAgent};
use sequence_game::{Game, GameConfig, Player};
use serde::Serialize;

use super::{
    Backend, last_seed,
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

struct Position {
    deal: u64,
    turn: usize,
    seat: usize,
    tactic: &'static str,
    observation: Observation,
    candidates: Vec<(usize, bool)>,
}

fn positions(options: &Options) -> Result<Vec<Position>> {
    let mut positions = Vec::new();
    for deal in options.first_shuffle_seed..=last_seed(options.first_shuffle_seed, options.deals)? {
        let mut game = Game::new(GameConfig {
            shuffle_seed: deal,
            starting_player: Player::One,
            policies: ["random".into(), "random".into()],
        });
        let mut agents: [RandomAgent; 2] = std::array::from_fn(|seat| {
            let offset = (deal - options.first_shuffle_seed).wrapping_mul(2);
            RandomAgent::new(
                options
                    .policy_seed
                    .wrapping_add(offset)
                    .wrapping_add(seat as u64),
            )
        });
        // Give each deal at most one vote per seat and tactic.
        let mut seen = [[false; 2]; 2];
        for turn in 1..=512 {
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
                for (i, tactic) in ["immediate_win", "first_sequence"].into_iter().enumerate() {
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

#[derive(Serialize)]
struct Score<'a> {
    run: &'a std::path::Path,
    episode: usize,
    shuffle_seed: u64,
    action_number: usize,
    seat: usize,
    tactic: &'a str,
    legal_actions: usize,
    tactical_actions: usize,
    best_tactical_rank: usize,
    best_tactical_q: f32,
    best_alternative_q: f32,
    gap: f32,
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

fn score(
    network: &Network,
    device: &Device,
    positions: &[Position],
    run: &std::path::Path,
    episode: usize,
    writer: &mut csv::Writer<impl io::Write>,
) -> Result<()> {
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
            writer.serialize(Score {
                run,
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
            })?;
        }
    }
    Ok(())
}

pub fn run(options: Options) -> Result<()> {
    let positions = positions(&options)?;
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
            score(
                &Network::load(&path, &device)?,
                &device,
                &positions,
                run,
                episode,
                &mut writer,
            )?;
        }
        writer.flush()?;
    }
    Ok(())
}
