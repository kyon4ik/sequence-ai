use burn::prelude::*;
use sequence_agent::mc::{McBatchPolicy, McDecision, McInferenceMetrics, McPolicyState, ValueNet};
use sequence_agent::{
    Agent, AgentConfig, AgentDescriptor, AgentError, EpisodeContext, InferenceDevice,
};
use sequence_core::{Action, Color, Game, Simple};
use sequence_runner::{MatchConfig, MatchRecord, MatchSession};
use std::time::{Duration, Instant};

struct ActiveMatch {
    index: usize,
    session: MatchSession<Simple>,
    states: [McPolicyState; 2],
    opponent: Option<Box<dyn Agent>>,
    neural_seat: Option<usize>,
    actions: Vec<Action>,
}

impl ActiveMatch {
    fn new(
        index: usize,
        config: &MatchConfig,
        epsilon: f64,
        baseline: Option<&AgentConfig>,
    ) -> Result<Self, AgentError> {
        let neural_seat = baseline.map(|_| index % 2);
        let opponent = if let Some(baseline) = baseline {
            let seat = 1 - neural_seat.unwrap();
            let mut agent = baseline.create(config.agent_seeds[seat], InferenceDevice::Cpu)?;
            agent.on_episode_start(EpisodeContext {
                game_seed: config.game_seed,
                agent_seed: config.agent_seeds[seat],
                color: [Color::Red, Color::Blue][seat],
            });
            Some(agent)
        } else {
            None
        };
        let mut players = [
            AgentDescriptor::new("self-play", "MC"),
            AgentDescriptor::new("self-play", "MC"),
        ];
        if let Some(seat) = neural_seat {
            players[seat] = AgentDescriptor::new("candidate", "Candidate");
            players[1 - seat] = AgentDescriptor::new("baseline", "Baseline");
        }
        Ok(Self {
            index,
            session: MatchSession::new(config.clone(), players)?,
            states: config
                .agent_seeds
                .map(|seed| McPolicyState::new(seed, epsilon)),
            opponent,
            neural_seat,
            actions: Vec::new(),
        })
    }

    fn neural_turn(&self) -> bool {
        self.neural_seat
            .is_none_or(|seat| self.session.game().current_player() as usize == seat)
    }
}

pub(super) struct Rollout {
    pub records: Vec<MatchRecord>,
    pub inference: McInferenceMetrics,
}

/// Refill a bounded window of games, preserving input game order in the result.
/// No rollout RNG is consumed here: episode seeds are assigned by the caller
/// before scheduling, and each seat owns its exploration RNG.
pub(super) fn play(
    net: &ValueNet,
    device: &Device,
    configs: &[MatchConfig],
    batch_games: usize,
    epsilon: f64,
    baseline: Option<&AgentConfig>,
) -> Result<Rollout, AgentError> {
    if batch_games == 0 {
        return Err("inference_batch_games must be positive".into());
    }
    let mut policy = McBatchPolicy::new(net.clone(), device.clone());
    let mut pending = configs.iter().enumerate();
    let mut active = pending
        .by_ref()
        .take(batch_games)
        .map(|(i, config)| ActiveMatch::new(i, config, epsilon, baseline))
        .collect::<Result<Vec<_>, _>>()?;
    let mut records: Vec<_> = std::iter::repeat_with(|| None)
        .take(configs.len())
        .collect();
    while !active.is_empty() {
        let mut chosen = vec![None; active.len()];
        let mut decision_times = vec![Duration::ZERO; active.len()];
        for (index, game) in active.iter_mut().enumerate() {
            game.session.game().legal_actions(&mut game.actions);
            if !game.neural_turn() {
                let started = Instant::now();
                chosen[index] = Some(
                    game.opponent
                        .as_mut()
                        .unwrap()
                        .choose_action(game.session.game().observation(), &game.actions)?,
                );
                decision_times[index] = started.elapsed();
            }
        }
        let indices: Vec<_> = active
            .iter()
            .enumerate()
            .filter_map(|(index, game)| game.neural_turn().then_some(index))
            .collect();
        let mut requests: Vec<_> = active
            .iter_mut()
            .filter(|game| game.neural_turn())
            .map(|game| {
                let actor = game.session.game().current_player() as usize;
                McDecision {
                    observation: game.session.game().observation(),
                    actions: &game.actions,
                    state: &mut game.states[actor],
                }
            })
            .collect();
        if !requests.is_empty() {
            let started = Instant::now();
            let actions = policy.choose_actions(&mut requests)?;
            let apportioned = started.elapsed().div_f64(actions.len() as f64);
            for (index, action) in indices.into_iter().zip(actions) {
                chosen[index] = Some(action);
                decision_times[index] = apportioned;
            }
        }
        for ((game, action), elapsed) in active.iter_mut().zip(chosen).zip(decision_times) {
            game.session.step(action.unwrap(), elapsed)?;
        }
        let mut index = 0;
        while index < active.len() {
            if active[index].session.is_finished() {
                let game = active.swap_remove(index);
                records[game.index] = Some(game.session.finish()?);
                if let Some((next_index, config)) = pending.next() {
                    active.push(ActiveMatch::new(next_index, config, epsilon, baseline)?);
                }
            } else {
                index += 1;
            }
        }
    }
    Ok(Rollout {
        records: records.into_iter().map(Option::unwrap).collect(),
        inference: policy.metrics().clone(),
    })
}
