use super::opponents::{CheckpointPool, OpponentPlan, OpponentPolicy};
use burn::prelude::*;
use sequence_agent::mc::{McBatchPolicy, McDecision, McInferenceMetrics, McPolicyState, ValueNet};
use sequence_agent::{
    Agent, AgentConfig, AgentDescriptor, AgentError, EpisodeContext, InferenceDevice,
};
use sequence_core::{Action, Color, Game, Simple};
use sequence_runner::{MatchConfig, MatchRecord, MatchSession};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum SeatPolicy {
    Neural(usize),
    Baseline,
}

struct Assignment {
    seats: [SeatPolicy; 2],
    descriptors: [AgentDescriptor; 2],
}

struct ActiveMatch {
    index: usize,
    session: MatchSession<Simple>,
    states: [McPolicyState; 2],
    opponent: Option<Box<dyn Agent>>,
    seats: [SeatPolicy; 2],
    actions: Vec<Action>,
}

impl ActiveMatch {
    fn new(
        index: usize,
        config: &MatchConfig,
        epsilon: f64,
        assignment: &Assignment,
        baseline: Option<&AgentConfig>,
    ) -> Result<Self, AgentError> {
        let opponent = if let Some(seat) = assignment
            .seats
            .iter()
            .position(|&p| p == SeatPolicy::Baseline)
        {
            let mut agent = baseline
                .ok_or("missing baseline policy")?
                .create(config.agent_seeds[seat], InferenceDevice::Cpu)?;
            agent.on_episode_start(EpisodeContext {
                game_seed: config.game_seed,
                agent_seed: config.agent_seeds[seat],
                color: [Color::Red, Color::Blue][seat],
            });
            Some(agent)
        } else {
            None
        };
        Ok(Self {
            index,
            session: MatchSession::new(config.clone(), assignment.descriptors.clone())?,
            states: std::array::from_fn(|seat| {
                McPolicyState::new(
                    config.agent_seeds[seat],
                    if assignment.seats[seat] == SeatPolicy::Neural(0) {
                        epsilon
                    } else {
                        0.
                    },
                )
            }),
            opponent,
            seats: assignment.seats,
            actions: Vec::new(),
        })
    }

    fn policy(&self) -> SeatPolicy {
        self.seats[self.session.game().current_player() as usize]
    }
}

pub(super) struct Rollout {
    pub records: Vec<MatchRecord>,
    pub inference: McInferenceMetrics,
}

pub(super) fn add_metrics(total: &mut McInferenceMetrics, metrics: &McInferenceMetrics) {
    total.decisions += metrics.decisions;
    total.batches += metrics.batches;
    total.candidates += metrics.candidates;
    total.encoding_time += metrics.encoding_time;
    total.prediction_time += metrics.prediction_time;
}

/// Legacy self-play/evaluation entry point, using the same multi-policy scheduler.
pub(super) fn play(
    net: &ValueNet,
    device: &Device,
    configs: &[MatchConfig],
    batch_games: usize,
    epsilon: f64,
    baseline: Option<&AgentConfig>,
) -> Result<Rollout, AgentError> {
    let assignments: Vec<_> = (0..configs.len())
        .map(|index| {
            let mut seats = [SeatPolicy::Neural(0); 2];
            let mut descriptors = std::array::from_fn(|_| AgentDescriptor::new("self-play", "MC"));
            if baseline.is_some() {
                seats[1 - index % 2] = SeatPolicy::Baseline;
                descriptors[index % 2] = AgentDescriptor::new("candidate", "Candidate");
                descriptors[1 - index % 2] = AgentDescriptor::new("baseline", "Baseline");
            }
            Assignment { seats, descriptors }
        })
        .collect();
    play_assigned(
        std::slice::from_ref(net),
        device,
        configs,
        batch_games,
        epsilon,
        &assignments,
        baseline,
    )
}

/// Each wave loads at most four frozen models. Games with a shared opponent are
/// collected together; results are restored to the preplanned game order before
/// target extraction and shuffling. No scheduling RNG is consumed here.
pub(super) fn play_opponents(
    net: &ValueNet,
    device: &Device,
    configs: &[MatchConfig],
    batch_games: usize,
    epsilon: f64,
    plans: &[OpponentPlan],
    pool: Option<&CheckpointPool>,
) -> Result<Rollout, AgentError> {
    if configs.len() != plans.len() {
        return Err("opponent schedule length differs from game count".into());
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut direct = Vec::new();
    for (index, plan) in plans.iter().enumerate() {
        match plan.policy {
            OpponentPolicy::Checkpoint(iteration) => {
                groups.entry(iteration).or_default().push(index)
            }
            _ => direct.push(index),
        }
    }
    let groups: Vec<_> = groups.into_iter().collect();
    let mut records: Vec<_> = std::iter::repeat_with(|| None)
        .take(configs.len())
        .collect();
    let mut inference = McInferenceMetrics::default();
    for wave in 0..groups.len().div_ceil(4).max(1) {
        let group_slice = &groups[(wave * 4).min(groups.len())..((wave + 1) * 4).min(groups.len())];
        let mut models = vec![net.clone()];
        for (iteration, _) in group_slice {
            let path = pool
                .ok_or("missing opponent checkpoint pool")?
                .path(*iteration);
            models.push(net.clone().valid().try_load_file(path)?);
        }
        let mut indices = if wave == 0 {
            std::mem::take(&mut direct)
        } else {
            Vec::new()
        };
        indices.extend(
            group_slice
                .iter()
                .flat_map(|(_, indices)| indices.iter().copied()),
        );
        let assignments: Vec<_> = indices
            .iter()
            .map(|&index| {
                let plan = &plans[index];
                let mut seats = [SeatPolicy::Neural(0); 2];
                let mut descriptors =
                    std::array::from_fn(|_| AgentDescriptor::new("self-play", "MC"));
                if plan.policy != OpponentPolicy::SelfPlay {
                    let seat = 1 - plan.learner_seat;
                    seats[seat] = match plan.policy {
                        OpponentPolicy::Checkpoint(iteration) => SeatPolicy::Neural(
                            1 + group_slice
                                .iter()
                                .position(|(i, _)| *i == iteration)
                                .unwrap(),
                        ),
                        OpponentPolicy::Greedy => SeatPolicy::Baseline,
                        OpponentPolicy::SelfPlay => unreachable!(),
                    };
                    descriptors[plan.learner_seat] = AgentDescriptor::new("learner", "Learner");
                    descriptors[seat] = AgentDescriptor::new(plan.policy.id(), plan.policy.id());
                }
                Assignment { seats, descriptors }
            })
            .collect();
        let configs: Vec<_> = indices.iter().map(|&i| configs[i].clone()).collect();
        let output = play_assigned(
            &models,
            device,
            &configs,
            batch_games,
            epsilon,
            &assignments,
            Some(&AgentConfig::greedy()),
        )?;
        add_metrics(&mut inference, &output.inference);
        for (index, record) in indices.into_iter().zip(output.records) {
            records[index] = Some(record);
        }
    }
    Ok(Rollout {
        records: records.into_iter().map(Option::unwrap).collect(),
        inference,
    })
}

/// Refill a bounded window of games and batch each model's pending decisions.
fn play_assigned(
    models: &[ValueNet],
    device: &Device,
    configs: &[MatchConfig],
    batch_games: usize,
    epsilon: f64,
    assignments: &[Assignment],
    baseline: Option<&AgentConfig>,
) -> Result<Rollout, AgentError> {
    if batch_games == 0 {
        return Err("inference_batch_games must be positive".into());
    }
    let mut policies: Vec<_> = models
        .iter()
        .map(|net| McBatchPolicy::new(net.clone(), device.clone()))
        .collect();
    let mut pending = configs.iter().enumerate();
    let mut active = pending
        .by_ref()
        .take(batch_games)
        .map(|(i, config)| ActiveMatch::new(i, config, epsilon, &assignments[i], baseline))
        .collect::<Result<Vec<_>, _>>()?;
    let mut records: Vec<_> = std::iter::repeat_with(|| None)
        .take(configs.len())
        .collect();
    while !active.is_empty() {
        let mut chosen = vec![None; active.len()];
        let mut decision_times = vec![Duration::ZERO; active.len()];
        for (index, game) in active.iter_mut().enumerate() {
            game.session.game().legal_actions(&mut game.actions);
            if game.policy() == SeatPolicy::Baseline {
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
        for (policy_index, policy) in policies.iter_mut().enumerate() {
            let indices: Vec<_> = active
                .iter()
                .enumerate()
                .filter_map(|(index, game)| {
                    (game.policy() == SeatPolicy::Neural(policy_index)).then_some(index)
                })
                .collect();
            let mut requests: Vec<_> = active
                .iter_mut()
                .filter(|game| game.policy() == SeatPolicy::Neural(policy_index))
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
                    active.push(ActiveMatch::new(
                        next_index,
                        config,
                        epsilon,
                        &assignments[next_index],
                        baseline,
                    )?);
                }
            } else {
                index += 1;
            }
        }
    }
    let mut inference = McInferenceMetrics::default();
    for policy in policies {
        add_metrics(&mut inference, policy.metrics());
    }
    Ok(Rollout {
        records: records.into_iter().map(Option::unwrap).collect(),
        inference,
    })
}
