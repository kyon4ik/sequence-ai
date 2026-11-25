use std::{collections::HashMap, num::NonZeroUsize};

use burn::tensor::Device;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sequence_agents::Policy;
use sequence_game::{Action, Card, Game, Outcome, Player, PlayerView};
use serde::{Deserialize, Serialize};

use super::{
    model::{GreedyPolicy, Network},
    observation::action_index,
};

const PRIOR_TEMPERATURE: f32 = 0.25;
const EXPLORATION: f32 = 1.25;
const DISCOUNT: f32 = 0.99;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SearchSettings {
    pub simulations: NonZeroUsize,
    pub depth: NonZeroUsize,
}

pub(crate) struct SearchPolicy<'a> {
    greedy: GreedyPolicy<'a>,
    settings: SearchSettings,
    rng: ChaCha8Rng,
}

impl<'a> SearchPolicy<'a> {
    pub fn new(
        network: &'a Network,
        device: &'a Device,
        settings: SearchSettings,
        seed: u64,
    ) -> Self {
        Self {
            greedy: GreedyPolicy { network, device },
            settings,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    fn node(&self, view: &PlayerView<'_>) -> Node {
        let network = self.greedy.network;
        let data = network.action_scores(&network.observe(view), self.greedy.device);
        let scores = data.as_slice::<f32>().expect("DQN scores are f32");
        Node::new(&view.legal_actions, scores)
    }

    fn simulate(&mut self, game: &mut Game, player: Player, node: &mut Node, depth: usize) -> f32 {
        let edge = node.select();
        game.apply(player, edge.action)
            .expect("root action is legal in the sample");

        let value = match self.next_decision(game, player) {
            Err(Outcome::Win(winner)) if winner == player => 1.0,
            Err(Outcome::Win(_)) => -1.0,
            Err(Outcome::Draw) => 0.0,
            Ok((view, observed_actions)) => {
                let mut hand = view.hand.to_vec();
                hand.sort_unstable_by_key(|card| (card.suit as u8, card.rank as u8));
                let observation = ObservationKey {
                    observed_actions,
                    hand,
                };
                let estimate = match edge.children.entry(observation) {
                    std::collections::hash_map::Entry::Occupied(mut child) if depth > 1 => {
                        self.simulate(game, player, child.get_mut(), depth - 1)
                    }
                    child => child.or_insert_with(|| self.node(&view)).value,
                };
                DISCOUNT * estimate
            }
        };

        edge.visits += 1;
        edge.total += value;
        node.visits += 1;
        value
    }

    fn next_decision<'g>(
        &mut self,
        game: &'g mut Game,
        player: Player,
    ) -> Result<(PlayerView<'g>, Vec<Action>), Outcome> {
        let mut observed_actions = Vec::new();
        while let Some(actor) = game.current_player() {
            let view = game.view(actor);
            let action = if actor == player {
                if view.legal_actions.as_slice() != [Action::Pass] {
                    return Ok((game.view(player), observed_actions));
                }
                Action::Pass
            } else {
                self.greedy
                    .choose_action(&view)
                    .expect("opponent has a legal action")
            };
            game.apply(actor, action)
                .expect("simulated action is legal");
            observed_actions.push(action);
        }
        Err(game.result().expect("simulation has ended").outcome)
    }
}

impl Policy for SearchPolicy<'_> {
    fn choose_action(&mut self, view: &PlayerView<'_>) -> Option<Action> {
        match view.legal_actions.as_slice() {
            [] => return None,
            [action] => return Some(*action),
            _ => {}
        }

        let mut tree = self.node(view);
        for _ in 0..self.settings.simulations.get() {
            let mut game = Game::sample_from(view, &mut self.rng);
            self.simulate(&mut game, view.player, &mut tree, self.settings.depth.get());
        }
        tree.edges
            .iter()
            .max_by(|a, b| {
                a.visits
                    .cmp(&b.visits)
                    .then_with(|| a.prior.total_cmp(&b.prior))
            })
            .map(|edge| edge.action)
    }
}

#[derive(Eq, Hash, PartialEq)]
struct ObservationKey {
    observed_actions: Vec<Action>,
    hand: Vec<Card>,
}

struct Node {
    edges: Vec<Edge>,
    value: f32,
    visits: usize,
}

struct Edge {
    action: Action,
    prior: f32,
    visits: usize,
    total: f32,
    children: HashMap<ObservationKey, Node>,
}

impl Node {
    fn new(actions: &[Action], scores: &[f32]) -> Self {
        let mut candidates: Vec<_> = actions
            .iter()
            .filter_map(|&action| action_index(action).map(|index| (index, action)))
            .collect();
        candidates.sort_unstable_by_key(|&(index, _)| index);

        let value = candidates
            .iter()
            .map(|&(index, _)| scores[index])
            .max_by(f32::total_cmp)
            .expect("a search node has a scored action");
        let mut edges: Vec<_> = candidates
            .into_iter()
            .map(|(index, action)| Edge {
                action,
                prior: ((scores[index] - value) / PRIOR_TEMPERATURE).exp(),
                visits: 0,
                total: 0.0,
                children: HashMap::new(),
            })
            .collect();
        let total: f32 = edges.iter().map(|edge| edge.prior).sum();
        for edge in &mut edges {
            edge.prior /= total;
        }
        Self {
            edges,
            value: value.clamp(-1.0, 1.0),
            visits: 0,
        }
    }

    fn select(&mut self) -> &mut Edge {
        let explore = EXPLORATION * (self.visits as f32 + 1.0).sqrt();
        let value = self.value;
        let score = |edge: &Edge| {
            let mean = if edge.visits == 0 {
                value
            } else {
                edge.total / edge.visits as f32
            };
            mean + explore * edge.prior / (1.0 + edge.visits as f32)
        };
        self.edges
            .iter_mut()
            .max_by(|a, b| score(a).total_cmp(&score(b)))
            .expect("a search node has a legal action")
    }
}
