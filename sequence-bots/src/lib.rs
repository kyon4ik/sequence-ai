use sequence_core::{Action, Observation};

use crate::greedy::GreedyBot;
use crate::random::RandomBot;

mod greedy;
mod random;

pub trait Bot {
    fn choose_action<'a>(&mut self, obs: Observation<'a>, actions: &[Action]) -> Action;
}

#[derive(Clone, Copy, Debug)]
pub enum BotKind {
    Random,
    Greedy,
}

impl BotKind {
    pub fn create(self, seed: u64) -> Box<dyn Bot> {
        match self {
            BotKind::Random => Box::new(RandomBot::from_seed(seed)),
            BotKind::Greedy => Box::new(GreedyBot::new()),
        }
    }
}
