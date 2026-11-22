mod heuristic;
mod random;

#[cfg(test)]
mod tests;

pub use heuristic::HeuristicAgent;
pub use random::RandomAgent;
use sequence_game::{Action, PlayerView};

pub trait Policy {
    /// Selects a legal action, or returns None when the view has no legal actions.
    fn choose_action(&mut self, view: &PlayerView<'_>) -> Option<Action>;
}
