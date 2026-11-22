mod evaluation;
mod replay;

#[cfg(test)]
mod tests;

use std::{error::Error, fmt, num::NonZeroUsize};

use sequence_agents::Policy;
use sequence_game::{Action, ActionError, Game, GameConfig, Outcome, Player, PlayerView};

pub use evaluation::{
    EvaluationConfig, EvaluationError, EvaluationReport, EvaluationTotals, evaluate,
};
pub use replay::{Replay, ReplayError};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatchConfig {
    pub game: GameConfig,
    /// Seeds used by the caller to initialize policies in Player 1, Player 2 order.
    pub policy_seeds: [Option<u64>; 2],
    /// Maximum accepted actions, including dead-card exchanges.
    pub action_limit: NonZeroUsize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchEnd {
    Completed(Outcome),
    ActionLimit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AppliedAction {
    pub player: Player,
    pub action: Action,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatchReport {
    pub config: MatchConfig,
    pub end: MatchEnd,
    pub action_count: usize,
    pub replay: Option<Replay>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchError {
    Finished(MatchEnd),
    MissingAction {
        player: Player,
    },
    InvalidAction {
        player: Player,
        action: Action,
        source: ActionError,
    },
}

impl fmt::Display for MatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Finished(end) => write!(f, "the match has ended ({end:?})"),
            Self::MissingAction { player } => {
                write!(f, "the policy for {player:?} chose no action")
            }
            Self::InvalidAction {
                player,
                action,
                source,
            } => write!(f, "could not apply {action:?} for {player:?} ({source})"),
        }
    }
}

impl Error for MatchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidAction { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub struct Match {
    config: MatchConfig,
    game: Game,
    end: Option<MatchEnd>,
    action_count: usize,
    recorded_actions: Option<Vec<AppliedAction>>,
}

impl Match {
    pub fn new(config: MatchConfig, capture_replay: bool) -> Self {
        let game = Game::new(config.game.clone());
        Self {
            config,
            game,
            end: None,
            action_count: 0,
            recorded_actions: capture_replay.then(Vec::new),
        }
    }

    pub fn config(&self) -> &MatchConfig {
        &self.config
    }

    /// The rules state for trusted callers. At the action limit it still has a current player.
    pub fn game(&self) -> &Game {
        &self.game
    }

    pub fn end(&self) -> Option<MatchEnd> {
        self.end
    }

    pub fn action_count(&self) -> usize {
        self.action_count
    }

    pub fn current_player(&self) -> Option<Player> {
        if self.end.is_some() {
            return None;
        }
        self.game.current_player()
    }

    pub fn active_view(&self) -> Option<PlayerView<'_>> {
        self.current_player().map(|player| self.game.view(player))
    }

    pub fn step(&mut self, player: Player, action: Action) -> Result<AppliedAction, MatchError> {
        if let Some(end) = self.end {
            return Err(MatchError::Finished(end));
        }

        self.game
            .apply(player, action)
            .map_err(|source| MatchError::InvalidAction {
                player,
                action,
                source,
            })?;

        let applied = AppliedAction { player, action };
        self.action_count += 1;
        if let Some(actions) = &mut self.recorded_actions {
            actions.push(applied);
        }

        if let Some(result) = self.game.result() {
            self.end = Some(MatchEnd::Completed(result.outcome));
        } else if self.action_count == self.config.action_limit.get() {
            self.end = Some(MatchEnd::ActionLimit);
        }

        Ok(applied)
    }

    /// Ask a policy to make the current player's decision using only that player's view.
    pub fn step_policy(&mut self, policy: &mut dyn Policy) -> Result<AppliedAction, MatchError> {
        if let Some(end) = self.end {
            return Err(MatchError::Finished(end));
        }

        let view = self
            .active_view()
            .expect("an unfinished match has an active player");
        let player = view.player;
        let action = policy
            .choose_action(&view)
            .ok_or(MatchError::MissingAction { player })?;

        self.step(player, action)
    }

    /// Run policies in Player 1, Player 2 order. The observer receives `None` at
    /// entry and the applied action after each successful step.
    pub fn run(
        &mut self,
        policies: [&mut dyn Policy; 2],
        mut observe: impl FnMut(Option<AppliedAction>, &Match),
    ) -> Result<MatchEnd, MatchError> {
        observe(None, self);

        while let Some(player) = self.current_player() {
            let applied = self.step_policy(policies[player.index()])?;
            observe(Some(applied), self);
        }

        Ok(self
            .end
            .expect("a match without a current player has ended"))
    }

    /// Return a report once the match has ended.
    pub fn into_report(self) -> Option<MatchReport> {
        let end = self.end?;
        let replay = self.recorded_actions.map(|actions| Replay {
            config: self.config.clone(),
            end,
            actions,
        });

        Some(MatchReport {
            config: self.config,
            end,
            action_count: self.action_count,
            replay,
        })
    }
}
