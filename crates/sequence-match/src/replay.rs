use std::{error::Error, fmt};

use crate::{AppliedAction, Match, MatchConfig, MatchEnd, MatchError};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Replay {
    pub config: MatchConfig,
    pub end: MatchEnd,
    pub actions: Vec<AppliedAction>,
}

impl Replay {
    /// Reconstruct the match using the current game rules. The observer sees the
    /// initial state and each successfully applied action, including on a valid
    /// prefix of a replay that later fails validation.
    pub fn play(
        &self,
        mut observe: impl FnMut(Option<AppliedAction>, &Match),
    ) -> Result<Match, ReplayError> {
        let mut game_match = Match::new(self.config.clone(), false);
        observe(None, &game_match);

        for (index, &action) in self.actions.iter().enumerate() {
            let applied = game_match
                .step(action.player, action.action)
                .map_err(|source| ReplayError::InvalidStep {
                    action_number: index + 1,
                    source,
                })?;
            observe(Some(applied), &game_match);
        }

        let actual = game_match.end();
        if actual == Some(self.end) {
            Ok(game_match)
        } else {
            Err(ReplayError::EndMismatch {
                expected: self.end,
                actual,
            })
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayError {
    InvalidStep {
        /// One-based position in the replay.
        action_number: usize,
        source: MatchError,
    },
    EndMismatch {
        expected: MatchEnd,
        actual: Option<MatchEnd>,
    },
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStep {
                action_number,
                source,
            } => write!(f, "replay action {action_number} failed ({source})"),
            Self::EndMismatch { expected, actual } => {
                write!(f, "replay expected {expected:?} and reached {actual:?}")
            }
        }
    }
}

impl Error for ReplayError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidStep { source, .. } => Some(source),
            Self::EndMismatch { .. } => None,
        }
    }
}
