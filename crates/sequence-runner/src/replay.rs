use crate::{FORMAT_VERSION, MatchRecord, RecordedGame, Termination};

#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    #[error("unsupported replay format {0}")]
    Format(u32),
    #[error("expected game {expected}, found {actual}")]
    Game {
        expected: &'static str,
        actual: String,
    },
    #[error("replay diverged at frame {frame}: {message}")]
    Divergence { frame: usize, message: String },
    #[error("frame {index} is outside replay with {frames} frames")]
    Frame { index: usize, frames: usize },
}

/// Reconstructed game states support seeking/backtracking and branching from
/// any frame. Policies and checkpoints are never invoked during reconstruction.
pub struct Replay<G> {
    states: Vec<G>,
    cursor: usize,
}

impl<G: RecordedGame> Replay<G> {
    pub fn new(record: &MatchRecord) -> Result<Self, ReplayError> {
        if record.header.format_version != FORMAT_VERSION {
            return Err(ReplayError::Format(record.header.format_version));
        }
        if record.header.game_id != G::ID {
            return Err(ReplayError::Game {
                expected: G::ID,
                actual: record.header.game_id.clone(),
            });
        }
        let diverged = |frame, message: &str| ReplayError::Divergence {
            frame,
            message: message.into(),
        };
        let mut game = G::from_seed(record.header.config.game_seed);
        if game.snapshot() != record.header.initial_state {
            return Err(diverged(
                0,
                "initial observation differs (seed or engine changed)",
            ));
        }
        let mut states = vec![game.clone()];
        let mut rewards = [0., 0.];
        for (index, step) in record.steps.iter().enumerate() {
            let frame = index + 1;
            if step.index != index || step.actor != game.current_player() {
                return Err(diverged(frame, "action index or actor differs"));
            }
            game.apply(step.action)
                .map_err(|e| diverged(frame, &e.to_string()))?;
            if game.snapshot() != step.state {
                return Err(diverged(
                    frame,
                    "observation differs (record or engine changed)",
                ));
            }
            for (total, reward) in rewards.iter_mut().zip(step.rewards) {
                *total += reward;
                if !reward.is_finite() || !total.is_finite() {
                    return Err(diverged(frame, "reward is not finite"));
                }
            }
            states.push(game.clone());
        }
        let frame = record.steps.len();
        if record.metrics.actions != frame
            || record.metrics.turns != game.turns()
            || rewards != record.metrics.players.each_ref().map(|m| m.reward)
        {
            return Err(diverged(frame, "summary metrics differ"));
        }
        match &record.termination {
            Termination::Completed(outcome)
                if !game.is_game_over() || *outcome != game.outcome() =>
            {
                return Err(diverged(frame, "completed outcome differs"));
            }
            Termination::ActionLimit
                if game.is_game_over() || frame != record.header.config.max_actions =>
            {
                return Err(diverged(frame, "action limit does not match final state"));
            }
            _ => {}
        }
        Ok(Self { states, cursor: 0 })
    }

    pub fn frame_count(&self) -> usize {
        self.states.len()
    }
    pub fn position(&self) -> usize {
        self.cursor
    }
    pub fn game(&self) -> &G {
        &self.states[self.cursor]
    }
    pub fn seek(&mut self, index: usize) -> Result<&G, ReplayError> {
        if index >= self.states.len() {
            return Err(ReplayError::Frame {
                index,
                frames: self.states.len(),
            });
        }
        self.cursor = index;
        Ok(self.game())
    }
    pub fn step_forward(&mut self) -> Option<&G> {
        if self.cursor + 1 >= self.states.len() {
            return None;
        }
        self.cursor += 1;
        Some(self.game())
    }
    pub fn step_back(&mut self) -> Option<&G> {
        if self.cursor == 0 {
            return None;
        }
        self.cursor -= 1;
        Some(self.game())
    }
}
