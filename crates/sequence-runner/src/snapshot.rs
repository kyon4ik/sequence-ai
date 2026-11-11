use sequence_core::{Board, Card, Cell, Classic, Color, Game, GameOutcome, Position, Simple};
use serde::{Deserialize, Serialize};

/// Portable public board representation; never serializes core's packed storage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardSnapshot {
    /// Row-major, top-left first; exactly 100 cells for the built-in games.
    pub cells: Vec<Cell>,
    pub sequences: [Vec<[Position; 5]>; 2],
}

impl From<&Board> for BoardSnapshot {
    fn from(board: &Board) -> Self {
        Self {
            cells: Position::all().map(|p| board.cell(p)).collect(),
            sequences: [Color::Red, Color::Blue]
                .map(|c| board.sequences(c).iter().map(|s| *s.positions()).collect()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassicSnapshot {
    pub exchanged_dead_card: bool,
    pub consecutive_passes: u8,
}

/// Owned actor observation for a replay viewer. Opponent hand and draw order
/// remain hidden. At a terminal frame, current_player is the next actor, not
/// necessarily the winner; use outcome to determine the result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameSnapshot {
    pub board: BoardSnapshot,
    pub turns: u16,
    pub current_player: Color,
    pub hand: Vec<Card>,
    pub opponent_hand_size: usize,
    pub draw_pile_size: usize,
    pub discard_pile: Vec<Card>,
    pub outcome: GameOutcome,
    pub classic: Option<ClassicSnapshot>,
}

/// Adds replay views without introducing recording concerns into core's Game.
/// Change ID if rules or seeded reconstruction compatibility changes.
pub trait RecordedGame: Game + Clone + 'static {
    const ID: &'static str;
    fn snapshot(&self) -> GameSnapshot;
}

impl RecordedGame for Simple {
    const ID: &'static str = "sequence-simple-v1";
    fn snapshot(&self) -> GameSnapshot {
        let obs = self.observation();
        GameSnapshot {
            board: obs.board.into(),
            turns: obs.turns,
            current_player: obs.current_player,
            hand: obs.hand.to_vec(),
            opponent_hand_size: obs.opponent_hand_size,
            draw_pile_size: obs.draw_pile_size,
            discard_pile: obs.discard_pile.to_vec(),
            outcome: obs.outcome,
            classic: None,
        }
    }
}

impl RecordedGame for Classic {
    const ID: &'static str = "sequence-classic-v1";
    fn snapshot(&self) -> GameSnapshot {
        let obs = self.observation();
        GameSnapshot {
            board: obs.board.into(),
            turns: obs.turns,
            current_player: obs.current_player,
            hand: obs.hand.to_vec(),
            opponent_hand_size: obs.opponent_hand_size,
            draw_pile_size: obs.draw_pile_size,
            discard_pile: obs.discard_pile.to_vec(),
            outcome: obs.outcome,
            classic: Some(ClassicSnapshot {
                exchanged_dead_card: obs.exchanged_dead_card,
                consecutive_passes: obs.consecutive_passes,
            }),
        }
    }
}
