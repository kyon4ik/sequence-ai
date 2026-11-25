use std::{collections::VecDeque, error::Error, fmt};

use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

use crate::{Board, Card, Position, Sequence, card::Effect, claims::new_claims};

const HAND_SIZE: usize = 7;
const SEQUENCES_TO_WIN: usize = 2;
const PASSES_TO_DRAW: u8 = 3;
pub const PLAY_HISTORY_LENGTH: usize = 4;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Player {
    One,
    Two,
}

impl Player {
    pub const fn index(self) -> usize {
        match self {
            Self::One => 0,
            Self::Two => 1,
        }
    }

    pub const fn opponent(self) -> Self {
        match self {
            Self::One => Self::Two,
            Self::Two => Self::One,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GameConfig {
    pub shuffle_seed: u64,
    pub starting_player: Player,
    /// Policy identifiers in Player 1, Player 2 order.
    pub policies: [String; 2],
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Action {
    Exchange(Card),
    Play { card: Card, target: Position },
    Pass,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CardPlay {
    pub card: Card,
    pub target: Position,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    Win(Player),
    Draw,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GameResult {
    pub config: GameConfig,
    pub outcome: Outcome,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActionError {
    GameOver,
    OutOfTurn,
    IllegalAction,
}

impl fmt::Display for ActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::GameOver => "the game has ended",
            Self::OutOfTurn => "the player cannot act on this turn",
            Self::IllegalAction => "the action is illegal in this position",
        })
    }
}

impl Error for ActionError {}

/// The information available to a policy sitting at one seat.
/// Arrays indexed by player are in Player 1, Player 2 order.
#[derive(Debug)]
pub struct PlayerView<'a> {
    pub player: Player,
    pub hand: &'a [Card],
    pub board: &'a Board,
    pub claimed_sequences: [&'a [Sequence]; 2],
    pub discard_piles: [&'a [Card]; 2],
    /// The opponent's most recent board plays, newest first, padded with `None`.
    pub opponent_plays: &'a [Option<CardPlay>; PLAY_HISTORY_LENGTH],
    pub hand_sizes: [usize; 2],
    pub draw_pile_size: usize,
    pub current_player: Option<Player>,
    pub consecutive_passes: u8,
    pub outcome: Option<Outcome>,
    pub legal_actions: Vec<Action>,
}

#[derive(Debug)]
pub struct PlayPreview {
    pub board: Board,
    pub new_claims: Vec<Sequence>,
    pub wins: bool,
}

impl PlayerView<'_> {
    /// Previews the public effects of a legal card play before any draw.
    pub fn preview_play(&self, action: Action) -> Option<PlayPreview> {
        let Action::Play { card, target } = action else {
            return None;
        };
        if !self.legal_actions.contains(&action) {
            return None;
        }

        let mut board = self.board.clone();
        let claimed = self.claimed_sequences[self.player.index()];
        let new_claims = play_on_board(&mut board, self.player, card, target, claimed);

        Some(PlayPreview {
            board,
            wins: claimed.len() + new_claims.len() >= SEQUENCES_TO_WIN,
            new_claims,
        })
    }
}

/// Full game state for the caller that runs the game and supplies views to policies.
pub struct Game {
    config: GameConfig,
    board: Board,
    hands: [Vec<Card>; 2],
    claimed: [Vec<Sequence>; 2],
    discards: [Vec<Card>; 2],
    recent_plays: [[Option<CardPlay>; PLAY_HISTORY_LENGTH]; 2],
    draw_pile: VecDeque<Card>,
    rng: ChaCha8Rng,
    current_player: Player,
    exchanged_this_turn: bool,
    consecutive_passes: u8,
    outcome: Option<Outcome>,
}

impl Game {
    pub fn new(config: GameConfig) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(config.shuffle_seed);
        let mut draw_pile: VecDeque<_> = Card::all().chain(Card::all()).collect();
        draw_pile.make_contiguous().shuffle(&mut rng);
        let hands = std::array::from_fn(|_| draw_pile.drain(..HAND_SIZE).collect());

        Self {
            current_player: config.starting_player,
            config,
            board: Board::empty(),
            hands,
            claimed: Default::default(),
            discards: Default::default(),
            recent_plays: Default::default(),
            draw_pile,
            rng,
            exchanged_this_turn: false,
            consecutive_passes: 0,
            outcome: None,
        }
    }

    pub fn current_player(&self) -> Option<Player> {
        self.outcome.is_none().then_some(self.current_player)
    }

    pub fn view(&self, player: Player) -> PlayerView<'_> {
        let current_player = self.current_player();
        PlayerView {
            player,
            hand: &self.hands[player.index()],
            board: &self.board,
            claimed_sequences: self.claimed.each_ref().map(Vec::as_slice),
            discard_piles: self.discards.each_ref().map(Vec::as_slice),
            opponent_plays: &self.recent_plays[player.opponent().index()],
            hand_sizes: self.hands.each_ref().map(Vec::len),
            draw_pile_size: self.draw_pile.len(),
            current_player,
            consecutive_passes: self.consecutive_passes,
            outcome: self.outcome,
            legal_actions: if current_player == Some(player) {
                self.legal_actions()
            } else {
                Vec::new()
            },
        }
    }

    pub fn result(&self) -> Option<GameResult> {
        self.outcome.map(|outcome| GameResult {
            config: self.config.clone(),
            outcome,
        })
    }

    pub fn apply(&mut self, player: Player, action: Action) -> Result<(), ActionError> {
        match self.current_player() {
            None => return Err(ActionError::GameOver),
            Some(current) if current != player => return Err(ActionError::OutOfTurn),
            Some(_) => {}
        }

        match action {
            Action::Exchange(card) => {
                if !self.exchange_available(card) {
                    return Err(ActionError::IllegalAction);
                }
                self.discard(card)?;
                self.draw_for_current_player();
                self.exchanged_this_turn = true;
            }
            Action::Play { card, target } => {
                if !self.play_targets(card).any(|position| position == target) {
                    return Err(ActionError::IllegalAction);
                }
                self.discard(card)?;
                self.play_card(card, target);
            }
            Action::Pass => {
                if self.hands[player.index()].iter().any(|&card| {
                    self.exchange_available(card) || self.play_targets(card).next().is_some()
                }) {
                    return Err(ActionError::IllegalAction);
                }
                self.consecutive_passes += 1;
                if self.consecutive_passes == PASSES_TO_DRAW {
                    self.outcome = Some(Outcome::Draw);
                } else {
                    self.end_turn();
                }
            }
        }

        Ok(())
    }

    fn legal_actions(&self) -> Vec<Action> {
        let hand = &self.hands[self.current_player.index()];
        let mut actions = Vec::new();

        for (index, &card) in hand.iter().enumerate() {
            if hand[..index].contains(&card) {
                continue;
            }

            if self.exchange_available(card) {
                actions.push(Action::Exchange(card));
            }
            actions.extend(
                self.play_targets(card)
                    .map(|target| Action::Play { card, target }),
            );
        }

        if actions.is_empty() {
            actions.push(Action::Pass);
        }
        actions
    }

    fn exchange_available(&self, card: Card) -> bool {
        !self.exchanged_this_turn
            && (!self.draw_pile.is_empty() || self.discards.iter().any(|pile| !pile.is_empty()))
            && self.board.is_dead_card(card)
    }

    fn play_targets(&self, card: Card) -> impl Iterator<Item = Position> + '_ {
        let effect = card.effect();
        let opponent = self.current_player.opponent();

        Board::candidate_positions(card)
            .iter()
            .copied()
            .filter(move |&target| {
                let chip = self.board.chip(target);
                match effect {
                    Effect::Match | Effect::Wild => !target.is_corner() && chip.is_none(),
                    Effect::Remove => {
                        chip == Some(opponent)
                            && !self.claimed[opponent.index()]
                                .iter()
                                .any(|sequence| sequence.spaces().contains(&target))
                    }
                }
            })
    }

    fn discard(&mut self, card: Card) -> Result<(), ActionError> {
        let player = self.current_player.index();
        let hand = &mut self.hands[player];
        let index = hand
            .iter()
            .position(|&held| held == card)
            .ok_or(ActionError::IllegalAction)?;

        hand.remove(index);
        self.discards[player].push(card);
        Ok(())
    }

    fn play_card(&mut self, card: Card, target: Position) {
        let player = self.current_player;
        let recent = &mut self.recent_plays[player.index()];
        recent.rotate_right(1);
        recent[0] = Some(CardPlay { card, target });
        self.consecutive_passes = 0;

        let claimed = &mut self.claimed[player.index()];
        claimed.extend(play_on_board(
            &mut self.board,
            player,
            card,
            target,
            claimed,
        ));

        if claimed.len() >= SEQUENCES_TO_WIN {
            self.outcome = Some(Outcome::Win(player));
            return;
        }

        self.draw_for_current_player();
        self.end_turn();
    }

    fn end_turn(&mut self) {
        self.current_player = self.current_player.opponent();
        self.exchanged_this_turn = false;
    }

    fn draw_for_current_player(&mut self) {
        self.replenish_draw_pile();
        if let Some(card) = self.draw_pile.pop_front() {
            self.hands[self.current_player.index()].push(card);
            self.replenish_draw_pile();
        }
    }

    fn replenish_draw_pile(&mut self) {
        if self.draw_pile.is_empty() {
            self.draw_pile
                .extend(self.discards.iter_mut().flat_map(|pile| pile.drain(..)));
            self.draw_pile.make_contiguous().shuffle(&mut self.rng);
        }
    }
}

fn play_on_board(
    board: &mut Board,
    player: Player,
    card: Card,
    target: Position,
    claimed: &[Sequence],
) -> Vec<Sequence> {
    if card.effect() == Effect::Remove {
        board.set_chip(target, None);
        return Vec::new();
    }

    board.set_chip(target, Some(player));
    new_claims(board, player, target, claimed)
}

#[cfg(test)]
mod tests;
