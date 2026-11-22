use std::cmp::Ordering;

use rand::{SeedableRng, seq::IndexedRandom};
use rand_chacha::ChaCha8Rng;
use sequence_game::{Action, Board, Player, PlayerView, Rank, SEQUENCE_LENGTH, Sequence, Space};

use crate::Policy;

const OPEN_LINE_WEIGHTS: [i32; SEQUENCE_LENGTH + 1] = [0, 64, 16, 4, 1, 0];

pub struct HeuristicAgent {
    rng: ChaCha8Rng,
}

impl HeuristicAgent {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

#[derive(Eq, Ord, PartialEq, PartialOrd)]
struct PlayScore {
    wins: bool,
    new_claims: usize,
    open_line_advantage: i32,
    saves_jack: bool,
}

impl Policy for HeuristicAgent {
    fn choose_action(&mut self, view: &PlayerView<'_>) -> Option<Action> {
        match view.legal_actions.as_slice() {
            [] => return None,
            [action] => return Some(*action),
            _ => {}
        }

        if let Some(&exchange) = view
            .legal_actions
            .iter()
            .find(|action| matches!(action, Action::Exchange(_)))
        {
            return Some(exchange);
        }

        let windows: Vec<_> = Sequence::all().collect();
        let mut best_score = None;
        let mut best_actions = Vec::new();

        for &action in &view.legal_actions {
            let Some(score) = score_play(view, action, &windows) else {
                continue;
            };

            match Some(&score).cmp(&best_score.as_ref()) {
                Ordering::Greater => {
                    best_score = Some(score);
                    best_actions.clear();
                    best_actions.push(action);
                }
                Ordering::Equal => best_actions.push(action),
                Ordering::Less => {}
            }
        }

        best_actions.choose(&mut self.rng).copied()
    }
}

fn score_play(view: &PlayerView<'_>, action: Action, windows: &[Sequence]) -> Option<PlayScore> {
    let Action::Play { card, .. } = action else {
        return None;
    };
    let preview = view.preview_play(action)?;
    let player = view.player;
    let opponent = player.opponent();

    let own_score = open_line_score(
        windows,
        &preview.board,
        player,
        view.claimed_sequences[player.index()],
        &preview.new_claims,
    );
    let opponent_score = open_line_score(
        windows,
        &preview.board,
        opponent,
        view.claimed_sequences[opponent.index()],
        &[],
    );

    Some(PlayScore {
        wins: preview.wins,
        new_claims: preview.new_claims.len(),
        open_line_advantage: own_score - opponent_score,
        saves_jack: card.rank != Rank::Jack,
    })
}

fn open_line_score(
    windows: &[Sequence],
    board: &Board,
    player: Player,
    claimed: &[Sequence],
    new_claims: &[Sequence],
) -> i32 {
    windows
        .iter()
        .filter(|window| {
            claimed
                .iter()
                .chain(new_claims)
                .all(|other| window.compatible_with(other))
        })
        .map(|window| {
            let mut empty = 0;
            for &position in window.spaces() {
                match board.space(position) {
                    Space::Labeled {
                        chip: Some(owner), ..
                    } if owner != player => return 0,
                    Space::Labeled { chip: None, .. } => empty += 1,
                    _ => {}
                }
            }
            OPEN_LINE_WEIGHTS[empty]
        })
        .sum()
}
