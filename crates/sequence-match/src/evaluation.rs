use std::{error::Error, fmt, num::NonZeroUsize};

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sequence_agents::Policy;
use sequence_game::{GameConfig, Outcome, Player};

use crate::{Match, MatchConfig, MatchEnd, MatchError, MatchReport};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvaluationConfig {
    /// Labels for the first and second policy factories.
    pub policy_ids: [String; 2],
    pub shuffle_seeds: Vec<u64>,
    pub starting_player: Player,
    /// Initializes the RNG that derives one seed per policy for each pair.
    pub policy_seed: u64,
    pub action_limit: NonZeroUsize,
    pub capture_replays: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EvaluationTotals {
    /// Wins in policy factory order.
    pub wins: [usize; 2],
    pub draws: usize,
    pub truncations: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvaluationReport {
    pub config: EvaluationConfig,
    /// Pairs in input seed order. The first policy occupies Player 1's seat
    /// for the first match of each pair and Player 2's seat for the second.
    pub pairs: Vec<[MatchReport; 2]>,
    pub totals: EvaluationTotals,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvaluationError {
    /// Zero-based index in the shuffle seed list.
    pub pair_index: usize,
    pub shuffle_seed: u64,
    /// Zero-based policy factory indices in Player 1, Player 2 order.
    pub policy_order: [usize; 2],
    pub source: MatchError,
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "evaluation failed at pair {} with shuffle seed {} and policy order {:?} ({})",
            self.pair_index, self.shuffle_seed, self.policy_order, self.source,
        )
    }
}

impl Error for EvaluationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

/// Run two games per shuffle seed with swapped seats and a fixed starting seat.
/// Each factory creates a fresh policy for each game. Its seed is reused within a pair.
pub fn evaluate<P1: Policy, P2: Policy>(
    config: &EvaluationConfig,
    mut make_first: impl FnMut(u64) -> P1,
    mut make_second: impl FnMut(u64) -> P2,
) -> Result<EvaluationReport, EvaluationError> {
    let mut rng = ChaCha8Rng::seed_from_u64(config.policy_seed);
    let mut pairs = Vec::with_capacity(config.shuffle_seeds.len());
    let mut totals = EvaluationTotals::default();

    for (pair_index, &shuffle_seed) in config.shuffle_seeds.iter().enumerate() {
        let policy_seeds = [rng.next_u64(), rng.next_u64()];
        let mut play = |policy_order: [usize; 2]| -> Result<MatchReport, EvaluationError> {
            let mut first = make_first(policy_seeds[0]);
            let mut second = make_second(policy_seeds[1]);
            let mut policies: [&mut dyn Policy; 2] = [&mut first, &mut second];
            if policy_order == [1, 0] {
                policies.swap(0, 1);
            }

            let match_config = MatchConfig {
                game: GameConfig {
                    shuffle_seed,
                    starting_player: config.starting_player,
                    policies: policy_order.map(|index| config.policy_ids[index].clone()),
                },
                policy_seeds: policy_order.map(|index| Some(policy_seeds[index])),
                action_limit: config.action_limit,
            };
            let mut game_match = Match::new(match_config, config.capture_replays);
            let end = game_match
                .run(policies, |_, _| {})
                .map_err(|source| EvaluationError {
                    pair_index,
                    shuffle_seed,
                    policy_order,
                    source,
                })?;

            match end {
                MatchEnd::Completed(Outcome::Win(player)) => {
                    totals.wins[policy_order[player.index()]] += 1;
                }
                MatchEnd::Completed(Outcome::Draw) => totals.draws += 1,
                MatchEnd::ActionLimit => totals.truncations += 1,
            }

            Ok(game_match
                .into_report()
                .expect("a completed run has a report"))
        };

        pairs.push([play([0, 1])?, play([1, 0])?]);
    }

    Ok(EvaluationReport {
        config: config.clone(),
        pairs,
        totals,
    })
}
