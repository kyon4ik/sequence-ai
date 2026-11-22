use std::num::NonZeroUsize;

use sequence_agents::{HeuristicAgent, Policy, RandomAgent};
use sequence_game::{Action, ActionError, GameConfig, Outcome, Player, PlayerView};

use crate::{
    EvaluationConfig, EvaluationError, EvaluationTotals, Match, MatchConfig, MatchEnd, MatchError,
    ReplayError, evaluate,
};

struct PreferExchange;

impl Policy for PreferExchange {
    fn choose_action(&mut self, view: &PlayerView<'_>) -> Option<Action> {
        view.legal_actions
            .iter()
            .copied()
            .find(|action| matches!(action, Action::Exchange(_)))
            .or_else(|| view.legal_actions.first().copied())
    }
}

struct MissingAction;

impl Policy for MissingAction {
    fn choose_action(&mut self, _: &PlayerView<'_>) -> Option<Action> {
        None
    }
}

fn match_config(limit: usize) -> MatchConfig {
    MatchConfig {
        game: GameConfig {
            shuffle_seed: 42,
            starting_player: Player::Two,
            policies: ["first".into(), "second".into()],
        },
        policy_seeds: [None; 2],
        action_limit: NonZeroUsize::new(limit).unwrap(),
    }
}

fn evaluation_config(limit: usize) -> EvaluationConfig {
    EvaluationConfig {
        policy_ids: ["random".into(), "heuristic".into()],
        shuffle_seeds: vec![42, 7],
        starting_player: Player::Two,
        policy_seed: 24,
        action_limit: NonZeroUsize::new(limit).unwrap(),
        capture_replays: true,
    }
}

fn assert_same_position(first: &Match, second: &Match) {
    assert_eq!(first.end(), second.end());
    assert_eq!(first.action_count(), second.action_count());
    assert_eq!(first.current_player(), second.current_player());

    for player in [Player::One, Player::Two] {
        let first = first.game().view(player);
        let second = second.game().view(player);

        assert_eq!(first.board, second.board);
        assert_eq!(first.hand, second.hand);
        assert_eq!(first.claimed_sequences, second.claimed_sequences);
        assert_eq!(first.discard_piles, second.discard_piles);
        assert_eq!(first.draw_pile_size, second.draw_pile_size);
        assert_eq!(first.consecutive_passes, second.consecutive_passes);
        assert_eq!(first.legal_actions, second.legal_actions);
    }
}

#[test]
fn manual_play_automatic_play_and_replay_share_the_same_transitions() {
    let mut config = match_config(1_000);
    // This deal exercises dead-card exchanges with the scripted policy.
    config.game.shuffle_seed = 0;
    let mut manual = Match::new(config.clone(), true);
    let mut actions = Vec::new();
    let mut saw_exchange = false;

    while let Some(view) = manual.active_view() {
        let player = view.player;
        let action = PreferExchange.choose_action(&view).unwrap();
        actions.push(Some(manual.step(player, action).unwrap()));

        if matches!(action, Action::Exchange(_)) {
            assert_eq!(manual.current_player(), Some(player));
            saw_exchange = true;
        }
    }
    assert!(saw_exchange);
    assert!(matches!(manual.end(), Some(MatchEnd::Completed(_))));

    let mut automatic = Match::new(config.clone(), true);
    let mut observed = Vec::new();
    automatic
        .run(
            [&mut PreferExchange, &mut PreferExchange],
            |applied, state| {
                assert_eq!(state.action_count(), observed.len());
                observed.push(applied);
            },
        )
        .unwrap();

    assert_eq!(observed[0], None);
    assert_eq!(observed[1..], actions);
    assert_same_position(&manual, &automatic);
    let report = automatic.into_report().unwrap();
    let replay = report.replay.as_ref().unwrap();
    let mut playback_events = Vec::new();
    let replayed = replay
        .play(|applied, state| {
            if applied.is_none() {
                assert_same_position(state, &Match::new(config.clone(), false));
            }
            playback_events.push(applied);
        })
        .unwrap();

    assert_eq!(playback_events, observed);
    assert_eq!(replayed.end(), Some(report.end));
    assert_eq!(replayed.action_count(), report.action_count);
    assert_same_position(&replayed, &manual);

    let mut exact_limit = replay.clone();
    exact_limit.config.action_limit = NonZeroUsize::new(report.action_count).unwrap();
    let last_action_finished = exact_limit.play(|_, _| {}).unwrap();
    assert_same_position(&last_action_finished, &replayed);
    assert_eq!(report, manual.into_report().unwrap());
}

#[test]
fn rejected_actions_leave_the_match_unchanged_and_the_limit_truncates() {
    let config = match_config(1);
    let mut game_match = Match::new(config.clone(), true);
    let mut observed = Vec::new();
    let failure = game_match.run([&mut PreferExchange, &mut MissingAction], |applied, _| {
        observed.push(applied);
    });

    assert_eq!(
        failure,
        Err(MatchError::MissingAction {
            player: Player::Two
        })
    );
    assert_eq!(observed, [None]);
    for (player, source) in [
        (Player::One, ActionError::OutOfTurn),
        (Player::Two, ActionError::IllegalAction),
    ] {
        assert_eq!(
            game_match.step(player, Action::Pass),
            Err(MatchError::InvalidAction {
                player,
                action: Action::Pass,
                source,
            })
        );
    }
    assert_same_position(&game_match, &Match::new(config, false));

    let applied = game_match.step_policy(&mut PreferExchange).unwrap();
    assert_eq!(game_match.end(), Some(MatchEnd::ActionLimit));
    assert!(game_match.active_view().is_none());
    assert!(game_match.game().current_player().is_some());
    assert_eq!(game_match.game().result(), None);
    assert_eq!(
        game_match.step(applied.player, applied.action),
        Err(MatchError::Finished(MatchEnd::ActionLimit))
    );
    assert_eq!(
        game_match.into_report().unwrap().replay.unwrap().actions,
        [applied]
    );
}

#[test]
fn replay_checks_action_legality_length_and_ending() {
    let mut game_match = Match::new(match_config(1), true);
    let applied = game_match.step_policy(&mut PreferExchange).unwrap();
    let mut replay = game_match.into_report().unwrap().replay.unwrap();

    replay.actions[0].action = Action::Pass;
    assert_eq!(
        replay.play(|_, _| {}).err(),
        Some(ReplayError::InvalidStep {
            action_number: 1,
            source: MatchError::InvalidAction {
                player: Player::Two,
                action: Action::Pass,
                source: ActionError::IllegalAction,
            }
        })
    );

    replay.actions.clear();
    assert_eq!(
        replay.play(|_, _| {}).err(),
        Some(ReplayError::EndMismatch {
            expected: MatchEnd::ActionLimit,
            actual: None,
        })
    );

    replay.actions.extend([applied, applied]);
    assert_eq!(
        replay.play(|_, _| {}).err(),
        Some(ReplayError::InvalidStep {
            action_number: 2,
            source: MatchError::Finished(MatchEnd::ActionLimit),
        })
    );

    replay.actions.pop();
    replay.end = MatchEnd::Completed(Outcome::Draw);
    assert_eq!(
        replay.play(|_, _| {}).err(),
        Some(ReplayError::EndMismatch {
            expected: replay.end,
            actual: Some(MatchEnd::ActionLimit),
        })
    );
}

#[test]
fn paired_evaluation_swaps_seats_and_repeats_with_fresh_seeded_policies() {
    let config = evaluation_config(1_000);
    let mut first_seeds = Vec::new();
    let mut second_seeds = Vec::new();
    let report = evaluate(
        &config,
        |seed| {
            first_seeds.push(seed);
            RandomAgent::new(seed)
        },
        |seed| {
            second_seeds.push(seed);
            HeuristicAgent::new(seed)
        },
    )
    .unwrap();
    assert_eq!(
        report,
        evaluate(&config, RandomAgent::new, HeuristicAgent::new).unwrap()
    );
    assert_eq!(report.config, config);
    assert_eq!(report.pairs.len(), config.shuffle_seeds.len());
    assert_eq!(first_seeds.len(), 2 * report.pairs.len());
    assert_eq!(second_seeds.len(), 2 * report.pairs.len());

    let mut totals = EvaluationTotals::default();
    for (index, [first, second]) in report.pairs.iter().enumerate() {
        let seed_index = 2 * index;
        let seeds = [
            Some(first_seeds[seed_index]),
            Some(second_seeds[seed_index]),
        ];
        assert_eq!(first.config.policy_seeds, seeds);
        assert_eq!(second.config.policy_seeds, [seeds[1], seeds[0]]);
        assert_eq!(
            second.config.policy_seeds,
            [
                Some(second_seeds[seed_index + 1]),
                Some(first_seeds[seed_index + 1])
            ]
        );
        assert_eq!(first.config.game.policies, config.policy_ids);
        assert_eq!(
            second.config.game.policies,
            [config.policy_ids[1].clone(), config.policy_ids[0].clone()]
        );

        for game in [first, second] {
            assert_eq!(game.config.game.shuffle_seed, config.shuffle_seeds[index]);
            assert_eq!(game.config.game.starting_player, config.starting_player);
            assert_eq!(
                game.replay.as_ref().unwrap().actions.len(),
                game.action_count
            );

            match game.end {
                MatchEnd::Completed(Outcome::Win(player)) => {
                    let winner = &game.config.game.policies[player.index()];
                    let index = config
                        .policy_ids
                        .iter()
                        .position(|id| id == winner)
                        .unwrap();
                    totals.wins[index] += 1;
                }
                MatchEnd::Completed(Outcome::Draw) => totals.draws += 1,
                MatchEnd::ActionLimit => totals.truncations += 1,
            }
        }
    }
    assert_eq!(report.totals, totals);
    assert!(totals.wins.iter().sum::<usize>() > 0);
}

#[test]
fn evaluation_reports_truncation_and_identifies_a_policy_failure_in_the_swapped_game() {
    let mut config = evaluation_config(1);
    config.capture_replays = false;

    let report = evaluate(&config, RandomAgent::new, HeuristicAgent::new).unwrap();
    assert_eq!(
        report.totals,
        EvaluationTotals {
            truncations: 2 * config.shuffle_seeds.len(),
            ..Default::default()
        }
    );
    assert!(
        report
            .pairs
            .iter()
            .flatten()
            .all(|game| game.replay.is_none())
    );

    assert_eq!(
        evaluate(&config, |_| MissingAction, RandomAgent::new),
        Err(EvaluationError {
            pair_index: 0,
            shuffle_seed: config.shuffle_seeds[0],
            policy_order: [1, 0],
            source: MatchError::MissingAction {
                player: Player::Two
            },
        })
    );
}
