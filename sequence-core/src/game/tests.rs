use super::*;
use crate::{Cell, get_card, positions_for_card};

fn p(row: u8, col: u8) -> Position {
    Position::new(row, col).unwrap()
}

fn jack(suit: CardSuit) -> Card {
    Card::new(CardRank::Jack, suit)
}

fn hand(game: &mut State, cards: &[Card]) {
    game.hands[game.current_player as usize].clear();
    game.hands[game.current_player as usize].extend(cards.iter().copied());
}

#[test]
fn jack_effects() {
    for suit in CardSuit::VARIANTS {
        let mut game = Classic::from_seed(7);
        let card = jack(*suit);
        hand(&mut game.state, &[card]);
        if card.is_one_eyed_jack() {
            game.state.board.place(p(4, 4), Color::Blue);
        }
        let action = Action::Play {
            card_index: 0,
            position: p(4, 4),
        };
        game.apply(action).unwrap();
        assert_eq!(
            game.state.board.cell(p(4, 4)),
            if card.is_one_eyed_jack() {
                Cell::Empty
            } else {
                Cell::Occupied(Color::Red)
            }
        );
    }
}

#[test]
fn illegal_actions_are_atomic_and_terminal_buffers_are_empty() {
    let mut game = Classic::from_seed(42);
    for action in [
        Action::Pass,
        Action::Exchange { card_index: 99 },
        Action::Play {
            card_index: 99,
            position: p(1, 1),
        },
        Action::Play {
            card_index: 0,
            position: p(0, 0),
        },
    ] {
        let before = format!("{game:?}");
        assert!(game.apply(action).is_err());
        assert_eq!(format!("{game:?}"), before);
    }
    let mut actions = vec![Action::Pass];
    game.legal_actions(&mut actions);
    assert!(!actions.contains(&Action::Pass));
    let count = actions.len();
    game.legal_actions(&mut actions);
    assert_eq!(actions.len(), count);
    game.state.outcome = GameOutcome::Draw;
    game.legal_actions(&mut actions);
    assert!(actions.is_empty());
    assert_eq!(game.apply(Action::Pass), Err(ActionError::GameOver));
}

#[test]
fn removal_cannot_target_own_chips_corners_or_sequences() {
    let mut game = Classic::from_seed(7);
    hand(&mut game.state, &[jack(CardSuit::Hearts)]);
    game.state.board.place(p(2, 2), Color::Red);
    for col in 0..5 {
        game.state.board.place(p(4, col), Color::Blue);
    }
    for position in [p(0, 0), p(1, 1), p(2, 2), p(4, 0)] {
        assert_eq!(
            game.validate(Action::Play {
                card_index: 0,
                position
            }),
            Err(ActionError::InvalidPlayPosition(position))
        );
    }
}

#[test]
fn ordinary_cards_match_only_empty_printed_squares() {
    let mut game = Simple::from_seed(7);
    let card = get_card(p(1, 1)).unwrap();
    hand(&mut game.state, &[card]);
    assert!(
        game.validate(Action::Play {
            card_index: 0,
            position: p(1, 2)
        })
        .is_err()
    );
    game.state.board.place(p(1, 1), Color::Blue);
    assert!(
        game.validate(Action::Play {
            card_index: 0,
            position: p(1, 1)
        })
        .is_err()
    );
    assert_eq!(
        game.validate(Action::Exchange { card_index: 0 }),
        Err(ActionError::ExchangeUnavailable)
    );
}

#[test]
fn one_dead_card_exchange_retains_turn_and_pass_counter() {
    let mut game = Classic::from_seed(7);
    let dead = get_card(p(1, 1)).unwrap();
    for position in positions_for_card(dead) {
        game.state.board.place(position, Color::Blue);
    }
    hand(&mut game.state, &[dead, dead]);
    game.state.draw_pile = vec![get_card(p(2, 2)).unwrap()];
    game.consecutive_passes = 3;
    let action = Action::Exchange { card_index: 0 };
    assert_eq!(
        game.validate(Action::Pass),
        Err(ActionError::UnexpectedPass)
    );
    game.apply(action).unwrap();
    assert_eq!(game.state.current_player, Color::Red);
    assert_eq!(game.state.turns, 0);
    assert_eq!(game.consecutive_passes, 3);
    assert!(game.exchanged_dead_card);
    assert_eq!(game.validate(action), Err(ActionError::ExchangeUnavailable));
    game.apply(Action::Play {
        card_index: 1,
        position: p(2, 2),
    })
    .unwrap();
    assert_eq!(game.state.current_player, Color::Blue);
    assert_eq!(game.consecutive_passes, 0);
    assert!(!game.exchanged_dead_card);
}

#[test]
fn live_cards_and_jacks_cannot_be_exchanged() {
    let mut game = Classic::from_seed(7);
    for card in [
        get_card(p(1, 1)).unwrap(),
        jack(CardSuit::Clubs),
        jack(CardSuit::Hearts),
    ] {
        hand(&mut game.state, &[card]);
        assert_eq!(
            game.validate(Action::Exchange { card_index: 0 }),
            Err(ActionError::CardNotDead)
        );
    }
}

#[test]
fn forced_passes_end_the_game_and_reset_exchange_availability() {
    let mut game = Classic::from_seed(7);
    for cards in &mut game.state.hands {
        cards.clear();
        cards.push(jack(CardSuit::Hearts));
    }
    for i in 1..=6 {
        game.exchanged_dead_card = true;
        let mut actions = vec![];
        game.legal_actions(&mut actions);
        assert_eq!(actions, [Action::Pass]);
        game.apply(Action::Pass).unwrap();
        assert_eq!(game.state.turns, i);
        assert!(!game.exchanged_dead_card);
    }
    assert_eq!(game.outcome(), GameOutcome::Draw);
}

#[test]
fn reshuffle_preserves_cards_and_clone_rng_state() {
    let mut game = Classic::from_seed(7);
    hand(&mut game.state, &[jack(CardSuit::Clubs)]);
    game.state.draw_pile.clear();
    game.state.discard_pile = vec![get_card(p(1, 1)).unwrap(), get_card(p(1, 2)).unwrap()];
    let action = Action::Play {
        card_index: 0,
        position: p(4, 4),
    };
    let mut clone = game.clone();
    game.apply(action).unwrap();
    clone.apply(action).unwrap();
    assert_eq!(format!("{game:?}"), format!("{clone:?}"));
    assert!(game.state.discard_pile.is_empty());
    assert_eq!(game.state.draw_pile.len(), 2);
    assert_eq!(game.state.hands[0].len(), 1);
}

#[test]
fn victory_precedes_turn_limit_and_limits_end_otherwise() {
    let mut game = Classic::from_seed(7);
    hand(&mut game.state, &[jack(CardSuit::Clubs)]);
    for col in (0..9).filter(|&col| col != 4) {
        game.state.board.place(p(4, col), Color::Red);
    }
    game.state.turns = Classic::MAX_TURNS - 1;
    let action = Action::Play {
        card_index: 0,
        position: p(4, 4),
    };
    game.apply(action).unwrap();
    assert_eq!(game.outcome(), GameOutcome::Won(Color::Red));
    let mut game = Classic::from_seed(7);
    game.state.turns = Classic::MAX_TURNS - 1;
    let mut actions = vec![];
    game.legal_actions(&mut actions);
    game.apply(actions[0]).unwrap();
    assert_eq!(game.outcome(), GameOutcome::Draw);
}

#[test]
fn simple_does_not_recycle_and_empty_hands_draw() {
    let mut game = Simple::from_seed(7);
    let card = get_card(p(1, 1)).unwrap();
    hand(&mut game.state, &[card]);
    game.state.hands[1].clear();
    game.state.draw_pile.clear();
    let action = Action::Play {
        card_index: 0,
        position: p(1, 1),
    };
    game.apply(action).unwrap();
    assert!(game.state.hands[0].is_empty());
    assert_eq!(game.state.discard_pile, [card]);
    assert_eq!(game.outcome(), GameOutcome::Draw);
}

#[test]
fn observations_do_not_depend_on_hidden_card_order_or_rng() {
    let game = Classic::from_seed(7);
    let mut other = game.clone();
    other.state.draw_pile.reverse();
    other.state.hands[1].reverse();
    other.rng = Xoroshiro128PlusPlus::seed_from_u64(99);
    assert_eq!(
        format!("{:?}", game.observation()),
        format!("{:?}", other.observation())
    );
}

#[test]
fn seeded_games_terminate_preserve_cards_and_match_validation() {
    for seed in 0..24 {
        check_seeded::<Simple>(seed, |game| &game.state, false);
        check_seeded::<Classic>(seed, |game| &game.state, true);
    }
}

fn check_seeded<G: Game + std::fmt::Debug>(seed: u64, state: fn(&G) -> &State, has_jacks: bool) {
    let mut game = G::from_seed(seed);
    assert_eq!(format!("{game:?}"), format!("{:?}", G::from_seed(seed)));
    let mut actions = vec![];
    let mut decisions = 0;
    while !game.is_game_over() {
        game.legal_actions(&mut actions);
        assert!(!actions.is_empty());
        for index in 0..state(&game).hands[game.current_player() as usize].len() {
            for position in Position::all() {
                let action = Action::Play {
                    card_index: index,
                    position,
                };
                assert_eq!(game.validate(action).is_ok(), actions.contains(&action));
            }
            let action = Action::Exchange { card_index: index };
            assert_eq!(game.validate(action).is_ok(), actions.contains(&action));
        }
        assert_eq!(
            game.validate(Action::Pass).is_ok(),
            actions.contains(&Action::Pass)
        );
        let action = actions[(seed as usize * 17 + decisions * 31) % actions.len()];
        game.apply(action).unwrap();
        let mut counts = [0; Card::COUNT];
        for card in state(&game)
            .hands
            .iter()
            .flatten()
            .chain(&state(&game).draw_pile)
            .chain(&state(&game).discard_pile)
        {
            counts[card.id()] += 1;
        }
        for &rank in CardRank::VARIANTS {
            for &suit in CardSuit::VARIANTS {
                let card = Card::new(rank, suit);
                assert_eq!(
                    counts[card.id()],
                    if !has_jacks && rank == CardRank::Jack {
                        0
                    } else {
                        2
                    }
                );
            }
        }
        decisions += 1;
        assert!(decisions <= 600);
    }
    game.legal_actions(&mut actions);
    assert!(actions.is_empty());
}
