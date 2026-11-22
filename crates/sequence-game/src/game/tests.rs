use super::*;
use crate::{BOARD_SIZE, Rank, Space, Suit};

const ACE: Card = Card::new(Rank::Ace, Suit::Clubs);
const TWO: Card = Card::new(Rank::Two, Suit::Clubs);
const WILD: Card = Card::new(Rank::Jack, Suit::Clubs);
const REMOVAL: Card = Card::new(Rank::Jack, Suit::Hearts);

fn config(starting_player: Player) -> GameConfig {
    GameConfig {
        shuffle_seed: 42,
        starting_player,
        policies: ["first".into(), "second".into()],
    }
}

fn position(row: usize, column: usize) -> Position {
    Position::new(row, column).unwrap()
}

fn play(card: Card, row: usize, column: usize) -> Action {
    Action::Play {
        card,
        target: position(row, column),
    }
}

#[test]
fn board_layout_labels_two_spaces_per_non_jack() {
    let game = Game::new(config(Player::One));
    let board = game.view(Player::One).board;

    for target in Position::all() {
        let reflected = position(
            BOARD_SIZE - 1 - target.row(),
            BOARD_SIZE - 1 - target.column(),
        );
        assert_eq!(board.space(target), board.space(reflected));
        assert_eq!(board.space(target) == Space::Corner, target.is_corner());
    }

    for card in Card::all() {
        let matching = Position::all()
            .filter(|&target| board.space(target) == Space::Labeled { card, chip: None })
            .collect::<Vec<_>>();
        if card.rank == Rank::Jack {
            assert!(matching.is_empty());
        } else {
            assert_eq!(
                Board::matching_positions(card).unwrap(),
                matching.as_slice()
            );
            assert_eq!(matching.len(), 2);
        }
    }
}

#[test]
fn setup_deals_seeded_deck_in_seat_order() {
    let game_config = config(Player::Two);
    let game = Game::new(game_config.clone());
    let one = game.view(Player::One);
    let two = game.view(Player::Two);
    let mut shuffled: Vec<_> = Card::all().chain(Card::all()).collect();
    shuffled.shuffle(&mut ChaCha8Rng::seed_from_u64(game_config.shuffle_seed));

    assert_eq!(one.current_player, Some(Player::Two));
    assert_eq!(one.hand_sizes, [7, 7]);
    assert_eq!(one.hand, &shuffled[..7]);
    assert_eq!(two.hand, &shuffled[7..14]);
    assert_eq!(
        game.draw_pile.iter().copied().collect::<Vec<_>>(),
        &shuffled[14..]
    );
    assert_eq!(one.draw_pile_size, 90);
    assert_eq!(one.discard_piles, [&[] as &[Card]; 2]);
    assert_eq!(one.claimed_sequences, [&[] as &[Sequence]; 2]);
    assert_eq!(one.consecutive_passes, 0);
    assert!(one.legal_actions.is_empty());
    assert!(!two.legal_actions.is_empty());
    assert!(game.result().is_none());
}

#[test]
fn card_effects_validate_turn_card_and_target() {
    let mut game = Game::new(config(Player::One));
    game.hands = [vec![ACE, ACE, WILD], vec![REMOVAL]];
    game.consecutive_passes = 2;

    let regular = play(ACE, 0, 1);
    let actions = game.view(Player::One).legal_actions;
    assert_eq!(
        actions.iter().filter(|&&action| action == regular).count(),
        1
    );
    assert!(!actions.contains(&play(WILD, 0, 0)));
    assert_eq!(
        game.apply(Player::Two, regular),
        Err(ActionError::OutOfTurn)
    );
    assert_eq!(
        game.apply(Player::One, play(ACE, 0, 2)),
        Err(ActionError::IllegalAction)
    );
    assert_eq!(
        game.apply(Player::One, play(WILD, 0, 0)),
        Err(ActionError::IllegalAction)
    );
    assert_eq!(
        game.apply(Player::One, play(TWO, 0, 2)),
        Err(ActionError::IllegalAction)
    );
    assert_eq!(game.view(Player::One).hand, &[ACE, ACE, WILD]);

    game.apply(Player::One, regular).unwrap();
    assert_eq!(game.view(Player::Two).consecutive_passes, 0);
    assert_eq!(game.view(Player::Two).discard_piles[0], &[ACE]);
    assert_eq!(game.board.chip(position(0, 1)), Some(Player::One));

    let removal = play(REMOVAL, 0, 1);
    let view = game.view(Player::Two);
    assert!(view.preview_play(play(REMOVAL, 0, 2)).is_none());
    let preview = view.preview_play(removal).unwrap();
    assert!(preview.new_claims.is_empty());
    assert!(!preview.wins);

    game.apply(Player::Two, removal).unwrap();
    assert_eq!(&preview.board, game.view(Player::One).board);
    assert_eq!(game.board.chip(position(0, 1)), None);
    game.apply(Player::One, play(WILD, 0, 2)).unwrap();
    assert_eq!(game.board.chip(position(0, 2)), Some(Player::One));
    assert_eq!(
        game.view(Player::Two).discard_piles,
        [&[ACE, WILD][..], &[REMOVAL]]
    );
}

#[test]
fn exchange_can_precede_a_legal_play_and_replenishes_from_both_discards() {
    let mut game = Game::new(config(Player::One));
    game.hands[0] = vec![ACE, WILD];
    game.draw_pile = VecDeque::from([TWO]);
    game.discards = [vec![REMOVAL], vec![WILD]];
    game.consecutive_passes = 1;
    for &target in Board::matching_positions(ACE).unwrap() {
        game.board.set_chip(target, Some(Player::Two));
    }

    let mut replenished = [REMOVAL, ACE, WILD];
    replenished.shuffle(&mut game.rng.clone());
    assert!(
        game.view(Player::One)
            .legal_actions
            .contains(&Action::Exchange(ACE))
    );
    assert!(
        game.view(Player::One)
            .legal_actions
            .contains(&play(WILD, 0, 2))
    );

    game.apply(Player::One, Action::Exchange(ACE)).unwrap();
    let view = game.view(Player::One);
    assert_eq!(view.current_player, Some(Player::One));
    assert_eq!(view.consecutive_passes, 1);
    assert_eq!(view.hand, &[WILD, TWO]);
    assert_eq!(view.draw_pile_size, 3);
    assert_eq!(view.discard_piles, [&[] as &[Card]; 2]);
    assert_eq!(
        game.draw_pile.iter().copied().collect::<Vec<_>>(),
        replenished
    );

    game.apply(Player::One, play(TWO, 0, 2)).unwrap();
    assert_eq!(game.view(Player::One).hand, &[WILD, replenished[0]]);
    assert_eq!(game.current_player(), Some(Player::Two));
}

#[test]
fn mandatory_exchange_allows_only_one_attempt_per_turn_and_three_passes_draw() {
    let game_config = config(Player::One);
    let mut game = Game::new(game_config.clone());
    game.hands = [vec![ACE], vec![REMOVAL]];
    game.draw_pile = VecDeque::from([ACE, ACE, WILD]);
    for &target in Board::matching_positions(ACE).unwrap() {
        game.board.set_chip(target, Some(Player::Two));
    }

    for passes in [0, 2] {
        assert_eq!(game.view(Player::One).consecutive_passes, passes);
        assert_eq!(
            game.view(Player::One).legal_actions,
            [Action::Exchange(ACE)]
        );
        assert_eq!(
            game.apply(Player::One, Action::Pass),
            Err(ActionError::IllegalAction)
        );
        game.apply(Player::One, Action::Exchange(ACE)).unwrap();
        assert_eq!(game.view(Player::One).consecutive_passes, passes);
        assert_eq!(game.view(Player::One).legal_actions, [Action::Pass]);
        assert_eq!(
            game.apply(Player::One, Action::Exchange(ACE)),
            Err(ActionError::IllegalAction)
        );
        game.apply(Player::One, Action::Pass).unwrap();

        if passes == 0 {
            assert_eq!(game.view(Player::Two).legal_actions, [Action::Pass]);
            assert_eq!(
                game.apply(Player::Two, Action::Exchange(REMOVAL)),
                Err(ActionError::IllegalAction)
            );
            game.apply(Player::Two, Action::Pass).unwrap();
        }
    }

    assert_eq!(game.view(Player::One).consecutive_passes, 3);
    assert_eq!(game.current_player(), None);
    assert_eq!(
        game.result(),
        Some(GameResult {
            config: game_config,
            outcome: Outcome::Draw
        })
    );
    assert_eq!(
        game.apply(Player::Two, Action::Pass),
        Err(ActionError::GameOver)
    );
}

#[test]
fn placement_claims_the_whole_largest_group_and_wins_before_drawing() {
    let game_config = config(Player::One);
    let mut game = Game::new(game_config.clone());
    game.hands[0] = vec![WILD];
    for index in 0..4 {
        game.board.set_chip(position(4, index), Some(Player::One));
        game.board
            .set_chip(position(4, index + 5), Some(Player::One));
        game.board.set_chip(position(index, 4), Some(Player::One));
        game.board
            .set_chip(position(index, 8 - index), Some(Player::One));
        if index != 0 {
            game.board
                .set_chip(position(index, index), Some(Player::One));
        }
    }

    let winning_play = play(WILD, 4, 4);
    let preview = game.view(Player::One).preview_play(winning_play).unwrap();
    assert!(preview.wins);

    game.apply(Player::One, winning_play).unwrap();
    let view = game.view(Player::One);
    assert_eq!(&preview.board, view.board);
    assert_eq!(preview.new_claims, view.claimed_sequences[0]);
    let lines: Vec<_> = view.claimed_sequences[0]
        .iter()
        .map(|sequence| *sequence.spaces())
        .collect();
    let expected = [
        std::array::from_fn(|index| position(index, index)),
        std::array::from_fn(|index| position(index, 4)),
        std::array::from_fn(|index| position(index, 8 - index)),
        std::array::from_fn(|index| position(4, index)),
        std::array::from_fn(|index| position(4, index + 4)),
    ];

    assert_eq!(lines, expected);
    assert_eq!(view.draw_pile_size, 90);
    assert!(view.hand.is_empty());
    assert!(view.legal_actions.is_empty());
    assert_eq!(
        game.result(),
        Some(GameResult {
            config: game_config,
            outcome: Outcome::Win(Player::One)
        })
    );
}

#[test]
fn claims_break_ties_in_board_order_and_protect_only_claimed_chips() {
    let mut game = Game::new(config(Player::One));
    game.hands = [vec![WILD, WILD], vec![REMOVAL]];
    for column in [1, 2, 4, 5, 6, 7] {
        game.board.set_chip(position(1, column), Some(Player::One));
    }

    game.apply(Player::One, play(WILD, 1, 3)).unwrap();
    let claims = game.view(Player::Two).claimed_sequences[0];
    assert_eq!(claims.len(), 1);
    assert_eq!(
        *claims[0].spaces(),
        std::array::from_fn(|index| position(1, index + 1))
    );
    assert_eq!(
        game.apply(Player::Two, play(REMOVAL, 1, 5)),
        Err(ActionError::IllegalAction)
    );
    game.apply(Player::Two, play(REMOVAL, 1, 7)).unwrap();

    game.apply(Player::One, play(WILD, 1, 0)).unwrap();
    assert_eq!(game.view(Player::One).claimed_sequences[0].len(), 1);
    assert!(game.result().is_none());
}

#[test]
fn a_seeded_game_runs_to_completion_using_player_views() {
    let mut game = Game::new(config(Player::Two));

    for _ in 0..1000 {
        let Some(player) = game.current_player() else {
            break;
        };
        let view = game.view(player);
        let cards_in_game = view.draw_pile_size
            + view.hand_sizes.into_iter().sum::<usize>()
            + view
                .discard_piles
                .into_iter()
                .map(<[Card]>::len)
                .sum::<usize>();
        assert_eq!(cards_in_game, 104);
        game.apply(player, view.legal_actions[0]).unwrap();
    }

    assert!(game.result().is_some());
}
