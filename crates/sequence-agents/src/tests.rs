use sequence_game::{Action, Card, Game, GameConfig, Player, Rank, Suit};

use crate::{HeuristicAgent, Policy, RandomAgent};

fn game(shuffle_seed: u64) -> Game {
    Game::new(GameConfig {
        shuffle_seed,
        starting_player: Player::One,
        policies: ["random".into(), "heuristic".into()],
    })
}

#[test]
fn random_agent_repeats_legal_choices_with_the_same_seed() {
    let game = game(42);
    let view = game.view(Player::One);
    let mut first = RandomAgent::new(7);
    let mut second = RandomAgent::new(7);

    for _ in 0..32 {
        let action = first.choose_action(&view).unwrap();
        assert!(view.legal_actions.contains(&action));
        assert_eq!(second.choose_action(&view), Some(action));
    }

    assert_eq!(first.choose_action(&game.view(Player::Two)), None);
}

#[test]
fn heuristic_saves_a_jack_when_a_matching_card_makes_the_same_play() {
    let wild = Card::new(Rank::Jack, Suit::Clubs);
    let game = (0..64)
        .map(game)
        .find(|game| game.view(Player::One).hand.contains(&wild))
        .unwrap();
    let mut view = game.view(Player::One);
    let (matching_play, target) = view
        .legal_actions
        .iter()
        .find_map(|&action| match action {
            Action::Play { card, target } if card.rank != Rank::Jack => Some((action, target)),
            _ => None,
        })
        .unwrap();
    let wild_play = Action::Play { card: wild, target };
    assert!(view.legal_actions.contains(&wild_play));
    view.legal_actions = vec![wild_play, matching_play];

    assert_eq!(
        HeuristicAgent::new(7).choose_action(&view),
        Some(matching_play)
    );
}
