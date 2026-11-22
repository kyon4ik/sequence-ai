use std::{env, error::Error, num::NonZeroUsize};

use sequence_agents::{HeuristicAgent, RandomAgent};
use sequence_game::{Outcome, Player};
use sequence_match::{EvaluationConfig, MatchEnd, evaluate};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let (count, inspect) = match args.as_slice() {
        [count] => (count.parse::<NonZeroUsize>()?.get(), None),
        [cmd, seed, seat] if cmd == "inspect" => {
            let seed = seed.parse::<usize>()?;
            let seat = match seat.as_str() {
                "first" => 0,
                "second" => 1,
                _ => return Err("heuristic seat must be first or second".into()),
            };
            (
                seed.checked_add(1).ok_or("seed too large")?,
                Some((seed, seat)),
            )
        }
        _ => {
            return Err(
                "usage: baseline PAIRS | baseline inspect SHUFFLE_SEED first|second".into(),
            );
        }
    };

    let config = EvaluationConfig {
        policy_ids: ["heuristic".into(), "random".into()],
        shuffle_seeds: (0..count as u64).collect(),
        starting_player: Player::One,
        policy_seed: 42,
        action_limit: NonZeroUsize::new(512).unwrap(),
        capture_replays: inspect.is_some(),
    };
    let report = evaluate(&config, HeuristicAgent::new, RandomAgent::new)?;
    if let Some((seed, seat)) = inspect {
        let replay = report.pairs[seed][seat].replay.as_ref().unwrap();
        let mut actions = replay.actions.iter();
        replay.play(|_, game_match| {
            if let Some(next) = actions.next() {
                let view = game_match.game().view(next.player);
                println!(
                    "\n{:?} ({})\nboard {:?}\nhand {:?}\nlegal {:?}\nchosen {:?}",
                    next.player,
                    replay.config.game.policies[next.player.index()],
                    view.board,
                    view.hand,
                    view.legal_actions,
                    next.action
                );
            }
        })?;
        println!("{:?}", replay.end);
        return Ok(());
    }

    println!(
        "shuffle_seed,first_policy,second_policy,first_policy_seed,second_policy_seed,outcome,actions"
    );
    for game in report.pairs.iter().flatten() {
        let [first, second] = &game.config.game.policies;
        let end = match game.end {
            MatchEnd::Completed(Outcome::Win(player)) => &game.config.game.policies[player.index()],
            MatchEnd::Completed(Outcome::Draw) => "draw",
            MatchEnd::ActionLimit => "truncated",
        };
        let [first_seed, second_seed] = game.config.policy_seeds.map(Option::unwrap);
        println!(
            "{},{first},{second},{first_seed},{second_seed},{end},{}",
            game.config.game.shuffle_seed, game.action_count
        );
    }
    Ok(())
}
