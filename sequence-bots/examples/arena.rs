use rand::{RngExt, SeedableRng, rngs::SmallRng};
use sequence_bots::BotKind;
use sequence_core::{GameOutcome, NoJacksGame};
use std::time::Instant;

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let games = args
        .first()
        .map(|n| n.parse::<usize>().expect("game count"))
        .unwrap_or(100);
    assert!(games > 0);
    if let Some(checkpoint) = args.get(1) {
        #[cfg(feature = "burn")]
        {
            let mc = BotKind::MonteCarlo {
                checkpoint: checkpoint.into(),
            };
            // Same game seeds, with the learned agent evaluated in both seats.
            for opponent in [BotKind::Random, BotKind::Greedy] {
                run_games(67, games, [mc.clone(), opponent.clone()], [43, 42]);
                run_games(67, games, [opponent, mc.clone()], [42, 43]);
            }
        }
        #[cfg(not(feature = "burn"))]
        panic!("checkpoint {checkpoint} requires --features burn");
    } else {
        run_games(67, games, [BotKind::Random, BotKind::Random], [43, 42]);
        run_games(67, games, [BotKind::Random, BotKind::Greedy], [43, 42]);
        run_games(67, games, [BotKind::Greedy, BotKind::Greedy], [43, 42]);
    }
}

fn run_games(seed: u64, n_games: usize, bot_kinds: [BotKind; 2], bot_seeds: [u64; 2]) {
    let mut game_rng = SmallRng::seed_from_u64(seed);
    let mut legal_actions = Vec::with_capacity(sequence_core::MAX_LEGAL_ACTIONS);

    let timer = Instant::now();

    let mut wins = [0; 2];
    let mut draws = 0;

    // Load checkpoints once per matchup, not once per game.
    let mut bots = [
        bot_kinds[0].create(bot_seeds[0]),
        bot_kinds[1].create(bot_seeds[1]),
    ];
    println!("Red: {:?}; Blue: {:?}", bot_kinds[0], bot_kinds[1]);

    let mut bot_timers = [0; 2];
    let mut bot_act_counts = [0; 2];
    for _ in 0..n_games {
        let mut game = NoJacksGame::from_seed(game_rng.random());

        while !game.is_game_over() {
            let bot_index = game.us() as usize;
            let obs = game.observation();
            game.legal_actions(&mut legal_actions);

            let t = Instant::now();
            let action = bots[bot_index].choose_action(obs, &legal_actions);
            bot_timers[bot_index] += t.elapsed().as_micros();
            bot_act_counts[bot_index] += 1;

            legal_actions.clear();
            game.apply_unchecked(action);
        }

        match game.outcome() {
            GameOutcome::Unknown => unreachable!(),
            GameOutcome::Won(color) => wins[color as usize] += 1,
            GameOutcome::Draw => draws += 1,
        }
    }

    println!(
        "Red avg. time: {}",
        bot_timers[0] as f64 / bot_act_counts[0] as f64
    );
    println!(
        "Blue avg. time: {}",
        bot_timers[1] as f64 / bot_act_counts[1] as f64
    );

    let execution_time = timer.elapsed();
    println!("{} games in {}s", n_games, execution_time.as_secs_f64());
    println!(
        "Red won {} | Blue won {} | Draws {}",
        wins[0], wins[1], draws,
    );
    println!();
}
