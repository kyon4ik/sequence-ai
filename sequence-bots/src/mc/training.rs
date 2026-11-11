use super::{McBot, ValueNet, model};
use burn::{
    optim::{AdamConfig, GradientsParams},
    prelude::*,
};
use rand::{RngExt, SeedableRng, rngs::SmallRng, seq::SliceRandom};
use sequence_core::{Color, Game, GameOutcome, Simple};

#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub iterations: usize,
    pub games_per_iteration: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            games_per_iteration: 128,
            batch_size: 256,
            learning_rate: 3e-4,
            epsilon_start: 0.2,
            epsilon_end: 0.05,
            seed: 42,
        }
    }
}

/// Collect with frozen weights, then regress on actor-relative terminal outcomes.
pub fn train(mut net: ValueNet, device: Device, config: &TrainingConfig) -> ValueNet {
    assert!(config.games_per_iteration > 0 && config.batch_size > 0);
    assert!(config.learning_rate.is_finite() && config.learning_rate > 0.);
    assert!((0.0..=1.0).contains(&config.epsilon_start));
    assert!((0.0..=1.0).contains(&config.epsilon_end));

    let device = device.autodiff();
    net = net.train();
    let mut optimizer = AdamConfig::new().init();
    let mut rng = SmallRng::seed_from_u64(config.seed);

    for iteration in 0..config.iterations {
        let progress = iteration as f64 / config.iterations.saturating_sub(1).max(1) as f64;
        let epsilon = config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start);

        let mut bot = McBot::new(net.clone(), device.clone(), rng.random(), epsilon);
        let mut samples = Vec::new();
        let mut results = [0usize; 3];

        for _ in 0..config.games_per_iteration {
            let mut game = Simple::from_seed(rng.random());
            let mut trajectory = Vec::new();
            let mut actions = Vec::new();

            while !game.is_game_over() {
                game.legal_actions(&mut actions);
                let actor = game.current_player();
                let (action, encoded) = bot.select(game.observation(), &actions);
                trajectory.push((encoded, actor));
                game.apply(action).unwrap();
            }

            let outcome = game.outcome();
            results[match outcome {
                GameOutcome::Won(c) => c as usize,
                GameOutcome::Draw => 2,
                _ => unreachable!(),
            }] += 1;

            // No opponent-value negation or bootstrap: each target uses the final result.
            samples.extend(trajectory.into_iter().map(|(encoded, actor)| {
                let target = terminal_target(outcome, actor);
                (encoded, target)
            }));
        }

        samples.shuffle(&mut rng);

        let mut total_loss = 0.;
        for minibatch in samples.chunks(config.batch_size) {
            let inputs: Vec<_> = minibatch.iter().map(|(x, _)| x.clone()).collect();
            let targets: Vec<_> = minibatch.iter().map(|(_, y)| *y).collect();
            let (board, context) = model::batch(&inputs, &device);
            let targets =
                Tensor::<2>::from_data(TensorData::new(targets, [inputs.len(), 1]), &device);
            let error = net.forward(board, context) - targets;
            let loss = (error.clone() * error).mean();
            total_loss += loss.clone().into_scalar::<f32>() as f64 * inputs.len() as f64;
            let gradients = GradientsParams::from_grads(loss.backward(), &net);
            net = optimizer.step(config.learning_rate, net, gradients);
        }

        println!(
            "iteration={} samples={} mse={:.4} epsilon={epsilon:.3} red/blue/draw={results:?}",
            iteration + 1,
            samples.len(),
            total_loss / samples.len() as f64
        );
    }

    net.valid()
}

fn terminal_target(outcome: GameOutcome, actor: Color) -> f32 {
    match outcome {
        GameOutcome::Won(winner) if winner == actor => 1.,
        GameOutcome::Won(_) => -1.,
        GameOutcome::Draw => 0.,
        GameOutcome::InProgress => panic!("Monte Carlo targets require a completed game"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mc::EncodedAfterstate;

    #[test]
    fn targets_use_the_actor_perspective() {
        for winner in [Color::Red, Color::Blue] {
            assert_eq!(terminal_target(GameOutcome::Won(winner), winner), 1.);
            assert_eq!(
                terminal_target(GameOutcome::Won(winner), winner.next()),
                -1.
            );
            assert_eq!(terminal_target(GameOutcome::Draw, winner), 0.);
        }
    }

    #[test]
    fn completed_game_updates_weights_and_checkpoint_roundtrips() {
        let device = Device::flex();
        device.seed(7);
        let net = ValueNet::new(&device);
        let game = Simple::from_seed(7);
        let mut actions = Vec::new();
        game.legal_actions(&mut actions);
        let input = EncodedAfterstate::new(game.observation(), actions[0]).unwrap();
        let predict = |net: &ValueNet| {
            let (board, context) = model::batch(std::slice::from_ref(&input), &device);
            net.forward(board, context).into_scalar::<f32>()
        };
        let before = predict(&net);
        let net = train(
            net,
            device.clone(),
            &TrainingConfig {
                iterations: 1,
                games_per_iteration: 1,
                batch_size: 128,
                epsilon_start: 1.,
                epsilon_end: 1.,
                seed: 7,
                ..Default::default()
            },
        );
        let after = predict(&net);
        assert!(after.is_finite() && (-1.0..=1.0).contains(&after));
        assert!((before - after).abs() > 1e-6);
        let path =
            std::env::temp_dir().join(format!("sequence-mc-test-{}.bpk", std::process::id()));
        net.clone().save_file(&path).unwrap();
        let loaded = ValueNet::new(&device).load_file(&path);
        assert!((predict(&loaded) - after).abs() < 1e-6);
        // Keep this round-trip test on its CPU backend; the bot factory uses CUDA.
        let mut bot = McBot::new(loaded, device.clone(), 7, 0.0);
        let selected = bot.select(game.observation(), &actions).0;
        assert!(actions.contains(&selected));
        std::fs::remove_file(path).unwrap();
    }
}
