//! Controlled research continuations. These are deliberately separate from normal
//! resume: learning settings change explicitly and partial runs are not resumable.
use super::*;
use std::collections::VecDeque;

type Sample = (EncodedAfterstate, f32, OpponentKind);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Intervention {
    Control,
    LowLearningRate,
    Replay,
    HistoricalWeight,
    FixedExploration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AblationConfig {
    pub source: std::path::PathBuf,
    pub intervention: Intervention,
    pub seed: u64,
    pub steps: usize,
    pub games: usize,
    pub updates: usize,
    pub batch_size: usize,
    pub replay_capacity: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AblationRow {
    pub iteration: usize,
    pub game_seed_schedule: Vec<MatchConfig>,
    pub epsilon: f64,
    pub learning_rate: f64,
    pub fresh_samples: usize,
    pub optimization_samples: usize,
    pub historical_samples: usize,
    pub retained_samples: usize,
    pub replay_size: usize,
    pub mse: f64,
    pub seconds: f64,
}

fn collect_samples(
    records: &[sequence_runner::MatchRecord],
    plans: &[opponents::OpponentPlan],
    state: &mut OpponentState,
    sampling: &HistoricalSampling,
) -> Result<Vec<Sample>, sequence_agent::AgentError> {
    let mut samples = Vec::new();
    for (record, plan) in records.iter().zip(plans) {
        let Termination::Completed(outcome) = record.termination else {
            return Err("incomplete ablation game".into());
        };
        state.observe(
            plan,
            outcome_index(outcome, plan.learner_seat) == 0,
            sampling,
        )?;
        let mut game = Simple::from_seed(record.header.config.game_seed);
        for step in &record.steps {
            if plan.learns_from(step.actor as usize) {
                samples.push((
                    EncodedAfterstate::new(game.observation(), step.action)?,
                    terminal_target(outcome, step.actor),
                    plan.source,
                ));
            }
            game.apply(step.action)?;
        }
    }
    Ok(samples)
}

fn retain(buffer: &mut VecDeque<Sample>, fresh: &[Sample], capacity: usize) {
    for sample in fresh {
        buffer.push_back(sample.clone());
    }
    while buffer.len() > capacity {
        buffer.pop_front();
    }
}

fn choose_samples(
    fresh: &[Sample],
    buffer: &VecDeque<Sample>,
    mix: &OpponentMix,
    intervention: Intervention,
    count: usize,
    iteration: usize,
    seed: u64,
) -> Vec<(Sample, bool)> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(iteration_seed(seed, iteration, 0x5A4F_1E50));
    let retained = if intervention == Intervention::Replay && !buffer.is_empty() {
        count / 4
    } else {
        0
    };
    let sources: Vec<_> = fresh.iter().map(|s| s.2).collect();
    let strategy = if intervention == Intervention::HistoricalWeight {
        SamplingStrategy::OpponentMix
    } else {
        SamplingStrategy::Natural
    };
    let selected = sampling::select(
        &sources,
        mix,
        strategy,
        Some(count - retained),
        iteration,
        &mut rng,
    );
    let mut result: Vec<_> = selected
        .into_iter()
        .map(|i| (fresh[i].clone(), false))
        .collect();
    for _ in 0..retained {
        result.push((buffer[rng.random_range(0..buffer.len())].clone(), true));
    }
    result.shuffle(&mut rng);
    result
}

/// Start from a complete checkpoint, preserving Adam and archived policies.
/// Uses independent per-iteration game, opponent and sample RNG domains shared
/// across interventions. No existing model or source checkpoint is modified.
pub fn run_ablation(
    config: &AblationConfig,
    device: Device,
    destination: &Path,
) -> Result<Vec<AblationRow>, sequence_agent::AgentError> {
    if config.steps == 0
        || config.games < 5
        || config.updates == 0
        || config.batch_size == 0
        || config.replay_capacity == 0
        || config.updates.checked_mul(config.batch_size).is_none()
    {
        return Err("invalid ablation budget".into());
    }
    let metadata: ResumeMetadata = ResumeMetadata::load(&config.source)?;
    let completed = metadata.report.iterations.len();
    let learning = metadata.config;
    learning.validate()?;
    let mut state = metadata.opponents;
    if state.checkpoints != (0..=completed).collect::<Vec<_>>()
        || !learning.opponents.uses_history()
    {
        return Err("ablation requires complete historical archive".into());
    }
    std::fs::create_dir(destination)?;
    std::fs::write(
        destination.join("experiment.json"),
        serde_json::to_vec_pretty(config)?,
    )?;
    let device = device.autodiff();
    let mut net = ValueNet::new(&device)
        .try_load_file(config.source.join("model.bpk"))?
        .train();
    let mut optimizer = optimizer::adam().load(config.source.join("optimizer.bpk"))?;
    let mut pool = CheckpointPool::new()?;
    pool.restore(&config.source, &state.checkpoints)?;
    let mut buffer = VecDeque::new();
    let mut report = Vec::new();
    for iteration in completed..completed + config.steps {
        let started = Instant::now();
        let mut rng =
            Xoshiro256PlusPlus::seed_from_u64(iteration_seed(config.seed, iteration, 0xE915_0DE5));
        let matches: Vec<_> = (0..config.games)
            .map(|_| MatchConfig {
                game_seed: rng.random(),
                agent_seeds: [rng.random(), rng.random()],
                ..Default::default()
            })
            .collect();
        // Reset only the scheduling RNG per iteration, keeping learned qualities
        // and archived weights. This avoids RNG drift when trajectories diverge.
        state.rng =
            Xoshiro256PlusPlus::seed_from_u64(iteration_seed(config.seed, iteration, 0x0FF0_4E47));
        let plans = learning.opponents.schedule_with(
            config.games,
            iteration,
            &mut state,
            &learning.historical_sampling,
        );
        let epsilon = if config.intervention == Intervention::FixedExploration {
            learning.epsilon_at(completed.saturating_sub(1))
        } else {
            learning.epsilon_at(iteration)
        };
        let lr = learning.learning_rate
            / if config.intervention == Intervention::LowLearningRate {
                3.0
            } else {
                1.0
            };
        let rollout =
            rollout::play_opponents(&net, &device, &matches, 64, epsilon, &plans, Some(&pool))?;
        let fresh = collect_samples(
            &rollout.records,
            &plans,
            &mut state,
            &learning.historical_sampling,
        )?;
        let selected = choose_samples(
            &fresh,
            &buffer,
            &learning.opponents,
            config.intervention,
            config.updates * config.batch_size,
            iteration,
            config.seed,
        );
        let mut total_loss = Tensor::<1>::zeros([1], &device.clone().without_autodiff());
        for minibatch in selected.chunks(config.batch_size) {
            let inputs: Vec<_> = minibatch.iter().map(|(s, _)| s.0.clone()).collect();
            let targets: Vec<_> = minibatch.iter().map(|(s, _)| s.1).collect();
            let (board, context) = batch(&inputs, &device);
            let targets =
                Tensor::<2>::from_data(TensorData::new(targets, [inputs.len(), 1]), &device);
            let error = net.forward(board, context) - targets;
            let loss = (error.clone() * error).mean();
            total_loss = total_loss + loss.clone().without_autodiff() * inputs.len() as f32;
            let gradients = GradientsParams::from_grads(loss.backward(), &net);
            net = optimizer.step(lr, net, gradients);
        }
        if config.intervention == Intervention::Replay {
            retain(&mut buffer, &fresh, config.replay_capacity);
        }
        pool.insert(iteration + 1, &net)?;
        state.checkpoints.push(iteration + 1);
        report.push(AblationRow {
            iteration: iteration + 1,
            game_seed_schedule: matches,
            epsilon,
            learning_rate: lr,
            fresh_samples: fresh.len(),
            optimization_samples: selected.len(),
            historical_samples: selected
                .iter()
                .filter(|(s, _)| s.2 == OpponentKind::RandomCheckpoint)
                .count(),
            retained_samples: selected.iter().filter(|(_, old)| *old).count(),
            replay_size: buffer.len(),
            mse: total_loss.try_into_scalar::<f32>()? as f64 / selected.len() as f64,
            seconds: started.elapsed().as_secs_f64(),
        });
        if (iteration + 1).is_multiple_of(10) || iteration + 1 == completed + config.steps {
            net.clone()
                .valid()
                .save_file(destination.join(format!("iteration-{:06}.bpk", iteration + 1)))?;
        }
        std::fs::write(
            destination.join("report.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        let row = report.last().unwrap();
        eprintln!(
            "{:?} seed={} iteration={} mse={:.4} samples={} retained={} {:.2}s",
            config.intervention,
            config.seed,
            iteration + 1,
            row.mse,
            row.fresh_samples,
            row.retained_samples,
            row.seconds
        );
    }
    net.valid().save_file(destination.join("model.bpk"))?;
    std::fs::write(
        destination.join("complete.json"),
        serde_json::to_vec(&true)?,
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ablation_restores_adam_and_archive_and_repeats_control() {
        let device = Device::flex();
        device.seed(781);
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("source");
        let config = TrainingConfig {
            iterations: 2,
            games_per_iteration: 5,
            evaluation_games: 2,
            opponents: OpponentMix {
                self_play: 80.,
                random_checkpoint: 20.,
                ..Default::default()
            },
            historical_sampling: HistoricalSampling {
                adaptive: true,
                ..Default::default()
            },
            ..Default::default()
        };
        train(
            ValueNet::new(&device),
            device.clone(),
            &config,
            None,
            |state| {
                if state.report().iterations.len() == 2 {
                    state.save(&source)?;
                }
                Ok(())
            },
        )
        .unwrap();
        let config = AblationConfig {
            source,
            intervention: Intervention::Control,
            seed: 813,
            steps: 2,
            games: 5,
            updates: 1,
            batch_size: 16,
            replay_capacity: 32,
        };
        let a = run_ablation(&config, device.clone(), &root.path().join("a")).unwrap();
        let b = run_ablation(&config, device.clone(), &root.path().join("b")).unwrap();
        for (a, b) in a.iter().zip(&b) {
            assert_eq!(a.game_seed_schedule, b.game_seed_schedule);
            assert!((a.mse - b.mse).abs() < 1e-6);
            assert_eq!(a.optimization_samples, 16);
        }
        let config = AblationConfig {
            intervention: Intervention::Replay,
            ..config
        };
        let replay = run_ablation(&config, device, &root.path().join("replay")).unwrap();
        assert_eq!(replay[0].retained_samples, 0);
        assert_eq!(replay[1].retained_samples, 4);
        assert_eq!(replay[1].replay_size, 32);
        assert_eq!(replay[0].game_seed_schedule, a[0].game_seed_schedule);
    }

    #[test]
    fn replay_uses_old_samples_with_fixed_budget_and_bounded_fifo() {
        let fresh = vec![
            (
                EncodedAfterstate {
                    board: [0.; 400],
                    context: [0.; 50]
                },
                1.,
                OpponentKind::SelfPlay
            );
            16
        ];
        let old = vec![(fresh[0].0.clone(), -1., OpponentKind::RandomCheckpoint); 12];
        let mut buffer = VecDeque::new();
        retain(&mut buffer, &old, 8);
        assert_eq!(buffer.len(), 8);
        let selected = choose_samples(
            &fresh,
            &buffer,
            &OpponentMix::default(),
            Intervention::Replay,
            128,
            50,
            42,
        );
        assert_eq!(selected.len(), 128);
        assert_eq!(selected.iter().filter(|(_, old)| *old).count(), 32);
        assert!(
            selected
                .iter()
                .all(|(s, old)| s.1 == if *old { -1. } else { 1. })
        );
        retain(&mut buffer, &fresh, 8);
        assert!(buffer.iter().all(|s| s.1 == 1.));
        let empty = choose_samples(
            &fresh,
            &VecDeque::new(),
            &OpponentMix::default(),
            Intervention::Replay,
            128,
            50,
            42,
        );
        assert!(empty.iter().all(|(_, old)| !old));
    }
}
