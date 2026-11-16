//! JSON manifest -> immutable model directory. Saves complete checkpoints every
//! N iterations, allowing long studies to avoid saving replay buffers every step.
#[cfg(feature = "burn")]
fn main() -> Result<(), sequence_agent::AgentError> {
    use burn::prelude::*;
    use sequence_agent::{AgentConfig, AgentDescriptor, AgentSpec, mc::ValueNet};
    use sequence_training::mc::{TrainingConfig, train};
    use serde::{Deserialize, Serialize};
    use std::path::PathBuf;
    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Manifest {
        id: String,
        training: TrainingConfig,
        checkpoint_interval: usize,
        #[serde(default)]
        resume: Option<PathBuf>,
    }
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: train_study <manifest.json> <new-output-directory>".into());
    }
    let manifest: Manifest = serde_json::from_reader(std::fs::File::open(&args[0])?)?;
    manifest.training.validate()?;
    if manifest.checkpoint_interval == 0 {
        return Err("checkpoint interval must be positive".into());
    }
    let output = PathBuf::from(&args[1]);
    std::fs::create_dir(&output)?;
    let checkpoints = output.join("checkpoints");
    std::fs::create_dir(&checkpoints)?;
    std::fs::write(
        output.join("experiment.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    let device = Device::cuda(0);
    device.seed(manifest.training.seed);
    let trained = train(
        ValueNet::new(&device),
        device,
        &manifest.training,
        manifest.resume.as_deref(),
        |state| {
            let metrics = state.report().iterations.last().unwrap();
            if metrics
                .iteration
                .is_multiple_of(manifest.checkpoint_interval)
                || metrics.iteration == manifest.training.iterations
            {
                state.save(&checkpoints.join(format!("iteration-{:06}", metrics.iteration)))?;
            }
            std::fs::write(output.join("progress.tmp"), serde_json::to_vec(metrics)?)?;
            std::fs::rename(output.join("progress.tmp"), output.join("progress.json"))?;
            eprintln!(
                "iteration={} mse={:.4} samples={} replay={} epsilon={:.3} evaluation={:?} seconds={:.2}",
                metrics.iteration,
                metrics.mse,
                metrics.samples,
                metrics.replay_samples,
                metrics.epsilon,
                metrics.evaluation_score,
                metrics.elapsed.as_secs_f64()
            );
            Ok(())
        },
    )?;
    trained.model.save_file(output.join("model.bpk"))?;
    if let Some(best) = trained.best_model {
        best.save_file(output.join("best.bpk"))?;
    }
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&trained.report)?,
    )?;
    let spec = AgentSpec {
        descriptor: AgentDescriptor::new(&manifest.id, &manifest.id),
        config: AgentConfig::MonteCarlo {
            checkpoint: "model.bpk".into(),
        },
    };
    std::fs::write(output.join("agent.json"), serde_json::to_vec_pretty(&spec)?)?;
    Ok(())
}
#[cfg(not(feature = "burn"))]
fn main() {
    eprintln!("train_study requires --features burn");
    std::process::exit(1);
}
