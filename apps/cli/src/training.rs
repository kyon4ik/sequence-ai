use std::path::Path;

use anyhow::Result;
use sequence_agent::InferenceDevice;

use crate::artifacts::Store;

#[cfg(feature = "burn")]
mod enabled {
    use super::*;
    use crate::{
        artifacts::{read_toml, write_json},
        parse_device, validate_id,
    };
    use anyhow::{Context, anyhow};
    use burn::prelude::*;
    use sequence_agent::{AgentConfig, AgentDescriptor, AgentSpec, mc::ValueNet};
    use sequence_training::mc::{TrainingConfig, train};
    use serde::{Deserialize, Serialize};

    #[derive(Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum Algorithm {
        Mc,
    }

    #[derive(Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct Experiment {
        algorithm: Algorithm,
        id: String,
        #[serde(default)]
        name: Option<String>,
        #[serde(default = "cpu")]
        device: String,
        #[serde(default)]
        training: TrainingConfig,
    }

    fn cpu() -> String {
        "cpu".into()
    }

    pub fn run(
        store: &Store,
        path: &Path,
        id: Option<String>,
        device: Option<InferenceDevice>,
        resume: Option<&Path>,
    ) -> Result<()> {
        let mut experiment: Experiment = read_toml(path)?;
        if let Some(id) = id {
            experiment.id = id;
        }
        validate_id(&experiment.id)?;
        experiment.training.validate().map_err(|e| anyhow!("{e}"))?;
        let device = device
            .map(Ok)
            .unwrap_or_else(|| parse_device(&experiment.device))
            .map_err(anyhow::Error::msg)?;
        experiment.device = match device {
            InferenceDevice::Cpu => "cpu".into(),
            InferenceDevice::Cuda(index) => format!("cuda:{index}"),
        };
        let dir = store.reserve_model(&experiment.id)?;
        write_json(&dir.join("experiment.json"), &experiment)?;
        eprintln!(
            "Training {} on {}: {} iterations × {} games (seed {})",
            experiment.id,
            experiment.device,
            experiment.training.iterations,
            experiment.training.games_per_iteration,
            experiment.training.seed
        );
        let device = device.burn_device();
        device.seed(experiment.training.seed);
        let checkpoints = dir.join("checkpoints");
        std::fs::create_dir(&checkpoints)?;
        let output = train(ValueNet::new(&device), device, &experiment.training, resume, |checkpoint| {
            let metrics = checkpoint.report().iterations.last().unwrap();
            checkpoint.save(&checkpoints.join(format!("iteration-{:06}", metrics.iteration)))?;
            eprintln!("iteration {}/{}  mse={:.6}  epsilon={:.3}  samples={}  evaluation={:?}  rollout={:.2}s  optimize={:.2}s  eval={:.2}s  inference_batches={}  elapsed={:.2}s", metrics.iteration, experiment.training.iterations, metrics.mse, metrics.epsilon, metrics.samples, metrics.evaluation_score, metrics.rollout_time.as_secs_f64(), metrics.optimization_time.as_secs_f64(), metrics.evaluation_time.as_secs_f64(), metrics.rollout_inference.batches, metrics.elapsed.as_secs_f64());
            let mut counts = std::collections::BTreeMap::new();
            let mut fallbacks = 0;
            for opponent in &metrics.opponents {
                *counts.entry(opponent.source).or_insert(0usize) += opponent.games;
                fallbacks += opponent.fallback_games;
            }
            eprintln!("  opponents={counts:?}  self-play fallbacks={fallbacks}");
            Ok(())
        }).map_err(|e| anyhow!("training failed: {e}"))?;
        if let Some(best) = output.best_model {
            best.save_file(dir.join("best.bpk"))?;
        }
        output
            .model
            .save_file(dir.join("model.bpk"))
            .context("saving model checkpoint")?;
        write_json(&dir.join("report.json"), &output.report)?;
        let mut descriptor = AgentDescriptor::new(
            &experiment.id,
            experiment.name.as_deref().unwrap_or(&experiment.id),
        );
        descriptor.metadata.insert("algorithm".into(), "mc".into());
        descriptor
            .metadata
            .insert("seed".into(), experiment.training.seed.to_string());
        descriptor
            .metadata
            .insert("device".into(), experiment.device);
        // Publish the portable spec last: its checkpoint is relative to this file.
        write_json(
            &dir.join("agent.json"),
            &AgentSpec {
                descriptor,
                config: AgentConfig::MonteCarlo {
                    checkpoint: "model.bpk".into(),
                },
            },
        )?;
        println!("Saved agent {} to {}", experiment.id, dir.display());
        println!(
            "Evaluate: sequence --data-dir {:?} eval {} greedy --games 100",
            store.root, experiment.id
        );
        Ok(())
    }
}

pub fn run(
    store: &Store,
    path: &Path,
    id: Option<String>,
    device: Option<InferenceDevice>,
    resume: Option<&Path>,
) -> Result<()> {
    #[cfg(feature = "burn")]
    {
        enabled::run(store, path, id, device, resume)
    }
    #[cfg(not(feature = "burn"))]
    {
        let _ = (store, path, id, device, resume);
        anyhow::bail!(
            "training requires Burn support; build sequence-cli with its default features or --features burn"
        )
    }
}
