use std::path::Path;

use anyhow::Result;
use sequence_agent::InferenceDevice;

use crate::artifacts::Store;

pub struct Overrides {
    pub id: Option<String>,
    pub device: Option<InferenceDevice>,
    pub seed: Option<u64>,
    pub checkpoint_interval: Option<usize>,
    pub keep_checkpoints: Option<usize>,
    pub resume: Option<std::path::PathBuf>,
}

/// Run only after a complete checkpoint has been published successfully.
#[cfg(any(feature = "burn", test))]
fn prune_checkpoints(directory: &Path, keep: usize) -> Result<()> {
    if keep == 0 {
        return Ok(());
    }
    let mut checkpoints = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(iteration) = name
            .to_str()
            .and_then(|name| name.strip_prefix("iteration-"))
        else {
            continue;
        };
        if entry.file_type()?.is_dir()
            && iteration.len() >= 6
            && iteration.bytes().all(|b| b.is_ascii_digit())
            && entry.path().join("state.json").is_file()
        {
            checkpoints.push((iteration.parse::<usize>()?, entry.path()));
        }
    }
    checkpoints.sort_by_key(|(iteration, _)| *iteration);
    let remove = checkpoints.len().saturating_sub(keep);
    for (_, path) in checkpoints.into_iter().take(remove) {
        std::fs::remove_dir_all(path)?;
    }
    Ok(())
}

#[cfg(feature = "burn")]
mod enabled {
    use super::*;
    use crate::{
        artifacts::{read_toml, replace_json, write_json},
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
        #[serde(default = "checkpoint_interval")]
        checkpoint_interval: usize,
        #[serde(default = "keep_checkpoints")]
        keep_checkpoints: usize,
    }

    fn cpu() -> String {
        "cpu".into()
    }

    fn checkpoint_interval() -> usize {
        10
    }

    fn keep_checkpoints() -> usize {
        2
    }

    pub fn run(store: &Store, path: &Path, overrides: Overrides) -> Result<()> {
        let mut experiment: Experiment = read_toml(path)?;
        if let Some(id) = overrides.id {
            experiment.id = id;
        }
        if let Some(seed) = overrides.seed {
            experiment.training.seed = seed;
        }
        if let Some(interval) = overrides.checkpoint_interval {
            experiment.checkpoint_interval = interval;
        }
        if let Some(keep) = overrides.keep_checkpoints {
            experiment.keep_checkpoints = keep;
        }
        anyhow::ensure!(
            experiment.checkpoint_interval > 0,
            "checkpoint_interval must be positive"
        );
        validate_id(&experiment.id)?;
        experiment.training.validate().map_err(|e| anyhow!("{e}"))?;
        let device = overrides
            .device
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
        let output = train(ValueNet::new(&device), device, &experiment.training, overrides.resume.as_deref(), |checkpoint| {
            let metrics = checkpoint.report().iterations.last().unwrap();
            if metrics.iteration.is_multiple_of(experiment.checkpoint_interval) || metrics.iteration == experiment.training.iterations {
                checkpoint.save(&checkpoints.join(format!("iteration-{:06}", metrics.iteration)))?;
                prune_checkpoints(&checkpoints, experiment.keep_checkpoints).map_err(|e| e.to_string())?;
            }
            replace_json(&dir.join("progress.json"), metrics).map_err(|e| e.to_string())?;
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
            .insert("device".into(), experiment.device.clone());
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
            "Evaluate: sequence --data-dir {:?} eval {} greedy --games 100 --device {}",
            store.root, experiment.id, experiment.device
        );
        Ok(())
    }
}

pub fn run(store: &Store, path: &Path, overrides: Overrides) -> Result<()> {
    #[cfg(feature = "burn")]
    {
        enabled::run(store, path, overrides)
    }
    #[cfg(not(feature = "burn"))]
    {
        let Overrides {
            id,
            device,
            seed,
            checkpoint_interval,
            keep_checkpoints,
            resume,
        } = overrides;
        let _ = (
            store,
            path,
            id,
            device,
            seed,
            checkpoint_interval,
            keep_checkpoints,
            resume,
        );
        anyhow::bail!(
            "training requires Burn support; build sequence-cli with its default features or --features burn"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retention_keeps_newest_complete_checkpoints_and_ignores_other_files() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "iteration-000001",
            "iteration-000002",
            "iteration-000003",
            "iteration-000004",
            "unrelated",
        ] {
            let path = dir.path().join(name);
            std::fs::create_dir(&path).unwrap();
            if name != "iteration-000004" {
                std::fs::write(path.join("state.json"), "{}").unwrap();
            }
        }
        prune_checkpoints(dir.path(), 0).unwrap();
        assert!(dir.path().join("iteration-000001").exists());
        prune_checkpoints(dir.path(), 2).unwrap();
        assert!(!dir.path().join("iteration-000001").exists());
        for name in [
            "iteration-000002",
            "iteration-000003",
            "iteration-000004",
            "unrelated",
        ] {
            assert!(dir.path().join(name).exists());
        }
    }
}
