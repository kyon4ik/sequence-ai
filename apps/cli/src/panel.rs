//! Fixed-panel evaluation with checkpoint-bound evidence for promotion.
use std::path::Path;

use anyhow::Result;
use sequence_agent::InferenceDevice;

#[cfg(feature = "burn")]
mod enabled {
    use super::*;
    use crate::{
        artifacts::{digest, read_json, write_json},
        parse_device, validate_id,
    };
    use anyhow::{anyhow, ensure};
    use burn::prelude::*;
    use sequence_agent::{AgentConfig, mc::ValueNet};
    use sequence_training::mc::{EvaluationConfig, evaluate_seeded_panel};
    use serde::{Deserialize, Serialize};
    use std::{
        collections::{BTreeMap, BTreeSet},
        path::PathBuf,
    };

    #[derive(Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct Checkpoint {
        id: String,
        path: PathBuf,
    }

    #[derive(Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct Manifest {
        #[serde(default = "cpu")]
        device: String,
        games: usize,
        inference_batch_games: usize,
        seeds: Vec<u64>,
        checkpoints: Vec<Checkpoint>,
        #[serde(default = "default_opponents")]
        opponents: Vec<(String, AgentConfig)>,
        #[serde(default)]
        diagnostic_games: usize,
    }

    fn cpu() -> String {
        "cpu".into()
    }
    fn default_opponents() -> Vec<(String, AgentConfig)> {
        vec![
            ("random".into(), AgentConfig::Random),
            ("greedy".into(), AgentConfig::greedy()),
        ]
    }

    pub fn run(config: &Path, output: &Path, device: Option<InferenceDevice>) -> Result<()> {
        let mut manifest: Manifest = read_json(config)?;
        let device = device
            .map(Ok)
            .unwrap_or_else(|| parse_device(&manifest.device))
            .map_err(anyhow::Error::msg)?;
        manifest.device = match device {
            InferenceDevice::Cpu => "cpu".into(),
            InferenceDevice::Cuda(i) => format!("cuda:{i}"),
        };
        ensure!(
            manifest.games > 0 && manifest.games.is_multiple_of(2),
            "panel games must be positive and even"
        );
        ensure!(
            manifest.inference_batch_games > 0,
            "inference_batch_games must be positive"
        );
        ensure!(
            !manifest.checkpoints.is_empty()
                && !manifest.seeds.is_empty()
                && !manifest.opponents.is_empty(),
            "panel requires checkpoints, seeds, and opponents"
        );
        ensure!(
            manifest.seeds.iter().collect::<BTreeSet<_>>().len() == manifest.seeds.len(),
            "duplicate evaluation seed"
        );
        let mut ids = BTreeSet::new();
        let mut hashes = BTreeMap::new();
        for checkpoint in &mut manifest.checkpoints {
            validate_id(&checkpoint.id)?;
            ensure!(ids.insert(checkpoint.id.clone()), "duplicate checkpoint ID");
            checkpoint.path = std::fs::canonicalize(&checkpoint.path)?;
            hashes.insert(checkpoint.id.clone(), digest(&checkpoint.path)?);
            for seed in &manifest.seeds {
                ensure!(
                    !output
                        .join(format!("{}-{seed}.json", checkpoint.id))
                        .exists(),
                    "panel output already exists for {} seed {seed}",
                    checkpoint.id
                );
            }
        }
        let mut names = BTreeSet::new();
        let mut opponent_hashes = BTreeMap::new();
        for (name, policy) in &mut manifest.opponents {
            ensure!(
                !name.is_empty() && names.insert(name.clone()),
                "empty or duplicate opponent name"
            );
            if let AgentConfig::MonteCarlo { checkpoint } = policy {
                *checkpoint = std::fs::canonicalize(&*checkpoint)?;
                opponent_hashes.insert(name.clone(), digest(checkpoint)?);
            }
        }
        std::fs::create_dir_all(output)?;
        let device = device.burn_device();
        for checkpoint in &manifest.checkpoints {
            let net = ValueNet::new(&device)
                .try_load_file(&checkpoint.path)
                .map_err(|e| anyhow!("{e}"))?;
            for &seed in &manifest.seeds {
                let config = EvaluationConfig {
                    games: manifest.games,
                    inference_batch_games: manifest.inference_batch_games,
                    seed,
                };
                eprintln!(
                    "Evaluating {} on {}: {} games per opponent, seed {seed}",
                    checkpoint.id, manifest.device, manifest.games
                );
                let baselines = evaluate_seeded_panel(
                    &net,
                    &device,
                    &config,
                    &manifest.opponents,
                    manifest.diagnostic_games,
                )
                .map_err(|e| anyhow!("{e}"))?;
                ensure!(
                    digest(&checkpoint.path)? == hashes[&checkpoint.id],
                    "candidate checkpoint changed during evaluation"
                );
                for (name, policy) in &manifest.opponents {
                    if let AgentConfig::MonteCarlo { checkpoint } = policy {
                        ensure!(
                            digest(checkpoint)? == opponent_hashes[name],
                            "opponent checkpoint changed during evaluation"
                        );
                    }
                }
                let report = serde_json::json!({"id": checkpoint.id, "checkpoint": checkpoint.path, "checkpoint_sha256": hashes[&checkpoint.id], "opponent_checkpoint_sha256": opponent_hashes, "device": manifest.device, "config": config, "baselines": baselines});
                let path = output.join(format!("{}-{seed}.json", checkpoint.id));
                write_json(&path, &report)?;
                println!("Saved {}", path.display());
            }
        }
        Ok(())
    }
}

pub fn run(config: &Path, output: &Path, device: Option<InferenceDevice>) -> Result<()> {
    #[cfg(feature = "burn")]
    {
        enabled::run(config, output, device)
    }
    #[cfg(not(feature = "burn"))]
    {
        let _ = (config, output, device);
        anyhow::bail!(
            "panel evaluation requires Burn support; build sequence-cli with default features"
        );
    }
}
