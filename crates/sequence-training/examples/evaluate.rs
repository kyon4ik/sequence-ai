//! Batched fixed-panel evaluations: evaluate <manifest.json> <output-directory>.
//! Manifest: {"device":"cuda:0", "games":1000, "inference_batch_games":128,
//! "seeds":[2026111401], "checkpoints":[{"id":"best", "path":".../best.bpk"}]}.
//! Optional opponents: [["name", AgentConfig], ...]; diagnostic_games defaults to 0.

#[cfg(feature = "burn")]
fn main() -> Result<(), sequence_agent::AgentError> {
    use burn::prelude::*;
    use sequence_agent::{AgentConfig, mc::ValueNet};
    use sequence_training::mc::{EvaluationConfig, evaluate_seeded_panel};
    use serde::{Deserialize, Serialize};
    use std::path::PathBuf;

    #[derive(Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct Checkpoint {
        id: String,
        path: PathBuf,
    }
    #[derive(Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct Manifest {
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
    fn default_opponents() -> Vec<(String, AgentConfig)> {
        vec![
            ("random".into(), AgentConfig::Random),
            ("greedy".into(), AgentConfig::greedy()),
        ]
    }
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: evaluate <manifest.json> <output-directory>".into());
    }
    let manifest: Manifest = serde_json::from_reader(std::fs::File::open(&args[0])?)?;
    let device = if manifest.device == "cpu" {
        Device::flex()
    } else if let Some(index) = manifest.device.strip_prefix("cuda:") {
        Device::cuda(index.parse::<usize>()?)
    } else {
        return Err("expected cpu or cuda:N".into());
    };
    let output = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&output)?;
    for checkpoint in &manifest.checkpoints {
        if checkpoint.id.is_empty()
            || !checkpoint
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b))
        {
            return Err("invalid checkpoint ID".into());
        }
        let net = ValueNet::new(&device).try_load_file(&checkpoint.path)?;
        for &seed in &manifest.seeds {
            let path = output.join(format!("{}-{seed}.json", checkpoint.id));
            if path.exists() {
                return Err(format!("evaluation already exists: {}", path.display()).into());
            }
            let config = EvaluationConfig {
                games: manifest.games,
                inference_batch_games: manifest.inference_batch_games,
                seed,
            };
            eprintln!(
                "Evaluating {} on {}: {} games per baseline, seed {}",
                checkpoint.id, manifest.device, manifest.games, seed
            );
            let baselines = evaluate_seeded_panel(
                &net,
                &device,
                &config,
                &manifest.opponents,
                manifest.diagnostic_games,
            )?;
            let report = serde_json::json!({"id": checkpoint.id, "checkpoint": checkpoint.path, "device": manifest.device, "config": config, "baselines": baselines});
            std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
            for baseline in &baselines {
                eprintln!(
                    "  {}: {:.3} ({:?})",
                    baseline.summary.opponent, baseline.summary.score, baseline.summary.outcomes
                );
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "burn"))]
fn main() {
    eprintln!("evaluate requires --features burn");
    std::process::exit(1);
}
