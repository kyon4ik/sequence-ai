#![deny(unsafe_code)]

use std::{net::IpAddr, path::PathBuf};

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use sequence_agent::InferenceDevice;

mod artifacts;
mod competition;
mod replay;
mod training;
mod web;

#[derive(Parser)]
#[command(
    name = "sequence",
    version,
    about = "Train, evaluate and explore Sequence agents"
)]
struct Cli {
    /// Root for model artifacts, replays and experiment reports.
    #[arg(long, global = true, default_value = "data")]
    data_dir: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train and save a neural agent from a TOML experiment (currently MC self-play).
    Train {
        /// Resume a full checkpoint into a new agent ID, retaining the original config.
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long)]
        config: PathBuf,
        /// Override the config's agent ID; existing IDs are never overwritten.
        #[arg(long)]
        id: Option<String>,
        /// Override the config's device: cpu or cuda:N.
        #[arg(long, value_parser = parse_device)]
        device: Option<InferenceDevice>,
    },
    /// Play two saved agent IDs/spec paths or built-ins: random, greedy, heuristic.
    Eval {
        agent_a: String,
        agent_b: String,
        #[arg(long, default_value = "100", value_parser = positive_usize)]
        games: usize,
        #[arg(long, default_value = "42")]
        seed: u64,
        #[arg(long, default_value = "1000", value_parser = positive_usize)]
        max_actions: usize,
        #[arg(long, default_value = "cpu", value_parser = parse_device)]
        device: InferenceDevice,
    },
    /// Run a round-robin league from TOML, with fresh Elo ratings for this run.
    Tournament {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, value_parser = parse_device)]
        device: Option<InferenceDevice>,
    },
    /// Validate a saved replay and display a board frame (0 = initial deal).
    Replay {
        /// Folder/game reference or an explicit JSON file path.
        game: String,
        /// Defaults to the final frame.
        #[arg(long)]
        frame: Option<usize>,
        /// Print the full, validated match record as JSON.
        #[arg(long, conflicts_with = "frame")]
        json: bool,
    },
    /// Start the local browser UI for saved agents, metrics and replay exploration.
    Serve {
        #[arg(long, default_value = "8080")]
        port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        host: IpAddr,
    },
}

fn positive_usize(value: &str) -> Result<usize, String> {
    let n = value.parse::<usize>().map_err(|e| e.to_string())?;
    if n == 0 {
        Err("must be positive".into())
    } else {
        Ok(n)
    }
}

fn parse_device(value: &str) -> Result<InferenceDevice, String> {
    if value == "cpu" {
        return Ok(InferenceDevice::Cpu);
    }
    if let Some(index) = value.strip_prefix("cuda:") {
        return index
            .parse()
            .map(InferenceDevice::Cuda)
            .map_err(|_| "expected cpu or cuda:N".into());
    }
    Err("expected cpu or cuda:N (for example cuda:0)".into())
}

fn run(cli: Cli) -> Result<()> {
    let store = artifacts::Store::new(cli.data_dir);
    match cli.command {
        Command::Train {
            config,
            id,
            device,
            resume,
        } => training::run(&store, &config, id, device, resume.as_deref()),
        Command::Eval {
            agent_a,
            agent_b,
            games,
            seed,
            max_actions,
            device,
        } => competition::evaluate(
            &store,
            [&agent_a, &agent_b],
            games,
            seed,
            max_actions,
            device,
        ),
        Command::Tournament { config, device } => competition::tournament(&store, &config, device),
        Command::Replay { game, frame, json } => replay::show(&store, &game, frame, json),
        Command::Serve { host, port } => web::serve(&store, host, port),
    }
}

fn main() -> std::process::ExitCode {
    match run(Cli::parse()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn validate_id(id: &str) -> Result<()> {
    if id.is_empty()
        || id.len() > 128
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b))
    {
        bail!("invalid ID {id:?}: use 1–128 ASCII letters, digits, hyphens or underscores");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn command_contract() {
        Cli::command().debug_assert();
        assert!(
            Cli::try_parse_from(["sequence", "eval", "random", "greedy", "--games", "0"]).is_err()
        );
        assert!(
            Cli::try_parse_from([
                "sequence", "train", "--config", "mc.toml", "--device", "cuda:x"
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "sequence",
                "--data-dir",
                "/tmp/test",
                "replay",
                "run/match",
                "--frame",
                "0"
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from(["sequence", "replay", "run/match", "--frame", "0", "--json"])
                .is_err()
        );
    }
}
