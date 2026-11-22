mod experiment;
mod plots;
mod summary;

use std::{
    error::Error,
    fs, io,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::{Parser, Subcommand};

use experiment::{ExperimentConfig, inspect_game, write_games};
use summary::Summary;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Parser)]
#[command(
    version,
    about = "Run reproducible Sequence policy matchups",
    long_about = concat!(
        "Run reproducible Sequence policy matchups. Copy experiments/random-vs-heuristic.json ",
        "to define your own experiment. Choose random or heuristic for each policy ",
        "and one or two for starting_seat.\n\n",
        "Each deal is played twice with the policies in swapped seats. The figures show ",
        "results for the first configured policy. Share the config and Git revision ",
        "to reproduce a run with cargo run --locked."
    )
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Evaluate a config and export per-game data, statistics, and report figures
    Run {
        /// JSON experiment config
        config: PathBuf,
        /// New output directory (defaults to results/<config file stem>)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Reconstruct a game from a run and print the acting player's view
    Inspect {
        /// Directory produced by the run command
        run: PathBuf,
        /// Game ID from games.csv
        game_id: NonZeroUsize,
        /// Show only this action number (counting from 1)
        #[arg(long)]
        action: Option<NonZeroUsize>,
    },
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Run { config, output } => run(&config, output),
        Command::Inspect {
            run,
            game_id,
            action,
        } => inspect_game(&run, game_id.get(), action.map(NonZeroUsize::get)),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(config_path: &Path, output: Option<PathBuf>) -> Result<()> {
    let output = match output {
        Some(output) => output,
        None => Path::new("results").join(
            config_path
                .file_stem()
                .ok_or("expected a config filename")?,
        ),
    };
    if output.try_exists()? {
        return Err(format!(
            "{} already exists. Choose a new --output directory.",
            output.display()
        )
        .into());
    }

    let config_bytes = fs::read(config_path).map_err(|error| {
        io::Error::new(error.kind(), format!("{}: {error}", config_path.display()))
    })?;
    let config: ExperimentConfig = serde_json::from_slice(&config_bytes)
        .map_err(|error| format!("{}: {error}", config_path.display()))?;
    let report = config.evaluate()?;
    let summary = Summary::from_report(&config, &report);

    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let staging = tempfile::tempdir_in(parent)?;
    let directory = staging.path();
    fs::write(directory.join("config.json"), config_bytes)?;
    write_games(&directory.join("games.csv"), &config, &report)?;
    fs::write(
        directory.join("summary.json"),
        serde_json::to_string_pretty(&summary)? + "\n",
    )?;
    plots::write_figures(directory, &report, &summary)?;
    fs::rename(directory, &output).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("could not publish results to {}: {error}", output.display()),
        )
    })?;
    summary.print(&output);
    Ok(())
}
