mod dqn;
mod experiment;
mod inspect;
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

use experiment::{ExperimentConfig, write_games};
use summary::Summary;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Parser)]
#[command(
    version,
    about = "Run reproducible Sequence policy matchups",
    long_about = concat!(
        "Run reproducible Sequence policy matchups. Copy experiments/random-vs-heuristic.json ",
        "to define your own experiment. Choose random, heuristic, dqn, or ismcts for each policy. ",
        "For dqn and ismcts, set checkpoint to a .bpk path relative to the config file. ",
        "For ismcts, also set search.simulations and search.depth (root-player decisions).\n\n",
        "Each deal is played twice with the policies in swapped seats. Seat one opens both games. The figures show ",
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
    /// Train DQN or Double DQN and evaluate checkpoints against Random and Heuristic
    TrainDqn {
        /// JSON experiment config with a training device and optional inference_device for play and evaluation
        config: PathBuf,
        /// New output directory (defaults to results/<config file stem>)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Override the training device specified in the config
        #[arg(long, value_enum)]
        device: Option<dqn::Backend>,
        /// Override the model initialization, exploration, and replay RNG seed
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Resume DQN training in the same run directory from its latest saved checkpoint
    ResumeDqn {
        /// Directory produced by the train-dqn command
        run: PathBuf,
        /// Total target episodes, including those already completed
        #[arg(long)]
        episodes: Option<NonZeroUsize>,
    },
    /// Evaluate a saved DQN checkpoint or training run and export an SVG report
    EvaluateDqn {
        /// Saved .bpk checkpoint or training run directory
        input: PathBuf,
        /// JSON training experiment config
        config: PathBuf,
        /// Override the evaluation device specified in the config
        #[arg(long, value_enum)]
        device: Option<dqn::Backend>,
        #[command(flatten)]
        options: dqn::EvaluationOptions,
    },
    /// Compare saved DQN runs on shared deals and export an SVG evaluation report
    EvaluateDqnSuite(dqn::suite::Options),
    /// Open a local browser to filter and replay games from a run
    Inspect {
        /// Directory produced by the run command
        run: PathBuf,
        /// Optionally open a game directly by ID from games.csv
        game_id: Option<NonZeroUsize>,
        /// Start just before this action (counting from 1)
        #[arg(long, requires = "game_id")]
        action: Option<NonZeroUsize>,
        /// Print the local address without opening a browser
        #[arg(long)]
        no_open: bool,
    },
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Run { config, output } => run(&config, output),
        Command::TrainDqn {
            config,
            output,
            device,
            seed,
        } => dqn::train(&config, output, device, seed),
        Command::ResumeDqn { run, episodes } => dqn::resume(&run, episodes),
        Command::EvaluateDqn {
            input,
            config,
            device,
            options,
        } => dqn::evaluate_report(&input, &config, device, options),
        Command::EvaluateDqnSuite(options) => dqn::suite::run(options),
        Command::Inspect {
            run,
            game_id,
            action,
            no_open,
        } => inspect::serve(
            &run,
            game_id.map(NonZeroUsize::get),
            action.map(NonZeroUsize::get),
            no_open,
        ),
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
    let mut config: ExperimentConfig = serde_json::from_slice(&config_bytes)
        .map_err(|error| format!("{}: {error}", config_path.display()))?;

    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let staging = tempfile::tempdir_in(parent)?;
    let directory = staging.path();
    let copied =
        config.save_checkpoint(config_path.parent().unwrap_or(Path::new(".")), directory)?;
    let checkpoint = config.load_checkpoint(directory)?;
    let (report, move_timing) = config.evaluate(checkpoint.as_ref())?;
    let summary = Summary::from_report(&config, &report);

    let config_bytes = if copied {
        (serde_json::to_string_pretty(&config)? + "\n").into_bytes()
    } else {
        config_bytes
    };
    fs::write(directory.join("config.json"), config_bytes)?;
    write_games(&directory.join("games.csv"), &config, &report)?;
    fs::write(
        directory.join("summary.json"),
        serde_json::to_string_pretty(&summary)? + "\n",
    )?;
    if let Some(move_timing) = move_timing {
        fs::write(
            directory.join("move-times.json"),
            serde_json::to_string_pretty(&move_timing)? + "\n",
        )?;
    }
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
