# Sequence AI

An RL course project on learning to play two-player Sequence under incomplete information. The Rust workspace contains the game, Random and Heuristic policies, a match evaluator and replay viewer, and a DQN trainer with self-play.

[Game rules](docs/RULES.md) · [Experiment catalog](docs/EXPERIMENTS.md)

## Install

Start with a clone or a copy of the repository. If you have access to the GitHub repository, run these commands in a terminal.

```text
git clone https://github.com/kyon4ik/sequence-ai.git
cd sequence-ai
```

Install current stable Rust with rustup. Cargo comes with Rust. Keep `Cargo.lock` when sharing the source and use `--locked` when building. [Rust installation](https://rust-lang.org/install/)

| Platform | Build prerequisites |
| --- | --- |
| Linux | Install your distribution's C compiler and linker, such as `build-essential` on Ubuntu or Debian. [Rust installation guide](https://doc.rust-lang.org/stable/book/ch01-01-installation.html) |
| macOS | Install the Xcode Command Line Tools with `xcode-select --install`. Use the CPU backend on Apple Silicon and Intel Macs. [Rust installation guide](https://doc.rust-lang.org/stable/book/ch01-01-installation.html) |
| Windows | Use the native MSVC Rust toolchain. Install the Visual Studio C++ Build Tools and a Windows SDK if rustup prompts for them. Open a new PowerShell window after installation. WSL users can follow the Linux CPU instructions. [MSVC prerequisites](https://rust-lang.github.io/rustup/installation/windows-msvc.html) |

Run the commands below from the repository root. The single-line Cargo commands work in Bash, Zsh, and PowerShell. Cargo selects the executable for your OS.

```text
rustc --version
cargo --version
cargo build --release --locked -p sequence-research
```

## Run a matchup and inspect games

```text
cargo run --release --locked -p sequence-research -- run experiments/random-vs-heuristic.json --output results/experiments/quick-match
cargo run --release --locked -p sequence-research -- inspect results/experiments/quick-match --no-open
```

Open the printed local address in your browser. The viewer filters games and replays their actions. Press Ctrl+C in the terminal to stop it. The run also saves `config.json`, `games.csv`, `summary.json`, and SVG plots. Use a new output path for each run.

## Train and evaluate a DQN

Start with the one-episode CPU smoke test. It uses an eight-action limit, so cutoffs are expected. Training writes a checkpoint at initialization and at the end.

```text
cargo run --release --locked -p sequence-research -- train-dqn crates/sequence-research/tests/fixtures/dqn.json --output results/dqn-smoke/seed-42
cargo run --release --locked -p sequence-research -- evaluate-dqn results/dqn-smoke/seed-42/checkpoints/episode-000001.bpk crates/sequence-research/tests/fixtures/dqn.json --device cpu --tactics
```

For a full CPU run, use an experiment preset and override its training device. Copy the JSON and reduce `episodes` and `evaluation.deal_count` if you want an intermediate check before committing to a long run. Keep `inference_device` set to `"cpu"` or omit it on CPU-only machines.

```text
cargo run --release --locked -p sequence-research -- train-dqn experiments/double-dqn-immediate-win-full.json --device cpu --seed 42 --output results/experiments/supervised-cpu/seed-42
```

### Optional NVIDIA CUDA training

The recent research runs used NVIDIA CUDA on Linux with CPU inference. Install a compatible NVIDIA driver and CUDA Toolkit, make `nvcc` available on your path, then check that the training environment sees the GPU. On Windows you can try the native CUDA setup with MSVC. For WSL 2, install the NVIDIA driver on the Windows host and the Linux CUDA Toolkit inside WSL. Use the CPU path if CUDA is unavailable. [Linux CUDA setup](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) · [Windows CUDA setup](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/) · [WSL CUDA setup](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

```text
nvidia-smi
nvcc --version
cargo run --release --locked -p sequence-research --features cuda -- train-dqn crates/sequence-research/tests/fixtures/dqn.json --device cuda --output results/dqn-cuda-smoke/seed-42
cargo run --release --locked -p sequence-research --features cuda -- train-dqn experiments/double-dqn-immediate-win-full.json --seed 42 --output results/experiments/supervised-cuda/seed-42
```

Use a separate output directory per seed.

### Inspect a full run

- `training.csv` records episode outcomes, the opponent, cumulative learner decisions and updates, and the exploration schedule.
- `evaluation.csv` and `win-rates.svg` show checkpoint performance against Random and Heuristic on the recurring monitoring panel.
- `checkpoints/` contains `.bpk` weights. Evaluate a checkpoint on a shared holdout with `evaluate-dqn`. The command prints summary JSON and `--games` saves a per-game CSV at a new path. `--take-immediate-wins` adds the rule override used in the tactical ablation.
- `probe-dqn` scans the saved checkpoints in its run directories and prints tactical action ranks on a common set of positions. The replay viewer takes directories produced by `run`.

```text
cargo run --release --locked -p sequence-research -- evaluate-dqn results/experiments/supervised-cuda/seed-42/checkpoints/episode-051200.bpk experiments/next-holdout.json --device cpu --tactics --games results/experiments/supervised-cuda/seed-42/holdout-games.csv
cargo run --release --locked -p sequence-research -- probe-dqn results/experiments/supervised-cuda/seed-42 --device cpu --first-shuffle-seed 13000000 --deals 256 --policy-seed 2029
```

Use the path of your own run if you trained on CPU or used a different name. Run any subcommand with `--help` to see its options.

## Design and share a new experiment

1. Copy an active JSON preset from `experiments/`. Pick `target` (`dqn` or `double_dqn`) and either all three `epsilon_*` fields or a `temperature` schedule. To change the opponent mix, set `heuristic_opponent_every` to one Heuristic game per that many episodes. `immediate_win_loss_weight` enables the tactical loss when positive.
2. Use distinct training and evaluation shuffle-seed ranges. For a serious comparison, budget at least 500,000 learner decisions per seed and aim for 1 million or more. Select checkpoints by recorded learner decisions, since two runs with the same episode count can have different numbers of decisions.
3. Use the same held-out deals and policy seeds for all arms. Each evaluation deal is played twice with the policies swapping seats. Report per-seed results and keep both seats together if bootstrapping by deal. Reserve a fresh panel if the existing holdout guided tuning.
4. Record the exact command, Git revision, any local patch, Rust version, device and hardware, and effective config saved in the run directory. Include the evaluation config and CSVs, the summary, and the checkpoints colleagues need to inspect or rerun evaluation.

For code changes, run `cargo fmt --all --check`, `cargo clippy --locked --workspace --all-targets -- -D warnings`, and `cargo test --locked --workspace`. The replay viewer test needs permission to bind a loopback port.
