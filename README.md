# Sequence AI

Train, evaluate, and inspect agents for two-player Sequence.

The CLI supports Monte Carlo training on **Simple rules**, random and greedy
baselines, head-to-head evaluation, round-robin tournaments, and a local replay
explorer. The Rust core also implements Classic rules. PPO and DQN are not
implemented.

## Install

Use a Rust toolchain with edition 2024 support. See
[development setup](docs/development.md) for optional linker settings and checks.

From the repository root:

```sh
cargo install --path apps/cli --locked
sequence --help
```

## Train your first agent

Run this small CPU experiment to check the complete workflow:

```sh
sequence train --config configs/experiments/mc.toml
sequence eval mc-v1 greedy --games 100
sequence serve --port 8080
```

Open `http://127.0.0.1:8080` to inspect the model, evaluation results, and replays.
The local explorer reads saved results; training runs through the CLI.

Artifacts are saved under `data/` (ignored by Git). Each model ID is immutable;
use `--id mc-v2` on a later training run. Use `--data-dir PATH` to choose another
artifact directory.

The introductory config is a workflow check. For the recommended full training
run, use the stable config (200 iterations, 128 games per iteration, learning
rate 0.0001, uniform historical opponents, retained-sample replay off):

```sh
sequence train --config configs/experiments/mc-stable.toml
sequence eval mc-stable-v1 greedy --games 1000 --seed 123 --device cuda:0
```

The stable config selects CUDA device 0. Pass `--device cpu` to train on CPU.
The introductory config selects CPU; add `--device cuda:0` to use a GPU.
Evaluation defaults to CPU unless you specify a device. CUDA failures do not
silently fall back to CPU.

## Guides

| Guide | Contents |
| --- | --- |
| [Usage](docs/usage.md) | Train, resume, evaluate, run tournaments, inspect replays, and plot metrics |
| [Configuration](docs/configuration.md) | Starting configs, defaults, paths, opponents, sampling, and inference batching |
| [Architecture and Rust APIs](docs/architecture.md) | Crate responsibilities and examples for matches and training |
| [Development](docs/development.md) | Build setup, feature flags, tests, and GPU checks |
| [Research](research/README.md) | Study protocols, prerequisites, results, checkpoint diagnostics, and benchmarks |

Detailed contracts live beside the code:
[game rules and observations](crates/sequence-core/README.md) and
[match execution and replay](crates/sequence-runner/README.md).
