# Development

[Start here](../README.md) · [Architecture and Rust APIs](architecture.md) · [Configuration](configuration.md)

## Build

```sh
cargo build --release -p sequence-cli
cargo run --release -p sequence-cli -- --help
```

CLI default features enable Burn for training and neural inference. The runner
has no direct Burn dependency. For a lightweight heuristic/replay-only executable:

```sh
cargo run -p sequence-cli --no-default-features -- eval random greedy --games 10
```

Python is optional and only used for offline analysis and plotting. Scripts with
inline dependency metadata run through `uv run`. Training, panel evaluation,
and champion promotion run through `sequence`.
Run commands from the repository root.

## Checks

```sh
cargo test --workspace
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
python3 -m unittest discover -s tools/tests
```

CUDA integration checks require an NVIDIA GPU:

```sh
cargo test -p sequence-training --features burn cuda_training_checkpoint_and_match -- --ignored
cargo test -p sequence-cli cuda_training_cli_roundtrip -- --ignored
```

For throughput measurements, see [the GPU benchmark](../research/benchmarks.md).
