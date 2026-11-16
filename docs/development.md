# Development

[Start here](../README.md) · [Architecture and Rust APIs](architecture.md) · [Configuration](configuration.md)

## Build

Use a Rust toolchain that supports edition 2024. Cargo uses its default linker
unless you configure another one locally.

For optional faster linking on `x86_64-unknown-linux-gnu`, install `clang` and
`/usr/bin/mold`, then enable the [example config](../.cargo/example.config.toml):

```sh
cp .cargo/example.config.toml .cargo/config.toml
```

`.cargo/config.toml` is ignored by Git. Adjust it for your local tools as needed.

```sh
cargo build --release -p sequence-cli
cargo run --release -p sequence-cli -- --help
```

CLI default features enable Burn for training and neural inference. The runner
has no direct Burn dependency. For a lightweight heuristic/replay-only executable:

```sh
cargo run -p sequence-cli --no-default-features -- eval random greedy --games 10
```

Python study drivers use Python 3. Plotting/analysis scripts with inline dependency
metadata run through `uv run`, which installs their dependencies separately.
Run commands from the repository root.

## Checks

```sh
cargo test --workspace
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
python3 -m unittest discover -s tools/tests
```

The CUDA integration check is explicit and requires host GPU access:

```sh
cargo test -p sequence-training --features burn cuda_training_checkpoint_and_match -- --ignored
cargo test -p sequence-cli cuda_training_cli_roundtrip -- --ignored
```

The local GPU is an NVIDIA GTX 1660 Ti with 6 GB VRAM. A sandbox may hide it,
causing `CUDA_ERROR_NO_DEVICE` or `nvidia-smi` failures despite a working host
driver. Verify GPU access outside the sandbox before concluding CUDA is
unavailable or changing GPU tests. Run GPU-dependent checks there when needed.

For throughput measurements, see [the GPU benchmark](../research/benchmarks.md).
