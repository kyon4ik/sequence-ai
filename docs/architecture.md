# Architecture and Rust APIs

[Start here](../README.md) · [Usage](usage.md) · [Configuration](configuration.md) · [Development](development.md)

The workspace has these responsibilities:

| Crate | Responsibility |
| --- | --- |
| `sequence-core` | Rules, state, legal actions, player observations |
| `sequence-agent` | Common policy/factory interface, heuristic and neural inference, agent definitions |
| `sequence-runner` | Individual matches, rewards, event recording, per-match metrics, replay |
| `sequence-training` | Learning algorithms; currently Monte Carlo training with configurable opponents |
| `apps/cli` | `sequence` commands, file artifacts, experiment orchestration, local replay explorer |

Dependencies point from training to runner to agent to core. Runner has no Burn
dependency of its own. Neural support is opt-in through the `burn` feature of
agent/training; the CLI enables it by default.

The CLI owns round-robin scheduling, run-local Elo, file persistence, and the
local explorer. Libraries return data and reports; the application chooses how
to display and persist them. See [artifact layout](usage.md#saved-artifacts).

The core implements Classic and Simple rules. Random agents support both;
greedy and Monte Carlo agents currently support Simple. CLI training, evaluation,
and tournaments use Simple.

## Run and replay a match

```rust
use sequence_agent::{AgentConfig, AgentDescriptor, AgentSpec};
use sequence_core::Simple;
use sequence_runner::{MatchConfig, MatchRunner, Replay};

let random = AgentSpec {
    descriptor: AgentDescriptor::new("random-v1", "Random"),
    config: AgentConfig::Random,
};
let greedy = AgentSpec {
    descriptor: AgentDescriptor::new("greedy-v1", "Greedy"),
    config: AgentConfig::greedy(),
};
let mut runner = MatchRunner::<Simple>::default();
let record = runner.run_factories(MatchConfig::default(), [&random, &greedy])?;
let json = serde_json::to_string(&record)?; // caller chooses where to store it
let saved = serde_json::from_str(&json)?;
let mut replay = Replay::<Simple>::new(&saved)?;
replay.seek(10)?;
let branch = replay.game().clone(); // inspect or manually continue this state
```

This snippet runs inside a function returning `Result<(), Box<dyn std::error::Error>>`.
See [runner contracts](../crates/sequence-runner/README.md) for events, rewards,
error handling, snapshots, and metrics. A runnable example writes the match JSON
to stdout:

```sh
cargo run -p sequence-runner --example match > /tmp/sequence-match.json
```

`AgentSpec` is serializable: heuristics save all policy parameters; neural agents
save a checkpoint reference. The caller resolves that path. Give each trained
checkpoint its own descriptor ID so future rankings distinguish model versions.
Specs use CPU inference by default. Use
`spec.on_device(InferenceDevice::Cuda(0))` to select CUDA explicitly. Missing
checkpoints and disabled Burn support return errors instead of panicking.

## Train and save a model

With `sequence-training/burn` enabled:

```rust
use burn::prelude::*;
use sequence_agent::mc::ValueNet;
use sequence_training::mc::{train, TrainingConfig};

let device = Device::cuda(0); // or Device::flex() for CPU
device.seed(42);
let output = train(
    ValueNet::new(&device),
    device,
    &TrainingConfig::default(),
    None, // Resume checkpoint directory, if any.
    |_| Ok(()), // Called after each completed iteration.
)?;
output.model.save_file("data/models/mc-v1.bpk")?;
// Save output.report and an AgentSpec pointing to the checkpoint alongside it.
```

The single `train` entry point handles fresh and resumed runs. Its callback
receives a `TrainingCheckpoint`; call `checkpoint.report()` for progress or
`checkpoint.save(path)` to publish complete state. Pass `Some(path)` as the
resume argument to continue a saved run.

The caller creates output directories. Reports contain per-iteration loss,
epsilon, sample and episode counts, outcomes, rewards, and execution timings.
Training reports are returned rather than printed, so a CLI or experiment store
can decide how to consume them. Seed the tensor device before model construction
as well as supplying the training seed. Exact GPU reproducibility also depends on
backend/kernel behavior.

## Contracts

- [Game rules and observations](../crates/sequence-core/README.md): legal actions, hidden information, variants, and checked state changes.
- [Match execution and replay](../crates/sequence-runner/README.md): events, rewards, failures, timings, snapshots, and format compatibility.
- [Training configuration](configuration.md): opponent scheduling, sample selection, and inference batching.
- [Resume behavior](usage.md#training-snapshots-and-evaluation): complete checkpoints and compatible configuration changes.
