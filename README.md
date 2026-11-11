# Sequence AI

Tools for training and evaluating agents for two-player Sequence.

The workspace follows the intended dependency boundaries:

| Crate | Responsibility |
| --- | --- |
| `sequence-core` | Rules, state, legal actions, player observations |
| `sequence-agent` | Common policy/factory interface, heuristic and neural inference, agent definitions |
| `sequence-runner` | Individual matches, rewards, event recording, per-match metrics, replay |
| `sequence-training` | Learning algorithms; currently Monte Carlo self-play using the runner |

Dependencies point from training to runner to agent to core. Runner has no Burn
dependency of its own. Neural support is opt-in through the `burn` feature of
agent/training. Existing rules, afterstate encoding, and checkpoint tensor layouts
are preserved. The former `sequence-bots` crate is now `sequence-agent`, under
`crates/`; its training API moved to `sequence-training::mc`.

The next layers remain separate: `sequence-tournament` will own scheduling, Elo,
and rankings; `sequence-storage` will own SQLite, replay persistence, and model
artifacts; `sequence-api` and `apps/{cli,web}` will consume these libraries.
No empty placeholder crates are required yet.

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
# Ok::<(), Box<dyn std::error::Error>>(())
```

See [runner contracts](crates/sequence-runner/README.md) for events, rewards,
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

```rust,ignore
use burn::prelude::*;
use sequence_agent::mc::ValueNet;
use sequence_training::mc::{train, TrainingConfig};

let device = Device::cuda(0); // or Device::flex() for CPU
device.seed(42);
let output = train(ValueNet::new(&device), device, &TrainingConfig::default())?;
output.model.save_file("data/models/mc-v1.bpk")?;
// Save output.report and an AgentSpec pointing to the checkpoint alongside it.
```

The caller creates output directories. Reports contain per-iteration loss,
epsilon, sample and episode counts, outcomes, rewards, and execution timings.
Training reports are returned rather than printed, so a CLI or experiment store
can decide how to consume them. Seed the tensor device before model construction
as well as supplying the training seed. Exact GPU reproducibility also depends on
backend/kernel behavior.

## Checks

```sh
cargo test --workspace
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

The CUDA integration check is explicit and requires host GPU access:

```sh
cargo test -p sequence-training --features burn cuda_training_checkpoint_and_match -- --ignored
```
