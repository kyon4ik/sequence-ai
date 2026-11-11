# Sequence AI

Tools for training and evaluating agents for two-player Sequence.

The workspace follows the intended dependency boundaries:

| Crate | Responsibility |
| --- | --- |
| `sequence-core` | Rules, state, legal actions, player observations |
| `sequence-agent` | Common policy/factory interface, heuristic and neural inference, agent definitions |
| `sequence-runner` | Individual matches, rewards, event recording, per-match metrics, replay |
| `sequence-training` | Learning algorithms; currently Monte Carlo self-play using the runner |
| `apps/cli` | `sequence` commands, file artifacts, experiment orchestration, local replay explorer |

Dependencies point from training to runner to agent to core. Runner has no Burn
dependency of its own. Neural support is opt-in through the `burn` feature of
agent/training. Existing rules, afterstate encoding, and checkpoint tensor layouts
are preserved. The former `sequence-bots` crate is now `sequence-agent`, under
`crates/`; its training API moved to `sequence-training::mc`.

The CLI currently orchestrates round-robin scheduling, run-local Elo, and JSON/file
persistence in small application modules. These can move into `sequence-tournament`
and `sequence-storage` when shared consumers need them; SQLite, a REST/WebSocket
service, and a standalone web application remain future layers.

## CLI: train your first agent

From the repository root:

```sh
# GPU training on this machine; omit --device for the config's CPU default.
cargo run -p sequence-cli -- train --config configs/experiments/mc.toml --device cuda:0

# Evaluate the saved checkpoint; seats alternate between games.
cargo run -p sequence-cli -- eval mc-v1 greedy --games 100 --device cuda:0

# Three-agent round robin with fresh Elo ratings.
cargo run -p sequence-cli -- tournament --config configs/experiments/league.toml --device cuda:0

# Use a replay ID printed by evaluation/tournament.
cargo run -p sequence-cli -- replay <game-id> --frame 10
cargo run -p sequence-cli -- serve --port 8080
```

Open `http://127.0.0.1:8080` to view trained-agent summaries and run standings,
select games, scrub through frames, play/pause, and inspect hands, moves, sequences,
rewards and decision times. Left/right arrow keys also step through the replay.
The UI is served locally by the CLI; it does not start training jobs.

Install the executable for the shorter `sequence ...` commands:

```sh
cargo install --path apps/cli --locked
sequence --help
```

Training currently supports **Monte Carlo (`algorithm = "mc"`) on Simple rules**.
PPO and DQN are not implemented; their algorithm names fail config validation.
[The example experiment](configs/experiments/mc.toml) contains the training
parameters. CLI training enables Burn by default. `--device cpu` and
`--device cuda:0` select training/inference hardware explicitly; CUDA failures do
not silently switch to CPU. Use a release build for longer experiments:
`cargo run --release -p sequence-cli -- train --config configs/experiments/mc.toml`.
An initial small training run verifies the workflow; playing strength requires
more training and evaluation.

Each model ID is immutable: choose a new `id` in the config or pass `--id mc-v2`
for the next run. Training prints iteration progress (MSE, epsilon, sample counts,
outcomes and elapsed time), then saves the model and report. Configs reject unknown
fields and invalid hyperparameters before creating artifacts. An interrupted or
failed training run may leave an incomplete directory without `agent.json`; choose
a new ID for a retry.

```text
data/
├── models/<agent-id>/
│   ├── model.bpk          # Neural checkpoint
│   ├── agent.json         # Shared AgentSpec; published after successful training
│   ├── experiment.json    # Resolved config, overrides and seed
│   └── report.json        # Iteration losses, outcomes, rewards and timings
├── replays/<game-id>.json # Every evaluation/tournament game
└── runs/<run-id>/
    ├── agents.json        # Input agent definitions
    └── report.json        # Seeds, game/replay IDs, results, timings, rewards, Elo
```

`--data-dir PATH` is a global option accepted before or after the subcommand;
relative paths use the working directory. The directory defaults to `data/` and is
ignored by Git. Storage currently uses files, not SQLite. A saved `agent.json`
references its checkpoint relative to itself, so the model directory is portable.
`eval` accepts a saved ID, an explicit `agent.json` path, or a built-in `random`,
`greedy` or `heuristic` (`heuristic` aliases `greedy`). Both heuristic and neural
policies run through the existing `Agent` / `AgentFactory` interfaces. Custom
heuristic weights can be stored as an `AgentSpec` JSON file.

[The league config](configs/experiments/league.toml) specifies agent references,
games per unordered pair, seed, action limit, device, initial Elo and K factor.
Agent spec paths inside a league config resolve relative to that config; model IDs
resolve in the selected data directory. Seats alternate for each pair (an odd
number gives one extra Red game to the first agent). Each game uses explicit game
and policy RNG seeds. Elo starts fresh for every evaluation or tournament, updates
in scheduling order, and is not a persistent global rating. Action-limited games
are counted as truncated, never as draws, and do not update Elo. Reports retain
both aggregate decision timings and each match's metrics; full replay records
also retain individual move times. Match elapsed time excludes model loading;
run elapsed time includes it. Exact timings and GPU results are not reproducible.

`replay ID` validates the record against the current engine before displaying it.
Use `--frame 0` for the initial deal, `--frame N` for a position, or `--json` for the
complete validated record. The default is the final frame. An explicit JSON file
path is also accepted. Replays need no checkpoint or inference backend. The browser
reads the recorded snapshots and shows the next actor's hand; opponent hand contents
are hidden by the runner's recording contract. The explorer lists saved evaluation
and tournament games; training currently saves aggregate metrics, not self-play
replays.

For a lightweight heuristic/replay-only executable:

```sh
cargo run -p sequence-cli --no-default-features -- eval random greedy --games 10
```

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
cargo test -p sequence-cli cuda_training_cli_roundtrip -- --ignored
```
