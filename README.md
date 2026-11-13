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

# Use a folder/game reference printed by evaluation/tournament.
cargo run -p sequence-cli -- replay <run-id>/<game-id> --frame 10
cargo run -p sequence-cli -- serve --port 8080
```

Open `http://127.0.0.1:8080` to view trained-agent summaries and run standings,
select a replay folder and then a game, scrub through frames, play/pause, and inspect
hands, moves, sequences, rewards and decision times. Left/right arrow keys also step
through the replay.
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
├── replays/<run-id>/game-NNNNNN.json # Games grouped by evaluation/tournament run
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

### Head-to-head tournament statistics

Select a run in the local explorer to see its **Head-to-head results** matrix.
Rows are the agent being measured and columns are its opponents. Switch between
win rate and score rate (win 1, draw 0.5, loss 0). Every cell shows wins/losses/draws;
select it for results as Red and Blue, 95% Wilson win-rate intervals, move timings,
and a replay list filtered to that pair. **Show all games** clears the filter.
The matrix includes a CSV download. A run can be opened directly with
`http://127.0.0.1:8080/?run=<run-id>`.

Analyze an existing tournament or evaluation without loading any model or replay:

```sh
sequence analyze <run-id>
sequence analyze <run-id> --format csv > pairwise.csv
sequence analyze data/runs/<run-id>/report.json --format json
```

`--data-dir` selects the store when using a run ID. Explicit report paths also
work independently of the store. JSON contains a directional `matrix[row][opponent]`;
CSV has one row per observed direction, so a seven-agent round robin has 42 rows.
Agent indices distinguish duplicate IDs in self-evaluations. Unplayed pairs are
null, and pairs with no completed games have null rates. Truncated and aborted
games are reported separately and excluded from rate denominators. Win-rate
intervals treat draws as non-wins; they are not confidence intervals for score rate.

The explorer reads the same summary through `/api/runs/<run-id>/pairwise` and
`/api/runs/<run-id>/pairwise.csv`, so reports from earlier runs need no migration.

`replay <folder>/<game>` validates the record against the current engine before
displaying it. Use `--frame 0` for the initial deal, `--frame N` for a position, or
`--json` for the complete validated record. The default is the final frame. An
explicit JSON file path is also accepted. Replays need no checkpoint or inference
backend. The browser reads the recorded snapshots and shows the next actor's hand;
opponent hand contents
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

### Training snapshots and evaluation

Training atomically publishes a directory after every completed iteration:
`models/<id>/checkpoints/iteration-NNNNNN/`. It contains `model.bpk`,
`optimizer.bpk` (Adam moments and step counters), `state.json` (configuration,
rollout/shuffle RNG state, iteration history and best score), and `best.bpk`
when a policy has been evaluated. Earlier snapshots survive interruption.

Resume into a new model ID using the original learning configuration:

```sh
sequence train --config configs/experiments/mc.toml --id mc-resumed \
  --resume data/models/mc-v1/checkpoints/iteration-000005
```

The iteration count remains the original total, preserving the exploration
schedule. Learning configuration changes are rejected. The execution setting
`training.inference_batch_games` may change on resume; old checkpoints and
configs that omit it use the default of 16. The current network has no
stochastic tensor layers; episode RNGs are recreated from seeds emitted by the
saved training RNG. CPU continuation is tested against uninterrupted training;
GPU numerical determinism still depends on kernels and hardware. Old weight-only
snapshots remain valid for inference, but cannot resume training.

Replays live in `replays/<folder>/<game>.json`; evaluation and tournament runs use
`<run-id>/game-NNNNNN`. Use `sequence replay <run-id>/game-NNNNNN` or an explicit
file path. In the browser, select a folder first, then a game inside it. Flat files
directly under `replays/` are ignored; move them into a folder to browse them.

`training.evaluation_interval` (default 10) controls periodic evaluation; the final
iteration is always evaluated. `training.evaluation_games` (default 20, positive
and even) is the number of games **per baseline**, against random and greedy.
Evaluation uses zero exploration, a fixed seed schedule independent of rollout
RNG, paired game seeds, and balanced seats. `evaluation_score` is the mean score
(win 1, draw 0.5, loss 0) across both baselines. `best.bpk` retains the earliest
highest-scoring evaluated checkpoint; `model.bpk` remains the latest model.
Use a separate unseen seed set for final comparisons, and repeat training across
multiple seeds before drawing strength conclusions.

### Batched self-play inference

`training.inference_batch_games` (default 16, positive) controls how many games
advance together during self-play and periodic evaluation. Candidate moves from
those games share one network forward pass and one synchronized result readback.
Set it to 1 for sequential inference. This is independent of `batch_size`, which
controls gradient updates. The policy remains frozen while collecting an
iteration's games. Episode seeds are assigned in input order, exploration RNGs
belong to individual seats, and samples are restored to game/action order before
shuffling. Changing inference batch shapes can change floating-point results and
therefore trajectories; bitwise agreement across batch sizes is not guaranteed.

The runner's `MatchSession` API permits pausing between actions without moving
game execution or rewards into training. Runner remains independent of Burn.
Regular matches still use the same execution code and record full replays.
In batched rollouts, decision time is the shared inference call's wall time
divided among its decisions; it measures attributed cost rather than individual
request latency.

Reports include `evaluation_time`, `rollout_inference` and
`evaluation_inference`. Inference metrics count decisions, network batches and
encoded candidates, plus CPU encoding time and prediction wall time. Prediction
includes packing, upload, GPU work and synchronized readback; these timers do not
isolate CUDA kernel time from transfer or launch latency. Console progress also
shows rollout, optimization and evaluation times.

Measured on a GTX 1660 Ti using the iteration-100 model, epsilon 0.175 and
128 fixed-seed self-play games (median of three warmed runs):

| Concurrent games | Self-play time | Games/second | Evaluation time (200 games) |
| --- | ---: | ---: | ---: |
| 1 | 8.82 s | 14.5 | 7.90 s |
| 8 | 3.60 s | 35.6 | 3.93 s |
| 16 | 3.17 s | 40.4 | 3.27 s |
| 32 | 2.94 s | 43.6 | 2.97 s |

All settings produced 7,737 self-play actions and evaluation score 0.75 on this
workload. CPU candidate encoding took about 0.14 s; inference service time still
dominated. These measurements exclude gradient updates and checkpoint writes.
The portable default remains 16.

An extended comparison warmed the full workload for each size and interleaved
five repetitions of both phases. With the same model and game seeds:

| Concurrent games | Median self-play time | Median evaluation time |
| --- | ---: | ---: |
| 32 | 2.968 s | 3.013 s |
| 64 | 2.888 s | 2.832 s |
| 128 | 2.795 s | 2.761 s |

128 was fastest among the tested sizes, giving about 6% greater self-play
throughput than 32. All evaluation scores remained 0.75. Use 128 for this
128-game workload on this GPU; larger settings cannot increase concurrency
beyond the number of games in the workload. This recommendation applies to
the tested checkpoint, network and hardware.

A separate CLI comparison resumed the same legacy iteration-100 checkpoint
through iteration 103. Excluding the first iteration's warmup, iteration time
averaged 9.91 s before batching and 4.07 s with 32 games (2.4× faster), excluding
checkpoint writes. Sample counts, outcomes, rewards and MSE matched exactly in
those three iterations; this does not guarantee identical trajectories on other
hardware or workloads.

Run the explicit GPU benchmark outside the sandbox, with no competing training:

```sh
SEQUENCE_BENCH_CHECKPOINT="$PWD/data/models/mc-v1-long/checkpoints/iteration-000100/model.bpk" \
  cargo test --release -p sequence-training --features burn cuda_rollout_throughput \
  -- --ignored --nocapture --test-threads=1
```

The checkpoint variable is optional; omit it to benchmark a seeded fresh model.
The benchmark warms each setting, compares 1/8/16/32/64/128 concurrent games on
the same 128 seeded episodes with five interleaved repetitions, validates their
replays, and times evaluation against both baselines. Set
`SEQUENCE_BENCH_BATCH_GAMES=32,64,128` to restrict the sweep. Its output is JSON
with throughput and inference timing breakdowns.

### Plot training metrics

Generate a standalone interactive dashboard, PNG, SVG, CSV and summary JSON:

```sh
uv run tools/plot_training.py data/models/mc-v1-long-batched/report.json \
  --output data/plots/mc-v1-long-batched
```

Open the generated `.html` file directly in a browser. It contains its plotting
library and needs no server or network connection. Hover for values, drag to
zoom, and toggle series through the legend. An interrupted run's
`checkpoints/iteration-NNNNNN/state.json` can also be used as input. Plotting
dependencies are declared in the script and installed separately by `uv`.

The plots separate evaluation strength, training MSE, exploration, self-play
outcomes, sample counts and execution timings. Historical reports lack detailed
inference timers and per-baseline evaluation scores; the exporter leaves missing
timings blank. A lower self-play MSE does not imply a stronger policy.

`agent.json` references the final `model.bpk`, even if an earlier policy evaluated
better. `best.bpk` retains the earliest highest-scoring evaluated policy. To test
it, create another AgentSpec with a distinct descriptor ID and a checkpoint path
pointing to `best.bpk`, then pass that spec's JSON path to `sequence eval`.

Optional evaluation comparisons use `--comparisons comparisons.json`, where the
file contains entries such as
`[{"label":"original","report":"data/runs/<eval-id>/report.json"}]`.
Report paths are relative to the working directory. The dashboard shows win-rate
bars with 95% Wilson intervals (draws count as non-wins). When replay files are
available, it also validates paired game/agent seeds, seats, opponent settings
and device before exporting exact paired win-test results in the summary JSON.
