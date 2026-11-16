# Sequence AI

Tools for training and evaluating agents for two-player Sequence.

The workspace follows the intended dependency boundaries:

| Crate | Responsibility |
| --- | --- |
| `sequence-core` | Rules, state, legal actions, player observations |
| `sequence-agent` | Common policy/factory interface, heuristic and neural inference, agent definitions |
| `sequence-runner` | Individual matches, rewards, event recording, per-match metrics, replay |
| `sequence-training` | Learning algorithms; currently Monte Carlo training with configurable opponents |
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
more training and evaluation. For the confirmed full-budget configuration, use
`configs/experiments/mc-stable.toml` (200 iterations, 128 games per iteration,
learning rate 0.0001, uniform historical opponents, replay off).

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

### Training against different opponents

Run the supplied 40% self-play, 20% best, 15% previous, 15% random checkpoint,
10% greedy experiment:

```sh
cargo run --release -p sequence-cli -- train --config configs/experiments/mc-mixed.toml
```

Any training config can specify relative game weights:

```toml
[training.opponents]
self_play = 40
best_checkpoint = 20
previous_checkpoint = 15
random_checkpoint = 15
greedy = 10
```

Omitting the table uses 100% self-play. Missing entries within a supplied table
are zero. Weights must be finite and nonnegative, with a positive sum. Each
iteration uses proportional game counts with largest-remainder rounding (128
games become 51/26/19/19/13), balanced learner seats per category, and a shuffled,
seeded schedule. Equal rounding remainders and odd seat counts rotate between
iterations. Weights describe games; sample proportions differ because self-play
trains on both seats and all other games train only on the learner's decisions.

The learner uses its epsilon schedule. Frozen opponents use zero exploration.
Best means the earliest highest-scoring checkpoint under the existing fixed
random/greedy evaluation. Previous is one snapshot older than the learner's
current weights: when the learner starts iteration N with checkpoint N−1,
previous is N−2. Random samples uniformly per game from checkpoints older than
the current learner. The initial network is checkpoint 0. Categories without an
available checkpoint temporarily use self-play; reports record these fallbacks.
Best may coincide with the current learner's weights, with exploration disabled
on the opponent seat.

Historical weights stay on disk. Rollouts group games into waves of at most four
distinct frozen opponents, batch inference by model, and restore game order for
target extraction. Every mixed-training snapshot includes an `opponents/` archive
and the pool manifest, best checkpoint ID and scheduler RNG in `state.json`.
The runtime archive moves onto the checkpoint filesystem on first publication;
immutable weights are hard-linked where possible, with copying as a fallback.
Each snapshot can be moved and resumed independently, including into a new model
ID. The opponent mix is part of the learning configuration and must match on
resume. Legacy self-play snapshots remain resumable with self-play configs.

`report.json` includes per-category and actual-opponent game/sample counts,
learner seats, learner wins/losses/draws and fallback counts. Self-play outcomes
use a designated learner seat, while samples include both seats. Periodic
evaluation separately records random, greedy, and historical scores against the
initial model, lagged model and incumbent best (duplicate checkpoint IDs are
evaluated once). Historical games use a separate fixed seed schedule, balanced
seats and zero exploration; they do not affect best-model selection. Each uses
`evaluation_games` games, so historical evaluation adds work. The plotting tool
shows category proportions, samples and training scores, plus fixed-opponent
evaluation. Use unseen seeds for final strength comparisons.

To give the requested opponent proportions the same influence on gradient
updates, use [the balanced-sampling config](configs/experiments/mc-balanced.toml),
or add these settings inside `[training]` (before `[training.opponents]`):

```toml
sampling = "opponent_mix"
updates_per_iteration = 64
```

`sampling = "natural"` is the default. Without a fixed update budget it uses all
fresh samples once. With `updates_per_iteration`, it samples uniformly with
replacement from the fresh samples. `opponent_mix` allocates optimization
samples proportionally to the requested opponent weights and samples uniformly
with replacement within each category. Categories without fresh samples in a
small rollout are excluded and the available weights are renormalized. Fallback
self-play samples keep their requested category. Each iteration shuffles the
selected samples before gradient batches are formed.

With 64 updates and a batch size of 128, both strategies use exactly 8,192
optimization samples per iteration. Repeated selections are not additional game
experience. Reports distinguish fresh `samples`, `optimization_samples` and
`updates`, including optimization sample counts by opponent. These settings are
part of the immutable resume configuration. Fixed-budget runs derive episode
seeds independently of optimization sampling, allowing paired experiments to
use the same episode seeds even after their policies diverge.

The controlled study runner alternates natural and balanced sampling across
three training seeds, checks equal update budgets, and evaluates each run's
best/final weights on an identical unseen seed schedule:

```sh
cargo build --release -p sequence-cli
cargo build --release -p sequence-training --features burn --example evaluate
python3 tools/opponent_study.py run --iterations 100 --seeds 42,43,44 \
  --updates 64 --games 1000 --device cuda:0
```

Configurations, training logs, models, exact per-game evaluation seeds/scores,
and Markdown/CSV/JSON summaries are saved under
`data/studies/opponent-sampling-20261113/`. Interrupted training resumes into a
new model ID on rerun; completed runs/evaluations are reused. Use `--output PATH`
for another study. The summaries include paired bootstrap comparisons grouped by
game-seed pairs and separate comparisons across training seeds. Three seeds and
100 iterations provide an exploratory comparison, rather than establishing
performance at the original 200-iteration horizon.

Rebuild the summary from existing results or export a standalone study figure:

```sh
python3 tools/opponent_study.py summarize
uv run tools/plot_opponent_study.py data/studies/opponent-sampling-20261113/summary.json
```

`sequence-training::mc::evaluate_seeded_baselines` also exposes this batched,
zero-exploration evaluation for Rust callers. The `evaluate` example accepts a
JSON manifest listing checkpoint paths, device, game count, inference batch size
and seeds. It saves summary statistics and per-game scores without full replays.

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

By default, exploration decays over the original total iteration count. To
extend a completed 200-iteration run, set `iterations = 1000` and
`epsilon_decay_iterations = 200` in the new config, then resume from its full
iteration-200 checkpoint into a new ID. Exploration stays at its final value
after iteration 200. Resume requires an unchanged effective decay horizon and
rejects other learning changes or a reduced iteration budget. The execution setting
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
those games using the same policy share one network forward pass and one
synchronized result readback.
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

### Adaptive historical opponents

`configs/experiments/mc-history-uniform.toml` and
`configs/experiments/mc-history-adaptive.toml` provide matched 80% self-play,
20% historical-opponent experiments. Compare both with `mc-mixed.toml`.
Use distinct model IDs and several training seeds for each condition.

```sh
cargo run --release -p sequence-cli -- train --config configs/experiments/mc-history-adaptive.toml
```

The optional `[training.historical_sampling]` table controls the
`random_checkpoint` category only:

- `adaptive = false` defaults to uniform sampling, preserving existing behavior.
- `snapshot_interval = 1` defaults to every historical checkpoint. The experiment
  configs use 10; iteration 0 is eligible, and the current checkpoint is excluded.
- `learning_rate = 0.01` controls quality decreases after learner wins.
- `uniform_fraction = 0.0` optionally mixes uniform exploration into softmax.

Adapted from [OpenAI Five, Appendix N](https://cdn.openai.com/dota-2.pdf),
new eligible opponents receive the maximum existing quality. Adaptive sampling
uses softmax quality probabilities; learner wins decrease quality by
`learning_rate / (eligible_count * sampling_probability)`. Losses and draws
leave it unchanged. With uniform mixing enabled, the denominator uses the actual
mixed probability. This trainer schedules games together: probabilities are
frozen for an iteration and updates affect the next iteration. Consequently this
is a batched adaptation, not an exact reproduction of Five's online scheduler.

Quality values and scheduler RNG are saved in checkpoint state. Sampling settings
must match when resuming. Reports include `historical_distribution` (checkpoint,
quality and probability before rollout), alongside per-opponent outcomes. The
snapshot interval controls sampling eligibility, not archive storage: all weights
remain archived for best/previous opponents and existing resume compatibility.

Compare equal interaction budgets, report optimization sample counts and elapsed
time, and evaluate selected checkpoints on a separate fixed opponent panel with
paired seeds and both seats. Adaptive training win rate is not a stable strength
metric. These configs do not establish that adaptive sampling improves Sequence;
that requires multi-seed held-out evaluation. Existing training-time best selection
still uses random/greedy baselines. Counter-strategy training and tactical curricula
are not enabled by these settings.

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

### Investigating checkpoint strength

The `sequence-training` example `evaluate` accepts an optional `opponents` list
of `[name, AgentConfig]` pairs. Defaults remain random and greedy. Frozen neural
opponents use GPU batching on the selected device. All candidates share the same
seed schedule; each seed is evaluated in both seats. For example, add to the JSON
manifest:

```json
{
  "opponents": [
    ["greedy", {"kind": "greedy", "weights": [0, 0, 1, 5, 10, 100]}],
    ["reference", {"kind": "monte_carlo", "checkpoint": "data/models/reference/model.bpk"}]
  ],
  "diagnostic_games": 200
}
```

Paths resolve from the working directory. Optional diagnostics inspect the first
N games per opponent, count legal immediate wins taken/missed, retain reproducible
miss examples, and summarize chosen value predictions versus actual outcomes.
These are matchup-specific descriptions; correlated decisions within games are
not independent calibration trials. Diagnostics never expose hidden opponent
hands to the policy. Simple rules have no jacks or exchanges.

`tools/analyze_history_diagnosis.py STUDY_DIRECTORY` reads `baselines/`, `panel/`,
and `diagnostics/` evaluation outputs, exports CSV/JSON, and plots scores and value
reliability. Its paired comparisons verify identical seeds and keep both seats
together in each bootstrap block. Pointwise intervals reflect game-seed uncertainty
for these checkpoints, not independent-training-seed uncertainty or a guarantee
against multiple-comparison error.

### Controlled regression interventions

`tools/regression_ablation.py` prepares and runs matched research continuations
from the seed-42 uniform/adaptive checkpoint-50 archives. Build the examples,
then run the driver from the repository root with host GPU access:

```sh
cargo build --release -p sequence-training --features burn --example ablation --example evaluate
python tools/regression_ablation.py run
uv run tools/analyze_regression_ablation.py data/studies/regression-ablation
```

The prespecified screening plan uses three continuation seeds, five conditions,
50 additional iterations, 64 games and 27 batches of 128 samples per iteration.
Conditions are control, 3x lower learning rate, 25% retained samples from a bounded
50,000-sample FIFO, 20% historical gradient samples, and exploration fixed at the
checkpoint-50 value. Replay starts empty and retains previous continuation
iterations with their original Monte Carlo labels; this tests coverage and
stale/off-policy targets together. The architecture and checkpoint weights are
identical at each matched start; Adam state and historical policies are restored.

This experimental path is separate from normal resume and never changes source
artifacts. Partial research runs are preserved but cannot resume; rerunning the
driver uses a new attempt directory. Normal training resume still rejects
learning changes. The default study output is `data/studies/regression-ablation`.
It contains the immutable plan, configuration/source hashes, logs, models,
per-game evaluation results, summary CSV/JSON, report and plot. Each final model
plays 400 games per reference on a shared seat-swapped seed schedule. Primary
results equally average greedy and the two frozen checkpoint-50 references;
random is secondary. These continuation replicates are conditional on two
starting policies from one initial training seed, not independent training runs.

### Full-budget learning-rate confirmation and champion selection

Run the prespecified study with host GPU access:

```sh
cargo build --release -p sequence-training --features burn --example train_study --example evaluate
python tools/learning_rate_confirmation.py run
```

Six fresh runs compare learning rates 0.0003 and 0.0001 with seeds 42–44,
128 games per iteration, 200 iterations, uniform historical sampling, and the
original exploration schedule. Checkpoints 50/100/150/200 share a curve evaluation
schedule. The final release comparison uses a separate schedule plus an additional
frozen mixed-training opponent. Each matchup uses 1,000 games, both seats, and
paired seeds. The plan and input/reference/binary hashes are saved before results.

If the prespecified learning-rate confirmation rule passes, the workflow runs
three further fresh 200-iteration low-rate agents with retained samples. That
comparison uses another release schedule. All results, logs, immutable model
versions, full checkpoints and decisions live under
`data/studies/learning-rate-confirmation/`. Interrupted runs resume from the latest
50-iteration snapshot into a new attempt directory.

Retained samples are an opt-in training setting:

```toml
[training.replay]
capacity = 50000
fraction = 0.25
```

The buffer starts empty. Later iterations replace the requested fraction of their
existing optimization slots with samples from previous iterations. Original Monte
Carlo outcomes are retained, so labels become off-policy. Total gradient samples
and batches per iteration remain unchanged; per-opponent optimization counts plus
`replay_samples` equal `optimization_samples`. Reports also include `replay_size`.
Complete replay-enabled snapshots include `replay.bin`; missing or malformed replay
state fails resumption rather than silently resetting the buffer. Normal resume
continues to require the same replay and other learning settings.

Training publishes a versioned `agent.json` for inspection, not a champion update.
`tools/checkpoint_gate.py` controls the separate champion pointer. It checks bound
checkpoint hashes, identical opponent policies, matching game/policy seeds and
seats, a positive lower bound on overall improvement, and per-opponent lower bounds
above a five-percentage-point regression margin. The study preselects seed 42 and
the final checkpoint; it does not search for a successful seed on release results.
A failed gate preserves an existing champion file byte-for-byte. The two possible
study gates each use family alpha 0.025, split across the primary and opponent
checks. An absent registry is initialized with the explicit incumbent.

The resulting champion pointer is a normal loadable AgentSpec with audit metadata:

```sh
sequence eval data/champions/sequence.json greedy --games 1000 --device cuda:0
```

The completed confirmation established a release-panel gain of 16.95 percentage
points for learning rate 0.0001 over 0.0003 (paired 95% interval +13.59 to +20.15).
The training default and introductory config now use 0.0001. Use
`configs/experiments/mc-stable.toml` for the full-budget settings. The original
mixed/history comparison configs retain their explicit 0.0003 rate to reproduce
those experiments; they override the new default.

Combining the lower rate with 25% retained samples showed no additional gain:
-0.27 points (95% interval -1.80 to +1.40) on its separate release schedule.
Replay remains opt-in and its candidate was not promoted. The champion pointer
retains the preselected low-rate seed-42 model, which passed the release gate.
Three training seeds still limit population-level inference. Mean late-training
retention improved, but its interval includes zero; endpoint strength is the
stronger finding. Natural optimization counts can vary with game length despite
equal interaction budgets.

Saved results: `data/studies/learning-rate-confirmation/report.md`,
`confirmation-curves.png`, `scores.csv`, and the two `*-decision.json` /
`promotion-*.json` pairs. Training and release decisions are separate: creating a
new model version does not replace `data/champions/sequence.json`.
