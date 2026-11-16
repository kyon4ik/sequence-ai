# Usage

[Start here](../README.md) · [Configuration](configuration.md) · [Architecture](architecture.md) · [Development](development.md)

Commands assume you installed `sequence` as described in the README and are
running from the repository root. Use `sequence <command> --help` for all flags.

## Train

```sh
sequence train --config configs/experiments/mc.toml
sequence eval mc-v1 greedy --games 100
```

The introductory config is a small CPU run to verify the workflow. For longer
training, use `configs/experiments/mc-stable.toml`, which selects CUDA by default:

```sh
sequence train --config configs/experiments/mc-stable.toml
sequence eval mc-stable-v1 greedy --games 1000 --seed 123 --device cuda:0
```

Pass `--device cpu` or `--device cuda:0` to override either training config.
Evaluation defaults to CPU independently of the training device. CUDA failures
do not silently switch to CPU. These runs need further evaluation before drawing
conclusions about playing strength.

Use `--seed 43 --id mc-seed-43` to repeat an experiment with another training
seed. Seeds and effective execution settings are saved in `experiment.json`.

Each model ID is immutable: choose a new `id` in the config or pass `--id mc-v2`
for the next run. Training prints iteration progress (MSE, epsilon, sample counts,
evaluation scores and timings), then saves the model and report. Configs reject
unknown fields and invalid hyperparameters before creating artifacts. An interrupted or
failed training run may leave an incomplete directory without `agent.json`; choose
a new ID for a retry.

## Saved artifacts

```text
data/
├── models/<agent-id>/
│   ├── model.bpk          # Neural checkpoint
│   ├── agent.json         # Shared AgentSpec; published after successful training
│   ├── experiment.json    # Resolved config, overrides and seed
│   ├── report.json        # Iteration losses, outcomes, rewards and timings
│   └── progress.json      # Latest completed iteration, updated during training
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
`greedy`. Both heuristic and neural
policies run through the existing `Agent` / `AgentFactory` interfaces. Custom
heuristic weights can be stored as an `AgentSpec` JSON file.

## Evaluate and run tournaments

```sh
sequence eval mc-v1 greedy --games 100 --seed 123
sequence tournament --config configs/experiments/league.toml
```

[The league config](../configs/experiments/league.toml) specifies agent references,
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

## Checkpoint panels

Evaluate several checkpoints against the same opponents and seat-swapped game
seeds without saving full replays:

```sh
sequence panel --config panel.json --output data/panel-evaluation
```

See [panel manifests and diagnostics](../research/evaluation.md) for the JSON
format. Reports include checkpoint hashes so the champion gate can verify that
its evidence belongs to the evaluated policies.

## Champion promotion

Training publishes a model version. To update a champion pointer, evaluate both
the candidate and incumbent with `sequence panel` on the same fixed panel and
seed schedule, then run:

```sh
sequence promote \
  --incumbent data/panel-evaluation/incumbent-123.json \
  --candidate data/panel-evaluation/candidate-123.json \
  --incumbent-spec data/models/incumbent/agent.json \
  --candidate-spec data/models/candidate/agent.json \
  --registry data/champions/sequence.json \
  --decision data/promotion/decision.json
```

Use at least 400 completed games per opponent with paired seats. The gate checks
checkpoint hashes, opponent policies, game/policy seeds, seats, action limits,
device, and evaluation configuration. It requires a positive lower bound on mean
improvement across opponents and a lower bound above -0.05 against each opponent.
Bounds use 5,000 paired seed-block bootstrap resamples with a fixed Rust RNG seed,
and Bonferroni adjustment across the primary and per-opponent checks.
`--family-alpha` defaults to 0.05.

An optional `--confirmation analysis-decision.json` also requires its `confirmed`
boolean to be true. The gate binds that file and both reports/specs by hash.
Decisions are immutable: rerunning identical inputs reuses the decision; changed
inputs or thresholds require a new decision path. A failed gate exits nonzero
and preserves an existing champion file byte-for-byte. An absent registry is
initialized with the explicitly evaluated incumbent. Successful promotion
atomically updates the registry to a loadable AgentSpec:

```sh
sequence eval data/champions/sequence.json greedy --games 1000
```

## Head-to-head tournament statistics

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

## Replay and the local explorer

```sh
sequence replay <run-id>/<game-id> --frame 10
sequence serve --port 8080
```

Open `http://127.0.0.1:8080` to view trained-agent summaries and run standings,
select a replay folder and game, scrub through frames, play/pause, and inspect
hands, moves, sequences, rewards and decision times. Left/right arrow keys step
through the replay. The UI does not start training jobs.

`replay <folder>/<game>` validates the record against the current engine before
displaying it. Use `--frame 0` for the initial deal, `--frame N` for a position, or
`--json` for the complete validated record. The default is the final frame. An
explicit JSON file path is also accepted. Replays need no checkpoint or inference
backend. The browser reads the recorded snapshots and shows the next actor's hand;
opponent hand contents are hidden by the runner's recording contract. The explorer
lists saved evaluation and tournament games; training currently saves aggregate
metrics, not self-play replays.

Replays live in `replays/<folder>/<game>.json`; evaluation and tournament runs use
`<run-id>/game-NNNNNN`. Use `sequence replay <run-id>/game-NNNNNN` or an explicit
file path. In the browser, select a folder first, then a game inside it. Flat files
directly under `replays/` are ignored; move them into a folder to browse them.

## Training snapshots and evaluation

The CLI atomically publishes complete version-3 training checkpoints. By default
it saves every iteration; set `--checkpoint-interval N` to save less often.
The final iteration is always saved, and `progress.json` is updated every iteration:
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
`training.inference_batch_games` may change on resume. The current network has no
stochastic tensor layers; episode RNGs are recreated from seeds emitted by the
saved training RNG. CPU continuation is tested against uninterrupted training;
GPU numerical determinism still depends on kernels and hardware. Resuming requires a complete version-3 checkpoint; inference weights alone
cannot restore training state.

`training.evaluation_interval` (default 10) controls periodic evaluation; the final
iteration is always evaluated. `training.evaluation_games` (default 20, positive
and even) is the number of games **per baseline**, against random and greedy.
Evaluation uses zero exploration, a fixed seed schedule independent of rollout
RNG, paired game seeds, and balanced seats. `evaluation_score` is the mean score
(win 1, draw 0.5, loss 0) across both baselines. `best.bpk` retains the earliest
highest-scoring evaluated checkpoint; `model.bpk` remains the latest model.
Use a separate unseen seed set for final comparisons, and repeat training across
multiple seeds before drawing strength conclusions.

## Plot training metrics

Generate a standalone interactive dashboard, PNG, SVG, CSV and summary JSON:

```sh
uv run tools/plot_training.py data/models/mc-v1/report.json \
  --output data/plots/mc-v1
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

For controlled checkpoint panels and tactical diagnostics, see [research evaluation](../research/evaluation.md).
