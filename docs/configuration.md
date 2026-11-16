# Configuration

[Start here](../README.md) · [Usage](usage.md) · [Architecture](architecture.md) · [Development](development.md)

## Choose a configuration

| Config in `configs/experiments/` | Purpose | Default device |
| --- | --- | --- |
| [`mc.toml`](../configs/experiments/mc.toml) | Small workflow check: 10 iterations, 8 games per iteration | CPU |
| [`mc-stable.toml`](../configs/experiments/mc-stable.toml) | Recommended full run: 200 iterations, 128 games, learning rate 0.0001, uniform history, replay off | CUDA 0 |
| [`league.toml`](../configs/experiments/league.toml) | Round robin using the introductory `mc-v1`, greedy, and random agents | CPU |

Other configs retain explicit parameters for earlier experiments. In particular,
mixed/history comparison configs retain learning rate 0.0003 and override the
current 0.0001 default. See [research protocols](../research/README.md).

## Experiment file and paths

Training reads TOML with `algorithm = "mc"`, a required `id`, an optional `name`,
a `device` (default `"cpu"`), and an optional `[training]` table. Monte Carlo on
Simple rules is the only implemented training algorithm; PPO and DQN names fail
validation. Unknown fields and invalid hyperparameters are rejected.

`--id` and `--device` override the experiment file. IDs contain 1–128 ASCII
letters, digits, hyphens, or underscores and cannot overwrite existing models.
The resolved experiment is saved as `experiment.json`.

| Path or reference | Resolved relative to |
| --- | --- |
| `--config`, `--resume`, `--data-dir`, explicit CLI agent/report/replay paths | Working directory |
| Agent spec path within a league config | League config directory |
| Saved model ID | `<data-dir>/models/<id>/agent.json` |
| Checkpoint path within a saved `agent.json` | That spec's directory when loaded by the CLI |

Paths inside research evaluation manifests instead resolve from the working
directory; see [checkpoint evaluation](../research/evaluation.md).

The default data directory is `data/`. `--data-dir` is global and may appear
before or after the subcommand. Library callers resolve checkpoint paths themselves.

## Training defaults

These are [`TrainingConfig::default()`](../crates/sequence-training/src/mc/mod.rs)
values. Example configs override them;
the introductory and stable configs are not aliases for these defaults.

| `[training]` field | Default | Meaning |
| --- | --- | --- |
| `iterations` | 100 | Total optimization iterations |
| `games_per_iteration` | 128 | Fresh rollout games per iteration |
| `batch_size` | 256 | Gradient batch size |
| `inference_batch_games` | 16 | Concurrent games sharing inference |
| `learning_rate` | 0.0001 | Optimizer learning rate |
| `epsilon_start`, `epsilon_end` | 0.2, 0.05 | Learner exploration schedule |
| `epsilon_decay_iterations` | Omitted | Decay over the configured total iterations |
| `seed` | 42 | Training seed |
| `evaluation_interval` | 10 | Evaluate every N iterations, and always at the end |
| `evaluation_games` | 20 | Games per baseline or historical evaluation opponent; positive and even |
| `sampling` | `"natural"` | Optimization sample selection |
| `updates_per_iteration` | Omitted | Use the fresh sample count rather than a fixed batch budget |
| `replay` | Omitted | Retained-sample buffer disabled |

Counts and learning rate must be positive. Exploration probabilities must be
in `[0, 1]`. See [resume restrictions](usage.md#training-snapshots-and-evaluation)
before extending a run or changing settings.

## Opponents and optimization sampling

Run the supplied 40% self-play, 20% best, 15% previous, 15% random checkpoint,
10% greedy experiment:

```sh
sequence train --config configs/experiments/mc-mixed.toml
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
updates, use [the balanced-sampling config](../configs/experiments/mc-balanced.toml),
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

## Historical opponent sampling

`configs/experiments/mc-history-uniform.toml` and
`configs/experiments/mc-history-adaptive.toml` provide matched 80% self-play,
20% historical-opponent experiments. Compare both with `mc-mixed.toml`.
Use distinct model IDs and several training seeds for each condition.

```sh
sequence train --config configs/experiments/mc-history-adaptive.toml
```

The optional `[training.historical_sampling]` table controls the
`random_checkpoint` category only:

- `adaptive = false` defaults to uniform sampling, preserving existing behavior.
- `snapshot_interval = 1` defaults to every historical checkpoint. The experiment
  configs use 10; iteration 0 is eligible, and the current checkpoint is excluded.
- `learning_rate = 0.01` controls quality decreases after learner wins.
- `uniform_fraction = 0.0` optionally mixes uniform exploration into softmax.

The scheduler is adapted from OpenAI Five, Appendix N (see the
[research reference](../research/README.md#reference)). In this implementation,
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

## Retained samples

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
`replay_samples` equal `optimization_samples`. Reports also include `replay_size`. Capacity must be positive, and fraction must
be strictly between zero and one.
Complete replay-enabled snapshots include `replay.bin`; missing or malformed replay
state fails resumption rather than silently resetting the buffer. Normal resume
continues to require the same replay and other learning settings.

## Inference batching and timing

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

See [GPU measurements and benchmark instructions](../research/benchmarks.md) for hardware-specific tuning.

## League defaults

Tournament TOML accepts `agents` (at least two references), `games_per_pair`
(default 100), `seed` (42), `max_actions` (1000), `device` (`"cpu"`),
`initial_elo` (1000), and `elo_k` (32). Unknown fields are rejected.
`--device` overrides the config. See [tournament behavior](usage.md#evaluate-and-run-tournaments)
for seat scheduling, truncation, and run-local Elo.
