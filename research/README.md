# Research protocols and results

[Start here](../README.md) · [Training configuration](../docs/configuration.md) · [Checkpoint evaluation](evaluation.md) · [GPU benchmarks](benchmarks.md)

These workflows describe controlled studies, including their selection rules and
limitations. For everyday training, start with `mc.toml` or `mc-stable.toml` in
[the usage guide](../docs/usage.md).

Run all commands from the repository root. Studies depend on local artifacts
under ignored `data/`; a fresh checkout does not include their input models or
saved results. Keep the prescribed seeds, input checkpoints, and configuration
values when reproducing a study. GPU workflows require host GPU access; see
[development setup](../docs/development.md).

## Opponent sampling study

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

See [checkpoint evaluation](evaluation.md) for the batched evaluation API and
the `evaluate` example's JSON manifest format.

## Controlled regression interventions

Required inputs are the complete iteration-50 checkpoint directories under
`data/models/mc-history-uniform-v1/` and
`data/models/mc-history-adaptive-v1/`, including optimizer state and historical
opponent archives. These originate from the corresponding history configs.

`tools/regression_ablation.py` prepares and runs matched research continuations
from the seed-42 uniform/adaptive checkpoint-50 archives. Build the examples,
then run the driver from the repository root with host GPU access:

```sh
cargo build --release -p sequence-training --features burn --example ablation --example evaluate
python3 tools/regression_ablation.py run
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

## Full-budget learning-rate confirmation and champion selection

Required local inputs include `data/models/mc-history-uniform-v1/experiment.json`,
the uniform/adaptive iteration-50 checkpoints, and
`data/models/mc-mixed-v1/model.bpk`. The workflow reads the existing
`data/champions/sequence.json` registry when present; otherwise it uses
`data/models/mc-history-uniform-v1/agent.json` as incumbent. The source
[driver](../tools/learning_rate_confirmation.py) records the resolved plan and
input hashes.

Run the prespecified study with host GPU access:

```sh
cargo build --release -p sequence-training --features burn --example train_study --example evaluate
python3 tools/learning_rate_confirmation.py run
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

Retained-sample behavior is documented in [configuration](../docs/configuration.md#retained-samples).

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

## Reference

The adaptive historical-opponent scheduler is adapted from
[OpenAI Five, Appendix N](https://cdn.openai.com/dota-2.pdf). The implementation's
batched scheduling and probability updates are described in
[historical opponent sampling](../docs/configuration.md#historical-opponent-sampling).
