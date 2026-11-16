# Research results and offline analysis

[Start here](../README.md) · [Training configuration](../docs/configuration.md) · [Checkpoint evaluation](evaluation.md) · [GPU benchmarks](benchmarks.md)

`sequence` is the execution interface. Python tools read saved artifacts and
export tables, figures, dashboards, or analysis decisions. They do not start
training, publish model specs, or update champion pointers.

## Run experiments

Train each condition with an explicit config, ID, and seed:

```sh
sequence train --config configs/experiments/mc-stable.toml --id stable-s42 --seed 42 --checkpoint-interval 50 --keep-checkpoints 0
sequence train --config configs/experiments/mc-stable.toml --id stable-s43 --seed 43 --checkpoint-interval 50 --keep-checkpoints 0
```

Use [resume](../docs/usage.md#training-snapshots-and-evaluation) to continue a
complete checkpoint, [panel evaluation](evaluation.md) for held-out comparisons,
and [champion promotion](../docs/usage.md#champion-promotion) to update a registry.
There is no study scheduler or automatic promotion after analysis.

Choose conditions, checkpoint-selection rules, interaction/update budgets, and
held-out evaluation seeds before inspecting results. Keep seats and game/policy
seeds paired across comparisons. Repeat training across independent seeds;
bootstrap intervals over game seeds alone do not measure training-seed uncertainty.

## Analyze saved studies

These tools retain the schemas of completed studies under `data/studies/`.
Their plans, model ledgers, evaluation files, and source checkpoints must already
exist. They regenerate analysis outputs from those artifacts; they do not recreate
historical training runs.

| Tool | Input and output |
| --- | --- |
| [`plot_training.py`](../tools/plot_training.py) | Training report or checkpoint state → interactive dashboard, figures, CSV, summary JSON |
| [`analyze_opponent_study.py`](../tools/analyze_opponent_study.py) | Opponent-sampling study → paired comparisons and Markdown/CSV/JSON summaries |
| [`plot_opponent_study.py`](../tools/plot_opponent_study.py) | Opponent summary → standalone figures |
| [`analyze_history_diagnosis.py`](../tools/analyze_history_diagnosis.py) | Fixed-panel evaluations and diagnostics → scores, calibration summaries, figures |
| [`analyze_regression_ablation.py`](../tools/analyze_regression_ablation.py) | Archived continuation evaluations → intervention comparisons and figures |
| [`analyze_learning_rate_confirmation.py`](../tools/analyze_learning_rate_confirmation.py) | Archived full-budget study → aggregate confirmation decision, tables, curves |

```sh
python3 tools/analyze_opponent_study.py data/studies/opponent-sampling-20261113
uv run tools/plot_opponent_study.py data/studies/opponent-sampling-20261113/summary.json
uv run tools/analyze_history_diagnosis.py data/studies/history-diagnosis
uv run tools/analyze_regression_ablation.py data/studies/regression-ablation
uv run tools/analyze_learning_rate_confirmation.py data/studies/learning-rate-confirmation --phase lr
```

The confirmation analyzer writes `lr-decision.json` or `replay-decision.json`.
Use it explicitly as `sequence promote --confirmation PATH` when the aggregate
study rule is a prerequisite for promotion. Analysis can read an existing
`promotion-lr.json` or `promotion-replay.json` to include the gate result in its
report. It never invokes promotion.

## Completed study protocols

The opponent-sampling comparison used three training seeds, equal fixed update
budgets, and the same opponent mix, comparing natural versus opponent-balanced
gradient sampling at 100 iterations. Both best and final checkpoints were
compared on identical held-out seeds. Three training seeds and the shorter horizon
limit conclusions about population strength and 200-iteration behavior.

The regression screening branched from two seed-42 iteration-50 checkpoints,
restoring weights, Adam, and historical opponents. Three continuation seeds tested
control, lower learning rate, retained samples, historical gradient weighting,
and fixed exploration over 50 more iterations. Each iteration used 64 games and
27 batches of 128 samples. Final checkpoints played 400 games per opponent on
paired seats. The primary score equally averaged greedy and the two frozen
checkpoint-50 policies. These were conditional continuations from two fixed
policies, rather than independent training runs. The specialized continuation
executor is no longer part of the supported interface.

Full-budget confirmation compared learning rates 0.0003 and 0.0001 across seeds
42–44, using 200 iterations and 128 games per iteration. Checkpoints 50/100/150/200
shared a curve-evaluation schedule. A separate release schedule added a frozen
mixed-training opponent, with 1,000 games per matchup. Final checkpoint 200 and
promotion candidate seed 42 were preselected. Conditional retained-sample runs
used a further release schedule. The two historical promotion gates used family
alpha 0.025 each; historical Python bootstrap results are retained as recorded.
New Rust gates record their RNG and protocol in each decision.

## Completed confirmation results

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
