# Checkpoint evaluation and diagnostics

[Research protocols](README.md) · [Everyday evaluation](../docs/usage.md#evaluate-and-run-tournaments)

For controlled panels without full replay storage, use `sequence panel`. Run from the repository root:

```sh
sequence panel --config panel.json --output data/panel-evaluation
```

Create `panel.json` with checkpoints that already exist locally:

```json
{
  "device": "cuda:0",
  "games": 1000,
  "inference_batch_games": 128,
  "seeds": [2026111401],
  "checkpoints": [
    {"id": "candidate", "path": "data/models/mc-stable-v1/model.bpk"}
  ]
}
```

Each seed is a base for a shared game schedule; games use both seats. Results
contain per-game scores and summaries and are saved as `<id>-<seed>.json`.
Use `"device": "cpu"` or pass `--device cpu` for CPU evaluation. Existing output
files, duplicate checkpoint IDs, duplicate seeds, and empty panels are rejected.
Checkpoint paths resolve from the working directory and are saved as absolute
paths. Reports bind candidate and neural-opponent files with SHA-256 hashes,
checked again before publication. These reports are inputs to
[`sequence promote`](../docs/usage.md#champion-promotion).
The Rust API also exposes `sequence_training::mc::evaluate_seeded_baselines`
and `evaluate_seeded_panel`.

## Custom opponents and diagnostics

`sequence panel` accepts an optional `opponents` list
of `[name, AgentConfig]` pairs. Defaults remain random and greedy. Frozen neural
opponents use batched inference on the selected device. All candidates share the same
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
