"""Run and summarize a paired, fixed-update opponent-sampling study.

python3 tools/opponent_study.py run --iterations 100 --seeds 42,43,44
python3 tools/opponent_study.py summarize

Uses the release sequence binary and sequence-training's evaluate example.
Evaluations use batched inference and retain per-game scores instead of replays.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = {"self_play": 40, "best_checkpoint": 20, "previous_checkpoint": 15,
           "random_checkpoint": 15, "greedy": 10}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(value, indent=2) + "\n")
    staging.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_payload_digest(path):
    """Burnpack v1 payload for this fixed network layout, excluding random IDs.

    burn-pack 0.22.1 uses a 10-byte header and a 256-byte-aligned data section.
    Whole-file hashes also include per-process parameter IDs in CBOR metadata.
    """
    data = path.read_bytes()
    magic, version, metadata_size = struct.unpack_from("<IHI", data)
    if magic != 0x4255524E or version != 1:
        raise ValueError("unsupported checkpoint format for parameter comparison")
    offset = ((10 + metadata_size + 255) // 256) * 256
    if offset >= len(data):
        raise ValueError("checkpoint has no tensor payload")
    return hashlib.sha256(data[offset:]).hexdigest()


def prepare(args):
    root = args.output.resolve()
    seeds = [int(s) for s in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("training seeds must be nonempty and distinct")
    if args.games <= 0 or args.games % 2 or args.iterations <= 0 or args.updates <= 0:
        raise ValueError("need positive iterations/updates and positive even evaluation games")
    cases = []
    for index, seed in enumerate(seeds):
        arms = ["natural", "opponent_mix"] if index % 2 == 0 else ["opponent_mix", "natural"]
        cases.extend({"id": f"{arm.replace('_', '-')}-s{seed}", "arm": arm, "seed": seed} for arm in arms)
    plan = {"device": args.device, "iterations": args.iterations, "training_seeds": seeds,
            "games_per_iteration": 128, "batch_size": 128, "updates_per_iteration": args.updates,
            "evaluation_games_per_baseline": args.games, "heldout_seed": 2026111402,
            "opponents": WEIGHTS, "cases": cases,
            "primary": "mean final-checkpoint score against random and greedy, averaged equally across training seeds",
            "secondary": ["final greedy score", "best-checkpoint mean score", "best greedy score", "best-to-final difference"],
            "selection": "earliest highest periodic mean random/greedy score; no held-out selection",
            "bootstrap": "paired resampling of game-seed pairs within a model; resampling training seeds for arm-mean differences"}
    path = root / "plan.json"
    if path.exists():
        if read_json(path) != plan:
            raise ValueError("study plan differs; use a new output directory")
    else:
        write_json(path, plan)
        write_json(root / "provenance.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "binaries": {str(p.relative_to(REPO)): digest(p) for p in [REPO / "target/release/sequence", REPO / "target/release/examples/evaluate"]},
        })
    (root / "configs").mkdir(exist_ok=True)
    for case in cases:
        text = f'''algorithm = "mc"
id = "{case['id']}"
name = "Sampling study: {case['arm']}, seed {case['seed']}"
device = "{args.device}"

[training]
iterations = {args.iterations}
games_per_iteration = 128
batch_size = 128
inference_batch_games = 128
learning_rate = 0.0003
epsilon_start = 0.3
epsilon_end = 0.05
seed = {case['seed']}
evaluation_interval = 10
evaluation_games = 100
sampling = "{case['arm']}"
updates_per_iteration = {args.updates}

[training.opponents]
'''
        text += "".join(f"{name} = {weight}\n" for name, weight in WEIGHTS.items())
        path = root / "configs" / (case["id"] + ".toml")
        if path.exists() and path.read_text() != text:
            raise ValueError(f"configuration differs: {path}")
        path.write_text(text)
    return root, plan


def existing_model(root, case):
    models = root / "training/models"
    candidates = [models / case["id"], *sorted(models.glob(case["id"] + "-resume-*"))]
    for path in candidates:
        if (path / "agent.json").exists():
            return path, None
    snapshots = [(path, snapshot) for path in candidates for snapshot in path.glob("checkpoints/iteration-*") if (snapshot / "state.json").is_file()]
    resume = max(snapshots, key=lambda item: int(item[1].name.split("-")[-1]))[1] if snapshots else None
    for retry in range(10000):
        name = case["id"] if retry == 0 else f"{case['id']}-resume-{retry}"
        if not (models / name).exists():
            return models / name, resume
    raise RuntimeError("too many interrupted model directories")


def validate_model(path, plan, case):
    rows = read_json(path / "report.json")["iterations"]
    cfg = read_json(path / "experiment.json")["training"]
    assert len(rows) == plan["iterations"]
    assert cfg["seed"] == case["seed"] and cfg["sampling"] == case["arm"]
    budget = plan["updates_per_iteration"] * plan["batch_size"]
    for row in rows:
        assert row["updates"] == plan["updates_per_iteration"]
        assert row["optimization_samples"] == budget
        assert sum(o["optimization_samples"] for o in row["opponents"]) == budget
        assert sum(o["games"] for o in row["opponents"]) == plan["games_per_iteration"]
        if case["arm"] == "opponent_mix":
            for source, weight in WEIGHTS.items():
                count = sum(o["optimization_samples"] for o in row["opponents"] if o["source"] == source)
                assert abs(count / budget - weight / 100) <= 1 / budget + 1e-12
    return rows


def run_study(root, plan):
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    resolved = {}
    for case in plan["cases"]:
        model, resume = existing_model(root, case)
        if not (model / "agent.json").exists():
            command = [str(REPO / "target/release/sequence"), "--data-dir", str(root / "training"),
                       "train", "--config", str(root / "configs" / (case["id"] + ".toml")), "--id", model.name,
                       "--device", plan["device"]]
            if resume:
                command.extend(["--resume", str(resume)])
            print(f"Training {case['id']}: {plan['iterations']} iterations, {plan['updates_per_iteration']} updates each", flush=True)
            write_json(root / "active.json", {"case": case, "model": str(model), "command": command,
                                               "started_utc": datetime.now(timezone.utc).isoformat()})
            with (logs / (model.name + ".log")).open("w") as log:
                subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = validate_model(model, plan, case)
        resolved[case["id"]] = str(model)
        write_json(root / "models.json", resolved)
        print(f"Completed {case['id']}: {sum(r['updates'] for r in rows)} updates, {sum(r['samples'] for r in rows)} fresh samples", flush=True)
    # Both arms must use identical preassigned episode seeds, even as their
    # policies produce different trajectories and different numbers of samples.
    for seed in plan["training_seeds"]:
        pair = [read_json(Path(resolved[f"{arm}-s{seed}"]) / "report.json")["iterations"] for arm in ["natural", "opponent-mix"]]
        assert [r["rollout_seed"] for r in pair[0]] == [r["rollout_seed"] for r in pair[1]]
        assert all(r["rollout_seed"] is not None for r in pair[0])
    checkpoints = []
    for case in plan["cases"]:
        model = Path(resolved[case["id"]])
        for which, filename in [("best", "best.bpk"), ("final", "model.bpk")]:
            identity = case["id"] + "-" + which
            if not (root / "evaluations" / f"{identity}-{plan['heldout_seed']}.json").exists():
                checkpoints.append({"id": identity, "path": str(model / filename)})
    if checkpoints:
        manifest = {"device": plan["device"], "games": plan["evaluation_games_per_baseline"], "inference_batch_games": 128,
                    "seeds": [plan["heldout_seed"]], "checkpoints": checkpoints}
        path = root / "study-evaluation.json"
        write_json(path, manifest)
        write_json(root / "study-checkpoint-hashes.json", {c["id"]: digest(Path(c["path"])) for c in checkpoints})
        print("Evaluating all best and final checkpoints on the held-out schedule", flush=True)
        with (logs / "evaluation.log").open("a") as log:
            subprocess.run([str(REPO / "target/release/examples/evaluate"), str(path), str(root / "evaluations")],
                           cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    write_json(root / "active.json", {"phase": "complete", "completed_utc": datetime.now(timezone.utc).isoformat()})


def baselines(report):
    return {b["summary"]["opponent"]: b for b in report["baselines"]}


def paired_clusters(left, right, names):
    assert left["device"] == right["device"] and left["config"] == right["config"]
    a, b = baselines(left), baselines(right)
    differences = []
    for name in names:
        assert a[name]["baseline"] == b[name]["baseline"]
        ga, gb = a[name]["games"], b[name]["games"]
        assert [(g["config"], g["learner_seat"]) for g in ga] == [(g["config"], g["learner_seat"]) for g in gb]
        assert len(ga) % 2 == 0
        values = []
        for i in range(0, len(ga), 2):
            assert ga[i]["config"]["game_seed"] == ga[i + 1]["config"]["game_seed"]
            values.append(sum(ga[j]["score"] - gb[j]["score"] for j in [i, i + 1]) / 2)
        differences.append(values)
    return [statistics.fmean(values) for values in zip(*differences)]


def bootstrap(values, seed=177, repetitions=5000):
    rng = random.Random(seed)
    draws = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(repetitions))
    return [draws[int(.025 * repetitions)], draws[int(.975 * repetitions)]]


def summarize(root, plan):
    resolved = read_json(root / "models.json")
    initial_weights = {}
    for seed in plan["training_seeds"]:
        weights = {}
        for arm in ["natural", "opponent-mix"]:
            model = Path(resolved[f"{arm}-s{seed}"])
            snapshots = sorted(model.glob("checkpoints/iteration-*/opponents/iteration-000000.bpk"))
            weights[arm] = parameter_payload_digest(snapshots[0])
        if weights["natural"] != weights["opponent-mix"]:
            raise ValueError(f"initial network parameters differ for seed {seed}")
        initial_weights[str(seed)] = weights
    write_json(root / "initial-weights.json", initial_weights)
    reports = {}
    rows = []
    for case in plan["cases"]:
        model = Path(resolved[case["id"]])
        training = validate_model(model, plan, case)
        evaluated = [m for m in training if m["evaluation_score"] is not None]
        best = max(evaluated, key=lambda m: m["evaluation_score"])
        for which in ["best", "final"]:
            identity = case["id"] + "-" + which
            report = read_json(root / "evaluations" / f"{identity}-{plan['heldout_seed']}.json")
            reports[(case["arm"], case["seed"], which)] = report
            scores = {name: b["summary"]["score"] for name, b in baselines(report).items()}
            counts = {name: sum(o["optimization_samples"] for m in training for o in m["opponents"] if o["source"] == name) for name in WEIGHTS}
            total = sum(counts.values())
            rows.append({"arm": case["arm"], "training_seed": case["seed"], "checkpoint": which,
                         "selected_iteration": best["iteration"] if which == "best" else plan["iterations"],
                         "random_score": scores["random"], "greedy_score": scores["greedy"], "mean_score": statistics.fmean(scores.values()),
                         "updates": sum(m["updates"] for m in training), "fresh_samples": sum(m["samples"] for m in training),
                         "optimization_samples": total, "optimization_source_fractions": {k: v / total for k, v in counts.items()}})
    contrasts = []
    for which in ["best", "final"]:
        for names in [["greedy"], ["random", "greedy"]]:
            deltas = []
            per_seed = []
            for seed in plan["training_seeds"]:
                clusters = paired_clusters(reports[("opponent_mix", seed, which)], reports[("natural", seed, which)], names)
                delta = statistics.fmean(clusters)
                deltas.append(delta)
                per_seed.append({"seed": seed, "difference": delta, "paired_game_seed_bootstrap_95": bootstrap(clusters)})
            contrasts.append({"checkpoint": which, "baselines": names, "balanced_minus_natural": statistics.fmean(deltas),
                              "training_seed_bootstrap_95": bootstrap(deltas), "per_training_seed": per_seed})
    original = None
    regressions = []
    for names in [["greedy"], ["random", "greedy"]]:
        drops = {arm: [] for arm in ["natural", "opponent_mix"]}
        per_seed = []
        for seed in plan["training_seeds"]:
            values = {}
            for arm in drops:
                clusters = paired_clusters(reports[(arm, seed, "best")], reports[(arm, seed, "final")], names)
                values[arm] = statistics.fmean(clusters)
                drops[arm].append(values[arm])
            per_seed.append({"seed": seed, "selected_best_minus_final": values})
        differences = [b - n for b, n in zip(drops["opponent_mix"], drops["natural"])]
        regressions.append({"baselines": names, "selected_best_minus_final": {arm: statistics.fmean(values) for arm, values in drops.items()},
                            "balanced_minus_natural_drop": statistics.fmean(differences),
                            "training_seed_bootstrap_95": bootstrap(differences), "per_training_seed": per_seed})
    original_paths = [root / "evaluations" / f"original-{which}-2026111401.json" for which in ["best", "final"]]
    if all(p.exists() for p in original_paths):
        a, b = map(read_json, original_paths)
        original = {"best": {k: v["summary"] for k, v in baselines(a).items()},
                    "final": {k: v["summary"] for k, v in baselines(b).items()}, "paired_comparisons": []}
        for names in [["greedy"], ["random", "greedy"]]:
            clusters = paired_clusters(a, b, names)
            original["paired_comparisons"].append({"baselines": names, "best_minus_final": statistics.fmean(clusters),
                                                    "paired_game_seed_bootstrap_95": bootstrap(clusters)})
    arm_means = {arm: {which: {metric: statistics.fmean(r[metric] for r in rows if r["arm"] == arm and r["checkpoint"] == which)
                              for metric in ["random_score", "greedy_score", "mean_score"]}
                       for which in ["best", "final"]} for arm in ["natural", "opponent_mix"]}
    provenance = read_json(root / "provenance.json")
    completed = read_json(root / "active.json")["completed_utc"]
    wall_minutes = (datetime.fromisoformat(completed) - datetime.fromisoformat(provenance["created_utc"])).total_seconds() / 60
    summary = {"plan": plan, "study_wall_minutes": wall_minutes, "initial_parameter_hashes": initial_weights, "original_checkpoints": original,
               "runs": rows, "arm_means": arm_means, "contrasts": contrasts,
               "regressions": regressions,
               "limits": [f"{len(plan['training_seeds'])} training seeds provide an exploratory comparison, not a precise population estimate.",
                          f"{plan['iterations']} iterations do not establish behavior at the original 200-iteration horizon.",
                          "Both arms use the same opponent pool; this isolates optimization sampling rather than pool versus pure self-play.",
                          "Both fixed-budget arms sample with replacement, unlike the original one-pass training regime.",
                          "Resampling uses repeated fresh-rollout samples; optimization samples are not additional game experience."]}
    write_json(root / "summary.json", summary)
    fields = [k for k in rows[0] if k != "optimization_source_fractions"]
    with (root / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
    lines = ["# Opponent sampling study", "", f"Device: {plan['device']}. {len(plan['training_seeds'])} training seeds; {plan['iterations']} iterations × {plan['updates_per_iteration']} updates × 128 samples per arm. Same 128-game opponent mix and preassigned episode seeds. Each checkpoint evaluated on {plan['evaluation_games_per_baseline']} games per baseline, with balanced seats.", ""]
    primary = next(c for c in contrasts if c["checkpoint"] == "final" and c["baselines"] == ["random", "greedy"])
    lo, hi = primary["training_seed_bootstrap_95"]
    interval_note = "includes zero" if lo <= 0 <= hi else "excludes zero; the small training-seed count still limits generalization"
    lines.extend([f"Final mean random/greedy score across training seeds: natural **{arm_means['natural']['final']['mean_score']:.2%}**, balanced **{arm_means['opponent_mix']['final']['mean_score']:.2%}**. Balanced minus natural: **{100 * primary['balanced_minus_natural']:+.2f} percentage points**, with a training-seed bootstrap interval of [{100 * lo:+.2f}, {100 * hi:+.2f}]. This exploratory interval {interval_note}.", ""])
    if original:
        lines.extend(["## Existing best versus final checkpoint", "", "| Checkpoint | Random score | Greedy score | Mean score |", "|---|---:|---:|---:|"])
        for which in ["best", "final"]:
            scores = [original[which][k]["score"] for k in ["random", "greedy"]]
            lines.append(f"| {which} | {scores[0]:.2%} | {scores[1]:.2%} | {statistics.fmean(scores):.2%} |")
        lines.append("")
    lines.extend(["## Controlled comparison", "", "| Seed | Sampling | Checkpoint | Iteration | Greedy score | Mean score |", "|---|---|---|---:|---:|---:|"])
    for row in sorted(rows, key=lambda row: (row["training_seed"], row["arm"], row["checkpoint"])):
        lines.append(f"| {row['training_seed']} | {row['arm']} | {row['checkpoint']} | {row['selected_iteration']} | {row['greedy_score']:.2%} | {row['mean_score']:.2%} |")
    lines.extend(["", "## Means across training seeds", "", "| Sampling | Checkpoint | Random score | Greedy score | Mean score |", "|---|---|---:|---:|---:|"])
    for arm, checkpoints in arm_means.items():
        for which, values in checkpoints.items():
            lines.append(f"| {arm} | {which} | {values['random_score']:.2%} | {values['greedy_score']:.2%} | {values['mean_score']:.2%} |")
    lines.extend(["", "## Balanced minus natural", "", "Differences and intervals below are in percentage points.", "", "| Checkpoint | Metric | Mean difference | Bootstrap interval across training seeds |", "|---|---|---:|---:|"])
    for contrast in contrasts:
        lo, hi = contrast["training_seed_bootstrap_95"]
        lines.append(f"| {contrast['checkpoint']} | {' + '.join(contrast['baselines'])} | {100 * contrast['balanced_minus_natural']:+.2f} | [{100 * lo:+.2f}, {100 * hi:+.2f}] |")
    lines.extend(["", "## Held-out change from selected best to final", "", "Values are percentage points; positive values indicate a lower final score. The best checkpoint was selected using periodic evaluation, independently of these held-out results.", "",
                  "| Metric | Natural drop | Balanced drop | Balanced minus natural drop |", "|---|---:|---:|---:|"])
    for regression in regressions:
        drops = regression["selected_best_minus_final"]
        lines.append(f"| {' + '.join(regression['baselines'])} | {100 * drops['natural']:+.2f} | {100 * drops['opponent_mix']:+.2f} | {100 * regression['balanced_minus_natural_drop']:+.2f} |")
    lines.extend(["", "## Limits", "", *[f"- {limit}" for limit in summary["limits"]], "",
                  f"Study wall time: {wall_minutes:.1f} minutes, including training, checkpoint writes and held-out evaluation. CUDA was used for all training and inference.", "",
                  "Per-game scores and exact episode seeds are in `evaluations/`; configurations, training logs, checkpoints and reports are retained under this study directory."])
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "summarize"])
    parser.add_argument("--output", type=Path, default=REPO / "data/studies/opponent-sampling-20261113")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "run":
        root, plan = prepare(args)
        run_study(root, plan)
    else:
        root = args.output.resolve()
        plan = read_json(root / "plan.json")
    summarize(root, plan)


if __name__ == "__main__":
    main()
