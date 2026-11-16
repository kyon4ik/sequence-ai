# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9,<4", "plotly>=6,<7"]
# ///
"""Export saved training metrics to offline HTML, PNG, SVG and CSV.

Run from the repository root:
    uv run tools/plot_training.py data/models/<id>/report.json

An interrupted run's checkpoints/iteration-NNNNNN/state.json also works.
--comparisons accepts a JSON list of {"label": ..., "report": ...} entries;
evaluation report paths in that list are relative to the working directory.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = ["#2563eb", "#ea580c", "#059669", "#7c3aed"]


def seconds(value):
    return value["secs"] + value["nanos"] / 1e9 if value is not None else None


def rolling(values, window=10):
    return [fmean(values[max(0, i - window + 1) : i + 1]) for i in range(len(values))]


def wilson(wins, total):
    """95% Wilson interval for win probability (draws count as non-wins)."""
    if not total:
        return None, None
    p, z = wins / total, 1.959963984540054
    scale = 1 + z * z / total
    center = (p + z * z / (2 * total)) / scale
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return center - half, center + half


def load_comparisons(path):
    result = []
    if not path:
        return result
    for item in json.loads(path.read_text()):
        source = Path(item["report"])
        report = json.loads(source.read_text())
        standing = report["standings"][0]
        n = standing["wins"] + standing["losses"] + standing["draws"]
        if not n:
            continue
        lower, upper = wilson(standing["wins"], n)
        wins = []
        seeds = []
        for game in report["games"]:
            seat = game["players"].index(0)
            replay_path = source.parent.parent.parent / "replays" / (game["replay"] + ".json")
            if replay_path.exists():
                replay = json.loads(replay_path.read_text())
                seeds.append((replay["header"]["config"]["game_seed"],
                              replay["header"]["config"]["agent_seeds"], seat))
            else:
                seeds.append(None)
            completed = game["termination"].get("Completed")
            winner = completed.get("Won") if isinstance(completed, dict) else None
            wins.append(winner == ["Red", "Blue"][seat])
        result.append({
            "label": item["label"], "agent": standing["agent"], "wins": standing["wins"],
            "losses": standing["losses"], "draws": standing["draws"],
            "truncated": standing["truncated"], "games": n,
            "win_rate": standing["wins"] / n, "lower": lower, "upper": upper,
            "seed": report["seed"], "device": report["device"], "source": str(source),
            "opponent_config": report["agents"][1]["config"],
            "outcomes": wins, "game_seeds": seeds,
        })
    return result


def paired_tests(comparisons):
    result = []
    for i, left in enumerate(comparisons):
        for right in comparisons[i + 1 :]:
            if (None in left["game_seeds"] or None in right["game_seeds"]
                    or left["game_seeds"] != right["game_seeds"]
                    or left["device"] != right["device"]
                    or left["opponent_config"] != right["opponent_config"]
                    or left["truncated"] or right["truncated"]):
                continue
            a = sum(x and not y for x, y in zip(left["outcomes"], right["outcomes"]))
            b = sum(y and not x for x, y in zip(left["outcomes"], right["outcomes"]))
            n = a + b
            # Exact paired test of differing binary win outcomes. A draw is a
            # non-win here, matching the chart's win-rate definition.
            p = min(1., 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2**n) if n else 1.
            result.append({"left": left["label"], "right": right["label"],
                           "left_only_wins": a, "right_only_wins": b,
                           "exact_p_value": p})
    return result


def line(name, x, y, color, dash=False):
    return {"name": name, "x": x, "y": y, "color": color, "dash": dash}


def build_panels(metrics, comparisons):
    x = [m["iteration"] for m in metrics]
    evaluated = [m for m in metrics if m.get("evaluation_score") is not None]
    timed_evaluations = [m for m in evaluated if seconds(m.get("evaluation_time"))]
    best = None
    best_line = []
    for m in metrics:
        score = m.get("evaluation_score")
        if score is not None:
            best = max(best, score) if best is not None else score
        best_line.append(best)
    panels = [
        {"title": "Periodic evaluation: mean score vs random and greedy", "ylabel": "Score", "range": [0, 1], "series": [
            line("Evaluation", [m["iteration"] for m in evaluated], [m["evaluation_score"] for m in evaluated], COLORS[0]),
            line("Best so far", x, best_line, COLORS[2], True)]},
        {"title": "Training error (not a playing-strength metric)", "ylabel": "MSE", "series": [
            line("Per iteration", x, [m["mse"] for m in metrics], "#94a3b8"),
            line("10-iteration mean", x, rolling([m["mse"] for m in metrics]), COLORS[1])]},
        {"title": "Exploration schedule", "ylabel": "Epsilon", "series": [
            line("Random-action probability", x, [m["epsilon"] for m in metrics], COLORS[3])]},
        {"title": "Training outcomes by seat", "ylabel": "Fraction of episodes", "range": [0, 1], "series": [
            line(name, x, [m["outcomes"][i] / m["episodes"] for m in metrics], color)
            for i, (name, color) in enumerate(zip(["Red wins", "Blue wins", "Draws"], [COLORS[1], COLORS[0], COLORS[2]]))]},
        {"title": "Iteration timing (excludes checkpoint writes)", "ylabel": "Seconds", "series": [
            line("Rollout + sample extraction", x, [seconds(m["rollout_time"]) for m in metrics], COLORS[0]),
            line("Weight updates", x, [seconds(m["optimization_time"]) for m in metrics], COLORS[1]),
            line("Periodic evaluation", [m["iteration"] for m in timed_evaluations], [seconds(m["evaluation_time"]) for m in timed_evaluations], COLORS[2])]},
        {"title": "Collected training data", "ylabel": "Afterstate samples", "series": [
            line("Fresh samples", x, [m["samples"] for m in metrics], COLORS[0]),
            line("Optimization samples", x, [m.get("optimization_samples") or None for m in metrics], COLORS[1])]},
        {"title": "Inference timing (available after the implementation change)", "ylabel": "Seconds per iteration", "series": [
            line("CPU candidate encoding", x, [seconds(m["rollout_inference"]["encoding_time"]) if m.get("rollout_inference", {}).get("decisions") else None for m in metrics], COLORS[1]),
            line("Pack + upload + predict + readback", x, [seconds(m["rollout_inference"]["prediction_time"]) if m.get("rollout_inference", {}).get("decisions") else None for m in metrics], COLORS[0])]},
    ]
    labels = [f'{c["label"]}\n{c["wins"]}/{c["games"]} wins' for c in comparisons]
    comparison_title = "Separate evaluation vs greedy (95% win-rate intervals)"
    if comparisons and len({(c["device"], c["seed"]) for c in comparisons}) == 1:
        comparison_title = f'Greedy evaluation: {comparisons[0]["device"]}, seed {comparisons[0]["seed"]} (95% intervals)'
    panels.append({"title": comparison_title, "ylabel": "Win rate", "range": [0, 1],
                   "bars": comparisons, "labels": labels, "series": []})
    if any(m.get("opponents") for m in metrics):
        sources = ["self_play", "best_checkpoint", "previous_checkpoint", "random_checkpoint", "greedy"]
        colors = COLORS + ["#b45309"]
        def grouped(metric, source):
            return [entry for entry in metric.get("opponents", []) if entry["source"] == source]

        panels.extend([
            {"title": "Evaluation against fixed opponents", "ylabel": "Score", "range": [0, 1], "series": [
                line(label, x, [next((e["score"] for e in m.get("evaluation_opponents", []) if e["opponent"] == opponent), None) for m in metrics], color)
                for opponent, label, color in [("random", "Random", COLORS[0]), ("greedy", "Greedy", COLORS[1]), ("iteration-000000", "Initial model", COLORS[2])]]},
            {"title": "Requested opponent mix (includes recorded fallbacks)", "ylabel": "Fraction of games", "range": [0, 1], "series": [
                line(source.replace("_", " "), x, [sum(e["games"] for e in grouped(m, source)) / m["episodes"] if m.get("opponents") else None for m in metrics], color)
                for source, color in zip(sources, colors)]},
            {"title": "Training score by category (exploration enabled)", "ylabel": "Learner score", "range": [0, 1], "series": [
                line(source.replace("_", " "), x, [
                    sum(e["outcomes"][0] + .5 * e["outcomes"][2] for e in grouped(m, source)) / sum(e["games"] for e in grouped(m, source))
                    if sum(e["games"] for e in grouped(m, source)) else None for m in metrics], color)
                for source, color in zip(sources, colors)]},
            {"title": "Training samples by requested opponent category", "ylabel": "Samples", "series": [
                line(source.replace("_", " "), x, [sum(e["samples"] for e in grouped(m, source)) if m.get("opponents") else None for m in metrics], color)
                for source, color in zip(sources, colors)]},
        ])
    if any(m.get("optimization_samples") for m in metrics):
        panels.extend([
            {"title": "Samples used for updates", "ylabel": "Optimization samples", "series": [
                line(source.replace("_", " "), x, [sum(e.get("optimization_samples", 0) for e in grouped(m, source)) if m.get("optimization_samples") else None for m in metrics], color)
                for source, color in zip(sources, colors)] + ([
                    line("Retained samples", x, [m.get("replay_samples", 0) for m in metrics], "#9333ea")
                ] if any(m.get("replay_samples", 0) for m in metrics) else [])},
            {"title": "Gradient update budget", "ylabel": "Updates", "series": [
                line("Updates per iteration", x, [m.get("updates") or None for m in metrics], COLORS[0])]},
        ])
    return panels


def export_csv(metrics, path):
    def flatten(value, prefix=""):
        if isinstance(value, dict):
            if set(value) == {"secs", "nanos"}:
                return {prefix + "_seconds": seconds(value)}
            return {key: item for name, child in value.items()
                    for key, item in flatten(child, f"{prefix}.{name}" if prefix else name).items()}
        if isinstance(value, list):
            return {key: item for i, child in enumerate(value)
                    for key, item in flatten(child, f"{prefix}.{i}").items()}
        return {prefix: value}

    rows = []
    for metric in metrics:
        m = dict(metric)
        for key in ["rollout_inference", "evaluation_inference"]:
            if not m.get(key, {}).get("decisions"):
                m.pop(key, None)
        if m.get("evaluation_score") is None or not seconds(m.get("evaluation_time")):
            m.pop("evaluation_time", None)
        rows.append(flatten(m))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render(panels, prefix, title, note, timing_start):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    rows = math.ceil(len(panels) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(17, rows * 4), constrained_layout=True)
    interactive = make_subplots(rows=rows, cols=2, subplot_titles=[p["title"] for p in panels], vertical_spacing=0.075, horizontal_spacing=0.09)
    for index, (panel, axis) in enumerate(zip(panels, axes.flat)):
        row, col = index // 2 + 1, index % 2 + 1
        axis.set_title(panel["title"], loc="left", fontsize=10, pad=12)
        axis.set_ylabel(panel["ylabel"])
        axis.grid(axis="y", color="#e2e8f0", linewidth=.7)
        axis.set_axisbelow(True)
        if "range" in panel:
            axis.set_ylim(panel["range"])
            interactive.update_yaxes(range=panel["range"], row=row, col=col)
        for series in panel["series"]:
            axis.plot(series["x"], [y if y is not None else math.nan for y in series["y"]],
                      label=series["name"], color=series["color"], linestyle="--" if series["dash"] else "-", linewidth=1.5)
            interactive.add_trace(go.Scatter(x=series["x"], y=series["y"], mode="lines+markers" if index == 0 else "lines",
                name=series["name"], legendgroup=str(index), line={"color": series["color"], "dash": "dash" if series["dash"] else "solid"},
                connectgaps=False, hovertemplate="Iteration %{x}<br>%{y:.4f}<extra>%{fullData.name}</extra>"), row=row, col=col)
        if panel["series"]:
            axis.set_xlabel("Iteration")
            axis.legend(loc="best", fontsize=8, frameon=False)
            interactive.update_xaxes(title_text="Iteration", row=row, col=col)
            if timing_start is not None:
                axis.axvline(timing_start, color="#64748b", linestyle=":", linewidth=1)
                interactive.add_vline(x=timing_start, line_dash="dot", line_color="#64748b", row=row, col=col)
        elif panel.get("bars"):
            comparisons = panel["bars"]
            rates = [c["win_rate"] for c in comparisons]
            lower = [c["win_rate"] - c["lower"] for c in comparisons]
            upper = [c["upper"] - c["win_rate"] for c in comparisons]
            axis.bar(panel["labels"], rates, yerr=[lower, upper], color=COLORS[:len(rates)], capsize=5, width=.55)
            axis.axhline(.5, color="#64748b", linestyle=":")
            for i, y in enumerate(rates):
                axis.text(i, comparisons[i]["upper"] + .025, f"{y:.1%}", ha="center")
            interactive.add_trace(go.Bar(x=[c["label"] for c in comparisons], y=rates, marker_color=COLORS[:len(rates)], showlegend=False,
                error_y={"type": "data", "symmetric": False, "array": upper, "arrayminus": lower},
                text=[f"{y:.1%}" for y in rates], textposition="outside",
                customdata=[[c["wins"], c["losses"], c["draws"], c["games"], c["seed"], c["device"]] for c in comparisons],
                hovertemplate="%{x}<br>Wins %{customdata[0]}, losses %{customdata[1]}, draws %{customdata[2]}<br>%{customdata[3]} games; seed %{customdata[4]}; %{customdata[5]}<extra></extra>"), row=row, col=col)
            interactive.add_hline(y=.5, line_dash="dot", line_color="#64748b", row=row, col=col)
        else:
            axis.text(.5, .5, "Supply --comparisons for evaluation reports", ha="center", transform=axis.transAxes)
        interactive.update_yaxes(title_text=panel["ylabel"], row=row, col=col)
    if timing_start is not None:
        note += f" · dotted line: detailed inference timing begins at iteration {int(timing_start + .5)}"
    fig.suptitle(title + "\n" + note, fontsize=11, ha="left", x=.03)
    fig.savefig(prefix.with_suffix(".png"), dpi=150)
    fig.savefig(prefix.with_suffix(".svg"))
    plt.close(fig)
    interactive.update_layout(title={"text": title}, height=rows * 425, width=None, template="plotly_white",
                              margin={"t": 100, "b": 50}, hovermode="closest", legend={"orientation": "h", "y": -.055})
    # Include Plotly itself so opening the artifact needs no server or internet.
    chart = interactive.to_html(full_html=False, include_plotlyjs=True, config={"responsive": True, "displaylogo": False})
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>body{{font:15px system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 20px;color:#172033}}p,li{{line-height:1.6}}a{{color:#2563eb}}.note{{padding:14px;background:#f1f5f9;border-radius:8px}}</style>
<h1>{html.escape(title)}</h1><p>{html.escape(note)}</p><p>Hover for values, drag to zoom, double-click to reset, and click legend entries to toggle lines.</p>
<p class="note">Periodic evaluation averages scores against random and greedy; historical reports do not separate those opponents. Training MSE measures fit to changing rollout data. Opponent weights describe games, while self-play contributes samples from both seats. Missing historical opponents fall back to self-play; the CSV retains actual opponents and fallback counts. It is not held-out playing strength. Earlier runs lack the new inference timers; gaps mean unmeasured, not zero.</p>
<p><a href="{html.escape(prefix.with_suffix('.png').name)}">PNG</a> · <a href="{html.escape(prefix.with_suffix('.svg').name)}">SVG</a> · <a href="{html.escape(prefix.with_suffix('.csv').name)}">All recorded metrics (CSV)</a></p>
{chart}<h2>What is recorded</h2><ul>
<li>Per iteration: episodes, afterstate samples, MSE, epsilon, Red/Blue wins and draws, cumulative rewards and decision time by seat.</li>
<li>Strength: periodic evaluation score (win 1, draw 0.5, loss 0), averaged over equal numbers of random and greedy games, with fixed evaluation seeds and balanced seats.</li>
<li>Timing: rollout/sample extraction, optimization, evaluation and iteration elapsed time. Iteration time excludes checkpoint writes.</li>
<li>Detailed inference: decisions, forward-pass batches, candidate afterstates, encoding time, and combined packing/upload/prediction/readback wall time. This does not isolate GPU kernel time.</li>
<li>New training reports: fresh and optimization sample counts, actual gradient updates, requested opponent categories, actual checkpoint IDs, fallback counts, games, samples, learner seats, wins/losses/draws, and separate baseline/historical evaluation scores. Training category scores use exploration and changing opponents; they are not held-out strength.</li>
<li>Separate evaluation/tournaments: wins/losses/draws/truncations, rewards, per-run Elo, decisions, mean/max move time, actions/turns, and complete game replays.</li></ul>
<p>Not recorded in older reports: per-opponent evaluation scores. Not recorded: held-out loss, policy entropy, gradient norms, or GPU-only compute/transfer times. Confidence bars show 95% Wilson intervals for win rate; draws count as non-wins.</p></html>'''
    prefix.with_suffix(".html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--comparisons", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/plots/training"), help="Output filename prefix")
    args = parser.parse_args()
    document = json.loads(args.report.read_text())
    metrics = document.get("report", document)["iterations"]
    if not metrics:
        parser.error("the report has no completed iterations")
    comparisons = load_comparisons(args.comparisons)
    evaluated = [m for m in metrics if m.get("evaluation_score") is not None]
    best = max(evaluated, key=lambda m: m["evaluation_score"]) if evaluated else None
    measured = [m["iteration"] for m in metrics if m.get("rollout_inference", {}).get("decisions")]
    timing_start = min(measured) - .5 if measured and min(measured) > metrics[0]["iteration"] else None
    title = f"Training history: {args.report.parent.name}"
    note = f'{len(metrics)} iterations · {sum(m["episodes"] for m in metrics):,} episodes · {sum(m["samples"] for m in metrics):,} samples'
    if best:
        note += f' · best evaluation {best["evaluation_score"]:.3f} at iteration {best["iteration"]}'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_csv(metrics, args.output.with_suffix(".csv"))
    render(build_panels(metrics, comparisons), args.output, title, note, timing_start)
    summary = {"source": str(args.report), "best_iteration": best["iteration"] if best else None,
               "best_evaluation_score": best["evaluation_score"] if best else None,
               "last_iteration": metrics[-1]["iteration"], "last_evaluation_score": evaluated[-1]["evaluation_score"] if evaluated else None,
               "comparisons": [{k: v for k, v in c.items() if k not in ["outcomes", "game_seeds"]} for c in comparisons],
               "paired_win_tests": paired_tests(comparisons)}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    for suffix in ["html", "png", "svg", "csv", "json"]:
        print(args.output.with_suffix("." + suffix))


if __name__ == "__main__":
    main()
