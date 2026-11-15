# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9,<4"]
# ///
"""Export a standalone figure from tools/opponent_study.py's summary.json."""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = ["#2563eb", "#ea580c", "#059669", "#7c3aed", "#b45309"]


def paired_panel(axis, rows, seeds, metric, title):
    for index, seed in enumerate(seeds):
        selected = {r["arm"]: r for r in rows if r["training_seed"] == seed and r["checkpoint"] == "final"}
        values = [selected[a][metric] for a in ["natural", "opponent_mix"]]
        axis.plot([0, 1], values, "o-", color=COLORS[index % len(COLORS)], label=f"Seed {seed}", linewidth=1.5)
    axis.set_xticks([0, 1], ["Natural sampling", "Opponent mix sampling"])
    axis.set_title(title, loc="left")
    axis.set_ylabel("Score (win 1, draw 0.5)")
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    rows = summary["runs"]
    seeds = summary["plan"]["training_seeds"]
    prefix = args.output or args.summary.parent / "figures/opponent-study"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    original = summary.get("original_checkpoints")
    if original:
        labels = ["Random", "Greedy", "Mean"]
        for i, which in enumerate(["best", "final"]):
            scores = [original[which][name]["score"] for name in ["random", "greedy"]]
            scores.append(statistics.fmean(scores))
            bars = axes[0, 0].bar([x + (i - .5) * .34 for x in range(3)], scores, width=.34,
                                   color=COLORS[i], label=which.capitalize())
            axes[0, 0].bar_label(bars, labels=[f"{v:.1%}" for v in scores], padding=3)
        axes[0, 0].set_xticks(range(3), labels)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_ylabel("Held-out score")
        axes[0, 0].legend(frameon=False)
    else:
        axes[0, 0].text(.5, .5, "Original checkpoint comparison unavailable", ha="center")
    axes[0, 0].set_title("Existing mixed run: best versus final", loc="left")
    paired_panel(axes[0, 1], rows, seeds, "mean_score", "Final checkpoints: mean random + greedy score")
    paired_panel(axes[1, 0], rows, seeds, "greedy_score", "Final checkpoints: score against greedy")
    sources = list(summary["plan"]["opponents"])
    bottom = [0, 0]
    for source, color in zip(sources, COLORS):
        values = [statistics.fmean(r["optimization_source_fractions"][source] for r in rows
                                  if r["arm"] == arm and r["checkpoint"] == "final")
                  for arm in ["natural", "opponent_mix"]]
        axes[1, 1].bar([0, 1], values, bottom=bottom, color=color, label=source.replace("_", " "), width=.5)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[1, 1].set_xticks([0, 1], ["Natural", "Opponent mix"])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("Fraction of optimization samples")
    axes[1, 1].set_title("Actual optimization mix (mean across seeds)", loc="left")
    axes[1, 1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    plan = summary["plan"]
    fig.suptitle(f"Opponent sampling: {len(seeds)} training seeds, {plan['iterations']} iterations, "
                 f"{plan['updates_per_iteration']} updates per iteration\n"
                 f"{plan['evaluation_games_per_baseline']:,} held-out games per baseline and checkpoint; paired seats", fontsize=13)
    for suffix in ["png", "svg", "pdf"]:
        output = prefix.with_suffix("." + suffix)
        fig.savefig(output, dpi=160)
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
