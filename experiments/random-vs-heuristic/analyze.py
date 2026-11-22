"""Summarize the saved baseline match, resampling both seats together."""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, quantiles

directory = Path(__file__).parent
config = json.loads((directory / "manifest.json").read_text())
with (directory / "games.csv").open() as source:
    games = list(csv.DictReader(source))

pairs = defaultdict(list)
for game in games:
    pairs[int(game["shuffle_seed"])].append(game)

scores = {"heuristic": 1, "draw": 0.5, "random": 0}
pair_scores = []
for pair in pairs.values():
    if sorted(game["first_policy"] for game in pair) != ["heuristic", "random"]:
        raise ValueError("incomplete seat swap")
    if any(game["outcome"] == "truncated" for game in pair):
        raise ValueError("unresolved game; report truncation bounds before scoring")
    pair_scores.append(mean(scores[game["outcome"]] for game in pair))

rng = random.Random(config["bootstrap_seed"])
samples = [mean(rng.choices(pair_scores, k=len(pair_scores)))
           for _ in range(config["bootstrap_samples"])]
percentiles = quantiles(samples, n=1000, method="inclusive")
print(f"Heuristic score {mean(pair_scores):.2%}, paired 95% interval "
      f"{percentiles[24]:.2%} to {percentiles[974]:.2%}")
for first in ["heuristic", "random"]:
    group = [g for g in games if g["first_policy"] == first]
    counts = {outcome: sum(g["outcome"] == outcome for g in group)
              for outcome in ["heuristic", "random", "draw", "truncated"]}
    print(f"{first} first ({len(group)} games): {counts}")
print("Median actions:", median(int(g["actions"]) for g in games))
