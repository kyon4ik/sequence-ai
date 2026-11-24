# Next Sequence experiments, in priority order

Run the shell blocks from the repository root. Build the executable once for CPU inference and CUDA training.

```sh
cargo build --release --locked -p sequence-research --features cuda
```

## 1. Measure the cost of missing an immediate win

On the original holdout, the final Double DQN checkpoint beat Heuristic in 66 of 512 games and took 66 of 225 immediate-win opportunities. Several opportunities can occur in one game. Replay the same 256 deals in both seats with the same checkpoint, first playing greedily and then taking any available win. The difference in Heuristic win rate measures the gain from this rule.

```sh
baseline=results/double-dqn-mixed-full-s42
checkpoint="$baseline/checkpoints/episode-051200.bpk"
panel="$baseline/transfer-holdout/config.json"
out=results/immediate-win-ablation-s42
mkdir "$out" && (
  set -e
  target/release/sequence-research evaluate-dqn "$checkpoint" "$panel" --device cpu --tactics \
    --games "$out/greedy.csv" > "$out/greedy.json"
  target/release/sequence-research evaluate-dqn "$checkpoint" "$panel" --device cpu --take-immediate-wins \
    --games "$out/take-win.csv" > "$out/take-win.json"
)
```

## 2. Compare DQN and Double DQN at the full budget

Change only the target algorithm. The archived comparison used short runs. Compare seeds 42, 43 and 44 to see whether the choice of target consistently affects performance after exploration decays. Reuse the completed seed-42 Double DQN baseline.

```sh
for seed in 42 43 44; do
  target/release/sequence-research train-dqn experiments/dqn-mixed-full.json \
    --seed "$seed" --output "results/dqn-mixed-full-s$seed" || break
done
for seed in 43 44; do
  target/release/sequence-research train-dqn experiments/double-dqn-mixed-full.json \
    --seed "$seed" --output "results/double-dqn-mixed-full-s$seed" || break
done
```

## 3. Increase Heuristic opponents from 20% to 50%

Use Double DQN with one Heuristic game per two episodes and self-play in the remaining games. Compare with the 20% Double DQN runs from experiment 2, seed by seed. Look for improvement against Heuristic and check whether it carries over against Random. A further head-to-head test against a frozen baseline would require adding checkpoint opponents to the evaluator.

```sh
for seed in 42 43 44; do
  target/release/sequence-research train-dqn experiments/double-dqn-half-heuristic-full.json \
    --seed "$seed" --output "results/double-dqn-half-heuristic-full-s$seed" || break
done
```

### Comparing the training runs

Use each run's `evaluation.csv` to inspect progress near 500,000 and 1 million learner decisions. For the main comparison, aim for the reference budget of 1,675,717 decisions. The trainer stops after 51,200 episodes, so totals can differ by opponent and seed. Use the smallest final decision count among the arms being compared if it falls below the reference. Evaluate the saved checkpoint nearest that common budget in each run and record its actual decision count.

After the runs in experiments 2 and 3 finish, this block evaluates all nine on the same fresh 256 deals (shuffle seeds 12,000,000–12,000,255, policy seed 2042, both seats). Each run gets its own holdout results. To compare experiment 2 alone, omit `double-dqn-half-heuristic-full` from the first loop.

```sh
(
  set -e
  runs=()
  budget=1675717
  for arm in dqn-mixed-full double-dqn-mixed-full double-dqn-half-heuristic-full; do
    for seed in 42 43 44; do
      run="results/${arm}-s${seed}"
      test -s "$run/checkpoints/episode-051200.bpk"
      runs+=("$run")
      decisions=$(awk -F, '$1 == 51200 {print $2; exit}' "$run/evaluation.csv")
      test -n "$decisions"
      if [ "$decisions" -lt "$budget" ]; then budget=$decisions; fi
    done
  done
  for run in "${runs[@]}"; do
    checkpoint=$(awk -F, -v target="$budget" '
      NR > 1 {
        gap = $2 - target
        if (gap < 0) gap = -gap
        if (NR == 2 || gap < best) {best = gap; episode = $1}
      }
      END {printf "episode-%06d", episode}
    ' "$run/evaluation.csv")
    out="$run/next-holdout-$checkpoint"
    mkdir "$out"
    target/release/sequence-research evaluate-dqn "$run/checkpoints/$checkpoint.bpk" \
      experiments/next-holdout.json --tactics --games "$out/games.csv" > "$out/summary.json"
  done
)
```

Use Bash or Zsh for this block. Report Heuristic win rate first and Random win rate alongside it, separately for each training seed. For a paired 95% interval, resample shuffle seeds with both seats and both policies kept together. Report any cutoffs.

## 4. Teach the network to rank immediate wins

Run this before changing the opponent mix. Taking immediate wins improved the three-seed holdout win rate against Heuristic by 7.7 percentage points. Use the full-budget Double DQN config with a 0.1 auxiliary loss weight and a 0.1 Q-value margin between the best immediate win and the best nonwinning legal move. Compare with the 20% Heuristic Double DQN baseline at matched decision counts. Measure immediate-win ranking on shared tactical positions and unaided win rates in full games.

```sh
for seed in 42 43 44; do
  out="results/double-dqn-immediate-win-full-s$seed"
  target/release/sequence-research train-dqn experiments/double-dqn-immediate-win-full.json \
    --seed "$seed" --output "$out" > "$out.log" 2>&1 &
done
wait
```

## 5. Test whether public move history helps

Add the opponent's last few publicly observed card plays and target squares to the observation. The current encoder records discard counts and loses their order. Train two otherwise identical networks with the same input dimensions, one receiving real history and one receiving zeros. Compare their held-out win rates at matched decision counts. This requires an observation encoding and model input for move history.
