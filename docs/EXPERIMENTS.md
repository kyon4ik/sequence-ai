# Experiment catalog

Start with the [installation and run guide](../README.md). The experiment configs record the training parameters, seeds, and evaluation setup.

Win rates in the main comparisons use paired deals with both seats played. The full-budget comparisons use checkpoints near a common learner-decision count. Percentages from different rows may use different checkpoints or evaluation panels.

## Completed experiments

| Experiment | Config or procedure | Result |
| --- | --- | --- |
| **Heuristic vs. Random** | [Match config](../experiments/random-vs-heuristic.json) | Heuristic won 510 of 512 games (99.6%) and won 255 of 256 in each seat. |
| **Double DQN with 20% Heuristic opponents** | [Training config](../experiments/double-dqn-mixed-full.json) · [Shared holdout](../experiments/next-holdout.json) | Seeds 42–44 reached 1.66–1.68 million learner decisions in 51,200 episodes each. Their final checkpoints averaged 15.3% wins against Heuristic and 91.1% against Random on a shared 256-deal holdout. |
| **Immediate-win rule at evaluation** | [Training config](../experiments/double-dqn-mixed-full.json) · [Shared holdout](../experiments/next-holdout.json) | Applying the rule to the original seed-42 checkpoint recovered 43 wins against Heuristic in 512 games (12.9% → 21.3%). On the shared three-seed holdout, the rate rose from 15.3% to 23.0%, a gain of 7.7 percentage points (paired 95% CI +6.4 to +9.0). |
| **Immediate-win supervision** | [Training config](../experiments/double-dqn-immediate-win-full.json) · [Shared holdout](../experiments/next-holdout.json) | With weight 0.1 and a 0.1 Q-value margin, the unaided win rate against Heuristic rose from 15.2% to 26.0% near 1.54 million decisions (paired 95% CI +8.3 to +13.2 points). Against Random it rose from 89.5% to 93.9%. On fixed tactical positions, the top choice was an immediate win 50.3% of the time, up from 29.7%. |
| **50% Heuristic opponents** | [Training config](../experiments/double-dqn-half-heuristic-full.json) · [Shared holdout](../experiments/next-holdout.json) | Increasing the Heuristic share from 20% to 50% moved the Heuristic holdout win rate from 15.5% to 13.7% (paired 95% CI −4.0 to +0.4 points). Random stayed near 90%. There was no clear gain. |
| **Temperature-controlled exploration pilot** | [Temperature preset](../experiments/double-dqn-temperature-full.json) · [ε-greedy control preset](../experiments/double-dqn-immediate-win-full.json) | Seed 42 used immediate-win supervision in both arms. Near one million decisions, ε-greedy won 20.7% against Heuristic and 93.6% against Random. Temperature sampling won 11.3% and 88.3%, respectively. At 35,840 episodes the gap narrowed. Temperature reached 21.3% against Heuristic at 1.19 million decisions, and ε-greedy reached 22.3% at 1.13 million. This is one seed and one temperature schedule. |

To reproduce the immediate-win rule comparison, use `evaluate-dqn` twice on each final baseline checkpoint with the shared holdout. Use `--tactics` for the greedy policy and `--take-immediate-wins` for the rule-assisted policy.

### Reproduce the temperature pilot

Copy the two presets into new JSON files and set `episodes` to **35840** in each. The temperature arm linearly anneals from **0.5 to 0.025** over **750,000** learner decisions and samples among legal actions using a softmax of their Q-values. The ε-greedy arm anneals from **1.0 to 0.05** over the same period. For each arm, train with seed 42 and use the checkpoint nearest 500,000 or 1 million decisions for each comparison. Use the shared holdout config for greedy evaluation.

## Planned experiments

1. **Full-budget DQN vs. Double DQN.** Run seeds 42–44 using the [DQN preset](../experiments/dqn-mixed-full.json) and compare to the completed [Double DQN baseline](../experiments/double-dqn-mixed-full.json). Then consider a DQN batch with the same 0.1 tactical loss to complete a two-by-two comparison of target algorithm and supervision.
2. **Public move history.** Add the opponent's last four publicly observed card plays and targets to the supervised agent. Compare against an otherwise identical model trained with zeroed history inputs. This tests whether public action order helps under incomplete information. Requires observation and model changes.
3. **Generalization to new opponents.** Evaluate existing checkpoints against a defense-heavy Heuristic variant and frozen checkpoints from other seeds. If the gains vary by opponent, try sampling older self-play checkpoints while keeping the Heuristic share at 20%. Requires evaluator support and, for the pool experiment, trainer support.
4. **Stronger immediate-win loss.** In a copy of the [supervised preset](../experiments/double-dqn-immediate-win-full.json), set `immediate_win_loss_weight` to 0.2 and keep the margin and opponent mix fixed. Compare tactical ranking and unaided game win rates against the completed 0.1 runs.
5. **Replicate temperature exploration if pursuing it further.** Repeat the tested schedule on seeds 43 and 44 and compare at matched decision counts. Keep ε-greedy as the reference based on the seed-42 pilot. [Temperature preset](../experiments/double-dqn-temperature-full.json)
