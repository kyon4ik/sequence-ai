# GPU inference benchmarks

[Research protocols](README.md) · [Inference configuration](../docs/configuration.md#inference-batching-and-timing) · [Development setup](../docs/development.md)

These are measurements of specific checkpoints and workloads. For portable
usage, the default `inference_batch_games` remains 16. Run benchmark commands
from the repository root.

Measured on a GTX 1660 Ti using the iteration-100 model, epsilon 0.175 and
128 fixed-seed self-play games (median of three warmed runs):

| Concurrent games | Self-play time | Games/second | Evaluation time (200 games) |
| --- | ---: | ---: | ---: |
| 1 | 8.82 s | 14.5 | 7.90 s |
| 8 | 3.60 s | 35.6 | 3.93 s |
| 16 | 3.17 s | 40.4 | 3.27 s |
| 32 | 2.94 s | 43.6 | 2.97 s |

All settings produced 7,737 self-play actions and evaluation score 0.75 on this
workload. CPU candidate encoding took about 0.14 s; inference service time still
dominated. These measurements exclude gradient updates and checkpoint writes.
The portable default remains 16.

An extended comparison warmed the full workload for each size and interleaved
five repetitions of both phases. With the same model and game seeds:

| Concurrent games | Median self-play time | Median evaluation time |
| --- | ---: | ---: |
| 32 | 2.968 s | 3.013 s |
| 64 | 2.888 s | 2.832 s |
| 128 | 2.795 s | 2.761 s |

128 was fastest among the tested sizes, giving about 6% greater self-play
throughput than 32. All evaluation scores remained 0.75. Use 128 for this
128-game workload on this GPU; larger settings cannot increase concurrency
beyond the number of games in the workload. This recommendation applies to
the tested checkpoint, network and hardware.

A separate CLI comparison resumed the same legacy iteration-100 checkpoint
through iteration 103. Excluding the first iteration's warmup, iteration time
averaged 9.91 s before batching and 4.07 s with 32 games (2.4× faster), excluding
checkpoint writes. Sample counts, outcomes, rewards and MSE matched exactly in
those three iterations; this does not guarantee identical trajectories on other
hardware or workloads.

Run the explicit GPU benchmark outside the sandbox, with no competing training:

```sh
SEQUENCE_BENCH_CHECKPOINT="$PWD/data/models/mc-v1-long/checkpoints/iteration-000100/model.bpk" \
  cargo test --release -p sequence-training --features burn cuda_rollout_throughput \
  -- --ignored --nocapture --test-threads=1
```

The checkpoint variable is optional; omit it to benchmark a seeded fresh model.
The benchmark warms each setting, compares 1/8/16/32/64/128 concurrent games on
the same 128 seeded episodes with five interleaved repetitions, validates their
replays, and times evaluation against both baselines. Set
`SEQUENCE_BENCH_BATCH_GAMES=32,64,128` to restrict the sweep. Its output is JSON
with throughput and inference timing breakdowns.
