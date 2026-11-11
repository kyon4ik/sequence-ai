# sequence-runner

Executes one two-player match. Tournament scheduling, Elo, storage, training,
and UI code belong to their own crates.

`MatchRunner<G>` accepts any `RecordedGame` and two `Participant`s containing
`Agent<G>` trait objects. `run_factories` builds fresh policies using the recorded
agent seeds. Red is seat 0 and Blue is seat 1. Agent descriptors identify model
versions independently of seats. Matching descriptor IDs are allowed for self-play.
RandomAgent supports both built-in variants; GreedyAgent, McAgent and the built-in
AgentSpec factory currently support Simple. Custom factories can support Classic.

Each episode starts from `G::from_seed`. The runner calls start/end hooks on both
agents, regenerates actions after each application, validates chosen actions, and
records every successful application. It always uses `current_player`, so a Classic
exchange retains the same actor. Random and MC agents reset exploration RNGs from
the episode context when reused.

`MatchConfig::max_actions` counts applications, including exchanges, rather than
completed turns. `Termination::ActionLimit` is unfinished, not a draw. Inference,
illegal-action, nonfinite-reward and event-sink errors return `RunError::Execution`
with a partial replay. Setup/factory errors return no record. Backend panics are
not caught, and an in-process inference call cannot be forcibly timed out. A future
process/remote agent wrapper can impose such a deadline.

`TerminalReward` emits +1/-1 on the winning move for both seats and zero otherwise.
`Reward<G>` supports shaping using before/after states and resets at each episode.
Every step records Red/Blue rewards; match totals sum them without discounting.
Both individual rewards and cumulative returns must be finite. An invalid reward
aborts the match; its already-applied action is retained with zero rewards.

`MatchMetrics` includes total elapsed time, successful actions, completed turns,
and per-player decision counts, total/maximum/mean decision time, and reward.
Decision time measures the policy call, including synchronous inference and
transfers; it excludes rule execution and recording. Failed attempts count as
decisions. Elapsed time includes episode hooks and recording except the final
sink call, and excludes factory/model creation. Timings vary between identical
seeded runs and are not replayed.

`EventSink` receives `Started`, `Action`, and `Finished` events in order. A
`Vec<MatchEvent>` is a built-in sink. Implement a match-scoped sink for SQLite,
JSONL, or live WebSocket consumers. Errors from a sink stop execution, preserve
the partial record, and prevent further calls into that sink. Agent/rule/reward
failures emit an aborted finish event if the sink is still healthy. End hooks run
before recording the finish event; a failure writing that final event changes the
returned record to aborted after those hooks have run. One sink instance should
represent one match; a scheduler can attach its own match identifier externally.

`MatchRecord`, events and agent specs implement serde. The runner does not choose
file locations, a database, or a model registry. A later storage layer can serialize
them without becoming part of match execution.

Replays have two complementary interfaces:

- `MatchRecord::frame(n)` returns a stored observation for any frame, without
  loading a model or running the engine. Frame 0 is the initial deal. Board cells,
  claimed sequences, the current actor's hand, public piles, counters and outcome
  are included. A replay contains observations from both seats over time, so it
  is an analysis artifact; do not expose a full replay to a live opponent agent.
- `Replay::<G>::new(&record)` reconstructs the game from its seed and actions and
  verifies every recorded observation, actor and action index, plus outcome and
  reward totals. It supports seek, forward, backward, and cloning a frame's game
  state for a continuation. No agent or checkpoint is needed. Failed inference
  attempts do not add frames because they did not apply an action.

`FORMAT_VERSION` identifies the record schema; `RecordedGame::ID` identifies the
variant/rules contract. Seed reproduction requires compatible core and RNG
versions. Snapshots remain viewable when reconstruction diverges; they do not
contain the hidden draw order and cannot alone restore an executable game after
an incompatible engine change. Full-state archival or migrations belong to a
future storage format. Custom reward values are recorded, not recomputed during
replay.
