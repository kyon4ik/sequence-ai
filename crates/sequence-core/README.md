# sequence-core

[Project overview](../../README.md) · [Architecture and Rust APIs](../../docs/architecture.md)

Two-player Sequence with seven-card hands and a fixed 10 × 10 board.

## Game variants

`Game` is a trait implemented independently by `Classic` and `Simple`. Import
`Game` to access seeded construction, board queries, observations, legal actions,
validation, and application. Its associated observation type allows each variant
to expose the information relevant to its rules. The game implementations share
card dealing and basic storage operations, without a runtime rules enum.

`Classic` uses two complete decks (104 cards). Hearts and spades jacks remove
an opposing chip unless it belongs to a claimed sequence. Clubs and diamonds
jacks place a chip on any empty noncorner square. A dead ordinary card has both
printed squares occupied. One dead-card exchange is permitted per turn; it draws
a replacement and retains the actor. Jacks cannot be exchanged. Used cards enter
a public discard pile, which is shuffled and recycled when a replacement is
needed and the draw pile is empty. The just-used card participates in recycling.

`Simple` uses two copies of each ordinary card (96 cards), with no jacks,
exchanges, or discard recycling. It does not store an RNG after dealing or any
Classic exchange/pass-limit state. A game without a winner draws when both
hands are empty.

Both variants need two sequences to win. A sequence has five squares; free
corners count for either player. Claims are automatic: scan horizontal, vertical,
down-right, then up-right directions. Within each direction, scan five-square
windows containing the new chip in increasing start-position order. A window may
share at most one square with each prior claim, including claims from this move.
All resulting claims are recorded before deciding victory.

Passing is legal only when no play or exchange is available. In Classic, a play
resets consecutive passes; an exchange does not. A pass or play completes a turn.
Classic draws after 300 completed turns or six consecutive passes. A winning
placement takes precedence over a draw limit reached on that move. Terminal
games reject all actions and offer no legal actions.

## Public interface

`SimpleObservation` and `ClassicObservation` expose only the actor's hand and
public information. Neither reveals opponent cards or draw order. Classic also
exposes exchange availability and consecutive passes. Cloning a Classic game
preserves the RNG state for reproducible future reshuffles. Seeds are reproducible
with the same crate and dependency versions, not a cross-version replay format.

`legal_actions` replaces its caller-owned buffer. `validate` checks an action
directly; `apply` returns `ActionError` and leaves all state unchanged when an
action is rejected. Actions refer to current hand indices: regenerate them after
each application, including exchanges. Duplicate cards retain separate actions.

Board queries use typed cells and immutable sequence records. `place_chip` is a
checked board operation that places a chip and claims sequences, independently
of hand or turn rules. It rejects occupied squares and corners without mutation.
`is_owned_by` includes free corners for either color. `is_in_sequence` reports
claimed membership; `is_protected` also includes corners. Position construction
and index conversion are checked. Card IDs and rank ordering remain unchanged.

## Research and bots

Afterstate construction belongs to `sequence-agent`, not the game trait or core
observations. `sequence_agent::Afterstate::new` accepts a `SimpleObservation` and
captures the board and remaining hand before replacement drawing or turn
advancement. It validates the action and uses the core board's placement rules.
Its result is not a complete next observation.

`sequence_agent::Agent<G>` associates a fallible policy with a game variant.
Random supports both variants; greedy and Monte Carlo currently support `Simple`.
Greedy uses the original own-run scoring weights and first-candidate tie breaking.
The Monte Carlo encoder retains its 400-board/50-context layout. Tensor layouts,
rewards, policies, and training remain outside `sequence-core`. Matches, metrics,
events, and replay views live in `sequence-runner`; Monte Carlo self-play training
lives in `sequence-training`. The optional `serde` feature serializes public
actions, colors, outcomes, cells and checked card/position identifiers, without
serializing private game state or packed board storage.
