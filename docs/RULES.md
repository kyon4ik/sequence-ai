# Two-player Sequence rules

## Objective

A sequence is a horizontal, vertical, or diagonal line of five consecutive spaces. Each space must contain one of the player's chips or be a free corner. A corner counts for both players, so a line through a corner needs four chips. A player wins immediately when they have at least two claimed sequences.

## Setup

- Start with an empty 10 × 10 board. Every non-Jack rank and suit combination labels two spaces. The four free corners never hold chips. Each player uses a distinct chip color.
- Shuffle two copies of each standard playing card, for 104 cards total. Use the game's seeded random generator for the initial shuffle and all later shuffles.
- Assign the first seven cards to Player 1 and the next seven to Player 2. The remaining cards form the draw pile. Both discard piles start empty, and the consecutive-pass count starts at zero.
- The game configuration specifies which player starts. Turns alternate between Player 1 and Player 2.

Record the shuffle seed, starting player, and policy assigned to each player with the game result.

## Player information

Each player sees their own hand, the board, the claimed sequences, both discard piles, whose turn it is, and the consecutive-pass count. Hand sizes and draw-pile size are public. The opponent's hand contents, draw-pile order, and shuffle seed are hidden from the players.

## Playing a turn

1. At the start of the turn, resolve a dead-card exchange if applicable.
2. Play one card face up to your discard pile and carry out its effect. If you have no legal play, follow the passing rule below.
3. If you placed a chip, resolve automatic claims and check for a win.
4. If play continues, draw one replacement card and end the turn.

When the draw pile runs out, shuffle the discard piles together to replenish it. This applies to both regular draws and exchanges.

| Card played | Effect |
| --- | --- |
| Any non-Jack | Place one of your chips on either unoccupied space matching the card's rank and suit. |
| Jack of clubs or diamonds (two-eyed) | Place one of your chips on any unoccupied non-corner space. |
| Jack of hearts or spades (one-eyed) | Remove one opposing chip that belongs to no claimed sequence. The space becomes empty. |

Each card requires a legal target for its effect. Chips in claimed sequences are protected from removal for the rest of the game.

## Dead-card exchange

A non-Jack card is dead when both board spaces matching it are occupied. At the start of a turn, the player may exchange one dead card from their hand if a replacement can be drawn. The exchange does not end the turn, and the player may play the replacement immediately. A second exchange is unavailable that turn, even if the replacement is dead.

To exchange a dead card, add it face up to your discard pile and draw one replacement.

## Passing and draws

- A player may pass only when their hand offers no legal chip placement or removal and they have no exchange available. If an exchange is available and the player has no legal play, they must exchange a dead card and check for legal plays again.
- A pass ends the turn and increases the consecutive-pass count by one. A chip placement or removal resets the count to zero. An exchange alone leaves the count unchanged.
- The third consecutive pass ends the game immediately in a draw.

## Sequence overlap

Any two sequences claimed by the same player may share at most one chip. Apply the limit separately to each pair, including sequences claimed on the same turn.

## Automatic claims

After each chip placement, identify the sequences completed by that placement. Automatically claim the largest group that satisfies the overlap limit among its members and with all sequences already claimed by that player.

If several largest groups are valid, compare them in board order. Order spaces by row from top to bottom, then by column from left to right. Order each sequence by its sorted list of spaces and each group by its sorted list of sequences. Compare the groups entry by entry and claim the group that comes first at the first difference.
