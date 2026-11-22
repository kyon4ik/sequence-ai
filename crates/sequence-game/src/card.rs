#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Suit {
    Clubs,
    Diamonds,
    Hearts,
    Spades,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Rank {
    Ace,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Queen,
    King,
    Jack,
}

const SUITS: [Suit; 4] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];
const RANKS: [Rank; 13] = [
    Rank::Ace,
    Rank::Two,
    Rank::Three,
    Rank::Four,
    Rank::Five,
    Rank::Six,
    Rank::Seven,
    Rank::Eight,
    Rank::Nine,
    Rank::Ten,
    Rank::Queen,
    Rank::King,
    Rank::Jack,
];
const NON_JACK_RANKS: usize = RANKS.len() - 1;
pub(crate) const BOARD_LABEL_COUNT: usize = SUITS.len() * NON_JACK_RANKS;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum Effect {
    Match,
    Wild,
    Remove,
}

impl Card {
    pub const fn new(rank: Rank, suit: Suit) -> Self {
        Self { rank, suit }
    }

    pub fn all() -> impl Iterator<Item = Self> {
        SUITS
            .into_iter()
            .flat_map(|suit| RANKS.into_iter().map(move |rank| Self::new(rank, suit)))
    }

    pub(crate) fn effect(self) -> Effect {
        match (self.rank, self.suit) {
            (Rank::Jack, Suit::Clubs | Suit::Diamonds) => Effect::Wild,
            (Rank::Jack, Suit::Hearts | Suit::Spades) => Effect::Remove,
            _ => Effect::Match,
        }
    }

    pub(crate) fn label_index(self) -> Option<usize> {
        (self.rank != Rank::Jack)
            .then_some(self.suit as usize * NON_JACK_RANKS + self.rank as usize)
    }

    pub(crate) const fn from_label_index(index: usize) -> Self {
        Self::new(RANKS[index % NON_JACK_RANKS], SUITS[index / NON_JACK_RANKS])
    }
}
