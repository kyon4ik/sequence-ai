use strum::{EnumCount, FromRepr, VariantArray};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Card(u8);

#[derive(Clone, Copy, PartialEq, Eq, Debug, EnumCount, VariantArray, FromRepr)]
#[repr(u8)]
pub enum CardSuit {
    Clubs = 0,
    Spades = 1,
    Hearts = 2,
    Diamonds = 3,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, EnumCount, VariantArray, FromRepr)]
#[repr(u8)]
pub enum CardRank {
    Ace = 0,
    Two = 1,
    Three = 2,
    Four = 3,
    Five = 4,
    Six = 5,
    Seven = 6,
    Eight = 7,
    Nine = 8,
    Ten = 9,
    Jack = 10,
    Queen = 11,
    King = 12,
}

impl Card {
    pub const COUNT: usize = CardSuit::COUNT * CardRank::COUNT;

    pub const fn new(rank: CardRank, suit: CardSuit) -> Self {
        let inner = (rank as u8) << 2 | suit as u8;
        Self(inner)
    }

    pub fn rank(&self) -> CardRank {
        CardRank::from_repr(self.0 >> 2).unwrap()
    }

    pub fn suit(&self) -> CardSuit {
        // Safety: guaranted by construction
        CardSuit::from_repr(self.0 & 3).unwrap()
    }

    pub const fn id(&self) -> usize {
        self.0 as usize
    }

    pub fn is_one_eyed_jack(&self) -> bool {
        *self == Card::new(CardRank::Jack, CardSuit::Hearts)
            || *self == Card::new(CardRank::Jack, CardSuit::Spades)
    }

    pub fn is_two_eyed_jack(&self) -> bool {
        *self == Card::new(CardRank::Jack, CardSuit::Diamonds)
            || *self == Card::new(CardRank::Jack, CardSuit::Clubs)
    }
}

impl Card {
    pub(crate) const fn from_str(token: &str) -> Option<Card> {
        let (rank, suit) = match token.as_bytes() {
            [b'_'] => return None,
            [rank, suit] => {
                let rank = match rank {
                    b'A' => CardRank::Ace,
                    b'2' => CardRank::Two,
                    b'3' => CardRank::Three,
                    b'4' => CardRank::Four,
                    b'5' => CardRank::Five,
                    b'6' => CardRank::Six,
                    b'7' => CardRank::Seven,
                    b'8' => CardRank::Eight,
                    b'9' => CardRank::Nine,
                    b'Q' => CardRank::Queen,
                    b'K' => CardRank::King,
                    _ => panic!("invalid board card value"),
                };
                (rank, *suit)
            }
            [b'1', b'0', suit] => (CardRank::Ten, *suit),
            _ => panic!("invalid board card token"),
        };

        let suit = match suit {
            b'c' => CardSuit::Clubs,
            b'd' => CardSuit::Diamonds,
            b'h' => CardSuit::Hearts,
            b's' => CardSuit::Spades,
            _ => panic!("invalid board card suit"),
        };

        Some(Card::new(rank, suit))
    }
}
