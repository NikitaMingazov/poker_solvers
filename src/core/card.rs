use std::cmp;
use std::fmt;
use std::mem;

use super::error::RSPokerError;

// #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(PartialEq, PartialOrd, Eq, Ord, Debug, Clone, Copy, Hash)]
pub enum Value {
    Two = 0,
    Three = 1,
    Four = 2,
    Five = 3,
    Six = 4,
    Seven = 5,
    Eight = 6,
    Nine = 7,
    Ten = 8,
    Jack = 9,
    Queen = 10,
    King = 11,
    Ace = 12,
}

const VALUES: [Value; 13] = [
    Value::Two,
    Value::Three,
    Value::Four,
    Value::Five,
    Value::Six,
    Value::Seven,
    Value::Eight,
    Value::Nine,
    Value::Ten,
    Value::Jack,
    Value::Queen,
    Value::King,
    Value::Ace,
];

impl Value {
    pub fn from_u8(v: u8) -> Self {
        Self::from(v)
    }
    pub const fn values() -> [Self; 13] {
        VALUES
    }

    pub fn from_char(c: char) -> Option<Self> {
        Self::try_from(c).ok()
    }

    pub fn to_char(self) -> char {
        char::from(self)
    }

    pub fn gap(self, other: Self) -> u8 {
        let min = cmp::min(self as u8, other as u8);
        let max = cmp::max(self as u8, other as u8);
        max - min
    }
}

impl From<u8> for Value {
    fn from(value: u8) -> Self {
        unsafe { mem::transmute(cmp::min(value, Self::Ace as u8)) }
    }
}

impl TryFrom<char> for Value {
    type Error = RSPokerError;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value.to_ascii_uppercase() {
            'A' => Ok(Self::Ace),
            'K' => Ok(Self::King),
            'Q' => Ok(Self::Queen),
            'J' => Ok(Self::Jack),
            'T' => Ok(Self::Ten),
            '9' => Ok(Self::Nine),
            '8' => Ok(Self::Eight),
            '7' => Ok(Self::Seven),
            '6' => Ok(Self::Six),
            '5' => Ok(Self::Five),
            '4' => Ok(Self::Four),
            '3' => Ok(Self::Three),
            '2' => Ok(Self::Two),
            _ => Err(RSPokerError::UnexpectedValueChar),
        }
    }
}

impl From<Value> for char {
    fn from(value: Value) -> Self {
        match value {
            Value::Ace => 'A',
            Value::King => 'K',
            Value::Queen => 'Q',
            Value::Jack => 'J',
            Value::Ten => 'T',
            Value::Nine => '9',
            Value::Eight => '8',
            Value::Seven => '7',
            Value::Six => '6',
            Value::Five => '5',
            Value::Four => '4',
            Value::Three => '3',
            Value::Two => '2',
        }
    }
}

impl From<Value> for u8 {
    fn from(value: Value) -> Self {
        value as u8
    }
}

// #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(PartialEq, PartialOrd, Eq, Ord, Debug, Clone, Copy, Hash)]
pub enum Suit {
    Spade = 0,
    Club = 1,
    Heart = 2,
    Diamond = 3,
}

const SUITS: [Suit; 4] = [Suit::Spade, Suit::Club, Suit::Heart, Suit::Diamond];

impl Suit {
    pub const fn suits() -> [Self; 4] {
        SUITS
    }

    pub fn from_u8(s: u8) -> Self {
        Self::from(s)
    }

    pub fn from_char(s: char) -> Option<Self> {
        TryFrom::try_from(s).ok()
    }

    pub fn to_char(self) -> char {
        char::from(self)
    }
}

impl From<u8> for Suit {
    fn from(value: u8) -> Self {
        unsafe { mem::transmute(cmp::min(value, Self::Diamond as u8)) }
    }
}

impl From<Suit> for u8 {
    fn from(value: Suit) -> Self {
        value as u8
    }
}

impl TryFrom<char> for Suit {
    type Error = RSPokerError;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value.to_ascii_lowercase() {
            'd' => Ok(Self::Diamond),
            's' => Ok(Self::Spade),
            'h' => Ok(Self::Heart),
            'c' => Ok(Self::Club),
            _ => Err(RSPokerError::UnexpectedSuitChar),
        }
    }
}

impl From<Suit> for char {
    fn from(value: Suit) -> Self {
        match value {
            Suit::Diamond => 'd',
            Suit::Spade => 's',
            Suit::Heart => 'h',
            Suit::Club => 'c',
        }
    }
}

#[derive(PartialEq, PartialOrd, Eq, Ord, Clone, Copy, Hash)]
pub struct Card {
    pub value: Value,
    pub suit: Suit,
}

impl Card {
    pub fn new(value: Value, suit: Suit) -> Self {
        Self { value, suit }
    }
}

impl From<Card> for u8 {
    fn from(card: Card) -> Self {
        u8::from(card.suit) * 13 + u8::from(card.value)
    }
}

impl From<u8> for Card {
    fn from(value: u8) -> Self {
        Self {
            value: Value::from(value % 13),
            suit: Suit::from(value / 13),
        }
    }
}

impl fmt::Debug for Card {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Card({}{})",
            char::from(self.value),
            char::from(self.suit)
        )
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", char::from(self.value), char::from(self.suit))
    }
}

impl TryFrom<&str> for Card {
    type Error = RSPokerError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let mut chars = value.chars();
        let value_char = chars.next().ok_or(RSPokerError::TooFewChars)?;
        let suit_char = chars.next().ok_or(RSPokerError::TooFewChars)?;
        Ok(Self {
            value: Value::try_from(value_char)?,
            suit: Suit::try_from(suit_char)?,
        })
    }
}

// #[cfg(feature = "serde")]
impl serde::Serialize for Card {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Use existing Display implementation which formats as "Ah", "Kc", etc
        serializer.serialize_str(&self.to_string())
    }
}

// #[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Card {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Card::try_from(s.as_str()).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constructor() {
        let c = Card {
            value: Value::Three,
            suit: Suit::Spade,
        };
        assert_eq!(Suit::Spade, c.suit);
        assert_eq!(Value::Three, c.value);
    }

    #[test]
    fn test_suit_from_u8() {
        assert_eq!(Suit::Spade, Suit::from_u8(0));
        assert_eq!(Suit::Club, Suit::from_u8(1));
        assert_eq!(Suit::Heart, Suit::from_u8(2));
        assert_eq!(Suit::Diamond, Suit::from_u8(3));
    }

    #[test]
    fn test_value_from_u8() {
        assert_eq!(Value::Two, Value::from_u8(0));
        assert_eq!(Value::Ace, Value::from_u8(12));
    }

    #[test]
    fn test_roundtrip_from_u8_all_cards() {
        for suit in SUITS {
            for value in VALUES {
                let c = Card { suit, value };
                let u = u8::from(c);
                assert_eq!(c, Card::from(u));
            }
        }
    }

    #[test]
    fn test_try_parse_card() {
        let expected = Card {
            value: Value::King,
            suit: Suit::Spade,
        };

        assert_eq!(expected, Card::try_from("Ks").unwrap())
    }

    #[test]
    fn test_parse_all_cards() {
        for suit in SUITS {
            for value in VALUES {
                let e = Card { suit, value };
                let card_string = format!("{}{}", char::from(value), char::from(suit));
                assert_eq!(e, Card::try_from(card_string.as_str()).unwrap());
            }
        }
    }

    #[test]
    fn test_compare() {
        let c1 = Card {
            value: Value::Three,
            suit: Suit::Spade,
        };
        let c2 = Card {
            value: Value::Four,
            suit: Suit::Spade,
        };
        let c3 = Card {
            value: Value::Four,
            suit: Suit::Club,
        };

        // Make sure that the values are ordered
        assert!(c1 < c2);
        assert!(c2 > c1);
        // Make sure that suit is used.
        assert!(c3 > c2);
    }

    #[test]
    fn test_value_cmp() {
        assert!(Value::Two < Value::Ace);
        assert!(Value::King < Value::Ace);
        assert_eq!(Value::Two, Value::Two);
    }

    #[test]
    fn test_from_u8() {
        assert_eq!(Value::Two, Value::from_u8(0));
        assert_eq!(Value::Ace, Value::from_u8(12));
    }

    #[test]
    fn test_size_card() {
        // Card should be really small. Hopefully just two u8's
        assert!(mem::size_of::<Card>() <= 2);
    }

    #[test]
    fn test_size_suit() {
        // One byte for Suit
        assert!(mem::size_of::<Suit>() <= 1);
    }

    #[test]
    fn test_size_value() {
        // One byte for Value
        assert!(mem::size_of::<Value>() <= 1);
    }

    #[test]
    fn test_gap() {
        // test on gap
        assert!(1 == Value::Ace.gap(Value::King));
        // test no gap at the high end
        assert!(0 == Value::Ace.gap(Value::Ace));
        // test no gap at the low end
        assert!(0 == Value::Two.gap(Value::Two));
        // Test one gap at the low end
        assert!(1 == Value::Two.gap(Value::Three));
        // test that ordering doesn't matter
        assert!(1 == Value::Three.gap(Value::Two));
        // Test things that are far apart
        assert!(12 == Value::Ace.gap(Value::Two));
        assert!(12 == Value::Two.gap(Value::Ace));
    }

    #[test]
    fn test_suit_to_char() {
        let s = Suit::Spade;
        assert_eq!('s', s.to_char());

        let s = Suit::Club;
        assert_eq!('c', s.to_char());

        let s = Suit::Heart;
        assert_eq!('h', s.to_char());

        let s = Suit::Diamond;
        assert_eq!('d', s.to_char());
    }

    #[test]
    fn test_value_to_char() {
        let v = Value::Ace;
        assert_eq!('A', v.to_char());

        let v = Value::King;
        assert_eq!('K', v.to_char());

        let v = Value::Queen;
        assert_eq!('Q', v.to_char());

        let v = Value::Jack;
        assert_eq!('J', v.to_char());

        let v = Value::Ten;
        assert_eq!('T', v.to_char());

        for i in 2..=9 {
            // The eunm is 0 based so the first value is 0 while cards
            // start at 2
            let v = Value::from(i - 2);
            assert_eq!(char::from_digit(u32::from(i), 10).unwrap(), v.to_char());
        }
    }
}
