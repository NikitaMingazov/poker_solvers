use crate::core::card::Card;
use crate::core::deck::Deck;
use std::ops::{Index, Range, RangeFrom, RangeFull, RangeTo};

extern crate rand;
use rand::Rng;
use rand::rng;
use rand::seq::{IndexedRandom, SliceRandom};

#[derive(Debug, Clone, PartialEq)]
pub struct FlatDeck {
    cards: Vec<Card>,
}

impl FlatDeck {
    pub fn len(&self) -> usize {
        self.cards.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }

    pub fn push(&mut self, c: Card) {
        self.cards.push(c);
    }

    pub fn sample(&self, n: usize) -> Vec<Card> {
        let mut rng = rng();
        self.cards.choose_multiple(&mut rng, n).cloned().collect()
    }

    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        self.cards.shuffle(rng)
    }

    pub fn deal(&mut self) -> Option<Card> {
        self.cards.pop()
    }
}

impl Index<usize> for FlatDeck {
    type Output = Card;
    fn index(&self, index: usize) -> &Card {
        &self.cards[index]
    }
}
impl Index<Range<usize>> for FlatDeck {
    type Output = [Card];
    fn index(&self, index: Range<usize>) -> &[Card] {
        &self.cards[index]
    }
}
impl Index<RangeTo<usize>> for FlatDeck {
    type Output = [Card];
    fn index(&self, index: RangeTo<usize>) -> &[Card] {
        &self.cards[index]
    }
}
impl Index<RangeFrom<usize>> for FlatDeck {
    type Output = [Card];
    fn index(&self, index: RangeFrom<usize>) -> &[Card] {
        &self.cards[index]
    }
}
impl Index<RangeFull> for FlatDeck {
    type Output = [Card];
    fn index(&self, index: RangeFull) -> &[Card] {
        &self.cards[index]
    }
}

impl From<Vec<Card>> for FlatDeck {
    fn from(value: Vec<Card>) -> Self {
        Self { cards: value }
    }
}

impl From<Deck> for FlatDeck {
    fn from(value: Deck) -> Self {
        // We sort the cards so that the same input
        // cards always result in the same starting flat deck
        let mut cards: Vec<Card> = value.into_iter().collect();
        cards.sort();
        Self { cards }
    }
}
impl Default for FlatDeck {
    fn default() -> Self {
        let mut cards: Vec<Card> = Deck::default().into_iter().collect();
        let mut rng = rng();
        cards.shuffle(&mut rng);
        Self { cards }
    }
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;
    use crate::core::card::{Suit, Value};

    #[test]
    fn test_deck_from() {
        let fd: FlatDeck = Deck::default().into();
        assert_eq!(52, fd.len());
    }

    #[test]
    fn test_from_vec() {
        let c = Card {
            value: Value::Nine,
            suit: Suit::Heart,
        };
        let v = vec![c];

        let mut flat_deck: FlatDeck = v.into();

        assert_eq!(1, flat_deck.len());
        assert_eq!(c, flat_deck.deal().unwrap());
    }

    #[test]
    fn test_shuffle_rng() {
        let mut fd_one: FlatDeck = Deck::default().into();
        let mut fd_two: FlatDeck = Deck::default().into();

        let mut rng_one = StdRng::seed_from_u64(420);
        let mut rng_two = StdRng::seed_from_u64(420);

        fd_one.shuffle(&mut rng_one);
        fd_two.shuffle(&mut rng_two);

        assert_eq!(fd_one, fd_two);
        assert_eq!(fd_one, fd_two);
    }

    #[test]
    fn test_index() {
        let mut fd: FlatDeck = Deck::new().into();

        let c = Card {
            value: Value::Nine,
            suit: Suit::Heart,
        };
        fd.push(c);
        assert_eq!(c, fd[0]);

        let mut fd: FlatDeck = Deck::new().into();
        let c = Card {
            value: Value::Nine,
            suit: Suit::Heart,
        };
        let c2 = Card {
            value: Value::Ten,
            suit: Suit::Heart,
        };
        fd.push(c);
        fd.push(c2);
        assert_eq!(c, fd[0]);
        assert_eq!(c2, fd[1]);
    }

    #[test]
    fn test_is_empty() {
        let mut fd: FlatDeck = Deck::new().into();
        assert!(fd.is_empty());

        fd.push(Card {
            value: Value::Nine,
            suit: Suit::Heart,
        });
        assert!(!fd.is_empty());
        let dealt_card = fd.deal();

        assert!(fd.is_empty());
        assert_eq!(
            Some(Card {
                value: Value::Nine,
                suit: Suit::Heart,
            }),
            dealt_card
        );
    }
}
