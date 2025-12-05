//! This is the core module. It exports the non-holdem
//! related code.

mod error;
pub use self::error::RSPokerError;
mod card;
pub use self::card::{Card, Suit, Value};

mod hand;
pub use self::hand::*;
mod flat_hand;
pub use self::flat_hand::*;

mod card_iter;
pub use self::card_iter::*;

mod deck;
pub use self::deck::Deck;

mod flat_deck;
pub use self::flat_deck::FlatDeck;

mod rank;
pub use self::rank::{Rank, Rankable};

// u16 backed player set.
mod player_bit_set;
// u64 backed card set.
mod card_bit_set;
// Export the bit set and the iterator
pub use self::player_bit_set::{ActivePlayerBitSetIter, PlayerBitSet};
// Export the bit set and the iterator used for cards (52 cards so u64 backed)
pub use self::card_bit_set::{CardBitSet, CardBitSetIter};
