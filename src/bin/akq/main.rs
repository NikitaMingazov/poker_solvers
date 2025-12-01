/*The fundamental elements of the game are these:

• There is a three-card deck, containing an Ace, a King, and a Queen.
• The players are each dealt one card without replacement and there is an initial ante.
• There is a round of betting, after which there is a showdown (if neither player folds). In
the showdown, the high card wins.*/

		#![allow(warnings)]
// I'm doing math, this variable standard is ridiculous if T violates it
#![allow(non_snake_case)]

#[allow(non_upper_case_globals)]
const ε : f64 = 0.0001;

use std::fs::File;
use std::io::BufReader;
use serde::{Deserialize};
use std::env;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
struct StreetActions {
	open : Vec<Action>,
	facing_bet : Vec<Vec<Action>>,
}

#[derive(Debug, Deserialize)]
struct ActionSet(Vec<StreetActions>);
impl ActionSet {
	fn actions_at(&self, street : u8, bet_level : u8) -> Option<Vec<Action>> {
		// bet level of 0 means check, 1=>bet, 2=>3bet etc.
		if bet_level == 0 {
			if usize::from(street) < self.0.len() {
				return Some(self.0[street as usize].open.clone());
			}
		} else {
			if usize::from(street) < self.0.len() {
				if usize::from(bet_level-1) < self.0[street as usize].facing_bet.len() {
					return Some(self.0[street as usize].facing_bet[usize::from(bet_level-1)].clone());
				}
			}
		}
		return None;
	}
}

fn parse_config(path: &str) -> Result<ActionSet, Box<dyn std::error::Error>> {
	let file = File::open(path)?;
	let reader = BufReader::new(file);
	let cfg = serde_json::from_reader(reader)?;
	Ok(cfg)
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
enum Action {
	Check,
	Call,
	Fold,
	Bet(f64), // pot%
	// Allin,
}

#[derive(Debug, Clone)]
struct Player {
	seat : u8,
	range : HashMap<Card, f64>,
	// committed : f64,
	// stack : f64,
}

impl Player {
	fn initlist(n : u8, deck : Vec<Card>) -> Vec<Player> {
		let mut players : Vec<Player> = Vec::new();
		for i in 0..n {
			players.push(Self::new(i, deck.clone()));
		}
		players
	}
	fn new(seat: u8, deck: Vec<Card>) -> Self {
		let mut range : HashMap<Card, f64> = HashMap::new();
		for card in deck {
			range.insert(card, 1.0);
		}
		Self {
			seat,
			range
		}
	}
}

#[derive(Debug, Clone)]
struct Strategy(HashMap<Card, Vec<f64>>);

impl Strategy {
	fn init(actionset: &ActionSet, street : u8, level : u8, hand_set : &Vec<Card>) -> Self {
		let mut strat : Self = Strategy(HashMap::new());
		match actionset.actions_at(street, level) {
			None => (),
			Some(actions) => {
				for card in hand_set {
					strat.0.insert(*card, vec![]);
					#[allow(non_snake_case)]
					let A = actions.len() as f64;
					for _ in &actions {
						strat.0.get_mut(&card).unwrap().push(1.0 / A);
					}
				}
			},
		}
		strat
	}

	// returns a permutation of the current strategy
	// should eventually use a smarter method than random step
	fn permutation_of(&self, delta : f64) -> Self {
		// todo: use None instead of empty?
		if self.0.is_empty() { return self.clone(); }
		let mut new = self.clone();
		use rand::{Rng, seq::IteratorRandom};
		let mut rng = rand::rng();
		// pick random card, get its weights
		let (_card, weights) = new.0.iter_mut().choose(&mut rng).unwrap();
		// increase/decrease one of the weights
		let idx = rng.random_range(0..weights.len());
		let sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
		weights[idx] = (weights[idx] + sign * delta).max(0.0);
		// renormalise
		let sum: f64 = weights.iter().sum();
		for w in weights.iter_mut() {
			*w /= sum;
		}
		new
	}
}

#[derive(Copy, Clone, Debug)]
struct LastChange {
	player_index : u8,
	bet : f64,
}

fn scale_f64_vec(a : &Vec<f64>, μ : f64) -> Vec<f64> {
	a.iter().map(|x| x * μ).collect()
}

fn add_f64_vec(a : &Vec<f64>, b : &Vec<f64>) -> Vec<f64> {
	a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn eq_of_range(a : &HashMap<Card, f64>, b : &HashMap<Card, f64>) -> f64 {
	// handling the degenerate case of either range being empty
	let a_has_range = a.values().any(|&x| x > ε);
	let b_has_range = b.values().any(|&x| x > ε);
	if a_has_range && !b_has_range { return 1.0; }
	if !a_has_range && !b_has_range { return 0.5; }
	if !a_has_range && b_has_range { return 0.0; }
	// calculate hand vs hand equity for each pair, weighted by occurence rate
	let mut equity = 0.0;
	let mut total_weight = 0.0;
	let mut played = false;
	// iterate over each hand pair
	for (a_card, a_weight) in a.iter() {
		for (b_card, b_weight) in b.iter() {
			if a_card == b_card { continue; } // blocked and impossible
			let pair_weight = a_weight * b_weight;
			if pair_weight < ε { // a card is absent from at least one player's side of pair
				continue;
			}
			played = true;
			if a_card > b_card {
				equity += pair_weight;
			} /* else if a card < b_card {
				equity[b] += pair_weight;
			} else {
				equity[a] += pair_weight / 2.0;
				equity[b] += pair_weight / 2.0;
			} */
			total_weight += pair_weight;
		}
	}
	// normalise result, such that eq(a, b) + eq(b, a) = 1
	if total_weight != 0.0 { equity /= total_weight; }
	// mutually blocked ranges split the pot
	if !played { equity = 0.5; }
	// hack for floating point precision
	if equity < ε { equity = 0.0 }
	if equity > 1.0-ε { equity = 1.0 }
	equity
}

// range vs range
// todo: board, multiway
fn r_v_r(a : &HashMap<Card, f64>, b : &HashMap<Card, f64>) -> Vec<f64> {
	vec![eq_of_range(a, b), eq_of_range(b, a)]
}

/*
struct GameTree {
	root: Node
}

impl GameTree {
	pub fn new(pot: f64, actionset: &ActionSet, deck : Vec<Card>) -> Self {
	}

	pub fn ev(&self) -> Vec<f64> {
	}
}
*/

#[derive(Debug, Clone)]
struct Node {
	// tree: &GameTree,
	pot : f64,
	ev : f64, // TODO: remove this. .ev() is the real interface, this is diagnostic only
	player_index : u8,
	action_change : Option<LastChange>,
	players : Vec<Player>,
	actions : Option<Vec<Action>>,
	strategy : Strategy,
	// current_seat : usize,
	// active_seats : Vec<u8>,
	// players : HashMap<u8, Player>,
	// history : History,
	root : *const Node,
	children : Vec<Node>,
}

impl Node {

	pub fn init_uniform(pot : f64, actionset : &ActionSet, deck : Vec<Card>) -> Box<Self> {
		let mut root = Box::new(Self::new(
			pot,
			0,
			Strategy::init(actionset, 0, 0, &deck),
			actionset.actions_at(0, 0),
			None,
			Player::initlist(2, deck),
			std::ptr::null(),
		));
		// todo: figure out if there's a better way to pass a reference to itself from root to children
		root.root = &*root as *const Node;
		root.populate_children_uniform(actionset, None, 0, 0);
		root
	}

	// no exploitation, just playing the tree as it is
	// returns the EV of each player
	pub fn ev(&self) -> Vec<f64> {
		// end recursion at leaf
		if self.actions == None {
			// folded out
			if self.players.len() == 1 {
				return match self.players[0].seat {
					0 => vec![self.pot, 0.0],
					1 => vec![0.0, self.pot],
					_ => panic!(),
				};
			} else { // showdown
				return scale_f64_vec(&r_v_r(&self.players[0].range, &self.players[1].range), self.pot);
			}
		}
		let mut ev : Vec<f64> = vec![0.0; 2];
		let mut out_prob : Vec<f64> = vec![];
		for (i, action) in self.actions.as_ref().unwrap().iter().enumerate() {
			out_prob.push(
				self.strategy.0.iter()
					// all the outward edges for an action, irrespective of card
					.map(|(card, weights)|
						weights[i]
						  *
						self.players[self.player_index as usize].range.get(card).unwrap()
					)
					.sum::<f64>()
					// divided by the total range, to normalise the vector
					  /
					self.players[self.player_index as usize].range.iter()
						.map(|(_, weight)| *weight)
						.sum::<f64>()
			);
			// subtract from EV for bets performed
			// TODO: bet committment
			match action {
				Action::Bet(bet) => ev[self.player_index as usize] -= bet * out_prob[i],
				Action::Call => ev[self.player_index as usize] -= self.action_change.unwrap().bet * out_prob[i],
				_ => (),
			}
		}
		for (i, child) in self.children.iter().enumerate() {
			ev = add_f64_vec(&ev, &scale_f64_vec(&child.ev(), out_prob[i]));
		}
		/* {
			println!("strategy: {:?}", self.strategy);
			println!("action_change: {:?}", self.action_change);
			println!("actions: {:?}", self.actions);
			println!("players: {:?}", self.players);
			println!("out_weights: {:?}\nev{:?}", out_prob, ev);
		} */
		ev
	}

	// creates a subtree with uniform actions taken
	fn populate_children_uniform(&mut self, actionset : &ActionSet, mut action_change : Option<LastChange>, mut street : u8, mut bet_level : u8) {
		if self.players.len() == 1 {
			return;
		}
		match action_change {
			Some(bet) => {
				if bet.player_index == self.player_index {
					street += 1;
					bet_level = 0;
					action_change = None;
				}
			},
			None => {
				action_change = Some(LastChange {player_index: 0, bet: 0.0});
			},
		}
		// useful for making the branches
		let n_player_index = if usize::from(self.player_index+1) < self.players.len() {self.player_index + 1} else {0};
		let deck = self.players[n_player_index as usize].range.keys().cloned().collect();
		let num_actions = {
			match actionset.actions_at(street, bet_level) {
				None => 0,
				Some(actions)=> actions.len(),
			}
		};
		let mut n_players = self.players.clone();
		for (_, j) in n_players[self.player_index as usize].range.iter_mut() {
			(*j) /= num_actions as f64;
		}
		// let n_actions = actionset.actions_at(street, bet_level);
		let mut new_children : Vec<(f64, u8, Option<LastChange>, u8, u8, Vec<Player>)> = Vec::new();
		if let Some(actions) = actionset.actions_at(street, bet_level) {
		for action in actions {
			match action {
				Action::Check => {
					// check-check has an initialised strategy and actions, have a "next player's actions" concept
					new_children.push(
						(
							self.pot,
							n_player_index,
							action_change,
							street,
							bet_level,
							n_players.clone(),
						)
					);
				}
				Action::Bet(value) => {
					let bet_level = bet_level+1;
					new_children.push(
						(
							self.pot+value,
							n_player_index,
							Some( LastChange {
								player_index: self.player_index,
								bet :1.0
							} ),
							street,
							bet_level,
							n_players.clone(),
						)
					);
				}
				Action::Fold => {
					let mut n_players = n_players.clone();
					n_players.remove(self.player_index as usize);
					let street = street+1; // todo: multiway
					let bet_level = 0;
					new_children.push(
						(
							self.pot,
							n_player_index,
							action_change,
							street,
							bet_level,
							n_players,
						)
					);
				}
				Action::Call => {
					let street = street+1;
					new_children.push(
						(
							self.pot+1.0,
							n_player_index,
							action_change,
							street,
							bet_level,
							n_players.clone(),
						)
					);
				}
			}
		}
		}
		let mut i : usize = 0;
		for (n_pot, n_player_index, n_action, n_street, n_bet_level, n_players) in new_children {
			self.children.push(
				Self::new(
					n_pot,
					n_player_index,
					Strategy::init(actionset, n_street, n_bet_level, &deck),
					actionset.actions_at(n_street, n_bet_level), // todo: proper multi-way logic
					n_action,
					n_players,
					self.root
				)
			);
			self.children[i].populate_children_uniform(actionset, n_action, n_street, n_bet_level);
			i += 1;
		}
		// a root node has no actions
		for child in &mut self.children {
			if child.children.len() == 0 {
				child.actions = None;
				child.strategy = Strategy(HashMap::new());
			}
		}
		for child in &mut self.children {
			child.ev = child.ev()[child.player_index as usize];
		}
		self.ev = self.ev()[self.player_index as usize];
	}

	fn new(pot : f64, player_index : u8, strategy : Strategy, actions : Option<Vec<Action>>, action_change : Option<LastChange>, players : Vec<Player>, root : *const Node) -> Self {
		Self {
			pot,
			ev : f64::MIN,
			player_index,
			action_change,
			players,
			actions,
			strategy,
			root,
			children : Vec::new(),
		}
	}

	fn set_strategy(&mut self, new_strat : &Strategy) {
		self.strategy = new_strat.clone();
		self.update_child_ranges(self.player_index as usize);
	}

	// update the subtree to use the new strategy for range calculations for the target player
	fn update_child_ranges(&mut self, target: usize) {
		if self.children.is_empty() { return; }
		for (index, child) in self.children.iter_mut().enumerate() {
			// skip if it's folded out, because the last player's range has no effect
			// TODO: handle players properly  for multiway allin, hashmap<player> instead of vec
			if child.players.len() == 1 { continue; }
			// set child range to this range, before factoring probability of moving there
			child.players[target].range = self.players[target].range.clone();
			if self.player_index as usize == target {
				let out_weight_into_child: Vec<(&Card, f64)> = self.strategy.0.iter()
					.map(|(card, weights) : (&Card, &Vec<f64>)| (card, weights[index]))
					.collect();
				for (card, out_weight) in out_weight_into_child {
					*child.players[target].range.get_mut(card).unwrap() *= out_weight;
				}
			}
			// cascade the change down
			child.update_child_ranges(target);
			// println!("\nout_weights: {:?}\nthis_range: {:?}\nchild_range: {:?}", &out_weights, &this_range, &child.players[self.player_index as usize].range);
		}
	}

	// fn update_child_ranges_helper(&mut self, target: usize) {
	// }

	pub fn counterexploitative_iteration(&mut self, target : usize, δ : f64) {
		if target == self.player_index as usize {
			// a future implementation might have self.try_ev(try_range)
			// save the previous state
			let old_ev = self.ev_after_exploitation();
			let old_strategy = self.strategy.clone();
			// try a new state
			let try_strategy = self.strategy.permutation_of(δ);
			self.set_strategy(&try_strategy);
			// revert if new strategy is worse
			let new_ev = self.ev_after_exploitation();
			if new_ev < old_ev {
				self.set_strategy(&old_strategy);
			}
		}
		for child in &mut self.children {
			child.counterexploitative_iteration(target, δ);
		};
		self.ev = self.ev_of_current(); // TODO: this is diagnostic and not accurate, remove/fix
	}

	// returns the EV of this node for the current player, using the global strategy
	fn ev_of_current(&self) -> f64 {
		return self.ev()[self.player_index as usize];
	}

	// returns the EV of the player with the current action if the opponent is maximally exploitative
	// EV is for the root, not the node itself
	fn ev_after_exploitation(&self) -> f64 {
		// todo: figure out if there's a better way to pass a reference to itself from root to children
		let mut exploit_tree = unsafe { (*self.root).clone() };
		let exploiter = if self.player_index == 1 { 0 } else { 1 };
		let max = 200;
		for i in 0..max {
			exploit_tree.exploitative_iteration(exploiter, iter_delta(i, max));
		}
		exploit_tree.ev()[self.player_index as usize]
	}

	fn exploitative_iteration(&mut self, exploiter : usize, δ : f64) {
		// a future implementation might have self.try_ev(try_range)
		if exploiter == self.player_index as usize {
			// save the previous state
			let old_ev = self.ev_of_current();
			let old_strategy = self.strategy.clone();
			// try a new state
			let try_strategy = self.strategy.permutation_of(δ);
			self.set_strategy(&try_strategy);
			// revert if new strategy has lower EV
			let new_ev = self.ev_of_current();
			if new_ev < old_ev {
				self.set_strategy(&old_strategy);
			}
		}
		for child in &mut self.children {
			child.exploitative_iteration(exploiter, δ);
		};
	}

}

use rs_poker::core::{Card, Suit, Value};

fn iter_delta(i : i32, max : i32) -> f64 {
	if i < 3*max/4 {
		0.2
	} else if i < 7*max/8 {
		0.1
	} else {
		0.03
	}
}

fn main() {
	let configjson = env::args().nth(1).expect("action config json required");
	let actionset = parse_config(&configjson).unwrap();
	// println!("{:#?}", actionset);
	let card_set = vec![
		Card::new(Value::Ace, Suit::Diamond),
		Card::new(Value::King, Suit::Diamond),
		Card::new(Value::Queen, Suit::Diamond),
	];
	let mut root = Node::init_uniform(2.0, &actionset, card_set);
	let T = 1000;
	let rounds = 1;
	let players = 2;
	for round in 0..rounds {
		if round % 1 == 0 {
			println!("player 0 ev: {}\nev while exploited: {}", root.ev_of_current(), root.ev_after_exploitation());
		}
		for i in 0..T {
			// root.counterexploitative_iteration(0, iter_delta(i, T));
			root.exploitative_iteration(1, iter_delta(i, T));
		}
		// for player in 0..players {
			// for i in 0..T {
			// 	root.counterexploitative_iteration(player, iter_delta(i, T));
			// }
			// for i in 0..T {
			// 	root.exploitative_iteration((player + 1 % 1), iter_delta(i, T));
			// }
		// }
	}
	println!("player 0 ev: {}\nev while exploited: {}", root.ev_of_current(), root.ev_after_exploitation());
	println!("{:#?}", root);
}

#[cfg(test)]
mod tests {
	use super::*;

	// helper functions
	fn akq_deck() -> Vec<Card> {
		vec![
			Card::new(Value::Ace, Suit::Diamond),
			Card::new(Value::King, Suit::Diamond),
			Card::new(Value::Queen, Suit::Diamond),
		]
	}

	fn is_zero(float : f64) -> bool {
		float.abs() < ε
	}

	fn random_range(cards : &Vec<Card>) -> HashMap<Card, f64>{
		let mut range : HashMap<Card, f64> = HashMap::new();
		use rand::Rng;
		let mut rng = rand::rng();
		for card in cards {
			let weight = rng.random();
			if rand::random_bool(1.0 / 3.0) { // 1/3 chance to zero it out
				range.insert(*card, 0.0);
			} else {
				range.insert(*card, weight);
			}
		}
		range
	}

	fn fixed_akq_range(ace : f64, king : f64, queen : f64) -> HashMap<Card, f64>{
		let mut range : HashMap<Card, f64> = HashMap::new();
		range.insert(Card::new(Value::Ace, Suit::Diamond), ace);
		range.insert(Card::new(Value::King, Suit::Diamond), king);
		range.insert(Card::new(Value::Queen, Suit::Diamond), queen);
		range
	}

	mod equity {
		use super::*;

		#[test]
		fn equities_sum_to_1() {
			let card_set = akq_deck();
			for _ in 0..1000 {
				let range0 = random_range(&card_set);
				let range1 = random_range(&card_set);
				let equity_sum = r_v_r(&range0, &range1).iter().sum::<f64>();
				let abs_diff = (equity_sum - f64::from(1.0)).abs();
				println!("range0: {:?}\nrange1: {:?}\nequity: {:?}", &range0, &range1, r_v_r(&range0, &range1));
				assert!(abs_diff < ε, "expected equities to sum to 1, they differ from 1 by {}", abs_diff);
			}
		}

		#[test]
		fn known_equities() {
			let range0 = fixed_akq_range(0.0, 0.1, 0.0);
			let range1 = fixed_akq_range(0.1, 0.0, 0.1);
			let eq = r_v_r(&range0, &range1);
			assert!(is_zero(eq[0] - 0.5));
			assert!(is_zero(eq[1] - 0.5));

			let range0 = fixed_akq_range(0.2, 0.1, 0.7);
			let range1 = fixed_akq_range(0.1, 0.5, 0.1);
			let eq = r_v_r(&range0, &range1);
			assert!(is_zero(eq[0] - 0.23214288));
			assert!(is_zero(eq[1] - 0.76785712));

			let range0 = fixed_akq_range(0.2, 0.1, 0.0);
			let range1 = fixed_akq_range(0.1, 0.5, 0.0);
			let eq = r_v_r(&range0, &range1);
			assert!(is_zero(eq[0] - 0.90909094));
			assert!(is_zero(eq[1] - 0.0909091));
		}

		#[test]
		fn equal_ranges_split_pot() {
			let range0 = fixed_akq_range(0.6, 0.2, 0.9);
			let range1 = fixed_akq_range(0.6, 0.2, 0.9);
			let eq = r_v_r(&range0, &range1);
			assert!(is_zero(eq[0] - 0.5));
			assert!(is_zero(eq[1] - 0.5));

			let range0 = fixed_akq_range(0.5, 0.5, 0.5);
			let range1 = fixed_akq_range(1.0, 1.0, 1.0);
			let eq = r_v_r(&range0, &range1);
			assert!(is_zero(eq[0] - 0.5));
			assert!(is_zero(eq[1] - 0.5));

			let range0 = fixed_akq_range(0.3, 0.0, 0.2);
			let range1 = fixed_akq_range(0.3, 0.0, 0.2);
			let eq = r_v_r(&range0, &range1);
			assert!(is_zero(eq[0] - 0.5));
			assert!(is_zero(eq[1] - 0.5));
		}

		#[test]
		fn empty_ranges_split_pot() {
			let range0 : HashMap<Card, f64> = HashMap::new();
			let range1 = fixed_akq_range(0.0, 0.0, 0.0);
			let eq = r_v_r(&range0, &range1);
			assert_eq!(eq[0], eq[1]);
		}

		#[test]
		fn mutually_blocked_ranges_split_pot() {
			let range0 = fixed_akq_range(0.3, 0.0, 0.0);
			let range1 = fixed_akq_range(0.9, 0.0, 0.0);
			let eq = r_v_r(&range0, &range1);
			assert_eq!(eq[0], eq[1]);
		}

		#[test]
		fn one_plays_one_wins() {
			let range0 = fixed_akq_range(0.3, 0.0, 0.0);
			let range1 = fixed_akq_range(0.0, 0.0, 0.0);
			let eq = r_v_r(&range0, &range1);
			assert_eq!(eq[0], 1.0);
		}

		#[test]
		fn dominator_wins() {
			let range0 = fixed_akq_range(0.3, 0.5, 0.0);
			let range1 = fixed_akq_range(0.0, 0.7, 0.8);
			let eq = r_v_r(&range0, &range1);
			assert_eq!(eq[0], 1.0);
		}
	}

	mod node {
		use super::*;

		fn uniform_akq_halfstreet_tree() -> Box<Node> {
			let action_set = ActionSet(vec![StreetActions {
				open: vec![Action::Check, Action::Bet(1.0)],
				facing_bet: vec![
					vec![Action::Check, Action::Bet(1.0)]
				],
			}]);
			// TODO: random pot size [1.0, 10.0]
			let root = Node::init_uniform(1.0, &action_set, akq_deck());
			root
		}

		fn permute_tree(node: &mut Node) {
			for _ in 0..4 {
				node.set_strategy(&node.strategy.permutation_of(0.2));
			}
			for child in &mut node.children {
				permute_tree(child);
			}
		}

		fn random_akq_halfstreet_tree() -> Box<Node> {
			let mut root = uniform_akq_halfstreet_tree();
			permute_tree(&mut *root);
			root
		}
			/*let rand_into = rng.random::<f64>();
			let mut seen_weight = 0.0;
			let mut chosen_action_index: usize = 100;
			for (i, weight) in node.strategy.0
				.get(&chosen_card)
				.unwrap()
				.iter()
				.enumerate() {
					seen_weight += weight;
					if rand_into < seen_weight {
						chosen_action_index = i;
					}
			}*/

		fn monte_carlo_round(node: &Node, target: usize) -> f64 {
			if node.actions == None {
				return node.ev()[target]; // assume that showdown EV is valid
			}
			use rand::Rng;
			let mut rng = rand::rng();
			let range = &node.players[node.player_index as usize].range;
			// choose card from range
			let rand_into = rng.random::<f64>() * range.values().sum::<f64>();
			let mut seen_weight = 0.0;
			let mut chosen_card = Card::new(Value::Jack, Suit::Spade); // junk value
			for (card, weight) in range {
				seen_weight += weight;
				if rand_into < seen_weight {
					chosen_card = *card;
					break;
				}
			}
			// choose action from strategy at card
			use rand::prelude::*;
			use rand::distr::weighted::WeightedIndex;
			let chosen_action_index = WeightedIndex::new(
				node.strategy.0.get(&chosen_card).unwrap()
			).unwrap().sample(&mut rand::rng());
			// bet/call have a cost TODO: pot commitment and last action
			let price = if node.player_index as usize != target { 0.0 }
				else {
					match node.actions.as_ref().unwrap()[chosen_action_index as usize] {
						Action::Bet(bet) => -1.0*bet,
						Action::Call => -1.0, // todo: bet size and invested
						_ => 0.0,
					}
			};
			let ev = price + monte_carlo_round(&node.children[chosen_action_index as usize], target);
			ev
		}

		// monte carlo is used for testing because it is independent to, and is simpler than the analytic weighted sum used by the nodes
		// it is not used in the real game due to inefficiency
		// ev refers to the turn player only
		fn monte_carlo_ev(node: &Node) -> f64 {
			let mut avg_ev = 0.0;
			let δ = 0.00001;
			let mut progress = 0.0;
			while progress < 1.0 {
				avg_ev += δ * monte_carlo_round(&node, node.player_index as usize);
				progress += δ;
			}
			avg_ev
		}

		/*fn random_node(player : u8, actions : Option<Vec<Action>>, cards : &Vec<Card>, action_change : Option<LastChange>, players : Vec<Player>) -> Node {
			let pot = 1.0;
			let player_index = 0;
			let mut strategy : Self = Strategy(HashMap::new());
			match actions {
				None => (),
				Some(actions) => {
					for card in cards {
						let mut weights: Vec<f64> = rand::random_iter().take(actions.len()).collect();
						let sum = weights.sum();
						&weights.iter_mut().map(|&x| x /= sum);
						strategy.0.insert(*card, weights);
					}
				},
			}
			Node::new(pot, player_index, strategy, actions, action_change, players)
		}*/

		/*#[test]
		fn ev_recurses_properly() {
			let mut players = vec![];
			for i in 0..2 {
				players.push(
					Player {
						seat: i as usize,
						// stack: rng.random() * 10.0,
						range: random_range(),
					}
				)
			}
			let open = Some(vec![Action::Check, Action::Bet(1.0)]);
			let defend = Some(vec![Action::Check, Action::Call]);
			let root = random_node(0, open, &akq_deck(), None, players);
		}*/

		#[test]
		fn exploitative_iteration_doesnt_decrease_monte_carlo_ev() {
			// let mut root = uniform_akq_halfstreet_tree();
			for _ in 0..50 {
				let mut root = random_akq_halfstreet_tree();
				let old_ev = monte_carlo_ev(&root);
				root.exploitative_iteration(0, 0.1);
				let new_ev = monte_carlo_ev(&root);
				assert!(old_ev <= new_ev + 0.01, "old ev:{old_ev}, new ev:{new_ev}");
				for child in &mut root.children {
					let old_ev = monte_carlo_ev(&child);
					child.exploitative_iteration(1, 0.1);
					let new_ev = monte_carlo_ev(&child);
					assert!(old_ev <= new_ev + 0.01, "old ev:{old_ev}, new ev:{new_ev}");
				}
			}
		}

		#[test]
		fn exploitative_iteration_doesnt_decrease_self_ev() {
			// let mut root = uniform_akq_halfstreet_tree();
			for _ in 0..10000 {
				let mut root = random_akq_halfstreet_tree();
				let old_ev = root.ev()[root.player_index as usize];
				// let old_state = root.clone();
				root.exploitative_iteration(0, 0.1);
				let new_ev = root.ev()[root.player_index as usize];
					// println!("before:\n{:?}\nafter\n{:?}\nold ev:{old_ev}, new ev:{new_ev}", &old_state, &root);
				assert!(old_ev <= new_ev + 0.01, "old ev:{old_ev}, new ev:{new_ev}");
				for child in &mut root.children {
					let old_ev = child.ev()[child.player_index as usize];
					// let old_state = child.clone();
					child.exploitative_iteration(1, 0.1);
					let new_ev = child.ev()[child.player_index as usize];
					// println!("before:\n{:?}\nafter\n{:?}\nold ev:{old_ev}, new ev:{new_ev}", &old_state, &root);
					assert!(old_ev <= new_ev + 0.01, "old ev:{old_ev}, new ev:{new_ev}");
				}
			}
		}

		// helper function for recursion
		fn recursive_ev_match(node: &Node) {
			let mc_ev = monte_carlo_ev(node);
			let self_ev = node.ev()[node.player_index as usize];
			let diff = (mc_ev - self_ev).abs();
			if diff > 0.01 {
				println!("state\n{:?}\n", &node);
				panic!("diff: {diff}, self ev:{self_ev}, monte carlo ev:{mc_ev}");
			}
			// assert!(diff < ε, "diff: {diff}, self ev:{self_ev}, monte carlo ev:{mc_ev}");
			for child in &node.children {
				recursive_ev_match(&child);
			}
		}

		#[test]
		fn self_ev_matches_numerical_approximation() {
			for _ in 0..100 {
				let root = random_akq_halfstreet_tree();
				recursive_ev_match(&root);
			}
		}

	}

}
