/*
   The fundamental elements of the game are these:

   • There is a three-card deck, containing an Ace, a King, and a Queen.
   • The players are each dealt one card without replacement and there is an initial ante.
   • There is a round of betting, after which there is a showdown (if neither player folds). In
   the showdown, the high card wins.
*/

// I'm doing math, this variable standard is ridiculous if T violates it
#![allow(non_snake_case)]
#![allow(confusable_idents)] // α DOES NOT look like a

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::BufReader;

// use std::fmt::Debug;

mod PureCFR {
	use crate::{Game, Node, Strategy, Action};
	use crate::{Hand, HashMap};

	/// Runs CFR on the provided Game tree for T iterations, then writes the solution to game.
	pub fn solve(game: &mut Game, T: u16) {
		let mut regrets = GameRegrets::for_game(&game);
		let mut strategy_sum = GameStrategySum::for_game(&game);
		for i in 1..=T {
			cfr_iteration(game, &mut regrets, &mut strategy_sum, i);
		}
		avg_strategy(game, &strategy_sum, T);
		// diagnostic: remove later
		game.update_evs();
		print_game_with_regrets(&game, &regrets);
	}

	fn cfr_iteration(game: &mut Game, regrets: &mut GameRegrets, strategy_sum: &mut GameStrategySum, T: u16) {
		// read game, write to regrets
		update_regrets(game, regrets);
		// read regrets, write to game
		update_strategy(game, regrets, T);
		// add the new strategy to the cumulative strategy
		for (idx, strategy_acc) in &mut strategy_sum.0.iter_mut().enumerate() {
			for (hand, action_weights) in strategy_acc.0.iter_mut() {
				for (i, weight) in action_weights.iter_mut().enumerate() {
					*weight += game.0[idx].strategy.0.get(hand).unwrap()[i]
					  *
					game.0[idx].players[game.0[idx].player_index as usize].range.get(hand).unwrap();
				}
			}
		}
	}

	pub fn avg_strategy(game: &mut Game, strategy_sum: &GameStrategySum, rounds: u16) {
		for (idx, node) in &mut game.0.iter_mut().enumerate() {
			for (hand, action_weights) in node.strategy.0.iter_mut() {
				for (i, weight) in action_weights.iter_mut().enumerate() {
					*weight = strategy_sum.0[idx].0.get(hand).unwrap()[i] / rounds as f64;
				}
			}
		}
		for idx in 0..game.0.len() {
			game.update_child_ranges_at(idx.try_into().unwrap());
		}
		game.update_evs();
	}

	fn update_strategy(game: &mut Game, regrets: &GameRegrets, T: u16) {
		let T = 1;
		for i in 0..game.0.len() {
			if game.0[i].children.len() == 0 {
				continue;
			}
			for (hand, actions) in game.0[i].strategy.0.iter_mut() {
				let all_action_regret_sum = regrets.0[i].0.get(hand).unwrap().values()
					.map(|&v| v.max(0.0))
					.sum::<f64>() / T as f64;
				let num_actions = actions.len() as f64;
				for (idx, weight) in actions.iter_mut().enumerate() {
					let action_regret: f64 = regrets.0[i].0.get(hand).unwrap()
						.get(&game.0[i].actions.as_ref().expect("nonterminal")[idx])
						.unwrap() / T as f64;
					let action_regret_pos = action_regret.max(0.0);
					let new_weight = if all_action_regret_sum > 0.0 {
						action_regret_pos / all_action_regret_sum // its portion of the positive regrets
					} else {
						1.0 / num_actions // random choice
					};
					*weight = new_weight;
				}
			}
		}
		game.update_child_ranges_at(0);
	}

	fn update_regrets(game: &Game, regrets: &mut GameRegrets) {
		for i in 0..game.0.len() {
			append_regret(game, regrets, i.try_into().unwrap());
		}
	}

	fn append_regret(game: &Game, regrets: &mut GameRegrets, idx: u32) {
		// non-borrowing alias
		macro_rules! cur_node { () => { game.0[idx as usize] }; }
		macro_rules! cur_player { () => { cur_node!().players[cur_node!().player_index as usize] }; }
		if cur_node!().children.len() == 0 {
			return;
		};
		let real_evs: HashMap<_, _> = cur_player!().range.keys()
			.map(|hand| (hand, game.ev_of_hand_of_player_at(idx, *hand, cur_node!().player_index)))
			.collect();
		for hand in cur_player!().range.clone().keys() {
			let hand_regrets: &mut HashMap<Action, f64> =
				regrets.0[idx as usize].0.get_mut(hand).unwrap();
			for action in cur_node!().clone().actions.as_ref().expect("nonterminal").iter() {
				let u_i_try = game.ev_of_action_for_hand_at(idx, *hand, *action);
				let u_i_old = real_evs.get(hand).unwrap();
				// println!("{:?}, {:?}", u_i_new, u_i_old);
				// reach of the opponents' ranges to get to this node
				// TODO: figure out if blocking needs to be accounted for
				let opponents_iter = cur_node!().players.iter()
					.enumerate()
					.filter(|(i, _)| *i != cur_node!().player_index as usize)
					.map(|(_, player)| player);
				let π = opponents_iter.fold(1.0, |acc, opponent| {
					let unblocked_weights = &opponent.range.iter()
						.filter(|(opp_hand, _)| **opp_hand & *hand == Hand::new())
						.map(|(_, hand_weight)| hand_weight);
					let unblocked_sum: f64 = unblocked_weights.clone().sum();
					let count = unblocked_weights.clone().count() as f64;
					acc * (unblocked_sum / count)
				});
				let regret = π * (u_i_try - u_i_old);
				// let regret = u_i_new - u_i_old;
				*hand_regrets.get_mut(action).unwrap() += regret;
			}
		}
	}

	/// Prints the game tree with each node's regrets shown directly below it. (debugging)
	/// Uses `{:?}` (Debug formatting) for both the node and its regrets.
	pub fn print_game_with_regrets(game: &Game, regrets: &GameRegrets) {
		let nodes = &game.0;
		let regrets_slice = &regrets.0;

		if nodes.is_empty() {
			println!("Empty game tree");
			return;
		}

		// Recursive helper that prints a node and its subtree.
		fn print_node(nodes: &[Node], regrets: &[Regrets], idx: usize, depth: usize) {
			let indent = "n  ".repeat(depth);
			let node = &nodes[idx];

			// Print the node using its Debug impl
			println!("{}{:#?}", indent, node);
			// Print the corresponding regrets using its Debug impl, indented one level
			println!("{}  regrets: {:#?}", indent, regrets[idx]);

			// Recurse into children (preserve order, but sort if desired)
			for &child_idx in &node.children {
				print_node(nodes, regrets, child_idx as usize, depth + 1);
			}
		}

		print_node(nodes, regrets_slice, 0, 0);
	}

	#[derive(Debug)]
	struct Regrets(HashMap<Hand, HashMap<Action, f64>>);
	impl Regrets {
		fn new() -> Self {
			Regrets(HashMap::new())
		}
	}

	#[derive(Debug)]
	pub struct GameStrategySum(Box<[Strategy]>); //TODO: this is not a real strategy, its invariants are different
	impl GameStrategySum {
		pub fn for_game(game: &Game) -> Self {
			let mut strategy_vec = vec![];
			strategy_vec.reserve(game.0.len());
			// populate the arena
			for node in game.0.iter() {
				strategy_vec.push(node.strategy.clone());
				for policy in strategy_vec.last_mut().unwrap().0.values_mut() {
					for scalar in policy {
						*scalar = 0.0;
					}
				}
			}
			GameStrategySum(strategy_vec.into_boxed_slice())
		}
	}

	#[derive(Debug)]
	pub struct GameRegrets(Box<[Regrets]>);
	impl GameRegrets {
		pub fn for_game(game: &Game) -> Self {
			let mut regret_vec = vec![];
			regret_vec.reserve(game.0.len());
			// populate the arena
			for node in game.0.iter() {
				if let Some(actions) = &node.actions {
					let mut regrets_init = Regrets::new();
					for hand in node.strategy.0.keys() {
						regrets_init.0.insert(hand.clone(), HashMap::new());
						for action in actions.clone() {
							regrets_init.0.get_mut(hand).unwrap().insert(action, 0.0);
						}
					}
					regret_vec.push(regrets_init);
				} else {
					regret_vec.push(Regrets::new());
				}
			}
			GameRegrets(regret_vec.into_boxed_slice())
		}
	}
}

#[derive(Debug, Clone)]
pub struct Game(Box<[Node]>);
impl Game {
	pub fn new_uniform(cfg: GameConfig, deck: Vec<Card>) -> Self {
		// let len = 0;
		// let mut nodes_uninit : Box<[MaybeUninit<Node>]> = Box::new_uninit_slice(len);
		let mut nodes_vec = vec![];
		// nodes_vec.reserve(len);
		// enumerate hands
		let hands = deck
			.iter()
			.map(|card| Hand::new_with_cards(vec![*card]))
			.collect();
		// create initial state of the game
		let root = Node::new(
			cfg.ante,
			0,
			Strategy::init(&cfg.actionset, 0, 0, &hands),
			cfg.actionset.actions_at(0, 0),
			None,
			Player::initlist(2, &hands),
		);
		nodes_vec.push(root);
		let mut new_idx = 1; // where to start inserting new nodes
		let mut iteration_stack = vec![(0, None, 0, 0)];
		while !iteration_stack.is_empty() {
			let (idx, action_change, street, bet_level) = iteration_stack.pop().unwrap();
			let children_added = Self::init_children_at(
				idx,
				&mut new_idx,
				&mut nodes_vec,
				&cfg.actionset,
				action_change,
				street,
				bet_level,
			);
			if let Some(children) = children_added {
				for state in children.iter().rev() {
					iteration_stack.push(*state);
				}
			}
		}
		for node in &mut nodes_vec.iter_mut() {
			if node.children.is_empty() {
				node.actions = None;
				node.strategy = Strategy(HashMap::new());
			}
		}
		// nodes_uninit[idx as usize].write(root);
		// SAFETY: every node is now initialised
		// let nodes_init = unsafe { nodes_uninit.assume_init() };
		// Game(nodes_init)
		Game(nodes_vec.into_boxed_slice())
	}

	fn init_children_at(
		idx: u32,
		new_idx: &mut u32,
		game_arena: &mut Vec<Node>,
		actionset: &ActionSet,
		mut action_change: Option<LastChange>,
		mut street: u8,
		mut bet_level: u8,
	) -> Option<Vec<(u32, Option<LastChange>, u8, u8)>> {
		// let mut cur_node = &mut game_arena[idx as usize];
		// non-borrowing alias
		macro_rules! cur_node {
			() => {
				game_arena[idx as usize]
			};
		}
		if cur_node!().players.len() == 1 {
			return None;
		}
		match action_change {
			Some(bet) => {
				if bet.player_index == cur_node!().player_index {
					street += 1;
					bet_level = 0;
					action_change = None;
				}
			}
			None => {
				action_change = Some(LastChange {
					player_index: 0,
					bet: 0.0,
				});
			}
		}
		// useful for making the branches
		let n_player_index =
			if usize::from(cur_node!().player_index + 1) < cur_node!().players.len() {
				cur_node!().player_index + 1
			} else {
				0
			};
		let deck = cur_node!().players[n_player_index as usize]
			.range
			.keys()
			.cloned()
			.collect();
		let num_actions = {
			match actionset.actions_at(street, bet_level) {
				None => 0,
				Some(actions) => actions.len(),
			}
		};
		let mut n_players = cur_node!().players.clone();
		for (_, j) in n_players[cur_node!().player_index as usize]
			.range
			.iter_mut()
		{
			(*j) /= num_actions as f64;
		}
		// let n_actions = actionset.actions_at(street, bet_level);
		let mut new_children: Vec<(f64, u8, Option<LastChange>, u8, u8, Vec<Player>)> = Vec::new();
		if let Some(actions) = actionset.actions_at(street, bet_level) {
			for action in actions {
				match action {
					Action::Check => {
						// check-check has an initialised strategy and actions, have a "next player's actions" concept
						new_children.push((
							cur_node!().pot,
							n_player_index,
							action_change,
							street,
							bet_level,
							n_players.clone(),
						));
					}
					Action::Bet(value) => {
						let bet_level = bet_level + 1;
						new_children.push((
							cur_node!().pot + value,
							n_player_index,
							Some(LastChange {
								player_index: cur_node!().player_index,
								bet: 1.0,
							}),
							street,
							bet_level,
							n_players.clone(),
						));
					}
					Action::Fold => {
						let mut n_players = n_players.clone();
						n_players.remove(cur_node!().player_index as usize);
						let street = street + 1; // todo: multiway
						let bet_level = 0;
						new_children.push((
							cur_node!().pot,
							n_player_index,
							action_change,
							street,
							bet_level,
							n_players,
						));
					}
					Action::Call => {
						let street = street + 1;
						new_children.push((
							cur_node!().pot + 1.0, //TODO: action_change - pot_investment
							n_player_index,
							action_change,
							street,
							bet_level,
							n_players.clone(),
						));
					}
				}
			}
		}
		let mut child_iters = vec![];
		for (n_pot, n_player_index, n_action, n_street, n_bet_level, n_players) in new_children {
			// create the child nodes in the arena
			cur_node!().children.push(*new_idx);
			game_arena.push(Node::new(
				n_pot,
				n_player_index,
				Strategy::init(actionset, n_street, n_bet_level, &deck),
				actionset.actions_at(n_street, n_bet_level), // todo: proper multi-way logic
				n_action,
				n_players,
			));
			// indicate to the caller that the new child was made, with such state
			child_iters.push((*new_idx, n_action, n_street, n_bet_level));
			*new_idx += 1;
		}
		Some(child_iters)
	}

	/// Returns the EV if the current player has X hand and takes Y action
	pub fn ev_of_action_for_hand_at(&self, idx: u32, hand: Hand, action: Action) -> f64 {
		// for a non-borrowing alias
		macro_rules! cur_node { () => { self.0[idx as usize] }; }
		// terminal node
		if cur_node!().children.is_empty() {
			return self.ev_of_hand_of_player_at(idx, hand, cur_node!().player_index);
		} else {
			let child_offset = cur_node!().actions.as_ref().expect("nonterminal").iter()
				.position(|cur_action| *cur_action == action).unwrap();
			let child_idx = cur_node!().children[child_offset];
			return self.ev_of_hand_of_player_at(child_idx, hand, cur_node!().player_index);
		}
	}

	/// Returns the EV of a hand at a node, belonging to any player, not scaled by their current range
	fn ev_of_hand_of_player_at(&self, idx: u32, hand: Hand, this_player: u8) -> f64 {
		// for a non-borrowing alias
		macro_rules! cur_node { () => { self.0[idx as usize] }; }
		if cur_node!().children.is_empty() { // terminal
			let opp_ranges: Vec<_> = cur_node!().players.iter()
				.enumerate()
				.filter(|(i, _)| *i != this_player as usize)
				.map(|(_, val)| &val.range)
				.collect();
			return cur_node!().pot * equity_of_hand_vs_ranges(hand, opp_ranges);
		} else { // nonterminal
			return cur_node!().children.iter()
				.enumerate()
				.map(|(i, child_idx)| (i, self.ev_of_hand_of_player_at(*child_idx, hand, this_player)))
				.map(|(i, child_ev)| child_ev * cur_node!().strategy.0.get(&hand).unwrap()[i])
				.sum::<f64>();
		}
	}

	fn update_evs(&mut self) {
		for i in 0..self.0.len() {
			self.0[i].ev = self.ev_at(i.try_into().unwrap())[self.0[i].player_index as usize];
		}
	}

	// fn set_strategy(&mut self, new_strat : &Strategy) {
	// 	self.strategy = new_strat.clone();
	// 	self.update_child_ranges(self.player_index as usize);
	// }

	/// Update the subtree to use the new strategy for range calculations.
	/// Used to restore invariants in children.
	fn update_child_ranges_at(&mut self, idx: u32) {
		// non-borrowing alias
		macro_rules! cur_node { () => { self.0[idx as usize] }; }
		macro_rules! cur_player { () => { cur_node!().players[cur_node!().player_index as usize] }; }
		if cur_node!().children.is_empty() {
			return;
		}
		for (index, &child_idx) in cur_node!().clone().children.iter().enumerate() {
			// skip if it's folded out, because the last player's range has no effect
			// TODO: handle players properly  for multiway allin, hashmap<player> instead of vec
			if self.0[child_idx as usize].players.len() == 1 { continue; }
			// set child range to this range, before factoring probability of moving there
			self.0[child_idx as usize].players[cur_node!().player_index as usize].range =
				cur_player!().range.clone();
			let out_weight_into_child: Vec<(Hand, f64)> = cur_node!().strategy.0.iter()
				.map(|(hand, weights): (&Hand, &Vec<f64>)| (hand.clone(), weights[index]))
				.collect();
			for (hand, out_weight) in &out_weight_into_child {
				*self.0[child_idx as usize].players[cur_node!().player_index as usize].range.get_mut(&hand).unwrap() *= out_weight;
			}
			// cascade the change down
			self.update_child_ranges_at(child_idx);
		}
	}

	/// returns the EV of each player at the designated node for the current global strategy
	pub fn ev_at(&self, idx: u32) -> Vec<f64> {
		// for a non-borrowing alias
		macro_rules! cur_node { () => { self.0[idx as usize] }; }
		macro_rules! cur_player { () => { cur_node!().players[cur_node!().player_index as usize] }; }
		// end recursion at leaf
		if cur_node!().actions == None {
			// folded out TODO: multiway match and seat reference
			if cur_node!().players.len() == 1 {
				return match cur_node!().players[0].seat {
					0 => vec![cur_node!().pot, 0.0],
					1 => vec![0.0, cur_node!().pot],
					_ => todo!(),
				};
			} else {
				// showdown
				return scale_f64_vec(
					&equities_of_ranges(vec![&cur_node!().players[0].range, &cur_node!().players[1].range]),
					cur_node!().pot,
				);
			}
		}
		let mut ev: Vec<f64> = vec![0.0; 2];
		let mut out_prob: Vec<f64> = vec![];
		let range_total = cur_player!().range.iter()
			.map(|(_, weight)| *weight)
			.sum::<f64>();
		if range_total != 0.0 {
			for (i, action) in cur_node!().actions.as_ref().unwrap().iter().enumerate() {
				out_prob.push(
					cur_node!().strategy.0.iter()
						// all the outward edges for an action, irrespective of card
						.map(|(card, weights)|
							weights[i]
						  	  *
							cur_player!().range.get(card).unwrap()
						)
						.sum::<f64>()
						// divided by the total range, to normalise the vector
						  /
						range_total,
				);
				// subtract from EV for bets performed
				// TODO: bet committment for call/bet
				match action {
					Action::Bet(bet) => ev[cur_node!().player_index as usize] -= bet * out_prob[i],
					Action::Call => {
						ev[cur_node!().player_index as usize] -=
						cur_node!().action_change.unwrap().bet * out_prob[i]
					}
					_ => (),
				}
			}
		} else {
			out_prob = vec![0.0; cur_player!().range.len()];
		}
		for (i, &child) in cur_node!().children.iter().enumerate() {
			ev = add_f64_vec(&ev, &scale_f64_vec(&self.ev_at(child), out_prob[i]));
		}
		ev
	}

	/*
	// returns the EV of this node for the current player, using the global strategy
	fn ev_of_current(&self) -> f64 {
		return self.ev()[self.player_index as usize];
	}

	// returns the EV of this node for the current player, using the global strategy
	fn min_ev_of_current(&self) -> f64 {
		return self.ev_after_exploitation()[self.player_index as usize];
	}

	// returns the EVs of the current node if the opponent is maximally exploitative starting at the current node
	pub fn ev_after_exploitation(&self) -> Vec<f64> {
		// TODO: pure strategy best response
		let mut exploit_tree = self.clone();
		let exploited = if self.player_index == 1 { 0 } else { 1 };
		exploit_tree.best_response(exploited);
		exploit_tree.ev()
	}

	*/
}

/// Parameters for the configuration of the game.
#[derive(Deserialize)]
pub struct GameConfig {
	ante: f64,
	// players: u8
	actionset: ActionSet,
}

/// The options that can be taken at a street
#[derive(Debug, Clone, Deserialize)]
struct StreetActions {
	open: Vec<Action>,
	facing_bet: Vec<Vec<Action>>,
}

/// The options that can be taken in the game
#[derive(Debug, Deserialize)]
struct ActionSet(Vec<StreetActions>);
impl ActionSet {
	fn actions_at(&self, street: u8, bet_level: u8) -> Option<Vec<Action>> {
		// bet level of 0 means check, 1=>bet, 2=>3bet etc.
		if bet_level == 0 {
			if usize::from(street) < self.0.len() {
				return Some(self.0[street as usize].open.clone());
			}
		} else {
			if usize::from(street) < self.0.len() {
				if usize::from(bet_level - 1) < self.0[street as usize].facing_bet.len() {
					return Some(
						self.0[street as usize].facing_bet[usize::from(bet_level - 1)].clone(),
					);
				}
			}
		}
		return None;
	}
}

fn parse_config(path: &str) -> Result<GameConfig, Box<dyn std::error::Error>> {
	let file = File::open(path)?;
	let reader = BufReader::new(file);
	let cfg = serde_json::from_reader(reader)?;
	Ok(cfg)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
	Check,
	Call,
	Fold,
	Bet(f64), // pot%
	// Allin,
}

impl Serialize for Action {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		let s = match self {
			Action::Check => "Check".to_string(),
			Action::Call => "Call".to_string(),
			Action::Fold => "Fold".to_string(),
			Action::Bet(amount) => {
				if amount.is_nan() {
					"Bet(NaN)".to_string()
				} else {
					format!("Bet({})", amount)
				}
			}
		};
		serializer.serialize_str(&s)
	}
}

use std::str::FromStr;
impl<'de> Deserialize<'de> for Action {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		let s = String::deserialize(deserializer)?;

		match s.as_str() {
			"Check" => Ok(Action::Check),
			"Call" => Ok(Action::Call),
			"Fold" => Ok(Action::Fold),
			"Bet(NaN)" => Ok(Action::Bet(f64::NAN)),
			s if s.starts_with("Bet(") && s.ends_with(')') => {
				let inner = &s[4..s.len() - 1]; // Get content inside Bet(...)
				match f64::from_str(inner) {
					Ok(amount) => Ok(Action::Bet(amount)),
					Err(_) => Err(serde::de::Error::custom("Invalid bet amount")),
				}
			}
			//"AllIn" => Ok(Action::AllIn),
			_ => Err(serde::de::Error::custom("Invalid action")),
		}
	}
}

impl Eq for Action {}

impl std::hash::Hash for Action {
	fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
		match self {
			Self::Check => 0u8.hash(state),
			Self::Call => 1u8.hash(state),
			Self::Fold => 2u8.hash(state),
			Self::Bet(percentage) => {
				3u8.hash(state);
				// Hash the bit representation to handle f64 properly
				percentage.to_bits().hash(state);
			}
		}
	}
}

#[derive(Debug, Clone, Serialize)]
struct Player {
	seat: u8,
	range: HashMap<Hand, f64>,
	// street_committment: f64,
	// stack : f64,
}

impl Player {
	fn initlist(n: u8, hands: &Vec<Hand>) -> Vec<Player> {
		let mut players: Vec<Player> = Vec::new();
		for i in 0..n {
			players.push(Self::new(i, hands));
		}
		players
	}
	fn new(seat: u8, hands: &Vec<Hand>) -> Self {
		let mut range: HashMap<Hand, f64> = HashMap::new();
		for hand in hands {
			range.insert(*hand, 1.0);
		}
		Self { seat, range }
	}
}

#[derive(Debug, Clone, Serialize)]
struct Strategy(HashMap<Hand, Vec<f64>>);

impl Strategy {
	fn init(actionset: &ActionSet, street: u8, level: u8, hand_set: &Vec<Hand>) -> Self {
		let mut strat: Self = Strategy(HashMap::new());
		if let Some(actions) = actionset.actions_at(street, level) {
			for hand in hand_set {
				strat.0.insert(*hand, vec![]);
				#[allow(non_snake_case)]
				let A = actions.len() as f64;
				for _ in &actions {
					strat.0.get_mut(&hand).unwrap().push(1.0 / A);
				}
			}
		}
		strat
	}
}

#[derive(Copy, Clone, Debug, Serialize)]
struct LastChange {
	player_index: u8,
	/// How much you need invested to match the action change (is 0.0 for check)
	bet: f64,
}

fn scale_f64_vec(a: &Vec<f64>, μ: f64) -> Vec<f64> {
	a.iter().map(|x| x * μ).collect()
}

fn add_f64_vec(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
	a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[derive(PartialEq)]
enum Showdown {
	Win,
	Lose,
	Tie
}

/// Whether the first argument wins, loses or ties against the second
fn hand_vs_hand_akq(a: Hand, b: Hand) -> Showdown {
	let a_card = a.iter().nth(0);
	let b_card = b.iter().nth(0);
	if a_card > b_card {
		return Showdown::Win;
	} else {
		return Showdown::Lose;
	}
}

/// Equity of a Hand vs a series of Ranges (only at showdown)
fn equity_of_hand_vs_ranges(a: Hand, opponents: Vec<&HashMap<Hand, f64>>) -> f64 {
	let combinations = opponents.iter()
        .map(|range| range.iter())
        .multi_cartesian_product();
	let mut equity = 0.0;
	let mut total_weight = 0.0;
	// TODO: global oncecell<u8> for playercount/passing down as parameter should simplify this
	for combo in combinations {
		let opponents_blocked = combo.iter()
			.enumerate()
			.any(|(i, (hand1, _))| {
				combo[(i + 1)..]
					.iter()
					.any(|(hand2, _)| (**hand1 & **hand2) != Hand::new())
			});
		if opponents_blocked { continue; }
		let self_blocked = combo.iter().any(|(hand, _)| a & **hand != Hand::new());
		if self_blocked { continue; }
		let π = combo.iter().fold(1.0, |acc, (_, weight)| acc * *weight);
		total_weight += π;
		if combo.iter().all(|(hand, _)| hand_vs_hand_akq(a, **hand) == Showdown::Win) {
			equity += π;
		} else if combo.iter().all(|(hand, _)| hand_vs_hand_akq(a, **hand) != Showdown::Lose) { // account for split pots
			let ties = combo.iter()
				.filter(|(hand, _)| hand_vs_hand_akq(a, **hand) == Showdown::Tie)
				.count();
			if ties > 0 {
				equity += π / (1+ties) as f64;
			}
		}
	}
	if total_weight == 0.0 { return 0.0; }
	equity / total_weight
}

/// All ranges equity (showdown only)
/// sums to 1 unless there was no play, in which case it sums to 0
fn equities_of_ranges(ranges: Vec<&HashMap<Hand, f64>>) -> Vec<f64> {
	let combinations = ranges.iter()
        .map(|range| range.iter())
        .multi_cartesian_product();
	// calculate hand vs hand equity for each pair, weighted by occurence rate
	let mut equity = vec![0.0; ranges.len()];
	let mut total_weight = 0.0;
	// let mut total_weight = 0.0;
	for combo in combinations.into_iter() {
		let blocked = combo.iter()
			.enumerate()
			.any(|(i, (hand1, _))| {
				combo[(i + 1)..].iter()
					.any(|(hand2, _)| (**hand1 & **hand2) != Hand::new())
			});
		if blocked { continue; }
		let π = combo.iter().fold(1.0, |acc, (_, weight)| acc * *weight);
		total_weight += π;
		for (i, (hand, _)) in combo.iter().enumerate() {
			if combo.iter()
				.enumerate()
				.filter(|(j, (_, _))| i != *j)
				.map(|(_, (hand, _))| hand)
				.all(|opp_hand| hand_vs_hand_akq(**hand, **opp_hand) == Showdown::Win) {
				equity[i] += π;
			} else if combo.iter().all(|(opp_hand, _)| hand_vs_hand_akq(**hand, **opp_hand) != Showdown::Lose) { // account for split pots
				let ties = combo.iter()
					.filter(|(opp_hand, _)| hand_vs_hand_akq(**hand, **opp_hand) == Showdown::Tie)
					.count();
				if ties > 0 {
					equity[i] += π / (1+ties) as f64;
				}
			}
		}
	}
	if total_weight == 0.0 { return vec![0.0; ranges.len()]; }
	equity.iter().map(|x| *x / total_weight).collect()
}

/// The equity that the first range has against all the other ranges at showdown
fn equity_of_range(a: &HashMap<Hand, f64>, b: &HashMap<Hand, f64>) -> f64 {
	let mut equity = 0.0;
	let mut total_weight = 0.0;
	for (a_hand, a_weight) in a.iter() {
		for (b_hand, b_weight) in b.iter() {
			// blocked and impossible
			if *a_hand & *b_hand != Hand::new() {
				continue;
			}
			let pair_weight = a_weight * b_weight;
			// a card is absent from at least one player's side of pair
			if pair_weight == 0.0 {
			// if pair_weight < ε {
				continue;
			}
			match hand_vs_hand_akq(*a_hand, *b_hand) {
				Showdown::Win => equity += pair_weight,
				Showdown::Tie => equity += pair_weight/2.0,
				_ => (),
				// Showdown::Lose => equity[b] += pair_weight,
			}
			total_weight += pair_weight;
		}
	}
	// normalise result, such that eq(a, b) + eq(b, a) = 1
	// but if there was no play, equity is already 0
	if total_weight != 0.0 {
		equity /= total_weight;
	}
	// hacks for floating point precision
	// if equity < ε {
	// 	equity = 0.0
	// }
	// if equity > 1.0 - ε {
	// 	equity = 1.0
	// }
	equity
}

#[derive(Debug, Clone, Serialize)]
struct Node {
	pot: f64,
	/// will be removed and replaced by an EV arena
	ev: f64, // TODO: remove this. .ev() is the real interface, this is diagnostic only
	player_index: u8,
	action_change: Option<LastChange>,
	actions: Option<Vec<Action>>,
	strategy: Strategy,
	players: Vec<Player>,
	// current_seat : usize,
	// active_seats : Vec<u8>,
	// players : HashMap<u8, Player>,
	// players : Box<[Option<Player>]>
	// history : History,
	children: Vec<u32>,
}

impl Node {
	fn new(
		pot: f64,
		player_index: u8,
		strategy: Strategy,
		actions: Option<Vec<Action>>,
		action_change: Option<LastChange>,
		players: Vec<Player>,
	) -> Self {
		let temp = Self {
			pot,
			ev: f64::MIN,
			player_index,
			action_change,
			players,
			actions: actions.clone(),
			strategy,
			children: Vec::new(),
		};
		temp
	}
}

use rs_poker::core::{Card, Hand, Value, Suit};

fn main() {
	let configjson = env::args().nth(1).expect("action config json required");
	let config = parse_config(&configjson).unwrap();
	let card_set = vec![
		Card::new(Value::King, Suit::Diamond),
		Card::new(Value::Queen, Suit::Diamond),
		Card::new(Value::Jack, Suit::Diamond),
	];
	let mut game = Game::new_uniform(config, card_set);
	let iterations = 100;
	PureCFR::solve(&mut game, iterations);
	// println!("player 0 ev: {}\nev while exploited: {}", root.ev_of_current(), root.min_ev_of_current());
	// let json_string = serde_json::to_string_pretty(&root).unwrap();
	// println!("{}", json_string);
	// let _ = fs::write("out.json", json_string);
}

#[cfg(test)]
mod tests {
use super::*;

#[allow(non_upper_case_globals)]
const ε: f64 = 0.000001;

// helper functions
impl Strategy {
	// returns a permutation of the current strategy
	fn permutation_of(&self, delta : f64) -> Self {
		// todo: use None instead of empty?
		if self.0.is_empty() { return self.clone(); }
		let mut new = self.clone();
		use rand::{Rng, seq::IteratorRandom};
		let mut rng = rand::rng();
		// pick random card, get its weights
		let (_card, weights) : (_, &mut Vec<f64>) = new.0.iter_mut().choose(&mut rng).unwrap();
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

fn akq_deck() -> Vec<Card> {
	vec![
		Card::new(Value::Ace, Suit::Diamond),
		Card::new(Value::King, Suit::Diamond),
		Card::new(Value::Queen, Suit::Diamond),
	]
}

fn is_equal(a: f64, b:f64) -> bool {
	(a-b).abs() < ε
}

fn random_range(cards : &Vec<Card>) -> HashMap<Hand, f64>{
	let mut range : HashMap<Hand, f64> = HashMap::new();
	use rand::Rng;
	let mut rng = rand::rng();
	for card in cards {
		let hand = Hand::new_with_cards(vec![*card]);
		let weight = rng.random();
		if rand::random_bool(1.0 / 3.0) { // 1/3 chance to zero it out
			range.insert(hand, 0.0);
		} else {
			range.insert(hand, weight);
		}
	}
	range
}

fn fixed_akq_range(ace : f64, king : f64, queen : f64) -> HashMap<Hand, f64>{
	let mut range : HashMap<Hand, f64> = HashMap::new();
	range.insert(Hand::new_with_cards(vec![Card::new(Value::Ace, Suit::Diamond)]), ace);
	range.insert(Hand::new_with_cards(vec![Card::new(Value::King, Suit::Diamond)]), king);
	range.insert(Hand::new_with_cards(vec![Card::new(Value::Queen, Suit::Diamond)]), queen);
	range
}

mod equity {
	use super::*;

	mod hand_vs_range {
		use super::*;

		// due to conditional probability and overlapping blockers this doesn't work
		/*
		fn r_v_r_using_h_v_r(a: &HashMap<Hand, f64>, b: &HashMap<Hand, f64>) -> Vec<f64> {
			let a_equity = a.iter()
				.map(|(hand, weight)| weight * hand_vs_ranges(*hand, vec![b]))
				.sum::<f64>();
			let b_equity = b.iter()
				.map(|(hand, weight)| weight * hand_vs_ranges(*hand, vec![a]))
				.sum::<f64>();
			let normalisation = a_equity + b_equity;
			if normalisation == 0.0 { return vec![0.0; 2]; }
			vec![a_equity/normalisation, b_equity/normalisation]
		}
		*/

		#[test]
		fn hand_vs_range_equal_to_solo_range_vs_range() {
			let card_set = akq_deck();
			for _ in 0..1000 {
				let rand_range = random_range(&card_set);
				for (i, card) in akq_deck().iter().enumerate() {
					let one_range = match i {
						0 => fixed_akq_range(1.0, 0.0, 0.0),
						1 => fixed_akq_range(0.0, 1.0, 0.0),
						2 => fixed_akq_range(0.0, 0.0, 1.0),
						_ => unreachable!(),
					};
					let one_range_vs_range = equities_of_ranges(vec![&one_range, &rand_range])[0];
					let hand_vs_range = equity_of_hand_vs_ranges(Hand::new_with_cards(vec![*card]), vec![&rand_range]);
					println!("direct hand vs range: {:?}", &hand_vs_range);
					println!("one card range vs range: {:?}", &one_range_vs_range);
					assert!(is_equal(hand_vs_range, one_range_vs_range))
				}
			}
		}
	}

	mod range_vs_range {
		use super::*;

		#[test]
		fn solo_range_vs_range_matches_all_ranges() {
			let card_set = akq_deck();
			for _ in 0..1000 {
				let a = random_range(&card_set);
				let b = random_range(&card_set);
				let solo_ranges = vec![equity_of_range(&a, &b), equity_of_range(&b, &a)];
				let coupled_ranges = equities_of_ranges(vec![&a, &b]);
				println!("each range vs ranges: {:?}", &coupled_ranges);
				println!("direct ranges vs ranges: {:?}", &solo_ranges);
				for i in 0..solo_ranges.len() {
					assert!(is_equal(solo_ranges[i], coupled_ranges[i]));
				}
			}
		}

		#[test]
		fn equities_sum_to_1_or_0() {
			let card_set = akq_deck();
			for _ in 0..1000 {
				let range0 = random_range(&card_set);
				let range1 = random_range(&card_set);
				let equity_sum = equities_of_ranges(vec![&range0, &range1]).iter().sum::<f64>();
				// println!("range0: {:?}\nrange1: {:?}\nequity: {:?}", &range0, &range1, evs_of_ranges(vec![&range0, &range1]));
				assert!(is_equal(equity_sum, 1.0) || is_equal(equity_sum, 0.0));
			}
		}

		#[test]
		fn known_equities() {
			let range0 = fixed_akq_range(0.0, 0.1, 0.0);
			let range1 = fixed_akq_range(0.1, 0.0, 0.1);
			let eq = equities_of_ranges(vec![&range0, &range1]);
			assert!(is_equal(eq[0], 0.5));
			assert!(is_equal(eq[1], 0.5));

			let range0 = fixed_akq_range(0.2, 0.1, 0.7);
			let range1 = fixed_akq_range(0.1, 0.5, 0.1);
			let eq = equities_of_ranges(vec![&range0, &range1]);
			assert!(is_equal(eq[0], 0.23214288));
			assert!(is_equal(eq[1], 0.76785712));

			let range0 = fixed_akq_range(0.2, 0.1, 0.0);
			let range1 = fixed_akq_range(0.1, 0.5, 0.0);
			let eq = equities_of_ranges(vec![&range0, &range1]);
			assert!(is_equal(eq[0], 0.90909094));
			assert!(is_equal(eq[1], 0.0909091));
		}

		#[test]
		fn equal_ranges_split_pot() {
			let range0 = fixed_akq_range(0.6, 0.2, 0.9);
			let range1 = fixed_akq_range(0.6, 0.2, 0.9);
			let eq = equities_of_ranges(vec![&range0, &range1]);
			assert!(is_equal(eq[0], 0.5));
			assert!(is_equal(eq[1], 0.5));

			let range0 = fixed_akq_range(0.5, 0.5, 0.5);
			let range1 = fixed_akq_range(1.0, 1.0, 1.0);
			let eq = equities_of_ranges(vec![&range0, &range1]);
			assert!(is_equal(eq[0], 0.5));
			assert!(is_equal(eq[1], 0.5));

			let range0 = fixed_akq_range(0.3, 0.0, 0.2);
			let range1 = fixed_akq_range(0.3, 0.0, 0.2);
			let eq = equities_of_ranges(vec![&range0, &range1]);
			assert!(is_equal(eq[0], 0.5));
			assert!(is_equal(eq[1], 0.5));
		}

		#[test]
		fn empty_ranges_both_0() {
			let range0 : HashMap<Hand, f64> = HashMap::new();
			let range1 = fixed_akq_range(0.0, 0.0, 0.0);
			let eq = equities_of_ranges(vec![&range0, &range1]);
			assert_eq!(eq[0], 0.0);
			assert_eq!(eq[1], 0.0);
		}

		#[test]
		fn mutually_blocked_ranges_both_0() {
			let range0 = fixed_akq_range(0.3, 0.0, 0.0);
			let range1 = fixed_akq_range(0.9, 0.0, 0.0);
			let eq = equities_of_ranges(vec![&range0, &range1]);
			assert_eq!(eq[0], 0.0);
			assert_eq!(eq[1], 0.0);
		}

		#[test]
		fn one_has_range_both_0() {
			let range0 = fixed_akq_range(0.3, 0.0, 0.0);
			let range1 = fixed_akq_range(0.0, 0.0, 0.0);
			let eq = equities_of_ranges(vec![&range0, &range1]);
			assert_eq!(eq[0], 0.0);
			assert_eq!(eq[1], 0.0);
		}

		#[test]
		fn dominator_wins() {
			let range0 = fixed_akq_range(0.3, 0.5, 0.0);
			let range1 = fixed_akq_range(0.0, 0.7, 0.0);
			let eq = equities_of_ranges(vec![&range0, &range1]);
			assert_eq!(eq[0], 1.0);

			let range0 = fixed_akq_range(0.3, 0.5, 0.0);
			let range1 = fixed_akq_range(0.0, 0.0, 0.8);
			let eq = equities_of_ranges(vec![&range0, &range1]);
			assert_eq!(eq[0], 1.0);
		}
	}
}

mod game {
	use super::*;
	// helper functions
	fn uniform_akq_halfstreet_game() -> Game {
		let actionset = ActionSet(vec![StreetActions {
			open: vec![Action::Check, Action::Bet(1.0)],
			facing_bet: vec![
				vec![Action::Check, Action::Bet(1.0)]
			],
		}]);
		let ante = 2.0;
		Game::new_uniform(GameConfig { ante, actionset }, akq_deck())
	}

	fn permute_node(node: &mut Node) {
		for _ in 0..4 {
			node.strategy = node.strategy.permutation_of(0.2);
		}
	}

	fn permute_game(game: &mut Game) {
		for node in game.0.iter_mut() {
			permute_node(node);
		}
		game.update_child_ranges_at(0);
	}

	fn random_akq_halfstreet_game() -> Game {
		let mut game = uniform_akq_halfstreet_game();
		permute_game(&mut game);
		game.update_evs();
		game
	}

	/*
	fn randomwalk_best_response(node: &mut Node, exploiter : usize) {
	if exploiter == node.player_index as usize {
	// save the previous state
	let old_ev = node.ev_of_current();
	let old_strategy = node.strategy.clone();
	// try a new state
	let try_strategy = node.strategy.permutation_of(1.0);
	node.set_strategy(&try_strategy);
	// revert if new strategy has lower EV
	let new_ev = node.ev_of_current();
	if new_ev < old_ev {
	node.set_strategy(&old_strategy);
	}
	}
	for child in &mut node.children {
	randomwalk_best_response(child, exploiter);
	};
	}
	*/

	fn monte_carlo_round_at(game: &mut Game, idx: u32, target: usize) -> f64 {
		if game.0[idx as usize].actions == None {
			return game.ev_at(idx)[target]; // assume that showdown EV is valid
		}
		use rand::Rng;
		let mut rng = rand::rng();
		let range = &game.0[idx as usize].players[game.0[idx as usize].player_index as usize].range;
		// choose card from range
			let rand_into = rng.random::<f64>() * range.values().sum::<f64>();
			let mut seen_weight = 0.0;
			let mut chosen_card = Hand::new_with_cards(vec![Card::new(Value::Jack, Suit::Spade)]); // junk value
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
				game.0[idx as usize].strategy.0.get(&chosen_card).unwrap()
			).unwrap().sample(&mut rand::rng());
			let child_node_index = game.0[idx as usize].children[chosen_action_index];
			// bet/call have a cost TODO: pot commitment and last action
			let price = if game.0[idx as usize].player_index as usize != target { 0.0 }
				else {
					match game.0[idx as usize].actions.as_ref().unwrap()[chosen_action_index as usize] {
						Action::Bet(bet) => -1.0*bet,
						Action::Call => -1.0, // todo: bet size and invested
						_ => 0.0,
					}
			};
			let ev = price + monte_carlo_round_at(game, child_node_index, target);
			ev
		}

		/// monte carlo is used for testing because it is independent to, and is simpler than the analytic weighted sum used by the nodes, but is inefficient for real use
		/// ev refers to the turn player only
		fn monte_carlo_ev_at(game: &Game, idx: u32) -> f64 {
			let mut game_clone = game.clone();
			let mut avg_ev = 0.0;
			let δ = 0.00001;
			let mut progress = 0.0;
			while progress < 1.0 {
				avg_ev += δ * monte_carlo_round_at(&mut game_clone, idx, game.0[idx as usize].player_index as usize);
				progress += δ;
			}
			avg_ev
		}

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

		/*
		#[test]
		fn best_response_doesnt_decrease_monte_carlo_ev() {
			// let mut root = uniform_akq_halfstreet_tree();
			for _ in 0..50 {
				let mut root = random_akq_halfstreet_tree();
				let old_ev = monte_carlo_ev(&root);
				root.cfr_exploitative_iteration(0);
				let new_ev = monte_carlo_ev(&root);
				assert!(old_ev <= new_ev + 0.01, "old ev:{old_ev}, new ev:{new_ev}");
				for child in &mut root.children {
					let old_ev = monte_carlo_ev(&child);
					child.cfr_exploitative_iteration(1);
					let new_ev = monte_carlo_ev(&child);
					assert!(old_ev <= new_ev + 0.01, "old ev:{old_ev}, new ev:{new_ev}");
				}
			}
		}
		*/

		/*
		#[test]
		fn best_response_doesnt_decrease_self_ev() {
			// let mut root = uniform_akq_halfstreet_tree();
			for _ in 0..10000 {
				let mut root = random_akq_halfstreet_tree();
				let old_ev = root.ev()[root.player_index as usize];
				// let old_state = root.clone();
				root.cfr_exploitative_iteration(0);
				let new_ev = root.ev()[root.player_index as usize];
					// println!("before:\n{:?}\nafter\n{:?}\nold ev:{old_ev}, new ev:{new_ev}", &old_state, &root);
				assert!(old_ev <= new_ev + 0.01, "old ev:{old_ev}, new ev:{new_ev}");
				for child in &mut root.children {
					let old_ev = child.ev()[child.player_index as usize];
					// let old_state = child.clone();
					child.cfr_exploitative_iteration(1);
					let new_ev = child.ev()[child.player_index as usize];
					// println!("before:\n{:?}\nafter\n{:?}\nold ev:{old_ev}, new ev:{new_ev}", &old_state, &root);
					assert!(old_ev <= new_ev + 0.01, "old ev:{old_ev}, new ev:{new_ev}");
				}
			}
		}
		*/

		// helper function for recursion
		fn recursive_ev_match(game: &Game, idx: u32) {
			let mc_ev = monte_carlo_ev_at(game, idx);
			let self_ev = game.ev_at(idx)[game.0[idx as usize].player_index as usize];
			let diff = (mc_ev - self_ev).abs();
			if diff > 0.01 {
				println!("state\n{:?}\n", &game.0[idx as usize]);
				panic!("diff: {diff}, self ev:{self_ev}, monte carlo ev:{mc_ev}");
			}
			// assert!(diff < ε, "diff: {diff}, self ev:{self_ev}, monte carlo ev:{mc_ev}");
			for child_idx in &game.0[idx as usize].children {
				recursive_ev_match(game, *child_idx);
			}
		}

		#[test]
		#[ignore]
		fn self_ev_matches_numerical_approximation() {
			for i in 0..100 {
				println!("{}", i);
				let root = random_akq_halfstreet_game();
				recursive_ev_match(&root, 0);
			}
		}

		/*
		#[test]
		fn cfr_exploit_matches_random_step_exploit() {
			for _ in 0..100 {
				let game = random_akq_halfstreet_game();
				let mut randomwalk_exploit = game.clone();
				for _ in 0..30 {
					randomwalk_exploitative_iteration(&mut randomwalk_exploit, 1);
				}
				let cfr_ev = root.min_ev_of_current();
				let rw_ev = randomwalk_exploit.ev()[0];
				assert!(is_equal(cfr_ev, rw_ev));
			}
		}
		*/

	}

}
