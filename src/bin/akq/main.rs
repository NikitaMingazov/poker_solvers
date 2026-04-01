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

#[allow(non_upper_case_globals)]
const ε : f64 = 0.0001;

use std::fs;
use std::fs::File;
use std::io::BufReader;
use serde::{Deserialize, Serialize};
use std::env;
use std::collections::{HashMap, hash_map};

struct Regrets(HashMap<Hand, HashMap<Action, Vec<f64>>>);
impl Regrets {
	fn new() -> Self {
		Regrets(HashMap::new())
	}
}

struct GameRegrets(Box<[Regrets]>);
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
						regrets_init.0.get_mut(hand).unwrap().insert(action, vec![]);
					}
				}
				regret_vec.push(regrets_init);
			}
		}
		GameRegrets(regret_vec.into_boxed_slice())
	}
}

#[derive(Debug)]
struct Game(Box<[Node]>);

impl Game {
	pub fn new_uniform(cfg: GameConfig, deck: Vec<Card>) -> Self {
		// let len = 0;
		// let mut nodes_uninit : Box<[MaybeUninit<Node>]> = Box::new_uninit_slice(len);
		let mut nodes_vec = vec![];
		// nodes_vec.reserve(len);

		// enumerate hands
		let hands = deck.iter().map(|card| Hand::new_with_cards(vec![*card])).collect();
		// create initial state of the game
		let root = Node::new(
			cfg.ante,
			0,
			Strategy::init(&cfg.actionset, 0, 0, &hands),
			cfg.actionset.actions_at(0, 0),
			None,
			Player::initlist(2, deck),
		);
		nodes_vec.push(root);
		let mut new_idx = 1; // where to start inserting new nodes
		let mut iteration_stack = vec![(0, None, 0, 0)];
		while !iteration_stack.is_empty() {
			let (idx, action_change, street, bet_level) = iteration_stack.pop().unwrap();
			let children_added = Self::init_children_at(idx, &mut new_idx, &mut nodes_vec, &cfg.actionset, action_change, street, bet_level);
			if let Some(children) = children_added {
				for state in children.iter().rev() {
					iteration_stack.push(*state);
				}
			}
		}
		// nodes_uninit[idx as usize].write(root);
		// SAFETY: every node is now initialised
		// let nodes_init = unsafe { nodes_uninit.assume_init() };
		// Game(nodes_init)
		Game(nodes_vec.into_boxed_slice())
	}

	fn init_children_at(idx: u32, new_idx: &mut u32, game_arena: &mut Vec<Node>, actionset: &ActionSet, mut action_change: Option<LastChange>, mut street: u8, mut bet_level: u8) -> Option<Vec<(u32, Option<LastChange>, u8, u8)>> {
		// let mut cur_node = &mut game_arena[idx as usize];
		if game_arena[idx as usize].players.len() == 1 {
			return None;
		}
		match action_change {
			Some(bet) => {
				if bet.player_index == game_arena[idx as usize].player_index {
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
		let n_player_index = if usize::from(game_arena[idx as usize].player_index+1) < game_arena[idx as usize].players.len() {game_arena[idx as usize].player_index + 1} else {0};
		let deck = game_arena[idx as usize].players[n_player_index as usize].range.keys().cloned().collect();
		let num_actions = {
			match actionset.actions_at(street, bet_level) {
				None => 0,
				Some(actions)=> actions.len(),
			}
		};
		let mut n_players = game_arena[idx as usize].players.clone();
		for (_, j) in n_players[game_arena[idx as usize].player_index as usize].range.iter_mut() {
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
								game_arena[idx as usize].pot,
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
								game_arena[idx as usize].pot+value,
								n_player_index,
								Some( LastChange {
									player_index: game_arena[idx as usize].player_index,
									bet: 1.0
								} ),
								street,
								bet_level,
								n_players.clone(),
							)
						);
					}
					Action::Fold => {
						let mut n_players = n_players.clone();
						n_players.remove(game_arena[idx as usize].player_index as usize);
						let street = street+1; // todo: multiway
						let bet_level = 0;
						new_children.push(
							(
								game_arena[idx as usize].pot,
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
								game_arena[idx as usize].pot+1.0, //TODO: action_change - pot_investment
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
		let mut child_iters = vec![];
		for (n_pot, n_player_index, n_action, n_street, n_bet_level, n_players) in new_children {
			// create the child nodes in the arena
			game_arena[idx as usize].children.push(*new_idx);
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
		// this is not a recursive function anymore
		// a root node has no actions
		// for child_idx in game_arena[idx as usize].children.clone() {
		// 	if game_arena[child_idx as usize].children.len() == 0 {
		// 		game_arena[child_idx as usize].actions = None;
		// 		game_arena[child_idx as usize].strategy = Strategy(HashMap::new());
		// 	}
		// }
		// for child in &cur_node.children {
		// 	// child.ev = child.ev()[child.player_index as usize];
		// }
		// self.ev = self.ev()[self.player_index as usize];
		Some(child_iters)
	}

	pub fn cfr_iteration(&mut self, regrets: &mut GameRegrets) {
		// read game, write to regrets
		self.append_regrets(regrets);
		// read from regrets, write to game
		self.update_strategy(regrets);
		self.update_evs();
	}

	fn update_strategy(&mut self, regrets: &GameRegrets) {
		for i in 0..self.0.len() {
			for (hand, actions) in self.0[i].strategy.0.iter_mut() {
				let T = regrets.0[i].0.get(hand).unwrap().iter().last().iter().count();
				let all_action_regret_sum = regrets.0[i].0.get(hand).unwrap().values()
					.map( |weights : &Vec<f64>| weights.iter().sum::<f64>())
					.map(|x| if x < 0.0 { 0.0 } else { x } )
					.sum::<f64>()
					  /
					T as f64;
				let num_actions = actions.iter().count() as f64;
				for (idx, weight) in actions.iter_mut().enumerate() {
					let action_regret_sum : f64 = regrets.0[i].0.get(hand).unwrap()
						.get(&self.0[i].actions.as_ref().expect("nonterminal")[idx]).unwrap().iter().sum::<f64>()
					  	  /
						T as f64;
				let action_regret_pos = if action_regret_sum < 0.0 { 0.0 } else { action_regret_sum };
				let new_weight = if all_action_regret_sum > 0.0 {
					action_regret_pos / all_action_regret_sum
				} else {
					1.0 / num_actions
				};
				*weight = new_weight;
				}
			}
			self.update_child_ranges_at(i.try_into().unwrap(), self.0[i].player_index as usize);
		}
	}

	fn update_evs(&mut self) {
		for i in 0..self.0.len() {
			self.0[i].ev = self.ev_at(i.try_into().unwrap())[self.0[i].player_index as usize];
		}
	}

	fn append_regrets(&mut self, regrets: &mut GameRegrets) {
		for i in 0..self.0.len() {
			self.append_regret(regrets, i.try_into().unwrap());
		}
	}

	// mutable to enable in-place modification of strategy
	fn append_regret(&mut self, regrets: &mut GameRegrets, idx: u32) {
		// non-borrowing alias
		macro_rules! cur_node {
			() => { self.0[idx as usize]};
		}
		if cur_node!().children.len() == 0 {return};
		let current_strategy = cur_node!().strategy.clone();
		for hand in cur_node!().players[cur_node!().player_index as usize].range.clone().keys() {
			let strategy_for_hand = current_strategy.0.get(hand).expect("nonterminal");
			// dumbass borrow checker doesn't check fields, regrets is disjoint from players
			let hand_regrets : &mut HashMap<Action, Vec<f64>> = regrets.0[idx as usize].0.get_mut(hand).unwrap();
			for (i, action) in cur_node!().clone().actions.as_ref().expect("nonterminal").iter().enumerate() {
				let mut α = vec![0.0; cur_node!().actions.as_ref().expect("non-terminal").len()];
				α[i] = 1.0;
				self.set_strategy_for_hand_at(idx, *hand, &α);
				let u_i_new = self.ev_of_hand_at(idx, &hand);
				self.set_strategy_for_hand_at(idx, *hand, &strategy_for_hand);
				let u_i_old = self.ev_of_hand_at(idx, &hand);
				// println!("{:?}, {:?}", u_i_new, u_i_old);
				// let regret = π * ( u_i_new - u_i_old );
				let regret = u_i_new - u_i_old;
				hand_regrets.get_mut(action).unwrap().push(regret);
			}
		}
	}

	// fn set_strategy(&mut self, new_strat : &Strategy) {
	// 	self.strategy = new_strat.clone();
	// 	self.update_child_ranges(self.player_index as usize);
	// }

	fn set_strategy_for_hand_at(&mut self, idx: u32, hand : Hand, action_weights : &Vec<f64>) {
		// non-borrowing alias
		macro_rules! cur_node {
			() => { self.0[idx as usize]};
		}
		cur_node!().strategy.0.insert(hand, action_weights.clone());
		self.update_child_ranges_at(idx, cur_node!().player_index as usize);
	}

	// update the subtree to use the new strategy/range for range calculations for the target player
	fn update_child_ranges_at(&mut self, idx: u32, target: usize) {
		// non-borrowing alias
		macro_rules! cur_node {
			() => { self.0[idx as usize]};
		}
		if cur_node!().children.is_empty() { return; }
		for (index, &child_idx) in cur_node!().clone().children.iter().enumerate() {
			// skip if it's folded out, because the last player's range has no effect
			// TODO: handle players properly  for multiway allin, hashmap<player> instead of vec
			if self.0[child_idx as usize].players.len() == 1 { continue; }
			// set child range to this range, before factoring probability of moving there
			self.0[child_idx as usize].players[target].range = cur_node!().players[target].range.clone();
			if cur_node!().player_index as usize == target {
				let out_weight_into_child: Vec<(Hand, f64)> = cur_node!().strategy.0.iter()
					.map(|(hand, weights) : (&Hand, &Vec<f64>)| (hand.clone(), weights[index]))
					.collect();
				for (hand, out_weight) in &out_weight_into_child {
					*self.0[child_idx as usize].players[target].range.get_mut(&hand).unwrap() *= out_weight;
				}
				// println!("\nout_weights: {:?}\nthis_range: {:?}\nchild_range: {:?}", &out_weight_into_child, &cur_node!().players[target].range, &child.players[cur_node!().player_index as usize].range);
			}
			// cascade the change down
			self.update_child_ranges_at(child_idx, target);
		}
	}

	// todo: properly optimised recursion
	fn ev_of_hand_at(&mut self, idx: u32, hand : &Hand) -> f64 {
		let range = self.0[idx as usize]
			.players[self.0[idx as usize].player_index as usize]
			.range.clone();
		for (hand_in_iter, weight) in self.0[idx as usize]
			.players[self.0[idx as usize].player_index as usize]
			.range.iter_mut() {
				if hand_in_iter != hand {
					*weight = 0.0;
				}
		}
		self.update_child_ranges_at(idx, self.0[idx as usize].player_index as usize);
		let ev_of_hand = self.ev_at(idx)[self.0[idx as usize].player_index as usize];
		self.0[idx as usize]
			.players[self.0[idx as usize].player_index as usize]
			.range = range;
		self.update_child_ranges_at(idx, self.0[idx as usize].player_index as usize);
		ev_of_hand
	}

	// no exploitation, just playing the tree as it is
	// returns the EV of each player
	pub fn ev_at(&self, idx: u32) -> Vec<f64> {
		// non-borrowing alias
		macro_rules! cur_node {
			() => { &self.0[idx as usize]};
		}
		// end recursion if the node is unreachable
		for player in &cur_node!().players {
			if player.range.values().sum::<f64>() == 0.0 {
				return vec![0.0, 0.0];
			}
		}
		// end recursion at leaf
		if cur_node!().actions == None {
			// folded out TODO: multiway
			if cur_node!().players.len() == 1 {
				return match cur_node!().players[0].seat {
					0 => vec![cur_node!().pot, 0.0],
					1 => vec![0.0, cur_node!().pot],
					_ => panic!(),
				};
			} else { // showdown
				return scale_f64_vec(&r_v_r(&cur_node!().players[0].range, &cur_node!().players[1].range), cur_node!().pot);
			}
		}
		let mut ev : Vec<f64> = vec![0.0; 2];
		let mut out_prob : Vec<f64> = vec![];
		// prevent division by 0
		let range_total = cur_node!().players[cur_node!().player_index as usize].range.iter()
			.map(|(_, weight)| *weight)
			.sum::<f64>();
		for (i, action) in cur_node!().actions.as_ref().unwrap().iter().enumerate() {
			if range_total != 0.0 {
				out_prob.push(
					cur_node!().strategy.0.iter()
						// all the outward edges for an action, irrespective of card
						.map(|(card, weights)|
							weights[i]
						  	*
							cur_node!().players[cur_node!().player_index as usize].range.get(card).unwrap()
						)
						.sum::<f64>()
						// divided by the total range, to normalise the vector
						  /
						range_total
				);
			} else {
				out_prob.push(0.0);
			}
			// subtract from EV for bets performed
			// TODO: bet committment for call/bet
			match action {
				Action::Bet(bet) => ev[cur_node!().player_index as usize] -= bet * out_prob[i],
				Action::Call => ev[cur_node!().player_index as usize] -= cur_node!().action_change.unwrap().bet * out_prob[i],
				_ => (),
			}
		}
		for (i, &child) in cur_node!().children.iter().enumerate() {
			ev = add_f64_vec(&ev, &scale_f64_vec(&self.ev_at(child), out_prob[i]));
		}
		ev
	}

	/*

	define_recursion_while!(append_regret, (|s: &mut Self| !s.children.is_empty()) );
	fn append_regret(&mut self) {
		let self_mut_ptr = self as *mut Self;
		let current_strategy = self.strategy.clone();
		for hand in self.players[self.player_index as usize].range.keys() {
			let strategy_for_hand = current_strategy.0.get(hand).expect("nonterminal");
			// dumbass borrow checker doesn't check fields, regrets is disjoint from players
			let hand_regrets : &mut HashMap<Action, Vec<f64>> = unsafe {
				(*self_mut_ptr).regrets.get_mut(hand).unwrap()
				// self.regrets.get_mut(hand).unwrap(); // my code should look like
			};
			for (i, action) in self.actions.as_ref().expect("nonterminal").iter().enumerate() {
				let mut α = vec![0.0; self.actions.as_ref().expect("non-terminal").len()];
				α[i] = 1.0;
				// self.set_strategy_for_hand(*hand, &α);
				// yet again, strategy is only read by .ev()
				unsafe {
					(*self_mut_ptr).set_strategy_for_hand(*hand, &α);
				}
				let u_i_new = self.ev_of_hand(&hand);
				// self.set_strategy_for_hand(*hand, &strategy_for_hand);
				unsafe {
					(*self_mut_ptr).set_strategy_for_hand(*hand, &strategy_for_hand);
				}
				let u_i_old = self.ev_of_hand(&hand);
				// println!("{:?}, {:?}", u_i_new, u_i_old);
				// let regret = π * ( u_i_new - u_i_old );
				let regret = u_i_new - u_i_old;
				hand_regrets.get_mut(action).unwrap().push(regret);
			}
		}
	}

	define_recursion_while!{ update_strategy, (|s: &mut Self| !s.children.is_empty()) }
	fn update_strategy(&mut self) {
		for (hand, actions) in self.strategy.0.iter_mut() {
			let T = self.regrets.get(hand).unwrap().iter().last().iter().count();
			let all_action_regret_sum = self.regrets.get(hand).unwrap().values()
				.map( |weights : &Vec<f64>| weights.iter().sum::<f64>())
				.map(|x| if x < 0.0 { 0.0 } else { x } )
				.sum::<f64>()
				  /
				T as f64;
			let num_actions = actions.iter().count() as f64;
			for (idx, weight) in actions.iter_mut().enumerate() {
				let action_regret_sum : f64 = self.regrets.get(hand).unwrap()
					.get(&self.actions.as_ref().expect("nonterminal")[idx]).unwrap().iter().sum::<f64>()
					  /
					T as f64;
				let action_regret_pos = if action_regret_sum < 0.0 { 0.0 } else { action_regret_sum };
				let new_weight = if all_action_regret_sum > 0.0 {
					action_regret_pos / all_action_regret_sum
				} else {
					1.0 / num_actions
				};
				*weight = new_weight;
			}
		}
		self.set_strategy(&self.strategy.clone());
	}

				/*let other_player = if self.player_index == 0 { 1 } else { 0 };
				let π = {
					let unblocked_range = self.players[other_player as usize].range.iter()
						.filter(|(other_hand, _)| !is_blocked(hand, other_hand))
						.collect();
					let unblocked_prob = unblocked_range.values().sum::<f64>();
					let count = unblocked_range.keys().count() as f64;
					unblocked_prob / count
				}*/


	// fn update_child_ranges_helper(&mut self, target: usize) {
	// }

	define_recursion_while!{
		update_evs,
		(|s: &mut Self| { s.ev = s.ev_of_current(); }),
		(|_s: &mut Self| true)
	}
	// returns the EV of this node for the current player, using the global strategy
	fn ev_of_current(&self) -> f64 {
		return self.ev()[self.player_index as usize];
	}


	// returns the EV of this node for the current player, using the global strategy
	fn min_ev_of_current(&self) -> f64 {
		return self.ev_after_exploitation()[self.player_index as usize];
	}

	define_recursion_while!{
		clear_regrets,
		(|s: &mut Self| {
			s.regrets
				.values_mut()
				.for_each(|hand_regrets| {
					hand_regrets
						.values_mut()
						.for_each(|action_regrets| action_regrets.clear())
				});
		}),
		(|_s: &mut Self| true)
	}
	// returns the EVs of the current node if the opponent is maximally exploitative starting at the current node
	pub fn ev_after_exploitation(&self) -> Vec<f64> {
		// TODO: pure strategy best response
		let mut exploit_tree = self.clone();
		let exploiter = if self.player_index == 1 { 0 } else { 1 };
		exploit_tree.cfr_exploit(exploiter, 10);
		exploit_tree.ev()
	}

	fn cfr_exploit(&mut self, exploiter: usize, n: u16) {
		for _ in 0..n {
			self.cfr_exploitative_iteration(exploiter);
		}
	}

	fn cfr_exploitative_iteration(&mut self, exploiter : usize) {
		recurse_method!(self, clear_regrets);
		self.update_strategy_for_exploiter(exploiter);
	}

	fn append_regrets_for_exploiter(&mut self, exploiter : usize) {
		if self.children.is_empty() { return; }
		if self.player_index as usize == exploiter {
			self.append_regret();
		}
		for child in &mut self.children {
			child.append_regrets_for_exploiter(exploiter);
		}
	}

	fn update_strategy_for_exploiter(&mut self, exploiter : usize) {
		if self.children.is_empty() { return; }
		if self.player_index as usize == exploiter {
			self.update_strategy();
		}
		for child in &mut self.children {
			child.update_strategy_for_exploiter(exploiter);
		}
	}
	*/
}

/// Parameters for the configuration of the game.
#[derive(Deserialize)]
struct GameConfig {
	ante: f64,
	// players:
	actionset: ActionSet,
}

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

fn parse_config(path: &str) -> Result<GameConfig, Box<dyn std::error::Error>> {
	let file = File::open(path)?;
	let reader = BufReader::new(file);
	let cfg = serde_json::from_reader(reader)?;
	Ok(cfg)
}

#[derive(Debug, Clone, PartialEq)]
enum Action {
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
	seat : u8,
	range : HashMap<Hand, f64>,
	// street_committment: f64,
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
		let mut range : HashMap<Hand, f64> = HashMap::new();
		for card in deck {
			range.insert(Hand::new_with_cards(vec![card]), 1.0);
		}
		Self {
			seat,
			range
		}
	}
}

#[derive(Debug, Clone, Serialize)]
struct Strategy(HashMap<Hand, Vec<f64>>);

impl Strategy {
	fn init(actionset: &ActionSet, street : u8, level : u8, hand_set : &Vec<Hand>) -> Self {
		let mut strat : Self = Strategy(HashMap::new());
		match actionset.actions_at(street, level) {
			None => (),
			Some(actions) => {
				for hand in hand_set {
					strat.0.insert(*hand, vec![]);
					#[allow(non_snake_case)]
					let A = actions.len() as f64;
					for _ in &actions {
						strat.0.get_mut(&hand).unwrap().push(1.0 / A);
					}
				}
			},
		}
		strat
	}

}

#[derive(Copy, Clone, Debug, Serialize)]
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

// equity of first hand
fn hand_vs_hand(a : &Hand, b : &Hand) -> f64 {
	let a_card = a.iter().nth(0);
	let b_card = b.iter().nth(0);
	if a_card > b_card { return 1.0 } else { return 0.0 }
}

// sums to 1 unless there was no play, and both players have 0
fn eq_of_range(a : &HashMap<Hand, f64>, b : &HashMap<Hand, f64>) -> f64 {
	// calculate hand vs hand equity for each pair, weighted by occurence rate
	let mut equity = 0.0;
	let mut total_weight = 0.0;
	// iterate over each hand pair
	for (a_hand, a_weight) in a.iter() {
		for (b_hand, b_weight) in b.iter() {
			// blocked and impossible
			if *a_hand & *b_hand != Hand::new() { continue; }
			let pair_weight = a_weight * b_weight;
			// a card is absent from at least one player's side of pair
			if pair_weight < ε { continue; }
			equity += pair_weight * hand_vs_hand(a_hand, b_hand);
			// equity[b] += pair_weight * hand_vs_hand(b_hand, a_hand);
			total_weight += pair_weight;
		}
	}
	// normalise result, such that eq(a, b) + eq(b, a) = 1
	// but if there was no play, equity is already 0
	if total_weight != 0.0 { equity /= total_weight; }
	// hack for floating point precision
	if equity < ε { equity = 0.0 }
	if equity > 1.0-ε { equity = 1.0 }
	equity
}

// range vs range
// todo: board, multiway
fn r_v_r(a : &HashMap<Hand, f64>, b : &HashMap<Hand, f64>) -> Vec<f64> {
	vec![eq_of_range(a, b), eq_of_range(b, a)]
}

#[derive(Debug, Clone, Serialize)]
struct Node {
	// tree: &GameTree,
	pot : f64,
	ev : f64, // TODO: remove this. .ev() is the real interface, this is diagnostic only
	player_index : u8,
	action_change : Option<LastChange>,
	actions : Option<Vec<Action>>,
	strategy : Strategy,
	players : Vec<Player>,
	// current_seat : usize,
	// active_seats : Vec<u8>,
	// players : HashMap<u8, Player>,
	// history : History,
	children: Vec<u32>,
	// children : Vec<Node>,
}

impl Node {
	fn new(pot : f64, player_index : u8, strategy : Strategy, actions : Option<Vec<Action>>, action_change : Option<LastChange>, players : Vec<Player>) -> Self {
		let temp = Self {
			pot,
			ev : f64::MIN,
			player_index,
			action_change,
			players,
			actions : actions.clone(),
			strategy,
			children : Vec::new(),
		};
		temp
	}
}

// use crate::core::{Card, Hand, Suit, Value};
use poker_solvers::core::*;

fn main() {
	let configjson = env::args().nth(1).expect("action config json required");
	let config = parse_config(&configjson).unwrap();
	let card_set = vec![
		Card::new(Value::Ace, Suit::Diamond),
		Card::new(Value::King, Suit::Diamond),
		Card::new(Value::Queen, Suit::Diamond),
	];
	let mut game = Game::new_uniform(config, card_set);
	let mut regrets = GameRegrets::for_game(&game);
	let rounds = 5;
	// let players = 2;
	for _ in 0..rounds {
		game.cfr_iteration(&mut regrets);
	}
	println!("{:#?}", game);
	// println!("player 0 ev: {}\nev while exploited: {}", root.ev_of_current(), root.min_ev_of_current());
	// let json_string = serde_json::to_string_pretty(&root).unwrap();
	// println!("{}", json_string);
	// let _ = fs::write("out.json", json_string);
}

#[cfg(test)]
mod tests {
	use super::*;

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

		#[test]
		fn equities_sum_to_1_or_0() {
			let card_set = akq_deck();
			for _ in 0..1000 {
				let range0 = random_range(&card_set);
				let range1 = random_range(&card_set);
				let equity_sum = r_v_r(&range0, &range1).iter().sum::<f64>();
				println!("range0: {:?}\nrange1: {:?}\nequity: {:?}", &range0, &range1, r_v_r(&range0, &range1));
				assert!(is_equal(equity_sum, 1.0) || is_equal(equity_sum, 0.0));
			}
		}

		#[test]
		fn known_equities() {
			let range0 = fixed_akq_range(0.0, 0.1, 0.0);
			let range1 = fixed_akq_range(0.1, 0.0, 0.1);
			let eq = r_v_r(&range0, &range1);
			assert!(is_equal(eq[0], 0.5));
			assert!(is_equal(eq[1], 0.5));

			let range0 = fixed_akq_range(0.2, 0.1, 0.7);
			let range1 = fixed_akq_range(0.1, 0.5, 0.1);
			let eq = r_v_r(&range0, &range1);
			assert!(is_equal(eq[0], 0.23214288));
			assert!(is_equal(eq[1], 0.76785712));

			let range0 = fixed_akq_range(0.2, 0.1, 0.0);
			let range1 = fixed_akq_range(0.1, 0.5, 0.0);
			let eq = r_v_r(&range0, &range1);
			assert!(is_equal(eq[0], 0.90909094));
			assert!(is_equal(eq[1], 0.0909091));
		}

		#[test]
		fn equal_ranges_split_pot() {
			let range0 = fixed_akq_range(0.6, 0.2, 0.9);
			let range1 = fixed_akq_range(0.6, 0.2, 0.9);
			let eq = r_v_r(&range0, &range1);
			assert!(is_equal(eq[0], 0.5));
			assert!(is_equal(eq[1], 0.5));

			let range0 = fixed_akq_range(0.5, 0.5, 0.5);
			let range1 = fixed_akq_range(1.0, 1.0, 1.0);
			let eq = r_v_r(&range0, &range1);
			assert!(is_equal(eq[0], 0.5));
			assert!(is_equal(eq[1], 0.5));

			let range0 = fixed_akq_range(0.3, 0.0, 0.2);
			let range1 = fixed_akq_range(0.3, 0.0, 0.2);
			let eq = r_v_r(&range0, &range1);
			assert!(is_equal(eq[0], 0.5));
			assert!(is_equal(eq[1], 0.5));
		}

		#[test]
		fn empty_ranges_both_0() {
			let range0 : HashMap<Hand, f64> = HashMap::new();
			let range1 = fixed_akq_range(0.0, 0.0, 0.0);
			let eq = r_v_r(&range0, &range1);
			assert_eq!(eq[0], 0.0);
			assert_eq!(eq[1], 0.0);
		}

		#[test]
		fn mutually_blocked_ranges_both_0() {
			let range0 = fixed_akq_range(0.3, 0.0, 0.0);
			let range1 = fixed_akq_range(0.9, 0.0, 0.0);
			let eq = r_v_r(&range0, &range1);
			assert_eq!(eq[0], 0.0);
			assert_eq!(eq[1], 0.0);
		}

		#[test]
		fn one_has_range_both_0() {
			let range0 = fixed_akq_range(0.3, 0.0, 0.0);
			let range1 = fixed_akq_range(0.0, 0.0, 0.0);
			let eq = r_v_r(&range0, &range1);
			assert_eq!(eq[0], 0.0);
			assert_eq!(eq[1], 0.0);
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

		fn uniform_akq_halfstreet_tree() -> Node {
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

		fn random_akq_halfstreet_tree() -> Node {
			let mut root = uniform_akq_halfstreet_tree();
			permute_tree(&mut root);
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

		fn randomwalk_exploitative_iteration(node: &mut Node, exploiter : usize) {
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
				randomwalk_exploitative_iteration(child, exploiter);
			};
		}


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

		// these fundamentally don't work for CFR which doesn't depend on current strategy
		/*
		#[test]
		fn exploitative_iteration_doesnt_decrease_monte_carlo_ev() {
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
		fn exploitative_iteration_doesnt_decrease_self_ev() {
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
		#[ignore]
		fn self_ev_matches_numerical_approximation() {
			for _ in 0..100 {
				let root = random_akq_halfstreet_tree();
				recursive_ev_match(&root);
			}
		}

		#[test]
		fn cfr_exploit_matches_random_step_exploit() {
			for _ in 0..100 {
				let root = random_akq_halfstreet_tree();
				let mut randomwalk_exploit = root.clone();
				for _ in 0..30 {
					randomwalk_exploitative_iteration(&mut randomwalk_exploit, 1);
				}
				let cfr_ev = root.min_ev_of_current();
				let rw_ev = randomwalk_exploit.ev()[0];
				assert!(is_equal(cfr_ev, rw_ev));
			}
		}

	}

}
