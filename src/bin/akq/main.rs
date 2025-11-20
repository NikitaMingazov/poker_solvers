/*The fundamental elements of the game are these:

• There is a three-card deck, containing an Ace, a King, and a Queen.
• The players are each dealt one card without replacement and there is an initial ante.
• There is a round of betting, after which there is a showdown (if neither player folds). In
the showdown, the high card wins.*/

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
	fn actions_at(&self, street : u8, bet_level : u8) -> Vec<Action> {
		// bet level of 0 means check, 1=>bet, 2=>3bet etc.
		if bet_level == 0 {
			if usize::from(street) < self.0.len() {
				return self.0[street as usize].open.clone();
			}
		} else {
			if usize::from(street) < self.0.len() {
				if usize::from(bet_level-1) < self.0[street as usize].facing_bet.len() {
					return self.0[street as usize].facing_bet[usize::from(bet_level-1)].clone()
				}
			}
		}
		return vec![];
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
	Bet(f32), // BBs preflop, pot% postflop
	// Allin,
}

#[derive(Debug, Clone)]
struct Player {
	seat : u8,
	range : HashMap<Card, f32>,
	// committed : f32,
	// stack : f32,
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
		let mut range : HashMap<Card, f32> = HashMap::new();
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
struct Strategy(HashMap<Card, Vec<f32>>);

impl Strategy {
	fn init(actionset: &ActionSet, street : u8, level : u8, range : &Vec<Card>) -> Self {
		let mut strat : Self = Strategy(HashMap::new());
		if actionset.actions_at(street, level).is_empty() { return strat; }
		for card in range {
			strat.0.insert(*card, vec![]);
			let actions = actionset.actions_at(street, level);
			#[allow(non_snake_case)]
			let A = actions.len() as f32;
			for _ in actions {
				strat.0.get_mut(&card).unwrap().push(1.0 / A);
			}
		}
		strat
	}

	fn permutation_of(&self, delta : f32) -> Self {
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
		let sum: f32 = weights.iter().sum();
		for w in weights.iter_mut() {
			*w /= sum;
		}
		new
	}
}

#[derive(Copy, Clone, Debug)]
struct LastChange {
	player_index : u8,
	bet : f32,
}

fn scale_f32_vec(a : &Vec<f32>, μ : f32) -> Vec<f32> {
	a.iter().map(|x| x * μ).collect()
}

fn add_f32_vec(a : &Vec<f32>, b : &Vec<f32>) -> Vec<f32> {
	a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

// range vs range
// todo: blockers, board, multiway
fn r_v_r(a : &HashMap<Card, f32>, b : &HashMap<Card, f32>) -> Vec<f32> {
	let mut equity = 0.0;
	for (a_card, a_weight) in a.iter() {
		for (b_card, b_weight) in b.iter() {
			if a_card > b_card {
				equity += {
					let a_prob = a_weight / a.values().sum::<f32>();
					let b_prob = b_weight / b.iter()
						.filter(|(card, _)| *card != a_card) // blocker effect
						.map(|(_, weight)| *weight)
						.sum::<f32>();
					a_prob * b_prob
				};
			}
		}
	}
	// println!("------------\nrange1: {:?} \nrange2: {:?}\nequity of 1: {:?}\n-----------", a, b, equity);
	vec![equity, 1.0-equity]
}

#[derive(Debug)]
struct Node {
	pot : f32,
	ev : f32,
	// exploitability : f32,
	player_index : u8,
	action_change : Option<LastChange>,
	players : Vec<Player>, // make this a reference to the parent
	actions : Vec<Action>,
	strategy : Strategy,
	// current_seat : usize,
	// active_seats : Vec<u8>,
	// players : HashMap<u8, Player>,
	// history : History,
	children : Vec<Node>,
}

impl Node {

	pub fn init(pot : f32, actionset : &ActionSet, deck : Vec<Card>) -> Self {
		let mut root = Self::new(
			pot,
			0,
			Strategy::init(actionset, 0, 0, &deck),
			actionset.actions_at(0, 0),
			None,
			Player::initlist(2, deck),
		);
		// unfortunately the first call must use a null value
		root.populate_children(actionset, None, 0, 0);
		root
	}

	fn populate_children(&mut self, actionset : &ActionSet, mut action_change : Option<LastChange>, mut street : u8, mut bet_level : u8) {
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
		let num_actions = actionset.actions_at(street, bet_level).len();
		let mut n_players = self.players.clone();
		for (_, j) in n_players[self.player_index as usize].range.iter_mut() {
			(*j) /= num_actions as f32;
		}
		// let n_actions = actionset.actions_at(street, bet_level);
		let mut new_children : Vec<(f32, u8, Option<LastChange>, u8, u8, Vec<Player>)> = Vec::new();
		for action in actionset.actions_at(street, bet_level) {
			match action {
				Action::Check => {
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
		let mut i = 0;
		for (n_pot, n_player_index, n_action, n_street, n_bet_level, n_players) in new_children {
			self.children.push(
				Self::new(
					n_pot,
					n_player_index,
					Strategy::init(actionset, n_street, n_bet_level, &deck),
					actionset.actions_at(n_street, n_bet_level), // todo: proper multi-way logic
					n_action,
					n_players
				)
			);
			self.children[i as usize].populate_children(actionset, n_action, n_street, n_bet_level);
			i += 1;
		}
	}

	fn new(pot : f32, player_index : u8, strategy : Strategy, actions : Vec<Action>, action_change : Option<LastChange>, players : Vec<Player>) -> Self {
		Self {
			pot,
			ev : f32::MIN,
			player_index,
			action_change,
			players,
			actions,
			strategy,
			children : Vec::new(),
		}
	}

	fn update_child_ranges(&mut self, new_strat : &Strategy) {
		if self.children.is_empty() { return; }
		// set child range to this range, before factoring probability of moving there
		for child in self.children.iter_mut() {
			for (i, player) in child.players.iter_mut().enumerate() {
				player.range = self.players[i].range.clone();
			}
		}
		// assign child weights to each child's range by multiplying the current range by the chance of the range going to the node
		for (idx, child) in self.children.iter_mut().enumerate() {
				//todo: make child.players a hashmap for seat as index because this hack won't scale for 3+ players
				if child.players.len() == 1 { continue; }
			// create tuple vector of % of hands that take an outgoing edge
			let out_weights: Vec<(&Card, f32)> = new_strat.0.iter()
				.filter_map(|(card, weights) : (&Card, &Vec<f32>)| Some((card, weights[idx])))
				.collect();
			// println!("\nout_weights: {:?}", &out_weights);
			// new range is scaled by probability of taking the edge
			for (card, out_weight) in out_weights {
				*child.players[self.player_index as usize].range.get_mut(card).unwrap() *=
					out_weight;
			}
			// println!("this_range: {:?}\nchild_range: {:?}", &this_range, &child.players[self.player_index as usize].range);
		}
		for child in self.children.iter_mut() {
			unsafe {
				child.update_child_ranges(&*(&child.strategy as *const Strategy));
			}
		}
	}

	pub fn iteration(&mut self, target : usize, δ : f32) {
		if target == self.player_index as usize {
			let try_strategy = self.strategy.permutation_of(δ);
			self.update_child_ranges(&try_strategy);
			let new_ev = self.ev()[self.player_index as usize];
			if new_ev > self.ev {
				self.strategy = try_strategy;
				self.ev = new_ev;
			} else {
				// the compiler is retarded if it can't figure that self.strategy isn't modified in update_child_ranges making this two disjoint borrows
				unsafe {
					self.update_child_ranges(&*(&self.strategy as *const Strategy)); // undo the range update
				}
			}
		} else {
			self.ev = self.ev()[self.player_index as usize];
		}
		for child in &mut self.children {
			child.iteration(target, δ);
		}
	}

	/*fn exploitability_of(&self, target : usize) -> f32 {
		if self.children.len() == 0 {
			if self.player_index == target {
				return self.pot - self.ev();
			} else {
				return self.ev();
			}
		}
		let mut exploitability = 0.0;
		if self.player_index == target {
			let mut out_prob : Vec<f32> = vec![];
			for (i, action) in self.actions.iter().enumerate() {
				out_prob.push(
					self.strategy.0.iter()
						.map(|(card, weights)|
							weights[i] *
							self.players[self.player_index as usize].range.get(card).unwrap()
						)
						.sum::<f32>() /
						self.strategy.0.iter()
							.map(|(card, weights)|
								self.players[self.player_index as usize].range.get(card).unwrap() *
								weights.iter().sum::<f32>()
							).sum::<f32>()
				);
			}
			for (i, child) in self.children.iter().enumerate() {
				exploitability += child.exploitability_of(target) * out_prob[i];
			}
		}
		else {
			for (card, weight) in self.players[self.player_index]
		}
		exploitability
	}*/

	// no exploitation, just playing the tree as it is
	fn ev(&self) -> Vec<f32> {
		// leaf node, base case of the recursion
		if self.children.len() == 0 {
			// folded out
			if self.players.len() == 1 {
				return match self.players[0].seat {
					0 => vec![self.pot, 0.0],
					1 => vec![0.0, self.pot],
					_ => panic!(),
				};
			} else { // showdown
				return scale_f32_vec(&r_v_r(&self.players[0].range, &self.players[1].range), self.pot);
			}
		}
		let mut ev : Vec<f32> = vec![0.0; 2];
		let mut out_prob : Vec<f32> = vec![];
		for (i, action) in self.actions.iter().enumerate() {
			out_prob.push(
				self.strategy.0.iter()
					.map(|(card, weights)|
						weights[i] *
						self.players[self.player_index as usize].range.get(card).unwrap()
					)
					.sum::<f32>() /
					self.strategy.0.iter()
						.map(|(card, weights)|
							self.players[self.player_index as usize].range.get(card).unwrap() *
							weights.iter().sum::<f32>()
						).sum::<f32>()
			);
			match action {
				Action::Bet(bet) => ev[self.player_index as usize] -= bet * out_prob[out_prob.len()-1],
				Action::Call => ev[self.player_index as usize] -= self.action_change.unwrap().bet * out_prob[out_prob.len()-1],
				_ => (),
			}
		}
		for (i, child) in self.children.iter().enumerate() {
			ev = add_f32_vec(&ev, &scale_f32_vec(&child.ev(), out_prob[i]));
		}
		// println!("out_weights: {:?}\nev{:?}", out_prob, ev);
		// println!("node for above: {:?}", self);
		ev
	}
}

use rs_poker::core::{Card, Suit, Value};

fn main() {
	let configjson = env::args().nth(1).expect("action config json required");
	let actionset = parse_config(&configjson).unwrap();
	// println!("{:#?}", actionset);
	let card_set = vec![
		Card::new(Value::Ace, Suit::Diamond),
		Card::new(Value::King, Suit::Diamond),
		Card::new(Value::Queen, Suit::Diamond),
	];
	let mut root = Node::init(1.0, &actionset, card_set);
	for _ in 0..10000 {
		root.iteration(0, 0.2);
		// root.iteration(1, 0.3);
	}
	println!("{:#?}", root);
}
