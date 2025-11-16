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
		if bet_level == 0 {
			if usize::from(street) < self.0.len() {
				return self.0[street as usize].open.clone();
			}
		} else {
			if usize::from(bet_level-1) < self.0[street as usize].facing_bet.len() {
				return self.0[street as usize].facing_bet[usize::from(bet_level-1)].clone()
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
	fn init(actionset: &ActionSet, street : u8, level : u8, range : Vec<Card>) -> Self {
		let mut strat : Self = Strategy(HashMap::new());
		for card in range {
			strat.0.insert(card, vec![]);
			let actions = actionset.actions_at(street.into(), level.into());
			#[allow(non_snake_case)]
			let A = actions.len() as f32;
			for _ in actions {
				strat.0.get_mut(&card).unwrap().push(1.0 / A);
			}
		}
		strat
	}
	fn permutation_of(&self, delta : f32) -> Self {
        let mut new = self.clone();
        use rand::{Rng, seq::IteratorRandom};
        let mut rng = rand::thread_rng();
		// pick random card, get its weights
        let (_card, weights) = new.0.iter_mut().choose(&mut rng).unwrap();
		// increase/decrease one of the weights
        let idx = rng.gen_range(0..weights.len());
        let sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        weights[idx] = (weights[idx] + sign * delta).max(0.0);
		// renormalise
        let sum: f32 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= sum;
        }
        new
    }
}

#[derive(Copy, Clone)]
struct LastChange {
	player_index : u8,
	bet : f32,
}

#[derive(Debug)]
struct Node {
	pot : f32,
	player_index : u8,
	strategy : Strategy,
	prev_ev : f32,
	actions : Vec<Action>,
	players : Vec<Player>, // make this a reference to the parent
	children : Vec<Node>,
}

fn scale_f32_vec(a : &Vec<f32>, μ : f32) -> Vec<f32> {
	a.iter().map(|x| x * μ).collect()
}

fn add_f32_vec(a : &Vec<f32>, b : &Vec<f32>) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn sub_f32_vec(a : &Vec<f32>, b : &Vec<f32>) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn sub_strategy_weights(a : &Strategy, b : &Strategy) -> Strategy {
	let mut diff = (*a).clone();
	for (card, weights) in diff.0.iter_mut() {
		(*weights) = sub_f32_vec(weights, b.0.get(&card).unwrap());
	}
	diff
}

impl Node {

	pub fn init(pot : f32, actionset : &ActionSet, deck : Vec<Card>) -> Self {
		let mut root = Self::new(
			pot,
			0,
			Strategy::init(actionset, 0, 0, deck.clone()),
			actionset.actions_at(0, 0),
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
				}
			},
			None => {
				action_change = Some(LastChange {player_index: 0, bet: 0.0});
			},
		}
		// useful for making the branches
		let n_player_index = if usize::from(self.player_index+1) < self.players.len() {self.player_index + 1} else {0};
		let n_strategy = Strategy::init(actionset, street, bet_level, self.players[n_player_index as usize].range.keys().cloned().collect());
		let num_actions = actionset.actions_at(street, bet_level).len();
		let mut n_players = self.players.clone();
		for (_, j) in n_players[self.player_index as usize].range.iter_mut() {
			(*j) /= num_actions as f32;
		}
		// let n_actions = actionset.actions_at(street, bet_level);
		let mut new_children : Vec<(f32, u8, Option<LastChange>, u8, u8, Strategy, Vec<Action>, Vec<Player>)> = Vec::new();
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
							n_strategy.clone(),
							actionset.actions_at(street, bet_level),
							n_players.clone(),
						)
					);
				}
				Action::Bet(value) => {
					new_children.push(
						(
							self.pot+value,
							n_player_index,
							Some(LastChange {
								player_index: self.player_index,
								bet :1.0},
							),
							street,
							bet_level+1,
							n_strategy.clone(),
							actionset.actions_at(street, bet_level+1),
							n_players.clone(),
						)
					);
				}
				Action::Fold => {
					let mut n_players = n_players.clone();
					n_players.remove(self.player_index as usize);
					new_children.push(
						(
							self.pot,
							n_player_index,
							action_change,
							street,
							bet_level,
							n_strategy.clone(),
							actionset.actions_at(street, bet_level),
							n_players,
						)
					);
				}
				Action::Call => {
					new_children.push(
						(
							self.pot+1.0,
							n_player_index,
							action_change,
							street,
							bet_level,
							n_strategy.clone(),
							actionset.actions_at(street, bet_level),
							n_players.clone(),
						)
					);
				}
			}
		}
		let mut i = 0;
		for (n_pot, n_player_index, n_action, street, bet_level, n_strategy, n_actions, n_players) in new_children {
			self.children.push(
				Self::new(
					n_pot, n_player_index, n_strategy, n_actions, n_players
				)
			);
			self.children[i as usize].populate_children(actionset, n_action, street, bet_level);
			i += 1;
		}
	}

	fn new(pot : f32, player_index : u8, strategy : Strategy, actions : Vec<Action>, players : Vec<Player>) -> Self {
		Self {
			pot,
			player_index,
			strategy,
			prev_ev : f32::MIN,
			actions,
			players,
			children : Vec::new(),
		}
	}

	pub fn iteration(&mut self) {
		let old_strategy = self.strategy.permutation_of(0.1);
		let delta = sub_strategy_weights(&self.strategy, &old_strategy);
		let new_evs = self.ev(delta);
		let new_ev = new_evs[self.player_index as usize];
		if new_ev < self.prev_ev {
			self.prev_ev = new_ev; // update previous result
		} else {
			self.strategy = old_strategy; // undo the permutation
		}
		for child in &mut self.children {
			child.iteration();
		}
	}

	fn ev(self : &Self, Δrange : Strategy) -> Vec<f32> {
		Δrange;
		vec![100000.0, 100000.0]
		// if self.children.len() == 0 {
		// 	return (self.player, self.pot * r_v_r(&self.players[0].range, &self.players[1].range));
		// }
		// for child in self.children {
		// 	// return highest of the returned EVs that match the node's player
		// 	let temp = child.ev();
		// 	if (self.player
		// }
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
	for _ in 0..1000 {
		root.iteration();
	}
	println!("{:#?}", root);
}
