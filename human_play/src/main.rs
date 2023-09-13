#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use alphazero::BoardGameModel;
use dfdx::tensor::Cpu;
use rust_games_games::Othello;
use rust_games_players::*;
use rust_games_shared::{Game, Strategy};

fn main() {
    type G = Othello;

    let human_player: Strategy<G> = Strategy::new("Human".to_string(), Human::new());

    // let bot = Strategy::new("Bot".to_string(), Corners);
    let dev: Cpu = Default::default();
    let bot = Strategy::new(
        "AlphaZero Best".to_string(),
        AlphaZero::new_from_file::<BoardGameModel<G>>("best", 1.0, &dev, false),
    );

    let players = vec![&human_player, &bot];

    let result = G::play_full_game(players, true);
    println!("{:?}", result);
}
