#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use alphazero::{AlphaZeroPlayer, BoardGameModel};
use dfdx::tensor::AutoDevice;
use rust_games_games::{Connect4, Othello};
use rust_games_players::*;
use rust_games_shared::{Game, Strategy};

fn main() {
    type G = Connect4;

    let human_player: Strategy<G> = Strategy::new("Human".to_string(), Human::new());

    // let bot = Strategy::new("Bot".to_string(), Corners);
    // let data_dir = "/Applications/Python 3.4/MyScripts/rust_games/data";
    // let dev: AutoDevice = Default::default();
    // let bot = Strategy::new(
    //     "AlphaZero Best".to_string(),
    //     AlphaZeroPlayer::new_from_file::<BoardGameModel<G>>("best",data_dir, 1.0, &dev, false, 100),
    // );
    let bot = Strategy::new("Bot".to_string(), Random::<G>::new());

    let players = vec![&bot, &human_player];

    let result = G::play_full_game(players, true);
    println!("{:?}", result);
}
