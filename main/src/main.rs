#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use alphazero::BoardGameModel;
use dfdx::prelude::*;
use rust_games_games::{Othello, TicTacToe};
use rust_games_main::Leaderboard;
use rust_games_players::{AlphaZero, Corners, Greedy, Random};
use rust_games_shared::Strategy;

fn main() {
    let dev: AutoDevice = Default::default();

    let data_dir = "/Applications/Python 3.4/MyScripts/rust_games/data";
    const AZ_DEPTH: usize = 50;
    type G = Othello;

    let corner_player = Strategy::new("Corners".to_string(), Corners::new());
    let rand_player_1 = Strategy::new("Random1".to_string(), Random::<G>::new());
    let rand_player_2: Strategy<G> = Strategy::new("Random2".to_string(), Random::new());
    let greedy_player = Strategy::new("Greedy".to_string(), Greedy::new());

    let dumb_az_player = Strategy::new(
        "Dumb Alphazero".to_string(),
        AlphaZero::new_from_file::<BoardGameModel<G>>("control", data_dir, 1.0, &dev, false, AZ_DEPTH),
    );

    let az_player_best = Strategy::new(
        "AlphaZero Best".to_string(),
        AlphaZero::new_from_file::<BoardGameModel<G>>("best", data_dir, 1.0, &dev, false, AZ_DEPTH),
    );

    let players = vec![
        corner_player,
        rand_player_1,
        rand_player_2,
        greedy_player,
        dumb_az_player,
        az_player_best,
    ];

    let mut arena = Leaderboard::new(players);
    arena.print();
    arena.play_random_games(500);
    arena.print();
}
