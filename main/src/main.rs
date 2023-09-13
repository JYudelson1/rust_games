#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use alphazero::BoardGameModel;
use dfdx::prelude::Cpu;
use rust_games_games::Othello;
use rust_games_main::Leaderboard;
use rust_games_players::{AlphaZero, Corners, Greedy, Random};
use rust_games_shared::Strategy;

fn main() {
    let corner_player = Strategy::new("Corners".to_string(), Corners::new());
    let rand_player_1 = Strategy::new("Random1".to_string(), Random::<Othello>::new());
    let rand_player_2: Strategy<Othello> = Strategy::new("Random2".to_string(), Random::new());
    let greedy_player = Strategy::new("Greedy".to_string(), Greedy::new());

    let dev: Cpu = Default::default();

    let dumb_az_player = Strategy::new(
        "Dumb Alphazero".to_string(),
        AlphaZero::new_from_file::<BoardGameModel<Othello>>("control", 1.0, &dev, false),
    );

    let az_player_best = Strategy::new(
        "AlphaZero Best".to_string(),
        AlphaZero::new_from_file::<BoardGameModel<Othello>>("best", 1.0, &dev, false),
    );

    let az_player_latest = Strategy::new(
        "AlphaZero latest".to_string(),
        AlphaZero::new_from_file::<BoardGameModel<Othello>>("latest", 1.0, &dev, false),
    );

    let players = vec![
        corner_player,
        rand_player_1,
        rand_player_2,
        greedy_player,
        dumb_az_player,
        az_player_best,
        az_player_latest
    ];

    let mut arena = Leaderboard::new(players);
    arena.print();
    arena.play_random_games(200);
    arena.print();
}
