#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod leaderboard;

use alphazero::{BoardGameModel, MCTS};
use dfdx::prelude::Cpu;
use leaderboard::Leaderboard;
use rust_games_games::Othello;
use rust_games_players::{AlphaZero, Corners, Greedy, Random};
use rust_games_shared::{Game, Strategy};

fn main() {
    let corner_player = Strategy::new("Corners".to_string(), Corners::new());
    let rand_player_1 = Strategy::new("Random1".to_string(), Random::<Othello>::new());
    let rand_player_2 = Strategy::new("Random2".to_string(), Random::new());
    let greedy_player = Strategy::new("Greedy".to_string(), Greedy::new());

    let dev: Cpu = Default::default();
    let mcts = MCTS::new_from_file::<BoardGameModel<Othello>>(
        Othello::new(),
        1.0,
        "test.safetensors",
        &dev,
    );
    let dumb_az_player = Strategy::new(
        "Dumb Alphazero".to_string(),
        AlphaZero { mcts: mcts.into() },
    );

    let players = vec![
        corner_player,
        rand_player_1,
        rand_player_2,
        greedy_player,
        dumb_az_player,
    ];

    let mut arena = Leaderboard::new(players);
    arena.print();
    arena.play_random_games(500);
    arena.print();
}
