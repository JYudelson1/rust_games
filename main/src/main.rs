#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use alphazero::BoardGameModel;
use dfdx::prelude::*;
use rust_games_games::Othello;
use rust_games_main::Leaderboard;
use rust_games_players::{AlphaZero, Corners, Greedy, Random};
use rust_games_shared::{Game, Strategy};

type Inp<G: Game> = Tensor<(Const<{ G::CHANNELS }>, G::BoardSize, G::BoardSize), f32, AutoDevice>;
//dfdx::tensor::Trace<f32, AutoDevice>>::Traced
type Test<G> = BoardGameModel<G>;

type Built<G> = <Test<G> as BuildOnDevice<AutoDevice, f32>>::Built;

type Out<G> = <Built<G> as ModuleMut<<Inp<G> as Trace<f32, AutoDevice>>::Traced>>::Output;

fn main() {
    print!("{}", std::any::type_name::<Out<Othello>>());
    assert!(false);
    let corner_player = Strategy::new("Corners".to_string(), Corners::new());
    let rand_player_1 = Strategy::new("Random1".to_string(), Random::<Othello>::new());
    let rand_player_2: Strategy<Othello> = Strategy::new("Random2".to_string(), Random::new());
    let greedy_player = Strategy::new("Greedy".to_string(), Greedy::new());

    let dev: AutoDevice = Default::default();

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
    arena.play_random_games(2000);
    arena.print();
}
