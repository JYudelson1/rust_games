#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use alphazero::{re_init_best_and_latest, BoardGameModel};
use rust_games_games::Othello;
use test_new::test_new_model;

mod games_list;
mod get_train_examples;
mod self_play;
mod test_new;

fn main() {
    type G = Othello;
    const TRAIN_ITER: usize = 15;
    // Full train loop

    // First, randomize "best" and "latest"
    let data_dir = "/Applications/Python 3.4/MyScripts/rust_games/data";
    re_init_best_and_latest::<G>(data_dir);

    // Then, any number of times
    for i in 0..TRAIN_ITER {
        println!("Iter {} of training.", i);
        println!("Gathering Examples...");
        self_play::self_play_iteration::<Othello, BoardGameModel<Othello>>("latest", data_dir, "latest", 100);
        let res = test_new_model::<Othello, BoardGameModel<Othello>>("best", "latest", data_dir, Some("best"), 5);
        println!("{:?}", res);
    }
}
