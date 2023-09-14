#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod games_list;
mod get_train_examples;
mod test_new;
mod train_utils;

use alphazero::{load_from_file, re_init_best_and_latest, BoardGameModel, MCTSConfig};
use dfdx::{optim::Adam, prelude::BuildOnDevice, tensor::AutoDevice};
use games_list::GamesHolder;
use indicatif::{ProgressBar, ProgressStyle};
use rust_games_games::Othello;
use test_new::test_new_model;
use train_utils::update_from_gamesholder;

fn main() {
    type G = Othello;
    const TRAIN_ITER: usize = 15;
    const BATCH_SIZE: usize = 20; // AlphaGo: 2048
    const NUM_BATCHES: usize = 100; // AlphaGo: 1000
    const CAPACITY: usize = 10_000; // AlphaGo: 500_000
    const NUM_SELF_PLAY_GAMES: usize = 1; // AlphaGo: 25_000
    const NUM_TEST_GAMES: usize = 1; // AlphaGo: ?? Maybe 20?

    let training_games_cfg = MCTSConfig {
        traversal_iter: 100, // AlphaGo: 1600
        temperature: 1.0,    // AlphaGo: 1.0, but lowers over time
    };
    let test_games_cfg = MCTSConfig {
        traversal_iter: 100, // AlphaGo: 1600
        temperature: 1.0,    // AlphaGo: 1.0, but lowers over time
    };
    // Full train loop

    // First, randomize "best" and "latest"
    let data_dir = "/Applications/Python 3.4/MyScripts/rust_games/data";
    re_init_best_and_latest::<G>(data_dir);
    let mut gh: GamesHolder<G> = GamesHolder {
        games: vec![],
        capacity: CAPACITY,
    };

    let dev: AutoDevice = Default::default();
    let mut model =
        load_from_file::<G, BoardGameModel<G>>(&format!("{}/latest.safetensors", data_dir), &dev);
    let mut opt: Adam<<BoardGameModel<G> as BuildOnDevice<AutoDevice, f32>>::Built, f32, AutoDevice>  = Adam::new(&model, Default::default());

    // Then, any number of times
    let progress_bar = ProgressBar::new(TRAIN_ITER as u64).with_style(
        ProgressStyle::default_bar()
            .template("Train Iters |{wide_bar}| {pos}/{len} [{elapsed_precise}>{eta_precise}]")
            .unwrap(),
    );
    progress_bar.inc(0);
    for _ in 0..TRAIN_ITER {
        // The current best player plays NUM_SELF_PLAY_GAMES against itself
        gh.add_n_games::<BoardGameModel<G>>("best", data_dir, NUM_SELF_PLAY_GAMES, &training_games_cfg);

        // Then, train the network on NUM_BATCHES batches of examples, each of size BATCH_SIZE
        update_from_gamesholder(
            &mut model, 
            &mut opt, 
            &dev, 
            &gh, 
            BATCH_SIZE, 
            NUM_BATCHES,
        );

        // Play the current network against the best network,
        let res = test_new_model::<G, BoardGameModel<G>>(
            &model,
            "best",
            data_dir,
            Some("best"),
            NUM_TEST_GAMES,
            &test_games_cfg
        );

        //Print the winner of this iteration
        println!("{:?}", res);

        // Progress bar
        progress_bar.inc(1);
    }
    progress_bar.finish();
}
