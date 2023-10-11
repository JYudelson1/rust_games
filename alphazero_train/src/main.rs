#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod games_list;
mod get_train_examples;
mod test_new;
mod train_utils;

use alphazero::{
    load_from_file, re_init_best_and_latest, AlphaZeroPlayer, BoardGameModel, MCTSConfig,
};
use clap::Parser;
use dfdx::{optim::Adam, prelude::*};
use games_list::GamesHolder;
use indicatif::{ProgressBar, ProgressStyle};
use rust_games_games::Othello;
use rust_games_main::Leaderboard;
use rust_games_players::Corners;
use rust_games_shared::Strategy;
use test_new::test_new_model;
use train_utils::update_from_gamesholder;

#[derive(Parser)]
struct TrainArgs {
    #[arg(short, long, default_value_t = 10)]
    train_iter: usize,

    #[arg(short, long)]
    batch_size: usize,

    #[arg(short, long)]
    num_batches: usize,

    #[arg(short, long, default_value_t = 4_000)]
    capacity: usize,

    #[arg(short, long)]
    num_self_play_games: usize,

    #[arg(short, long)]
    num_test_games: usize,

    #[arg(short, long)]
    training_traversal_iter: usize,

    #[arg(short, long, default_value_t = 1.0)]
    training_temp: f32,

    #[arg(short, long)]
    test_traversal_iter: usize,

    #[arg(short, long, default_value_t = 0.001)]
    test_temp: f32,
}

fn main() {
    type G = Othello;

    let args = TrainArgs::parse();

    // const TRAIN_ITER: usize = 15;
    // const BATCH_SIZE: usize = 40; // AlphaGo: 2048
    // const NUM_BATCHES: usize = 20; // AlphaGo: 1000
    // const CAPACITY: usize = 4_000; // AlphaGo: 500_000
    // const NUM_SELF_PLAY_GAMES: usize = 2; // AlphaGo: 25_000
    // const NUM_TEST_GAMES: usize = 2; // AlphaGo: ?? Maybe 20?

    let training_games_cfg = MCTSConfig {
        traversal_iter: args.training_traversal_iter, // AlphaGo: 1600
        temperature: args.training_temp, // AlphaGo: 1.0, but lowers over time
    };
    let test_games_cfg = MCTSConfig {
        traversal_iter: args.test_traversal_iter, // AlphaGo: 1600
        temperature: args.test_temp,    // AlphaGo: 1.0, but lowers over time
    };
    ////// Full train loop

    //// First, randomize "best" and "latest"
    let dev: AutoDevice = AutoDevice::seed_from_u64(0);
    let data_dir = "/Applications/Python 3.4/MyScripts/rust_games/data";
    re_init_best_and_latest::<G>(data_dir, &dev);

    let mut gh: GamesHolder<G> = GamesHolder {
        games: vec![],
        capacity: args.capacity,
    };

    let mut model =
        load_from_file::<G, BoardGameModel<G>>(&format!("{}/latest.safetensors", data_dir), &dev);
    let mut opt: Adam<
        <BoardGameModel<G> as BuildOnDevice<AutoDevice, f32>>::Built,
        f32,
        AutoDevice,
    > = Adam::new(
        &model,
        AdamConfig {
            lr: 1e-3,                                  //TODO: Try futzing
            weight_decay: Some(WeightDecay::L2(1e-3)), // AlphaGo param: 1e-4 TODO: Try fucking around
            ..Default::default()
        },
    );

    //// Then, any number of times
    let progress_bar = ProgressBar::new(args.train_iter as u64).with_style(
        ProgressStyle::default_bar()
            .template("Train Iters |{wide_bar}| {pos}/{len} [{elapsed_precise}>{eta_precise}]")
            .unwrap(),
    );
    progress_bar.inc(0);
    for _ in 0..args.train_iter {
        // The current best player plays NUM_SELF_PLAY_GAMES against itself
        gh.add_n_games::<BoardGameModel<G>>("best", data_dir, args.num_self_play_games, &training_games_cfg);

        // Then, train the network on NUM_BATCHES batches of examples, each of size BATCH_SIZE
        update_from_gamesholder(
            &mut model, 
            &mut opt, 
            &dev, 
            &gh, 
            args.batch_size, 
            args.num_batches,
        );

        //// Play the current network against the best network
        // 
        // Save & load the model to end up with a "clean" validation model,
        // untracked by optimizer. This makes it much faster.
        model.save_safetensors(format!("{}/latest.safetensors", data_dir)).unwrap();
        let test_model = load_from_file::<G, BoardGameModel<G>>(&format!("{}/latest.safetensors", data_dir), &dev);
        let res = test_new_model::<G, BoardGameModel<G>>(
            &test_model,
            "best",
            data_dir,
            args.num_test_games,
            &test_games_cfg
        );

        // If the new model won, save it to best
        match res {
            test_new::NewModelResults::OldModelWon(_) => {},
            test_new::NewModelResults::NewModelWon(_) => {
                model.save_safetensors(format!("{}/best.safetensors", data_dir)).unwrap();
            },
        }

        //Print the winner of this iteration
        println!("{:?}", res);

        //// Play some games against Corners
        let bot = Strategy::new(
            "AlphaZero Best".to_string(),
            AlphaZeroPlayer::new_from_file::<BoardGameModel<G>>("best",data_dir, 1.0, &dev, false, 100),
        );

        let corner_player = Strategy::new("Corners".to_string(), Corners::new());

        let players = vec![bot, corner_player];

        let mut arena = Leaderboard::new(players);
        arena.play_random_games(args.num_test_games);
        arena.print();
        

        // Progress bar
        progress_bar.inc(1);
    }
    progress_bar.finish();
}
