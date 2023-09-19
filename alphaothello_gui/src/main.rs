#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
mod othello_druid;

use alphazero::{load_from_file, BoardGameModel};
use dfdx::prelude::*;
use druid::{AppLauncher, WindowDesc};
use othello_druid::{analysis_board, SimpleOthello};
use rust_games_games::Othello;

fn main() {
    let dev: AutoDevice = AutoDevice::seed_from_u64(0);
    let data_dir = "/Applications/Python 3.4/MyScripts/rust_games/data/918";

    let model = load_from_file::<Othello, BoardGameModel<Othello>>(
        &format!("{}/best.safetensors", data_dir),
        &dev,
    );

    let window: WindowDesc<_> = WindowDesc::new(analysis_board)
        .window_size((800., 800.))
        .resizable(false)
        .title("Alpha Othello");
    let othello_state = SimpleOthello::new(model);
    AppLauncher::with_window(window)
        .launch(othello_state)
        .expect("launch failed");
}
