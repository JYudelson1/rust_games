#![allow(incomplete_features)]
#![allow(type_alias_bounds)]
#![feature(generic_const_exprs)]

mod az_player;
mod mcts;
mod nn;
mod train;

pub use az_player::AlphaZeroPlayer;
pub use mcts::{MCTSConfig, MCTS};
pub use nn::{load_from_file, re_init_best_and_latest, BoardGameModel};
pub use train::{update_on_batch, update_on_many, TrainingExample, UnfinishedTrainingExample};
