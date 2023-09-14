#![allow(incomplete_features)]
#![allow(type_alias_bounds)]
#![feature(generic_const_exprs)]

mod mcts;
mod nn;
mod train;

pub use mcts::{MCTS, MCTSConfig};
pub use nn::{load_from_file, re_init_best_and_latest, BoardGameModel};
pub use train::{update_on_batch, update_on_many, TrainingExample, UnfinishedTrainingExample};
