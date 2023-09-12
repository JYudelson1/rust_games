#![allow(incomplete_features)]
#![allow(type_alias_bounds)]
#![feature(generic_const_exprs)]

mod mcts;
mod nn;
mod train;

pub use mcts::MCTS;
pub use nn::BoardGameModel;
pub use train::{update_on_many, TrainingExample};
