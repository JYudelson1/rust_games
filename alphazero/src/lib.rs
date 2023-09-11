#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod mcts;
mod nn;
mod train;

pub use mcts::MCTS;
pub use nn::BoardGameModel;
