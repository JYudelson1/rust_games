#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
mod alphazero;
mod corners;
mod greedy;
mod random;

pub use alphazero::AlphaZero;
pub use corners::Corners;
pub use greedy::Greedy;
pub use random::Random;
