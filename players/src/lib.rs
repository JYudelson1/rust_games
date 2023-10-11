#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
mod corners;
mod greedy;
mod human;
mod random;

pub use corners::Corners;
pub use greedy::Greedy;
pub use human::Human;
pub use random::Random;
