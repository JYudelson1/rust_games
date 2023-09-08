#![feature(generic_const_exprs)]

use std::{fmt::Debug, rc::Rc};

#[derive(Debug)]
pub enum GameResult {
    Winner(String),
    Tie,
    Ranking(Vec<String>),
}

pub trait Game: Clone {
    type Move: Copy;
    type Board;

    const NUM_PLAYERS: usize;
    const BOARD_SIZE: usize; //TODO: Make this concept more generic?
    const CHANNELS: usize;

    fn new() -> Self;
    fn print(&self);
    fn nn_input(&self) -> dfdx::tensor::Tensor3D<{Self::CHANNELS}, {Self::BOARD_SIZE}, {Self::BOARD_SIZE}>;
    fn legal_moves(&self) -> Vec<Self::Move>;
    fn make_move(&mut self, m: Self::Move);
    fn is_over(&self) -> bool;
    fn play_full_game<'a>(players: Vec<&Strategy<Self>>, verbose: bool) -> GameResult
    where
        Self: Sized;
}

pub enum PlayerError {
    NoLegalMoves,
}

pub trait Player<G: Game> {
    fn new() -> Self
    where
        Self: Sized;
    fn choose_move(&self, game: &G) -> Result<G::Move, PlayerError>;
}

//TODO: Playing against each other, ELO, etc
#[derive(Copy, Clone, Debug)]
pub struct Elo {
    pub rating: f64,
    pub games_played: usize,
    pub wins: usize
}

impl Elo {
    fn new() -> Self {
        Elo {
            rating: 1200.0,
            games_played: 0,
            wins: 0
        }
    }
}

#[derive(Clone)]
pub struct Strategy<G: Game> {
    pub name: String,
    pub player: Rc<dyn Player<G>>,
    pub elo: Elo,
}

impl<G: Game> Strategy<G> {
    pub fn new<P: 'static + Player<G>>(name: String) -> Self {
        Strategy {
            name: name,
            player: Rc::new(P::new()),
            elo: Elo::new(),
        }
    }
}

impl<G: Game> Debug for Strategy<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Strategy")
            .field("name", &self.name)
            .field("elo", &self.elo)
            .finish()
    }
}
