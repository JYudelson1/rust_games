#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::{
    any::Any,
    collections::HashMap,
    fmt::{Debug, Display},
    hash::Hash,
    rc::Rc,
};

use dfdx::prelude::{Tensor, ConstDim, AutoDevice, Const};

#[derive(Debug)]
pub enum GameResult {
    Winner((PlayerId, String)),
    Tie,
    Ranking(Vec<(PlayerId, String)>),
}

#[derive(Debug, Copy, Clone, PartialEq, Hash, Eq)]
pub enum PlayerId {
    //TODO: Multiplayer?
    First,
    Second,
}

pub trait Game: Clone + Debug {
    type Move: Copy + Debug + Display + Hash + Eq;
    type Board: Eq;

    type BoardSize: ConstDim;
    type TotalBoardSize: ConstDim;

    const CHANNELS: usize;
    const NUM_PLAYERS: usize;
    const TOTAL_MOVES: usize;

    fn new() -> Self;
    fn print(&self);
    fn to_nn_input(&self) -> Tensor<(Const<{Self::CHANNELS}>, Self::BoardSize, Self::BoardSize), f32, AutoDevice>;
    fn get_board(&self) -> Self::Board;
    fn legal_moves(&self) -> Vec<Self::Move>;
    fn make_move(&mut self, m: Self::Move);
    fn is_over(&self) -> bool;
    fn get_result(&self) -> Option<GameResult>;
    fn current_player(&self) -> PlayerId;

    fn all_possible_moves() -> [Self::Move; Self::TOTAL_MOVES];

    fn associate_players(players: Vec<&Strategy<Self>>) -> HashMap<PlayerId, &Strategy<Self>>;
    fn play_full_game<'a>(players: Vec<&Strategy<Self>>, verbose: bool) -> GameResult
    where
        Self: Sized,
    {
        assert!(players.len() == Self::NUM_PLAYERS);
        let player_map: HashMap<PlayerId, &Strategy<Self>> = Self::associate_players(players);

        let mut game = Self::new();
        while !game.is_over() {
            if verbose {
                game.print();
            }

            let current_player = game.current_player();
            let next_move = (*player_map.get(&current_player).unwrap())
                .player
                .choose_move(&game);
            match next_move {
                Ok(m) => {
                    game.make_move(m);
                }
                // No legal moves skips the turn
                Err(err) => {
                    panic!("Err: {:?}", err);
                }
            }

        }
        if verbose {
            game.print();
        }

        game.get_result().expect("Game should have finished!")
    }
}

#[derive(Debug)]
pub enum PlayerError {
    NoLegalMoves,
}

pub trait Player<G: Game> {
    fn choose_move(&self, game: &G) -> Result<G::Move, PlayerError>;
    fn reset(&mut self) {}
    fn as_any(&self) -> &dyn Any;
}

#[derive(Copy, Clone, Debug)]
pub struct Elo {
    pub rating: f64,
    pub games_played: usize,
    pub wins: usize,
}

impl Elo {
    fn new() -> Self {
        Elo {
            rating: 1200.0,
            games_played: 0,
            wins: 0,
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
    pub fn new<P: 'static + Player<G>>(name: String, player: P) -> Self {
        Strategy {
            name: name,
            player: Rc::new(player),
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
