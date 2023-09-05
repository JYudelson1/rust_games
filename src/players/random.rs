use rand::seq::SliceRandom;
use rust_games::{Game, Player, PlayerError};

use std::marker::PhantomData;

pub struct Random<G: Game> {
    game: PhantomData<G>,
}

impl<G: Game> Player<G> for Random<G> {
    fn new() -> Random<G> {
        Random { game: PhantomData}
    }
    fn choose_move(&self, game: &G) -> Result<<G as Game>::Move, PlayerError> {
        let moves = game.legal_moves();

        let mut rng = rand::thread_rng();
        
        let m = moves.choose(&mut rng);
        match m {
            Some(mv) => Ok(*mv),
            None => Err(PlayerError::NoLegalMoves)
        }
    }
}
