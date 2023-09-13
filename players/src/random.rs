use rand::seq::SliceRandom;
use rust_games_shared::{Game, Player, PlayerError};

use std::marker::PhantomData;

pub struct Random<G: Game> {
    game: PhantomData<G>,
}

impl<G: Game> Random<G> {
    pub fn new() -> Random<G> {
        Random { game: PhantomData }
    }
}

impl<G: Game + 'static> Player<G> for Random<G> {
    fn choose_move(&self, game: &G) -> Result<<G as Game>::Move, PlayerError> {
        let moves = game.legal_moves();

        if moves.is_empty() {
            return Err(PlayerError::NoLegalMoves);
        }

        let mut rng = rand::thread_rng();

        let m = moves.choose(&mut rng);
        Ok(*m.unwrap())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
