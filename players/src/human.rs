use dialoguer::{theme::ColorfulTheme, Select};
use rust_games_shared::{Game, Player, PlayerError};

use std::marker::PhantomData;

pub struct Human<G: Game> {
    game: PhantomData<G>,
}

impl<G: Game> Human<G> {
    pub fn new() -> Human<G> {
        Human { game: PhantomData }
    }
}

impl<G: Game + 'static> Player<G> for Human<G> {
    fn choose_move(&self, game: &G) -> Result<<G as Game>::Move, PlayerError> {
        let moves = game.legal_moves();

        if moves.is_empty() {
            return Err(PlayerError::NoLegalMoves);
        }

        let selection_idx = Select::with_theme(&ColorfulTheme::default())
            .items(&moves)
            .interact()
            .unwrap();

        Ok(moves[selection_idx])
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
