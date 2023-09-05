use super::super::games::othello::OthelloMove;
use crate::Othello;
use rust_games::{Game, Player, PlayerError};

pub struct Greedy;

impl Player<Othello> for Greedy {
    fn new() -> Greedy {
        Greedy
    }
    fn choose_move(&self, game: &Othello) -> Result<<Othello as Game>::Move, PlayerError> {
        let mut best_move: Option<<Othello as Game>::Move> = None;
        let mut flipped_of_best = 0;

        for x in 0..8 {
            for y in 0..8 {
                let new_move = OthelloMove::new(x, y);
                let amt = game.tiles_would_flip(new_move).len();
                if amt > flipped_of_best {
                    best_move = Some(new_move);
                    flipped_of_best = amt;
                }
            }
        }

        match best_move {
            Some(mv) => Ok(mv),
            None => Err(PlayerError::NoLegalMoves),
        }
    }
}
