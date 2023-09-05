use super::super::games::othello::OthelloMove;
use crate::Othello;
use rand::seq::SliceRandom;
use rust_games::{Game, Player, PlayerError};

pub struct Corners;

impl Player<Othello> for Corners {
    fn new() -> Corners {
        Corners
    }
    fn choose_move(&self, game: &Othello) -> Result<<Othello as Game>::Move, PlayerError> {
        let mut rng = rand::thread_rng();

        let moves = game.legal_moves();
        let edges: Vec<OthelloMove> = moves.iter().filter(|m| m.is_edge()).copied().collect();
        let corners: Vec<OthelloMove> = moves.iter().filter(|m| m.is_corner()).copied().collect();

        let m = if !corners.is_empty(){
            corners.choose(&mut rng)
        } else if !edges.is_empty(){
            edges.choose(&mut rng)
        } else {
            moves.choose(&mut rng)
        };
        match m {
            Some(mv) => Ok(*mv),
            None => Err(PlayerError::NoLegalMoves),
        }
    }
}
