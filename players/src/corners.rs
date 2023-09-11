use rand::seq::SliceRandom;
use rust_games_games::Othello;
use rust_games_shared::{Game, Player, PlayerError};

pub struct Corners;

impl Corners {
    pub fn new() -> Corners {
        Corners
    }
}

impl Player<Othello> for Corners {
    fn choose_move(&self, game: &Othello) -> Result<<Othello as Game>::Move, PlayerError> {
        let mut rng = rand::thread_rng();

        let moves = game.legal_moves();
        let edges: Vec<<Othello as Game>::Move> =
            moves.iter().filter(|m| m.is_edge()).copied().collect();
        let corners: Vec<<Othello as Game>::Move> =
            moves.iter().filter(|m| m.is_corner()).copied().collect();

        let m = if !corners.is_empty() {
            corners.choose(&mut rng)
        } else if !edges.is_empty() {
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
