use rust_games_games::Othello;
use rust_games_games::OthelloMove;
use rust_games_shared::{Game, Player, PlayerError};

pub struct Greedy;

impl Greedy {
    pub fn new() -> Greedy {
        Greedy
    }
}

impl Player<Othello> for Greedy {
    fn choose_move(&self, game: &Othello) -> Result<<Othello as Game>::Move, PlayerError> {
        let mut best_move: Option<<Othello as Game>::Move> = None;
        let mut flipped_of_best: Option<usize> = None;

        for mv in game.legal_moves() {
            match mv {
                OthelloMove::Pass => return Ok(mv),
                OthelloMove::Move(_, _) => {
                    let amt = game.tiles_would_flip(mv).len();
                    match flipped_of_best {
                        Some(flipped_num) => {
                            if amt > flipped_num {
                                best_move = Some(mv);
                                flipped_of_best = Some(amt);
                            }
                        }
                        None => {
                            flipped_of_best = Some(amt);
                            best_move = Some(mv);
                        }
                    }
                }
            }
        }

        match best_move{
            Some(mv) => Ok(mv),
            None => Err(PlayerError::NoLegalMoves),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
