use dfdx::prelude::*;
use rust_games_shared::{Game, GameResult, Strategy};
use std::fmt;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum OthelloState {
    Empty,
    White,
    Black,
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum PlayerColor {
    White,
    Black,
}

impl PlayerColor {
    fn to_color(&self) -> OthelloState {
        match self {
            PlayerColor::White => OthelloState::White,
            PlayerColor::Black => OthelloState::Black,
        }
    }
}

impl fmt::Display for OthelloState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let icon = match self {
            OthelloState::Empty => "🟩",
            OthelloState::White => "⬜",
            OthelloState::Black => "⬛",
        };

        write!(f, "{}", icon)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OthelloMove {
    Pass,
    Move(usize, usize),
}

impl OthelloMove {
    pub fn is_corner(&self) -> bool {
        match self {
            OthelloMove::Pass => false,
            &OthelloMove::Move(x, y) => (x == 0 || x == 7) && (y == 0 || y == 7),
        }
    }

    pub fn is_edge(&self) -> bool {
        match self {
            OthelloMove::Pass => false,
            &OthelloMove::Move(x, y) => x == 0 || x == 7 || y == 0 || y == 7,
        }
        
    }
}

#[derive(Debug, Clone)]
pub struct Othello {
    board: [[OthelloState; 8]; 8],
    playing: PlayerColor,
    last_was_pass: bool
}

impl Othello {
    pub fn tiles_would_flip(&self, m: OthelloMove) -> Vec<OthelloMove> {
        let mut tiles = vec![];
        match m {
            OthelloMove::Pass => tiles,
            OthelloMove::Move(x, y) => {
                if self.board[y][x] != OthelloState::Empty {
                    return tiles;
                }
        // travel in each direction
        for dx in [-1i8, 0, 1] {
            for dy in [-1i8, 0, 1] {
                if (dx, dy) == (0, 0) {
                    continue;
                }

                for i in 1.. {
                    // If the surrounding tile is of the same color, or empty, or a wall, ignore it
                    if (y as i8 + dy * i) < 0
                        || (y as i8 + dy * i) > 7
                        || (x as i8 + dx * i) < 0
                        || (x as i8 + dx * i) > 7
                    {
                        break;
                    }
                    // The tile is in scope, so we don't need to worry about usizes going negative

                    // If the next tile's empty, terminate quietly
                    let tile_state =
                        self.board[(y as i8 + dy * i) as usize][(x as i8 + dx * i) as usize];

                    if tile_state == OthelloState::Empty {
                        break;
                    }
                    // If the next tile is of the opposite color, keep going
                    if tile_state != self.playing.to_color() {
                        continue;
                    }
                    // If of the player's color, terminate and add all the prev tiles to tiles
                    if tile_state == self.playing.to_color() {
                        for d in 1..i {
                            tiles.push(OthelloMove::Move((x as i8 + (dx * d)) as usize, (y as i8 + (dy * d)) as usize));
                        }
                        break;
                    }
                }
            }
        }
    tiles}
        }
}
}

impl Game for Othello {
    type Move = OthelloMove;

    type Board = [[OthelloState; 8]; 8];

    const NUM_PLAYERS: usize = 2;

    fn new() -> Othello {
        let mut g = Othello {
            board: [[OthelloState::Empty; 8]; 8],
            playing: PlayerColor::Black,
            last_was_pass: false
        };
        g.board[3][3] = OthelloState::Black;
        g.board[3][4] = OthelloState::White;
        g.board[4][3] = OthelloState::White;
        g.board[4][4] = OthelloState::Black;

        g
    }

    fn print(&self) {
        for row in self.board {
            for icon in row {
                print!("{}", icon);
            }
            println!()
        }
        println!("Currently playing: {}", self.playing.to_color())
    }

    fn legal_moves(&self) -> Vec<Self::Move> {
        let mut moves = vec![];

        for x in 0..8 {
            for y in 0..8 {
                let new_move = OthelloMove::Move(x, y);
                if !self.tiles_would_flip(new_move).is_empty() {
                    moves.push(new_move)
                }
            }
        }

        if moves.is_empty() {
            if !self.last_was_pass {
                moves.push(OthelloMove::Pass);
            }
        }
        moves
    }

    fn make_move(&mut self, m: Self::Move) {
        //println!("{:?}", m);
        match m {
            OthelloMove::Pass => {
                self.last_was_pass = true;
            }
            OthelloMove::Move(x, y) => {
                self.last_was_pass = false;
                for tile in self.tiles_would_flip(m) {
                    match tile {
                        OthelloMove::Pass => {},
                        OthelloMove::Move(x, y) => {self.board[y][x] = self.playing.to_color();}
                    }
                    
                }

                self.board[y][x] = self.playing.to_color();
            }
        }

        self.playing = match self.playing {
            PlayerColor::Black => PlayerColor::White,
            PlayerColor::White => PlayerColor::Black,
        };

        // self
    }

    fn play_full_game<'a>(strategies: Vec<&Strategy<Othello>>, verbose: bool) -> GameResult {
        assert!(strategies.len() == Self::NUM_PLAYERS);
        let black = &strategies[0].player;
        let white = &strategies[1].player;

        let mut game = Self::new();
        while !game.is_over() {
            let next_move = if game.playing == PlayerColor::Black {
                black.choose_move(&game)
            } else {
                white.choose_move(&game)
            };
            match next_move {
                Ok(m) => {
                    game.make_move(m);
                    if !matches!(m, OthelloMove::Pass) {}
                    if verbose {
                        game.print()
                    }
                }
                // No legal moves skips the turn
                Err(err) => {
                    panic!("Err: {:?}", err);
                }
            }

            if verbose {
                game.print();
            }
        }

        // Count up
        let mut black_tiles = 0;
        let mut white_tiles = 0;
        for x in 0..8 {
            for y in 0..8 {
                if game.board[y][x] == OthelloState::Black {
                    black_tiles += 1
                } else if game.board[y][x] == OthelloState::White {
                    white_tiles += 1
                }
            }
        }

        if black_tiles < white_tiles {
            GameResult::Winner(strategies[1].name.clone())
        } else if black_tiles == white_tiles {
            GameResult::Tie
        } else {
            GameResult::Winner(strategies[0].name.clone())
        }
    }

    fn is_over(&self) -> bool {
        if !self.legal_moves().is_empty() {
            return false;
        }
        if !self.last_was_pass {
            return false;
        }
        true
    }

    const BOARD_SIZE: usize = 8;

    const CHANNELS: usize = 3;

    fn to_nn_input(
        &self,
    ) -> Tensor3D<{ Self::CHANNELS }, { Self::BOARD_SIZE }, { Self::BOARD_SIZE }> {
        let dev: Cpu = Default::default();
        let mut black_channel = [[0.0; 8]; 8];
        for (x, row) in self.board.iter().enumerate() {
            for (y, tile) in row.iter().enumerate() {
                black_channel[x][y] = match tile {
                    OthelloState::Empty => 0.0,
                    OthelloState::White => 0.0,
                    OthelloState::Black => 1.0,
                }
            }
        }

        let mut white_channel = [[0.0; 8]; 8];
        for (x, row) in self.board.iter().enumerate() {
            for (y, tile) in row.iter().enumerate() {
                white_channel[x][y] = match tile {
                    OthelloState::Empty => 0.0,
                    OthelloState::White => 1.0,
                    OthelloState::Black => 0.0,
                }
            }
        }

        let player_num = match self.playing {
            PlayerColor::White => 1.0_f32,
            PlayerColor::Black => 0.0,
        };

        let player_channel = [[player_num; 8]; 8];

        dev.tensor([black_channel, white_channel, player_channel])
    }

    fn get_board(&self) -> Self::Board {
        self.board
    }
}

#[cfg(test)]
mod test {
    use super::Othello;
    use rust_games_shared::Game;

    #[test]
    fn empty_board() {
        Othello::new().print()
    }
}
