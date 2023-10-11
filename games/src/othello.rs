use dfdx::prelude::*;
use rust_games_shared::{Game, GameResult, PlayerId, Strategy};
use std::{collections::HashMap, fmt};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum OthelloState {
    Empty,
    White,
    Black,
}

fn to_color(id: &PlayerId) -> OthelloState {
    match id {
        PlayerId::Second => OthelloState::White,
        PlayerId::First => OthelloState::Black,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OthelloMove {
    Pass,
    Move(usize, usize),
}

impl fmt::Display for OthelloMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OthelloMove::Pass => write!(f, "Pass"),
            OthelloMove::Move(x, y) => write!(f, "{} {}", x + 1, 8 - y),
        }
    }
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
    playing: PlayerId,
    last_was_pass: bool,
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
                    if tile_state != to_color(&self.playing) {
                        continue;
                    }
                    // If of the player's color, terminate and add all the prev tiles to tiles
                    if tile_state == to_color(&self.playing) {
                        for d in 1..i {
                            tiles.push(OthelloMove::Move((x as i8 + (dx * d)) as usize, (y as i8 + (dy * d)) as usize));
                        }
                        break;
                    }
                }
            }
        }
        tiles
            }
        }
    }

    pub fn new_from_board(board: [[OthelloState; 8]; 8], playing: PlayerId) -> Othello {
        Othello {
            board,
            playing,
            last_was_pass: false,
        }
    }
}

impl Game for Othello {
    type Move = OthelloMove;

    type Board = [[OthelloState; 8]; 8];

    const NUM_PLAYERS: usize = 2;
    const TOTAL_MOVES: usize = 64 /* Board spots */ + 1 /* Passing */;

    fn new() -> Othello {
        let mut g = Othello {
            board: [[OthelloState::Empty; 8]; 8],
            playing: PlayerId::First,
            last_was_pass: false
        };
        g.board[3][3] = OthelloState::Black;
        g.board[3][4] = OthelloState::White;
        g.board[4][3] = OthelloState::White;
        g.board[4][4] = OthelloState::Black;

        g
    }

    fn print(&self) {
        for (i, row) in self.board.iter().enumerate() {
            print!("{}", 8 - i);
            for icon in row {
                print!("{}", icon);
            }
            println!()
        }
        println!("  1 2 3 4 5 6 7 8");
        println!("Currently playing: {}", to_color(&self.playing))
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
                        OthelloMove::Move(x, y) => {self.board[y][x] = to_color(&self.playing);}
                    }
                    
                }

                self.board[y][x] = to_color(&self.playing);
            }
        }

        self.playing = match self.playing {
            PlayerId::First => PlayerId::Second,
            PlayerId::Second => PlayerId::First,
        };

        // self
    }

    //fn play_full_game<'a>(strategies: Vec<&Strategy<Self>>, verbose: bool) -> GameResult 

    fn is_over(&self) -> bool {
        if !self.legal_moves().is_empty() {
            return false;
        }
        if !self.last_was_pass {
            return false;
        }
        true
    }

    type BoardSize = Const<8>;
    type TotalBoardSize = Const<64>;
    const CHANNELS: usize = 3;

    fn to_nn_input(
        &self,
    ) -> Tensor<(Const<{Self::CHANNELS}>, Self::BoardSize, Self::BoardSize), f32, AutoDevice>{
        let dev: AutoDevice = Default::default();
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
            PlayerId::First => 1.0_f32,
            PlayerId::Second => 0.0,
        };

        let player_channel = [[player_num; 8]; 8];

        dev.tensor([black_channel, white_channel, player_channel])
    }

    fn get_board(&self) -> Self::Board {
        self.board
    }

    fn get_result(&self) -> Option<GameResult> {
        // Make sure the game is over
        if !self.is_over() {
            return None;
        }

        // Count up
        let mut black_tiles = 0;
        let mut white_tiles = 0;
        for x in 0..8 {
            for y in 0..8 {
                if self.board[y][x] == OthelloState::Black {
                    black_tiles += 1
                } else if self.board[y][x] == OthelloState::White {
                    white_tiles += 1
                }
            }
        }

        if black_tiles < white_tiles {
            Some(GameResult::Winner((PlayerId::Second, "".to_string())))
        } else if black_tiles == white_tiles {
            Some(GameResult::Tie)
        } else {
            Some(GameResult::Winner((PlayerId::First, "".to_string())))
        }
    }

    fn current_player(&self) -> PlayerId {
        self.playing
    }

    fn associate_players(players: Vec<&Strategy<Self>>) -> HashMap<PlayerId, &Strategy<Self>> {
        let mut players_map = HashMap::new();
        players_map.insert(PlayerId::First, players[0]);
        players_map.insert(PlayerId::Second, players[1]);

        players_map
    }

    fn all_possible_moves() -> [Self::Move; Self::TOTAL_MOVES]{
        let mut moves = [OthelloMove::Pass; Self::TOTAL_MOVES];

        for x in 0..8 {
            for y in 0..8 {
                moves[(8 * x) + y] = OthelloMove::Move(x, y)
            }
        }

        moves
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
