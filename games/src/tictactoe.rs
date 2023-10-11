use core::fmt;
use std::collections::HashMap;

use dfdx::prelude::*;
use rust_games_shared::{Game, GameResult, PlayerId, Strategy};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TTTState {
    Empty,
    X,
    O,
}

impl TTTState {
    fn to_player(&self) -> Option<PlayerId> {
        match self {
            TTTState::Empty => None,
            TTTState::X => Some(PlayerId::First),
            TTTState::O => Some(PlayerId::Second),
        }
    }
}

fn to_state(id: &PlayerId) -> TTTState {
    match id {
        PlayerId::First => TTTState::X,
        PlayerId::Second => TTTState::O,
    }
}

impl fmt::Display for TTTState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let icon = match self {
            TTTState::Empty => " ",
            TTTState::X => "X",
            TTTState::O => "O",
        };

        write!(f, "{}", icon)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TTTMove {
    x: usize,
    y: usize,
}

impl fmt::Display for TTTMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.x + 1, 3 - self.y)
    }
}

#[derive(Debug, Clone)]
pub struct TicTacToe {
    board: [[TTTState; 3]; 3],
    playing: PlayerId,
}

impl Game for TicTacToe {
    type Move = TTTMove;

    type Board = [[TTTState; 3]; 3];

    type BoardSize = Const<3>;

    type TotalBoardSize = Const<9>;

    const CHANNELS: usize = 3;

    const NUM_PLAYERS: usize = 2;

    const TOTAL_MOVES: usize = 9;

    fn new() -> Self {
        Self {
            board: [[TTTState::Empty; 3]; 3],
            playing: PlayerId::First,
        }
    }

    fn print(&self) {
        for (i, row) in self.board.iter().enumerate() {
            print!("{} ", 3 - i);
            for icon in row {
                print!(" {}", icon);
            }
            println!()
        }
        println!();
        println!("   1 2 3");
        println!("Currently playing: {}", to_state(&self.playing));
    }

    fn to_nn_input(
        &self,
    ) -> Tensor<(Const<{ Self::CHANNELS }>, Self::BoardSize, Self::BoardSize), f32, AutoDevice>
    {
        let dev: AutoDevice = Default::default();
        let mut o_channel = [[0.0; 3]; 3];
        for (x, row) in self.board.iter().enumerate() {
            for (y, tile) in row.iter().enumerate() {
                o_channel[x][y] = match tile {
                    TTTState::Empty => 0.0,
                    TTTState::X => 0.0,
                    TTTState::O => 1.0,
                }
            }
        }

        let mut x_channel = [[0.0; 3]; 3];
        for (x, row) in self.board.iter().enumerate() {
            for (y, tile) in row.iter().enumerate() {
                x_channel[x][y] = match tile {
                    TTTState::Empty => 0.0,
                    TTTState::X => 1.0,
                    TTTState::O => 0.0,
                }
            }
        }

        let player_num = match self.playing {
            PlayerId::First => 1.0_f32,
            PlayerId::Second => 0.0,
        };

        let player_channel = [[player_num; 3]; 3];

        dev.tensor([x_channel, o_channel, player_channel])
    }

    fn get_board(&self) -> Self::Board {
        self.board
    }

    fn legal_moves(&self) -> Vec<Self::Move> {
        let mut moves = vec![];

        for x in 0..3 {
            for y in 0..3 {
                match self.board[x][y] {
                    TTTState::Empty => moves.push(TTTMove { x, y }),
                    TTTState::X => (),
                    TTTState::O => (),
                }
            }
        }
        moves
    }

    fn make_move(&mut self, m: Self::Move) {
        self.board[m.x][m.y] = to_state(&self.playing);
        self.playing = match self.playing {
            PlayerId::First => PlayerId::Second,
            PlayerId::Second => PlayerId::First,
        };
    }

    fn is_over(&self) -> bool {
        self.get_result().is_some() ||
            self.legal_moves().is_empty()
    }

    fn get_result(&self) -> Option<GameResult> {
        // Horizontal
        for d in 0..3 {
            if self.board[d][0] == self.board[d][1] && self.board[d][0] == self.board[d][2] {
                return Some(GameResult::Winner(self.board[d][0].to_player().unwrap()));
            }
        }
        // Vertical
        for d in 0..3 {
            if self.board[0][d] == self.board[1][d] && self.board[0][d] == self.board[2][d] {
                return Some(GameResult::Winner(self.board[0][d].to_player().unwrap()));
            }
        }
        // Diagonal
        if self.board[0][0] == self.board[1][1] && self.board[0][0] == self.board[2][2] {
            return Some(GameResult::Winner(self.board[1][1].to_player().unwrap()));
        }
        if self.board[0][2] == self.board[1][1] && self.board[0][2] == self.board[2][0] {
            return Some(GameResult::Winner(self.board[1][1].to_player().unwrap()));
        }

        None
    }

    fn current_player(&self) -> PlayerId {
        self.playing
    }

    fn all_possible_moves() -> [Self::Move; Self::TOTAL_MOVES] {
        let mut moves = [TTTMove { x: 0, y: 0 }; Self::TOTAL_MOVES];

        for x in 0..3 {
            for y in 0..3 {
                moves[x + 3 * y] = TTTMove { x, y };
            }
        }

        moves
    }

    fn associate_players(
        players: Vec<&Strategy<Self>>,
    ) -> std::collections::HashMap<PlayerId, &Strategy<Self>> {
        let mut players_map = HashMap::new();
        players_map.insert(PlayerId::First, players[0]);
        players_map.insert(PlayerId::Second, players[1]);

        players_map
    }
}
