use core::fmt;
use std::collections::HashMap;

use dfdx::prelude::*;
use rust_games_shared::{Game, Strategy};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TTTState {
    Empty,
    X,
    O,
}

impl TTTState {
    fn to_player(&self) -> Option<PlayerSymbol> {
        match self {
            TTTState::Empty => None,
            TTTState::X => Some(PlayerSymbol::X),
            TTTState::O => Some(PlayerSymbol::O),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Hash, Eq)]
pub enum PlayerSymbol {
    X,
    O,
}

impl PlayerSymbol {
    fn to_state(&self) -> TTTState {
        match self {
            PlayerSymbol::X => TTTState::X,
            PlayerSymbol::O => TTTState::O,
        }
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
    playing: PlayerSymbol,
}

impl Game for TicTacToe {
    type Move = TTTMove;

    type Board = [[TTTState; 3]; 3];

    type PlayerId = PlayerSymbol;

    type BoardSize = Const<3>;

    type TotalBoardSize = Const<9>;

    const CHANNELS: usize = 3;

    const NUM_PLAYERS: usize = 2;

    const TOTAL_MOVES: usize = 9;

    fn new() -> Self {
        Self {
            board: [[TTTState::Empty; 3]; 3],
            playing: PlayerSymbol::X,
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
        println!("Currently playing: {}", self.playing.to_state());
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
            PlayerSymbol::X => 1.0_f32,
            PlayerSymbol::O => 0.0,
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
        self.board[m.x][m.y] = self.playing.to_state();
        self.playing = match self.playing {
            PlayerSymbol::X => PlayerSymbol::O,
            PlayerSymbol::O => PlayerSymbol::X,
        };
    }

    fn is_over(&self) -> bool {
        self.get_winner().is_some() ||
            self.legal_moves().is_empty()
    }

    fn get_winner(&self) -> Option<Self::PlayerId> {
        // Horizontal
        for d in 0..3 {
            if self.board[d][0] == self.board[d][1] && self.board[d][0] == self.board[d][2] {
                return self.board[d][0].to_player();
            }
        }
        // Vertical
        for d in 0..3 {
            if self.board[0][d] == self.board[1][d] && self.board[0][d] == self.board[2][d] {
                return self.board[0][d].to_player();
            }
        }
        // Diagonal
        if self.board[0][0] == self.board[1][1] && self.board[0][0] == self.board[2][2] {
            return self.board[1][1].to_player();
        }
        if self.board[0][2] == self.board[1][1] && self.board[0][2] == self.board[2][0] {
            return self.board[1][1].to_player();
        }

        None
    }

    fn current_player(&self) -> Self::PlayerId {
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
    ) -> std::collections::HashMap<Self::PlayerId, &Strategy<Self>> {
        let mut players_map = HashMap::new();
        players_map.insert(PlayerSymbol::X, players[0]);
        players_map.insert(PlayerSymbol::O, players[1]);

        players_map
    }
}
