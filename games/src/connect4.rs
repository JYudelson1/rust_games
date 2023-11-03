use core::fmt;
use std::collections::HashMap;

use dfdx::prelude::*;
use rust_games_shared::{Game, GameResult, PlayerId, Strategy};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Connect4State {
    Empty,
    First,
    Second,
}

impl Connect4State {
    fn to_player(&self) -> Option<PlayerId> {
        match self {
            Connect4State::Empty => None,
            Connect4State::First => Some(PlayerId::First),
            Connect4State::Second => Some(PlayerId::Second),
        }
    }
}

fn to_state(id: &PlayerId) -> Connect4State {
    match id {
        PlayerId::First => Connect4State::First,
        PlayerId::Second => Connect4State::Second,
    }
}

impl fmt::Display for Connect4State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let icon = match self {
            Connect4State::Empty => "⚪️",
            Connect4State::First => "🔴",
            Connect4State::Second => "🟡",
        };

        write!(f, "{}", icon)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Connect4Move {
    column: usize,
}

impl fmt::Display for Connect4Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Column {}", self.column + 1)
    }
}

#[derive(Debug, Clone)]
pub struct Connect4 {
    board: [[Connect4State; 7]; 8],
    playing: PlayerId,
    heights: [u8; 7],
}

impl Connect4 {
    fn check_horizontal(&self) -> Option<PlayerId> {
        for row in self.board {
            let mut current = row[0];
            let mut i = 0;

            for spot in row {
                if spot == Connect4State::Empty {
                    i = 0;
                    current = spot;
                } else if current == Connect4State::Empty {
                    current = spot;
                    i = 1;
                } else if current == spot {
                    i += 1;
                } else if current != spot {
                    current = spot;
                    i = 1;
                }

                if i == 4 {
                    return Some(current.to_player().unwrap());
                }
            }
        }

        None
    }
    fn check_vertical(&self) -> Option<PlayerId> {
        for col in 0..7 {
            let mut current = self.board[0][col];
            let mut i = 0;

            for spot_idx in 0_usize..8 {
                let spot = self.board[spot_idx][col];
                if spot == Connect4State::Empty {
                    i = 0;
                    current = spot;
                } else if current == Connect4State::Empty {
                    current = spot;
                    i = 1;
                } else if current == spot {
                    i += 1;
                } else if current != spot {
                    current = spot;
                    i = 1;
                }

                if i == 4 {
                    return Some(current.to_player().unwrap());
                }
            }
        }

        None
    }
    fn check_diag_upright(&self) -> Option<PlayerId> {
        for x in 0..5 {
            for y in 0..4 {
                if self.board[x][y] == self.board[x + 1][y + 1]
                    && self.board[x][y] == self.board[x + 2][y + 2]
                    && self.board[x][y] == self.board[x + 3][y + 3]
                    && self.board[x][y] != Connect4State::Empty
                {
                    return Some(self.board[x][y].to_player().unwrap());
                }
            }
        }
        None
    }
    fn check_diag_upleft(&self) -> Option<PlayerId> {
        for x in 0..5 {
            for y in 3..7 {
                if self.board[x][y] == self.board[x + 1][y - 1]
                    && self.board[x][y] == self.board[x + 2][y - 2]
                    && self.board[x][y] == self.board[x + 3][y - 3]
                    && self.board[x][y] != Connect4State::Empty
                {
                    return Some(self.board[x][y].to_player().unwrap());
                }
            }
        }
        None
    }
}

impl Game for Connect4 {
    type Move = Connect4Move;

    type Board = [[Connect4State; 7]; 8];

    type BoardSizeX = Const<7>;
    type BoardSizeY = Const<8>;

    type TotalBoardSize = Const<56>;

    const CHANNELS: usize = 3;

    const NUM_PLAYERS: usize = 2;

    const TOTAL_MOVES: usize = 7;

    fn new() -> Self {
        Self {
            board: [[Connect4State::Empty; 7]; 8],
            playing: PlayerId::First,
            heights: [0; 7],
        }
    }

    fn print(&self) {
        for (i, row) in self.board.iter().enumerate() {
            print!("{}", 8 - i);
            for (j, _) in row.iter().enumerate() {
                print!(" {}", self.board[7 - i][j]);
            }
            println!()
        }
        println!("   1  2  3  4  5  6  7");
        println!("Currently playing: {}", to_state(&self.playing));
    }

    fn to_nn_input(
        &self,
    ) -> Tensor<
        (
            Const<{ Self::CHANNELS }>,
            Self::BoardSizeX,
            Self::BoardSizeY,
        ),
        f32,
        AutoDevice,
    > {
        let dev: AutoDevice = Default::default();
        let mut o_channel = [[0.0; 8]; 7];
        for (x, row) in self.board.iter().enumerate() {
            for (y, tile) in row.iter().enumerate() {
                o_channel[x][y] = match tile {
                    Connect4State::Empty => 0.0,
                    Connect4State::First => 0.0,
                    Connect4State::Second => 1.0,
                }
            }
        }

        let mut x_channel = [[0.0; 8]; 7];
        for (x, row) in self.board.iter().enumerate() {
            for (y, tile) in row.iter().enumerate() {
                x_channel[x][y] = match tile {
                    Connect4State::Empty => 0.0,
                    Connect4State::First => 1.0,
                    Connect4State::Second => 0.0,
                }
            }
        }

        let player_num = match self.playing {
            PlayerId::First => 1.0_f32,
            PlayerId::Second => 0.0,
        };

        let player_channel = [[player_num; 8]; 7];

        dev.tensor([x_channel, o_channel, player_channel])
    }

    fn get_board(&self) -> Self::Board {
        self.board
    }

    fn legal_moves(&self) -> Vec<Self::Move> {
        let mut moves = vec![];

        for col in 0..7 {
            if self.board[7][col] == Connect4State::Empty {
                moves.push(Connect4Move { column: col })
            }
        }
        moves
    }

    fn make_move(&mut self, m: Self::Move) {
        let col = m.column;
        let height = self.heights[col];

        self.board[height as usize][col] = to_state(&self.playing);
        self.heights[col] += 1;

        self.playing = match self.playing {
            PlayerId::First => PlayerId::Second,
            PlayerId::Second => PlayerId::First,
        };
    }

    fn is_over(&self) -> bool {
        self.get_result().is_some()
    }

    fn get_result(&self) -> Option<GameResult> {
        // Check for fours in a row
        let mut possible_winner: Option<PlayerId>;
        possible_winner = self.check_horizontal();
        if possible_winner.is_some() {
            return Some(GameResult::Winner(possible_winner.unwrap()));
        }
        possible_winner = self.check_vertical();
        if possible_winner.is_some() {
            return Some(GameResult::Winner(possible_winner.unwrap()));
        }
        possible_winner = self.check_diag_upleft();
        if possible_winner.is_some() {
            return Some(GameResult::Winner(possible_winner.unwrap()));
        }
        possible_winner = self.check_diag_upright();
        if possible_winner.is_some() {
            return Some(GameResult::Winner(possible_winner.unwrap()));
        }

        // Failing that, if no moves are legal, the game is a tie
        if self.legal_moves().is_empty() {
            return Some(GameResult::Tie);
        }

        // If that's also not true, no one has won
        None
    }

    fn current_player(&self) -> PlayerId {
        self.playing
    }

    fn all_possible_moves() -> [Self::Move; Self::TOTAL_MOVES] {
        let mut moves = [Connect4Move { column: 0 }; Self::TOTAL_MOVES];

        for (i, mv) in moves.iter_mut().enumerate() {
            mv.column = i;
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
