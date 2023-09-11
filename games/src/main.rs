#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod games;
mod leaderboard;
mod players;

use games::othello::Othello;
use leaderboard::Leaderboard;
use players::Corners;
use players::Greedy;
use players::Random;
use shared::Strategy;

fn main() {
    let corner_player = Strategy::new("Corners".to_string(), Corners::new());
    let rand_player_1 = Strategy::new("Random1".to_string(), Random::new());
    let rand_player_2 = Strategy::new("Random2".to_string(), Random::new());
    let greedy_player = Strategy::new("Greedy".to_string(), Greedy::new());

    let players = vec![corner_player, rand_player_1, rand_player_2, greedy_player];
    //println!("{:?}", Othello::play_full_game(players, true));
    let mut arena = Leaderboard::new(players);
    arena.print();
    arena.play_random_games(4000);
    arena.print();
}
