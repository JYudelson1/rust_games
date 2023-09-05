//mod alphazero;
mod games;
mod leaderboard;
mod players;

use games::othello::Othello;
use leaderboard::Leaderboard;
use players::corners::Corners;
use players::greedy::Greedy;
use players::random::Random;
use rust_games::Strategy;

fn main() {
    let corner_player = Strategy::new::<Corners>("Corners".to_string());
    let rand_player_1 = Strategy::new::<Random<Othello>>("Random1".to_string());
    let rand_player_2 = Strategy::new::<Random<Othello>>("Random2".to_string());
    let greedy_player = Strategy::new::<Greedy>("Greedy".to_string());

    let players = vec![corner_player, rand_player_1, rand_player_2, greedy_player];
    //println!("{:?}", Othello::play_full_game(players, true));
    let mut arena = Leaderboard::new(players);
    arena.print();
    arena.play_random_games(4000);
    arena.print();
}
