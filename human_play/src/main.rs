use rust_games_games::Othello;
use rust_games_players::{Corners, Human};
use rust_games_shared::{Game, Strategy};

fn main() {
    type G = Othello;

    let human_player: Strategy<G> = Strategy::new("Human".to_string(), Human::new());

    let bot = Strategy::new("Bot".to_string(), Corners);

    let players = vec![&human_player, &bot];

    let result = G::play_full_game(players, true);
    println!("{:?}", result);
}
