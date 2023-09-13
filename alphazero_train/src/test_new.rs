use alphazero::BoardGameModel;
use dfdx::tensor::Cpu;
use rust_games_main::Leaderboard;
use rust_games_players::AlphaZero;
use rust_games_shared::{Game, Strategy};

#[derive(Debug)]
pub enum NewModelResults {
    OldModelWon(f32),
    NewModelWon(f32),
}

pub fn test_new_model<G: Game + 'static>(
    champ_model_name: &str,
    new_model_name: &str,
    save_winner_to: Option<&str>,
    num_iterations: usize,
) -> NewModelResults
where
    [(); G::CHANNELS * G::BOARD_SIZE * G::BOARD_SIZE]: Sized,
    [(); G::TOTAL_MOVES]: Sized
{
    let dev: Cpu = Default::default();

    let old_az = Strategy::new(
        "Old Alphazero".to_string(),
        AlphaZero::new_from_file::<BoardGameModel<G>>(champ_model_name, 1.0, &dev, false),
    );

    let new_az = Strategy::new(
        "New AlphaZero".to_string(),
        AlphaZero::new_from_file::<BoardGameModel<G>>(new_model_name, 1.0, &dev, false),
    );

    let players = vec![old_az, new_az];
    let mut arena = Leaderboard::new(players);
    arena.play_random_games(num_iterations);

    let new_model_wins = arena.strategies.get("New AlphaZero").unwrap().elo.wins;

    let new_win_percentage = (new_model_wins as f32) / (num_iterations as f32);
    let result = if new_win_percentage >= 0.6 {
        NewModelResults::NewModelWon(new_win_percentage)
    } else {
        NewModelResults::OldModelWon(new_win_percentage)
    };

    if save_winner_to.is_some() {
        let winner_name = match result {
            NewModelResults::OldModelWon(_) => champ_model_name,
            NewModelResults::NewModelWon(_) => new_model_name,
        };
        AlphaZero::new_from_file::<BoardGameModel<G>>(winner_name, 1.0, &dev, false)
            .mcts
            .borrow()
            .save_nn(save_winner_to.unwrap());
    }

    result
}
