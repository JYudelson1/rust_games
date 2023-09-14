use alphazero::MCTSConfig;
use dfdx::prelude::*;
use rust_games_main::Leaderboard;
use rust_games_players::AlphaZero;
use rust_games_shared::{Game, Strategy};

#[derive(Debug)]
pub enum NewModelResults {
    OldModelWon(f32),
    NewModelWon(f32),
}

pub fn test_new_model<G: Game + 'static, B: BuildOnDevice<AutoDevice, f32> + 'static>(
    new_model: &<B as BuildOnDevice<AutoDevice, f32>>::Built,
    best_model_name: &str,
    data_dir: &str,
    save_winner_to: Option<&str>,
    num_iterations: usize,
    mcts_cfg: &MCTSConfig
) -> NewModelResults
where
    [(); G::TOTAL_MOVES]: Sized,
    [(); G::CHANNELS]: Sized,
    [(); <G::TotalBoardSize as ConstDim>::SIZE]: Sized,
    [(); 2 * <G::TotalBoardSize as ConstDim>::SIZE]: Sized,
    [(); <G::BoardSize as ConstDim>::SIZE]: Sized,
    <B as BuildOnDevice<AutoDevice, f32>>::Built: Module<
        Tensor<
            (
                Const<{ G::CHANNELS }>,
                <G as Game>::BoardSize,
                <G as Game>::BoardSize,
            ),
            f32,
            AutoDevice,
        >,
        Output = (
            Tensor<(Const<{ G::TOTAL_MOVES }>,), f32, AutoDevice>,
            Tensor<(Const<1>,), f32, AutoDevice>,
        ),
        Error = <AutoDevice as HasErr>::Err,
    > + Clone,
{
    let dev: AutoDevice = Default::default();

    let old_az = Strategy::new(
        "Old Alphazero".to_string(),
        AlphaZero::new_from_file::<B>(
            best_model_name, 
            data_dir, 
            mcts_cfg.temperature, 
            &dev, 
            false, 
            mcts_cfg.traversal_iter),
    );

    let new_az = Strategy::new(
        "New AlphaZero".to_string(),
        AlphaZero::new(new_model.clone(), mcts_cfg.temperature, false, mcts_cfg.traversal_iter),
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
            NewModelResults::OldModelWon(_) => best_model_name,
            NewModelResults::NewModelWon(_) => "New Model",
        };
        AlphaZero::new_from_file::<B>(winner_name, data_dir, 1.0, &dev, false, 100)
            .mcts
            .borrow()
            .save_nn(save_winner_to.unwrap(), data_dir);
    }

    result
}
