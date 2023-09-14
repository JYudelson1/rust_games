use alphazero::TrainingExample;
use dfdx::{
    prelude::{AutoDevice, BuildOnDevice, Const, ConstDim, Module, Tensor},
    tensor::HasErr,
};
use indicatif::ProgressStyle;
use rust_games_players::AlphaZero;
use rust_games_shared::{Game, Strategy};

fn training_game<G: Game + 'static, B: BuildOnDevice<AutoDevice, f32> + 'static>(
    model_name: &str,
    data_dir: &str
) -> Vec<TrainingExample<G>>
where
    [(); G::TOTAL_MOVES]: Sized,
    [(); G::CHANNELS]: Sized,
    [(); <G::BoardSize as ConstDim>::SIZE]: Sized,
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
    >,
{
    // For now, only support two player games
    // Fix that later
    assert!(G::NUM_PLAYERS == 2);

    //let x: ;

    let dev: AutoDevice = Default::default();

    let az1: AlphaZero<G, _> =
        AlphaZero::new_from_file::<B>(model_name, data_dir,  1.0, &dev, true, 100);
    let player1 = Strategy::new("Player1".to_string(), az1);

    let az2: AlphaZero<G, _> =
        AlphaZero::new_from_file::<B>(model_name, data_dir, 1.0, &dev, true, 100);
    let player2 = Strategy::new("Player2".to_string(), az2);

    let players = vec![&player1, &player2];

    let result = G::play_full_game(players, false);

    let winner_num: f32;
    match result {
        rust_games_shared::GameResult::Winner(name) => {
            winner_num = if name == "Player1" {
                1.0
            } else if name == "Player2" {
                -1.0
            } else {
                panic!("Weird player name issue");
            };
        }
        rust_games_shared::GameResult::Tie => winner_num = 0.0,
        rust_games_shared::GameResult::Ranking(_) => todo!(),
    }

    let examples1 = &player1
        .player
        .as_any()
        .downcast_ref::<AlphaZero<G, <B as BuildOnDevice<AutoDevice, f32>>::Built>>()
        .unwrap()
        .mcts
        .borrow_mut()
        .train_examples;

    let examples2 = &player2
        .player
        .as_any()
        .downcast_ref::<AlphaZero<G, <B as BuildOnDevice<AutoDevice, f32>>::Built>>()
        .unwrap()
        .mcts
        .borrow_mut()
        .train_examples;

    let mut all_examples = vec![];

    if let Some(examples) = examples1 {
        for ex in examples {
            all_examples.push(ex)
        }
    }
    if let Some(examples) = examples2 {
        for ex in examples {
            all_examples.push(ex)
        }
    }

    let finished_examples: Vec<TrainingExample<G>> = all_examples
        .into_iter()
        .map(|ex| ex.clone().finish(winner_num))
        .collect();

    finished_examples
}

pub fn get_examples_until<G: Game + 'static, B: BuildOnDevice<AutoDevice, f32> + 'static>(
    model_name: &str,
    data_dir: &str,
    min_examples: usize,
) -> Vec<TrainingExample<G>>
where
    [(); G::TOTAL_MOVES]: Sized,
    [(); G::CHANNELS]: Sized,
    [(); 2 * <G::TotalBoardSize as ConstDim>::SIZE]: Sized,
    [(); <G::TotalBoardSize as ConstDim>::SIZE]: Sized,
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
    >,
{
    let mut all_examples = vec![];

    let progress_bar = indicatif::ProgressBar::new(min_examples as u64).with_style(
        ProgressStyle::default_bar()
            .template("{msg} |{wide_bar}| {pos}/{len} [{elapsed_precise}>{eta_precise}]")
            .unwrap(),
    );
    progress_bar.inc(0);

    loop {
        let new_examples = training_game::<G, B>(model_name, data_dir);
        progress_bar.inc(new_examples.len() as u64);
        all_examples.extend(new_examples);
        if all_examples.len() >= min_examples {
            break;
        }
    }

    progress_bar.finish();
    all_examples
}

#[cfg(test)]
mod tests {
    use super::{get_examples_until, training_game};
    use alphazero::BoardGameModel;
    use rust_games_games::Othello;
    #[test]
    fn works_at_all() {
        training_game::<Othello, BoardGameModel<Othello>>("test", "/Applications/Python 3.4/MyScripts/rust_games/data");
    }

    #[test]
    fn get_200_ex() {
        let ex = get_examples_until::<Othello, BoardGameModel<Othello>>("test", "/Applications/Python 3.4/MyScripts/rust_games/data", 200);
        assert!(ex.len() >= 200);
    }
}
