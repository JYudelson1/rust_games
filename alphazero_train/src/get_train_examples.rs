use alphazero::{BoardGameModel, TrainingExample};
use dfdx::prelude::{BuildOnDevice, AutoDevice, Tensor3D};
use rust_games_players::AlphaZero;
use rust_games_shared::{Game, Strategy};

fn training_game<G: Game + 'static>(model_name: &str) -> Vec<TrainingExample<G>>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
    [(); G::CHANNELS * G::BOARD_SIZE * G::BOARD_SIZE]: Sized,
    [(); G::TOTAL_MOVES]: Sized
{
    // For now, only support two player games
    // Fix that later
    assert!(G::NUM_PLAYERS == 2);

    //let x: ;

    let dev: AutoDevice = Default::default();

    let az1: AlphaZero<G, _> =
        AlphaZero::new_from_file::<BoardGameModel<G>>(model_name, 1.0, &dev, true);
    let player1 = Strategy::new("Player1".to_string(), az1);

    let az2: AlphaZero<G, _> =
        AlphaZero::new_from_file::<BoardGameModel<G>>(model_name, 1.0, &dev, true);
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
        .downcast_ref::<AlphaZero<G, <BoardGameModel<G> as BuildOnDevice<AutoDevice, f32>>::Built>>()
        .unwrap()
        .mcts
        .borrow_mut()
        .train_examples;

    let examples2 = &player2
        .player
        .as_any()
        .downcast_ref::<AlphaZero<G, <BoardGameModel<G> as BuildOnDevice<AutoDevice, f32>>::Built>>()
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

pub fn get_examples_until<G: Game + 'static>(
    model_name: &str,
    min_examples: usize,
) -> Vec<TrainingExample<G>>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
    [(); G::CHANNELS * G::BOARD_SIZE * G::BOARD_SIZE]: Sized,
    [(); G::TOTAL_MOVES]: Sized
{
    let mut all_examples = vec![];

    loop {
        all_examples.extend(training_game(model_name));
        if all_examples.len() >= min_examples {
            break;
        }
    }

    all_examples
}

#[cfg(test)]
mod tests {
    use super::{get_examples_until, training_game};
    use rust_games_games::Othello;
    #[test]
    fn works_at_all() {
        training_game::<Othello>("test");
    }

    #[test]
    fn get_200_ex() {
        let ex = get_examples_until::<Othello>("test", 200);
        assert!(ex.len() >= 200);
    }
}
