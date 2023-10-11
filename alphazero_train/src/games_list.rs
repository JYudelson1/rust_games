use crate::get_train_examples::training_games;
use alphazero::{MCTSConfig, TrainingExample};
use dfdx::prelude::*;
use rand::seq::SliceRandom;
use rust_games_shared::Game;

pub(crate) struct GamesHolder<G: Game>
where
    [(); G::CHANNELS]: Sized,
{
    pub games: Vec<TrainingExample<G>>,
    pub capacity: usize,
}

impl<'a, G: Game + 'static> GamesHolder<G>
where
    [(); G::CHANNELS]: Sized,
{
    pub(crate) fn extend(&mut self, new_games: Vec<TrainingExample<G>>) {
        let new_len = self.games.len() + new_games.len();

        if new_len > self.capacity {
            let to_delete = new_len - self.capacity;
            self.games.drain(0..to_delete);
        }

        self.games.extend(new_games);
    }

    pub(crate) fn get_random(&self, batch_size: usize) -> Vec<&TrainingExample<G>> {
        let mut rng = rand::thread_rng();
        self.games.choose_multiple(&mut rng, batch_size).collect()
    }

    pub(crate) fn add_n_games<B: BuildOnDevice<AutoDevice, f32> + 'static>(
        &mut self,
        model_name: &str,
        data_dir: &str,
        num_games: usize,
        mcts_cfg: &MCTSConfig,
    ) where
        [(); G::TOTAL_MOVES]: Sized,
        [(); <G::BoardSize as ConstDim>::SIZE]: Sized,
        [(); <G::TotalBoardSize as ConstDim>::SIZE]: Sized,
        [(); 2 * <G::TotalBoardSize as ConstDim>::SIZE]: Sized,
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
        let examples = training_games::<G, B>(model_name, data_dir, num_games, mcts_cfg);
        self.extend(examples);
        // TODO: Should totally be possible to rotate these boards iff we're playing othello
    }
}

#[test]
fn test_cap() {
    use dfdx::tensor::TensorFrom;
    use rust_games_games::Othello;
    let dev: AutoDevice = Default::default();
    let ex = TrainingExample::<Othello>::new(dev.tensor([[[0.0; 8]; 8]; 3]), 0.0, vec![]);

    let mut gh = GamesHolder::<Othello> {
        games: vec![ex.clone(), ex.clone()],
        capacity: 3,
    };

    gh.extend(vec![ex.clone(), ex.clone()]);

    assert!(gh.games.len() == gh.capacity);
}
