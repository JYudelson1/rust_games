use dfdx::prelude::*;
use rust_games_shared::{Game, Player};

use crate::MCTS;
use std::cell::RefCell;

pub struct AlphaZeroPlayer<
    G: Game,
    M: Module<
        Tensor<(Const<{ G::CHANNELS }>, G::BoardSize, G::BoardSize), f32, AutoDevice>,
        Output = (
            Tensor<(Const<{ G::TOTAL_MOVES }>,), f32, AutoDevice>,
            Tensor<(Const<1>,), f32, AutoDevice>,
        ),
        Error = <AutoDevice as HasErr>::Err,
    >,
> {
    pub mcts: RefCell<MCTS<G, M>>,
}

impl<
        G: Game,
        M: Module<
            Tensor<(Const<{ G::CHANNELS }>, G::BoardSize, G::BoardSize), f32, AutoDevice>,
            Output = (
                Tensor<(Const<{ G::TOTAL_MOVES }>,), f32, AutoDevice>,
                Tensor<(Const<1>,), f32, AutoDevice>,
            ),
            Error = <AutoDevice as HasErr>::Err,
        >,
    > AlphaZeroPlayer<G, M>
{
    pub fn new(model: M, temperature: f32, training: bool, traversal_iter: usize) -> Self {
        Self {
            mcts: MCTS::new(G::new(), model, temperature, training, traversal_iter).into(),
        }
    }

    //TODO: There should be a Config struct for this, which implements default
    pub fn new_from_file<B: BuildOnDevice<AutoDevice, f32, Built = M>>(
        model_name: &str,
        data_dir: &str,
        temperature: f32,
        dev: &AutoDevice,
        training: bool,
        traversal_iter: usize,
    ) -> Self
    where
        M: TensorCollection<f32, AutoDevice>,
    {
        let file_name = format!("{}/{}.safetensors", data_dir, model_name);
        Self {
            mcts: MCTS::new_from_file::<B>(
                G::new(),
                temperature,
                &file_name,
                dev,
                training,
                traversal_iter,
            )
            .into(),
        }
    }
}

impl<
        G: Game + 'static,
        M: Module<
                Tensor<(Const<{ G::CHANNELS }>, G::BoardSize, G::BoardSize), f32, AutoDevice>,
                Output = (
                    Tensor<(Const<{ G::TOTAL_MOVES }>,), f32, AutoDevice>,
                    Tensor<(Const<1>,), f32, AutoDevice>,
                ),
                Error = <AutoDevice as HasErr>::Err,
            > + 'static,
    > Player<G> for AlphaZeroPlayer<G, M>
{
    fn choose_move(&self, game: &G) -> Result<<G as Game>::Move, rust_games_shared::PlayerError> {
        let mv = self.mcts.borrow_mut().choose_move(game);

        //println!("the v-value of the board is now: {}", self.mcts.borrow_mut().root.get_mut().v);

        mv
    }

    fn reset(&mut self) {
        self.mcts.get_mut().reset_board();
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::AlphaZeroPlayer;
    use crate::{BoardGameModel, MCTS};
    use dfdx::prelude::*;
    use rust_games_games::Othello;
    use rust_games_shared::{Game, Player};

    #[test]
    fn first_move() {
        let dev: AutoDevice = Default::default();
        let nn = dev.build_module::<BoardGameModel<Othello>, f32>();
        let player = AlphaZeroPlayer::new(nn.clone(), 1.0, false, 100);
        let mut g = Othello::new();
        let m = player.choose_move(&g);
        g.make_move(m.unwrap());
        g.print();

        player
            .mcts
            .into_inner()
            .save_nn("test", "/Applications/Python 3.4/MyScripts/rust_games/data");
    }

    #[test]
    fn load_mcts_test() {
        let g = Othello::new();
        let dev: AutoDevice = Default::default();
        MCTS::new_from_file::<BoardGameModel<Othello>>(g, 1.0, "test", &dev, false, 100);
    }

    #[test]
    fn load_player_test() {
        let mut g = Othello::new();
        let dev: AutoDevice = Default::default();
        let player: AlphaZeroPlayer<Othello, _> =
            AlphaZeroPlayer::new_from_file::<BoardGameModel<Othello>>(
                "test",
                "/Applications/Python 3.4/MyScripts/rust_games/data",
                1.0,
                &dev,
                false,
                100,
            );

        let m = player.choose_move(&g);
        g.make_move(m.unwrap());
        g.print();
    }
}
