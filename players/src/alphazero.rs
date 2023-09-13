use dfdx::prelude::*;
use rust_games_shared::{Game, Player};

use alphazero::MCTS;
use std::cell::RefCell;

pub struct AlphaZero<
    G: Game,
    M: Module<
            Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>,
            Output = (Tensor<(Const<{G::TOTAL_MOVES}>,), f32, AutoDevice>, Tensor<(Const<1>,), f32, AutoDevice>),
            Error = <AutoDevice as HasErr>::Err,
        >,
> where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    pub mcts: RefCell<MCTS<G, M>>,
}

impl<
        G: Game,
        M: Module<
                Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>,
                Output = (Tensor<(Const<{G::TOTAL_MOVES}>,), f32, AutoDevice>, Tensor<(Const<1>,), f32, AutoDevice>),
                Error = <AutoDevice as HasErr>::Err,
            > + TensorCollection<f32, AutoDevice>,
    > AlphaZero<G, M>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    pub fn new(model: M, temperature: f32, training: bool) -> Self {
        Self {
            mcts: MCTS::new(G::new(), model, temperature, training).into(),
        }
    }

    pub fn new_from_file<B: BuildOnDevice<AutoDevice, f32, Built = M>>(
        model_name: &str,
        temperature: f32,
        dev: &AutoDevice,
        training: bool
    ) -> Self 
    where [(); G::CHANNELS * G::BOARD_SIZE * G::BOARD_SIZE]: Sized
    {
        Self {
            mcts: MCTS::new_from_file::<B>(G::new(), temperature, model_name, dev, training).into(),
        }
    }
}

impl<
        G: Game + 'static,
        M: Module<
                Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>,
                Output = (Tensor<(Const<{G::TOTAL_MOVES}>,), f32, AutoDevice>, Tensor<(Const<1>,), f32, AutoDevice>),
                Error = <AutoDevice as HasErr>::Err,
            > + 'static,
    > Player<G> for AlphaZero<G, M>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    fn choose_move(&self, game: &G) -> Result<<G as Game>::Move, rust_games_shared::PlayerError> {
        self.mcts.borrow_mut().choose_move(game)
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
    use super::AlphaZero;
    use alphazero::{BoardGameModel, MCTS};
    use dfdx::prelude::*;
    use rust_games_games::Othello;
    use rust_games_shared::{Game, Player};

    #[test]
    fn first_move() {
        let dev: AutoDevice = Default::default();
        let nn = dev.build_module::<BoardGameModel<Othello>, f32>();
        let player = AlphaZero::new(nn.clone(), 1.0, false);
        let mut g = Othello::new();
        let m = player.choose_move(&g);
        g.make_move(m.unwrap());
        g.print();

        player.mcts.into_inner().save_nn("test");
    }

    #[test]
    fn load_mcts_test() {
        let g = Othello::new();
        let dev: AutoDevice = Default::default();
        MCTS::new_from_file::<BoardGameModel<Othello>>(
            g,
            1.0,
            "test",
            &dev,
            false
        );
    }

    #[test]
    fn load_player_test() {
        let mut g = Othello::new();
        let dev: AutoDevice = Default::default();
        let player: AlphaZero<Othello, _> =
            AlphaZero::new_from_file::<BoardGameModel<Othello>>("test", 1.0, &dev, false);

        let m = player.choose_move(&g);
        g.make_move(m.unwrap());
        g.print();
    }
}
