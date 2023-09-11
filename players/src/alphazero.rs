use dfdx::prelude::{BuildOnDevice, Cpu, Tensor, Tensor3D};
use dfdx::prelude::{Module, TensorCollection, ZeroSizedModule};
use dfdx::shapes::{Const, Rank0};
use rust_games_shared::{Game, Player};

use alphazero::MCTS;
use std::cell::RefCell;
use std::rc::Rc;

pub struct AlphaZero<
    G: Game,
    M: Module<
            Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>,
            Output = (Tensor<(Const<1>,), f32, Cpu>, Tensor<(Const<1>,), f32, Cpu>),
            Error = dfdx::prelude::CpuError,
        > + TensorCollection<f32, Cpu>,
> where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    mcts: RefCell<MCTS<G, M>>,
}

impl<
        G: Game,
        M: Module<
                Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>,
                Output = (Tensor<(Const<1>,), f32, Cpu>, Tensor<(Const<1>,), f32, Cpu>),
                Error = dfdx::prelude::CpuError,
            > + TensorCollection<f32, Cpu>,
    > AlphaZero<G, M>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    pub fn new(model: M, temperature: f32) -> Self {
        Self {
            mcts: MCTS::new(G::new(), model, temperature).into(),
        }
    }

    pub fn new_from_file<B: BuildOnDevice<Cpu, f32, Built = M>>(
        model_name: &str,
        temperature: f32,
        dev: &Cpu,
    ) -> Self {
        Self {
            mcts: MCTS::new_from_file::<B>(G::new(), temperature, model_name, dev).into(),
        }
    }
}

impl<
        G: Game,
        M: Module<
                Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>,
                Output = (Tensor<(Const<1>,), f32, Cpu>, Tensor<(Const<1>,), f32, Cpu>),
                Error = dfdx::prelude::CpuError,
            > + TensorCollection<f32, Cpu>,
    > Player<G> for AlphaZero<G, M>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    fn choose_move(&self, game: &G) -> Result<<G as Game>::Move, rust_games_shared::PlayerError> {
        self.mcts.borrow_mut().choose_move(game)
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
        let dev: Cpu = Default::default();
        let nn = dev.build_module::<BoardGameModel<Othello>, f32>();
        let player = AlphaZero::new(nn.clone(), 1.0);
        let mut g = Othello::new();
        let m = player.choose_move(&g);
        g.make_move(m.unwrap());
        g.print();

        nn.save_safetensors("test.safetensors").unwrap();
    }

    #[test]
    fn load_mcts_test() {
        let g = Othello::new();
        let dev: Cpu = Default::default();
        MCTS::new_from_file::<BoardGameModel<Othello>>(
            g,
            1.0,
            "/Applications/Python 3.4/MyScripts/rust_games/games/test.safetensors",
            &dev,
        );
    }
}
