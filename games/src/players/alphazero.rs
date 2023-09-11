use dfdx::prelude::{Module, TensorCollection, ZeroSizedModule};
use dfdx::shapes::{Const, Rank0};
use dfdx::tensor::{Cpu, Tensor, Tensor3D};
use shared::{Game, Player};

use alphazero::MCTS;
use std::cell::RefCell;
use std::rc::Rc;

pub struct AlphaZero<G: Game, M: Module<
                Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>,
                Output = (
                    Tensor<(Const<1>,), f32, dfdx::tensor::Cpu>,
                    Tensor<(Const<1>,), f32, dfdx::tensor::Cpu>,
                ),
                Error = dfdx::prelude::CpuError,
            >>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    mcts: RefCell<MCTS<G, M>>,
}

impl<G: Game, M: Module<
                Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>,
                Output = (
                    Tensor<(Const<1>,), f32, dfdx::tensor::Cpu>,
                    Tensor<(Const<1>,), f32, dfdx::tensor::Cpu>,
                ),
                Error = dfdx::prelude::CpuError,
            >> AlphaZero<G, M>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    pub fn new(
        model: M,
        temperature: f32,
    ) -> Self {
        Self {
            mcts: MCTS::new(G::new(), model, temperature).into(),
        }
    }
}

impl<G: Game, M: Module<
                Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>,
                Output = (
                    Tensor<(Const<1>,), f32, dfdx::tensor::Cpu>,
                    Tensor<(Const<1>,), f32, dfdx::tensor::Cpu>,
                ),
                Error = dfdx::prelude::CpuError,
            > + TensorCollection<f32, Cpu>> Player<G> for AlphaZero<G, M>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    fn choose_move(&self, game: &G) -> Result<<G as Game>::Move, shared::PlayerError> {
        self.mcts.borrow_mut().choose_move(game)
    }
}

#[cfg(test)]
mod tests {
    use super::AlphaZero;
    use crate::Othello;
    use alphazero::BoardGameModel;
    use dfdx::prelude::*;
    use shared::{Game, Player};

    #[test]
    fn first_move() {
        let dev: Cpu = Default::default();
        let nn = dev.build_module::<BoardGameModel<Othello>, f32>();
        let player = AlphaZero::new(nn, 1.0);
        let mut g = Othello::new();
        let m = player.choose_move(&g);
        g.make_move(m.unwrap());
        g.print()
    }
}
