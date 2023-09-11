use dfdx::prelude::Module;
use dfdx::shapes::{Const, Rank0};
use dfdx::tensor::{Tensor, Tensor3D};
use shared::{Game, Player};

use alphazero::MCTS;
use std::cell::RefCell;
use std::rc::Rc;

pub struct AlphaZero<G: Game>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    mcts: RefCell<MCTS<G>>,
}

impl<G: Game> AlphaZero<G>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    pub fn new(
        model: impl Module<
                Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>,
                Output = (
                    Tensor<(Const<1>,), f32, dfdx::tensor::Cpu>,
                    Tensor<(Const<1>,), f32, dfdx::tensor::Cpu>,
                ),
                Error = dfdx::prelude::CpuError,
            > + 'static,
        temperature: f32,
    ) -> Self {
        Self {
            mcts: MCTS::new(G::new(), Rc::new(model), temperature).into(),
        }
    }
}

impl<G: Game> Player<G> for AlphaZero<G>
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
