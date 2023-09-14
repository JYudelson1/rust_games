use crate::games_list::GamesHolder;
use alphazero::update_on_batch;
use dfdx::{optim::Adam, prelude::*};
use lazy_pbar::pbar;
use rust_games_shared::Game;

pub(crate) fn update_from_gamesholder<G: Game + 'static,
    Model: ModuleMut<
            <Tensor<
                (
                    usize,
                    Const<{G::CHANNELS}>,
                    G::BoardSize,
                    G::BoardSize,
                ),
                f32,
                AutoDevice,
            > as dfdx::tensor::Trace<f32, AutoDevice>>::Traced,
            Error = <AutoDevice as HasErr>::Err,
            Output = (
                Tensor<(usize, Const<{G::TOTAL_MOVES}>,), f32, AutoDevice, OwnedTape<f32, AutoDevice>>,
                Tensor<(usize, Const<1>,), f32, AutoDevice, OwnedTape<f32, AutoDevice>>,
            ),
> + TensorCollection<f32, AutoDevice>>
(
    model: &mut Model,
    opt: &mut Adam<Model, f32, AutoDevice>,
    dev: &AutoDevice,
    games_holder: &GamesHolder<G>,
    batch_size: usize,
    num_batches: usize)
{
    for _ in pbar(0..num_batches) {
        let examples = games_holder.get_random(batch_size);
        update_on_batch(model, examples, opt, dev).unwrap();
    }
}
