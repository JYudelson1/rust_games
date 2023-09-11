use dfdx::{optim::Adam, prelude::*};
use rust_games_shared::Game;

struct TrainingExample<G: Game>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    position: Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>,
    winner: f32,
    next_move_probs: Vec<(
        Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>,
        usize,
    )>,
}

impl<G: Game> TrainingExample<G>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    fn to_true_probs(&self, dev: &Cpu) -> Tensor<(usize,), f32, Cpu, NoneTape> {
        let move_counts = self
            .next_move_probs
            .iter()
            .map(|(_board, count)| *count as f32)
            .collect::<Vec<f32>>();

        let move_counts_tensor = dev.tensor_from_vec(move_counts.clone(), (move_counts.len(),));
        move_counts_tensor.softmax()
    }
}

fn to_model_probs<
    G: Game,
    M: ModuleMut<
            <Tensor<
                (
                    usize,
                    Const<{ G::CHANNELS }>,
                    Const<{ G::BOARD_SIZE }>,
                    Const<{ G::BOARD_SIZE }>,
                ),
                f32,
                Cpu,
            > as dfdx::tensor::Trace<f32, Cpu>>::Traced,
            Error = <Cpu as HasErr>::Err,
            Output = (
                Tensor<(usize,), f32, Cpu, OwnedTape<f32, Cpu>>,
                Tensor<(usize,), f32, Cpu, OwnedTape<f32, Cpu>>,
            ),
        > + TensorCollection<f32, Cpu>,
>(
    ex: &TrainingExample<G>,
    model: &mut M,
    grads: Gradients<f32, Cpu>,
) -> Tensor<(usize,), f32, Cpu, OwnedTape<f32, Cpu>>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    let mut all_boards = vec![];

    for (board, _count) in ex.next_move_probs.iter() {
        all_boards.push(board.clone());
    }

    let all_boards_tensor = all_boards.stack();

    let (move_probs, _) = model.forward_mut(all_boards_tensor.traced(grads));

    move_probs.softmax()
}

fn loss(
    model_v: Tensor<(Const<1>,), f32, Cpu, OwnedTape<f32, Cpu>>,
    model_probs: Tensor<(usize,), f32, Cpu, OwnedTape<f32, Cpu>>,
    winner: Tensor<(Const<1>,), f32, Cpu, NoneTape>,
    true_probs: Tensor<(usize,), f32, Cpu, NoneTape>,
) -> Tensor<Rank0, f32, Cpu, OwnedTape<f32, Cpu>> {
    let mse_loss = mse_loss(model_v, winner);
    let bce_loss = binary_cross_entropy_with_logits_loss(model_probs, true_probs);

    mse_loss + bce_loss
}

fn update_on_many<
    G: Game,
    Model: ModuleMut<
            <Tensor<
                (
                    Const<{ G::CHANNELS }>,
                    Const<{ G::BOARD_SIZE }>,
                    Const<{ G::BOARD_SIZE }>,
                ),
                f32,
                Cpu,
            > as dfdx::tensor::Trace<f32, Cpu>>::Traced,
            Error = <Cpu as HasErr>::Err,
            Output = (
                Tensor<(Const<1>,), f32, Cpu, OwnedTape<f32, Cpu>>,
                Tensor<(Const<1>,), f32, Cpu, OwnedTape<f32, Cpu>>,
            ),
        > + ModuleMut<
            <Tensor<
                (
                    usize,
                    Const<{ G::CHANNELS }>,
                    Const<{ G::BOARD_SIZE }>,
                    Const<{ G::BOARD_SIZE }>,
                ),
                f32,
                Cpu,
            > as dfdx::tensor::Trace<f32, Cpu>>::Traced,
            Error = <Cpu as HasErr>::Err,
            Output = (
                Tensor<(usize,), f32, Cpu, OwnedTape<f32, Cpu>>,
                Tensor<(usize,), f32, Cpu, OwnedTape<f32, Cpu>>,
            ),
        > + TensorCollection<f32, Cpu>,
>(
    model: &mut Model,
    mut examples: Vec<TrainingExample<G>>,
    opt: &mut Adam<Model, f32, Cpu>,
    batch_accum: usize,
    dev: Cpu,
) -> Result<(), <Cpu as dfdx::tensor::HasErr>::Err>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
    Tensor1D<{ G::CHANNELS * G::BOARD_SIZE * G::BOARD_SIZE }>: Sized,
{
    let mut grads = model.try_alloc_grads()?;
    for (i, example) in examples.iter_mut().enumerate() {
        let (_, v) = model.try_forward_mut(example.position.clone().traced(grads.clone()))?;
        let loss: Tensor<(), f32, Cpu, OwnedTape<f32, Cpu>> = loss(
            v,
            to_model_probs(example, model, grads),
            dev.tensor([example.winner]),
            example.to_true_probs(&dev),
        );
        let loss_value = loss.array();
        grads = loss.try_backward()?;
        if i % batch_accum == 0 {
            opt.update(model, &grads).unwrap();
            model.try_zero_grads(&mut grads)?;
        }
        println!("batch {i} | loss = {loss_value:?}");
    }
    Ok(())
}
