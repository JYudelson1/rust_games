use std::collections::HashMap;

use dfdx::{optim::Adam, prelude::*};
use rust_games_shared::Game;

#[derive(Clone)]
pub struct UnfinishedTrainingExample<G: Game>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    position: Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>,
    next_move_probs: Vec<(
        G::Move,
        usize,
    )>,
}

impl<G: Game> UnfinishedTrainingExample<G>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    pub fn new(
        position: Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>,
        next_move_probs: Vec<(
            G::Move,
            usize,
        )>,
    ) -> Self {
        UnfinishedTrainingExample {
            position,
            next_move_probs,
        }
    }

    pub fn finish(self, winner: f32) -> TrainingExample<G> {
        TrainingExample {
            position: self.position,
            winner,
            next_move_probs: self.next_move_probs,
        }
    }
}

pub struct TrainingExample<G: Game>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    position: Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>,
    winner: f32,
    next_move_probs: Vec<(
        G::Move,
        usize,
    )>,
}

impl<G: Game> TrainingExample<G>
where
    Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>: Sized,
{
    fn to_true_probs(&self, dev: &Cpu) -> Tensor<(Const<{G::TOTAL_MOVES}>,), f32, Cpu, NoneTape> {

        let mut board_to_count: HashMap<G::Move, usize> = HashMap::new();

        for (action, count) in &self.next_move_probs {
            board_to_count.insert(action.clone(), *count);
        }

        let all_moves = G::all_possible_moves();

        let mut all_counts = [0.0_f32; G::TOTAL_MOVES];

        for (i, mv) in all_moves.iter().enumerate() {
            if let Some(count) = board_to_count.get(mv) {
                all_counts[i] = *count as f32;
            }
        }

        let move_counts_tensor = dev.tensor(all_counts);
        move_counts_tensor.softmax()
    }

    pub fn new(
        position: Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>, 
        winner: f32, 
        next_move_probs: Vec<(
        G::Move,
        usize,
    )>) -> Self {
        TrainingExample { position, winner, next_move_probs }
    }
}

fn loss<G: Game>(
    model_v: Tensor<(Const<1>,), f32, Cpu, OwnedTape<f32, Cpu>>,
    model_probs: Tensor<(Const<{G::TOTAL_MOVES}>,), f32, Cpu, OwnedTape<f32, Cpu>>,
    winner: Tensor<(Const<1>,), f32, Cpu, NoneTape>,
    true_probs: Tensor<(Const<{G::TOTAL_MOVES}>,), f32, Cpu, NoneTape>,
) -> Tensor<Rank0, f32, Cpu, OwnedTape<f32, Cpu>> {
    let mse_loss = mse_loss(model_v, winner);
    let bce_loss = binary_cross_entropy_with_logits_loss(model_probs, true_probs);

    mse_loss + bce_loss
}

pub fn update_on_many<
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
                Tensor<(Const<{G::TOTAL_MOVES}>,), f32, Cpu, OwnedTape<f32, Cpu>>,
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
                Tensor<(usize,Const<{G::TOTAL_MOVES}>), f32, Cpu, OwnedTape<f32, Cpu>>,
                Tensor<(usize,Const<1>), f32, Cpu, OwnedTape<f32, Cpu>>,
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
        let (p, v) = model.try_forward_mut(example.position.clone().traced(grads.clone()))?;
        let loss: Tensor<(), f32, Cpu, OwnedTape<f32, Cpu>> = loss(
            v,
            p,
            dev.tensor([example.winner]),
            example.to_true_probs(&dev),
        );
        let loss_value = loss.array();
        if i % 100 == 0 {
            println!("batch {i} | loss = {loss_value:?}");
        }
        grads = loss.try_backward()?;
        if i % batch_accum == 0 {
            opt.update(model, &grads).unwrap();
            model.try_zero_grads(&mut grads)?;
        }
    }
    Ok(())
}
