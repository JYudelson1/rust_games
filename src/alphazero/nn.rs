use dfdx::prelude::*;
use rust_games::Game;

use super::shared::ModelError;

//TODO: Make more generic??
type OthelloBaseModel = (Flatten2D, Linear<256, 10>);
type OthelloPolicyHead = (Linear<10, 1>); //TODO
type OthelloValueHead = (Linear<10, 1>); //TODO

pub struct OthelloModel(
    (
        OthelloBaseModel,
        SplitInto<(
            OthelloPolicyHead, //TODO
            OthelloValueHead,  //TODO
        )>,
    ),
);

// pub struct AlphaZero{
//     model: OthelloModel,
//     grads: Gradients<f32, Cpu>
// }

// impl AlphaZero {
//     fn new(dev: Cpu) -> Self {
//         let mut model = dev.build_module::<OthelloModel, f32>();
//         model.reset_params();
//         let mut grads = model.alloc_grads();
//         AlphaZero { model, grads }
//     }
// }
