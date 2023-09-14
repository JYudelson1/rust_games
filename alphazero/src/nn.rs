use dfdx::prelude::*;
use rust_games_shared::Game;

type ConvLayer<G: Game> = (
    Conv2D<{ G::CHANNELS }, 256, 3, 1, 1>, //TODO: Fix here
    BatchNorm2D<256>,
    ReLU,
);
type InnerResidual = (
    Conv2D<256, 256, 3, 1, 1>,
    BatchNorm2D<256>,
    ReLU,
    Conv2D<256, 256, 3, 1, 1>,
    BatchNorm2D<256>,
);
type ResidualLayer = (Residual<InnerResidual>, ReLU);
type ResidualStack = Repeated<ResidualLayer, 2>;
type BaseModel<G: Game> = (ConvLayer<G>, ResidualStack);

type PolicyHead<G: Game> = (
    Conv2D<256, 2, 1>,
    BatchNorm2D<2>,
    ReLU,
    Flatten2D,
    Linear<{ 2 * <G::TotalBoardSize as ConstDim>::SIZE }, { G::TOTAL_MOVES }>, //TODO: FIx here
);

type ValueHead<G: Game> = (
    (
        Conv2D<256, 1, 1>,
        BatchNorm2D<1>,
        ReLU,
        Flatten2D,
        Linear<{ <G::TotalBoardSize as ConstDim>::SIZE }, 256>, //TODO: Fix here
    ),
    (ReLU, Linear<256, 1>, Tanh),
);
pub type BoardGameModel<G: Game> = ((BaseModel<G>, SplitInto<(PolicyHead<G>, ValueHead<G>)>),);

// type Inp<G: rust_games_shared::Game> =
//     Tensor<(G::Channels, G::BoardSize, G::BoardSize), f32, AutoDevice>;

// type Test<G: rust_games_shared::Game> = BoardGameModel<G>;

// type Built<G> = <Test<G> as BuildOnDevice<AutoDevice, f32>>::Built;

// type Out<G: rust_games_shared::Game> = <Built<G> as Module<Inp<G>>>::Output;

//print!("{}", std::any::type_name::<Out<Othello>>());

// type BoardGameBaseModel<G: Game> = (
//     Flatten2D,
//     Linear<{ G::CHANNELS * G::BOARD_SIZE * G::BOARD_SIZE }, 100>,
//     Linear<100, 10>
// );
// type BoardGamePolicyHead<G: Game> = Linear<10, { G::TOTAL_MOVES }>; //TODO
// type BoardGameValueHead = Linear<10, 1>; //TODO
// pub type BoardGameModel<G: Game> = ((
//     BoardGameBaseModel<G>,
//     SplitInto<(BoardGamePolicyHead<G>, BoardGameValueHead)>,
// ),);

pub fn load_from_file<G: Game, B: BuildOnDevice<AutoDevice, f32>>(
    file_name: &str,
    dev: &AutoDevice,
) -> B::Built
where
    B::Built: TensorCollection<f32, AutoDevice>,
{
    let mut model = dev.build_module::<B, f32>();
    <B::Built as LoadFromSafetensors<f32, AutoDevice>>::load_safetensors::<&str>(&mut model, &file_name).unwrap();
    model
}

pub fn re_init_best_and_latest<G: Game>(data_dir: &str)
where
    [(); G::TOTAL_MOVES]: Sized,
    [(); G::CHANNELS]: Sized,
    [(); <G::TotalBoardSize as ConstDim>::SIZE]: Sized,
    [(); 2 * <G::TotalBoardSize as ConstDim>::SIZE]: Sized,
    [(); <G::BoardSize as ConstDim>::SIZE]: Sized,
    BoardGameModel<G>: BuildOnDevice<AutoDevice, f32>,
{
    let file_name_best = format!("{}/best.safetensors", data_dir);
    let file_name_latest = format!("{}/latest.safetensors", data_dir);
    let file_name_control = format!("{}/control.safetensors", data_dir);

    let dev: AutoDevice = Default::default();

    let model = dev.build_module::<BoardGameModel<G>, f32>();

    model.save_safetensors(file_name_best).unwrap();
    model.save_safetensors(file_name_latest).unwrap();
    model.save_safetensors(file_name_control).unwrap();
}
