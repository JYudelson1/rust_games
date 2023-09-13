use dfdx::prelude::*;
use rust_games_shared::Game;

type BoardGameBaseModel<G: Game> = (
    Flatten2D,
    Linear<{ G::CHANNELS * G::BOARD_SIZE * G::BOARD_SIZE }, 100>,
    Linear<100, 10>
);
type BoardGamePolicyHead<G: Game> = Linear<10, {G::TOTAL_MOVES}>; //TODO
type BoardGameValueHead = Linear<10, 1>; //TODO

pub type BoardGameModel<G: Game> = ((
    BoardGameBaseModel<G>,
    SplitInto<(BoardGamePolicyHead<G>, BoardGameValueHead)>,
),);

pub fn load_from_file<
    G: Game,
    M: Module<
        Tensor3D<{ G::CHANNELS }, { G::BOARD_SIZE }, { G::BOARD_SIZE }>,
        Output = (Tensor<(Const<{G::TOTAL_MOVES}>,), f32, Cpu>, Tensor<(Const<1>,), f32, Cpu>),
        Error = CpuError,
    >,
    B: BuildOnDevice<Cpu, f32, Built = M>,
>(
    model_name: &str,
    dev: &Cpu,
) -> M
where
    [(); G::CHANNELS * G::BOARD_SIZE * G::BOARD_SIZE]: Sized,
    M: TensorCollection<f32, Cpu>,
{
    let mut file_name = "/Applications/Python 3.4/MyScripts/rust_games/data/".to_string();
    file_name.push_str(model_name);
    file_name.push_str(".safetensors");

    let mut model = dev.build_module::<B, f32>();
    <M as LoadFromSafetensors<f32, Cpu>>::load_safetensors::<&str>(&mut model, &file_name).unwrap();
    model
}

pub fn re_init_best_and_latest<G: Game>()
where
    [(); G::CHANNELS * G::BOARD_SIZE * G::BOARD_SIZE]: Sized,
    [(); G::TOTAL_MOVES]: Sized,
{
    let file_name_best = "/Applications/Python 3.4/MyScripts/rust_games/data/best.safetensors";
    let file_name_latest = "/Applications/Python 3.4/MyScripts/rust_games/data/latest.safetensors";
    let file_name_control =
        "/Applications/Python 3.4/MyScripts/rust_games/data/control.safetensors";

    let dev: Cpu = Default::default();

    let model = dev.build_module::<BoardGameModel<G>, f32>();

    model.save_safetensors(file_name_best).unwrap();
    model.save_safetensors(file_name_latest).unwrap();
    model.save_safetensors(file_name_control).unwrap();
}
