use crate::get_train_examples::get_examples_until;
use alphazero::{load_from_file, update_on_many};
use dfdx::{optim::Adam, prelude::*};
use rust_games_shared::Game;

pub fn self_play_iteration<G: Game + 'static, B: BuildOnDevice<AutoDevice, f32> + 'static>(
    in_model_name: &str,
    out_model_name: &str,
    data_dir: &str,
    num_examples: usize,
) where
    [(); G::TOTAL_MOVES]: Sized,
    [(); G::CHANNELS]: Sized,
    [(); <G::TotalBoardSize as ConstDim>::SIZE]: Sized,
    [(); 2 * <G::TotalBoardSize as ConstDim>::SIZE]: Sized,
    [(); <G::BoardSize as ConstDim>::SIZE]: Sized,
    <B as BuildOnDevice<AutoDevice, f32>>::Built: Module<
        Tensor<
            (
                Const<{ G::CHANNELS }>,
                <G as Game>::BoardSize,
                <G as Game>::BoardSize,
            ),
            f32,
            AutoDevice,
        >,
        Output = (
            Tensor<(Const<{ G::TOTAL_MOVES }>,), f32, AutoDevice>,
            Tensor<(Const<1>,), f32, AutoDevice>,
        ),
        Error = <AutoDevice as HasErr>::Err,
    >,
    <B as BuildOnDevice<AutoDevice, f32>>::Built: ModuleMut<
        <Tensor<
            (
                Const<{ G::CHANNELS }>,
                <G as Game>::BoardSize,
                <G as Game>::BoardSize,
            ),
            f32,
            AutoDevice,
        > as Trace<f32, AutoDevice>>::Traced,
        Output = (
            Tensor<(Const<{ G::TOTAL_MOVES }>,), f32, AutoDevice, OwnedTape<f32, AutoDevice>>,
            Tensor<(Const<1>,), f32, AutoDevice, OwnedTape<f32, AutoDevice>>,
        ),
        Error = <AutoDevice as HasErr>::Err,
    >,
{
    let examples = get_examples_until::<G, B>(in_model_name, data_dir, num_examples);

    let dev: AutoDevice = Default::default();

    let in_file_name = format!("{}/{}.safetensors", data_dir, in_model_name);

    let mut model: <B as BuildOnDevice<AutoDevice, f32>>::Built =
        load_from_file::<G, B>(&in_file_name, &dev);
    let mut opt = Adam::new(&model, Default::default());

    let examples_ref: Vec<&alphazero::TrainingExample<G>> = examples.iter().map(|ex| ex).collect();
    update_on_many(&mut model, examples_ref, &mut opt, 1, &dev).unwrap();

    let out_file_name = format!("{}/{}.safetensors", data_dir, out_model_name);
    model.save_safetensors(out_file_name).unwrap();
}
