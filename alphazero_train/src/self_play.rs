use crate::get_train_examples::get_examples_until;
use alphazero::{load_from_file, update_on_many, BoardGameModel};
use dfdx::{
    optim::Adam,
    prelude::{BuildOnDevice, SaveToSafetensors},
    tensor::AutoDevice,
};
use rust_games_shared::Game;

pub fn self_play_iteration<G: Game + 'static>(
    in_model_name: &str,
    out_model_name: &str,
    num_examples: usize,
) where
    [(); G::CHANNELS * G::BOARD_SIZE * G::BOARD_SIZE]: Sized,
    [(); G::TOTAL_MOVES]: Sized
{
    let examples = get_examples_until::<G>(in_model_name, num_examples);

    let dev: AutoDevice = Default::default();

    let mut model = load_from_file::<
        G,
        <BoardGameModel<G> as BuildOnDevice<AutoDevice, f32>>::Built,
        BoardGameModel<G>,
    >(in_model_name, &dev);
    let mut opt = Adam::new(&model, Default::default());

    let _ = update_on_many(&mut model, examples, &mut opt, 1, dev);

    let mut file_name = "/Applications/Python 3.4/MyScripts/rust_games/data/".to_string();
    file_name.push_str(out_model_name);
    file_name.push_str(".safetensors");

    model.save_safetensors(file_name).unwrap();
}
