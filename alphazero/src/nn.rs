use dfdx::prelude::*;
use rust_games_shared::Game;

//TODO: Make more generic??
type BoardGameBaseModel<G: Game> = (
    Flatten2D,
    Linear<{ G::CHANNELS * G::BOARD_SIZE * G::BOARD_SIZE }, 10>,
);
type BoardGamePolicyHead = Linear<10, 1>; //TODO
type BoardGameValueHead = Linear<10, 1>; //TODO

pub type BoardGameModel<G: Game> = ((
    BoardGameBaseModel<G>,
    SplitInto<(
        BoardGamePolicyHead, //TODO
        BoardGameValueHead,  //TODO
    )>,
),);
