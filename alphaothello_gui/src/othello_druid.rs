use std::rc::Rc;

use alphazero::BoardGameModel;
use dfdx::prelude::*;
use druid::widget::{Button, CrossAxisAlignment, Flex, Label, Painter};
use druid::{Color, Data, Env, Lens, RenderContext, Widget, WidgetExt};
use rust_games_games::{Othello, OthelloState};
use rust_games_shared::{Game, PlayerId};

#[derive(Clone, Data, Lens)]
pub(crate) struct SimpleOthello {
    board: [[u8; 8]; 8],
    playing: u8,
    p_board: [[String; 8]; 8],
    model: Rc<<BoardGameModel<Othello> as BuildOnDevice<AutoDevice, f32>>::Built>,
    last_v_value: f32,
}

impl SimpleOthello {
    pub(crate) fn cycle(&mut self, x: usize, y: usize) {
        self.board[x][y] = match self.board[x][y] {
            0 => 1,
            1 => 2,
            2 => 0,
            _ => panic!("Bad GUI"),
        };
    }

    pub(crate) fn cycle_player(&mut self) {
        self.playing = match self.playing {
            0 => 1,
            1 => 0,
            _ => panic!("Bad GUI"),
        };
    }

    pub(crate) fn new(
        model: <BoardGameModel<Othello> as BuildOnDevice<AutoDevice, f32>>::Built,
    ) -> Self {
        let mut p: [[String; 8]; 8] = Default::default();
        for x in 0..8 {
            for y in 0..8 {
                p[x][y] = "0.0".to_string();
            }
        }

        Self {
            board: [[0; 8]; 8],
            playing: 0,
            p_board: p,
            model: Rc::new(model),
            last_v_value: 0.0,
        }
    }

    pub(crate) fn to_game(&self) -> Othello {
        let playing = match self.playing {
            0 => PlayerId::First,
            1 => PlayerId::Second,
            _ => panic!(),
        };

        let mut board = [[OthelloState::Empty; 8]; 8];

        for x in 0..8 {
            for y in 0..8 {
                board[x][y] = match self.board[x][y] {
                    0 => OthelloState::Empty,
                    1 => OthelloState::Black,
                    2 => OthelloState::White,
                    _ => panic!("Bad GUI"),
                }
            }
        }

        Othello::new_from_board(board, playing)
    }

    fn run_model(&mut self) {
        let input = self.to_game().to_nn_input();
        let (p, v) = self.model.forward(input);

        self.last_v_value = v.array()[0];

        for x in 0..8 {
            for y in 0..8 {
                self.p_board[x][y] = format!("{:.3}", p.array()[(8 * x) + y]); // Maybe swap?
            }
        }
    }
}

fn new_othello_space(x: usize, y: usize) -> impl Widget<SimpleOthello> {
    let painter = Painter::new(move |ctx, data: &SimpleOthello, _env| {
        let bounds = ctx.size().to_rect();

        match data.board[x][y] {
            0 => ctx.fill(bounds, &Color::GREEN),
            1 => ctx.fill(bounds, &Color::BLACK),
            2 => ctx.fill(bounds, &Color::WHITE),
            _ => panic!("Bad GUI"),
        }

        if ctx.is_hot() {
            ctx.stroke(bounds.inset(-0.5), &Color::WHITE, 1.0);
        }

        // if ctx.is_active() {
        //     ctx.fill(bounds, &env.get(theme::PRIMARY_LIGHT));
        // }
    });

    Button::dynamic(move |data: &SimpleOthello, _: &Env| data.p_board[x][y].clone())
        .center()
        .background(painter)
        .expand()
        .on_click(move |_ctx, data: &mut SimpleOthello, _env| data.cycle(x, y))
}

fn new_othello_row(y: usize) -> impl Widget<SimpleOthello> {
    Flex::row()
        .with_flex_child(new_othello_space(0, y), 1.0)
        .with_spacer(1.0)
        .with_flex_child(new_othello_space(1, y), 1.0)
        .with_spacer(1.0)
        .with_flex_child(new_othello_space(2, y), 1.0)
        .with_spacer(1.0)
        .with_flex_child(new_othello_space(3, y), 1.0)
        .with_spacer(1.0)
        .with_flex_child(new_othello_space(4, y), 1.0)
        .with_spacer(1.0)
        .with_flex_child(new_othello_space(5, y), 1.0)
        .with_spacer(1.0)
        .with_flex_child(new_othello_space(6, y), 1.0)
        .with_spacer(1.0)
        .with_flex_child(new_othello_space(7, y), 1.0)
}

fn playing() -> impl Widget<SimpleOthello> {
    let painter = Painter::new(move |ctx, data: &SimpleOthello, _env| {
        let bounds = ctx.size().to_rect();

        match data.playing {
            0 => ctx.fill(bounds, &Color::BLACK),
            1 => ctx.fill(bounds, &Color::WHITE),
            _ => panic!("Bad GUI"),
        }

        if ctx.is_hot() {
            ctx.stroke(bounds.inset(-0.5), &Color::WHITE, 1.0);
        }

        // if ctx.is_active() {
        //     ctx.fill(bounds, &env.get(theme::PRIMARY_LIGHT));
        // }
    });

    Button::dynamic(move |data: &SimpleOthello, _: &Env| format!("Player: {}", data.playing))
        .center()
        .background(painter)
        .expand()
        .on_click(move |_ctx, data: &mut SimpleOthello, _env| data.cycle_player())
}

pub(crate) fn new_othello_board() -> impl Widget<SimpleOthello> {
    Flex::column()
        .cross_axis_alignment(CrossAxisAlignment::End)
        .with_flex_child(new_othello_row(0), 1.0)
        .with_flex_child(new_othello_row(1), 1.0)
        .with_flex_child(new_othello_row(2), 1.0)
        .with_flex_child(new_othello_row(3), 1.0)
        .with_flex_child(new_othello_row(4), 1.0)
        .with_flex_child(new_othello_row(5), 1.0)
        .with_flex_child(new_othello_row(6), 1.0)
        .with_flex_child(new_othello_row(7), 1.0)
        .with_flex_child(playing(), 0.5)
}

fn analysis_area() -> impl Widget<SimpleOthello> {
    Flex::column()
        .cross_axis_alignment(CrossAxisAlignment::End)
        .with_flex_child(
            Button::new("Run model")
                .center()
                .expand()
                .on_click(move |_ctx, data: &mut SimpleOthello, _env| data.run_model()),
            1.0,
        )
        .with_spacer(0.4)
        .with_flex_child(
            Label::dynamic(|data: &SimpleOthello, _: &Env| format!("{:.3}", data.last_v_value))
                .center()
                .expand(),
            0.7,
        )
}

pub(crate) fn analysis_board() -> impl Widget<SimpleOthello> {
    Flex::row()
        .with_flex_child(new_othello_board(), 1.0)
        .with_spacer(1.0)
        .with_flex_child(analysis_area(), 0.2)
}
