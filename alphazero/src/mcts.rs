use std::cell::Cell;

use dfdx::prelude::*;

use rust_games_shared::{Game, PlayerError};

use crate::{UnfinishedTrainingExample, nn::load_from_file};

#[derive(Clone, Debug)]
struct ActionNode<G: Game> {
    pub action: Option<G::Move>,
    post_state: G,
    q: f32,   //q-value
    n: usize, //number of times this action was taken
    v: f32,   // initial estimate by the model of the position's value
    children: Vec<Self>,
}

impl<G: Game> ActionNode<G>
where
    Tensor<(Const<{G::CHANNELS}>, G::BoardSize, G::BoardSize), f32, AutoDevice>: Sized,
{
    pub fn best_child(&mut self, temperature: f32) -> Option<(&mut ActionNode<G>, usize)> {
        let mut best = None;
        let mut highest_u: Option<f32> = None;
        let mut best_index: usize = 0;

        let root_sum_visits: usize = self.children.iter().map(|act| act.n).sum();

        for (i, child) in &mut self.children.iter_mut().enumerate() {
            let u = child.q
                + (temperature * child.v * (root_sum_visits as f32) / (1.0 + child.n as f32));

            if highest_u == None || u > highest_u.unwrap() {
                best = Some(child);
                highest_u = Some(u);
                best_index = i;
            }
        }

        match best {
            Some(child) => Some((child, best_index)),
            None => None,
        }
    }

    fn spawn_children(
        &mut self,
        model: &impl Module<
                Tensor<(Const<{G::CHANNELS}>, G::BoardSize, G::BoardSize), f32, AutoDevice>,
                Output = (
                    Tensor<(Const<{G::TOTAL_MOVES}>,), f32, AutoDevice>,
                    Tensor<(Const<1>,), f32, AutoDevice>,
                ),
                Error = <AutoDevice as HasErr>::Err,
            >,
    ) {
        if !self.children.is_empty(){
            return;
        }

        let next_actions = self.post_state.legal_moves();
        for action in next_actions {
            let mut subgame = self.post_state.clone();
            subgame.make_move(action);

            let (_, v) = model.forward(subgame.to_nn_input());

            let new_node: ActionNode<G> = ActionNode {
                action: Some(action),
                post_state: subgame,
                q: 0.0,
                n: 0,
                v: v.array()[0],
                children: vec![],
            };

            self.children.push(new_node);
        }
    }

    pub fn to_unfinished_example(&self) -> UnfinishedTrainingExample<G> {
        let position = self.post_state.to_nn_input();
        let next_move_probs = self.children
            .iter()
            .map(|node| (node.action.expect("Child must have an associated action"), node.n))
            .collect();
        UnfinishedTrainingExample::new(position, next_move_probs)
    }
}

pub struct MCTS<G: Game, M: Module<
            Tensor<(Const<{G::CHANNELS}>, G::BoardSize, G::BoardSize), f32, AutoDevice>,
            Output = (
                Tensor<(Const<{G::TOTAL_MOVES}>,), f32, AutoDevice>,
                Tensor<(Const<1>,), f32, AutoDevice>,
            ),
            Error = <AutoDevice as HasErr>::Err,
        >> {
    root: Cell<ActionNode<G>>,
    pub model: M,
    pub temperature: f32,
    pub train_examples: Option<Vec<UnfinishedTrainingExample<G>>>
}

impl<'a, G: Game, M: Module<
            Tensor<(Const<{G::CHANNELS}>, G::BoardSize, G::BoardSize), f32, AutoDevice>,
            Output = (
                Tensor<(Const<{G::TOTAL_MOVES}>,), f32, AutoDevice>,
                Tensor<(Const<1>,), f32, AutoDevice>,
            ),
            Error = <AutoDevice as HasErr>::Err,
        >> MCTS<G, M>
where
    Tensor<(Const<{G::CHANNELS}>, G::BoardSize, G::BoardSize), f32, AutoDevice>: Sized,
{
    fn traverse(&mut self, n: usize) {
        // n times:
        // traverse from the root until a leaf node is reached
        // expand it, backup the v-values to update q's and n's
        for _ in 0..n {
            let mut path: Vec<usize> = vec![];
            let child_v;

            {
                // Inner-ing scope to make compiler happy
                let mut current = self.root.get_mut();
                let mut index;

                while !current.children.is_empty() {
                    (current, index) = current
                        .best_child(self.temperature)
                        .expect("There should be a child of current node!");
                    path.push(index);
                }

                // Store this v for updating parents in path
                child_v = current.v;

                //Expand leaf node at non-terminal state
                if !current.post_state.is_over() {
                    current.spawn_children(&self.model);
                }
            }

            // Update parents in the path
            let mut parent = self.root.get_mut();
            for parent_index in path.iter() {
                (*parent).q = (parent.q * (parent.n as f32) + child_v) / ((parent.n + 1) as f32);
                (*parent).n = parent.n + 1;
                parent = parent.children.get_mut(*parent_index).unwrap();
            }
        }
    }

    pub fn choose_move(&mut self, game: &G) -> Result<G::Move, PlayerError> {
        // First, update on opponent's move (if applicable)
        let root = self.root.get_mut();
        if game.get_board() != root.post_state.get_board() {
            // If the game board is a child of this state:
            // Use that subtree as the root
            // But first, make sure children are spawned
            root.spawn_children(&self.model);
            let mut found = None;
            for child in self.root.get_mut().children.iter() {
                if child.post_state.get_board() == game.get_board() {
                    found = Some(child);
                }
            }

            // Change the root node to the current game
            // (Which is a child of the old root)
            self.root = Cell::new(
                found
                    .expect(
                        "The game given as input is not one move after the game stored in the MCTS",
                    )
                    .clone(),
            );
            // Else:
            // Should maybe use the new game as a new root
            // But for now, panic
        }
        self.traverse(100);
        let r = self.root.get_mut();
        if let Some(examples) = &mut self.train_examples {
            let ex = r.to_unfinished_example();
            examples.push(ex);
        }
        let best_child = r.best_child(self.temperature);

        match best_child {
            Some((child, _)) => {
                let action = child.action.unwrap();
                // Update board based on move made
                *r = child.clone();
                Ok(action)
            },
            None => Err(PlayerError::NoLegalMoves),
        }
    }

    pub fn new(
        root: G,
        model: M,
        temperature: f32,
        training: bool
    ) -> Self {
        let (_, v) = model.forward(root.to_nn_input());
        let train_examples = if training {Some(vec![])} else {None};
        Self {
            root: Cell::new(ActionNode {
                action: None,
                post_state: root,
                q: 0.0,
                n: 0,
                v: v.array()[0],
                children: vec![],
            }),
            model: model,
            temperature: temperature,
            train_examples
        }
    }

    pub fn new_from_file<B: BuildOnDevice<AutoDevice, f32, Built = M>>(root: G, temperature: f32, model_name: &str, dev: &AutoDevice, training: bool) -> Self
    where M: TensorCollection<f32, AutoDevice>,
        [(); G::TOTAL_MOVES]: Sized
        {
            let model = load_from_file::<G, B>(model_name, dev);
        
            Self::new(root, model, temperature, training)
    }

    pub fn save_nn(&self, model_name: &str) 
    where M: TensorCollection<f32, AutoDevice>
    {
        let mut file_name = "/Applications/Python 3.4/MyScripts/rust_games/data/".to_string();
        file_name.push_str(model_name);
        file_name.push_str(".safetensors");

        self.model.save_safetensors(file_name).unwrap();
    }

    pub fn reset_board(&mut self) {
        let g = G::new();
        let (_, v) = self.model.forward(g.to_nn_input());
        self.root.replace(ActionNode {
            action: None,
            post_state: g,
            q: 0.0,
            n: 0,
            v: v.array()[0],
            children: vec![],
        });
    }
}
