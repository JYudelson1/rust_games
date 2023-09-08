use dfdx::prelude::*;

use super::nn;
use super::nn::OthelloModel;
use super::shared::ModelError;
use rust_games::Game;

#[derive(Clone)]
struct ActionNode<G: Game> {
    action: Option<G::Move>,
    post_state: G,
    q: f32,   //q-value
    n: usize, //number of times this action was taken
    v: f32,   // initial estimate by the model of the position's value
    p: f32,   // prior probability this move is chosen
    children: Vec<Self>,
}

impl<G: Game> ActionNode<G> {
    pub fn best_child(&mut self, temperature: f32) -> (Option<(&mut ActionNode<G>, usize)>) {
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
}

struct MCTS<G: Game> {
    root: ActionNode<G>,
    model: Box<dyn Module<G, Output = (f32, f32), Error = ModelError>>,
    num_searches: usize,
    temperature: f32,
}

impl<'a, G: Game> MCTS<G> {
    fn spawn_children(&mut self, mut node: ActionNode<G>) {
        assert!(node.children.is_empty());

        let next_actions = node.post_state.legal_moves();
        for action in next_actions {
            let mut subgame = node.post_state.clone();
            subgame.make_move(action);
            // TODO: terminal states
            // TODO: understand when p turns negative
            let (p, v) = self.model.forward(subgame.clone());
            // self.model.forward(subgame.nn_input());
            let new_node: ActionNode<G> = ActionNode {
                action: Some(action),
                post_state: subgame,
                q: 0.0,
                n: 0,
                v: v,
                p: p,
                children: vec![],
            };

            node.children.push(new_node);
        }
    }

    fn traverse(&mut self, n: usize) {
        // n times:
        // traverse from the root until a leaf node is reached
        // expand it, backup the v-values to update q's and n's
        for _ in 0..n {
            let mut path: Vec<usize> = vec![];
            let mut current = &mut self.root.clone();
            let mut index = 0;

            while !current.children.is_empty() {
                (current, index) = current
                    .best_child(self.temperature)
                    .expect("There should be a child of current node!");
                path.push(index);
            }

            //Expand leaf node at non-terminal state
            if !current.post_state.is_over() {
                self.spawn_children(current.clone());
            }

            // Update parents in the path
            let mut parent = &mut self.root;
            for parent_index in path.iter() {
                (*parent).q = (parent.q * (parent.n as f32) + current.v) / ((parent.n + 1) as f32);
                (*parent).n = parent.n + 1;
                parent = parent.children.get_mut(*parent_index).unwrap();
            }
        }
    }
}
