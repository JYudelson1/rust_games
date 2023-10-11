use crate::{nn::load_from_file, UnfinishedTrainingExample};
use dfdx::prelude::*;
use rand::{distributions::WeightedIndex, prelude::Distribution, thread_rng};
use rust_games_shared::{Game, PlayerError};
use std::{cell::Cell, collections::HashMap};

#[derive(Clone, Debug)]
pub struct ActionNode<G: Game> 
where [(); G::TOTAL_MOVES]: Sized{
    pub action: Option<G::Move>,
    post_state: G,
    q: f32,     //q-value
    n: usize,   //number of times this action was taken
    pub v: f32, // initial estimate by the model of the position's value
    pub p: [f32; G::TOTAL_MOVES], // prior probability array over all this nodes possible children
    children: Vec<Self>,
}

impl<G: Game> ActionNode<G>
where
    Tensor<(Const<{G::CHANNELS}>, G::BoardSize, G::BoardSize), f32, AutoDevice>: Sized,
    [(); G::TOTAL_MOVES]: Sized
{
    /*
    Chooses the next child to examine during the MCTS traversal.
    This is calculated using AlphaGo's variant PUCT algorithm, described [here](https://gwern.net/doc/reinforcement-learning/model/alphago/2017-silver.pdf#page=8).
     */
    pub fn best_child_traversal(&mut self, temperature: f32, index_map: &HashMap<G::Move, usize>) -> Option<(&mut ActionNode<G>, usize)> {
        let mut best = None;
        let mut highest_u: Option<f32> = None;
        let mut best_index: usize = 0;

        let root_visits: f32 = (self.n as f32).sqrt();

        for (i, child) in &mut self.children.iter_mut().enumerate() {

            // If the move being selected is for second, invert the q-value when deciding to go that way.
            // I.e. first player chooses whats best for first player, same for second
            let turn_order_factor = match self.post_state.current_player() {
                rust_games_shared::PlayerId::First => 1.0_f32,
                rust_games_shared::PlayerId::Second => -1.0_f32,
            };
            // TODO: This temperature should probably depend on the number of moves throughout the game, like in the AG paper
            let p_index = index_map.get(&child.action.unwrap()).unwrap();
            let u = child.q * turn_order_factor
                + (temperature * (child.p)[*p_index] * root_visits / (1.0 + child.n as f32));

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

    /*
    Chooses the next child to examine during the MCTS traversal.
    This is calculated using exponentiated visit count, described [here](https://gwern.net/doc/reinforcement-learning/model/alphago/2017-silver.pdf#page=8).
     */
    fn best_child_visitcount(&mut self, temperature: f32) -> Option<&mut ActionNode<G>> {
        let mut visit_counts = vec![];

        for child in self.children.iter(){
            visit_counts.push((child.n as f32).powf(temperature));
        }

        let dist = WeightedIndex::new(&visit_counts).unwrap();
        let chosen_index = dist.sample(&mut thread_rng());

        Some(&mut self.children[chosen_index])
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

            let (p, v) = model.forward(subgame.to_nn_input());

            let new_node: ActionNode<G> = ActionNode {
                action: Some(action),
                post_state: subgame,
                q: 0.0,
                n: 0,
                v: v.array()[0],
                p: p.array(),
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
    pub root: Cell<ActionNode<G>>,
    pub model: M,
    pub temperature: f32,
    pub train_examples: Option<Vec<UnfinishedTrainingExample<G>>>,
    pub traverse_iter: usize,
    index_map: HashMap<G::Move, usize>
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
                        .best_child_traversal(self.temperature, &self.index_map)
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
        self.traverse(self.traverse_iter);
        let r = self.root.get_mut();
        if let Some(examples) = &mut self.train_examples {
            let ex = r.to_unfinished_example();
            examples.push(ex);
        }
    
        let best_child = r.best_child_visitcount(self.temperature);

        match best_child {
            Some(child) => {
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
        training: bool,
        traverse_iter: usize
    ) -> Self {
        let (p, v) = model.forward(root.to_nn_input());
        let train_examples = if training {Some(vec![])} else {None};
        let mut index_map = HashMap::new();

        for (i, action) in G::all_possible_moves().iter().enumerate() {
            index_map.insert(*action, i);
        }

        Self {
            root: Cell::new(ActionNode {
                action: None,
                post_state: root,
                q: 0.0,
                n: 0,
                v: v.array()[0],
                p: p.array(),
                children: vec![],
            }),
            model: model,
            temperature: temperature,
            train_examples,
            traverse_iter,
            index_map
        }
    }

    pub fn new_from_file<B: BuildOnDevice<AutoDevice, f32, Built = M>>(root: G, temperature: f32, file_name: &str, dev: &AutoDevice, training: bool, traverse_iter: usize) -> Self
    where M: TensorCollection<f32, AutoDevice>,
        [(); G::TOTAL_MOVES]: Sized
        {
            let model = load_from_file::<G, B>(file_name, dev);
        
            Self::new(root, model, temperature, training, traverse_iter)
    }

    pub fn save_nn(&self, model_name: &str, data_dir: &str) 
    where M: TensorCollection<f32, AutoDevice>
    {
        let file_name = format!("{}/{}.safetensors", data_dir, model_name);

        self.model.save_safetensors(file_name).unwrap();
    }

    pub fn reset_board(&mut self) {
        let g = G::new();
        let (p, v) = self.model.forward(g.to_nn_input());
        self.root.replace(ActionNode {
            action: None,
            post_state: g,
            q: 0.0,
            n: 0,
            v: v.array()[0],
            p: p.array(),
            children: vec![],
        });
    }
}

pub struct MCTSConfig {
    pub traversal_iter: usize,
    pub temperature: f32,
}
