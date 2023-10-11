use std::{collections::HashMap, rc::Rc};

use indicatif::{ProgressBar, ProgressStyle};
use rand::{rngs::ThreadRng, seq::IteratorRandom};
use rust_games_shared::{Elo, Game, GameResult, Player, Strategy};
pub struct Leaderboard<G: Game> {
    pub strategies: HashMap<usize, Strategy<G>>,
    rng: ThreadRng,
}

impl<'a, G: Game> Leaderboard<G> {
    pub fn new(strategies: Vec<Strategy<G>>) -> Self {
        let rng = rand::thread_rng();
        let mut strats: HashMap<usize, Strategy<G>> = HashMap::new();
        for (i, strat) in strategies.iter().enumerate() {
            strats.insert(i, strat.clone());
        }
        Leaderboard {
            strategies: strats,
            rng: rng,
        }
    }

    fn update(&mut self, players: Vec<usize>, result: GameResult) {
        match result {
            GameResult::Winner(winner_id) => {

                let winner_index = players[winner_id as usize];

                let old_winner_elo = self.strategies.get(&winner_index).unwrap().elo.clone();
                for player in players {
                    if player.eq(&winner_index) {
                        continue;
                    }
                    let player_elo = self.strategies.get(&player).unwrap().elo.clone();
                    let winner = self.strategies.get_mut(&winner_index).unwrap();
                    (*winner).elo = elo_win(winner.elo, player_elo);
                    (*self.strategies.get_mut(&player).unwrap()).elo =
                        elo_loss(player_elo, old_winner_elo)
                }
            }
            GameResult::Tie => {
                assert!(players.len() == 2); //TODO: extend this
                let player_1_elo = self.strategies.get(&players[0]).unwrap().elo.clone();
                let player_2 = self.strategies.get_mut(&players[1]).unwrap();
                let old_player_2_elo = player_2.elo.clone();
                (*player_2).elo = elo_tie(player_2.elo, player_1_elo);
                (*self.strategies.get_mut(&players[0]).unwrap()).elo =
                    elo_tie(player_1_elo, old_player_2_elo)
            }
            GameResult::Ranking(_ordered_names) => todo!(),
        }
    }

    pub fn play_random_game(&mut self, verbose: bool) {
        let player_indices: Vec<usize> = self
            .strategies
            .keys()
            .choose_multiple(&mut self.rng, G::NUM_PLAYERS)
            .iter()
            .map(|k| **k)
            .collect();

        let players: Vec<Strategy<G>> = player_indices
            .iter()
            .map(|key| {
                Rc::<(dyn Player<G> + 'static)>::get_mut(&mut self.strategies.get_mut(key).unwrap().player)
                .unwrap()
                .reset();
            
                self.strategies.get_mut(key).unwrap().clone()
            })
            .collect();

        let players = players.iter().map(|strat| strat).collect();

        //println!("{:?}", player_names);
        let result = G::play_full_game(players, verbose);

        self.update(player_indices, result);
    }

    pub fn play_random_games(&mut self, n: usize) {
        let progress_bar = ProgressBar::new(n as u64).with_style(
            ProgressStyle::default_bar()
                .template(
                    "Playing Games... |{wide_bar}| {pos}/{len} [{elapsed_precise}>{eta_precise}]",
                )
                .unwrap(),
        );
        progress_bar.inc(0);
        for _ in 0..n {
            self.play_random_game(false);
            progress_bar.inc(1);
        }
        progress_bar.finish_and_clear();
    }

    pub fn print(&self) {
        for strat_id in self.strategies.keys() {
            let strat = self.strategies.get(&strat_id).unwrap();
            let elo = strat.elo;
            let strat_name = &strat.name;
            println!(
                "{} has ELO {:.0} and {} wins",
                strat_name, elo.rating, elo.wins
            )
        }
    }
}

fn elo_helper(rating1: f64, rating2: f64, score: f64, k_factor: f64) -> f64 {
    let prob_win = 1.0 / (1.0 + 10.0_f64.powf((rating2 - rating1) / 400.0));
    let new_rating = rating1 + k_factor * (score - prob_win);
    new_rating
}

fn k_factor(elo: Elo) -> f64 {
    if elo.games_played <= 100 {
        64.0
    } else if elo.games_played <= 1000 {
        32.0
    } else {
        16.0
    }
}

fn elo_win(winner: Elo, loser: Elo) -> Elo {
    let new_rating = elo_helper(winner.rating, loser.rating, 1.0, k_factor(winner));
    Elo {
        rating: new_rating,
        games_played: winner.games_played + 1,
        wins: winner.wins + 1,
    }
}
fn elo_loss(loser: Elo, winner: Elo) -> Elo {
    let new_rating = elo_helper(loser.rating, winner.rating, 0.0, k_factor(loser));
    Elo {
        rating: new_rating,
        games_played: loser.games_played + 1,
        wins: loser.wins,
    }
}
fn elo_tie(player1: Elo, player2: Elo) -> Elo {
    let new_rating = elo_helper(player1.rating, player2.rating, 0.5, k_factor(player1));
    Elo {
        rating: new_rating,
        games_played: player1.games_played + 1,
        wins: player1.wins,
    }
}
