use std::collections::HashMap;

use lazy_pbar::pbar;
use rand::{rngs::ThreadRng, seq::IteratorRandom};
use rust_games_shared::{Elo, Game, GameResult, Strategy};
pub struct Leaderboard<G: Game> {
    pub strategies: HashMap<String, Strategy<G>>,
    rng: ThreadRng,
}

impl<'a, G: Game> Leaderboard<G> {
    pub fn new(strategies: Vec<Strategy<G>>) -> Self {
        let rng = rand::thread_rng();
        let mut strats = HashMap::new();
        for strat in strategies {
            strats.insert(strat.name.clone(), strat);
        }
        Leaderboard {
            strategies: strats,
            rng: rng,
        }
    }

    fn update(&mut self, players: Vec<String>, result: GameResult) {
        match result {
            GameResult::Winner(winner_name) => {
                let old_winner_elo = self.strategies.get(&winner_name).unwrap().elo.clone();
                for player in players {
                    if player.eq(&winner_name) {
                        continue;
                    }
                    let player_elo = self.strategies.get(&player).unwrap().elo.clone();
                    let winner = self.strategies.get_mut(&winner_name).unwrap();
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
        let players = self
            .strategies
            .values()
            .choose_multiple(&mut self.rng, G::NUM_PLAYERS);
        let result = G::play_full_game(players.clone(), verbose);
        let player_names: Vec<String> = players.iter().map(|p| p.name.clone()).collect();
        self.update(player_names, result);
    }

    pub fn play_random_games(&mut self, n: usize) {
        for _ in pbar(0..n) {
            self.play_random_game(false);
        }
    }

    pub fn print(&self) {
        for strat_name in self.strategies.keys() {
            let elo = self.strategies.get(&strat_name.clone()).unwrap().elo;
            println!(
                "{} has ELO {} and {} wins",
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
