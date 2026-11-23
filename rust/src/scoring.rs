use anyhow::{Result, Error};
use sm64_binds::{GamePad, GameState, SM64GameGenerator};
use serde::{Serialize, Deserialize};
use std::fmt::Display;

pub fn load_from_path<S: ScoringStrategy>(mut scoring: S, path: Vec<GamePad>) -> Result<S, Error> {
    let game_gen = SM64GameGenerator::new(crate::util::rom_bytes()?)?;
    let mut base_game = game_gen.create_game()?;
    
    for pad in path {
        base_game.step_game(pad)?;
        let state = base_game.get_game_state()?;
        scoring.add_state(state);
    }
    Ok(scoring)
}

pub trait ScoringStrategy: Clone + Send + Sync + Serialize + Display {
    fn default() -> Self;
    fn get_fitness(&self, tmp_states: &[GameState]) -> f32;
    fn add_state(&mut self, state: GameState);
    fn load(filename: &str) -> Result<Self, Error>;
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultScoring;

impl ScoringStrategy for DefaultScoring {
    fn default() -> Self {Self{}}
    fn get_fitness(&self, tmp_states: &[GameState]) -> f32 {
        let mut sum = 0.0;
        let mut discount = 1.0;
        let mut max_fitness = -f32::INFINITY;
        for s in tmp_states {
            let score = -s.pos[2];
            sum += score * discount;
            discount *= 0.99;
            max_fitness = f32::max(max_fitness, sum)
        }
        max_fitness
    }

    fn add_state(&mut self, _state: GameState) {}
    fn load(_filename: &str) -> Result<Self, Error> {Ok(Self {})}
}

impl std::fmt::Display for DefaultScoring {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Scoring Strategy: Default (Z-Position)")
    }
}

