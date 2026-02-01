
use anyhow::{Result, Error};
use sm64_binds::GameState;
use serde::{Serialize, Deserialize};
use std::{fs::File, io::Read};

use crate::scoring::ScoringStrategy;

pub type Vec3d = (f32, f32, f32);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceScoring {
    goal_point: (f32, f32, f32)
}

impl DistanceScoring {
    pub fn new(goal_point: Vec3d) -> Self {
        Self {
            goal_point
        }
    }
}

fn distance(a: Vec3d, b: Vec3d) -> f32 {
    let d2 = (a.0-b.0)*(a.0-b.0) + (a.1-b.1)*(a.1-b.1) + (a.2-b.2)*(a.2-b.2);
    f32::sqrt(d2)
}

impl ScoringStrategy for DistanceScoring {
    fn default() -> Self {
        let goal_point: Vec3d = (0.0,0.0,0.0);
        Self::new(goal_point)
    }

    fn get_fitness(&self, tmp_states: &[GameState]) -> f32 {
        let distances: Vec<f32> = tmp_states.iter()
            .map(|s| f32::sqrt(distance(s.pos.into(), self.goal_point) ))
            .collect();


        let mut sum = 0.0;
        let mut discount = 1.0;
        let mut max_fitness = -f32::INFINITY;
        for d in distances {
            let score = -d;
            sum += score * discount;
            discount *= 0.99;
            max_fitness = f32::max(max_fitness, sum)
        }
        max_fitness
    }

    fn add_state(&mut self, _state: GameState) {}
    fn load(filename: &str) -> Result<Self, Error> {
        let mut file = File::open(filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let scoring: Self = postcard::from_bytes(&buffer)?;
        Ok(scoring)
    }
}

impl std::fmt::Display for DistanceScoring {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f, 
            "Scoring Strategy: Distance, [{}, {}, {}])", 
            self.goal_point.0, self.goal_point.1, self.goal_point.2
        )
    }
}