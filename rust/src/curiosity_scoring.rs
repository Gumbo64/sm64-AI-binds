
use anyhow::{Result, Error};
use sm64_binds::GameState;
use serde::{Serialize, Deserialize};
use std::{collections::HashMap, fs::File, io::Read};

use crate::scoring::ScoringStrategy;

type BucketKey = (ZoneKey, i32, i32, i32);
type ZoneKey = i32; // levelNum

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuriosityScoring {
    circle_radius: i32,
    pub bucket_counts: HashMap<BucketKey, u32>,
    /// Tracks total visits per (Course, Act, Area)
    zone_counts: HashMap<ZoneKey, u32>, 
    discretize_step: f32,
    discretize_min: f32,
}

impl CuriosityScoring {
    /// Helper to extract just the Zone from a full BucketKey
    fn get_zone_from_key(key: BucketKey) -> ZoneKey {
        key.0
    }

    fn add_circle(
        radius: i32, 
        key: BucketKey, 
        bucket_dict: &mut HashMap<BucketKey, u32>,
        zone_dict: &mut HashMap<ZoneKey, u32>
    ) {
        let (level_num, x, y, z) = key;
        
        // Update the zone count once per call (not inside the 3D loop)
        *zone_dict.entry(level_num).or_insert(0) += 1;

        for i in -radius..=radius {
            for j in -radius..=radius {
                for k in -radius..=radius {
                    let neighbor_key = (level_num, x + i, y + j, z + k);
                    *bucket_dict.entry(neighbor_key).or_insert(0) += 1;
                }
            }
        }
    }

    fn get_bucket_key(&self, state: &GameState) -> BucketKey {
        let x_bucket = self.discretize_position(state.pos[0]);
        let y_bucket = self.discretize_position(state.pos[1]);
        let z_bucket = self.discretize_position(state.pos[2]);

        (state.level_num, x_bucket, y_bucket, z_bucket)
    }

    fn discretize_position(&self, value: f32) -> i32 {
        ((value - self.discretize_min) / self.discretize_step).floor() as i32
    }
}

impl ScoringStrategy for CuriosityScoring {
    fn default() -> Self {
        let circle_radius = 1;
        let discretize_step = 1000.0 / ((2 * circle_radius + 1) as f32);
        
        Self {
            circle_radius,
            bucket_counts: HashMap::new(),
            zone_counts: HashMap::new(),
            discretize_step,
            discretize_min: -8002.0,
        }
    }
    fn get_fitness(&self, tmp_states: &[GameState]) -> f32 {
        let bucket_keys: Vec<BucketKey> = tmp_states.iter()
            .map(|s| self.get_bucket_key(s))
            .collect();

        let mut tmp_bucket_counts: HashMap<BucketKey, u32> = HashMap::new();
        let mut tmp_zone_counts: HashMap<ZoneKey, u32> = HashMap::new();
        let mut scores: Vec<f32> = Vec::with_capacity(bucket_keys.len());

        for (i, &key) in bucket_keys.iter().enumerate() {
            let zone_key = Self::get_zone_from_key(key);

            // 1. Get visits from global + local temporary counts
            let global_visits = self.bucket_counts.get(&key).copied().unwrap_or(0);
            let local_visits = tmp_bucket_counts.get(&key).copied().unwrap_or(0);
            let visits = global_visits + local_visits;

            // Visits in the zone (without coordinates)
            // Should 
            let global_zone_visits = self.zone_counts.get(&zone_key).copied().unwrap_or(0);
            let local_zone_visits = tmp_zone_counts.get(&zone_key).copied().unwrap_or(0);
            let total_zone_visits = global_zone_visits + local_zone_visits;

            // Example: Using both for curiosity. You could adjust this formula.
            let s1 = 1.0 / f32::sqrt(1.0 + (visits as f32));
            let s2 = 1.0 / f32::sqrt(1.0 + (total_zone_visits as f32));
            let s3 = (tmp_states[i].num_stars + tmp_states[i].in_credits) as f32;

            let s = s1 + s2 + 1000.0 * s3;
            scores.push(s);

            // 3. Update local counts for both buckets and zones
            Self::add_circle(self.circle_radius, key, &mut tmp_bucket_counts, &mut tmp_zone_counts);
        }

        let mut sum = 0.0;
        let mut discount = 1.0;
        let mut max_fitness = -f32::INFINITY;
        for score in scores {
            sum += score * discount;
            discount *= 0.99;
            max_fitness = f32::max(max_fitness, sum)
        }
        max_fitness
    }

    fn add_state(&mut self, state: GameState) {
        let key = self.get_bucket_key(&state);
        Self::add_circle(self.circle_radius, key, &mut self.bucket_counts, &mut self.zone_counts);
    }
    fn load(filename: &str) -> Result<Self, Error> {
        // 1. Open the file
        let mut file = File::open(filename)?;
        
        // 2. Read the entire file into a buffer
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // 3. Deserialize using postcard
        let scoring: CuriosityScoring = postcard::from_bytes(&buffer)?;
        
        Ok(scoring)
    }
}

impl std::fmt::Display for CuriosityScoring {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f, 
            "Scoring Strategy: Curiosity (Visited Buckets: {}, Zones: {})", 
            self.bucket_counts.len(), 
            self.zone_counts.len(),
        )
    }
}