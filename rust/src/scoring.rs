use sm64_binds::GameState;

pub trait ScoringStrategy: Clone + Send + Sync {
    fn get_fitness(&self, tmp_states: &[GameState]) -> f32;
    fn add_state(&mut self, state: GameState);
}
#[derive(Debug, Clone)]
pub struct DefaultScoring;

impl ScoringStrategy for DefaultScoring {
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
}



// CURIOSITY SCORING
use std::collections::HashMap;
type BucketKey = (i32, i32, i32, i32, i32, i32);

#[derive(Debug, Clone)]
pub struct CuriosityScoring {
    circle_radius: i32,
    bucket_counts: HashMap<BucketKey, u32>,
    discretize_step: f32,
    discretize_min: f32,
}

impl CuriosityScoring {
    pub fn new() -> Self {
        let circle_radius = 1;
        // Python: 1000 / (2*radius + 1)
        let discretize_step = 1000.0 / ((2 * circle_radius + 1) as f32);
        
        Self {
            circle_radius,
            bucket_counts: HashMap::new(),
            discretize_step,
            discretize_min: -8002.0,
        }
    }

    fn add_circle(radius: i32, key: BucketKey, bucket_dict: &mut HashMap<BucketKey, u32>) {
        let (n, an, ai, x, y, z) = key;
        for i in -radius..=radius {
            for j in -radius..=radius {
                for k in -radius..=radius {
                    let neighbor_key = (n, an, ai, x + i, y + j, z + k);
                    *bucket_dict.entry(neighbor_key).or_insert(0) += 1;
                }
            }
        }
    }

    fn get_bucket_key(&self, state: &GameState) -> BucketKey {
        let x_bucket = self.discretize_position(state.pos[0]);
        let y_bucket = self.discretize_position(state.pos[1]);
        let z_bucket = self.discretize_position(state.pos[2]);

        (
            state.course_num,
            state.act_num,
            state.area_index,
            x_bucket,
            y_bucket,
            z_bucket,
        )
    }

    /// Replaces np.digitize with direct arithmetic for performance
    fn discretize_position(&self, value: f32) -> i32 {
        // Equivalent to finding the index in np.arange(min, max, step)
        ((value - self.discretize_min) / self.discretize_step).floor() as i32
    }
}

impl ScoringStrategy for CuriosityScoring {
    fn get_fitness(&self, tmp_states: &[GameState]) -> f32 {
        let bucket_keys: Vec<BucketKey> = tmp_states.iter()
            .map(|s| self.get_bucket_key(s))
            .collect();

        // Temporary local counts for this batch
        let mut tmp_bucket_counts: HashMap<BucketKey, u32> = HashMap::new();
        let mut scores: Vec<f32> = Vec::with_capacity(bucket_keys.len());

        for &key in bucket_keys.iter() {
            // Get visits from global + local temporary counts
            let global_visits = self.bucket_counts.get(&key).copied().unwrap_or(0);
            let local_visits = tmp_bucket_counts.get(&key).copied().unwrap_or(0);
            let visits = global_visits + local_visits;

            // let s = -(visits as f32);
            let s = 1.0 / f32::sqrt(1.0 + (visits as f32));
            scores.push(s);

            // Update local counts with the "circle" influence
            Self::add_circle(self.circle_radius, key, &mut tmp_bucket_counts);
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
        Self::add_circle(self.circle_radius, key, &mut self.bucket_counts);
    }
}