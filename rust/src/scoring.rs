use sm64_binds::GameState;

pub trait ScoringStrategy: Clone + Send + Sync {
    fn get_fitness(&self, tmp_states: &[GameState]) -> f32;
    fn add_state(&mut self, state: GameState);
}
#[derive(Debug, Clone)]
pub struct DefaultScoring;

impl ScoringStrategy for DefaultScoring {
    fn get_fitness(&self, tmp_states: &[GameState]) -> f32 {
        // if tmp_states.is_empty() {
        //     return -f32::INFINITY;
        // }

        // Equivalent to: scores = np.array([-state.pos[2] for state in tmp_states])
        // i = np.argmax(scores); return scores[i] - i
        // let best_val = tmp_states.iter().enumerate()
        //     .map(|(i, s)| {
        //         -s.pos[2] - (i as f32)
        //     })
        //     .max_by(|a, b| a.partial_cmp(b).unwrap())
        //     .unwrap();
        // best_val
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
