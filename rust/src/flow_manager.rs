use std::fs::File;
use std::io::Write;

use rand::prelude::*;
use rayon::prelude::*;
use anyhow::{Result, Error};
use papaya::HashMap;

use sm64_binds::{GamePad, SM64Game, SM64GameGenerator};

type Genome = Vec<GamePad>;

use crate::distance_scoring::{DistanceScoring, Vec3d};
use crate::population::Population;
use crate::scoring::ScoringStrategy;
use crate::util::{random_gamepad, write_gamepads_to_file};

type BucketKey = (i32, i32, i32, i32);


struct FlowNode {
    memory_bytes: Vec<u8>,
    path: Vec<GamePad>
}


struct SM64ThreadState {
    tmp_game: SM64Game
}

impl SM64ThreadState {
    fn new() -> Result<Self, Error> {
        let game_gen = SM64GameGenerator::new(include_bytes!("../../baserom.us.z64").to_vec())?;
        let tmp_game = game_gen.create_game()?;
        Ok(Self { tmp_game })
    }

    fn get_fitness(&mut self, genome: &Genome, node: &FlowNode, goal_point: Vec3d) -> Result<f32, Error> {
        let scoring = DistanceScoring::new(goal_point);
        self.tmp_game.write_memory_bytes(node.memory_bytes.as_slice());

        let mut tmp_states = Vec::with_capacity(genome.len());
        for pad in genome {
            self.tmp_game.step_game(*pad)?;
            tmp_states.push(self.tmp_game.get_game_state()?);
        }

        Ok(scoring.get_fitness(&tmp_states))
    }
}


// --- POPULATION ---

pub struct FlowManager {
    pop: Population<DistanceScoring>
}

impl FlowManager {
    pub fn new(pop_size: usize, genome_length: usize) -> Result<Self, Error> {
        println!("making temp games");

        let thread_count = rayon::current_num_threads();
        let thread_states: Result<Vec<SM64ThreadState>, Error> = (0..thread_count)
            .into_par_iter()
            .map(|_| SM64ThreadState::new())
            .collect();
        let thread_states = thread_states?;


        let mut rng = rand::rng();
        let pop = (0..pop_size)
            .map(|_| (0..genome_length).map(|_| random_gamepad(&mut rng)).collect())
            .collect();

        Ok(Self {
            pop_size,
            genome_length,
            mutation_rate: 0.05,
            pop_elite: (pop_size / 5) + 1,
            pop,
            buckets: HashMap::new(),
            thread_states,
        })
    }

    pub fn iterate(&mut self) -> Result<f32, Error> {
        let num_workers = self.thread_states.len();
        let chunk_size = (self.pop.len() + num_workers - 1) / num_workers;

        // We iterate over the GAMES, not the population. 
        // This guarantees we use exactly one game per thread.
        let results: Result<Vec<Vec<(usize, f32)>>, Error> = self.thread_states
            .par_iter_mut()
            .enumerate()
            .map(|(worker_idx, thread_state)| {
                let start = worker_idx * chunk_size;
                let end = (start + chunk_size).min(self.pop.len());
                
                // If this worker has no work (pop_size < num_workers), skip
                if start >= self.pop.len() { return Ok(vec![]); }

                let mut local_results = Vec::with_capacity(end - start);

                for (local_idx, genome) in self.pop[start..end].iter().enumerate() {
                    let global_idx = start + local_idx;
                    let score = thread_state.get_fitness(genome)?;
                    local_results.push((global_idx, score));
                }
                Ok(local_results)
            })
            .collect();

        // Flatten results and find best fitness
        let mut fitness_results: Vec<(usize, f32)> = results?.into_iter().flatten().collect();
        fitness_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        // Sort by fitness descending (best first)
        let best_fitness = fitness_results[0].1;

        // 2. Handle Elitism (Sequential is fine here as it's usually a small slice)
        let mut new_pop: Vec<Genome> = fitness_results.iter()
            .take(self.pop_elite)
            .map(|(idx, _)| self.pop[*idx].clone())
            .collect();

        // 3. Parallel Offspring Generation
        let offspring_needed = self.pop_size - new_pop.len();

        let mut offspring: Vec<Genome> = (0..offspring_needed)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng(); // Thread-local RNG
                let idx1 = rng.random_range(0..self.pop_size);
                let idx2 = rng.random_range(0..self.pop_size);
                
                let parent1 = &self.pop[idx1];
                let parent2 = &self.pop[idx2];
                
                // Use map to generate the genome in parallel if genome_length is huge, 
                // but usually a standard loop here is faster due to lower overhead.
                (0..self.genome_length).map(|j| {
                    let r: f64 = rng.random();
                    let mut gene = if r < 0.4 {
                        parent1[j].clone()
                    } else if r < 0.8 {
                        parent2[j].clone()
                    } else if r < 0.9 {
                        let mut g = parent1[j].clone();
                        g.button = parent2[j].button;
                        g
                    } else {
                        let mut g = parent2[j].clone();
                        g.button = parent1[j].button;
                        g
                    };

                    if rng.random::<f64>() < self.mutation_rate {
                        gene = random_gamepad(&mut rng);
                    }
                    gene
                }).collect()
            })
            .collect();

        new_pop.append(&mut offspring);
        self.pop = new_pop;

        Ok(best_fitness)
    }
    pub fn save_base_path(&self, filename: &str) -> Result<(), Error> {
        Ok(write_gamepads_to_file(&self.base_path, &filename)?)
    }

    pub fn save_flow(&self, filename: &str) -> Result<(), Error> {
        let bytes = postcard::to_allocvec(&self.flow)?;
        let mut file = File::create(filename)?;
        file.write_all(&bytes)?;
        Ok(())
    }

}

