use std::fs::File;
use std::io::Write;

use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use anyhow::{Result, Error};
use sm64_binds::{GamePad, SM64Game, SM64GameGenerator};

type Genome = Vec<GamePad>;

use crate::scoring::ScoringStrategy;
use crate::noise::{Slot, Variant, random_gamepad};
use crate::util::write_gamepads_to_file;

struct SM64ThreadState<S: ScoringStrategy> {
    scoring: S,
    base_game: SM64Game,
    tmp_game: SM64Game
}

impl<S: ScoringStrategy> SM64ThreadState<S> {
    fn new(game_gen: &SM64GameGenerator, initial_base_path: Vec<GamePad>, scoring: S) -> Result<Self, Error> {
        let mut base_game = game_gen.create_game()?;

        for pad in initial_base_path {
            base_game.step_game(pad)?;
        }
        
        let tmp_game = game_gen.create_game()?;
        Ok(Self { scoring, base_game, tmp_game })
    }

    fn base_step(&mut self, path: &[GamePad]) -> Result<(), Error> {
        for pad in path {
            self.base_game.step_game(*pad)?;
            let state = self.base_game.get_game_state()?;
            self.scoring.add_state(state);
        }
        Ok(())
    }

    fn get_fitness(&mut self, genome: &Genome) -> Result<f32, Error> {
        // Copies only the pages the previous genome wrote, while the base is
        // unchanged; a full copy otherwise.
        self.tmp_game.sync_from(&self.base_game)?;

        let mut tmp_states = Vec::with_capacity(genome.len());
        for pad in genome {
            self.tmp_game.step_game(*pad)?;
            tmp_states.push(self.tmp_game.get_game_state()?);
        }

        Ok(self.scoring.get_fitness(&tmp_states))
    }
}


// --- POPULATION ---

pub struct Population<S: ScoringStrategy> {
    pop_size: usize,
    genome_length: usize,
    mutation_rate: f64,
    pop_elite: usize,
    
    pop: Vec<Genome>,
    
    pub base_path: Vec<GamePad>,
    thread_states: Vec<SM64ThreadState<S>>,
    rng: SmallRng,
    variant: Variant,
    slots: Vec<Slot>,
}

impl<S: ScoringStrategy> Population<S> {
    pub fn new(pop_size: usize, genome_length: usize, scoring: S, initial_base_path: Vec<GamePad>) -> Result<Self, Error> {
        let game_gen = SM64GameGenerator::new(crate::util::rom_bytes()?)?;
        let white = Variant::named("white").unwrap();
        Self::with_options(pop_size, genome_length, scoring, initial_base_path, &game_gen, white, rand::rng().random())
    }

    /// `new` with a game generator, the inputs' noise (`crate::noise`) and a
    /// seed: the same seed gives the same run.
    pub fn with_options(pop_size: usize, genome_length: usize, scoring: S, initial_base_path: Vec<GamePad>,
                        game_gen: &SM64GameGenerator, variant: Variant, seed: u64) -> Result<Self, Error> {
        println!("making temp games");

        let thread_count = rayon::current_num_threads();
        let thread_states: Result<Vec<SM64ThreadState<S>>, Error> = (0..thread_count)
            .into_par_iter()
            .map(|_| SM64ThreadState::new(game_gen, initial_base_path.clone(), scoring.clone()))
            .collect();
        let thread_states = thread_states?;


        let mut rng = SmallRng::seed_from_u64(seed);
        let mut slots: Vec<Slot> = (0..pop_size).map(|_| Slot::new(&mut rng, variant)).collect();
        let pop = slots.iter_mut()
            .map(|slot| (0..genome_length).map(|_| slot.next(&mut rng)).collect())
            .collect();

        Ok(Self {
            pop_size,
            genome_length,
            mutation_rate: 0.05,
            pop_elite: (pop_size / 5) + 1,
            pop,
            base_path: initial_base_path,
            thread_states,
            rng,
            variant,
            slots,
        })
    }

    // fn base_step_single(&mut self, pad: GamePad) -> Result<(), Error> {
    //     self.base_game.step_game(pad)?;
    //     self.base_path.push(pad);
    //     let state = self.base_game.get_game_state()?;
    //     self.scoring.add_state(state);
    //     Ok(())
    // }

    pub fn base_step_n(&mut self, n: usize) -> Result<(), Error> {
        // Apply fittest genome steps (assuming sorted, index 0 is best)
        let fittest_genome = &self.pop[0].clone();

        let step_path = &fittest_genome[0..n];
        self.base_path.extend(step_path);

        let results: Vec<Result<_, Error>> = self.thread_states
            .par_iter_mut()
            .map(|s| s.base_step(step_path)) // Collect results
            .collect();
        for result in results {
            result?; // This will return the first error encountered, if any
        }
    
        // Roll genomes: remove first n, add n randoms at end
        for (genome, slot) in self.pop.iter_mut().zip(&mut self.slots) {
            genome.drain(0..n);
            for _ in 0..n {
                genome.push(slot.next(&mut self.rng));
            }
        }
        Ok(())
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

        // One seed per child, drawn in order, so a run doesn't depend on threads.
        let seeds: Vec<u64> = (0..offspring_needed).map(|_| self.rng.random()).collect();
        let mut offspring: Vec<Genome> = seeds
            .into_par_iter()
            .map(|seed| {
                let mut rng = SmallRng::seed_from_u64(seed);
                let idx1 = rng.random_range(0..self.pop_size);
                let idx2 = rng.random_range(0..self.pop_size);
                
                let parent1 = &self.pop[idx1];
                let parent2 = &self.pop[idx2];
                
                // Use map to generate the genome in parallel if genome_length is huge, 
                // but usually a standard loop here is faster due to lower overhead.
                // Block variants copy runs of about 30 frames from one parent.
                let mut r: f64 = rng.random();
                (0..self.genome_length).map(|j| {
                    if !self.variant.block || j == 0 || rng.random_bool(1.0 / 30.0) {
                        r = rng.random();
                    }
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

    pub fn scoring(&self) -> &S {
        &self.thread_states[0].scoring
    }

    pub fn save_scoring(&self, filename: &str) -> Result<(), Error> {
        let bytes = postcard::to_allocvec(&self.thread_states[0].scoring)?;
        let mut file = File::create(filename)?;
        file.write_all(&bytes)?;
        Ok(())
    }

}

