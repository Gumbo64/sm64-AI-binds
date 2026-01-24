use std::fs::{self, File};
use std::path::Path;
use rand::prelude::*;
use rayon::prelude::*;
use byteorder::{BigEndian, WriteBytesExt};
use anyhow::{Result, Error};
use sm64_binds::{GamePad, SM64Game, SM64GameGenerator, GameState, buttons};
use tqdm::tqdm;

fn random_gamepad(rng: &mut ThreadRng) -> GamePad {
    let stick_x = rng.random_range(-128..=127);
    let stick_y = rng.random_range(-128..=127);

    let mut button = 0;
    if rng.random_bool(0.04) { button |= buttons::A_BUTTON; }
    if rng.random_bool(0.04) { button |= buttons::B_BUTTON; }
    if rng.random_bool(0.04) { button |= buttons::Z_TRIG; }
    return GamePad::new(button, stick_x, stick_y)
}

type Genome = Vec<GamePad>;

fn skip_menu_pads() -> Vec<GamePad> {
    let mut path = Vec::new();
    for i in 0..1680 {
        let mut button = 0;
        if (150..160).contains(&i) || (200..300).contains(&i) {
            button = buttons::START_BUTTON;
        }
        if i % 2 == 0 {
            button = buttons::A_BUTTON; 
        }
        let pad = GamePad::new(button, 0, 0);
        path.push(pad);
    }
    path
}

trait ScoringStrategy: Clone + Send + Sync {
    fn get_fitness(&self, tmp_states: &[GameState]) -> f32;
    fn add_state(&mut self, state: GameState);
}
#[derive(Debug, Clone)]
struct DefaultScoring;

impl ScoringStrategy for DefaultScoring {
    fn get_fitness(&self, tmp_states: &[GameState]) -> f32 {
        if tmp_states.is_empty() {
            return -f32::INFINITY;
        }

        // Equivalent to: scores = np.array([-state.pos[2] for state in tmp_states])
        // i = np.argmax(scores); return scores[i] - i
        let (best_idx, best_val) = tmp_states.iter().enumerate()
            .map(|(i, state)| (i, -state.pos[2]))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        best_val - (best_idx as f32)
    }

    fn add_state(&mut self, _state: GameState) {}
}

struct SM64ThreadState<S: ScoringStrategy> {
    scoring: S,
    base_game: SM64Game,
    tmp_game: SM64Game
}

impl<S: ScoringStrategy> SM64ThreadState<S> {
    fn new(skip_menu: bool, scoring: S) -> Result<Self, Error> {
        let game_gen = SM64GameGenerator::new(include_bytes!("../../baserom.us.z64").to_vec())?;
        let mut base_game = game_gen.create_game()?;
        if skip_menu {
            for pad in skip_menu_pads() {
                base_game.step_game(pad)?;
            }
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
        self.base_game.copy_to(&mut self.tmp_game)?;
                    
        let mut tmp_states = Vec::with_capacity(genome.len());
        for pad in genome {
            self.tmp_game.step_game(*pad)?;
            tmp_states.push(self.tmp_game.get_game_state()?);
        }

        Ok(self.scoring.get_fitness(&tmp_states))
    }
}


// --- POPULATION ---

struct Population<S: ScoringStrategy> {
    pop_size: usize,
    genome_length: usize,
    mutation_rate: f64,
    pop_elite: usize,
    
    pop: Vec<Genome>,
    
    // Wrapped in Arc so it can be shared across threads
    base_path: Vec<GamePad>,
    thread_states: Vec<SM64ThreadState<S>>
}

impl<S: ScoringStrategy> Population<S> {
    fn new(pop_size: usize, genome_length: usize, scoring: S, skip_menu: bool) -> Result<Self, Error> {
        println!("making temp games");

        let base_path: Vec<GamePad> = match skip_menu {
            true => skip_menu_pads(),
            false => Vec::new(),
        };

        let thread_count = rayon::current_num_threads();
        let thread_states: Result<Vec<SM64ThreadState<S>>, Error> = (0..thread_count)
            .into_par_iter()
            .map(|_| SM64ThreadState::new(skip_menu, scoring.clone()))
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
            base_path,
            thread_states,
        })
    }

    // fn base_step_single(&mut self, pad: GamePad) -> Result<(), Error> {
    //     self.base_game.step_game(pad)?;
    //     self.base_path.push(pad);
    //     let state = self.base_game.get_game_state()?;
    //     self.scoring.add_state(state);
    //     Ok(())
    // }

    fn base_step_n(&mut self, n: usize) -> Result<(), Error> {
        // Apply fittest genome steps (assuming sorted, index 0 is best)
        let fittest_genome = &self.pop[0].clone();

        let step_path = &fittest_genome[0..n];
        self.base_path.extend(fittest_genome);


        let results: Vec<Result<_, Error>> = self.thread_states
            .par_iter_mut()
            .map(|s| s.base_step(step_path)) // Collect results
            .collect();
        for result in results {
            result?; // This will return the first error encountered, if any
        }
    
        // Roll genomes: remove first n, add n randoms at end
        let mut rng = rand::rng();
        for genome in self.pop.iter_mut() {
            genome.drain(0..n);
            for _ in 0..n {
                genome.push(random_gamepad(&mut rng));
            }
        }
        Ok(())
    }

    fn iterate(&mut self) -> Result<f32, Error> {
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
}

// --- IO UTILS ---

fn write_gamepads_to_file(gamepads: &[GamePad], filename: &str) -> std::io::Result<()> {
    if let Some(parent) = Path::new(filename).parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(filename)?;
    for gp in gamepads {
        file.write_u16::<BigEndian>(gp.button)?;
        file.write_i8(gp.stick_x)?;
        file.write_i8(gp.stick_y)?;
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    let pop_size = 1000;
    let genome_length = 50;
    let step_amount = 10;

    println!("Initialising population");

    // Initialize population (loads ROM)
    let mut pop = Population::new(
        pop_size, 
        genome_length,
        DefaultScoring{},
        true // Python's main block doesn't seem to pass skip_menu=True, but calls base_step manually
    )?;

    println!("Starting loop...");
    for _i in tqdm(0..100) {
        let _best_fit = pop.iterate()?;
        // println!("Best Fitness: {:.4}", best_fit);

        pop.base_step_n(step_amount)?;

        let filename = format!("runs/pop{}-genlen{}-step{}.m64", pop_size, genome_length, step_amount);
        // if let Err(e) = write_gamepads_to_file(&pop.base_path, &filename) {
        //     eprintln!("Failed to write m64 file: {}", e);
        // }
    }
    Ok(())
}