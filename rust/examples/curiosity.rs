use sm64_ai::util::read_gamepads_from_file;
use sm64_ai::{population::Population, util::skip_menu_pads};

use sm64_ai::scoring::{ScoringStrategy, load_from_path};
use sm64_ai::curiosity_scoring::CuriosityScoring;
use anyhow::{Result, Error};
use tqdm::tqdm;

fn main() -> Result<(), Error> {
    let pop_size = 128;
    let genome_length = 600;
    let step_amount = 1;

    println!("Initialising population");

    let config_str = format!("curiosity-pop{}-genlen{}-step{}", pop_size, genome_length, step_amount);
    let base_path_filename = format!("runs/{}.m64", config_str);
    let scoring_path = format!("runs-scoring/{}.scoring", config_str);

    let initial_base_path = match read_gamepads_from_file(&base_path_filename) {
        Ok(path) => {println!("Loaded Path. Length: {}", path.len()); path},
        Err(_) => {println!("New Base Path"); skip_menu_pads()},
    };

    // let scoring = match CuriosityScoring::load(&scoring_path) {
    //     Ok(s) => {println!("Loaded Scoring."); s},
    //     Err(_) => {println!("New Scoring"); CuriosityScoring::new()},
    // };
    let scoring: CuriosityScoring = load_from_path(CuriosityScoring::default(), initial_base_path.clone())?;

    // Initialize population (loads ROM)
    let mut pop = Population::new(
        pop_size, 
        genome_length,
        scoring,
        initial_base_path
    )?;

    println!("Hot start time...");
    // let hot_start_time = genome_length / step_amount;
    let hot_start_time = 10;
    for _i in tqdm(0..hot_start_time) {
        pop.iterate()?;
    }

    println!("Starting loop...");
    let mut i = 0;
    loop {
        let _best_fit = pop.iterate()?;
        pop.base_step_n(step_amount)?;

        i = (i + 1) % 100;
        if i == 0 {
            if let Err(e) = pop.save_base_path(&base_path_filename) {
                println!("Base path saving failed {}", e)
            }
            if let Err(e) = pop.save_scoring(&scoring_path) {
                println!("Scoring saving failed {}", e)
            }
        }
    }

    // println!("Starting loop...");
    // for _i in tqdm(0..1000) {
    //     let _best_fit = pop.iterate()?;
    //     pop.base_step_n(step_amount)?;
    // }

    // let filename = format!("runs/pop{}-genlen{}-step{}.m64", pop_size, genome_length, step_amount);
    // pop.write_base_path_to_file(&filename);
    // Ok(())
}