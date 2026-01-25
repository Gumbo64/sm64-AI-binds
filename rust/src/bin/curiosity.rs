use sm64_ai::population::Population;
use sm64_ai::scoring::CuriosityScoring;
use anyhow::{Result, Error};
use tqdm::tqdm;

fn main() -> Result<(), Error> {
    let pop_size = 128;
    let genome_length = 250;
    let step_amount = 1;

    println!("Initialising population");

    // Initialize population (loads ROM)
    let mut pop = Population::new(
        pop_size, 
        genome_length,
        CuriosityScoring::new(),
        true
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
            let filename = format!("runs/curiosity-pop{}-genlen{}-step{}.m64", pop_size, genome_length, step_amount);
            pop.write_base_path_to_file(&filename);
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