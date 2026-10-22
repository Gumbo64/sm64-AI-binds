use std::fs::File;

use sm64_ai::util::read_gamepads_from_file;
use sm64_ai::{population::Population, util::skip_menu_pads};
use sm64_ai::scoring::ScoringStrategy;
use sm64_ai::curiosity_scoring::CuriosityScoring;
use anyhow::{Result, Error};
use tqdm::tqdm;
use std::io::Write;
fn main() -> Result<(), Error> {
    let pop_size = 128;
    let genome_length = 600;
    let step_amount = 1;

    println!("Initialising population");

    let config_str = format!("curiosity-pop{}-genlen{}-step{}", pop_size, genome_length, step_amount);
    let base_path_filename = format!("runs/{}.m64", config_str);
    let scoring_path = format!("runs-scoring/{}.scoring", config_str);


    let scoring = match CuriosityScoring::load(&scoring_path) {
        Ok(s) => {println!("Loaded Scoring."); s},
        Err(_) => {println!("New Scoring"); CuriosityScoring::default()},
    };

    let mut file = File::create("curiosity_output.csv")?;
    writeln!(file, "level,x,y,z,visits")?;
    for (key, count) in &scoring.bucket_counts {
        // key = (course, act, area, x, y, z)
        writeln!(file, "{},{},{},{},{}",key.0, key.1, key.2, key.3, count)?;
    }

    println!("{:}", scoring);

    Ok(())
}