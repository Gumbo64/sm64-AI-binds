use sm64_ai::distance_scoring::DistanceScoring;
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

    // init

    let pop = Population::new(pop_size, genome_length, DistanceScoring::default(), initial_base_path)?;
        

    // loop
    loop {
        // generate goal point. BucketKey esque, in zones that have been reached
        let goal_point = (0,0,0,0);
        // Get nearest visited bucket
        
        let nearest_key, nearest_node = (0,0,0,0); 
        
        // make/load nearest game
        let nearest_node_game = ???;

        let scoring = DistanceScoring::new(goal_point);
        pop.set_scoring(scoring);
        pop.set_base(nearest_node);
        pop.reset_pop();

        // Iterate
        for i in 0..100 {
            pop.iterate();
        }

    }
    let true_goal_point = (1,2,3,4);
    pop.save_path_to(true_goal_point);


}