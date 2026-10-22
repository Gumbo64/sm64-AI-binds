//! Prints where a replay changes level or area (sampled every N frames) and
//! Mario's state there, to pick search start points.
//!
//! cargo run --release --bin level_timeline -- <rom> <m64> [every=30]

use anyhow::{Result, bail};
use sm64_ai::util::read_gamepads_from_file;
use sm64_binds::SM64GameGenerator;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        bail!("usage: level_timeline <rom> <m64> [every]");
    }
    let pads = read_gamepads_from_file(&args[1])?;
    let every: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(30);
    let mut game = SM64GameGenerator::from_file(&args[0])?.create_game()?;
    let mut last = (i32::MIN, i32::MIN);
    for (i, pad) in pads.iter().enumerate() {
        game.step_game(*pad)?;
        if (i + 1) % every != 0 {
            continue;
        }
        let s = game.get_game_state()?;
        if (s.level_num, s.area_index) != last {
            println!(
                "frame {:>6}: level {:>2} area {} coins {:>3} stars {:>2} pos [{:.0}, {:.0}, {:.0}]",
                i + 1,
                s.level_num,
                s.area_index,
                s.coins,
                s.num_stars,
                s.pos[0],
                s.pos[1],
                s.pos[2]
            );
            last = (s.level_num, s.area_index);
        }
    }
    Ok(())
}
