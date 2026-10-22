//! Replays an input file and prints Mario's state at a frame and at the end
//! (coins, stars, lives, health, position, level).
//!
//! cargo run --release --bin replay_info -- <rom> <m64> [frame]

use anyhow::{Result, bail};
use sm64_ai::util::read_gamepads_from_file;
use sm64_binds::{GameState, SM64GameGenerator};

fn show(label: &str, frame: usize, s: &GameState) {
    println!(
        "{label} (frame {frame}): coins {} stars {} lives {} health {:#x} pos [{:.0}, {:.0}, {:.0}] level {} area {}",
        s.coins,
        s.num_stars,
        s.lives,
        s.health,
        s.pos[0],
        s.pos[1],
        s.pos[2],
        s.level_num,
        s.area_index
    );
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        bail!("usage: replay_info <rom> <m64> [frame]");
    }
    let pads = read_gamepads_from_file(&args[1])?;
    let mark: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(0);
    let mut game = SM64GameGenerator::from_file(&args[0])?.create_game()?;
    for (i, pad) in pads.iter().enumerate() {
        if i == mark && mark > 0 {
            show("at", i, &game.get_game_state()?);
        }
        game.step_game(*pad)?;
    }
    show("end", pads.len(), &game.get_game_state()?);
    Ok(())
}
