//! Replays an .m64 and prints the final state and every object.
//!
//! cargo run --release --bin probe -- <rom> <file.m64> [frames]

use anyhow::Result;
use obrrt_core::{GamePad, check_layout, read_state};
use sm64_binds::SM64GameGenerator;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let game_gen = SM64GameGenerator::from_file(&args[1])?;
    let mut game = game_gen.create_game()?;
    let bytes = std::fs::read(&args[2])?;
    let pads: Vec<GamePad> =
        bytes.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect();
    let n = args.get(3).map_or(pads.len(), |s| s.parse().unwrap());
    for p in &pads[..n] {
        game.step_game(*p)?;
    }
    check_layout(&game)?;
    let s = read_state(&game, n as u32, true);
    println!(
        "frame {n} level {} area {} act {} pos {:?} action {:#x} stars {} keys {} coins {} lives {} health {:#x} controllable {}",
        s.level, s.area, s.act, s.pos, s.action, s.stars, s.keys, s.coins, s.lives, s.health, s.controllable()
    );
    for o in s.objects() {
        println!(
            "{:>3} {:<34} pos {:>7.0} {:>7.0} {:>7.0}  d {:>6.0}  interact {:#9x} act {:>3} hp {:>3} params {:#010x} raw_1ac {:?}",
            o.slot, o.bhv_name(), o.pos[0], o.pos[1], o.pos[2], obrrt_core::dist(o.pos, s.pos), o.interact_type, o.action, o.health, o.beh_params, o.raw_1ac
        );
    }
    Ok(())
}
