//! Replays an .m64 and prints the final state and every object.
//!
//! cargo run --release --bin probe -- <rom> <file.m64> [frames]

use anyhow::Result;
use obrrt_core::{GamePad, check_layout, read_state};
use sm64_binds::SM64GameGenerator;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let game_gen = SM64GameGenerator::from_file(&args[1])?;
    obrrt_objectives::extra::missions::twostar::install();
    let mut game = game_gen.create_game()?;
    let bytes = std::fs::read(&args[2])?;
    let pads: Vec<GamePad> =
        bytes.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect();
    let n = args.get(3).map_or(pads.len(), |s| s.parse().unwrap());
    let two = obrrt_objectives::extra::missions::twostar::on();
    for (i, p) in pads[..n].iter().enumerate() {
        game.step_game(*p)?;
        // OBRRT_TWOSTAR: read every 30th frame so the zone's static floors get stored (geo::observe_floors).
        if two && i % 30 == 0 {
            let _ = read_state(&game, i as u32, false);
        }
    }
    check_layout(&game)?;
    let s = read_state(&game, n as u32, true);
    println!(
        "frame {n} level {} area {} act {} pos {:?} action {:#x} stars {} keys {} coins {} lives {} health {:#x} controllable {} water {}",
        s.level, s.area, s.act, s.pos, s.action, s.stars, s.keys, s.coins, s.lives, s.health, s.controllable(), s.water_level
    );
    if obrrt_objectives::extra::missions::twostar::on() {
        print!("{}", obrrt_objectives::extra::missions::twostar::debug_plan(&s));
    }
    for o in s.objects() {
        println!(
            "{:>3} {:<34} pos {:>7.0} {:>7.0} {:>7.0}  d {:>6.0}  interact {:#9x} act {:>3} hp {:>3} params {:#010x} value {} loot {} raw_1ac {:?}",
            o.slot, o.bhv_name(), o.pos[0], o.pos[1], o.pos[2], obrrt_core::dist(o.pos, s.pos), o.interact_type, o.action, o.health, o.beh_params, o.value, o.loot_coins, o.raw_1ac
        );
    }
    Ok(())
}
