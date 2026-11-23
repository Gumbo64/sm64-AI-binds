//! clean_scan <rom> <out.tsv> <movie>...: frames of each movie where the `speed` problem has a start
//! (BoB, controllable, forward speed 60..110). One line per frame: movie frame fv action x y z grounded
use obrrt_core::{GamePad, read_state};
use obrrt_gp::problems::read_movie;
use sm64_binds::SM64GameGenerator;
use std::io::Write;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let gg = SM64GameGenerator::from_file(&a[1])?;
    let mut game = gg.create_game()?;
    let power_on = game.save_state();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[2])?);
    for m in &a[3..] {
        let pads: Vec<GamePad> = match read_movie(m) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("skip {m}: {e}");
                continue;
            }
        };
        game.load_state(&power_on)?;
        let mut n = 0;
        for (i, p) in pads.iter().enumerate() {
            game.step_game(*p)?;
            let s = read_state(&game, i as u32 + 1, false);
            if s.level as u32 == obrrt_core::layout::LEVEL_BOB as u32 && s.controllable() && s.forward_vel >= 60.0 && s.forward_vel <= 110.0 && s.pos[0] < 1190.0 {
                writeln!(out, "{m}\t{}\t{:.1}\t{:#x}\t{:.0}\t{:.0}\t{:.0}", i + 1, s.forward_vel, s.action, s.pos[0], s.pos[1], s.pos[2])?;
                n += 1;
            }
        }
        eprintln!("{m}: {} frames, {n} candidates", pads.len());
    }
    Ok(())
}
