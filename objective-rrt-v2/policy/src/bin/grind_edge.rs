//! grind_edge <rom> <out.tsv> <every> <movie>...: looks for edge grinds (an airborne/ground action alternation on a
//! ledge). At every `every`-th frame of each movie where Mario is controllable and on the ground, tries 12 constant stick
//! directions (30 degrees apart, from his heading) for 40 frames with no buttons and counts flips of the air flag.
//! Line: movie frame dir_idx flips yrange mean_fv holding start_action end_pos actions
use obrrt_core::{GamePad, read_state};
use obrrt_gp::problems::read_movie;
use obrrt_objectives::moves::arcs::stick;
use sm64_binds::SM64GameGenerator;
use std::io::Write;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let gg = SM64GameGenerator::from_file(&a[1])?;
    let mut game = gg.create_game()?;
    let power_on = game.save_state();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[2])?);
    let every: usize = a[3].parse()?;
    let only_hold = std::env::var("ONLY_HOLD").is_ok();
    // LEVELS="9,16": only states in these levels
    let levels: Vec<u16> = std::env::var("LEVELS").map(|l| l.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    // REPLAY="frame dir_idx": print the per-frame rows of that one try and stop (the movie is the first argument)
    let replay: Option<(usize, usize)> = std::env::var("REPLAY").ok().and_then(|r| {
        let v: Vec<usize> = r.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        (v.len() == 2).then(|| (v[0], v[1]))
    });
    // CTRL=file: lines "movie<TAB>frame<TAB>dir"; runs controller variants from each state (see below) and prints metrics
    let ctrl: Vec<(String, usize, usize)> = std::env::var("CTRL")
        .map(|f| {
            std::fs::read_to_string(f)
                .unwrap_or_default()
                .lines()
                .filter_map(|l| {
                    let v: Vec<&str> = l.split('\t').collect();
                    (v.len() >= 3).then(|| (v[0].to_string(), v[1].parse().unwrap_or(0), v[2].parse().unwrap_or(0)))
                })
                .collect()
        })
        .unwrap_or_default();
    for m in &a[4..] {
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
            if !ctrl.is_empty() {
                for (cm, cf, ck) in &ctrl {
                    if cm != m || *cf != i + 1 {
                        continue;
                    }
                    let snap = game.save_state();
                    let s0 = read_state(&game, 0, false);
                    for (vname, lean, dive_fv) in [("const", 0.0f32, -1.0f32), ("face", 0.0, -1.0), ("face+10", 10.0, -1.0), ("face-10", -10.0, -1.0), ("face+dive32", 0.0, 32.0), ("face+dive20", 0.0, 20.0)] {
                        game.load_state(&snap)?;
                        let mut flips = 0;
                        let mut prev_air = s0.action & 0x800 != 0;
                        let mut fvs = 0.0;
                        let mut last_b = false;
                        let mut last_a = false;
                        let mut acts = std::collections::BTreeMap::<u32, u32>::new();
                        let mut fv_end = 0.0;
                        let mut ymin = f32::MAX;
                        for f in 0..80 {
                            let st = read_state(&game, 0, false);
                            let yaw = if vname == "const" || f < 4 {
                                st.yaw.wrapping_add((*ck as i32 * 5461) as i16)
                            } else {
                                st.yaw.wrapping_add((lean * 182.04) as i16)
                            };
                            let (sx, sy) = stick(st.cam_yaw, yaw, 1.0);
                            let mut b = 0u16;
                            if dive_fv > 0.0 && f >= 4 && st.action == 0x0100088C && st.forward_vel >= dive_fv && !last_b {
                                b |= obrrt_core::buttons::B_BUTTON;
                            }
                            if st.action == 0x0088_0456 && !last_a && dive_fv > 0.0 {
                                b |= obrrt_core::buttons::A_BUTTON;
                            }
                            last_b = b & obrrt_core::buttons::B_BUTTON != 0;
                            last_a = b & obrrt_core::buttons::A_BUTTON != 0;
                            game.step_game(GamePad::new(b, sx, sy))?;
                            let t = read_state(&game, 0, false);
                            let air = t.action & 0x800 != 0;
                            if f > 0 && air != prev_air {
                                flips += 1;
                            }
                            prev_air = air;
                            fvs += t.forward_vel;
                            fv_end = t.forward_vel;
                            ymin = ymin.min(t.pos[1]);
                            *acts.entry(t.action).or_default() += 1;
                        }
                        let e = read_state(&game, 0, false);
                        let d = ((e.pos[0] - s0.pos[0]).powi(2) + (e.pos[2] - s0.pos[2]).powi(2)).sqrt();
                        let top: Vec<String> = { let mut v: Vec<_> = acts.iter().collect(); v.sort_by(|a, b| b.1.cmp(a.1)); v.iter().take(4).map(|(a, n)| format!("{a:#x}x{n}")).collect() };
                        println!("CTRL\t{cm}\t{cf}\t{ck}\tL{}\t{vname}\tdisp80 {d:.0}\tmean_fv {:.1}\tend_fv {fv_end:.1}\tflips {flips}\tdy {:.0}\t{}", s0.level, fvs / 80.0, e.pos[1] - ymin.min(e.pos[1]), top.join(","));
                    }
                    game.load_state(&snap)?;
                }
                continue;
            }
            if let Some((rf, rk)) = replay {
                if i + 1 == rf {
                    let s = read_state(&game, 0, false);
                    println!("start level {} area {} pos {:?} yaw {} cam {} act {:#x} fv {} holding {}", s.level, s.area, s.pos, s.yaw, s.cam_yaw, s.action, s.forward_vel, s.holding);
                    for f in 0..60 {
                        let st = read_state(&game, 0, false);
                        let yaw = st.yaw.wrapping_add((rk as i32 * 5461) as i16);
                        let (sx, sy) = stick(st.cam_yaw, yaw, 1.0);
                        game.step_game(GamePad::new(0, sx, sy))?;
                        let t = read_state(&game, 0, false);
                        println!("f {f:2} act {:#010x} fv {:6.2} vy {:6.1} pos ({:7.1},{:7.1},{:7.1}) yaw {} stick ({sx},{sy}) worldyaw {}", t.action, t.forward_vel, t.vel[1], t.pos[0], t.pos[1], t.pos[2], t.yaw, yaw);
                    }
                    return Ok(());
                }
                continue;
            }
            if i % every != 0 {
                continue;
            }
            let s = read_state(&game, i as u32 + 1, false);
            if (!levels.is_empty() && !levels.contains(&(s.level as u16))) || !s.controllable() || s.action & 0x800 != 0 || (only_hold && !s.holding) {
                continue;
            }
            let snap = game.save_state();
            for k in 0..12 {
                game.load_state(&snap)?;
                let mut flips = 0;
                let (mut ymin, mut ymax) = (f32::MAX, f32::MIN);
                let mut fvs = 0.0;
                let mut prev_air = s.action & 0x800 != 0;
                let mut acts: Vec<u32> = vec![];
                let mut last = s.clone();
                for f in 0..40 {
                    let st = read_state(&game, 0, false);
                    let yaw = st.yaw.wrapping_add((k as i32 * 5461) as i16);
                    let (sx, sy) = stick(st.cam_yaw, yaw, 1.0);
                    game.step_game(GamePad::new(0, sx, sy))?;
                    let t = read_state(&game, 0, false);
                    let air = t.action & 0x800 != 0;
                    if f > 0 && air != prev_air {
                        flips += 1;
                    }
                    prev_air = air;
                    ymin = ymin.min(t.pos[1]);
                    ymax = ymax.max(t.pos[1]);
                    fvs += t.forward_vel;
                    if acts.last() != Some(&t.action) {
                        acts.push(t.action);
                    }
                    last = t;
                }
                if flips >= 6 {
                    writeln!(
                        out,
                        "{m}\t{}\t{k}\t{flips}\t{:.0}\t{:.1}\t{}\t{:#x}\t({:.0},{:.0},{:.0})\tL{}\t{}",
                        i + 1,
                        ymax - ymin,
                        fvs / 40.0,
                        s.holding as u8,
                        s.action,
                        last.pos[0],
                        last.pos[1],
                        last.pos[2],
                        s.level,
                        acts.iter().take(8).map(|a| format!("{a:#x}")).collect::<Vec<_>>().join(",")
                    )?;
                    n += 1;
                }
            }
            game.load_state(&snap)?;
        }
        out.flush()?;
        eprintln!("{m}: {} frames, {n} hits", pads.len());
    }
    Ok(())
}
