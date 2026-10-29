//! Scratch lab (lakitu skip): traces Mario against the intro Lakitu's
//! trigger box, and measures `Tech::LakituSkip` attempt success rates.
//!
//! lakitu_lab <rom> <movie.m64|raw> trace <from> <to>
//!   replay the movie, printing frames from..to (pos, action, Lakitu act)
//! lakitu_lab <rom> <movie.m64|raw> ga <trials> [seed] [scale]
//!   play the movie to its end (no idle), then run `trials` GA attempts of
//!   every generated Tech::LakituSkip objective from there; prints each
//!   attempt's success, frames and progress.
//! lakitu_lab <rom> boot steer [x,z ...]
//!   boot (menu-skipping inputs, idle until still; the intro Lakitu still
//!   waiting), then for each start (the boot spot, or Mario teleported to
//!   x,z standing still facing the castle) and each generated
//!   Tech::LakituSkip mode, play the closed loop the GA's steered rollouts
//!   use (steer() + the mode's pad overlay) for up to 600 frames; prints
//!   frames to success (or how it failed), takeoff / strip landing points.
use std::sync::Arc;

use anyhow::Result;
use obrrt_core::{GamePad, Generator, Objective, read_state};
use obrrt_objectives::Registry;
use obrrt_search::ga::{Fitness, GaParams, Worker};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use sm64_binds::SM64GameGenerator;

fn read_movie(p: &str) -> Result<Vec<GamePad>> {
    let b = std::fs::read(p)?;
    let b = if b.starts_with(b"M64\x1a") { &b[0x400..] } else { &b[..] };
    Ok(b.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect())
}

fn boot_inputs() -> Vec<GamePad> {
    use obrrt_core::buttons::{A_BUTTON, START_BUTTON};
    (0..1680)
        .map(|i| {
            let mut button = 0;
            if (150..160).contains(&i) || (200..300).contains(&i) {
                button = START_BUTTON;
            }
            if i % 2 == 0 {
                button = A_BUTTON;
            }
            GamePad::new(button, 0, 0)
        })
        .collect()
}

const M: usize = obrrt_core::layout::ADDR_gMarioStates as usize;

fn write(g: &mut sm64_binds::SM64Game, addr: usize, bytes: &[u8]) -> Result<()> {
    let (store, _, memory) = g.wasm_parts();
    memory.view(store).write(addr as u64, bytes)?;
    Ok(())
}

fn steer_lab(gg: &SM64GameGenerator, starts: &[String]) -> Result<()> {
    use obrrt_core::St;
    let mut g = gg.create_game()?;
    let mut f = 0u32;
    for p in boot_inputs() {
        g.step_game(p)?;
        f += 1;
    }
    let mut st = read_state(&g, f, false);
    let mut still = 0;
    while still < 10 || !st.controllable() {
        let before = st.pos;
        g.step_game(GamePad::new(0, 0, 0))?;
        f += 1;
        st = read_state(&g, f, false);
        still = if st.pos == before { still + 1 } else { 0 };
    }
    let boot_save = g.save_state();
    let boot_f = f;
    let starts: Vec<String> = if starts.is_empty() { vec!["boot".into()] } else { starts.to_vec() };
    // The objectives (generated near the bridge: the registry offers them
    // within reach of the Lakitu), reused for every start.
    let mut objs: Vec<Arc<dyn Objective>> = Vec::new();
    for spec in std::iter::once(&"0,-100,0".to_string()).chain(starts.iter()) {
        let first = objs.is_empty();
        g.load_state(&boot_save)?;
        let mut f = boot_f;
        if let Some(path) = spec.strip_prefix("state:") {
            // An emulator snapshot (gunzipped `slope_lab savestate:`).
            g.load_state(&sm64_binds::SaveState::from_bytes(std::fs::read(path)?)?)?;
            f = 1648;
        } else if spec != "boot" {
            let v: Vec<f32> = spec.split(',').map(|t| t.parse().unwrap()).collect();
            let (x, z) = (v[0], v[1]);
            // Optional facing (degrees, 0 = +z, 180 = towards the castle).
            let yaw = v.get(2).map_or(0x8000u16 as i16, |d| ((d / 360.0 * 65536.0) as i32) as i16);
            let tris = g.with_memory(obrrt_core::geo::read_floors).0;
            let y = obrrt_core::geo::floor_below(&tris, x, z, 3000.0).unwrap_or(811.0);
            for _ in 0..4 {
                write(&mut g, M + 0x3C, &[x.to_le_bytes(), y.to_le_bytes(), z.to_le_bytes()].concat())?;
                write(&mut g, M + 0x48, &[0f32.to_le_bytes(); 3].concat())?;
                write(&mut g, M + 0x54, &0f32.to_le_bytes())?;
                write(&mut g, M + 0x2E, &yaw.to_le_bytes())?;
                write(&mut g, M + 0x0C, &0x0C40_0201u32.to_le_bytes())?;
                g.step_game(GamePad::new(0, 0, 0))?;
                f += 1;
            }
            for _ in 0..45 {
                g.step_game(GamePad::new(0, 0, 0))?;
                f += 1;
            }
        }
        let st0 = read_state(&g, f, true);
        let save = g.save_state();
        if first {
            let mut rng = SmallRng::seed_from_u64(1);
            objs = Registry.generate(&st0, &mut rng).into_iter().filter(|o| o.kind() == "Tech::LakituSkip").collect();
            continue;
        }
        println!("start {spec}: pos ({:.0}, {:.0}, {:.0}) frame {}", st0.pos[0], st0.pos[1], st0.pos[2], st0.frame);
        for o in &objs {
            if std::env::var("ONLY").is_ok_and(|v| !o.key().ends_with(&v)) {
                continue;
            }
            g.load_state(&save)?;
            let overlay = o.pad_overlay();
            let mut traj: Vec<St> = vec![st0.clone()];
            let mut played: Vec<GamePad> = Vec::new();
            let mut outcome = "timeout".to_string();
            let mut takeoff: Option<[f32; 3]> = None;
            let mut strip: Option<([f32; 3], usize)> = None;
            let mut caught_at: Option<[f32; 3]> = None;
            let mut path: Vec<String> = Vec::new();
            let trace = std::env::var("TRACE").is_ok_and(|v| o.key().ends_with(&v));
            for k in 0..600usize {
                let s = traj.last().unwrap();
                let pad = obrrt_search::steer::steer(o.as_ref(), s, k, 16);
                let pad = match &overlay {
                    Some(ov) => ov.pad(&traj, &played, pad),
                    None => pad,
                };
                g.step_game(pad)?;
                let n = read_state(&g, st0.frame + k as u32 + 1, false);
                if trace {
                    println!(
                        "    {k:>3} pad {:04x} {:>4} {:>4} pos {:>7.1} {:>6.1} {:>7.1} vy {:>5.1} fv {:>5.1} yaw {:#06x} act {:#010x} floor {:>6.1}",
                        pad.button, pad.stick_x, pad.stick_y, n.pos[0], n.pos[1], n.pos[2], n.vel[1], n.forward_vel, n.yaw as u16, n.action, n.floor_y
                    );
                }
                let prev_air = s.action & 0x800 != 0;
                if n.action == 0x0300_0888 && !prev_air && takeoff.is_none() {
                    takeoff = Some(s.pos);
                }
                if n.action == 0x0000_0479 && strip.is_none() && n.pos[2] < -177.0 && n.pos[2] > -2000.0 {
                    strip = Some((n.pos, k + 1));
                }
                let in_box = n.pos[0] > -544.0 && n.pos[0] < 545.0 && n.pos[1] > 800.0 && n.pos[2] > -2000.0 && n.pos[2] < -177.0;
                if in_box && n.action & 0x800 == 0 && caught_at.is_none() {
                    caught_at = Some(n.pos);
                }
                if k % 10 == 9 {
                    path.push(format!("{:.0},{:.0}", n.pos[0], n.pos[2]));
                }
                played.push(pad);
                traj.push(n);
                if o.success(&traj).is_some() {
                    // Walk on in (steer() at the door) until the castle loads.
                    let mut inside = None;
                    for j in 0..150usize {
                        let s = traj.last().unwrap();
                        if s.level == 6 {
                            inside = Some(k + 1 + j);
                            break;
                        }
                        let pad = obrrt_search::steer::steer(o.as_ref(), s, k + j, 16);
                        g.step_game(pad)?;
                        traj.push(read_state(&g, st0.frame + (k + 2 + j) as u32, false));
                    }
                    outcome = format!("SUCCESS {} frames (castle {})", k + 1, inside.map_or("-".into(), |f| f.to_string()));
                    break;
                }
                let n = traj.last().unwrap();
                if n.dialog == 34 || matches!(n.action, 0x0000_1308 | 0x2000_1306) {
                    outcome = format!("caught at frame {} ({:.0}, {:.0}, {:.0})", k + 1, n.pos[0], n.pos[1], n.pos[2]);
                    break;
                }
                if n.pos[1] < 700.0 && n.pos[2] < -177.0 {
                    outcome = format!("fell at frame {} ({:.0}, {:.0}, {:.0})", k + 1, n.pos[0], n.pos[1], n.pos[2]);
                    break;
                }
            }
            let fmt = |p: [f32; 3]| format!("({:.1}, {:.0}, {:.0})", p[0], p[1], p[2]);
            println!(
                "  {:<22} {outcome}; takeoff {} strip {} box-ground {}",
                o.key(),
                takeoff.map_or("-".into(), fmt),
                strip.map_or("-".into(), |(p, k)| format!("{} @{k}", fmt(p))),
                caught_at.map_or("-".into(), fmt)
            );
            if std::env::var("PATHS").is_ok() {
                println!("    path (x,z every 10 f): {}", path.join(" "));
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    // The steering controller (objectives/src/moves; move library off).
    obrrt_objectives::moves::install_controller();
    let a: Vec<String> = std::env::args().collect();
    let gg = SM64GameGenerator::from_file(&a[1])?;
    if a[2] == "boot" && a[3] == "walls" {
        // Static wall triangles with a vertex in x0..x1, z0..z1.
        let v: Vec<f32> = a[4..8].iter().map(|t| t.parse().unwrap()).collect();
        let mut g = gg.create_game()?;
        for p in boot_inputs() {
            g.step_game(p)?;
        }
        for _ in 0..60 {
            g.step_game(GamePad::new(0, 0, 0))?;
        }
        for t in g.with_memory(obrrt_core::geo::read_walls) {
            if t.v.iter().any(|p| p[0] >= v[0] && p[0] <= v[1] && p[2] >= v[2] && p[2] <= v[3]) {
                println!("wall {:?} n {:?}", t.v, t.n);
            }
        }
        return Ok(());
    }
    if a[2] == "boot" && a[3] == "steer" {
        return steer_lab(&gg, &a[4..]);
    }
    let pads = read_movie(&a[2])?;
    let mut game = gg.create_game()?;
    match a[3].as_str() {
        "trace" => {
            let (from, to): (usize, usize) = (a[4].parse()?, a[5].parse()?);
            for (i, p) in pads.iter().enumerate().take(to + 1) {
                game.step_game(*p)?;
                let f = i + 1;
                if f < from {
                    continue;
                }
                let s = read_state(&game, f as u32, true);
                let lak = s
                    .objects()
                    .iter()
                    .find(|o| o.bhv_name() == "bhvCameraLakitu")
                    .map(|o| o.action)
                    .unwrap_or(-1);
                let inbox = s.pos[0] > -544.0 && s.pos[0] < 545.0 && s.pos[1] > 800.0 && s.pos[2] > -2000.0 && s.pos[2] < -177.0;
                println!(
                    "f{f} pad {:04x} {:>4} {:>4} pos {:>7.1} {:>6.1} {:>7.1} v {:>6.1} fv {:>5.1} act {:#010x} floor {:>6.1} lak {lak} dlg {} {}",
                    p.button, p.stick_x, p.stick_y, s.pos[0], s.pos[1], s.pos[2], s.vel[1], s.forward_vel, s.action, s.floor_y, s.dialog,
                    if inbox { "BOX" } else { "" }
                );
            }
        }
        "ga" => {
            let trials: usize = a[4].parse()?;
            let seed: u64 = a.get(5).map_or(Ok(1), |s| s.parse())?;
            let scale: f64 = a.get(6).map_or(Ok(1.0), |s| s.parse())?;
            for p in &pads {
                game.step_game(*p)?;
            }
            let st = read_state(&game, pads.len() as u32, true);
            println!("start pos {:?} action {:#x} level {} frame {}", st.pos, st.action, st.level, st.frame);
            let save = game.save_state();
            let mut rng = SmallRng::seed_from_u64(seed);
            let objs: Vec<Arc<dyn Objective>> =
                Registry.generate(&st, &mut rng).into_iter().filter(|o| o.kind() == "Tech::LakituSkip").collect();
            let mut w = Worker { base: gg.create_game()?, tmp: gg.create_game()? };
            let params = GaParams { scale, fitness: Fitness::Met, new_ops: true, pink: 0.0, per_objective_budget: false };
            for o in &objs {
                println!("objective {} base_criteria {}", o.key(), o.base_criteria(&st));
                let (mut ok, mut frames, mut chained) = (0, 0u64, 0usize);
                for t in 0..trials {
                    let att = w.attempt(&[o.as_ref()], &st, &save, &mut rng, params)?;
                    let seg = &att.segments[0];
                    // CHAIN=n: up to n more attempts from each kept state
                    // nearer by base_heuristic (as the tree would).
                    let chain: usize = std::env::var("CHAIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                    if chain > 0 && !seg.success {
                        let mut cur = seg.end.clone();
                        let mut h = o.base_heuristic(&st);
                        let mut done = false;
                        for _ in 0..chain {
                            let Some((s2, sv2)) = cur.take() else { break };
                            if !o.base_criteria(&s2) || o.base_heuristic(&s2) >= h - 0.5 {
                                break;
                            }
                            h = o.base_heuristic(&s2);
                            let a2 = w.attempt(&[o.as_ref()], &s2, &sv2, &mut rng, params)?;
                            frames += a2.frames;
                            if a2.segments[0].success {
                                done = true;
                                break;
                            }
                            cur = a2.segments[0].end.clone();
                        }
                        chained += done as usize;
                    }
                    frames += att.frames;
                    ok += seg.success as usize;
                    let end = seg.end.as_ref().map(|(s, _)| (s.pos, s.frame - st.frame));
                    println!("  trial {t}: success {} progress {:.0} frames {} end {:?}", seg.success, seg.progress, att.frames, end);
                    if std::env::var("TRACE").is_ok_and(|v| v.parse() == Ok(t)) {
                        let mut g = gg.create_game()?;
                        g.load_state(&save)?;
                        for (k, p) in seg.inputs.iter().enumerate() {
                            g.step_game(*p)?;
                            let s = read_state(&g, st.frame + k as u32 + 1, false);
                            println!(
                                "    {k:>3} pad {:04x} {:>4} {:>4} pos {:>7.1} {:>6.1} {:>7.1} vy {:>5.1} fv {:>5.1} act {:#010x} floor {:>6.1} dlg {}",
                                p.button, p.stick_x, p.stick_y, s.pos[0], s.pos[1], s.pos[2], s.vel[1], s.forward_vel, s.action, s.floor_y, s.dialog
                            );
                        }
                    }
                }
                println!("{}: {ok}/{trials} successes, {} more by chaining, {frames} frames", o.key(), chained);
            }
        }
        m => anyhow::bail!("mode {m}?"),
    }
    Ok(())
}
