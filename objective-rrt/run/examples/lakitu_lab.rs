//! Scratch lab (lakitu skip): traces Mario against the intro Lakitu's
//! trigger box, and measures `Tech::LakituSkip` attempt success rates.
//!
//! lakitu_lab <rom> <movie.m64|raw> trace <from> <to>
//!   replay the movie, printing frames from..to (pos, action, Lakitu act)
//! lakitu_lab <rom> <movie.m64|raw> ga <trials> [seed] [scale]
//!   play the movie to its end (no idle), then run `trials` GA attempts of
//!   every generated Tech::LakituSkip objective from there; prints each
//!   attempt's success, frames and progress.
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

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let gg = SM64GameGenerator::from_file(&a[1])?;
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
