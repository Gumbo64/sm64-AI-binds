//! Lab for fast exits after a star (`Tech::FastExit`, objectives/src/extra/
//! tech/fastexit.rs) and for C-button camera turns in the GA
//! (`OBRRT_GA_CBUTTONS`, search/src/ga.rs).
//!
//! `<run>` may also be a movie (.m64): `<node>` is then the frame to play to.
//!
//! fastexit_lab <rom> <run> <node> scan
//!   replay the run's tree to `node`; for every star in the level: the floor
//!   under it, and the nearest point (xz, at the star's height) over a death
//!   plane (SURFACE_DEATH_PLANE, check_death_barrier) or over no floor (OOB),
//!   with whether the star is under the death line (plane + 2048).
//! fastexit_lab <rom> <run> <node> cost <star#> [doom]
//!   teleport Mario onto star `star#` (as listed by scan) and play the
//!   search's tail (A on alternate frames) until he's in control in the
//!   castle: frames from the grab to control, with each action change.
//!   With `doom`, first one frame below the death line over the nearest death
//!   plane (queues WARP_OP_WARP_FLOOR, 20 frames), then the star: the
//!   death-fade exit.
//! fastexit_lab <rom> <run> <node> camyaw <frames> <button hex> <sx> <sy>
//!   hold one pad and print cam_yaw (gLakituState.yaw, St::cam_yaw), the
//!   game's camera->yaw and Mario's intendedYaw each frame: does St read the
//!   yaw Mario's next frame steers by?
//! fastexit_lab <rom> <run> <node> ga <kind> <trials> [seed]
//!   GA attempts of every generated objective of `kind` from the node
//!   (Registry.generate), e.g. Tech::FastExit; success / progress / frames.
use std::sync::Arc;

use anyhow::{Result, bail};
use obrrt_core::layout::*;
use obrrt_core::{GamePad, Generator, Objective, St, check_layout, read_state};
use obrrt_objectives::Registry;
use obrrt_search::ga::{Fitness, GaParams, Worker};
use obrrt_search::route::read_tree_bin;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use sm64_binds::{SM64Game, SM64GameGenerator};

const SURFACE_DEATH_PLANE: i16 = 0x0A;
#[allow(non_upper_case_globals)]
const MARIO_intendedYaw: usize = 36;
const LEVEL_CASTLE_INSIDE: u16 = 6;
const LEVEL_CASTLE_GROUNDS: u16 = 16;
const LEVEL_CASTLE_COURTYARD: u16 = 26;

fn replay(gg: &SM64GameGenerator, run: &str, node: usize) -> Result<(SM64Game, u32)> {
    if run.ends_with(".m64") {
        // A movie instead of a run: play its first `node` frames.
        let b = std::fs::read(run)?;
        let b = if b.starts_with(b"M64\x1a") { &b[0x400..] } else { &b[..] };
        let mut game = gg.create_game()?;
        for c in b.chunks_exact(4).take(node) {
            game.step_game(GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8))?;
        }
        check_layout(&game)?;
        return Ok((game, node as u32));
    }
    let records = read_tree_bin(&format!("{run}/tree.bin"))?;
    let mut chain = vec![node + 1];
    while let Some(&r) = chain.last() {
        if r == 0 {
            break;
        }
        let parent = records[r].0;
        chain.push(if parent >= 0 { parent as usize + 1 } else { 0 });
    }
    let mut game = gg.create_game()?;
    let mut frame = 0u32;
    for r in chain.iter().rev() {
        for p in &records[*r].1 {
            game.step_game(*p)?;
            frame += 1;
        }
    }
    check_layout(&game)?;
    Ok((game, frame))
}

fn write(g: &mut SM64Game, addr: usize, bytes: &[u8]) -> Result<()> {
    let (store, _, memory) = g.wasm_parts();
    memory.view(store).write(addr as u64, bytes)?;
    Ok(())
}

fn teleport(g: &mut SM64Game, p: [f32; 3]) -> Result<()> {
    let at = ADDR_gMarioStates as usize + MARIO_pos;
    let mut b = Vec::new();
    for v in p {
        b.extend_from_slice(&v.to_le_bytes());
    }
    write(g, at, &b)?;
    // Standing still, so the teleport lands where it was put.
    let mut z = Vec::new();
    for _ in 0..3 {
        z.extend_from_slice(&0f32.to_le_bytes());
    }
    write(g, ADDR_gMarioStates as usize + MARIO_vel, &z)?;
    write(g, ADDR_gMarioStates as usize + MARIO_forwardVel, &0f32.to_le_bytes())
}

fn stars(s: &St) -> Vec<obrrt_core::Obj> {
    s.objects()
        .iter()
        .filter(|o| o.interact_type == INTERACT_STAR_OR_KEY || matches!(o.bhv_name(), "bhvSpawnedStar" | "bhvSpawnedStarNoLevelExit" | "bhvStarSpawnCoordinates"))
        .cloned()
        .collect()
}

/// The nearest xz point (rings of 25 units out to 2000) at height `y` over a
/// floor matching `want`, with that floor's height and type.
fn nearest(g: &SM64Game, c: [f32; 3], want: impl Fn(Option<i16>) -> bool) -> Option<([f32; 3], f32, f32)> {
    g.with_memory(|m| {
        let mut best: Option<([f32; 3], f32, f32)> = None;
        for r in (0..=2000).step_by(25) {
            let n = if r == 0 { 1 } else { (r as f32 * std::f32::consts::TAU / 25.0).ceil() as usize };
            for k in 0..n {
                let a = k as f32 * std::f32::consts::TAU / n as f32;
                let (x, z) = (c[0] + r as f32 * a.sin(), c[2] + r as f32 * a.cos());
                let hit = obrrt_core::geo::floor_at(m, x, c[1], z);
                if want(hit.as_ref().map(|h| h.kind)) {
                    best = Some(([x, c[1], z], r as f32, hit.map_or(-11000.0, |h| h.y)));
                    return best;
                }
            }
        }
        best
    })
}

fn main() -> Result<()> {
    obrrt_objectives::moves::install_controller();
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        bail!("usage: fastexit_lab <rom> <run> <node> scan|cost|camyaw|ga ...");
    }
    let gg = SM64GameGenerator::from_file(&a[1])?;
    let node: usize = a[3].parse()?;
    let (mut game, frame) = replay(&gg, &a[2], node)?;
    let st = read_state(&game, frame, true);
    println!(
        "node {node}: frame {frame} level {} area {} pos {:.0?} lives {} stars {} action {:#x}",
        st.level, st.area, st.pos, st.lives, st.stars, st.action
    );
    match a[4].as_str() {
        "scan" => {
            for (i, o) in stars(&st).iter().enumerate() {
                let under = game.with_memory(|m| obrrt_core::geo::floor_at(m, o.pos[0], o.pos[1], o.pos[2]));
                let dp = nearest(&game, o.pos, |k| k == Some(SURFACE_DEATH_PLANE));
                let oob = nearest(&game, o.pos, |k| k.is_none());
                println!(
                    "star#{i} {} params {:#010x} at {:.0?} (dist {:.0}); floor under {}",
                    o.bhv_name(),
                    o.beh_params,
                    o.pos,
                    st.dist(o.pos),
                    under.map_or("none".into(), |h| format!("{:.0} type {:#x}", h.y, h.kind))
                );
                match dp {
                    Some((p, r, y)) => println!(
                        "    death plane {r:.0} away at ({:.0}, {:.0}), plane y {y:.0}, line {:.0}: star {} the line by {:.0}",
                        p[0],
                        p[2],
                        y + 2048.0,
                        if o.pos[1] < y + 2048.0 { "under" } else { "above" },
                        (o.pos[1] - (y + 2048.0)).abs()
                    ),
                    None => println!("    no death plane within 2000"),
                }
                match oob {
                    Some((p, r, _)) => println!("    no floor (OOB) {r:.0} away at ({:.0}, {:.0})", p[0], p[2]),
                    None => println!("    floor everywhere within 2000"),
                }
            }
        }
        "cost" => {
            let list = stars(&st);
            let i: usize = a[5].parse()?;
            let star = list.get(i).ok_or_else(|| anyhow::anyhow!("no star#{i}"))?.clone();
            let doom = a.get(6).is_some_and(|v| v == "doom");
            let mut f = frame;
            let step = |g: &mut SM64Game, p: GamePad, f: &mut u32| -> Result<St> {
                g.step_game(p)?;
                *f += 1;
                Ok(read_state(g, *f, false))
            };
            // Wait for control (a node can end mid-transition).
            for _ in 0..300 {
                if read_state(&game, f, false).controllable() {
                    break;
                }
                step(&mut game, GamePad::new(0, 0, 0), &mut f)?;
            }
            let frame = f;
            if doom {
                let Some((p, _, y)) = nearest(&game, star.pos, |k| k == Some(SURFACE_DEATH_PLANE)) else { bail!("no death plane near") };
                teleport(&mut game, [p[0], y + 2000.0, p[2]])?;
                let s = step(&mut game, GamePad::new(0, 0, 0), &mut f)?;
                println!("doom frame: pos {:.0?} floor {:.0} type {:#x} action {:#x}", s.pos, s.floor_y, s.floor_type, s.action);
            }
            teleport(&mut game, [star.pos[0], star.pos[1] - 40.0, star.pos[2]])?;
            let (mut grab, mut last_act, mut last_level) = (None, 0u32, st.level);
            let s0 = st.stars;
            for k in 0..1200u32 {
                let pad = GamePad::new(if k % 2 == 0 { obrrt_core::buttons::A_BUTTON } else { 0 }, 0, 0);
                let s = step(&mut game, pad, &mut f)?;
                if grab.is_none() && s.stars > s0 {
                    grab = Some(f);
                    println!("  grab at {f} (+{k}), action {:#x}", s.action);
                }
                if s.action != last_act || s.level != last_level {
                    println!("  {f} (+{}) level {} action {:#x} lives {} play_mode {} menu {}", f - frame, s.level, s.action, s.lives, s.play_mode, s.menu_mode);
                    last_act = s.action;
                    last_level = s.level;
                }
                let castle = matches!(s.level, LEVEL_CASTLE_INSIDE | LEVEL_CASTLE_GROUNDS | LEVEL_CASTLE_COURTYARD);
                if castle && s.controllable() {
                    let g = grab.unwrap_or(frame);
                    println!(
                        "{}: control in the castle at {f}: {} frames after the grab; lives {} -> {}, stars {} -> {}",
                        if doom { "death-fade exit" } else { "normal exit" },
                        f - g,
                        st.lives,
                        s.lives,
                        s0,
                        s.stars
                    );
                    break;
                }
            }
        }
        "camyaw" => {
            let n: usize = a[5].parse()?;
            let pad = GamePad::new(u16::from_str_radix(&a[6], 16)?, a[7].parse()?, a[8].parse()?);
            let mut f = frame;
            let mut prev = st.clone();
            for k in 0..n {
                game.step_game(pad)?;
                f += 1;
                let s = read_state(&game, f, false);
                let iy = game.with_memory(|m| {
                    let at = ADDR_gMarioStates as usize + MARIO_intendedYaw;
                    i16::from_le_bytes([m[at], m[at + 1]])
                });
                // intendedYaw = atan2s(-stickY, stickX) + camera->yaw: the
                // camera yaw Mario used this frame, from the stick.
                let stick = (pad.stick_x as f32).atan2(-(pad.stick_y as f32)); // atan2s(y=-sy, x=sx) in sm64: angle from +z
                let stick_s = (stick * 65536.0 / std::f32::consts::TAU) as i32;
                let used = (iy as i32 - stick_s) as i16;
                println!(
                    "{k:>3} pad {:04x} cam_yaw {:#06x} (d {:+5}) intendedYaw {:#06x} -> camera used {:#06x} (prev St cam_yaw {:#06x}, diff {:+})",
                    pad.button,
                    s.cam_yaw as u16,
                    s.cam_yaw.wrapping_sub(prev.cam_yaw),
                    iy as u16,
                    used as u16,
                    prev.cam_yaw as u16,
                    used.wrapping_sub(prev.cam_yaw)
                );
                prev = s;
            }
        }
        "ga" => {
            let kind = a[5].as_str();
            let trials: usize = a[6].parse()?;
            let seed: u64 = a.get(7).map_or(Ok(1), |s| s.parse())?;
            let save = game.save_state();
            let mut rng = SmallRng::seed_from_u64(seed);
            let objs: Vec<Arc<dyn Objective>> = Registry.generate(&st, &mut rng).into_iter().filter(|o| o.kind() == kind).collect();
            if objs.is_empty() {
                println!("no {kind} objective generated here");
            }
            let mut w = Worker::new(&gg)?;
            let params = GaParams { scale: 1.0, fitness: Fitness::Met, new_ops: true, pink: 0.0, per_objective_budget: false };
            for o in &objs {
                println!("objective {} base_criteria {} weight {:.1}", o.key(), o.base_criteria(&st), o.obj_weight(&st));
                let (mut ok, mut frames) = (0, 0u64);
                for t in 0..trials {
                    let att = w.attempt(&[o.as_ref()], &st, &save, &mut rng, params)?;
                    let seg = &att.segments[0];
                    frames += att.frames;
                    ok += seg.success as usize;
                    let end = seg.end.as_ref().map(|(s, _)| format!("lvl {} pos {:.0?} +{}f stars {} lives {}", s.level, s.pos, s.frame - st.frame, s.stars, s.lives));
                    println!("  trial {t}: success {} progress {:.0} frames {} end {}", seg.success, seg.progress, att.frames, end.unwrap_or("-".into()));
                }
                println!("{}: {ok}/{trials} successes, {frames} frames", o.key());
            }
        }
        m => bail!("mode {m}?"),
    }
    Ok(())
}
