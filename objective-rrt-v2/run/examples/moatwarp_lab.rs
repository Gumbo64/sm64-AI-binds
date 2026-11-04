//! Lab for the 2026 120-star TAS's opening: the moat-bank BLJ that walks
//! into the Vanish Cap Under the Moat warp (ref/tas/120_first2_phases.txt,
//! frames 1756-1852; the warp is castle_grounds/script.c's bhvWarp at
//! (-3379, -815, -2025), behaviour byte 0x3C = radius 600, height 50, behind
//! the moat grills).
//!
//! moatwarp_lab <rom> <movie.m64> tas
//!   replay the movie to f1856 and print f1800-1856 (must reach VCutM, level 18)
//! moatwarp_lab <rom> <movie.m64> walkgrid <F> [step]
//!   from the TAS state after frame F, one frame with stick (x, y) on a grid
//!   (no buttons), then the TAS's own next inputs for 30 frames: how many
//!   sticks warp. F = 1826 is the TAS's (its walking frame is 1827).
//! moatwarp_lab <rom> <movie.m64> chain <F> <buffer 0|1> [cycles]
//!   from the TAS state after frame F (mid-chain, F >= 1787), drive the BLJ
//!   with a generic policy (press A+Z with the stick back on a landing whose
//!   previous frame had A up; otherwise Z, stick neutral), optionally
//!   pause-buffered (press frame + START, then Z / Z+START / Z), for up to
//!   `cycles` presses; after every landing with |fv| > 800 try a 16-step
//!   walkgrid (5 wait frames, as the TAS) and report hits.
//! moatwarp_lab <rom> <movie.m64> start <trials> [seed]
//!   from the TAS's bank stop (after f1759), random setups: 0-8 frames of
//!   walking toward a random yaw near the TAS's, crouch-slide long jump with
//!   the stick back from a random frame, then the generic chain (unbuffered)
//!   and the final walkgrid. Reports how many reach |fv| > 2000, how many
//!   stall, and how many warp.
//! moatwarp_lab <rom> <movie.m64> pause
//!   the pause buffer's frame pattern: plays the TAS to f1803, then each
//!   candidate buffer pattern, and prints play mode / A presses / speed.
//! moatwarp_lab <rom> <movie.m64> overlay <F> <buffer 0|1> <stop> [x,y,z yaw] [frames]
//!   the objectives' own BLJ overlay (blj::overlay_exit, free exit, genome
//!   stick neutral; with buffer 1 wrapped in pausebuf::wrap) from the TAS
//!   state after frame F (F may be anywhere in the movie, e.g. the JRB dock
//!   BLJ at f76660); start/yaw default to Mario's position and the downhill
//!   of his floor. Prints the chain and its peak speed, real frames used and
//!   game (unpaused) frames, and whether it warped.
//! (mwoverlay / mwmc: the same with Tech::MoatWarpBlj's own overlay,
//! moatwarp::overlay; its buffering follows OBRRT_PAUSE_BUF; mwmc's 9th
//! argument is the side offset from downhill, e.g. -0x2000.)
//! moatwarp_lab <rom> <movie.m64> mc <F> <buffer 0|1> <stop> <trials> [seed]
//!   Monte Carlo of the GA's first generation: the `overlay` run with random
//!   held-stick genomes (a new random stick every 1-10 frames, as the GA's
//!   random segments), 140 frames each. Counts chains past |fv| 1000 / 2000,
//!   stalls (|fv| > 1000 with no movement over a landing) and warps.
use anyhow::{Result, bail};
use obrrt_core::{GamePad, St, read_state};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use sm64_binds::{SM64Game, SM64GameGenerator, SaveState};

const LEVEL_VCUTM: u16 = 18;
const ACT_DISAPPEARED: u32 = 0x0000_1300;
const ACT_LONG_JUMP: u32 = 0x0300_0888;
const ACT_LONG_JUMP_LAND: u32 = 0x0000_0479;
const A: u16 = 0x8000;
const Z: u16 = 0x2000;
const START: u16 = 0x1000;
const L: u16 = 0x0020;
const WARP: [f32; 3] = [-3379.0, -815.0, -2025.0];

fn pads_of(path: &str) -> Result<Vec<GamePad>> {
    let bytes = std::fs::read(path)?;
    let body = if bytes.len() >= 0x400 && &bytes[..4] == b"M64\x1a" {
        let n = u32::from_le_bytes(bytes[0x18..0x1C].try_into().unwrap()) as usize;
        let d = &bytes[0x400..];
        d[..(n * 4).min(d.len() / 4 * 4)].to_vec()
    } else {
        bytes[..bytes.len() / 4 * 4].to_vec()
    };
    Ok(body.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect())
}

struct G {
    game: SM64Game,
    frame: u32,
    prev: u16,
    st: St,
}

impl G {
    fn step(&mut self, p: GamePad) -> Result<&St> {
        self.game.step_game(p)?;
        self.frame += 1;
        self.prev = p.button;
        self.st = read_state(&self.game, self.frame, false);
        Ok(&self.st)
    }
    fn snap(&self) -> (SaveState, u32, u16, St) {
        (self.game.save_state(), self.frame, self.prev, self.st.clone())
    }
    fn restore(&mut self, s: &(SaveState, u32, u16, St)) -> Result<()> {
        self.game.load_state(&s.0)?;
        self.frame = s.1;
        self.prev = s.2;
        self.st = s.3.clone();
        Ok(())
    }
}

fn warped(s: &St) -> bool {
    s.level == LEVEL_VCUTM || (s.level == 16 && s.action == ACT_DISAPPEARED && hd(s.pos, WARP) < 700.0)
}

fn hd(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn line(f: u32, p: GamePad, s: &St) {
    println!(
        "{:5} {:#06x} {:4} {:4} | {:8.1} {:7.1} {:8.1} fv {:9.2} act {:#010x} lvl {:2} pm {} yaw {:#06x} cam {:#06x}",
        f, p.button, p.stick_x, p.stick_y, s.pos[0], s.pos[1], s.pos[2], s.forward_vel, s.action, s.level, s.play_mode,
        s.yaw as u16, s.cam_yaw as u16
    );
}

/// Plays `pads[from..]` until a warp or `n` frames.
fn tail(g: &mut G, pads: &[GamePad], from: usize, n: usize) -> Result<bool> {
    for k in 0..n {
        let p = pads.get(from + k).copied().unwrap_or(GamePad::new(0, 0, 0));
        if warped(g.step(p)?) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// From the snapshot: `wait` neutral Z frames, one walking frame with each
/// stick of the grid, then 30 neutral frames. Returns (hits, tried, a hit).
fn walkgrid(g: &mut G, base: &(SaveState, u32, u16, St), wait: usize, step: i32) -> Result<(usize, usize, Option<(i8, i8)>)> {
    let (mut hits, mut tried, mut one) = (0, 0, None);
    let mut x = -128;
    while x <= 127 {
        let mut y = -128;
        while y <= 127 {
            g.restore(base)?;
            for _ in 0..wait {
                g.step(GamePad::new(Z, 0, 0))?;
            }
            g.step(GamePad::new(0, x as i8, y as i8))?;
            let mut ok = warped(&g.st);
            for k in 0..30 {
                if ok {
                    break;
                }
                // The TAS presses A on the frame after the walk (f1828).
                let b = if k == 0 { A } else { 0 };
                ok = warped(g.step(GamePad::new(b, 0, 0))?);
            }
            tried += 1;
            if ok {
                hits += 1;
                one.get_or_insert((x as i8, y as i8));
            }
            y += step;
        }
        x += step;
    }
    g.restore(base)?;
    Ok((hits, tried, one))
}

/// The stick (raw x, y) that points world yaw `yaw` at full tilt, given the
/// camera yaw (intendedYaw = atan2s(-y, x) + camera yaw).
fn stick_to(s: &St, yaw: i32) -> (i8, i8) {
    let rel = (yaw - s.cam_yaw as i32) as f32 * std::f32::consts::TAU / 65536.0;
    ((127.0 * rel.sin()).round() as i8, (-127.0 * rel.cos()).round() as i8)
}

/// One step of the generic BLJ chain policy. Returns the pads to play.
fn blj_pads(g: &G, buffer: bool) -> Vec<GamePad> {
    let s = &g.st;
    let back = stick_to(s, s.yaw as i32 + 0x8000);
    if s.action == ACT_LONG_JUMP_LAND && g.prev & A == 0 {
        if buffer {
            vec![GamePad::new(A | Z | START, back.0, back.1), GamePad::new(Z | L, 0, 0), GamePad::new(Z | START | L, 0, 0), GamePad::new(Z, 0, 0)]
        } else {
            vec![GamePad::new(A | Z, back.0, back.1)]
        }
    } else if s.action == ACT_LONG_JUMP {
        vec![GamePad::new(Z, back.0, back.1)]
    } else {
        vec![GamePad::new(Z, 0, 0)]
    }
}

/// Runs the chain policy for up to `presses` presses, trying the walkgrid on
/// landings past 800. Returns (best |fv|, stalled frames, warp hits, a hit).
fn chain(g: &mut G, buffer: bool, presses: usize, grid_step: i32, verbose: bool) -> Result<(f32, usize, usize, Option<(u32, i8, i8)>)> {
    let (mut best, mut stalled, mut hits, mut one) = (0f32, 0usize, 0usize, None);
    let mut n = 0;
    let mut tried_fv: Vec<f32> = Vec::new();
    for _ in 0..400 {
        if n >= presses || !g.st.alive() || g.st.level != 16 {
            break;
        }
        let pads = blj_pads(g, buffer);
        let pos0 = g.st.pos;
        for p in &pads {
            if p.button & A != 0 {
                n += 1;
            }
            g.step(*p)?;
            if verbose {
                line(g.frame, *p, &g.st);
            }
        }
        let s = g.st.clone();
        if hd(pos0, s.pos) < 1.0 && s.forward_vel < -100.0 {
            stalled += 1;
        }
        if !matches!(s.action, ACT_LONG_JUMP | ACT_LONG_JUMP_LAND) {
            break;
        }
        best = best.max(-s.forward_vel);
        if s.action == ACT_LONG_JUMP_LAND && s.forward_vel < -800.0 && !tried_fv.iter().any(|v| (v - s.forward_vel).abs() < 1.0) {
            tried_fv.push(s.forward_vel);
            let base = g.snap();
            let (h, t, o) = walkgrid(g, &base, 5, grid_step)?;
            if verbose || h > 0 {
                println!(
                    "  walkgrid at f{} pos ({:.0},{:.0},{:.0}) fv {:.0}: {}/{} hit{}",
                    base.1, s.pos[0], s.pos[1], s.pos[2], s.forward_vel, h, t,
                    o.map_or(String::new(), |o| format!(" e.g. stick {:?}", o))
                );
            }
            if h > 0 {
                hits += h;
                one.get_or_insert((base.1, o.unwrap().0, o.unwrap().1));
            }
        }
    }
    Ok((best, stalled, hits, one))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        bail!("usage: moatwarp_lab <rom> <movie.m64> tas|walkgrid|chain|start|pause ...");
    }
    let pads = pads_of(&args[2])?;
    let gg = SM64GameGenerator::from_file(&args[1])?;
    let mut game = gg.create_game()?;
    let mw = args[3].starts_with("mw");
    let boot = if args[3].ends_with("overlay") { args[4].parse::<usize>()?.min(1755) } else { 1755usize };
    for p in &pads[..boot] {
        game.step_game(*p)?;
    }
    let st = read_state(&game, boot as u32, false);
    let mut g = G { game, frame: boot as u32, prev: pads[boot - 1].button, st };
    let to = |g: &mut G, f: u32| -> Result<()> {
        while g.frame < f {
            let p = pads[g.frame as usize];
            g.step(p)?;
        }
        Ok(())
    };
    match args[3].as_str() {
        "tas" => {
            while g.frame < 1856 {
                let p = pads[g.frame as usize];
                g.step(p)?;
                if g.frame >= 1756 {
                    line(g.frame, p, &g.st);
                }
            }
            println!("reached VCutM: {}", g.st.level == LEVEL_VCUTM);
        }
        "walkgrid" => {
            let f: u32 = args[4].parse()?;
            let step: i32 = args.get(5).map_or(Ok(8), |s| s.parse())?;
            to(&mut g, f)?;
            let base = g.snap();
            let (mut hits, mut tried) = (0, 0);
            let mut x = -128;
            let mut rows = Vec::new();
            while x <= 127 {
                let mut y = -128;
                let mut row = String::new();
                while y <= 127 {
                    g.restore(&base)?;
                    g.step(GamePad::new(0, x as i8, y as i8))?;
                    let ok = warped(&g.st) || tail(&mut g, &pads, f as usize + 1, 30)?;
                    tried += 1;
                    hits += ok as usize;
                    row.push(if ok { '#' } else { '.' });
                    y += step;
                }
                rows.push(format!("x {:4} {}", x, row));
                x += step;
            }
            for r in rows {
                println!("{r}");
            }
            println!("walkgrid after f{f}: {hits}/{tried} sticks warp (columns: y from -128 by {step})");
        }
        "chain" => {
            let f: u32 = args[4].parse()?;
            let buffer = args[5] == "1";
            let presses: usize = args.get(6).map_or(Ok(30), |s| s.parse())?;
            to(&mut g, f)?;
            let t0 = g.frame;
            let (best, stalled, hits, one) = chain(&mut g, buffer, presses, 4, true)?;
            println!(
                "chain from f{t0} buffer {buffer}: best |fv| {best:.0}, stalled cycles {stalled}, walkgrid hits {hits}, first {:?}",
                one
            );
        }
        "start" => {
            let trials: usize = args[4].parse()?;
            let seed: u64 = args.get(5).map_or(Ok(1), |s| s.parse())?;
            let mut rng = SmallRng::seed_from_u64(seed);
            to(&mut g, 1759)?;
            let base = g.snap();
            let tas_yaw = 0x2400i32; // overwritten below from the TAS's own walk frame
            let _ = tas_yaw;
            // The TAS's facing on its crouch slide (f1761).
            let mut probe = G { game: gg.create_game()?, frame: 0, prev: 0, st: g.st.clone() };
            probe.game.load_state(&base.0)?;
            probe.frame = base.1;
            probe.prev = base.2;
            for f in 1759..1761 {
                probe.step(pads[f])?;
            }
            let yaw0 = probe.st.yaw as i32;
            println!("TAS crouch-slide facing {:#06x} at ({:.0},{:.0},{:.0})", yaw0 as u16, probe.st.pos[0], probe.st.pos[1], probe.st.pos[2]);
            let (mut fast, mut stall_n, mut warp_n) = (0, 0, 0);
            for t in 0..trials {
                g.restore(&base)?;
                let yaw = yaw0 + rng.random_range(-0x1800..0x1800);
                let walk = rng.random_range(1..8);
                for _ in 0..walk {
                    let (x, y) = stick_to(&g.st, yaw);
                    g.step(GamePad::new(0, x, y))?;
                }
                // Crouch slide, long jump, stick back from a random frame.
                let (x, y) = stick_to(&g.st, yaw);
                g.step(GamePad::new(Z, x, y))?;
                g.step(GamePad::new(A | Z, x, y))?;
                let fwd = rng.random_range(0..4);
                for _ in 0..fwd {
                    g.step(GamePad::new(A | Z, x, y))?;
                }
                for _ in 0..40 {
                    if g.st.action != ACT_LONG_JUMP {
                        break;
                    }
                    let b = stick_to(&g.st, g.st.yaw as i32 + 0x8000);
                    g.step(GamePad::new(Z, b.0, b.1))?;
                }
                let (best, stalled, hits, one) = chain(&mut g, false, 24, 32, false)?;
                fast += (best > 2000.0) as usize;
                stall_n += (stalled > 0) as usize;
                warp_n += (hits > 0) as usize;
                println!(
                    "trial {t:3} yaw {:#06x} walk {walk} fwd {fwd}: best |fv| {best:7.0} stalled {stalled:2} hits {hits:3} {:?} end ({:.0},{:.0},{:.0})",
                    yaw as u16, one, g.st.pos[0], g.st.pos[1], g.st.pos[2]
                );
            }
            println!("start: {trials} trials, |fv|>2000 {fast}, stalled {stall_n}, warped {warp_n}");
        }
        "pause" => {
            to(&mut g, 1803)?;
            let base = g.snap();
            let pats: Vec<(&str, Vec<GamePad>)> = vec![
                ("tas: AZS, ZL, ZSL, Z, AZ", vec![GamePad::new(A | Z | START, 0, 0), GamePad::new(Z | L, 0, 0), GamePad::new(Z | START | L, 0, 0), GamePad::new(Z, 0, 0), GamePad::new(A | Z, 0, 0)]),
                ("no L: AZS, Z, ZS, Z, AZ", vec![GamePad::new(A | Z | START, 0, 0), GamePad::new(Z, 0, 0), GamePad::new(Z | START, 0, 0), GamePad::new(Z, 0, 0), GamePad::new(A | Z, 0, 0)]),
                ("short: AZS, ZS, AZ", vec![GamePad::new(A | Z | START, 0, 0), GamePad::new(Z | START, 0, 0), GamePad::new(A | Z, 0, 0)]),
                ("short2: AZS, Z, ZS, AZ", vec![GamePad::new(A | Z | START, 0, 0), GamePad::new(Z, 0, 0), GamePad::new(Z | START, 0, 0), GamePad::new(A | Z, 0, 0)]),
                ("A held: AZS, AZ, AZS, AZ, AZ", vec![GamePad::new(A | Z | START, 0, 0), GamePad::new(A | Z, 0, 0), GamePad::new(A | Z | START, 0, 0), GamePad::new(A | Z, 0, 0), GamePad::new(A | Z, 0, 0)]),
                ("unbuffered: AZ, Z, AZ", vec![GamePad::new(A | Z, 0, 0), GamePad::new(Z, 0, 0), GamePad::new(A | Z, 0, 0)]),
            ];
            for (name, ps) in pats {
                g.restore(&base)?;
                println!("-- {name}");
                for p in ps {
                    g.step(p)?;
                    line(g.frame, p, &g.st);
                }
            }
        }
        "overlay" | "mwoverlay" => {
            let f: u32 = args[4].parse()?;
            let buffer = args[5] == "1";
            let stop: f32 = args[6].parse()?;
            to(&mut g, f)?;
            let s0 = g.st.clone();
            let down = (s0.floor_normal[0].atan2(s0.floor_normal[2]) * 32768.0 / std::f32::consts::PI) as i32;
            let (start, yaw) = match args.get(7) {
                Some(p) if p.contains(',') => {
                    let v: Vec<f32> = p.split(',').map(|x| x.parse().unwrap()).collect();
                    (([v[0], v[1], v[2]]), args.get(8).map_or(down, |y| i32::from_str_radix(y.trim_start_matches("0x"), 16).unwrap()))
                }
                _ => (s0.pos, down),
            };
            let frames: usize = args.iter().rev().next().and_then(|a| a.parse().ok()).filter(|_| args.len() > 7).unwrap_or(120);
            let base = if mw { obrrt_objectives::extra::tech::moatwarp::overlay(yaw, stop) } else { obrrt_objectives::extra::tech::blj::overlay_exit(start, yaw, 0.0, None, stop, true) };
            let ov = if buffer && !mw { obrrt_objectives::extra::tech::pausebuf::wrap(base) } else { base };
            let mut traj = vec![s0.clone()];
            let mut played: Vec<GamePad> = Vec::new();
            let (mut peak, mut game_frames, mut warp_at) = (0f32, 0usize, None);
            println!("overlay from f{f} start ({:.0},{:.0},{:.0}) yaw {:#06x} buffer {buffer} stop {stop}", start[0], start[1], start[2], yaw as u16);
            for k in 0..frames {
                let (p, _key) = ov.pad_keyed(&traj, &played, GamePad::new(0, 0, 0));
                g.step(p)?;
                line(g.frame, p, &g.st);
                played.push(p);
                traj.push(g.st.clone());
                if g.st.play_mode == 0 {
                    game_frames += 1;
                }
                if matches!(g.st.action, ACT_LONG_JUMP | ACT_LONG_JUMP_LAND) {
                    peak = peak.max(-g.st.forward_vel);
                }
                if warped(&g.st) {
                    warp_at = Some(k + 1);
                    break;
                }
            }
            println!("overlay: peak |fv| {peak:.0}, {} real frames, {game_frames} unpaused, warp {:?}", played.len(), warp_at);
        }
        "mc" | "mwmc" => {
            let f: u32 = args[4].parse()?;
            let buffer = args[5] == "1";
            let stop: f32 = args[6].parse()?;
            let trials: usize = args[7].parse()?;
            let seed: u64 = args.get(8).map_or(Ok(1), |s| s.parse())?;
            let mut rng = SmallRng::seed_from_u64(seed);
            to(&mut g, f)?;
            let base = g.snap();
            let s0 = g.st.clone();
            let down = (s0.floor_normal[0].atan2(s0.floor_normal[2]) * 32768.0 / std::f32::consts::PI) as i32;
            let (mut k1, mut k2, mut stall, mut warp) = (0, 0, 0, 0);
            for t in 0..trials {
                g.restore(&base)?;
                // mwmc: the objective's overlay at downhill + `side` (args[9], hex,
                // default -0x2000); buffering then follows OBRRT_PAUSE_BUF.
                let side = args.get(9).map_or(-0x2000, |v| i32::from_str_radix(v.trim_start_matches("-0x"), 16).map(|x| if v.starts_with('-') { -x } else { x }).unwrap());
                let ov0 = if mw { obrrt_objectives::extra::tech::moatwarp::overlay(down + side, stop) } else { obrrt_objectives::extra::tech::blj::overlay_exit(s0.pos, down, 0.0, None, stop, true) };
                let ov = if buffer && !mw { obrrt_objectives::extra::tech::pausebuf::wrap(ov0) } else { ov0 };
                let mut genome = Vec::new();
                while genome.len() < 140 {
                    let a = rng.random_range(0.0..std::f32::consts::TAU);
                    let m = rng.random_range(0.0..127.0f32);
                    let n = rng.random_range(1..=10);
                    for _ in 0..n {
                        genome.push(GamePad::new(0, (m * a.cos()) as i8, (m * a.sin()) as i8));
                    }
                }
                let mut traj = vec![s0.clone()];
                let mut played: Vec<GamePad> = Vec::new();
                let (mut peak, mut stalled, mut w) = (0f32, false, None);
                for k in 0..140 {
                    let (p, _) = ov.pad_keyed(&traj, &played, genome[k]);
                    let before = g.st.pos;
                    g.step(p)?;
                    played.push(p);
                    traj.push(g.st.clone());
                    if matches!(g.st.action, ACT_LONG_JUMP | ACT_LONG_JUMP_LAND) {
                        peak = peak.max(-g.st.forward_vel);
                        stalled |= -g.st.forward_vel > 1000.0 && g.st.play_mode == 0 && p.button & A != 0 && hd(before, g.st.pos) < 1.0;
                    }
                    if warped(&g.st) {
                        w = Some(k + 1);
                        break;
                    }
                }
                k1 += (peak > 1000.0) as usize;
                k2 += (peak > 2000.0) as usize;
                stall += stalled as usize;
                warp += w.is_some() as usize;
                if peak > 1000.0 || w.is_some() || t < 5 {
                    println!("trial {t:4}: peak {peak:7.0} stalled {stalled} warp {w:?} end ({:.0},{:.0},{:.0}) lvl {}", g.st.pos[0], g.st.pos[1], g.st.pos[2], g.st.level);
                }
            }
            println!("mc from f{f} buffer {buffer} stop {stop}: {trials} trials, peak>1000 {k1}, >2000 {k2}, stalled {stall}, warped {warp}");
        }
        _ => bail!("unknown mode"),
    }
    Ok(())
}
