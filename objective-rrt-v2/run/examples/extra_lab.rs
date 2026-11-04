//! Movement policies lab (objectives/src/moves/extra.rs): from a real state on
//! a recorded route, rolls each policy out alone towards a target and prints
//! what it got, next to the plain steering and the move library.
//!
//! extra_lab <rom> <movie.m64|raw> <frame=N | near=X,Y,Z,R> <X,Y,Z> <N> [opts]
//!
//! - `frame=N`: the movie to frame N; `near=X,Y,Z,R`: to its first
//!   controllable state within R of X,Y,Z.
//! - `<X,Y,Z>`: the target (`+dX,dY,dZ`: relative to the start); N frames.
//! - `idle=K`: K neutral frames first (a standing start);
//!   `run=X,Z,K`: K frames of full stick towards X,Z first (speed).
//! - `only=a,b`: these policies instead of all; `trace=a`: its frames too;
//!   `arrival=stop`: the target is stopped at (default pass).
//!
//! Columns: `in` frames to within 150 of the target (`-` not); `closest`
//! its distance then; `up` the highest above the start; `end up` height at the
//! last frame (grounded: what is kept), `hd gain` the horizontal distance won
//! at the last frame; `fv` the last and the highest forward speed.

use anyhow::{Result, bail};
use obrrt_core::{Arrival, GamePad, Objective, St, dist, read_state};
use obrrt_objectives::moves;
use obrrt_search::steer::{steer, steer_policy};
use sm64_binds::SM64GameGenerator;

struct Point([f32; 3], Arrival);

impl Objective for Point {
    fn kind(&self) -> &'static str {
        "Moves::ExtraLab"
    }
    fn key(&self) -> String {
        format!("ExtraLab {:?}", self.0)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        0.0
    }
    fn base_criteria(&self, _s: &St) -> bool {
        false
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut obrrt_core::Mem) -> obrrt_core::Eta {
        obrrt_core::Eta::Frames(obrrt_core::frames_for(s.dist(self.0)))
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.0)
    }
    fn arrival(&self, _s: &St) -> Arrival {
        self.1
    }
}

fn floats(s: &str) -> Result<Vec<f32>> {
    Ok(s.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?)
}

fn main() -> Result<()> {
    // SAFETY: before any thread is started.
    unsafe {
        std::env::set_var("OBRRT_MOVES_EXTRA", "1");
        std::env::set_var("OBRRT_MOVES_MODES", "1");
    }
    moves::install_policies();
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        bail!("usage: extra_lab <rom> <movie> <frame=N|near=X,Y,Z,R> <X,Y,Z> <N> [idle=K] [run=X,Z,K] [only=a,b] [trace=a] [arrival=stop]");
    }
    let opt = |k: &str| a.iter().find_map(|s| s.strip_prefix(k).map(str::to_string));
    let gen_ = SM64GameGenerator::from_file(&a[1])?;
    let mut g = gen_.create_game()?;
    let b = std::fs::read(&a[2])?;
    let body = if b.starts_with(b"M64\x1a") { &b[0x400..] } else { &b[..] };
    let pads: Vec<GamePad> = body.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect();
    let mut f = 0u32;
    let want_frame = a[3].strip_prefix("frame=").map(|v| v.parse::<u32>()).transpose()?;
    let near = a[3].strip_prefix("near=").map(floats).transpose()?;
    let mut found = false;
    for p in &pads {
        g.step_game(*p)?;
        f += 1;
        if want_frame == Some(f) {
            found = true;
            break;
        }
        if let Some(n) = &near {
            let s = read_state(&g, f, false);
            if s.controllable() && dist(s.pos, [n[0], n[1], n[2]]) < n[3] {
                found = true;
                break;
            }
        }
    }
    if !found {
        bail!("movie ended before {}", a[3]);
    }
    if let Some(k) = opt("idle=") {
        for _ in 0..k.parse::<usize>()? {
            g.step_game(GamePad::new(0, 0, 0))?;
            f += 1;
        }
    }
    if let Some(v) = opt("run=") {
        let v = floats(&v)?;
        for _ in 0..v[2] as usize {
            let s = read_state(&g, f, false);
            let (x, y) = moves::camera_stick(&s, [v[0], s.pos[1], v[1]]);
            g.step_game(GamePad::new(0, x, y))?;
            f += 1;
        }
    }
    // With objects: snapshots the zone's static floors.
    let s0 = read_state(&g, f, true);
    moves::observe(&s0);
    obrrt_core::mover::publish();
    let target = if let Some(rel) = a[4].strip_prefix('+') {
        let v = floats(rel)?;
        [s0.pos[0] + v[0], s0.pos[1] + v[1], s0.pos[2] + v[2]]
    } else {
        let v = floats(&a[4])?;
        [v[0], v[1], v[2]]
    };
    let n: usize = a[5].parse()?;
    let arrival = if a.iter().any(|x| x == "arrival=stop") { Arrival::Stop } else { Arrival::Pass };
    let obj = Point(target, arrival);
    println!(
        "start: frame {f} pos {:.0?} action {:#010x} fv {:.1} yaw {:#06x} floor ny {:.3} kind {:#x}; target {target:.0?} ({:.0} away, {:+.0} up), {n} frames",
        s0.pos, s0.action, s0.forward_vel, s0.yaw as u16, s0.floor_normal[1], s0.floor_type, s0.hdist(target), target[1] - s0.pos[1]
    );
    let save = g.save_state();
    let m = obrrt_core::mover::mover().ok_or_else(|| anyhow::anyhow!("no mover"))?;
    let only = opt("only=").map(|v| v.split(',').map(str::to_string).collect::<Vec<_>>());
    let trace = opt("trace=");
    println!("{:<14} {:>5} {:>4} {:>7} {:>6} {:>7} {:>8} {:>6} {:>6}", "policy", "plaus", "in", "closest", "up", "end up", "hd gain", "fv", "max fv");
    // -2 plain steering, -1 the library, then the policies.
    let rows: Vec<i32> = (-2..m.policies() as i32).collect();
    for r in rows {
        let name = match r {
            -2 => "(steering)".to_string(),
            -1 => "(library)".to_string(),
            p => m.policy_name(p as usize).to_string(),
        };
        let plausible = r < 0 || m.policy_plausible(r as usize, &s0, target, arrival);
        let listed = only.as_ref().map_or(plausible, |o| r < 0 || o.contains(&name));
        if !listed {
            continue;
        }
        moves::set_steering(r == -1);
        g.load_state(&save)?;
        let mut s = s0.clone();
        let mut fr = f;
        let (mut hit, mut closest, mut up, mut maxfv) = (None, dist(s0.pos, target), 0f32, s0.forward_vel);
        for k in 0..n {
            let pad = if r >= 0 { steer_policy(&obj, &s, k, 8, Some(r as usize)) } else { steer(&obj, &s, k, 8) };
            g.step_game(pad)?;
            fr += 1;
            s = read_state(&g, fr, false);
            let d = dist(s.pos, target);
            if trace.as_deref() == Some(name.as_str()) {
                println!(
                    "{:4} pad {:04x} {:4} {:4} | {:7.0} {:7.0} {:7.0} fv {:6.1} vy {:6.1} act {:#010x} t {:2} ny {:.3} yaw {:#06x} d {:6.0}",
                    k + 1, pad.button, pad.stick_x, pad.stick_y, s.pos[0], s.pos[1], s.pos[2], s.forward_vel, s.vel[1], s.action, s.action_timer, s.floor_normal[1], s.yaw as u16, d
                );
            }
            closest = closest.min(d);
            up = up.max(s.pos[1] - s0.pos[1]);
            maxfv = maxfv.max(s.forward_vel);
            if d < 150.0 && hit.is_none() {
                hit = Some(k + 1);
                break;
            }
            if s.zone() != s0.zone() {
                break;
            }
        }
        println!(
            "{:<14} {:>5} {:>4} {:>7.0} {:>6.0} {:>7.0} {:>8.0} {:>6.1} {:>6.1}",
            name,
            if plausible { "yes" } else { "no" },
            hit.map_or("-".to_string(), |h| h.to_string()),
            closest,
            up,
            s.pos[1] - s0.pos[1],
            s0.hdist(target) - s.hdist(target),
            s.forward_vel,
            maxfv
        );
    }
    Ok(())
}
