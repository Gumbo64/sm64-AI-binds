//! Movement library lab (objectives/src/moves): steers from a start state to
//! a fixed target with the generic controller and with the library, and
//! reports frames to get within 100 units, the closest approach and which
//! moves fired.
//!
//! cargo run --release --example moves_lab -- <rom> <start> <mode> [args]
//!
//! start:
//! - `boot`: power-on with the fixed menu-skipping inputs, idle until Mario
//!   stands still (as run/src/main.rs; castle grounds);
//! - `tree:<run_dir>:<node>`: replay a run's tree.bin to that node.
//!
//! modes:
//! - `steer X,Y,Z N [off|on|both] [every=8] [trace]`: N frames towards
//!   X,Y,Z (`dX,dY,dZ` with a leading `+` is relative to the start);
//! - `slopes R`: static floor triangles within R of Mario that walking
//!   doesn't climb (normal y < 0.866), nearest first;
//! - `scan X Z`: the static floors under (X, Z).
//!
//! - `gaps R`: places within R where a level floor ends and the same
//!   height continues 150-450 further (gap jumps), and where it drops 1000+
//!   (fall damage / ground pound spots).
//!
//! Options (any mode): `pre=X,Y,Z,N` first steers N frames towards X,Y,Z
//! (generic controller, mover off) to set up speed or a place;
//! `prestick=X,Z,N` instead holds the stick towards (X, Z) N frames;
//! `health=W` sets Mario's health to W wedges; `arrival=pass|stop` (default
//! stop) how the target is to be reached (`door`: a door).
//!
//! `OBRRT_MOVES_UPHILL=kick|airkick|jump` picks the uphill-kick variant.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use obrrt_core::buttons::{A_BUTTON, START_BUTTON};
use obrrt_core::{GamePad, Objective, St, check_layout, dist, read_state};
use obrrt_objectives::moves::{self, MoveKind};
use obrrt_search::route::read_tree_bin;
use obrrt_search::steer::steer;
use sm64_binds::{SM64Game, SM64GameGenerator};

struct Point([f32; 3]);

impl Objective for Point {
    fn kind(&self) -> &'static str {
        "Moves::Lab"
    }
    fn key(&self) -> String {
        format!("MovesLab {:?}", self.0)
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
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        arrival()
    }
}

/// `arrival=pass|stop` (default stop: the library's rules before arrivals);
/// `door` makes it a door.
fn arrival() -> obrrt_core::Arrival {
    if args_door() {
        return obrrt_core::Arrival::Door;
    }
    match std::env::args().find_map(|a| a.strip_prefix("arrival=").map(str::to_string)).as_deref() {
        Some("pass") => obrrt_core::Arrival::Pass,
        _ => obrrt_core::Arrival::Stop,
    }
}

/// run/src/main.rs's fixed boot.
fn boot_inputs() -> Vec<GamePad> {
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

fn start(game: &mut SM64Game, spec: &str) -> Result<u32> {
    let mut frame = 0u32;
    if spec == "boot" {
        for p in boot_inputs() {
            game.step_game(p)?;
            frame += 1;
        }
        let mut st = read_state(game, frame, false);
        let mut still = 0;
        while still < 10 || !st.controllable() {
            let before = st.pos;
            game.step_game(GamePad::new(0, 0, 0))?;
            frame += 1;
            st = read_state(game, frame, false);
            still = if st.pos == before { still + 1 } else { 0 };
        }
    } else if let Some(rest) = spec.strip_prefix("tree:") {
        let (dir, node) = rest.rsplit_once(':').ok_or_else(|| anyhow::anyhow!("tree:<run_dir>:<node>"))?;
        let node: usize = node.parse()?;
        let records = read_tree_bin(&format!("{dir}/tree.bin"))?;
        let mut chain = vec![node + 1];
        while let Some(&r) = chain.last() {
            if r == 0 {
                break;
            }
            let parent = records[r].0;
            chain.push(if parent >= 0 { parent as usize + 1 } else { 0 });
        }
        for r in chain.iter().rev() {
            for p in &records[*r].1 {
                game.step_game(*p)?;
                frame += 1;
            }
        }
    } else {
        bail!("start: boot or tree:<run_dir>:<node>");
    }
    check_layout(game)?;
    Ok(frame)
}

struct Outcome {
    frames: usize,
    /// First frame in a door action (pulling / pushing a door).
    door: Option<usize>,
    within100: Option<usize>,
    closest: (f32, usize),
    end: St,
    moves: BTreeMap<&'static str, usize>,
    max_y: f32,
    min_wedges: i32,
}

fn run(game: &mut SM64Game, frame0: u32, s0: &St, target: [f32; 3], n: usize, every: usize, lib: bool, trace: bool) -> Result<Outcome> {
    let obj = Point(target);
    let mut s = s0.clone();
    let mut frame = frame0;
    let mut out = Outcome { frames: 0, door: None, within100: None, closest: (dist(s.pos, target), 0), end: s.clone(), moves: BTreeMap::new(), max_y: s.pos[1], min_wedges: s.wedges() };
    for k in 0..n {
        if lib {
            *out.moves.entry(moves::classify_with(&s, target, arrival()).name()).or_default() += 1;
        }
        let pad = steer(&obj, &s, k, every);
        game.step_game(pad)?;
        frame += 1;
        out.frames += 1;
        s = read_state(game, frame, false);
        let d = dist(s.pos, target);
        if trace {
            println!(
                "{:4} pad {:04x} {:4} {:4} | pos {:7.0} {:7.0} {:7.0} fv {:5.1} vy {:6.1} act {:#010x} t {:3} fny {:.3} d {:6.0}{}",
                k + 1,
                pad.button,
                pad.stick_x,
                pad.stick_y,
                s.pos[0],
                s.pos[1],
                s.pos[2],
                s.forward_vel,
                s.vel[1],
                s.action,
                s.action_timer,
                s.floor_normal[1],
                d,
                if lib { format!(" {}", moves::classify_with(&s, target, arrival()).name()) } else { String::new() }
            );
        }
        out.max_y = out.max_y.max(s.pos[1]);
        if out.door.is_none() && matches!(s.action, 0x1320 | 0x1321) {
            out.door = Some(k + 1);
            if args_door() {
                out.end = s.clone();
                break;
            }
        }
        out.min_wedges = out.min_wedges.min(s.wedges());
        if d < out.closest.0 {
            out.closest = (d, k + 1);
        }
        if d < 100.0 && out.within100.is_none() {
            out.within100 = Some(k + 1);
            out.end = s.clone();
            break;
        }
        out.end = s.clone();
    }
    Ok(out)
}

/// `door`: a run ends when Mario enters a door.
fn args_door() -> bool {
    std::env::args().any(|a| a == "door")
}

fn parse3(s: &str) -> Result<[f32; 3]> {
    let v: Vec<f32> = s.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?;
    if v.len() != 3 {
        bail!("want X,Y,Z");
    }
    Ok([v[0], v[1], v[2]])
}

fn main() -> Result<()> {
    // The steering controller (objectives/src/moves; move library off).
    obrrt_objectives::moves::install_controller();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        bail!("usage: moves_lab <rom> <start> <steer|slopes|scan> [args]");
    }
    obrrt_core::geo::want_static_floors();
    let gen_ = SM64GameGenerator::from_file(&args[1])?;
    let mut game = gen_.create_game()?;
    let frame0 = start(&mut game, &args[2])?;
    // Read with objects: snapshots the zone's static floors.
    let s0 = read_state(&game, frame0, true);
    println!(
        "start: frame {frame0} level {} area {} pos {:.0?} action {:#010x} cam_yaw {:#06x}",
        s0.level, s0.area, s0.pos, s0.action, s0.cam_yaw as u16
    );
    let mut frame0 = frame0;
    let mut s0 = s0;
    for a in &args {
        if let Some(v) = a.strip_prefix("pre=") {
            let v: Vec<f32> = v.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?;
            let o = run(&mut game, frame0, &s0, [v[0], v[1], v[2]], v[3] as usize, 8, false, false)?;
            frame0 += o.frames as u32;
            s0 = read_state(&game, frame0, true);
        }
        if let Some(v) = a.strip_prefix("prestick=") {
            let v: Vec<f32> = v.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?;
            for _ in 0..v[2] as usize {
                let st = read_state(&game, frame0, false);
                let (sx, sy) = moves::camera_stick(&st, [v[0], st.pos[1], v[1]]);
                game.step_game(GamePad::new(0, sx, sy))?;
                frame0 += 1;
            }
            s0 = read_state(&game, frame0, true);
        }
        if let Some(v) = a.strip_prefix("health=") {
            let w: i16 = v.parse()?;
            let at = obrrt_core::layout::ADDR_gMarioStates as u64 + obrrt_core::layout::MARIO_health as u64;
            let (store, _, memory) = game.wasm_parts();
            memory.view(store).write(at, &((w << 8) | 0x80).to_le_bytes())?;
            s0 = read_state(&game, frame0, true);
        }
    }
    if args.iter().any(|a| a.starts_with("pre") || a.starts_with("health=")) {
        println!(
            "set up: frame {frame0} pos {:.0?} action {:#010x} fv {:.1} wedges {} wall {:.2?}",
            s0.pos, s0.action, s0.forward_vel, s0.wedges(), s0.wall_normal
        );
    }
    let tris = obrrt_core::geo::static_floors(s0.zone());
    match args[3].as_str() {
        "slopes" => {
            let r: f32 = args.get(4).map_or(Ok(3000.0), |v| v.parse())?;
            let tris = tris.ok_or_else(|| anyhow::anyhow!("no floor snapshot"))?;
            let mut v: Vec<_> = tris
                .iter()
                .filter(|t| t.n[1] < 0.866 && t.n[1] > 0.2)
                .map(|t| {
                    let c = [0, 1, 2].map(|k| (t.v[0][k] + t.v[1][k] + t.v[2][k]) / 3.0);
                    (dist(c, s0.pos), c, t.n, t.kind)
                })
                .filter(|x| x.0 < r)
                .collect();
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            for (d, c, n, k) in v.iter().take(60) {
                // Uphill direction: against the normal's horizontal part.
                println!("d {d:6.0} centre {:7.0} {:7.0} {:7.0} ny {:.3} ({:4.1}°) uphill ({:.2},{:.2}) kind {k:#x}", c[0], c[1], c[2], n[1], n[1].acos().to_degrees(), -n[0], -n[2]);
            }
        }
        "walls" => {
            // Level floor whose next 100 ahead rises 400-3000 (a cliff or wall),
            // nearest first; with `pair`, only those with another such
            // wall facing them within 250-600 behind (wall-kick chains).
            let r: f32 = args.get(4).map_or(Ok(2000.0), |v| v.parse())?;
            let pair = args.iter().any(|a| a == "pair");
            let tris = tris.ok_or_else(|| anyhow::anyhow!("no floor snapshot"))?;
            let fl = |x: f32, z: f32, y: f32| obrrt_core::geo::floor_below(&tris, x, z, y);
            let mut out = Vec::new();
            let n = (r / 100.0) as i32;
            for ix in -n..=n {
                for iz in -n..=n {
                    let (x, z) = (s0.pos[0] + ix as f32 * 100.0, s0.pos[2] + iz as f32 * 100.0);
                    let Some(y) = fl(x, z, s0.pos[1] + 3000.0) else { continue };
                    for k in 0..8 {
                        let a = k as f32 * std::f32::consts::TAU / 8.0;
                        let d = [a.sin(), a.cos()];
                        let wall = |t: f32| fl(x + d[0] * t, z + d[1] * t, y + 3000.0).is_some_and(|h| h > y + 400.0);
                        if !(fl(x + d[0] * 50.0, z + d[1] * 50.0, y + 50.0).is_some_and(|h| (h - y).abs() < 30.0) && wall(100.0)) {
                            continue;
                        }
                        if pair {
                            let back = |t: f32| fl(x - d[0] * t, z - d[1] * t, y + 3000.0).is_some_and(|h| h > y + 400.0);
                            if !(2..=6).any(|j| back(j as f32 * 100.0)) {
                                continue;
                            }
                        }
                        out.push((((x - s0.pos[0]).powi(2) + (z - s0.pos[2]).powi(2)).sqrt(), x, y, z, d));
                    }
                }
            }
            out.sort_by(|a, b| a.0.total_cmp(&b.0));
            for (dd, x, y, z, d) in out.iter().take(25) {
                println!("wall {dd:5.0} away: from ({x:.0}, {y:.0}, {z:.0}) towards ({:.2},{:.2})", d[0], d[1]);
            }
        }
        "doors" => {
            for o in s0.objects().iter().filter(|o| o.interact_type & (obrrt_core::layout::INTERACT_DOOR | obrrt_core::layout::INTERACT_WARP_DOOR) != 0) {
                println!("door {} at {:.0?} yaw {:#06x} ({:.0} away)", o.bhv_name(), o.pos, o.yaw, dist(o.pos, s0.pos));
            }
        }
        "probe" => {
            // What the library sees from the start state towards X,Y,Z.
            obrrt_core::mover::publish_zone(s0.zone());
            let t = parse3(&args[4])?;
            let ctx = moves::Ctx::with(&s0, t, arrival());
            println!("key {:?}", moves::key(&s0, t));
            println!("classify {}", moves::classify_with(&s0, t, arrival()).name());
            println!("ctx: hd {:.0} dy {:.0} fv {:.1} yaw_err {:#06x} grade {:.2} base {:.0}", ctx.hd, ctx.dy, ctx.fv, ctx.yaw_err, ctx.grade, ctx.base);
            if let Some(ah) = &ctx.ahead {
                println!("ahead wall {:?}", ah.wall);
                for ((d, f), (_, h)) in ah.low.iter().zip(&ah.high) {
                    println!("  {d:5.0}: low {:?} high {:?}", f.map(|f| (f.y.round(), f.normal[1])), h.map(|h| h.round()));
                }
            }
            for k in 0..8 {
                let b = k as f32 * std::f32::consts::TAU / 8.0;
                println!("  dir ({:.2},{:.2}): wall {:?}", b.sin(), b.cos(), moves::geom::wall_ahead(&s0, [b.sin(), b.cos()], 2000.0, ctx.base, 0.1));
            }
            let a = s0.yaw as f32 * std::f32::consts::TAU / 65536.0;
            println!(
                "facing wall (head-on) {:?}, walls in zone {}",
                moves::geom::wall_ahead(&s0, [a.sin(), a.cos()], 1000.0, ctx.base, 0.7),
                obrrt_core::geo::static_walls(s0.zone()).map_or(0, |w| w.len())
            );
        }
        "gaps" => {
            let r: f32 = args.get(4).map_or(Ok(2000.0), |v| v.parse())?;
            let tris = tris.ok_or_else(|| anyhow::anyhow!("no floor snapshot"))?;
            let fl = |x: f32, z: f32, y: f32| obrrt_core::geo::floor_below(&tris, x, z, y);
            let mut found = 0;
            let step = 100.0;
            let n = (r / step) as i32;
            'outer: for ix in -n..=n {
                for iz in -n..=n {
                    let (x, z) = (s0.pos[0] + ix as f32 * step, s0.pos[2] + iz as f32 * step);
                    let Some(y) = fl(x, z, s0.pos[1] + 2000.0) else { continue };
                    for k in 0..8 {
                        let a = k as f32 * std::f32::consts::TAU / 8.0;
                        let d = [a.sin(), a.cos()];
                        let at = |t: f32| fl(x + d[0] * t, z + d[1] * t, y + 100.0);
                        // Level run-up of 100.
                        if !at(100.0).is_some_and(|h| (h - y).abs() < 30.0) {
                            continue;
                        }
                        let hole = at(200.0).is_none_or(|h| h < y - 300.0);
                        if !hole {
                            continue;
                        }
                        let far = (3..=6).map(|j| j as f32 * 100.0).find(|&t| at(t).is_some_and(|h| (h - y).abs() < 60.0));
                        let below = at(250.0).map_or(f32::NAN, |h| y - h);
                        if far.is_some() || below > 1000.0 {
                            println!(
                                "{} at ({x:.0}, {y:.0}, {z:.0}) dir ({:.2},{:.2}): far side {:?}, drop {below:.0}",
                                if far.is_some() { "gap " } else { "drop" },
                                d[0],
                                d[1],
                                far
                            );
                            found += 1;
                            if found > 80 {
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
        "scan" => {
            let (x, z): (f32, f32) = (args[4].parse()?, args[5].parse()?);
            let tris = tris.ok_or_else(|| anyhow::anyhow!("no floor snapshot"))?;
            let mut y = 20000.0;
            while let Some(f) = obrrt_core::geo::floor_below(&tris, x, z, y) {
                println!("floor y {f:.0}");
                y = f - 1.0;
            }
        }
        "steer" => {
            let rel = args[4].starts_with('+');
            let mut target = parse3(args[4].trim_start_matches('+'))?;
            if rel {
                for k in 0..3 {
                    target[k] += s0.pos[k];
                }
            }
            let n: usize = args[5].parse()?;
            let which = args.get(6).map_or("both", |s| s.as_str());
            let every = args.iter().find_map(|a| a.strip_prefix("every=")).map_or(Ok(8), |v| v.parse())?;
            let trace = args.iter().any(|a| a == "trace");
            println!("target {target:.0?}: start distance {:.0}, {n} frames, generic A period {every}", dist(s0.pos, target));
            let save = game.save_state();
            let report = |name: &str, o: &Outcome| {
                println!(
                    "{name:4}: within 100 at {} | closest {:.0} at {} | end {:.0?} action {:#010x} | max y {:.0} | wedges {}{}",
                    o.within100.map_or("never".to_string(), |f| format!("frame {f}")),
                    o.closest.0,
                    o.closest.1,
                    o.end.pos,
                    o.end.action,
                    o.max_y,
                    o.min_wedges,
                    o.door.map_or(String::new(), |f| format!(" | door at frame {f}"))
                );
                if !o.moves.is_empty() {
                    let m: Vec<String> = o.moves.iter().map(|(k, v)| format!("{k} {v}")).collect();
                    println!("      moves: {}", m.join(", "));
                }
            };
            if which == "off" || which == "both" {
                let o = run(&mut game, frame0, &s0, target, n, every, false, trace)?;
                report("off", &o);
                game.load_state(&save)?;
            }
            if which == "on" || which == "both" {
                moves::install();
                moves::observe(&s0);
                obrrt_core::mover::publish();
                let o = run(&mut game, frame0, &s0, target, n, every, true, trace)?;
                report("on", &o);
            }
        }
        m => bail!("unknown mode {m}"),
    }
    let _ = MoveKind::ALL;
    Ok(())
}
