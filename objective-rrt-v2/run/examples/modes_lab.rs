//! Control-mode lab: measures the cannon, swimming, flying, slides and
//! poles from a start state, for the mode policies (objectives/src/moves:
//! cannon.rs, water_air.rs, slide.rs, pole.rs) and ref/arcs/modes.json.
//!
//! cargo run --release --example modes_lab -- <rom> <start> <cmd> [args]
//!
//! start: `tree:<run_dir>:<node>`, `raw:<inputs>` (headerless prefix),
//! `tas:<file.m64>:<N>` (the first N frames of an m64).
//!
//! - `cannon [X,Y,Z]`: in the cannon: for pitches 0x0400..0x3800 aims
//!   (yaw unchanged), fires, follows the shot; prints per pitch the
//!   frames, landing, and the closed-form model's worst error before the
//!   first collision (cannon::at_frame), and cannon::predict's landing vs
//!   the game's. With a target, also each cannon policy's flight (frames
//!   fire -> end, miss).
//! - `swim`: in water: A every 4 frames / the held-A TAS cadence / A
//!   held (flutter), heading level for 150 frames: mean speed over the
//!   last 100.
//! - `fly`: flying: stickY in -64..64 held 120 frames: speed and height
//!   after 30/60/120, and water_air::fly_step's position error at 30.
//! - `slide X,Y,Z N`: sliding: each slide policy (and the generic steering)
//!   to the target: frames to within 300, mean speed; then constant stick
//!   angles off the velocity (0/30/45/60/90/180°) for 30 frames: speed
//!   ratio and turn; then a switch to a stomach slide (A, B in the air)
//!   with slide-course after it.
//! - `pole [X,Y,Z]`: on a pole: climb patterns (full stick, release 1/4,
//!   1/16) to the top: frames and the rate; then the kick-off arc from
//!   where Mario is (A while holding): per frame height and distance.
//!
//! Output lines starting `json ` are the measurements (one JSON object
//! each) for ref/arcs/modes.json.

use anyhow::{Result, bail};
use obrrt_core::buttons::{A_BUTTON, B_BUTTON};
use obrrt_core::{Arrival, GamePad, St, check_layout, dist, read_state};
use obrrt_objectives::moves::{self, cannon, pole, slide, water_air};
use obrrt_search::route::read_tree_bin;
use sm64_binds::{SM64Game, SM64GameGenerator};

fn start(game: &mut SM64Game, spec: &str) -> Result<u32> {
    let mut frame = 0u32;
    if let Some(rest) = spec.strip_prefix("tree:") {
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
    } else if let Some(path) = spec.strip_prefix("raw:") {
        let bytes = std::fs::read(path)?;
        for c in bytes.chunks_exact(4) {
            game.step_game(GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8))?;
            frame += 1;
        }
    } else if let Some(rest) = spec.strip_prefix("tas:") {
        let (path, n) = rest.rsplit_once(':').ok_or_else(|| anyhow::anyhow!("tas:<file.m64>:<N>"))?;
        let n: usize = n.parse()?;
        let bytes = std::fs::read(path)?;
        for c in bytes[0x400..].chunks_exact(4).take(n) {
            game.step_game(GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8))?;
            frame += 1;
        }
    } else {
        bail!("start: tree:<run_dir>:<node>, raw:<inputs> or tas:<file.m64>:<N>");
    }
    check_layout(game)?;
    Ok(frame)
}

fn parse3(s: &str) -> Result<[f32; 3]> {
    let v: Vec<f32> = s.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?;
    if v.len() != 3 {
        bail!("X,Y,Z");
    }
    Ok([v[0], v[1], v[2]])
}

struct Lab {
    game: SM64Game,
    frame0: u32,
    save: sm64_binds::SaveState,
    s0: St,
}

impl Lab {
    fn reset(&mut self) -> Result<St> {
        self.game.load_state(&self.save)?;
        Ok(self.s0.clone())
    }
    fn step(&mut self, k: u32, p: GamePad) -> Result<St> {
        self.game.step_game(p)?;
        Ok(read_state(&self.game, self.frame0 + k + 1, false))
    }
}

/// A pad with the game's stick (±64) values (as modes.rs `pad`).
fn pad(b: u16, cx: f32, cy: f32) -> GamePad {
    let raw = |c: f32| {
        let c = c.round();
        if c.abs() < 2.0 { 0 } else { (c + 6.0 * c.signum()).clamp(-127.0, 127.0) as i8 }
    };
    GamePad::new(b, raw(cx), raw(cy))
}

fn main() -> Result<()> {
    moves::install_controller();
    obrrt_core::geo::want_static_floors();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        bail!("usage: modes_lab <rom> <start> cannon|swim|fly|slide|pole [args]");
    }
    let gen_ = SM64GameGenerator::from_file(&args[1])?;
    let mut game = gen_.create_game()?;
    let frame0 = start(&mut game, &args[2])?;
    let s0 = read_state(&game, frame0, true);
    moves::observe(&s0);
    obrrt_core::mover::publish();
    let save = game.save_state();
    println!(
        "start: frame {frame0} level {} area {} pos {:.0?} action {:#010x} fv {:.1} pitch {} yaw {} water {} floor {:#x} ny {:.3}",
        s0.level, s0.area, s0.pos, s0.action, s0.forward_vel, s0.pitch, s0.yaw, s0.water_level, s0.floor_type, s0.floor_normal[1]
    );
    let mut lab = Lab { game, frame0, save, s0 };
    let rest = &args[4..];
    match args[3].as_str() {
        "cannon" => cannon_cmd(&mut lab, rest),
        "swim" => swim_cmd(&mut lab),
        "fly" => fly_cmd(&mut lab),
        "slide" => slide_cmd(&mut lab, rest),
        "pole" => pole_cmd(&mut lab, rest),
        c => bail!("unknown command {c}"),
    }
}

// ---------------------------------------------------------------------------

const ACT_IN_CANNON: u32 = 0x1371;
const ACT_SHOT: u32 = 0x0088_0898;

/// Holds pitch/yaw towards (pitch, yaw) until aimed (or 300 frames), then
/// A; returns (frames used, state after the fire frame) or None.
fn aim_and_fire(lab: &mut Lab, pitch: i32, yaw: i32) -> Result<Option<(u32, St, St)>> {
    let mut s = lab.reset()?;
    let mut k = 0u32;
    while k < 400 {
        let pe = pitch - s.pitch as i32;
        let ye = ((yaw - s.yaw as i32) as i16) as i32;
        let aimed = pe.abs() < 20 && ye.abs() < 20;
        let cx = if ye.abs() < 20 { 0.0 } else { (-(ye as f32) / 10.0).clamp(-64.0, 64.0) };
        let cy = if pe.abs() < 20 { 0.0 } else { (-(pe as f32) / 10.0).clamp(-64.0, 64.0) };
        let fire = aimed && k % 2 == 0 && k > 0;
        let before = s.clone();
        let pd = pad(if fire { A_BUTTON } else { 0 }, cx, cy);
        if std::env::var("LAB_DEBUG").is_ok() && k % 20 == 0 {
            println!("  k {k} pitch {} yaw {} pad {:?} act {:#x}", s.pitch, s.yaw, pd, s.action);
        }
        s = lab.step(k, pd)?;
        k += 1;
        if s.action == ACT_SHOT {
            return Ok(Some((k, before, s)));
        }
        if s.action != ACT_IN_CANNON {
            if std::env::var("LAB_DEBUG").is_ok() {
                println!("  left the cannon at {k}: act {:#x}", s.action);
            }
            return Ok(None);
        }
    }
    if std::env::var("LAB_DEBUG").is_ok() {
        println!("  not aimed after 400: pitch {} yaw {} (want {pitch} {yaw})", s.pitch, s.yaw);
    }
    Ok(None)
}

struct Shot {
    frames: usize,
    end: [f32; 3],
    end_action: u32,
    model_err: f32,
    trace: Vec<[f32; 3]>,
}

/// Follows a shot from the state after the fire frame (`after`, at `k`).
fn follow(lab: &mut Lab, k: u32, before: &St, after: &St, pitch: i32) -> Result<Shot> {
    let mut s = after.clone();
    let origin = before.pos;
    let yaw = after.yaw as i32;
    let (mut n, mut err) = (0usize, 0f32);
    let mut trace = vec![s.pos];
    while s.action == ACT_SHOT && n < 400 {
        s = lab.step(k + n as u32, GamePad::new(0, 0, 0))?;
        n += 1;
        trace.push(s.pos);
        if s.action == ACT_SHOT {
            let (h, y) = cannon::at_frame(pitch, n as f32);
            let mp = [origin[0] + h * cannon::sins(yaw), origin[1] + y, origin[2] + h * cannon::coss(yaw)];
            err = err.max(dist(mp, s.pos));
        }
    }
    Ok(Shot { frames: n, end: s.pos, end_action: s.action, model_err: err, trace })
}

fn cannon_cmd(lab: &mut Lab, rest: &[String]) -> Result<()> {
    if lab.s0.action != ACT_IN_CANNON {
        bail!("not in a cannon");
    }
    // Wait (neutral stick) for the cannon to open: the aim moves only then
    // (the state's yaw is Mario's before it, maybe out of the cannon's
    // range); continue from there.
    {
        let mut s = lab.reset()?;
        let mut k = 0u32;
        while k < 300 {
            s = lab.step(k, pad(0, 0.0, 64.0))?;
            k += 1;
            if s.pitch != lab.s0.pitch || s.yaw != lab.s0.yaw {
                break;
            }
        }
        // Back to pitch 0 with the stick held the other way a few frames.
        for _ in 0..4 {
            s = lab.step(k, pad(0, 0.0, 64.0))?;
            k += 1;
        }
        println!("cannon open after {k} frames: pitch {} yaw {}", s.pitch, s.yaw);
        lab.save = lab.game.save_state();
        lab.frame0 += k;
        lab.s0 = read_state(&lab.game, lab.frame0, false);
    }
    let yaw = lab.s0.yaw as i32;
    let mut p = 0x400;
    while p <= 0x3800 {
        let Some((k, before, after)) = aim_and_fire(lab, p, yaw)? else {
            println!("pitch {p:#06x}: no fire");
            p += 0x400;
            continue;
        };
        let real_pitch = before.pitch as i32;
        let pred = cannon::predict(&before, real_pitch, before.yaw as i32, 400);
        let shot = follow(lab, k, &before, &after, real_pitch)?;
        let hd = before.hdist(shot.end);
        println!(
            "pitch {real_pitch:#06x}: aim frames {k} flight {} end {:.0?} ({:#x}) hd {hd:.0} dy {:.0} | model max err {:.2} | predict {}",
            shot.frames,
            shot.end,
            shot.end_action,
            shot.end[1] - before.pos[1],
            shot.model_err,
            pred.map_or("-".into(), |f| format!("{} frames end {:.0?} landed {} miss {:.0}", f.frames, f.end, f.landed, dist(f.end, shot.end)))
        );
        println!(
            "json {{\"cannon\":{{\"zone\":[{},{}],\"pitch\":{real_pitch},\"frames\":{},\"hd\":{hd:.1},\"dy\":{:.1},\"model_err\":{:.3},\"predict_frames\":{},\"predict_miss\":{:.1}}}}}",
            before.level,
            before.area,
            shot.frames,
            shot.end[1] - before.pos[1],
            shot.model_err,
            pred.map_or(-1, |f| f.frames as i64),
            pred.map_or(-1.0, |f| dist(f.end, shot.end))
        );
        p += 0x400;
    }
    // Solve check: aim at each target given (pass, stop, lob, edge).
    for t in rest {
        let target = parse3(t)?;
        println!("target {target:.0?} ({:.0} away)", lab.s0.hdist(target));
        for (name, k, arr) in [
            ("cannon-aim pass", cannon::Kind::Aim, Arrival::Pass),
            ("cannon-aim stop", cannon::Kind::Aim, Arrival::Stop),
            ("cannon-lob pass", cannon::Kind::Lob, Arrival::Pass),
            ("cannon-lob stop", cannon::Kind::Lob, Arrival::Stop),
            ("cannon-edge", cannon::Kind::Edge, Arrival::Stop),
        ] {
            let Some((pitch, yaw)) = cannon::aim(k, &lab.s0, target, arr) else {
                println!("{name}: no aim");
                continue;
            };
            let Some((kk, before, after)) = aim_and_fire(lab, pitch, yaw)? else {
                println!("{name}: no fire");
                continue;
            };
            let shot = follow(lab, kk, &before, &after, before.pitch as i32)?;
            let closest = shot.trace.iter().map(|q| dist(*q, target)).fold(f32::MAX, f32::min);
            // After the shot: frames until the dive slide stops (stick
            // neutral) and where.
            let mut s = read_state(&lab.game, lab.frame0 + kk + shot.frames as u32, false);
            let mut m = 0;
            while m < 200 && s.forward_vel.abs() > 0.5 && s.action & 0x800 == 0 {
                s = lab.step(kk + shot.frames as u32 + m, GamePad::new(0, 0, 0))?;
                m += 1;
            }
            println!(
                "{name}: pitch {pitch:#06x} (fired {:#06x}) flight {} end {:.0?} ({:#x}) closest {closest:.0} land-miss {:.0} | rest after {m} at {:.0?} miss {:.0}",
                before.pitch,
                shot.frames,
                shot.end,
                shot.end_action,
                dist(shot.end, target),
                s.pos,
                dist(s.pos, target)
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------

fn swim_cmd(lab: &mut Lab) -> Result<()> {
    if !water_air::swimming(&lab.s0) {
        bail!("not swimming");
    }
    for name in ["a-every-4", "tas-hold", "flutter-hold", "a-every-2"] {
        let mut s = lab.reset()?;
        let (mut sum, mut n, mut top) = (0f32, 0, 0f32);
        let start = s.pos;
        for k in 0..150u32 {
            let a = match name {
                "a-every-4" => k % 4 == 0,
                "a-every-2" => k % 2 == 0,
                "tas-hold" => moves::modes::hold_a(&s, k as usize),
                _ => true,
            };
            // Level, straight on: pitch towards 0 (stick y = pitch/252).
            let cy = (s.pitch as f32 / 252.0).clamp(-58.0, 64.0);
            s = lab.step(k, pad(if a { A_BUTTON } else { 0 }, 0.0, cy))?;
            if k >= 50 {
                sum += s.forward_vel;
                n += 1;
            }
            top = top.max(s.forward_vel);
        }
        let mean = sum / n as f32;
        println!("swim {name}: mean speed (frames 50-150) {mean:.2} max {top:.2} distance {:.0}", dist(start, s.pos));
        println!("json {{\"swim\":{{\"cadence\":\"{name}\",\"mean\":{mean:.3},\"max\":{top:.3}}}}}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------

fn fly_cmd(lab: &mut Lab) -> Result<()> {
    if lab.s0.action != obrrt_core::layout::ACT_FLYING {
        bail!("not flying");
    }
    for cy in [-64.0f32, -32.0, -16.0, 0.0, 16.0, 32.0, 64.0] {
        let mut s = lab.reset()?;
        let mut model = water_air::Flyer::of(&s);
        let mut err30 = -1.0;
        let mut marks = Vec::new();
        for k in 0..120u32 {
            s = lab.step(k, pad(0, 0.0, cy))?;
            water_air::fly_step(&mut model, 0.0, cy);
            if k + 1 == 30 {
                err30 = if s.action == obrrt_core::layout::ACT_FLYING { dist(model.pos, s.pos) } else { -1.0 };
            }
            if matches!(k + 1, 30 | 60 | 120) {
                marks.push((k + 1, s.forward_vel, s.pos[1] - lab.s0.pos[1], s.pitch, s.action));
            }
            if s.action != obrrt_core::layout::ACT_FLYING {
                break;
            }
        }
        let m: Vec<String> = marks.iter().map(|(f, v, dy, p, a)| format!("@{f}: fv {v:.1} dy {dy:.0} pitch {p} act {a:#x}")).collect();
        println!("fly stickY {cy}: {} | model err @30 {err30:.2}", m.join(" | "));
        let js: Vec<String> = marks.iter().map(|(f, v, dy, _, _)| format!("{{\"frame\":{f},\"fv\":{v:.2},\"dy\":{dy:.1}}}")).collect();
        println!("json {{\"fly\":{{\"stick_y\":{cy},\"fv0\":{:.2},\"marks\":[{}],\"model_err30\":{err30:.3}}}}}", lab.s0.forward_vel, js.join(","));
    }
    Ok(())
}

// ---------------------------------------------------------------------------

fn slide_cmd(lab: &mut Lab, rest: &[String]) -> Result<()> {
    let target = parse3(rest.first().ok_or_else(|| anyhow::anyhow!("slide X,Y,Z N"))?)?;
    let n: u32 = rest.get(1).map_or(Ok(600), |v| v.parse())?;
    let obj = moves_target(target);
    let pols = [
        ("generic", None),
        ("slide-hold", Some(slide::Slide::Hold)),
        ("slide-course", Some(slide::Slide::Course)),
        ("slide-sharp", Some(slide::Slide::Sharp)),
        ("slide-jump", Some(slide::Slide::Jump)),
    ];
    for (name, k) in pols {
        let mut s = lab.reset()?;
        let (mut hit, mut sum, mut m) = (None, 0f32, 0);
        for f in 0..n {
            let p = k.and_then(|k| slide::slide_pad(k, &s, target, f as usize)).unwrap_or_else(|| obrrt_search::steer::steer(&obj, &s, f as usize, 8));
            s = lab.step(f, p)?;
            sum += (s.vel[0] * s.vel[0] + s.vel[2] * s.vel[2]).sqrt();
            m += 1;
            if dist(s.pos, target) < 300.0 {
                hit = Some(f + 1);
                break;
            }
        }
        println!("slide {name}: to target {hit:?} mean hspeed {:.1} end {:.0?} act {:#x} d {:.0}", sum / m as f32, s.pos, s.action, dist(s.pos, target));
        println!("json {{\"slide\":{{\"policy\":\"{name}\",\"frames\":{},\"mean_hspeed\":{:.2}}}}}", hit.map_or(-1, |h| h as i64), sum / m as f32);
    }
    // Constant stick angle off the velocity for 30 frames.
    for deg in [0.0f32, 30.0, 45.0, 60.0, 90.0, 180.0] {
        let mut s = lab.reset()?;
        let v0 = (s.vel[0] * s.vel[0] + s.vel[2] * s.vel[2]).sqrt();
        let h0 = s.vel[0].atan2(s.vel[2]);
        let mut neutral = deg < 0.0;
        neutral |= false;
        for f in 0..30u32 {
            let h = s.vel[0].atan2(s.vel[2]) + deg.to_radians();
            let (sx, sy) = moves::stick_dir(&s, h);
            s = lab.step(f, if neutral { GamePad::new(0, 0, 0) } else { GamePad::new(0, sx, sy) })?;
        }
        let v1 = (s.vel[0] * s.vel[0] + s.vel[2] * s.vel[2]).sqrt();
        let turn = (s.vel[0].atan2(s.vel[2]) - h0).to_degrees();
        let turn = (turn + 540.0) % 360.0 - 180.0;
        println!("slide stick {deg}° off velocity, 30 frames: speed {v0:.1} -> {v1:.1} ({:.3}) turned {turn:.1}° act {:#x}", v1 / v0.max(1e-3), s.action);
        println!("json {{\"slide_angle\":{{\"deg\":{deg},\"v0\":{v0:.2},\"v1\":{v1:.2},\"turn_deg\":{turn:.2}}}}}");
    }
    {
        let mut s = lab.reset()?;
        let v0 = (s.vel[0] * s.vel[0] + s.vel[2] * s.vel[2]).sqrt();
        for f in 0..30u32 {
            s = lab.step(f, GamePad::new(0, 0, 0))?;
        }
        let v1 = (s.vel[0] * s.vel[0] + s.vel[2] * s.vel[2]).sqrt();
        println!("slide neutral stick 30 frames: speed {v0:.1} -> {v1:.1} act {:#x}", s.action);
    }
    // Form switch: A (butt slide jump, after 5 frames) then B in the air
    // (dive) -> a dive slide; slide-course before and after.
    {
        let mut s = lab.reset()?;
        let (mut hit, mut phase, mut switched) = (None, 0, None);
        for f in 0..n {
            let base = slide::slide_pad(slide::Slide::Course, &s, target, f as usize).unwrap_or_else(|| obrrt_search::steer::steer(&obj, &s, f as usize, 8));
            let p = match phase {
                0 if s.action == slide::ACT_BUTT_SLIDE && s.action_timer >= 5 && f % 2 == 0 => {
                    phase = 1;
                    GamePad::new(A_BUTTON, base.stick_x, base.stick_y)
                }
                1 if s.action & 0x800 != 0 && f % 2 == 0 => {
                    phase = 2;
                    GamePad::new(B_BUTTON, base.stick_x, base.stick_y)
                }
                _ => base,
            };
            s = lab.step(f, p)?;
            if phase == 2 && switched.is_none() && slide::slide_action(s.action) {
                switched = Some((f + 1, s.action, (s.vel[0] * s.vel[0] + s.vel[2] * s.vel[2]).sqrt()));
            }
            if dist(s.pos, target) < 300.0 {
                hit = Some(f + 1);
                break;
            }
        }
        println!("slide switch to stomach (A, B): back on a slide {switched:?}, to target {hit:?} end act {:#x}", s.action);
    }
    Ok(())
}

struct Target([f32; 3]);
impl obrrt_core::Objective for Target {
    fn kind(&self) -> &'static str {
        "Modes::Lab"
    }
    fn key(&self) -> String {
        String::new()
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
}

fn moves_target(t: [f32; 3]) -> Target {
    Target(t)
}

// ---------------------------------------------------------------------------

fn pole_cmd(lab: &mut Lab, rest: &[String]) -> Result<()> {
    if !pole::on_pole(&lab.s0) {
        bail!("not on a pole");
    }
    use obrrt_core::layout::{ACT_CLIMBING_POLE, ACT_HOLDING_POLE, ACT_TOP_OF_POLE};
    for (name, every) in [("full", 0u32), ("release-1/4", 4), ("release-1/8", 8), ("release-1/16", 16)] {
        let mut s = lab.reset()?;
        let y0 = s.pos[1];
        let (mut top, mut peak) = (None, (y0, 0u32));
        for f in 0..400u32 {
            let release = every > 0 && f % every == every - 1;
            s = lab.step(f, pad(0, 0.0, if release { 0.0 } else { 64.0 }))?;
            if s.pos[1] > peak.0 + 0.01 {
                peak = (s.pos[1], f + 1);
            }
            if s.action == ACT_TOP_OF_POLE {
                top = Some(f + 1);
                break;
            }
            if !matches!(s.action, ACT_HOLDING_POLE | ACT_CLIMBING_POLE) && s.action != 0x0010_0344 {
                break;
            }
        }
        let rate = (peak.0 - y0) / peak.1.max(1) as f32;
        println!("pole climb {name}: top (handstand) at {top:?}; rose {:.0} by frame {} ({rate:.2}/frame) act {:#x}", peak.0 - y0, peak.1, s.action);
        println!("json {{\"pole_climb\":{{\"pattern\":\"{name}\",\"top\":{},\"rise\":{:.1},\"rise_frames\":{},\"rate\":{rate:.3}}}}}", top.map_or(-1, |t| t as i64), peak.0 - y0, peak.1);
    }
    // Kick-off arc from here: hold still a frame (holding), A, then stick
    // forward (camera-relative along the facing) with A held.
    let mut s = lab.reset()?;
    let mut f = 0u32;
    while s.action != ACT_HOLDING_POLE && f < 60 {
        s = lab.step(f, pad(0, 0.0, 0.0))?;
        f += 1;
    }
    let o = s.pos;
    s = lab.step(f, GamePad::new(A_BUTTON, 0, 0))?;
    f += 1;
    let mut pts = Vec::new();
    for k in 1..80u32 {
        let face = s.yaw as f32 * std::f32::consts::TAU / 65536.0;
        let (sx, sy) = moves::stick_dir(&s, face);
        s = lab.step(f, GamePad::new(A_BUTTON, sx, sy))?;
        f += 1;
        let h = ((s.pos[0] - o[0]).powi(2) + (s.pos[2] - o[2]).powi(2)).sqrt();
        pts.push((k, h, s.pos[1] - o[1], s.forward_vel, s.action));
        if s.action & 0x800 == 0 {
            break;
        }
    }
    for (k, h, dy, fv, a) in pts.iter().filter(|p| p.0 % 5 == 0 || p.4 & 0x800 == 0) {
        println!("kick t {k}: out {h:.0} up {dy:.0} fv {fv:.1} act {a:#x}");
    }
    let js: Vec<String> = pts.iter().map(|(k, h, dy, fv, _)| format!("[{k},{h:.1},{dy:.1},{fv:.2}]")).collect();
    println!("json {{\"pole_kick\":{{\"t_out_up_fv\":[{}]}}}}", js.join(","));
    if let Some(t) = rest.first() {
        let target = parse3(t)?;
        for (name, k) in [("pole-top", pole::Pole::Top), ("pole-kick", pole::Pole::Kick), ("pole-drop", pole::Pole::Drop)] {
            let mut s = lab.reset()?;
            let mut hit = None;
            let obj = moves_target(target);
            for f in 0..600u32 {
                let p = pole::pole_pad(k, &s, target, f as usize).unwrap_or_else(|| obrrt_search::steer::steer(&obj, &s, f as usize, 8));
                s = lab.step(f, p)?;
                if dist(s.pos, target) < 100.0 {
                    hit = Some(f + 1);
                    break;
                }
            }
            println!("{name}: to target {hit:?} end {:.0?} act {:#x}", s.pos, s.action);
        }
    }
    Ok(())
}
