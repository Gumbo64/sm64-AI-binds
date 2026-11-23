//! Checks the skill library on the real emulator.
//!
//!   skill_check <rom> run   [problem=speed] [frames=120] [threads=3]   each skill alone from every case start
//!   skill_check <rom> match [problem=speed] [max_frames=3000]          keep_best vs the TAS/route pads on fast frames
//!   skill_check <rom> hand  [problem=speed] [frames=200] [threads=3]   a hand-written state->skill table
//!
//! The Sense is computed here from emulator memory (floor probes, wall rays,
//! edge, landing) because sense.rs belongs to another module; the stick/button
//! realisation is local too (menu::realize is not written yet).

use anyhow::Result;
use obrrt_core::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use obrrt_core::{Eta, GamePad, Mem, St, geo, read_state};
use obrrt_gp::lang::{Ctx, RefNow};
use obrrt_gp::problems::load;
use obrrt_gp::rollout::{Case, in_action};
use obrrt_policy::skills::{self, F, Skill, act::*};
use obrrt_policy::types::*;
use sm64_binds::{SM64Game, SM64GameGenerator};
use std::f32::consts::TAU;

fn yaw_of(dx: f32, dz: f32) -> i16 {
    ((dx.atan2(dz) * 65536.0 / TAU).round() as i64 as i32) as i16
}

// ----------------------------------------------------------- local Sense

fn probe_at(game: &SM64Game, s: &St, x: f32, z: f32, d: f32) -> FloorProbe {
    match game.with_memory(|m| geo::floor_at(m, x, s.pos[1] + 20.0, z)) {
        Some(h) => FloorProbe { dist: d, y: h.y, ny: h.normal[1], ty: h.kind as u16, class: 0, has_floor: true },
        None => FloorProbe { dist: d, has_floor: false, ..Default::default() },
    }
}

/// Ray (horizontal, at wall height) against the zone's static wall triangles.
fn wall_ray(walls: &[geo::Tri], o: [f32; 3], yaw: i16) -> (f32, i16) {
    let r = yaw as f32 * TAU / 65536.0;
    let d = [r.sin(), 0.0, r.cos()];
    let mut best = (f32::INFINITY, 0i16);
    for t in walls {
        // Moller-Trumbore
        let (a, b, c) = (t.v[0], t.v[1], t.v[2]);
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let p = [d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]];
        let det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
        if det.abs() < 1e-6 {
            continue;
        }
        let tv = [o[0] - a[0], o[1] - a[1], o[2] - a[2]];
        let u = (tv[0] * p[0] + tv[1] * p[1] + tv[2] * p[2]) / det;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = [tv[1] * e1[2] - tv[2] * e1[1], tv[2] * e1[0] - tv[0] * e1[2], tv[0] * e1[1] - tv[1] * e1[0]];
        let v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) / det;
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let dist = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) / det;
        if dist > 0.0 && dist < best.0 {
            best = (dist, yaw_of(t.n[0], t.n[2]));
        }
    }
    best
}

fn local_sense(game: &SM64Game, s: &St) -> Sense {
    let mut sn = Sense::default();
    let r = s.yaw as f32 * TAU / 65536.0;
    for (i, d) in [24.0f32, 48.0, 96.0, 192.0, 384.0, 768.0].iter().enumerate() {
        sn.ahead[i] = probe_at(game, s, s.pos[0] + r.sin() * d, s.pos[2] + r.cos() * d, *d);
    }
    let vr = if s.vel[0].hypot(s.vel[2]) > 0.5 { s.vel[0].atan2(s.vel[2]) } else { r };
    for (i, d) in [96.0f32, 384.0].iter().enumerate() {
        sn.vel_ahead[i] = probe_at(game, s, s.pos[0] + vr.sin() * d, s.pos[2] + vr.cos() * d, *d);
    }
    let o = [s.pos[0], s.pos[1] + 80.0, s.pos[2]];
    sn.wall = [f32::INFINITY; 4];
    if let Some(w) = geo::static_walls(s.zone()) {
        for (i, off) in [0i32, -0x4000, 0x4000, 0x8000].iter().enumerate() {
            let (d, n) = wall_ray(&w, o, (s.yaw as i32 + off) as i16);
            sn.wall[i] = d;
            if i == 0 {
                sn.wall_normal_yaw = n;
            }
        }
    }
    sn.ceil_clear = (s.ceil_y - s.pos[1]).max(1.0);
    sn.edge_ahead = f32::INFINITY;
    for p in sn.ahead.iter() {
        if !p.has_floor || p.y < s.floor_y - 100.0 {
            sn.edge_ahead = p.dist;
            break;
        }
    }
    sn.ledge_ahead = f32::INFINITY;
    // Ballistic landing under the current heading, from the probes.
    if s.action & 0x800 != 0 {
        let (mut y, mut vy, mut x, mut z) = (s.pos[1], s.vel[1], s.pos[0], s.pos[2]);
        let (vx, vz) = (s.vel[0], s.vel[2]);
        let mut dist = 0.0f32;
        let mut fy = s.floor_y;
        for t in 1..=90u32 {
            x += vx;
            z += vz;
            y += vy;
            vy = (vy - 4.0).max(-75.0);
            dist += vx.hypot(vz);
            let i = [24.0f32, 48.0, 96.0, 192.0, 384.0, 768.0].iter().position(|d| *d >= dist).unwrap_or(5);
            if sn.ahead[i].has_floor {
                fy = sn.ahead[i].y;
            }
            if y <= fy && vy < 0.0 {
                sn.land = Some(Landing { frames: t, pos: [x, fy, z], fv: s.forward_vel, probe: sn.ahead[i] });
                break;
            }
        }
    }
    for o in s.objects().iter().take(64) {
        let (dx, dz) = (o.pos[0] - s.pos[0], o.pos[2] - s.pos[2]);
        let name = o.bhv_name();
        let cat = if o.interact_type & 16 != 0 {
            skills::cat::COIN
        } else if o.interact_type & 64 != 0 {
            skills::cat::POLE
        } else if o.interact_type & 4096 != 0 {
            skills::cat::STAR
        } else if name.contains("Bomb") {
            skills::cat::BOMB
        } else if o.interact_type & 8 != 0 {
            skills::cat::ENEMY
        } else {
            continue;
        };
        sn.objs.push(ObjProbe { cat, dist: dx.hypot(dz), rel_yaw: (yaw_of(dx, dz) as i32 - s.yaw as i32) as i16, dy: o.pos[1] - s.pos[1], radius: 50.0 });
    }
    sn
}

// --------------------------------------------------------- local realize

fn base_yaw(c: &Choice, s: &St, pts0: [f32; 3]) -> i16 {
    match c.cell.stick.r {
        Ref::Target => yaw_of(pts0[0] - s.pos[0], pts0[2] - s.pos[2]),
        Ref::Face => s.yaw,
        Ref::Vel => if s.vel[0].hypot(s.vel[2]) > 0.5 { yaw_of(s.vel[0], s.vel[2]) } else { s.yaw },
        Ref::Downhill => yaw_of(s.floor_normal[0], s.floor_normal[2]),
        Ref::Uphill => yaw_of(-s.floor_normal[0], -s.floor_normal[2]),
        Ref::Cam => s.cam_yaw,
        Ref::World => c.yaw,
    }
}

fn realize_local(c: &Choice, s: &St, pts0: [f32; 3]) -> GamePad {
    let mut b = 0u16;
    if c.cell.btn.a() {
        b |= A_BUTTON;
    }
    if c.cell.btn.b() {
        b |= B_BUTTON;
    }
    if c.cell.btn.z() {
        b |= Z_TRIG;
    }
    let st = c.cell.stick;
    if st.mag == Mag::Neutral {
        return GamePad::new(b, 0, 0);
    }
    let off: i32 = match st.ang {
        Ang::Aim => (c.fine_ang * 0x400 as f32) as i32,
        Ang::Left => 0x1400 + (c.fine_ang * 0xC00 as f32) as i32,
        Ang::Right => -0x1400 + (c.fine_ang * 0xC00 as f32) as i32,
        Ang::SideLeft => 0x4000 + (c.fine_ang * 0x1000 as f32) as i32,
        Ang::SideRight => -0x4000 + (c.fine_ang * 0x1000 as f32) as i32,
        Ang::Back => 0x8000 + (c.fine_ang * 0x3000 as f32) as i32,
    };
    let yaw = (base_yaw(c, s, pts0) as i32 + off) as i16;
    let tilt = match st.mag {
        Mag::Neutral => 0.0,
        Mag::Short => 0.30 + 0.08 * c.fine_mag.clamp(0.0, 1.0),
        Mag::Full => 1.0,
    };
    let (x, y) = obrrt_gp::lang::stick(s.cam_yaw, yaw, tilt);
    GamePad::new(b, x, y)
}

// ------------------------------------------------------------ the runner

struct Run {
    fv0: f32,
    fv_end: f32,
    fv_min: f32,
    fv_mean: f32,
    frames: usize,
    alive: bool,
    done: Option<usize>,
    actions: Vec<u32>,
}

fn run_case(game: &mut SM64Game, case: &Case, task: &dyn obrrt_gp::rollout::Task, frames: usize, mut pick: impl FnMut(&F) -> usize, lib: &[Box<dyn Skill>], params: &[Vec<f32>]) -> Result<Run> {
    game.load_state(&case.save)?;
    for p in &case.noise {
        game.step_game(*p)?;
    }
    let mut traj = case.pre_states.clone();
    let mut played = case.pre_pads.clone();
    let mut mem = Mem::default();
    let start = case.start.clone();
    let base = case.frame as u32;
    let mut r = Run { fv0: start.forward_vel, fv_end: start.forward_vel, fv_min: start.forward_vel, fv_mean: 0.0, frames: 0, alive: true, done: None, actions: vec![] };
    let mut sum = 0.0;
    let horizon = task.horizon().unwrap_or(frames).min(frames);
    for t in 0..horizon {
        let cur = traj.last().unwrap().clone();
        let pts = task.points(&cur, &mem);
        let ctx = Ctx { s: &cur, last: played.last().copied().unwrap_or(GamePad::new(0, 0, 0)), in_act: in_action(&traj, &played), pts, pace: 0.0, rf: RefNow::default() };
        let sn = local_sense(game, &cur);
        let obs = Obs { ctx: &ctx, sense: &sn };
        let f = F::new(&obs);
        let k = pick(&f);
        let ch = lib[k].choose(&obs, &params[k]);
        let pad = realize_local(&ch, &cur, pts[0]);
        if std::env::var("TRACE").is_ok() {
            println!(
                "t{t:>3} act {:#010x} fv {:>7.2} vy {:>6.1} ny {:.3} n {:>2} -> {} btn {}{}{} stick ({:>4},{:>4}) cell {:?}/{:?}/{:?}",
                cur.action, cur.forward_vel, cur.vel[1], cur.floor_normal[1], ctx.in_act, lib[k].name(),
                if pad.button & A_BUTTON != 0 { "A" } else { "-" }, if pad.button & B_BUTTON != 0 { "B" } else { "-" }, if pad.button & Z_TRIG != 0 { "Z" } else { "-" },
                pad.stick_x, pad.stick_y, ch.cell.stick.mag, ch.cell.stick.ang, ch.cell.stick.r
            );
            println!("      pos ({:.0},{:.0},{:.0}) yaw {:#x} wallF {:.0} edge {:.0} floor {:?} walls_known {}", cur.pos[0], cur.pos[1], cur.pos[2], cur.yaw as u16, sn.wall[0], sn.edge_ahead, sn.ahead[3].has_floor, geo::static_walls(cur.zone()).is_some());
        }
        game.step_game(pad)?;
        played.push(pad);
        let st = read_state(game, base + t as u32 + 1, true);
        r.frames += 1;
        r.actions.push(cur.action);
        r.fv_end = st.forward_vel;
        r.fv_min = r.fv_min.min(st.forward_vel);
        sum += st.forward_vel;
        if !st.alive() {
            r.alive = false;
            break;
        }
        let eta = task.eta(&st, &start, &mut mem);
        if matches!(eta, Eta::Lost) {
            r.alive = false;
            break;
        }
        if eta.done() {
            r.done = Some(t + 1);
            traj.push(st);
            break;
        }
        traj.push(st);
    }
    r.fv_mean = sum / r.frames.max(1) as f32;
    Ok(r)
}

fn hand_table(lib: &[Box<dyn Skill>]) -> impl Fn(&F) -> usize + use<> {
    let idx = |name: &str| lib.iter().position(|s| s.name() == name).unwrap();
    let (kb, rc, kc, skc, ap) = (idx("keep_best"), idx("rollout_cancel"), idx("kick_chain"), idx("slide_kick_chain"), idx("zone_safe_hop"));
    move |f: &F| -> usize {
        match f.a {
            DIVE_SLIDE | STOMACH_SLIDE => rc,
            SLIDE_KICK_SLIDE | CROUCH_SLIDE | SLIDE_KICK => skc,
            FORWARD_ROLLOUT | FREEFALL_LAND | JUMP_LAND | FREEFALL_LAND_STOP | JUMP_KICK => kc,
            DOUBLE_JUMP_LAND | LONG_JUMP_LAND => ap,
            _ => kb,
        }
    }
}

fn median(v: &mut Vec<f32>) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Which actions a skill is meant for (applicability).
fn applies(name: &str, a: u32) -> bool {
    let air = a & 0x800 != 0;
    let land = matches!(a, JUMP_LAND | FREEFALL_LAND | DOUBLE_JUMP_LAND | LONG_JUMP_LAND | SIDE_FLIP_LAND | TRIPLE_JUMP_LAND | BACKFLIP_LAND | JUMP_LAND_STOP | FREEFALL_LAND_STOP | DOUBLE_JUMP_LAND_STOP | LONG_JUMP_LAND_STOP);
    let slide = matches!(a, DIVE_SLIDE | BUTT_SLIDE | STOMACH_SLIDE | SLIDE_KICK_SLIDE | CROUCH_SLIDE);
    match name {
        "keep_best" | "hold_speed" | "steer_target" | "zone_safe_hop" | "cap_aware" | "health_guard" | "carry" | "wait" | "exit_to" => true,
        "slide_optimise" | "rollout_cancel" | "butt_slide_chain" => slide || a == FORWARD_ROLLOUT,
        "kick_chain" => air || land || matches!(a, PUNCHING | MOVE_PUNCHING),
        "slide_kick_chain" => slide || a == SLIDE_KICK || a == FORWARD_ROLLOUT || land,
        "edge_dive_hop" => air || a == DIVE_SLIDE || a == WALKING,
        "wall_kick_chain" => air || a == AIR_HIT_WALL,
        "jump_chain" | "exit_jump" | "exit_dive" => land || air || a == WALKING || a == IDLE,
        "long_jump_builder" => a == LONG_JUMP || a == CROUCH_SLIDE || a == LONG_JUMP_LAND || a == WALKING,
        "ground_pound" => air,
        "brake_stop" | "slow_bleed" | "turn_around" => matches!(a, WALKING | DECELERATING | BRAKING | TURNING_AROUND) || slide || land,
        "ledge_grab" => air || a == LEDGE_GRAB,
        "pole" => matches!(a, GRAB_POLE_SLOW | GRAB_POLE_FAST | HOLDING_POLE | CLIMBING_POLE | TOP_OF_POLE) || a == WALKING,
        "swim_cadence" => a & 0x2000 != 0,
        "avoid_wall" | "avoid_edge" | "approach_object" | "avoid_object" => a != IDLE,
        _ => false,
    }
}

fn main() -> Result<()> {
    unsafe {
        std::env::set_var("OBRRT_LAKITU_ARC", "1");
        std::env::set_var("OBRRT_GROUNDS_FIX", "1");
    }
    obrrt_objectives::moves::install_controller();
    geo::want_static_floors();
    let args: Vec<String> = std::env::args().collect();
    let gg = SM64GameGenerator::from_file(&args[1])?;
    let mode = args.get(2).map_or("run", |s| s.as_str()).to_string();
    let pname = args.get(3).map_or("speed", |s| s.as_str()).to_string();
    let n: usize = args.get(4).map_or(Ok(if mode == "match" { 3000 } else { 120 }), |s| s.parse())?;
    let threads: usize = args.get(5).map_or(Ok(3), |s| s.parse())?;
    let pb = load(&pname, ".", &gg, false)?;
    let lib = skills::library();
    let params: Vec<Vec<f32>> = lib.iter().map(|s| s.defaults()).collect();
    println!("{} skills, problem {pname}: {} cases", lib.len(), pb.cases.len());
    let mut games: Vec<SM64Game> = (0..threads).map(|_| gg.create_game()).collect::<Result<_>>()?;

    match mode.as_str() {
        "run" => {
            // Each skill alone from every case.
            let nsk = lib.len();
            let total = pb.cases.len() * nsk;
            let res = obrrt_gp::rollout::par(&mut games, total, |g, i| {
                let (k, ci) = (i / pb.cases.len(), i % pb.cases.len());
                let c = &pb.cases[ci];
                run_case(g, c, pb.tasks[c.task].as_ref(), n, |_| k, &lib, &params)
            })?;
            // references: the movie's own pads
            println!("{:<18} {:>6} {:>7} {:>7} {:>7} {:>6} {:>6} {:>8}", "skill", "apply%", "keep", "min", "mean", "alive", "done", "(fv_end/fv0 median)");
            for k in 0..nsk {
                let rows: Vec<&Run> = (0..pb.cases.len()).map(|ci| &res[k * pb.cases.len() + ci]).collect();
                let tot: usize = rows.iter().map(|r| r.frames).sum();
                let app: usize = rows.iter().map(|r| r.actions.iter().filter(|a| applies(lib[k].name(), **a)).count()).sum();
                let mut keep: Vec<f32> = rows.iter().filter(|r| r.fv0 > 10.0).map(|r| r.fv_end / r.fv0).collect();
                let mut mins: Vec<f32> = rows.iter().filter(|r| r.fv0 > 10.0).map(|r| r.fv_min / r.fv0).collect();
                let mut means: Vec<f32> = rows.iter().filter(|r| r.fv0 > 10.0).map(|r| r.fv_mean / r.fv0).collect();
                let alive = rows.iter().filter(|r| r.alive).count();
                let done = rows.iter().filter(|r| r.done.is_some()).count();
                println!(
                    "{:<18} {:>5.0}% {:>7.2} {:>7.2} {:>7.2} {:>3}/{:<2} {:>3}/{:<2}",
                    lib[k].name(),
                    100.0 * app as f32 / tot.max(1) as f32,
                    median(&mut keep),
                    median(&mut mins),
                    median(&mut means),
                    alive,
                    rows.len(),
                    done,
                    rows.len()
                );
            }
        }
        "trace" => {
            let name = std::env::var("SK").unwrap_or("keep_best".into());
            let ci: usize = std::env::var("CASE").ok().and_then(|x| x.parse().ok()).unwrap_or(0);
            let hand = hand_table(&lib);
            let c = &pb.cases[ci];
            println!("case {ci} {} start fv {:.1} act {:#x}", c.name, c.start.forward_vel, c.start.action);
            let kk = lib.iter().position(|s| s.name() == name);
            let r = run_case(&mut games[0], c, pb.tasks[c.task].as_ref(), n, |f| kk.unwrap_or_else(|| hand(f)), &lib, &params)?;
            println!("end fv {:.1} alive {} done {:?}", r.fv_end, r.alive, r.done);
        }
        "ref" => {
            // the movie's own pads from each case start, over n frames
            let res = obrrt_gp::rollout::par(&mut games, pb.cases.len(), |g, ci| {
                let c = &pb.cases[ci];
                g.load_state(&c.save)?;
                for p in &c.noise { g.step_game(*p)?; }
                let mut mem = Mem::default();
                let (mut fv_min, mut sum, mut done, mut end) = (c.start.forward_vel, 0.0, None, c.start.forward_vel);
                for t in 0..n {
                    let pad = c.ref_pads.get(t).copied().unwrap_or(GamePad::new(0, 0, 0));
                    g.step_game(pad)?;
                    let st = read_state(g, c.frame as u32 + t as u32 + 1, true);
                    end = st.forward_vel; fv_min = fv_min.min(end); sum += end;
                    if std::env::var("TRACE").ok().and_then(|x| x.parse::<usize>().ok()) == Some(ci) {
                        println!("t{t:>3} act {:#010x} fv {:>7.2} pos ({:.0},{:.0},{:.0}) pad {}{}{} stick ({:>4},{:>4})", st.action, st.forward_vel, st.pos[0], st.pos[1], st.pos[2],
                            if pad.button & A_BUTTON != 0 { "A" } else { "-" }, if pad.button & B_BUTTON != 0 { "B" } else { "-" }, if pad.button & Z_TRIG != 0 { "Z" } else { "-" }, pad.stick_x, pad.stick_y);
                    }
                    let eta = pb.tasks[c.task].eta(&st, &c.start, &mut mem);
                    if eta.done() { done = Some(t + 1); break; }
                }
                Ok((c.start.forward_vel, end, fv_min, sum / n as f32, done))
            })?;
            let mut keep: Vec<f32> = res.iter().map(|r| r.1 / r.0).collect();
            let mut mins: Vec<f32> = res.iter().map(|r| r.2 / r.0).collect();
            println!("movie ref: keep {:.2} min {:.2} done {}/{}", median(&mut keep), median(&mut mins), res.iter().filter(|r| r.4.is_some()).count(), res.len());
            for (c, r) in pb.cases.iter().zip(&res) { println!("{:<44} fv0 {:>6.1} end {:>6.1} done {:?}", c.name, r.0, r.1, r.4); }
        }
        "hand" => {
            // A hand-written state -> skill table.
            let table = hand_table(&lib);
            let res = obrrt_gp::rollout::par(&mut games, pb.cases.len(), |g, ci| {
                let c = &pb.cases[ci];
                run_case(g, c, pb.tasks[c.task].as_ref(), n, &table, &lib, &params)
            })?;
            for (c, r) in pb.cases.iter().zip(&res) {
                println!("{:<4} {:<44} fv0 {:>6.1} end {:>6.1} min {:>6.1} mean {:>6.1} frames {:>4} alive {} done {}", if c.train { "tr" } else { "TE" }, c.name, r.fv0, r.fv_end, r.fv_min, r.fv_mean, r.frames, r.alive, r.done.map_or("-".into(), |d| d.to_string()));
            }
            let succ = res.iter().filter(|r| r.done.is_some()).count();
            println!("hand table: done {succ}/{}", res.len());
        }
        "match" => {
            // keep_best vs the pads of the movies on fast frames.
            let kb = lib.iter().position(|s| s.name() == "keep_best").unwrap();
            let mut stats: std::collections::BTreeMap<u32, (u32, u32, u32, u32)> = Default::default();
            let g = &mut games[0];
            for (mi, (mname, pads)) in pb.movies.iter().enumerate() {
                // replay the movie from power-on: states come from the game
                g.load_state(&pb.cases.iter().find(|c| c.movie == mi).map(|c| c.save.clone()).unwrap())?;
                let first = pb.cases.iter().filter(|c| c.movie == mi).map(|c| c.frame - c.noise.len()).min().unwrap_or(0);
                let mut played: Vec<GamePad> = vec![GamePad::new(0, 0, 0)];
                let mut traj: Vec<St> = vec![read_state(g, first as u32, true)];
                let mut seen = 0u32;
                for (t, pad) in pads.iter().enumerate().skip(first).take(n) {
                    let cur = traj.last().unwrap().clone();
                    if cur.forward_vel >= 55.0 && cur.controllable() {
                        let ctx = Ctx { s: &cur, last: *played.last().unwrap(), in_act: in_action(&traj, &played), pts: [[0.0; 3]; obrrt_gp::lang::MAX_POINTS], pace: 0.0, rf: RefNow::default() };
                        let sn = local_sense(g, &cur);
                        let obs = Obs { ctx: &ctx, sense: &sn };
                        let ch = lib[kb].choose(&obs, &params[kb]);
                        let mine = realize_local(&ch, &cur, [0.0; 3]);
                        let e = stats.entry(cur.action).or_default();
                        e.0 += 1;
                        let (ma, mb, mz) = (mine.button & A_BUTTON != 0, mine.button & B_BUTTON != 0, mine.button & Z_TRIG != 0);
                        let (ta, tb, tz) = (pad.button & A_BUTTON != 0, pad.button & B_BUTTON != 0, pad.button & Z_TRIG != 0);
                        e.1 += (ma == ta && mb == tb && mz == tz) as u32;
                        // stick class: both neutral/short/full
                        let mag = |p: &GamePad| ((p.stick_x as f32).hypot(p.stick_y as f32) * 64.0 / 80.0).min(64.0);
                        let cls = |m: f32| if m < 4.0 { 0 } else if m <= 48.0 { 1 } else { 2 };
                        e.2 += (cls(mag(&mine)) == cls(mag(pad))) as u32;
                        e.3 += 1;
                        seen += 1;
                    }
                    g.step_game(*pad)?;
                    played.push(*pad);
                    traj.push(read_state(g, (t + 1) as u32, true));
                    if traj.len() > 40 {
                        traj.remove(0);
                        if played.len() > 40 {
                            played.remove(0);
                        }
                    }
                }
                println!("movie {mname}: {seen} fast frames");
            }
            println!("{:<26} {:>6} {:>10} {:>12}", "action", "frames", "btn match", "stick class");
            let nm = |a: u32| format!("{a:#010x}");
            for (a, (nn, bm, sm, _)) in &stats {
                println!("{:<26} {:>6} {:>9.0}% {:>11.0}%", nm(*a), nn, 100.0 * *bm as f32 / *nn as f32, 100.0 * *sm as f32 / *nn as f32);
            }
        }
        m => anyhow::bail!("mode {m}"),
    }
    Ok(())
}
