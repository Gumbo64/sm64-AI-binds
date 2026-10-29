//! Arc lab: measures Mario's moves (jump arcs, wall kicks, slide kicks,
//! slides, ground pounds, falls) on the castle grounds, writes them as the
//! predictor's table, and checks the approach planner of
//! objectives/src/moves/arcs.rs against the game.
//!
//! arc_lab <rom> probe                  boot; print Mario, doors, long flat rays
//! arc_lab <rom> column x0 x1 z0 z1 d   static floor heights on a grid
//! arc_lab <rom> walls                  run into walls from a few starts; print them
//! arc_lab <rom> slopes                 static floor triangles by steepness
//! arc_lab <rom> measure                measure everything; write ref/arcs/arcs.json
//!                                      and ref/arcs/arcs_table.rs (the predictor's table)
//! arc_lab <rom> validate               predict() vs the game at speeds not in the table
//! arc_lab <rom> door [trace]           plan_approach vs running straight, from
//!                                      starts 800-2500 from castle doors; per-move
//!                                      landing errors
//!
//! Boot: the fixed menu-skipping inputs (as `obrrt`), then idle until still,
//! in the castle grounds. Jumps are measured on the grounds' flat y=260
//! plateau (Mario teleported to the start of a ~4950-unit flat ray).
//! Everything is deterministic: rerunning `measure` regenerates the table.
#[path = "../../objectives/src/moves/arcs.rs"]
#[allow(dead_code)]
mod arcs;

use anyhow::{Result, bail};
use arcs::{AirStick, ArcKind};
use obrrt_core::buttons::{A_BUTTON, B_BUTTON, START_BUTTON, Z_TRIG};
use obrrt_core::layout::*;
use obrrt_core::{GamePad, St, read_state};
use serde_json::{Value, json};
use sm64_binds::{SM64Game, SM64GameGenerator, SaveState};
use std::fmt::Write as _;

const M: usize = ADDR_gMarioStates as usize;
const AIR: u32 = 0x800;
const ACT_WALKING: u32 = 0x0400_0440;
const ACT_TURNING_AROUND: u32 = 0x0000_0443;
const ACT_DIVE_SLIDE: u32 = 0x0088_0456;
const ACT_AIR_HIT_WALL: u32 = 0x0000_08A7;
const ACT_WALL_KICK: u32 = 0x0300_0886;
const ACT_LONG_JUMP_LAND: u32 = 0x0000_0479;
const ACT_IDLE_: u32 = 0x0C40_0201;
const ACT_FREEFALL: u32 = 0x0100_088C;
const ACT_CROUCH_SLIDE: u32 = 0x0480_8459;
const ACT_SLIDE_KICK: u32 = 0x0180_08AA;
const ACT_SLIDE_KICK_SLIDE: u32 = 0x0080_045A;
const ACT_GROUND_POUND: u32 = 0x0080_08A9;

/// The plateau runway: start, yaw (0x2000: +x +z).
const RUN_START: [f32; 3] = [-5800.0, 260.0, 1150.0];
/// Mid-plateau start for moves that travel backwards (backflip, side flip)
/// or sideways.
const RUN_MID: [f32; 3] = [-3900.0, 260.0, 2950.0];
const RUN_YAW: i32 = 0x2000;

pub fn act_name(a: u32) -> String {
    let n = match a {
        0x0C40_0201 => "IDLE",
        0x0400_0440 => "WALKING",
        0x0400_044A => "DECELERATING",
        0x0400_0445 => "BRAKING",
        0x0000_0443 => "TURNING_AROUND",
        0x0000_0444 => "FINISH_TURNING_AROUND",
        0x0480_8459 => "CROUCH_SLIDE",
        0x0C00_8220 => "CROUCHING",
        0x0C00_8221 => "START_CROUCHING",
        0x0300_0880 => "JUMP",
        0x0300_0881 => "DOUBLE_JUMP",
        0x0100_0882 => "TRIPLE_JUMP",
        0x0100_0883 => "BACKFLIP",
        0x0100_0887 => "SIDE_FLIP",
        0x0300_0888 => "LONG_JUMP",
        0x0188_088A => "DIVE",
        0x0100_088C => "FREEFALL",
        0x0100_08A6 => "FORWARD_ROLLOUT",
        0x0000_08A7 => "AIR_HIT_WALL",
        0x0080_08A9 => "GROUND_POUND",
        0x0180_08AA => "SLIDE_KICK",
        0x0180_08AC => "JUMP_KICK",
        0x0300_0886 => "WALL_KICK_AIR",
        0x0400_0470 => "JUMP_LAND",
        0x0400_0471 => "FREEFALL_LAND",
        0x0400_0472 => "DOUBLE_JUMP_LAND",
        0x0400_0473 => "SIDE_FLIP_LAND",
        0x0400_0478 => "TRIPLE_JUMP_LAND",
        0x0000_0479 => "LONG_JUMP_LAND",
        0x0400_047A => "BACKFLIP_LAND",
        0x0080_023C => "GROUND_POUND_LAND",
        0x0088_0456 => "DIVE_SLIDE",
        0x0080_045A => "SLIDE_KICK_SLIDE",
        0x0084_0452 => "BUTT_SLIDE",
        0x008C_0453 => "STOMACH_SLIDE",
        0x0080_0457 => "MOVE_PUNCHING",
        0x0000_08A8 => "SOFT_BONK",
        0x0100_08A4 => "BACKWARD_AIR_KB",
        _ => return format!("{a:#x}"),
    };
    n.to_string()
}

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

/// Power-on to the grounds' first still, controllable frame.
fn boot(gg: &SM64GameGenerator) -> Result<(SM64Game, St)> {
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
    let st = read_state(&g, f, true);
    Ok((g, st))
}

fn write(g: &mut SM64Game, addr: usize, bytes: &[u8]) -> Result<()> {
    let (store, _, memory) = g.wasm_parts();
    memory.view(store).write(addr as u64, bytes)?;
    Ok(())
}

fn iyaw(g: &SM64Game) -> i16 {
    g.with_memory(|m| i16::from_le_bytes(m[M + 0x24..M + 0x26].try_into().unwrap()))
}

fn wrap(a: i32) -> i32 {
    a as i16 as i32
}

fn rad(yaw: i32) -> f32 {
    yaw as f32 * std::f32::consts::TAU / 65536.0
}

fn r1(v: f32) -> f32 {
    (v * 10.0).round() / 10.0
}
fn r2(v: f32) -> f32 {
    (v * 100.0).round() / 100.0
}

/// A game being driven: the stick is given as a world yaw; the offset
/// between the read camera yaw and the one the game turns the stick by is
/// learned from `intendedYaw` each frame.
struct Sim {
    g: SM64Game,
    f: u32,
    st: St,
    bias: i32,
}

impl Sim {
    fn new(gg: &SM64GameGenerator, save: &SaveState, st: &St) -> Result<Sim> {
        let mut g = gg.create_game()?;
        g.load_state(save)?;
        Ok(Sim { g, f: st.frame, st: st.clone(), bias: 0 })
    }

    fn step(&mut self, yaw: Option<i32>, tilt: f32, btn: u16) -> Result<()> {
        let (x, y) = match yaw {
            Some(w) => arcs::stick(self.st.cam_yaw, (w + self.bias) as i16, tilt),
            None => (0, 0),
        };
        self.pad(GamePad::new(btn, x, y))?;
        if let Some(w) = yaw
            && tilt > 0.3
        {
            let err = wrap(iyaw(&self.g) as i32 - w);
            if err.abs() < 0x2000 {
                self.bias -= err / 2;
            }
        }
        Ok(())
    }

    fn pad(&mut self, p: GamePad) -> Result<()> {
        self.g.step_game(p)?;
        self.f += 1;
        self.st = read_state(&self.g, self.f, false);
        Ok(())
    }

    /// Teleports Mario (idle, still) to `p` facing `yaw`, then idles until
    /// the camera settles.
    fn teleport(&mut self, p: [f32; 3], yaw: i32) -> Result<()> {
        for _ in 0..4 {
            self.put(p, yaw, ACT_IDLE_)?;
            self.pad(GamePad::new(0, 0, 0))?;
        }
        for _ in 0..45 {
            self.pad(GamePad::new(0, 0, 0))?;
        }
        Ok(())
    }

    fn put(&mut self, p: [f32; 3], yaw: i32, action: u32) -> Result<()> {
        write(&mut self.g, M + 0x3C, &[p[0].to_le_bytes(), p[1].to_le_bytes(), p[2].to_le_bytes()].concat())?;
        write(&mut self.g, M + 0x48, &[0f32.to_le_bytes(); 3].concat())?;
        write(&mut self.g, M + 0x54, &0f32.to_le_bytes())?;
        write(&mut self.g, M + 0x2E, &(yaw as i16).to_le_bytes())?;
        write(&mut self.g, M + 0x0C, &action.to_le_bytes())?;
        Ok(())
    }

    fn air(&self) -> bool {
        self.st.action & AIR != 0
    }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        bail!("usage: arc_lab <rom> probe|column|walls|slopes|measure|validate|door [args]");
    }
    let gg = SM64GameGenerator::from_file(&a[1])?;
    let (g, st) = boot(&gg)?;
    match a[2].as_str() {
        "probe" => probe(&g, &st),
        "column" => {
            let v: Vec<f32> = a[3..8].iter().map(|s| s.parse().unwrap()).collect();
            let (tris, _) = g.with_memory(obrrt_core::geo::read_floors);
            let mut z = v[2];
            while z <= v[3] {
                let mut row = format!("z {z:6.0}:");
                let mut x = v[0];
                while x <= v[1] {
                    let y = obrrt_core::geo::floor_below(&tris, x, z, 3000.0);
                    row += &y.map_or("    --".into(), |y| format!(" {y:5.0}"));
                    x += v[4];
                }
                println!("{row}");
                z += v[4];
            }
        }
        "walls" => walls(&gg, &g, &st)?,
        "wallone" => {
            let save0 = g.save_state();
            let kind = if a[3] == "lj" { ArcKind::LongJump } else { ArcKind::Single };
            let m = wall_kick_one(&gg, &save0, &st, kind, a[4].parse()?, a[5].parse()?, a[6].parse()?)?;
            println!("{}", m.json());
        }
        "slopes" => slopes(&g),
        "measure" => measure_all(&gg, &g, &st)?,
        "validate" => validate(&gg, &g, &st)?,
        "door" => door_test(&gg, &g, &st, a.get(3).is_some_and(|s| s == "trace"))?,
        "chain" => {
            let (save, st0) = runway(&gg, &g, &st)?;
            println!("{}", chain(&gg, &save, &st0, &a[3], a[4].parse()?)?);
        }
        m => bail!("mode {m}?"),
    }
    Ok(())
}

fn probe(g: &SM64Game, st: &St) {
    println!(
        "frame {} pos {:?} yaw {:#06x} cam {:#06x} act {:#x} level {} area {} floor {} n {:?} water {}",
        st.frame, st.pos, st.yaw as u16, st.cam_yaw as u16, st.action, st.level, st.area, st.floor_y, st.floor_normal, st.water_level
    );
    for o in st.objects() {
        if o.interact_type & (INTERACT_DOOR | INTERACT_WARP_DOOR | INTERACT_WARP) != 0 || o.bhv_name().contains("Door") {
            println!("  {} slot {} pos {:?} yaw {:#06x} it {:#x} bp {:#010x}", o.bhv_name(), o.slot, o.pos, o.yaw as u16, o.interact_type, o.beh_params);
        }
    }
    let (tris, _) = g.with_memory(obrrt_core::geo::read_floors);
    println!("{} floor tris", tris.len());
    let mut best = Vec::new();
    for ix in -40..=40 {
        for iz in -40..=40 {
            let (x, z) = (ix as f32 * 200.0, iz as f32 * 200.0);
            let Some(y0) = obrrt_core::geo::floor_below(&tris, x, z, 20000.0) else { continue };
            for k in 0..16 {
                let ang = k as f32 * std::f32::consts::TAU / 16.0;
                let mut len = 0.0;
                let mut d = 50.0;
                while d < 12000.0 {
                    let (px, pz) = (x + d * ang.sin(), z + d * ang.cos());
                    match obrrt_core::geo::floor_below(&tris, px, pz, y0 + 30.0) {
                        Some(y) if (y - y0).abs() < 2.0 => len = d,
                        _ => break,
                    }
                    d += 50.0;
                }
                best.push((len, x, y0, z, k));
            }
        }
    }
    best.sort_by(|a, b| b.0.total_cmp(&a.0));
    for b in best.iter().filter(|b| b.2 > 0.0).take(10) {
        println!("  flat ray {:.0} from ({:.0}, {:.0}, {:.0}) dir {}/16", b.0, b.1, b.2, b.3, b.4);
    }
}

/// Runs into walls from a few starts in 16 directions; prints where Mario
/// stopped against one, its normal, and the floor.
fn walls(gg: &SM64GameGenerator, g: &SM64Game, st: &St) -> Result<()> {
    let save = g.save_state();
    let starts = [[0.0, 803.0, -2500.0], [-1328.0, 260.0, 4354.0], [-3500.0, 260.0, 2500.0], [0.0, 906.0, -1200.0], [-4000.0, 260.0, 4000.0]];
    for s in starts {
        for k in 0..16 {
            let yaw = k * 0x1000;
            let mut sim = Sim::new(gg, &save, st)?;
            sim.teleport(s, yaw)?;
            let mut n = 0;
            while n < 400 {
                sim.step(Some(yaw), 1.0, 0)?;
                n += 1;
                if n > 20 && sim.st.forward_vel < 4.0 {
                    break;
                }
            }
            if let Some(w) = sim.st.wall_normal
                && n < 400
            {
                println!(
                    "start {:?} yaw {:#06x}: stop {:.0},{:.0},{:.0} after {n} wall n {:.2},{:.2},{:.2} act {} floor {:.0} water {}",
                    s, yaw, sim.st.pos[0], sim.st.pos[1], sim.st.pos[2], w[0], w[1], w[2], act_name(sim.st.action), sim.st.floor_y, sim.st.water_level
                );
            }
        }
    }
    Ok(())
}

/// The area's static floor triangles by steepness (large ones).
fn slopes(g: &SM64Game) {
    let (tris, _) = g.with_memory(obrrt_core::geo::read_floors);
    let mut v: Vec<_> = tris
        .iter()
        .map(|t| {
            let [a, b, c] = t.v;
            let e1 = [b[0] - a[0], b[2] - a[2]];
            let e2 = [c[0] - a[0], c[2] - a[2]];
            let area = (e1[0] * e2[1] - e1[1] * e2[0]).abs() / 2.0;
            let cen = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0];
            (t.n[1], area, cen, t.kind, t.n)
        })
        .filter(|t| t.0 < 0.99 && t.0 > 0.3 && t.1 > 40000.0)
        .collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    for t in v {
        println!("ny {:.3} area {:>8.0} centroid {:.0},{:.0},{:.0} type {:#x} n {:.2},{:.2}", t.0, t.1, t.2[0], t.2[1], t.2[2], t.3, t.4[0], t.4[2]);
    }
}

// ===================================================================== jumps

#[derive(Clone, Copy, Debug)]
struct Spec {
    kind: ArcKind,
    v: f32,
    held: u8,
    stick: AirStick,
    chain: u8,
}

/// One measured move: the trajectory from the state before its first input
/// (index 0) to `walk` (first ACT_WALKING after landing) or the end.
struct Meas {
    spec: Spec,
    traj: Vec<St>,
    travel: i32,
    prep: usize,
    takeoff: usize,
    land: usize,
    walk: Option<usize>,
    takeoff_action: u32,
}

impl Meas {
    /// Landed on the runway's own height without touching a wall.
    fn valid(&self) -> bool {
        self.rel(self.land).2.abs() < 2.0 && self.traj.iter().all(|s| !matches!(s.action, ACT_AIR_HIT_WALL | 0x0000_08A8 | 0x0100_08A4))
    }
    fn rel(&self, i: usize) -> (f32, f32, f32) {
        rel(&self.traj[0], &self.traj[i], self.travel)
    }

    fn json(&self) -> Value {
        let n = self.land;
        let along: Vec<f32> = (1..=n).map(|i| r1(self.rel(i).0)).collect();
        let height: Vec<f32> = (1..=n).map(|i| r1(self.rel(i).2)).collect();
        let lat: Vec<f32> = (1..=n).map(|i| r1(self.rel(i).1)).collect();
        let (peak_i, peak) = height.iter().enumerate().fold((0, f32::MIN), |b, (i, &h)| if h > b.1 { (i + 1, h) } else { b });
        let land = self.rel(n);
        let ls = &self.traj[n];
        let (lag, lag_dist, walk_speed) = match self.walk {
            Some(w) => (w - n, self.rel(w).0 - land.0, self.traj[w].forward_vel),
            None => (255, 0.0, 0.0),
        };
        let yaw_change = wrap(self.traj[n].yaw as i32 - self.traj[self.takeoff].yaw as i32);
        // Slide kicks: the bounce (vy turning from falling to rising in SLIDE_KICK).
        let bounce = (1..self.traj.len())
            .find(|&i| self.traj[i].action == ACT_SLIDE_KICK && self.traj[i - 1].action == ACT_SLIDE_KICK && self.traj[i - 1].vel[1] < 0.0 && self.traj[i].vel[1] > 0.0);
        json!({
            "kind": format!("{:?}", self.spec.kind), "stick": format!("{:?}", self.spec.stick), "chain": self.spec.chain,
            "held": self.spec.held, "v_in": r2(self.traj[0].forward_vel),
            "v_take": r2(self.traj[self.takeoff].forward_vel), "vy_take": r2(self.traj[self.takeoff].vel[1]),
            "takeoff_action": format!("{:#x}", self.takeoff_action), "takeoff_name": act_name(self.takeoff_action),
            "prep": self.prep, "frames": n, "land_dist": r1(land.0), "land_lat": r1(land.1), "land_dy": r1(land.2),
            "land_speed": r2(ls.forward_vel), "land_action": format!("{:#x}", ls.action), "land_name": act_name(ls.action),
            "lag": lag, "lag_dist": r1(lag_dist), "walk_speed": r2(walk_speed),
            "peak": peak, "peak_frame": peak_i, "yaw_change": yaw_change,
            "bounce_frame": bounce, "bounce_vy": bounce.map(|i| r2(self.traj[i].vel[1])),
            "actions": self.traj[1..=n].iter().map(|s| act_name(s.action)).collect::<Vec<_>>(),
            "along": along, "height": height, "lat": lat,
        })
    }
}

/// (along, lateral (right), height) of `p` from `o` along world yaw `travel`.
fn rel(o: &St, p: &St, travel: i32) -> (f32, f32, f32) {
    let (s, c) = rad(travel).sin_cos();
    let (dx, dz) = (p.pos[0] - o.pos[0], p.pos[2] - o.pos[2]);
    (dx * s + dz * c, dx * c - dz * s, p.pos[1] - o.pos[1])
}

/// Runs at full tilt along `yaw` until forward speed reaches `v`.
fn run_to(sim: &mut Sim, yaw: i32, v: f32) -> Result<()> {
    let mut n = 0;
    while sim.st.forward_vel < v - 0.01 && n < 200 {
        sim.step(Some(yaw), 1.0, 0)?;
        n += 1;
    }
    Ok(())
}

fn stick_yaw(stick: AirStick, travel: i32) -> Option<i32> {
    match stick {
        AirStick::Fwd => Some(travel),
        AirStick::Back => Some(travel + 0x8000),
        AirStick::Side => Some(travel + 0x4000),
        AirStick::Neutral => None,
    }
}

/// Plays one airborne phase until Mario is on the ground: A held for the
/// first `held` frames (counting the takeoff frame, already played), B on
/// air frame `b_at` (if any), stick per `stick`.
fn fly(sim: &mut Sim, traj: &mut Vec<St>, travel: i32, held: u8, stick: AirStick, b_at: Option<usize>) -> Result<()> {
    let mut k = 1usize;
    while sim.air() && k < 200 {
        let mut btn = if (k as u8) < held { A_BUTTON } else { 0 };
        if b_at == Some(k) {
            btn |= B_BUTTON;
        }
        sim.step(stick_yaw(stick, travel), 1.0, btn)?;
        traj.push(sim.st.clone());
        k += 1;
    }
    Ok(())
}

/// After landing: stick along `travel`, no buttons, until ACT_WALKING.
fn settle(sim: &mut Sim, traj: &mut Vec<St>, travel: i32) -> Result<Option<usize>> {
    for _ in 0..90 {
        if sim.st.action == ACT_WALKING {
            return Ok(Some(traj.len() - 1));
        }
        sim.step(Some(travel), 1.0, 0)?;
        traj.push(sim.st.clone());
    }
    Ok(None)
}

/// Flies with A held while rising, released when falling.
fn fly_rising(sim: &mut Sim, yaw: i32) -> Result<()> {
    let mut n = 0;
    while sim.air() && n < 200 {
        let b = if sim.st.vel[1] > 0.0 { A_BUTTON } else { 0 };
        sim.step(Some(yaw), 1.0, b)?;
        n += 1;
    }
    Ok(())
}

/// Measures one move from the runway start.
fn measure(sim: &mut Sim, spec: Spec, yaw: i32) -> Result<Option<Meas>> {
    use ArcKind as K;
    let mut travel = yaw;
    // Lead-in: speed, earlier jumps of the chain.
    match spec.kind {
        K::Backflip => {}
        K::Double => {
            run_to(sim, yaw, spec.v)?;
            sim.step(Some(yaw), 1.0, A_BUTTON)?;
            fly_rising(sim, yaw)?;
        }
        K::Triple => {
            run_to(sim, yaw, spec.v)?;
            sim.step(Some(yaw), 1.0, A_BUTTON)?;
            fly_rising(sim, yaw)?;
            sim.step(Some(yaw), 1.0, A_BUTTON)?;
            fly_rising(sim, yaw)?;
        }
        K::LongJump => {
            run_to(sim, yaw, spec.v)?;
            for _ in 0..spec.chain {
                if sim.st.action != ACT_LONG_JUMP_LAND {
                    sim.step(Some(yaw), 1.0, Z_TRIG)?;
                }
                sim.step(Some(yaw), 1.0, Z_TRIG | A_BUTTON)?;
                let mut n = 0;
                while sim.air() && n < 200 {
                    sim.step(Some(yaw), 1.0, Z_TRIG)?;
                    n += 1;
                }
                if sim.st.action != ACT_LONG_JUMP_LAND {
                    return Ok(None);
                }
            }
        }
        _ => run_to(sim, yaw, spec.v)?,
    }
    let mut traj = vec![sim.st.clone()];
    let mut b_at = None;
    match spec.kind {
        K::Single | K::Double | K::Triple | K::JumpKick => {
            sim.step(Some(yaw), 1.0, A_BUTTON)?;
            if spec.kind == K::JumpKick {
                b_at = Some(spec.held as usize);
            }
        }
        K::LongJump => {
            if sim.st.action != ACT_LONG_JUMP_LAND {
                sim.step(Some(yaw), 1.0, Z_TRIG)?;
                traj.push(sim.st.clone());
            }
            sim.step(Some(yaw), 1.0, Z_TRIG | A_BUTTON)?;
        }
        K::Backflip => {
            travel = yaw + 0x8000;
            sim.step(None, 0.0, Z_TRIG)?;
            traj.push(sim.st.clone());
            sim.step(None, 0.0, Z_TRIG | A_BUTTON)?;
        }
        K::Sideflip => {
            travel = yaw + 0x8000;
            let mut n = 0;
            while sim.st.action != ACT_TURNING_AROUND && n < 10 {
                sim.step(Some(travel), 1.0, 0)?;
                traj.push(sim.st.clone());
                n += 1;
            }
            sim.step(Some(travel), 1.0, A_BUTTON)?;
        }
        K::Dive | K::DiveRollout => sim.step(Some(yaw), 1.0, B_BUTTON)?,
        K::SlideKick | K::SlideKickHop => {
            // Z (crouch slide), Z held, then Z+B.
            for _ in 0..spec.chain.max(1) {
                sim.step(Some(yaw), 1.0, Z_TRIG)?;
                traj.push(sim.st.clone());
            }
            sim.step(Some(yaw), 1.0, Z_TRIG | B_BUTTON)?;
        }
        K::WallKick => unreachable!("wall kicks: wall_kicks()"),
    }
    traj.push(sim.st.clone());
    if !sim.air() {
        return Ok(None);
    }
    let takeoff = traj.len() - 1;
    let takeoff_action = sim.st.action;
    let prep = takeoff - 1;
    let held = match spec.kind {
        K::LongJump => 1,
        K::JumpKick => 255,
        K::SlideKick | K::SlideKickHop => 0,
        _ => spec.held,
    };
    fly(sim, &mut traj, travel, held, spec.stick, b_at)?;
    match spec.kind {
        K::DiveRollout => {
            if sim.st.action != ACT_DIVE_SLIDE {
                return Ok(None);
            }
            sim.step(Some(travel), 1.0, A_BUTTON)?;
            traj.push(sim.st.clone());
            fly(sim, &mut traj, travel, 1, spec.stick, None)?;
        }
        K::SlideKick => {
            // Ride the slide out.
            let mut n = 0;
            while sim.st.action == ACT_SLIDE_KICK_SLIDE && n < 120 {
                sim.step(stick_yaw(spec.stick, travel), 1.0, 0)?;
                traj.push(sim.st.clone());
                n += 1;
            }
        }
        K::SlideKickHop => {
            if sim.st.action != ACT_SLIDE_KICK_SLIDE {
                return Ok(None);
            }
            for _ in 1..spec.held.max(1) {
                sim.step(stick_yaw(spec.stick, travel), 1.0, 0)?;
                traj.push(sim.st.clone());
            }
            sim.step(stick_yaw(spec.stick, travel), 1.0, A_BUTTON)?;
            traj.push(sim.st.clone());
            fly(sim, &mut traj, travel, 1, spec.stick, None)?;
        }
        _ => {}
    }
    let land = traj.len() - 1;
    let walk = settle(sim, &mut traj, travel)?;
    Ok(Some(Meas { spec, traj, travel, prep, takeoff, land, walk, takeoff_action }))
}

fn runway(gg: &SM64GameGenerator, g: &SM64Game, st: &St) -> Result<(SaveState, St)> {
    runway_at(gg, g, st, RUN_START)
}

fn runway_at(gg: &SM64GameGenerator, g: &SM64Game, st: &St, p: [f32; 3]) -> Result<(SaveState, St)> {
    let mut sim = Sim { g: gg.copy_from(g)?, f: st.frame, st: st.clone(), bias: 0 };
    sim.teleport(p, RUN_YAW)?;
    Ok((sim.g.save_state(), sim.st.clone()))
}

fn all_specs() -> Vec<Spec> {
    use AirStick as S;
    use ArcKind as K;
    let mut v = Vec::new();
    let sp = |kind, v, held, stick, chain| Spec { kind, v, held, stick, chain };
    for &s in &[0.0, 8.0, 16.0, 24.0, 32.0] {
        for &h in &[1u8, 3, 6, 10, 255] {
            v.push(sp(K::Single, s, h, S::Fwd, 0));
        }
        for &st in &[S::Neutral, S::Back, S::Side] {
            v.push(sp(K::Single, s, 255, st, 0));
        }
        for &h in &[1u8, 6, 255] {
            v.push(sp(K::Double, s, h, S::Fwd, 0));
        }
        v.push(sp(K::Double, s, 255, S::Neutral, 0));
        v.push(sp(K::Double, s, 255, S::Back, 0));
        for &h in &[1u8, 255] {
            v.push(sp(K::Triple, s, h, S::Fwd, 0));
        }
        v.push(sp(K::Triple, s, 255, S::Neutral, 0));
        v.push(sp(K::Triple, s, 255, S::Back, 0));
        for &k in &[1u8, 4, 8] {
            v.push(sp(K::JumpKick, s, k, S::Fwd, 0));
        }
    }
    for &s in &[12.0, 16.0, 20.0, 24.0, 28.0, 32.0] {
        for &st in &[S::Fwd, S::Neutral, S::Back, S::Side] {
            v.push(sp(K::LongJump, s, 1, st, 0));
        }
    }
    for &s in &[16.0, 24.0, 32.0] {
        for c in 1..=3u8 {
            for &st in &[S::Fwd, S::Neutral, S::Back, S::Side] {
                v.push(sp(K::LongJump, s, 1, st, c));
            }
        }
    }
    for &st in &[S::Neutral, S::Fwd, S::Back, S::Side] {
        v.push(sp(K::Backflip, 0.0, 255, st, 0));
    }
    v.push(sp(K::Backflip, 0.0, 1, S::Neutral, 0));
    for &s in &[16.0, 24.0, 32.0] {
        for &st in &[S::Fwd, S::Neutral, S::Back, S::Side] {
            v.push(sp(K::Sideflip, s, 255, st, 0));
        }
        v.push(sp(K::Sideflip, s, 1, S::Fwd, 0));
    }
    for &s in &[29.5, 31.0, 32.0] {
        for &st in &[S::Fwd, S::Neutral, S::Back] {
            v.push(sp(K::Dive, s, 1, st, 0));
            v.push(sp(K::DiveRollout, s, 1, st, 0));
        }
    }
    // Slide kicks: entry speed, Z frames before B (chain), hop frame (held).
    for &s in &[10.5, 16.0, 24.0, 32.0] {
        for &z in &[1u8, 3, 6] {
            v.push(sp(K::SlideKick, s, 0, S::Fwd, z));
            for &h in &[1u8, 4, 8] {
                v.push(sp(K::SlideKickHop, s, h, S::Fwd, z));
            }
        }
        v.push(sp(K::SlideKick, s, 0, S::Neutral, 1));
        v.push(sp(K::SlideKickHop, s, 1, S::Back, 1));
    }
    v
}

// ================================================================ wall kicks

struct WallMeas {
    in_kind: ArcKind,
    v: f32,
    in_speed: f32,
    in_angle: f32,
    press: u8,
    hit: bool,
    kicked: bool,
    out_speed: f32,
    out_angle: f32,
    vy: f32,
    traj: Vec<St>,
    hit_at: usize,
    kick_at: usize,
    land: usize,
    walk: Option<usize>,
    hit_action: u32,
}

/// Contact point on the east wall of the castle door's platform (x 554,
/// normal -x; floor 803 from x -450 to 500, z -3400 to -2350; its twin
/// faces it at x -554).
const WALL_HIT: [f32; 3] = [504.0, 803.0, -2750.0];
const WALL_NORMAL_YAW: i32 = 0xC000;

fn inject_walk(sim: &mut Sim, p: [f32; 3], yaw: i32, v: f32) -> Result<()> {
    sim.teleport(p, yaw)?;
    place(sim, p, yaw, ACT_WALKING, v)
}

/// Puts Mario at `p` in `action` at forward speed `v` (no settling: for
/// slopes, where idling would slide him off first).
fn place(sim: &mut Sim, p: [f32; 3], yaw: i32, action: u32, v: f32) -> Result<()> {
    sim.put(p, yaw, action)?;
    write(&mut sim.g, M + 0x54, &v.to_le_bytes())?;
    // Slides move by slideVelX/Z (0x58/0x5C), walking sets them each frame.
    let (sn, cs) = rad(yaw).sin_cos();
    let (vx, vz) = (v * sn, v * cs);
    write(&mut sim.g, M + 0x58, &[vx.to_le_bytes(), vz.to_le_bytes()].concat())?;
    write(&mut sim.g, M + 0x48, &[vx.to_le_bytes(), 0f32.to_le_bytes(), vz.to_le_bytes()].concat())?;
    // The state the next frame's inputs are chosen from.
    sim.st.pos = p;
    sim.st.action = action;
    sim.st.forward_vel = v;
    sim.st.yaw = yaw as i16;
    sim.st.vel = [0.0; 3];
    Ok(())
}

fn wall_kick_one(gg: &SM64GameGenerator, save0: &SaveState, st0: &St, in_kind: ArcKind, v: f32, angle: f32, press: u8) -> Result<WallMeas> {
    let dyaw = (angle * 65536.0 / 360.0).round() as i32;
    let yaw = 0x4000 + dyaw;
    let (s, c) = rad(yaw).sin_cos();
    let back = if in_kind == ArcKind::LongJump { 600.0 } else { 380.0 };
    let start = [WALL_HIT[0] - s * back, WALL_HIT[1], WALL_HIT[2] - c * back];
    let mut sim = Sim::new(gg, save0, st0)?;
    inject_walk(&mut sim, start, yaw, v)?;
    let jump_at = if v >= 30.0 { 260.0 } else if v >= 20.0 { 200.0 } else { 150.0 };
    if in_kind == ArcKind::LongJump {
        sim.step(Some(yaw), 1.0, Z_TRIG)?;
        sim.step(Some(yaw), 1.0, Z_TRIG | A_BUTTON)?;
    } else {
        let mut n = 0;
        while n < 100 && (WALL_HIT[0] - sim.st.pos[0]) / s.max(0.2) > jump_at {
            sim.step(Some(yaw), 1.0, 0)?;
            n += 1;
        }
        sim.step(Some(yaw), 1.0, A_BUTTON)?;
    }
    let mut k = 1;
    let mut in_speed = sim.st.forward_vel;
    let tr = std::env::var("WALL_TRACE").is_ok();
    while sim.air() && sim.st.action != ACT_AIR_HIT_WALL && k < 80 {
        in_speed = sim.st.forward_vel;
        let b = if in_kind != ArcKind::LongJump && k < 4 { A_BUTTON } else { 0 };
        sim.step(Some(yaw), 1.0, b)?;
        if tr {
            eprintln!("  {k} {} pos {:.0},{:.0},{:.0} fv {:.1} yaw {:#x} wall {:?}", act_name(sim.st.action), sim.st.pos[0], sim.st.pos[1], sim.st.pos[2], sim.st.forward_vel, sim.st.yaw, sim.st.wall_normal);
        }
        k += 1;
    }
    let hit_action = sim.st.action;
    let hit = sim.st.action == ACT_AIR_HIT_WALL;
    let mut traj = vec![sim.st.clone()];
    let mut kicked = false;
    if sim.air() {
        for _ in 0..press {
            sim.step(None, 0.0, 0)?;
            traj.push(sim.st.clone());
        }
        // Stick away from the wall with the press (as the moves library does).
        sim.step(Some(WALL_NORMAL_YAW), 1.0, A_BUTTON)?;
        traj.push(sim.st.clone());
        kicked = sim.st.action == ACT_WALL_KICK;
    }
    let kick_at = traj.len() - 1;
    let (out_speed, vy) = (sim.st.forward_vel, sim.st.vel[1]);
    let out_yaw = sim.st.yaw as i32;
    let out_angle = wrap(out_yaw - WALL_NORMAL_YAW) as f32 * 360.0 / 65536.0;
    if kicked {
        fly(&mut sim, &mut traj, out_yaw, 255, AirStick::Fwd, None)?;
    } else {
        let mut n = 0;
        while sim.air() && n < 120 {
            sim.step(None, 0.0, 0)?;
            traj.push(sim.st.clone());
            n += 1;
        }
    }
    let land = traj.len() - 1;
    let walk = settle(&mut sim, &mut traj, out_yaw)?;
    Ok(WallMeas { in_kind, v, in_speed, in_angle: angle, press, hit, kicked, out_speed, out_angle, vy, traj, hit_at: 0, kick_at, land, walk, hit_action })
}

impl WallMeas {
    /// (distance out from the hit along the wall's normal, height above the hit).
    fn prof(&self) -> (Vec<f32>, Vec<f32>) {
        let o = &self.traj[self.hit_at];
        let k0 = self.kick_at;
        let a = (k0..=self.land).map(|i| r1(rel(o, &self.traj[i], WALL_NORMAL_YAW).0)).collect();
        let h = (k0..=self.land).map(|i| r1(self.traj[i].pos[1] - o.pos[1])).collect();
        (a, h)
    }
    fn json(&self) -> Value {
        let (along, height) = self.prof();
        let peak = height.iter().cloned().fold(f32::MIN, f32::max);
        let o = &self.traj[self.hit_at];
        let land_dist = rel(o, &self.traj[self.land], WALL_NORMAL_YAW).0;
        json!({"in_kind": format!("{:?}", self.in_kind), "run_speed": self.v, "in_speed": r2(self.in_speed), "in_angle": self.in_angle,
            "press": self.press, "hit": self.hit, "hit_action": act_name(self.hit_action), "kicked": self.kicked,
            "out_speed": r2(self.out_speed), "out_angle": r1(self.out_angle), "vy": r2(self.vy),
            "frames": self.land - self.kick_at, "land_dist": r1(land_dist), "peak": peak,
            "lag": self.walk.map_or(255, |w| w - self.land), "land_action": act_name(self.traj[self.land].action),
            "reached_far_wall": land_dist > 1000.0,
            "along": along, "height": height})
    }
}

fn wall_kicks(gg: &SM64GameGenerator, save0: &SaveState, st0: &St) -> Result<Vec<WallMeas>> {
    let mut out = vec![];
    for &(kind, v) in &[(ArcKind::Single, 16.0f32), (ArcKind::Single, 24.0), (ArcKind::Single, 32.0), (ArcKind::LongJump, 32.0)] {
        for &ang in &[0.0f32, 15.0, 30.0, 35.0, 40.0, 44.0, 45.0, 55.0] {
            for &press in &[0u8, 1, 2, 3, 4, 5, 6] {
                let m = wall_kick_one(gg, save0, st0, kind, v, ang, press)?;
                eprintln!(
                    "wall {:?} v{v} ang {ang} press {press}: in {:.1} hit {} kicked {} out {:.1} {:.1} vy {:.0} land {}",
                    kind, m.in_speed, m.hit, m.kicked, m.out_speed, m.out_angle, m.vy, m.land
                );
                out.push(m);
            }
        }
    }
    Ok(out)
}

// ==================================================================== slides

/// A slope triangle per steepness band: (ny, kind, start point, downhill yaw).
fn slope_sites(g: &SM64Game) -> Vec<(f32, i16, [f32; 3], i32)> {
    let (tris, _) = g.with_memory(obrrt_core::geo::read_floors);
    let bands = [(0.60, 0.67), (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 0.985)];
    let mut out = vec![];
    for (lo, hi) in bands {
        let best = tris
            .iter()
            .filter(|t| t.n[1] >= lo && t.n[1] < hi)
            .filter_map(|t| {
                let [a, b, c] = t.v;
                let area = ((b[0] - a[0]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[0] - a[0])).abs() / 2.0;
                let cen = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0];
                // Top floor there, above the moat's water (-81).
                let top = obrrt_core::geo::floor_below(&tris, cen[0], cen[2], 20000.0)?;
                ((top - cen[1]).abs() < 2.0 && cen[1] > 100.0).then_some((area, t.n, t.kind, cen))
            })
            .max_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((_, n, kind, cen)) = best {
            let h = (n[0] * n[0] + n[2] * n[2]).sqrt();
            let down = [n[0] / h, n[2] / h];
            let yaw = (down[0].atan2(down[1]) * 65536.0 / std::f32::consts::TAU).round() as i32;
            // Start 250 uphill of the centroid, on the floor there.
            let (x, z) = (cen[0] - down[0] * 250.0, cen[2] - down[1] * 250.0);
            let y = obrrt_core::geo::floor_below(&tris, x, z, cen[1] + 400.0).unwrap_or(cen[1]);
            out.push((n[1], kind, [x, y, z], yaw));
        }
    }
    out
}

fn slide_one(gg: &SM64GameGenerator, save0: &SaveState, st0: &St, site: (f32, i16, [f32; 3], i32), entry: arcs::SlideEntry, mode: u8) -> Result<Vec<St>> {
    // mode 0: straight; 1: stick 90 degrees right from frame 5; 2: A on frame 20.
    let (_, _, p, yaw) = site;
    let mut sim = Sim::new(gg, save0, st0)?;
    match entry {
        arcs::SlideEntry::Stand => place(&mut sim, p, yaw, ACT_IDLE_, 0.0)?,
        arcs::SlideEntry::Crouch => place(&mut sim, p, yaw, ACT_WALKING, 16.0)?,
        arcs::SlideEntry::Dive => place(&mut sim, p, yaw, ACT_WALKING, 30.0)?,
    }
    let mut traj = vec![sim.st.clone()];
    for k in 0..60 {
        let mut b = match entry {
            arcs::SlideEntry::Crouch => Z_TRIG,
            arcs::SlideEntry::Dive if k == 0 => B_BUTTON,
            _ => 0,
        };
        if mode == 2 && k == 20 {
            b |= A_BUTTON;
        }
        let y = if mode == 1 && k >= 5 { yaw + 0x4000 } else { yaw };
        sim.step(Some(y), 1.0, b)?;
        traj.push(sim.st.clone());
        if mode == 2 && k > 20 && !sim.air() {
            break;
        }
    }
    Ok(traj)
}

fn slides(gg: &SM64GameGenerator, g: &SM64Game, save0: &SaveState, st0: &St) -> Result<Vec<Value>> {
    use arcs::SlideEntry as E;
    let mut out = vec![];
    for site in slope_sites(g) {
        for entry in [E::Stand, E::Crouch, E::Dive] {
            let t = slide_one(gg, save0, st0, site, entry, 0)?;
            let o = &t[0];
            let fv: Vec<f32> = t[1..].iter().map(|s| r2(s.forward_vel)).collect();
            let dist: Vec<f32> = t[1..].iter().map(|s| r1(rel(o, s, site.3).0)).collect();
            // The slide action and its speed gain per frame (frames 3..20 on the slope).
            let slide_action = t.iter().skip(2).find(|s| s.action & AIR == 0).map_or(0, |s| s.action);
            let gains: Vec<f32> = (2..=30.min(t.len() - 1))
                .filter(|&i| (t[i].floor_normal[1] - site.0).abs() < 0.03 && t[i].action == slide_action && t[i - 1].action == slide_action)
                .map(|i| t[i].forward_vel - t[i - 1].forward_vel)
                .collect();
            let accel = if gains.is_empty() { 0.0 } else { gains.iter().sum::<f32>() / gains.len() as f32 };
            let top = fv.iter().cloned().fold(0.0f32, f32::max);
            let ts = slide_one(gg, save0, st0, site, entry, 1)?;
            let steer = if ts.len() > 16 { wrap(ts[16].yaw as i32 - ts[6].yaw as i32) / 10 } else { 0 };
            let tj = slide_one(gg, save0, st0, site, entry, 2)?;
            let (jump_action, jump_frames, jump_dist) = if tj.len() > 22 && tj[22].action & AIR != 0 {
                let j0 = 21;
                let land = (j0..tj.len()).find(|&i| tj[i].action & AIR == 0).unwrap_or(tj.len() - 1);
                (tj[j0].action, land - j0 + 1, rel(&tj[j0 - 1], &tj[land], site.3).0)
            } else {
                (tj.get(21).map_or(0, |s| s.action), 0, 0.0)
            };
            eprintln!(
                "slide ny {:.3} kind {:#x} {:?}: action {} accel {:.2} top {:.1} steer {} jump {} {}f {:.0}",
                site.0, site.1, entry, act_name(slide_action), accel, top, steer, act_name(jump_action), jump_frames, jump_dist
            );
            out.push(json!({"entry": format!("{:?}", entry), "ny": r2(site.0 * 10.0) / 10.0, "ny_exact": site.0, "floor_kind": site.1,
                "start": site.2, "yaw": site.3,
                "slide_action": format!("{:#x}", slide_action), "slide_name": act_name(slide_action), "accel": r2(accel), "top_speed": r2(top),
                "steer_rate": steer, "jump_action": format!("{:#x}", jump_action), "jump_name": act_name(jump_action),
                "jump_frames": jump_frames, "jump_dist": r1(jump_dist),
                "actions": t[1..].iter().map(|s| act_name(s.action)).collect::<Vec<_>>(),
                "ny_under": t[1..].iter().map(|s| r2(s.floor_normal[1] * 10.0) / 10.0).collect::<Vec<_>>(),
                "fv": fv, "dist": dist}));
        }
        // Slide kick downhill: the bounce on a slope.
        let mut sim = Sim::new(gg, save0, st0)?;
        place(&mut sim, site.2, site.3, ACT_WALKING, 24.0)?;
        let o = sim.st.clone();
        let mut t = vec![o.clone()];
        sim.step(Some(site.3), 1.0, Z_TRIG)?;
        t.push(sim.st.clone());
        sim.step(Some(site.3), 1.0, Z_TRIG | B_BUTTON)?;
        t.push(sim.st.clone());
        let mut n = 0;
        while n < 80 && (sim.air() || sim.st.action == ACT_SLIDE_KICK_SLIDE) {
            sim.step(Some(site.3), 1.0, 0)?;
            t.push(sim.st.clone());
            n += 1;
        }
        let bounce = (2..t.len()).find(|&i| t[i].action == ACT_SLIDE_KICK && t[i - 1].vel[1] < 0.0 && t[i].vel[1] > 0.0);
        let air_end = (2..t.len()).find(|&i| t[i].action & AIR == 0).unwrap_or(t.len() - 1);
        out.push(json!({"entry": "SlideKick", "ny": site.0, "floor_kind": site.1, "start": site.2, "yaw": site.3,
            "bounce_frame": bounce, "bounce_vy": bounce.map(|i| r2(t[i].vel[1])),
            "air_frames": air_end, "air_dist": r1(rel(&o, &t[air_end], site.3).0), "air_speed": r2(t[air_end].forward_vel),
            "total_frames": t.len() - 1, "total_dist": r1(rel(&o, &t[t.len() - 1], site.3).0), "end_action": act_name(t[t.len() - 1].action)}));
    }
    Ok(out)
}

/// A slide kick run off a ledge (a drop of 150-1000 onto a floor above the
/// moat's water), against the same kick on the flat.
fn slide_kick_ledge(gg: &SM64GameGenerator, g: &SM64Game, save: &SaveState, st0: &St) -> Result<Value> {
    let (tris, _) = g.with_memory(obrrt_core::geo::read_floors);
    let fl = |x: f32, z: f32, y: f32| obrrt_core::geo::floor_below(&tris, x, z, y);
    // Search the area for a straight flat 800-unit run-up ending at a drop.
    let mut found = None;
    'search: for ix in -40..=40 {
        for iz in -40..=40 {
            let (x, z) = (ix as f32 * 200.0, iz as f32 * 200.0);
            let Some(y0) = fl(x, z, 20000.0) else { continue };
            if y0 < 0.0 {
                continue;
            }
            for k in 0..16 {
                let yaw = k * 0x1000;
                let (s, c) = rad(yaw).sin_cos();
                let mut d = 10.0;
                while d < 3000.0 {
                    match fl(x + s * d, z + c * d, y0 + 30.0) {
                        Some(y) if (y - y0).abs() < 2.0 => d += 10.0,
                        Some(y) if y < y0 - 100.0 && y > y0 - 1100.0 && y > -60.0 && d >= 600.0 => {
                            // A flat landing zone beyond.
                            let flat = (1..=6).all(|j| fl(x + s * (d + 50.0 * j as f32), z + c * (d + 50.0 * j as f32), y + 30.0).is_some_and(|v| (v - y).abs() < 5.0));
                            if flat {
                                found = Some(([x + s * (d - 600.0), y0, z + c * (d - 600.0)], yaw, y0 - y));
                                break 'search;
                            }
                            break;
                        }
                        _ => break,
                    }
                }
            }
        }
    }
    let Some((p0, yaw, drop)) = found else { return Ok(json!(null)) };
    eprintln!("ledge: run-up from {p0:?} yaw {yaw:#x}, drop {drop:.0}");
    let mut res = vec![];
    for &(before, label) in &[(140.0f32, "edge"), (60.0, "edge_late"), (500.0, "flat")] {
        let mut sim = Sim::new(gg, save, st0)?;
        inject_walk(&mut sim, p0, yaw, 32.0)?;
        let start = sim.st.clone();
        let mut n = 0;
        while n < 60 && rel(&start, &sim.st, yaw).0 < 600.0 - before {
            sim.step(Some(yaw), 1.0, 0)?;
            n += 1;
        }
        let o = sim.st.clone();
        let mut t = vec![o.clone()];
        sim.step(Some(yaw), 1.0, Z_TRIG)?;
        t.push(sim.st.clone());
        sim.step(Some(yaw), 1.0, Z_TRIG | B_BUTTON)?;
        t.push(sim.st.clone());
        let mut n = 0;
        while n < 120 && sim.air() {
            sim.step(Some(yaw), 1.0, 0)?;
            t.push(sim.st.clone());
            n += 1;
        }
        let bounce = (2..t.len()).find(|&i| t[i].action == ACT_SLIDE_KICK && t[i - 1].vel[1] < 0.0 && t[i].vel[1] > 0.0);
        let last = &t[t.len() - 1];
        res.push(json!({"case": label, "start": p0, "yaw": yaw, "drop": drop, "kick_before_edge": before,
            "bounce_frame": bounce, "bounce_vy": bounce.map(|i| r2(t[i].vel[1])),
            "bounce_dy": bounce.map(|i| r1(t[i].pos[1] - o.pos[1])), "bounce_dist": bounce.map(|i| r1(rel(&o, &t[i], yaw).0)),
            "air_frames": t.len() - 1, "air_dist": r1(rel(&o, last, yaw).0), "dy": r1(last.pos[1] - o.pos[1]), "land_speed": r2(last.forward_vel),
            "land_action": act_name(last.action), "health_loss": o.health - last.health}));
    }
    Ok(json!(res))
}

// ========================================================== pounds and falls

fn drop_from(gg: &SM64GameGenerator, save: &SaveState, st0: &St, h: f32, pound: bool, after: u8) -> Result<(Vec<St>, usize, usize)> {
    // after: 0 stick held from the landing; 1 A every other frame; 2 nothing.
    let mut sim = Sim::new(gg, save, st0)?;
    let p = [st0.pos[0], st0.pos[1] + h, st0.pos[2]];
    sim.put(p, RUN_YAW, ACT_FREEFALL)?;
    let mut t = vec![sim.st.clone()];
    sim.step(None, 0.0, 0)?;
    t.push(sim.st.clone());
    let z_at = t.len() - 1;
    if pound {
        sim.step(None, 0.0, Z_TRIG)?;
        t.push(sim.st.clone());
    }
    let mut n = 0;
    while sim.air() && n < 400 {
        sim.step(None, 0.0, 0)?;
        t.push(sim.st.clone());
        n += 1;
    }
    let land = t.len() - 1;
    for k in 0..120 {
        match after {
            0 => sim.step(Some(RUN_YAW), 1.0, 0)?,
            1 => sim.step(None, 0.0, if k % 2 == 0 { A_BUTTON } else { 0 })?,
            _ => sim.step(None, 0.0, 0)?,
        }
        t.push(sim.st.clone());
    }
    Ok((t, z_at, land))
}

fn pounds_and_falls(gg: &SM64GameGenerator, save: &SaveState, st0: &St) -> Result<(Vec<Value>, Vec<Value>, Value)> {
    let mut gp = vec![];
    for &h in &[150.0f32, 300.0, 600.0, 1000.0, 1500.0, 2500.0, 3500.0] {
        let (t, z, land) = drop_from(gg, save, st0, h, true, 0)?;
        // Spin: frames from Z until he starts to come down (the game raises
        // him 20, 18, ... a frame meanwhile and sets peakHeight there).
        let spin = (z + 2..t.len()).find(|&i| t[i].pos[1] < t[i - 1].pos[1]).map_or(0, |i| i - z - 1);
        let rise = t.iter().skip(z).map(|s| s.pos[1]).fold(f32::MIN, f32::max) - t[z].pos[1];
        let lag_stick = (land..t.len()).find(|&i| t[i].action == ACT_WALKING).map_or(255, |i| i - land);
        let loss = t[0].health - t.iter().skip(land).map(|s| s.health).min().unwrap_or(t[0].health);
        let (ta, _, land_a) = drop_from(gg, save, st0, h, true, 1)?;
        let lag_a = (land_a + 1..ta.len()).find(|&i| ta[i].action & AIR != 0).map_or(255, |i| i - land_a);
        let (tn, _, land_n) = drop_from(gg, save, st0, h, true, 2)?;
        let idle = (land_n..tn.len()).find(|&i| tn[i].action == ACT_IDLE_).map_or(255, |i| i - land_n);
        let vy_fall = t.iter().skip(z + 1).map(|s| s.vel[1]).fold(0.0f32, f32::min);
        eprintln!("gp h{h}: spin {spin} land {} {} lag stick {lag_stick} A {lag_a} idle {idle} loss {loss:#x} vy {vy_fall}", land - z, act_name(t[land].action));
        gp.push(json!({"height": h, "spin": spin, "rise": r1(rise), "to_land": land - z, "land_action": format!("{:#x}", t[land].action), "land_name": act_name(t[land].action),
            "lag_stick": lag_stick, "lag_a": lag_a, "lag_idle": idle, "fall_vy": vy_fall, "health_loss": loss,
            "after_stick": t[land..(land + 30).min(t.len())].iter().map(|s| act_name(s.action)).collect::<Vec<_>>(),
            "after_a": ta[land_a..(land_a + 30).min(ta.len())].iter().map(|s| act_name(s.action)).collect::<Vec<_>>()}));
    }
    // Ground pound out of a running jump: how far he still drifts after Z.
    let mut drift = vec![];
    for &zf in &[2usize, 6, 10] {
        let mut sim = Sim::new(gg, save, st0)?;
        run_to(&mut sim, RUN_YAW, 32.0)?;
        sim.step(Some(RUN_YAW), 1.0, A_BUTTON)?;
        for _ in 1..zf {
            sim.step(Some(RUN_YAW), 1.0, A_BUTTON)?;
        }
        let at_z = sim.st.clone();
        sim.step(Some(RUN_YAW), 1.0, Z_TRIG)?;
        let gp_ok = sim.st.action == ACT_GROUND_POUND;
        let mut n = 0;
        while sim.air() && n < 200 {
            sim.step(Some(RUN_YAW), 1.0, 0)?;
            n += 1;
        }
        drift.push(json!({"z_at_air_frame": zf, "pounded": gp_ok, "drift_after_z": r1(rel(&at_z, &sim.st, RUN_YAW).0), "frames_to_land": n + 1}));
    }
    let mut falls = vec![];
    for &h in &[500.0f32, 1000.0, 1140.0, 1160.0, 1300.0, 2000.0, 2990.0, 3010.0, 3500.0, 5000.0] {
        for &pound in &[false, true] {
            let (t, _, land) = drop_from(gg, save, st0, h, pound, 0)?;
            let loss = t[0].health - t.iter().skip(land).take(60).map(|s| s.health).min().unwrap_or(t[0].health);
            let lag = (land..t.len()).find(|&i| t[i].action == ACT_WALKING).map_or(255, |i| i - land);
            eprintln!("fall h{h} pound {pound}: {} loss {loss:#x} lag {lag}", act_name(t[land].action));
            falls.push(json!({"height": h, "mode": if pound { "GroundPound" } else { "Freefall" }, "health_loss": loss,
                "land_action": format!("{:#x}", t[land].action), "land_name": act_name(t[land].action), "lag": lag,
                "after": t[land..(land + 12).min(t.len())].iter().map(|s| act_name(s.action)).collect::<Vec<_>>()}));
        }
    }
    Ok((gp, falls, json!(drift)))
}

// ==================================================================== chains

/// Distance covered in `frames` by a repeated move, from walking at 32.
fn chain(gg: &SM64GameGenerator, save: &SaveState, st0: &St, which: &str, frames: usize) -> Result<f32> {
    let mut sim = Sim::new(gg, save, st0)?;
    inject_walk(&mut sim, RUN_START, RUN_YAW, 32.0)?;
    let o = sim.st.clone();
    for _ in 0..frames {
        let a = sim.st.action;
        let ground = a & AIR == 0;
        let b = match which {
            "run" => 0,
            "long_jump_chain" => match a {
                ACT_CROUCH_SLIDE | ACT_LONG_JUMP_LAND => Z_TRIG | A_BUTTON,
                _ if ground => Z_TRIG,
                _ => Z_TRIG,
            },
            "slide_kick_chain" => match a {
                ACT_CROUCH_SLIDE => Z_TRIG | B_BUTTON,
                ACT_SLIDE_KICK_SLIDE => A_BUTTON,
                ACT_WALKING => Z_TRIG,
                _ => 0,
            },
            "slide_kick_ride_chain" => match a {
                ACT_CROUCH_SLIDE => Z_TRIG | B_BUTTON,
                ACT_WALKING => Z_TRIG,
                _ => 0,
            },
            "dive_rollout_chain" => match a {
                ACT_WALKING if sim.st.forward_vel >= 29.0 => B_BUTTON,
                ACT_DIVE_SLIDE => A_BUTTON,
                _ => 0,
            },
            _ => 0,
        };
        sim.step(Some(RUN_YAW), 1.0, b)?;
        if std::env::var("CHAIN_TRACE").is_ok_and(|v| v == which) {
            eprintln!("  {which} b {b:04x} {} fv {:.1} along {:.0} y {:.0}", act_name(sim.st.action), sim.st.forward_vel, rel(&o, &sim.st, RUN_YAW).0, sim.st.pos[1]);
        }
    }
    Ok(rel(&o, &sim.st, RUN_YAW).0)
}

// ==================================================================== ground

fn ground(gg: &SM64GameGenerator, save: &SaveState, st0: &St) -> Result<Value> {
    let yaw = RUN_YAW;
    let mut sim = Sim::new(gg, save, st0)?;
    let o = sim.st.clone();
    let (mut fv, mut dist, mut acts) = (vec![], vec![], vec![]);
    for _ in 0..60 {
        sim.step(Some(yaw), 1.0, 0)?;
        fv.push(r2(sim.st.forward_vel));
        dist.push(r1(rel(&o, &sim.st, yaw).0));
        acts.push(act_name(sim.st.action));
    }
    let snap = sim.g.save_state();
    let st32 = sim.st.clone();
    let o = sim.st.clone();
    let mut stop = (0, 0.0, String::new());
    for k in 1..60 {
        sim.step(None, 0.0, 0)?;
        if sim.st.forward_vel <= 0.0 || sim.st.action == ACT_IDLE_ {
            stop = (k, rel(&o, &sim.st, yaw).0, act_name(sim.st.action));
            break;
        }
    }
    let mut sim = Sim::new(gg, &snap, &st32)?;
    let mut skid = (0, 0.0, 0.0);
    let mut max_along = 0f32;
    for k in 1..60 {
        sim.step(Some(yaw + 0x8000), 1.0, 0)?;
        max_along = max_along.max(rel(&o, &sim.st, yaw).0);
        if sim.st.action == ACT_WALKING && wrap(sim.st.yaw as i32 - (yaw + 0x8000)).abs() < 0x1000 {
            skid = (k, max_along, rel(&o, &sim.st, yaw).0);
            break;
        }
    }
    let mut turns = vec![];
    for &off in &[0x2000, 0x4000, 0x6000] {
        let mut sim = Sim::new(gg, &snap, &st32)?;
        let mut rec = json!({"off_deg": off as f32 * 360.0 / 65536.0, "frames": 255});
        for k in 1..60 {
            sim.step(Some(yaw + off), 1.0, 0)?;
            if wrap(sim.st.yaw as i32 - (yaw + off)).abs() < 0x100 && sim.st.action == ACT_WALKING {
                let (a, l, _) = rel(&o, &sim.st, yaw);
                rec = json!({"off_deg": off as f32 * 360.0 / 65536.0, "frames": k, "fv": r2(sim.st.forward_vel),
                    "fwd": r1(a), "lat": r1(l), "action": act_name(sim.st.action)});
                break;
            }
        }
        turns.push(rec);
    }
    let mut sim = Sim::new(gg, &snap, &st32)?;
    let y0 = sim.st.yaw as i32;
    sim.step(Some(yaw + 0x4000), 1.0, 0)?;
    let rate = wrap(sim.st.yaw as i32 - y0);
    Ok(json!({"run_fv": fv, "run_dist": dist, "run_actions": acts,
        "stop": {"frames": stop.0, "dist": r1(stop.1), "action": stop.2},
        "skid": {"frames": skid.0, "max_dist": r1(skid.1), "end_dist": r1(skid.2)},
        "turns": turns, "turn_rate": rate}))
}

fn air_steer(ms: &[Meas]) -> Value {
    let mut v = vec![];
    for m in ms.iter().filter(|m| m.spec.stick == AirStick::Side) {
        let Some(f) = ms.iter().find(|f| {
            f.spec.kind == m.spec.kind && f.spec.v == m.spec.v && f.spec.chain == m.spec.chain && f.spec.held == m.spec.held && f.spec.stick == AirStick::Fwd
        }) else {
            continue;
        };
        let (a, l, _) = m.rel(m.land);
        let (af, lf, _) = f.rel(f.land);
        let air = (m.land - m.takeoff + 1) as f32;
        v.push(json!({"kind": format!("{:?}", m.spec.kind), "v": m.spec.v, "chain": m.spec.chain, "frames": m.land,
            "lat": r1(l - lf), "lat_per_frame": r2((l - lf) / air), "along_loss": r1(af - a),
            "yaw_change": wrap(m.traj[m.land].yaw as i32 - m.traj[m.takeoff].yaw as i32)}));
    }
    json!(v)
}

// ================================================================== measure

fn repo(p: &str) -> String {
    format!("{}/../{p}", env!("CARGO_MANIFEST_DIR"))
}

fn measure_all(gg: &SM64GameGenerator, g: &SM64Game, st: &St) -> Result<()> {
    let (save, st0) = runway(gg, g, st)?;
    let (save_mid, st_mid) = runway_at(gg, g, st, RUN_MID)?;
    let save0 = g.save_state();
    let mut moves = vec![];
    let mut metas: Vec<Meas> = vec![];
    for spec in all_specs() {
        let mid = matches!(spec.kind, ArcKind::Backflip | ArcKind::Sideflip) || spec.stick == AirStick::Side;
        let mut sim = if mid { Sim::new(gg, &save_mid, &st_mid)? } else { Sim::new(gg, &save, &st0)? };
        match measure(&mut sim, spec, RUN_YAW)? {
            Some(m) if m.valid() => {
                let j = m.json();
                eprintln!(
                    "{:?} {:?} v{} h{} c{}: v_in {} {} frames {} land {} lag {} {}",
                    spec.kind, spec.stick, spec.v, spec.held, spec.chain, j["v_in"], j["takeoff_name"], j["frames"], j["land_dist"], j["lag"], j["land_name"]
                );
                moves.push(j);
                metas.push(m);
            }
            _ => eprintln!("{:?} {:?} v{} h{} c{}: FAILED", spec.kind, spec.stick, spec.v, spec.held, spec.chain),
        }
    }
    let walls: Vec<Value> = wall_kicks(gg, &save0, st)?.iter().map(|w| w.json()).collect();
    let slides = slides(gg, g, &save0, st)?;
    let ledge = slide_kick_ledge(gg, g, &save, &st0)?;
    let (gp, falls, gp_drift) = pounds_and_falls(gg, &save, &st0)?;
    let mut chains = vec![];
    for which in ["run", "long_jump_chain", "slide_kick_chain", "slide_kick_ride_chain", "dive_rollout_chain"] {
        let d = chain(gg, &save, &st0, which, 60)?;
        eprintln!("chain {which}: {d:.0} in 60 frames");
        chains.push(json!({"name": which, "frames": 60, "dist": r1(d)}));
    }
    let run_d = chains[0]["dist"].clone();
    for c in chains.iter_mut() {
        c["run_dist"] = run_d.clone();
    }
    let ground = ground(gg, &save, &st0)?;
    let steer = air_steer(&metas);
    let doc = json!({
        "generated_by": "arc_lab <rom> measure (run/examples/arc_lab.rs)",
        "frame_of_reference": "moves: along/lat/height relative to Mario's position before the move's first input; along = the travel direction (backflip, side flip: opposite the facing); per frame from the first input's frame to the landing frame. wall_kicks: along = out from the hit along the wall's normal, height above the hit, from the kick frame.",
        "runway": {"start": RUN_START, "yaw": RUN_YAW},
        "wall": {"hit": WALL_HIT, "normal_yaw": WALL_NORMAL_YAW, "facing_wall_gap": 1108.0},
        "moves": moves, "wall_kicks": walls, "slides": slides, "slide_kick_ledge": ledge,
        "ground_pound": gp, "ground_pound_drift": gp_drift, "falls": falls, "chains": chains,
        "ground": ground, "air_steer": steer,
    });
    std::fs::write(repo("ref/arcs/arcs.json"), serde_json::to_string_pretty(&doc)? + "\n")?;
    std::fs::write(repo("ref/arcs/arcs_table.rs"), table_rs(&doc))?;
    eprintln!("wrote ref/arcs/arcs.json and ref/arcs/arcs_table.rs");
    Ok(())
}

fn fj(v: &Value) -> f32 {
    v.as_f64().unwrap_or(0.0) as f32
}
fn uj(v: &Value) -> u64 {
    v.as_u64().unwrap_or(0)
}
fn hexj(v: &Value) -> u32 {
    v.as_str().and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok()).unwrap_or(0)
}
fn list(v: &Value) -> String {
    v.as_array().map_or(String::new(), |a| a.iter().map(|x| format!("{:?}", fj(x))).collect::<Vec<_>>().join(","))
}

fn table_rs(doc: &Value) -> String {
    let mut s = String::from(
        "// Generated by `arc_lab <rom> measure` (run/examples/arc_lab.rs) from the\n// measurements in ref/arcs/arcs.json. Do not edit; rerun the lab.\n#[rustfmt::skip]\npub const RECS: &[Rec] = &[\n",
    );
    for m in doc["moves"].as_array().unwrap() {
        writeln!(
            s,
            "    Rec {{ kind: ArcKind::{}, stick: AirStick::{}, chain: {}, held: {}, v_in: {:?}, v_take: {:?}, vy_take: {:?}, takeoff_action: {:#x}, prep: {}, frames: {}, land_dist: {:?}, land_lat: {:?}, land_speed: {:?}, land_action: {:#x}, lag: {}, lag_dist: {:?}, walk_speed: {:?}, peak: {:?}, peak_frame: {}, yaw_change: {}, along: &[{}], height: &[{}] }},",
            m["kind"].as_str().unwrap(), m["stick"].as_str().unwrap(), uj(&m["chain"]), uj(&m["held"]), fj(&m["v_in"]), fj(&m["v_take"]), fj(&m["vy_take"]),
            hexj(&m["takeoff_action"]), uj(&m["prep"]), uj(&m["frames"]).min(255), fj(&m["land_dist"]), fj(&m["land_lat"]), fj(&m["land_speed"]), hexj(&m["land_action"]),
            uj(&m["lag"]).min(255), fj(&m["lag_dist"]), fj(&m["walk_speed"]), fj(&m["peak"]), uj(&m["peak_frame"]), m["yaw_change"].as_i64().unwrap(),
            list(&m["along"]), list(&m["height"])
        )
        .unwrap();
    }
    s += "];\n#[rustfmt::skip]\npub const WALL_KICKS: &[WallRec] = &[\n";
    for w in doc["wall_kicks"].as_array().unwrap().iter().filter(|w| !w["kicked"].as_bool().unwrap() || (fj(&w["out_angle"]) + fj(&w["in_angle"])).abs() < 10.0) {
        writeln!(
            s,
            "    WallRec {{ in_kind: ArcKind::{}, in_speed: {:?}, in_angle: {:?}, press: {}, hit: {}, kicked: {}, out_speed: {:?}, out_angle: {:?}, vy: {:?}, frames: {}, land_dist: {:?}, peak: {:?}, lag: {}, along: &[{}], height: &[{}] }},",
            w["in_kind"].as_str().unwrap(), fj(&w["in_speed"]), fj(&w["in_angle"]), uj(&w["press"]), w["hit"].as_bool().unwrap(), w["kicked"].as_bool().unwrap(),
            fj(&w["out_speed"]), fj(&w["out_angle"]), fj(&w["vy"]), uj(&w["frames"]).min(255), fj(&w["land_dist"]), fj(&w["peak"]), uj(&w["lag"]).min(255),
            list(&w["along"]), list(&w["height"])
        )
        .unwrap();
    }
    s += "];\n#[rustfmt::skip]\npub const SLIDES: &[SlideRec] = &[\n";
    let slide_names = ["BUTT_SLIDE", "CROUCH_SLIDE", "STOMACH_SLIDE", "DIVE_SLIDE"];
    for r in doc["slides"].as_array().unwrap().iter().filter(|r| r["entry"] != "SlideKick" && slide_names.contains(&r["slide_name"].as_str().unwrap_or(""))) {
        writeln!(
            s,
            "    SlideRec {{ entry: SlideEntry::{}, ny: {:?}, floor_kind: {}, accel: {:?}, top_speed: {:?}, slide_action: {:#x}, steer_rate: {}, fv: &[{}], dist: &[{}], jump_action: {:#x}, jump_frames: {}, jump_dist: {:?} }},",
            r["entry"].as_str().unwrap(), fj(&r["ny_exact"]), r["floor_kind"].as_i64().unwrap(), fj(&r["accel"]), fj(&r["top_speed"]), hexj(&r["slide_action"]),
            r["steer_rate"].as_i64().unwrap(), list(&r["fv"]), list(&r["dist"]), hexj(&r["jump_action"]), uj(&r["jump_frames"]), fj(&r["jump_dist"])
        )
        .unwrap();
    }
    s += "];\npub const GP: &[GpRec] = &[\n";
    for r in doc["ground_pound"].as_array().unwrap() {
        writeln!(
            s,
            "    GpRec {{ height: {:?}, spin: {}, to_land: {}, land_action: {:#x}, lag_stick: {}, lag_a: {}, health_loss: {} }},",
            fj(&r["height"]), uj(&r["spin"]), uj(&r["to_land"]).min(255), hexj(&r["land_action"]), uj(&r["lag_stick"]).min(255), uj(&r["lag_a"]).min(255),
            r["health_loss"].as_i64().unwrap()
        )
        .unwrap();
    }
    s += "];\npub const FALLS: &[FallRec] = &[\n";
    for r in doc["falls"].as_array().unwrap() {
        writeln!(
            s,
            "    FallRec {{ height: {:?}, mode: FallMode::{}, health_loss: {}, land_action: {:#x}, lag: {} }},",
            fj(&r["height"]), r["mode"].as_str().unwrap(), r["health_loss"].as_i64().unwrap(), hexj(&r["land_action"]), uj(&r["lag"]).min(255)
        )
        .unwrap();
    }
    s += "];\npub const CHAINS: &[ChainRec] = &[\n";
    for r in doc["chains"].as_array().unwrap() {
        writeln!(s, "    ChainRec {{ name: {:?}, frames: {}, dist: {:?}, run_dist: {:?} }},", r["name"].as_str().unwrap(), uj(&r["frames"]), fj(&r["dist"]), fj(&r["run_dist"])).unwrap();
    }
    s += "];\n";
    let g = &doc["ground"];
    writeln!(s, "pub const RUN_FV: &[f32] = &[{}];", list(&g["run_fv"])).unwrap();
    writeln!(s, "pub const RUN_DIST: &[f32] = &[{}];", list(&g["run_dist"])).unwrap();
    let t90 = &g["turns"][1];
    writeln!(
        s,
        "pub const GROUND: Ground = Ground {{ stop_frames: {}, stop_dist: {:?}, skid_frames: {}, skid_dist: {:?}, turn90_frames: {}, turn90_fv: {:?}, turn_rate: {} }};",
        uj(&g["stop"]["frames"]), fj(&g["stop"]["dist"]), uj(&g["skid"]["frames"]), fj(&g["skid"]["max_dist"]),
        uj(&t90["frames"]).min(255), fj(&t90["fv"]), g["turn_rate"].as_i64().unwrap().abs()
    )
    .unwrap();
    s += "pub const AIR_STEER: &[Steer] = &[\n";
    for a in doc["air_steer"].as_array().unwrap() {
        writeln!(
            s,
            "    Steer {{ kind: ArcKind::{}, v: {:?}, chain: {}, lat: {:?}, along_loss: {:?}, yaw_change: {} }},",
            a["kind"].as_str().unwrap(), fj(&a["v"]), uj(&a["chain"]), fj(&a["lat"]).abs(), fj(&a["along_loss"]), a["yaw_change"].as_i64().unwrap().abs()
        )
        .unwrap();
    }
    s += "];\n";
    s
}

// ================================================================= validate

/// predict() against the game at entry speeds between the table's, and the
/// air simulator `arcs::fly` against every frame of those flights.
fn validate(gg: &SM64GameGenerator, g: &SM64Game, st: &St) -> Result<()> {
    use AirStick as S;
    use ArcKind as K;
    let (save, st0) = runway(gg, g, st)?;
    let (save_mid, st_mid) = runway_at(gg, g, st, RUN_MID)?;
    let specs: Vec<Spec> = [
        (K::Single, 4.0, 255, 0), (K::Single, 12.0, 255, 0), (K::Single, 20.0, 255, 0), (K::Single, 28.0, 255, 0), (K::Single, 20.0, 4, 0),
        (K::Double, 12.0, 255, 0), (K::Double, 20.0, 255, 0), (K::Double, 28.0, 255, 0),
        (K::Triple, 20.0, 255, 0), (K::Triple, 28.0, 255, 0),
        (K::LongJump, 14.0, 1, 0), (K::LongJump, 18.0, 1, 0), (K::LongJump, 22.0, 1, 0), (K::LongJump, 26.0, 1, 0), (K::LongJump, 30.0, 1, 0),
        (K::LongJump, 20.0, 1, 1), (K::LongJump, 28.0, 1, 2),
        (K::Backflip, 0.0, 255, 0), (K::Sideflip, 20.0, 255, 0), (K::Sideflip, 28.0, 255, 0),
        (K::Dive, 30.0, 1, 0), (K::DiveRollout, 30.0, 1, 0), (K::JumpKick, 12.0, 4, 0), (K::JumpKick, 20.0, 4, 0),
        (K::SlideKick, 20.0, 0, 1), (K::SlideKickHop, 20.0, 1, 1), (K::SlideKickHop, 28.0, 4, 1),
    ]
    .iter()
    .map(|&(kind, v, held, chain)| Spec { kind, v, held, stick: if kind == K::Backflip { S::Neutral } else { S::Fwd }, chain })
    .collect();
    println!("predict() vs game (entry speeds off the table's grid; stick held forward)");
    println!("{:<12} {:>5} {:>4} {:>3} | {:>7} {:>7} {:>6} {:>6} | {:>4} {:>4} | {:>4} {:>4}", "kind", "v_in", "held", "ch", "land", "pred", "err", "peakE", "frm", "pred", "lag", "pred");
    let mut errs: std::collections::BTreeMap<String, (f32, f32, usize)> = Default::default();
    let mut fly_err: std::collections::BTreeMap<String, (f32, i32, usize)> = Default::default();
    for spec in specs {
        let mid = matches!(spec.kind, K::Backflip | K::Sideflip);
        let mut sim = if mid { Sim::new(gg, &save_mid, &st_mid)? } else { Sim::new(gg, &save, &st0)? };
        let Some(m) = measure(&mut sim, spec, RUN_YAW)? else {
            println!("{:?} v{}: failed", spec.kind, spec.v);
            continue;
        };
        let v_in = m.traj[0].forward_vel;
        let p = arcs::predict_with(spec.kind, v_in, spec.held, spec.stick, spec.chain > 0);
        let (land, _, _) = m.rel(m.land);
        let peak = (1..=m.land).map(|i| m.rel(i).2).fold(f32::MIN, f32::max);
        let lag = m.walk.map_or(255, |w| w - m.land);
        let err = land - p.land_dist;
        let e = errs.entry(format!("{:?}", spec.kind)).or_insert((0.0, 0.0, 0));
        e.0 += err.abs();
        e.1 = e.1.max(err.abs());
        e.2 += 1;
        println!(
            "{:<12} {:>5.1} {:>4} {:>3} | {:>7.1} {:>7.1} {:>6.2} {:>6.2} | {:>4} {:>4} | {:>4} {:>4}",
            format!("{:?}", spec.kind), v_in, spec.held, spec.chain, land, p.land_dist, err, peak - p.peak, m.land, p.frames, lag, p.lag
        );
        // fly() from each air frame of plain jumps (one air phase, A released
        // once falling: held 255 jumps only from their falling frames).
        if matches!(spec.kind, K::Single | K::Double | K::Triple | K::LongJump | K::Sideflip | K::Backflip) {
            let land_s = &m.traj[m.land];
            for k in m.takeoff..m.land {
                let s = &m.traj[k];
                // Only once A is released (the simulation assumes it is).
                let a_held = spec.kind != K::LongJump && (k - m.takeoff + 1 < spec.held as usize) && (spec.held != 255 || s.vel[1] > 0.0);
                if s.action & AIR == 0 || a_held {
                    continue;
                }
                let fwd = if spec.stick == S::Fwd { 1.0 } else { 0.0 };
                let (n, pos, _) = arcs::fly(s, land_s.pos[1], fwd, 0.0);
                let d = ((pos[0] - land_s.pos[0]).powi(2) + (pos[2] - land_s.pos[2]).powi(2)).sqrt();
                let fe = fly_err.entry(format!("{:?}", spec.kind)).or_insert((0.0, 0, 0));
                fe.0 = fe.0.max(d);
                fe.1 = fe.1.max((n as i32 - (m.land - k) as i32).abs());
                fe.2 += 1;
            }
        }
    }
    println!("\nlanding distance error of predict() by kind (units):  mean  max");
    for (k, (sum, max, n)) in &errs {
        println!("  {k:<12} {:>6.2} {:>6.2}  ({n} runs)", sum / *n as f32, max);
    }
    println!("\nwall kick chains between parallel walls (head-on, first-frame A): height per kick, kicks to climb 1000");
    for gap in [200.0f32, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 1000.0, 1108.0] {
        println!("  gap {gap:>5}: {:?} per kick, {:?} kicks for +1000", arcs::wall_kick_chain(gap).map(r1), arcs::wall_kicks_to_climb(gap, 1000.0));
    }
    // Planner cost per call.
    let t0 = std::time::Instant::now();
    let mut s = st0.clone();
    s.action = 0x0400_0440;
    s.forward_vel = 32.0;
    let mut n = 0;
    for d in (500..3000).step_by(50) {
        let _ = arcs::plan_approach(&s, [s.pos[0], s.pos[1], s.pos[2] + d as f32], Some([0.0, 1.0]), arcs::DOOR_STANDOFF);
        n += 1;
    }
    println!("\nplan_approach: {:.0} us a call (targets 500-3000 away)", t0.elapsed().as_secs_f64() * 1e6 / n as f64);
    println!("\nfly() from mid-air states: worst landing position error, worst frame error");
    for (k, (d, f, n)) in &fly_err {
        println!("  {k:<12} {:>6.2} units {:>2} frames  ({n} states)", d, f);
    }
    Ok(())
}

// ===================================================================== doors

const ACT_PULLING_DOOR_: u32 = 0x0000_1320;
const ACT_PUSHING_DOOR_: u32 = 0x0000_1321;

fn entered(s: &St, zone: (u16, u8)) -> bool {
    s.zone() != zone || matches!(s.action, ACT_PULLING_DOOR_ | ACT_PUSHING_DOOR_ | 0x0000_1331 | 0x0000_132F | 0x0000_132E)
}

struct DoorRun {
    frames: Option<u32>,
    moves: Vec<(String, f32, f32)>,
    end: St,
}

/// Plays from `st0` until Mario goes through a door: `planned` per frame
/// `plan_approach(..).now.pad(st)`, else the stick straight at the door.
fn door_run(gg: &SM64GameGenerator, save: &SaveState, st0: &St, door: [f32; 3], door_yaw: i32, planned: bool, tris: &[obrrt_core::geo::Tri], trace: bool) -> Result<DoorRun> {
    let mut sim = Sim::new(gg, save, st0)?;
    let zone = st0.zone();
    let floor = |x: f32, z: f32, y: f32| obrrt_core::geo::floor_below(tris, x, z, y);
    let (_, dir) = arcs::door_approach(door, door_yaw, st0.pos, 0.0);
    let mut moves = vec![];
    // An open move: (kind, its first-input state, predicted landing distance).
    let mut open: Option<(ArcKind, St, f32, bool)> = None;
    for f in 0..900u32 {
        if entered(&sim.st, zone) {
            return Ok(DoorRun { frames: Some(f), moves, end: sim.st.clone() });
        }
        let s = sim.st.clone();
        let pad = if planned {
            match arcs::plan_approach_with(&s, door, Some(dir), arcs::DOOR_STANDOFF, Some(&floor)) {
                Some(plan) => {
                    if trace {
                        eprintln!(
                            "  f{f} {} pos {:.0},{:.0},{:.0} fv {:.1} now {:?} frames {:.0} run {:.0} steps {}",
                            act_name(s.action), s.pos[0], s.pos[1], s.pos[2], s.forward_vel, plan.now, plan.frames, plan.run_frames, plan.steps.len()
                        );
                    }
                    // Landing error bookkeeping: a long jump / dive starts now.
                    let starts = match plan.now {
                        arcs::Intent::LongJump { .. } if s.action & AIR == 0 && s.action != ACT_CROUCH_SLIDE => Some((ArcKind::LongJump, s.action == ACT_LONG_JUMP_LAND)),
                        arcs::Intent::DiveRollout { .. } if s.action == ACT_WALKING && s.forward_vel >= 29.0 => Some((ArcKind::DiveRollout, false)),
                        arcs::Intent::Jump { .. } if s.action & AIR == 0 => Some((ArcKind::Single, false)),
                        _ => None,
                    };
                    if let Some((k, chained)) = starts
                        && open.is_none()
                    {
                        // What the plan means this jump to cover (shortened last jumps
                        // included), else the full-stick prediction.
                        let planned = plan.steps.iter().find_map(|st| match st {
                            arcs::Planned::Move { dist, .. } => Some(*dist),
                            _ => None,
                        });
                        let p = arcs::predict_with(k, s.forward_vel, if k == ArcKind::Single { 255 } else { 1 }, AirStick::Fwd, chained);
                        open = Some((k, s.clone(), planned.unwrap_or(p.land_dist), true));
                    }
                    if let (Some(o), arcs::Intent::Air { fwd, side, .. }) = (open.as_mut(), plan.now)
                        && (fwd < 0.99 || side.abs() > 0.05)
                    {
                        o.3 = false; // steered: not an open-loop flight
                    }
                    plan.now.pad(&s)
                }
                None => GamePad::new(0, 0, 0),
            }
        } else {
            let to = [door[0] - s.pos[0], door[2] - s.pos[2]];
            let yaw = (to[0].atan2(to[1]) * 65536.0 / std::f32::consts::TAU) as i16;
            let (x, y) = arcs::stick(s.cam_yaw, yaw, 1.0);
            GamePad::new(0, x, y)
        };
        sim.pad(pad)?;
        // A landing closes the open move (dive + rollout: the rollout's landing).
        if let Some((k, o, pred, open_loop)) = open.clone() {
            let landed = sim.st.action & AIR == 0 && s.action & AIR != 0 && !(k == ArcKind::DiveRollout && sim.st.action == ACT_DIVE_SLIDE);
            if landed {
                let yaw = o.yaw as i32;
                let d = rel(&o, &sim.st, yaw).0;
                moves.push((format!("{:?}{}", k, if open_loop { "" } else { " (steered)" }), d, pred));
                open = None;
            }
        }
    }
    Ok(DoorRun { frames: None, moves, end: sim.st.clone() })
}

fn door_test(gg: &SM64GameGenerator, g: &SM64Game, st: &St, trace: bool) -> Result<()> {
    let (tris, _) = g.with_memory(obrrt_core::geo::read_floors);
    let doors: Vec<_> = st.objects().iter().filter(|o| o.bhv_name() == "bhvDoorWarp" && o.pos[1] > 700.0).cloned().collect();
    println!("castle front doors: {:?}", doors.iter().map(|o| (o.pos, o.yaw)).collect::<Vec<_>>());
    // The bridge is the intro Lakitu's trigger (x -544..545, y > 800, z
    // -2000..-177): get his dialog over with first (A on alternate frames).
    let mut sim = Sim { g: gg.copy_from(g)?, f: st.frame, st: st.clone(), bias: 0 };
    sim.teleport([0.0, 906.0, -1200.0], 0x8000)?;
    let mut n = 0;
    while n < 3000 && !(sim.st.controllable() && sim.st.action == ACT_IDLE_ && n > 30) {
        sim.pad(GamePad::new(if n % 2 == 0 { A_BUTTON } else { 0 }, 0, 0))?;
        n += 1;
    }
    println!("Lakitu's dialog done after {n} frames ({})", act_name(sim.st.action));
    let st = &sim.st.clone();
    let save0 = sim.g.save_state();
    let mut rows = vec![];
    let mut all_moves: Vec<(String, f32, f32)> = vec![];
    // Starts on the door's axis (the doors' normal is +z), 800-2500 out, and off it.
    for &d in &[800.0f32, 1200.0, 1600.0, 2000.0, 2500.0] {
        for &(dx, face) in &[(0.0f32, 0x8000i32), (-300.0, 0x8000), (300.0, 0x8000), (150.0, 0x0000)] {
            let door = doors.iter().min_by(|a, b| (a.pos[0] - dx).abs().total_cmp(&(b.pos[0] - dx).abs())).unwrap();
            let z = door.pos[2] + d;
            let Some(y) = obrrt_core::geo::floor_below(&tris, dx, z, 3000.0) else { continue };
            let p = [dx, y, z];
            let mut sim = Sim::new(gg, &save0, st)?;
            sim.teleport(p, face)?;
            let save = sim.g.save_state();
            let s0 = sim.st.clone();
            if trace {
                eprintln!("teleported: {} dialog {} pos {:?}", act_name(s0.action), s0.dialog, s0.pos);
            }
            let base = door_run(gg, &save, &s0, door.pos, door.yaw, false, &tris, false)?;
            if trace {
                eprintln!("start {p:?} face {face:#x}");
            }
            let plan = door_run(gg, &save, &s0, door.pos, door.yaw, true, &tris, trace)?;
            let moves: Vec<String> = plan.moves.iter().map(|m| m.0.clone()).collect();
            println!(
                "start d {d:>4} x {dx:>5} face {face:#06x}: run straight {:>4} frames, plan {:>4} frames ({}) end {} moves {:?}",
                base.frames.map_or("--".into(), |f| f.to_string()),
                plan.frames.map_or("--".into(), |f| f.to_string()),
                match (base.frames, plan.frames) {
                    (Some(a), Some(b)) => format!("{:+}", b as i32 - a as i32),
                    _ => "n/a".into(),
                },
                act_name(plan.end.action),
                moves
            );
            rows.push((d, dx, base.frames, plan.frames));
            all_moves.extend(plan.moves);
        }
    }
    // Lobby doors: walk in through the front door, then plan to every
    // ordinary door that needs no stars from starts along its normal.
    {
        let door = &doors[0];
        let p = [0.0, 803.0, door.pos[2] + 600.0];
        let mut sim = Sim::new(gg, &save0, st)?;
        sim.teleport(p, 0x8000)?;
        let mut n = 0;
        while n < 600 && sim.st.zone() == st.zone() {
            let s = sim.st.clone();
            let to = [door.pos[0] - s.pos[0], door.pos[2] - s.pos[2]];
            let yaw = (to[0].atan2(to[1]) * 65536.0 / std::f32::consts::TAU) as i16;
            let (x, y) = arcs::stick(s.cam_yaw, yaw, 1.0);
            sim.pad(GamePad::new(0, x, y))?;
            n += 1;
        }
        let mut k = 0;
        while k < 600 && !(sim.st.controllable() && sim.st.action == ACT_IDLE_ && k > 30) {
            sim.pad(GamePad::new(0, 0, 0))?;
            k += 1;
        }
        let lobby = read_state(&sim.g, sim.f, true);
        let lsave = sim.g.save_state();
        let (ltris, _) = sim.g.with_memory(obrrt_core::geo::read_floors);
        println!("\nlobby: level {} area {} pos {:?} after {} + {k} frames", lobby.level, lobby.area, lobby.pos, n);
        for o in lobby.objects().iter().filter(|o| matches!(o.bhv_name(), "bhvDoor" | "bhvStarDoor") && o.beh_params >> 24 == 0) {
            let a = o.yaw as f32 * std::f32::consts::TAU / 65536.0;
            let nrm = [a.sin(), a.cos()];
            for side in [1.0f32, -1.0] {
                for &d in &[600.0f32, 900.0, 1200.0] {
                    let (x, z) = (o.pos[0] + nrm[0] * side * d, o.pos[2] + nrm[1] * side * d);
                    let Some(y) = obrrt_core::geo::floor_below(&ltris, x, z, o.pos[1] + 100.0) else { continue };
                    // Flat, floored straight path to the door.
                    let flat = (1..20).all(|j| {
                        let t = j as f32 / 20.0;
                        obrrt_core::geo::floor_below(&ltris, x + (o.pos[0] - x) * t, z + (o.pos[2] - z) * t, o.pos[1] + 100.0).is_some_and(|v| (v - o.pos[1]).abs() < 30.0)
                    });
                    if (y - o.pos[1]).abs() > 30.0 || !flat {
                        continue;
                    }
                    let mut sim = Sim::new(gg, &lsave, &lobby)?;
                    let face = ((-nrm[0] * side).atan2(-nrm[1] * side) * 65536.0 / std::f32::consts::TAU) as i32;
                    sim.teleport([x, y, z], face)?;
                    let save = sim.g.save_state();
                    let s0 = sim.st.clone();
                    let base = door_run(gg, &save, &s0, o.pos, o.yaw, false, &ltris, false)?;
                    let plan = door_run(gg, &save, &s0, o.pos, o.yaw, true, &ltris, false)?;
                    println!(
                        "lobby door {:?} side {side:+} d {d:>4}: run straight {:>4} frames, plan {:>4} frames moves {:?}",
                        o.pos, base.frames.map_or("--".into(), |f| f.to_string()), plan.frames.map_or("--".into(), |f| f.to_string()),
                        plan.moves.iter().map(|m| m.0.clone()).collect::<Vec<_>>()
                    );
                    rows.push((d, x, base.frames, plan.frames));
                    all_moves.extend(plan.moves);
                }
            }
        }
    }
    let ok = rows.iter().filter(|r| r.3.is_some()).count();
    let base_ok = rows.iter().filter(|r| r.2.is_some()).count();
    let both: Vec<_> = rows.iter().filter_map(|r| Some((r.2?, r.3?))).collect();
    let saved: i32 = both.iter().map(|&(a, b)| a as i32 - b as i32).sum();
    println!("\nentered: plan {ok}/{} starts, run straight {base_ok}/{}; frames saved over {} starts entered both ways: {} total ({:.1} a start)",
        rows.len(), rows.len(), both.len(), saved, saved as f32 / both.len().max(1) as f32);
    println!("\nlanding distance, game vs predicted, in the door runs (units):");
    let mut by: std::collections::BTreeMap<String, Vec<f32>> = Default::default();
    for (k, d, p) in &all_moves {
        by.entry(k.clone()).or_default().push(d - p);
    }
    for (k, v) in by {
        let mean = v.iter().map(|e| e.abs()).sum::<f32>() / v.len() as f32;
        let max = v.iter().map(|e| e.abs()).fold(0.0f32, f32::max);
        println!("  {k:<24} n {:>3}  mean |err| {:>7.2}  max {:>7.2}", v.len(), mean, max);
    }
    Ok(())
}
