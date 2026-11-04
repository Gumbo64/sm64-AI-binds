//! Scratch lab (BLJ research, ref/blj-conditions.md): plays a scripted
//! backwards long jump from chosen start states and prints the speed per
//! jump with the floor/ceiling under Mario.
//!
//! blj_lab <rom> <movie.m64|raw> <frames|settle> <mode> [args]
//!   trace <from> <to>            replay the movie, printing frames from..to
//!                                (instrumented: floor y/normal/type/object, ceil)
//!   script <n> [opts]            from the boot state, play the scripted BLJ n frames
//!   opts: inject=x,y,z,yaw,v0    teleport (idle 3 frames) then set LONG_JUMP_LAND
//!         v0=<f>                 set LONG_JUMP_LAND with this speed in place
//!         gnd=back|neutral       stick on non-press landing frames (default neutral)
//!         quiet                  per-jump summary only
//!         mag=<0..1>             stick tilt in the air (default 1)
//!   slopes <n> <v0>              inject on many static floor triangles of the
//!                                boot area (facing downhill) and summarise
//!   scan x z                     static floor column at (x, z)
use anyhow::Result;
use obrrt_core::buttons::{A_BUTTON, Z_TRIG};
use obrrt_core::layout::ADDR_gMarioStates;
use obrrt_core::{GamePad, St, read_state};
use sm64_binds::{SM64Game, SM64GameGenerator};

const ACT_LONG_JUMP: u32 = 0x0300_0888;
const ACT_LONG_JUMP_LAND: u32 = 0x0000_0479;
const ACT_IDLE: u32 = 0x0C40_0201;
const M: usize = ADDR_gMarioStates as usize;

#[derive(Clone, Copy, Debug, Default)]
struct Ext {
    timer: u16,
    imag: f32,
    iyaw: i16,
    floor_y: f32,
    ceil_y: f32,
    fny: f32,
    ftype: i16,
    fobj: bool,
    wall: bool,
}

fn rd<const N: usize>(m: &[u8], a: usize) -> [u8; N] {
    m[a..a + N].try_into().unwrap()
}
fn f32_(m: &[u8], a: usize) -> f32 {
    f32::from_le_bytes(rd(m, a))
}
fn u32_(m: &[u8], a: usize) -> u32 {
    u32::from_le_bytes(rd(m, a))
}
fn i16_(m: &[u8], a: usize) -> i16 {
    i16::from_le_bytes(rd(m, a))
}

fn ext(g: &SM64Game) -> Ext {
    g.with_memory(|m| {
        let floor = u32_(m, M + 0x68) as usize;
        let (fny, ftype, fobj) = if floor != 0 && floor + 0x30 < m.len() {
            (f32_(m, floor + 0x20), i16_(m, floor), u32_(m, floor + 0x2C) != 0)
        } else {
            (f32::NAN, -1, false)
        };
        Ext {
            timer: u16::from_le_bytes(rd(m, M + 0x1A)),
            imag: f32_(m, M + 0x20),
            iyaw: i16_(m, M + 0x24),
            floor_y: f32_(m, M + 0x70),
            ceil_y: f32_(m, M + 0x6C),
            fny,
            ftype,
            fobj,
            wall: u32_(m, M + 0x60) != 0,
        }
    })
}

fn write(g: &mut SM64Game, addr: usize, bytes: &[u8]) -> Result<()> {
    let (store, _, memory) = g.wasm_parts();
    memory.view(store).write(addr as u64, bytes)?;
    Ok(())
}

/// The raw stick for intendedYaw = `yaw` at `tilt` (adjust_analog_stick
/// removes 6 per axis), camera-relative (as objectives/src/extra/tech/blj.rs).
fn stick(s: &St, yaw: i32, tilt: f32) -> (i8, i8) {
    let rel = (yaw - s.cam_yaw as i32) as f32 * std::f32::consts::TAU / 65536.0;
    let t = tilt * 121.0;
    let raw = |v: f32| if v.abs() < 0.5 { 0 } else { (v + 6.0 * v.signum()).round().clamp(-127.0, 127.0) as i8 };
    (raw(t * rel.sin()), raw(-t * rel.cos()))
}

struct Boot {
    game: SM64Game,
    frame: u32,
}

fn boot(rom: &str, movie: &str, frames: &str) -> Result<(SM64GameGenerator, Boot, Vec<GamePad>)> {
    let gen_ = SM64GameGenerator::from_file(rom)?;
    let mut game = gen_.create_game()?;
    let bytes = std::fs::read(movie)?;
    let body = if bytes.len() >= 0x400 && &bytes[..4] == b"M64\x1a" {
        let n = u32::from_le_bytes(bytes[0x18..0x1C].try_into().unwrap()) as usize;
        let d = &bytes[0x400..];
        d[..(n * 4).min(d.len() / 4 * 4)].to_vec()
    } else {
        bytes[..bytes.len() / 4 * 4].to_vec()
    };
    let pads: Vec<GamePad> =
        body.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect();
    let mut frame = 0u32;
    if frames == "settle" {
        for p in &pads {
            game.step_game(*p)?;
            frame += 1;
        }
        let mut still = 0;
        let mut st = read_state(&game, frame, false);
        while still < 10 || !st.controllable() {
            let before = st.pos;
            game.step_game(GamePad::new(0, 0, 0))?;
            frame += 1;
            st = read_state(&game, frame, false);
            still = if st.pos == before { still + 1 } else { 0 };
        }
    } else {
        let n: usize = frames.parse()?;
        for p in &pads[..n.min(pads.len())] {
            game.step_game(*p)?;
            frame += 1;
        }
    }
    Ok((gen_, Boot { game, frame }, pads))
}

fn line(f: u32, s: &St, e: &Ext, pad: Option<GamePad>) -> String {
    let p = pad.map_or(String::from("            "), |p| format!("{:04x}{:>4}{:>4}", p.button, p.stick_x, p.stick_y));
    let dyaw = (e.iyaw as i32 - s.yaw as i32) as i16;
    format!(
        "{f:6} {p} act {:#010x} fv {:8.2} vy {:6.1} pos {:8.1} {:7.1} {:8.1} flo {:7.1} ny {:.3} t{:#04x}{} ceil {:7.0} tmr {} dyaw {:#06x} mag {:.0}{}",
        s.action, s.forward_vel, s.vel[1], s.pos[0], s.pos[1], s.pos[2], e.floor_y, e.fny, e.ftype,
        if e.fobj { " OBJ" } else { "" }, e.ceil_y.min(99999.0), e.timer, dyaw as u16, e.imag, if e.wall { " wall" } else { "" }
    )
}

#[derive(Default, Debug, Clone)]
struct Jump {
    start_f: u32,
    v_in: f32,     // speed on the frame before the jump (landing)
    v_after: f32,  // speed after the jump frame (x1.5 + one air frame)
    air: u32,      // frames in LONG_JUMP incl. the jump frame (0 if landed same frame -> 1)
    y0: f32,
    y_land: f32,
    gnd: u32,      // landing frames before the next press
    pred: Option<u32>, // predicted air frames (static geometry)
}

struct Opts {
    smart: Option<u32>,
    gnd_back: bool,
    quiet: bool,
    mag: f32,
}

/// Scripted BLJ: Z held; in LONG_JUMP stick back; in LONG_JUMP_LAND press A
/// on the first frame A wasn't held the frame before (stick back with it),
/// stick neutral otherwise (common_landing_action: no friction < 16).
fn script(g: &mut SM64Game, mut frame: u32, n: u32, o: &Opts, geo: &Geo) -> Result<Vec<Jump>> {
    let mut jumps: Vec<Jump> = Vec::new();
    let mut held_a = false;
    let mut s = read_state(g, frame, false);
    let mut e = ext(g);
    let mut stop_after = None;
    let mut peak = 0f32;
    let mut traj = vec![s.clone()];
    for _ in 0..n {
        let back = stick(&s, s.yaw as i32 + 0x8000, o.mag);
        let pred = (s.action == ACT_LONG_JUMP_LAND).then(|| predict(geo, s.pos, s.yaw, s.forward_vel, o.mag, 40).map(|p| p.0)).flatten();
        let want = match o.smart {
            None => true,
            Some(k) => pred.is_some_and(|p| p <= k) || e.timer >= 4,
        };
        let pad = if s.action == ACT_LONG_JUMP_LAND && !held_a && want {
            GamePad::new(A_BUTTON | Z_TRIG, back.0, back.1)
        } else if s.action == ACT_LONG_JUMP_LAND && !o.gnd_back {
            GamePad::new(Z_TRIG, 0, 0)
        } else if s.action == ACT_LONG_JUMP || s.action == ACT_LONG_JUMP_LAND {
            GamePad::new(Z_TRIG, back.0, back.1)
        } else {
            GamePad::new(0, 0, 0)
        };
        let prev = s.clone();
        let pe = e;
        g.step_game(pad)?;
        frame += 1;
        held_a = pad.button & A_BUTTON != 0;
        s = read_state(g, frame, false);
        e = ext(g);
        traj.push(s.clone());
        // A jump happened this frame if A was pressed in LJ_LAND and the
        // new action is LONG_JUMP, or LJ_LAND again with vy > 0 (same-frame landing)
        let jumped = held_a && prev.action == ACT_LONG_JUMP_LAND
            && (s.action == ACT_LONG_JUMP || (s.action == ACT_LONG_JUMP_LAND && (s.forward_vel - prev.forward_vel * 1.5).abs() < 6.0 && prev.forward_vel.abs() > 1.0) || s.vel[1] > 20.0);
        if jumped {
            jumps.push(Jump { start_f: frame, v_in: prev.forward_vel, v_after: s.forward_vel, air: 1, y0: prev.pos[1], y_land: f32::NAN, gnd: 0, pred: predict(geo, prev.pos, prev.yaw, prev.forward_vel, o.mag, 40).map(|p| p.0) });
        } else if let Some(j) = jumps.last_mut() {
            if prev.action == ACT_LONG_JUMP {
                j.air += 1;
            }
            if s.action == ACT_LONG_JUMP_LAND && j.y_land.is_nan() {
                j.y_land = s.pos[1];
            }
            if s.action == ACT_LONG_JUMP_LAND && prev.action == ACT_LONG_JUMP_LAND {
                j.gnd += 1;
            }
        }
        if let Some(j) = jumps.last_mut() {
            if jumped && s.action == ACT_LONG_JUMP_LAND {
                j.y_land = s.pos[1];
            }
        }
        let _ = pe;
        if !o.quiet {
            println!("{}", line(frame, &s, &e, Some(pad)));
        }
        if s.action == ACT_LONG_JUMP || s.action == ACT_LONG_JUMP_LAND {
            peak = peak.min(s.forward_vel);
        }
        if stop_after.is_none() && !(s.action == ACT_LONG_JUMP || s.action == ACT_LONG_JUMP_LAND) {
            println!("  chain left at frame {frame}: action {:#010x} fv {:.1} pos {:?} ceil {:.0} wall {}", s.action, s.forward_vel, s.pos, e.ceil_y, e.wall);
            stop_after = Some(3);
        }
        if s.forward_vel.abs() > 5000.0 {
            println!("  |fv| > 5000, stop");
            break;
        }
        if let Some(k) = stop_after.as_mut() {
            if *k == 0 {
                break;
            }
            *k -= 1;
        }
    }
    let m = metric(&traj);
    println!("  peak fv {peak:.1}; jumps {}; metric {:.2} (peak {:.0} jumps {} grown {} alive {})", jumps.len(), m.score, m.best_peak, m.best_jumps, m.best_grown, m.alive);
    for (i, j) in jumps.iter().enumerate() {
        println!(
            "  jump {i:2} f{} v_in {:8.2} -> {:8.2} (x{:.3})  air {:2} pred {:>4} gnd {}  y0 {:7.1} land {:7.1} dy {:+6.1}",
            j.start_f, j.v_in, j.v_after, j.v_after / j.v_in.min(-0.01), j.air, j.pred.map_or("-".into(), |p| p.to_string()), j.gnd, j.y0, j.y_land, j.y_land - j.y0
        );
    }
    Ok(jumps)
}

fn set_lj_land(g: &mut SM64Game, v0: f32) -> Result<()> {
    write(g, M + 0x54, &v0.to_le_bytes())?;
    write(g, M + 0x0C, &ACT_LONG_JUMP_LAND.to_le_bytes())?;
    write(g, M + 0x1A, &0u16.to_le_bytes())?;
    write(g, M + 0x18, &0u16.to_le_bytes())?;
    Ok(())
}

fn inject(g: &mut SM64Game, frame: &mut u32, x: f32, y: f32, z: f32, yaw: i16, v0: f32) -> Result<()> {
    for k in 0..4 {
        write(g, M + 0x3C, &[x.to_le_bytes(), y.to_le_bytes(), z.to_le_bytes()].concat())?;
        write(g, M + 0x48, &[0f32.to_le_bytes(); 3].concat())?;
        write(g, M + 0x54, &0f32.to_le_bytes())?;
        write(g, M + 0x2E, &yaw.to_le_bytes())?;
        write(g, M + 0x0C, &ACT_IDLE.to_le_bytes())?;
        let _ = k;
        g.step_game(GamePad::new(0, 0, 0))?;
        *frame += 1;
    }
    write(g, M + 0x2E, &yaw.to_le_bytes())?;
    set_lj_land(g, v0)?;
    Ok(())
}

/// Highest static floor at (x, z) with height <= y + 78 (find_floor), from
/// the static partition triangles.
fn floor_at(tris: &[obrrt_core::geo::Tri], x: f32, y: f32, z: f32) -> Option<(f32, usize)> {
    let mut best: Option<(f32, usize)> = None;
    for (i, t) in tris.iter().enumerate() {
        if t.n[1] < 0.01 {
            continue;
        }
        let [a, b, c] = t.v;
        let cross = |p: [f32; 3], q: [f32; 3]| (q[2] - p[2]) * (x - q[0]) - (q[0] - p[0]) * (z - q[2]);
        let (c1, c2, c3) = (cross(a, b), cross(b, c), cross(c, a));
        if !((c1 >= 0.0 && c2 >= 0.0 && c3 >= 0.0) || (c1 <= 0.0 && c2 <= 0.0 && c3 <= 0.0)) {
            continue;
        }
        let off = -(t.n[0] * a[0] + t.n[1] * a[1] + t.n[2] * a[2]);
        let h = -(x * t.n[0] + t.n[2] * z + off) / t.n[1];
        if y - (h - 78.0) < 0.0 {
            continue;
        }
        if best.is_none_or(|(bh, _)| h > bh) {
            best = Some((h, i));
        }
    }
    best
}

/// Static ceilings (partition list 1), as Tri.
fn read_ceils(m: &[u8]) -> Vec<obrrt_core::geo::Tri> {
    let part = obrrt_core::layout::ADDR_gStaticSurfacePartition as usize;
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for cell in 0..256 {
        let mut node = u32_(m, part + cell * 24 + 8) as usize;
        let mut guard = 0;
        while node != 0 && guard < 100_000 {
            guard += 1;
            let surf = u32_(m, node + 4) as usize;
            if surf != 0 && seen.insert(surf) {
                let v = |k: usize| [i16_(m, surf + k) as f32, i16_(m, surf + k + 2) as f32, i16_(m, surf + k + 4) as f32];
                out.push(obrrt_core::geo::Tri { v: [v(0x0A), v(0x10), v(0x16)], n: [f32_(m, surf + 0x1C), f32_(m, surf + 0x20), f32_(m, surf + 0x24)], kind: i16_(m, surf) });
            }
            node = u32_(m, node) as usize;
        }
    }
    out
}

fn inside(t: &obrrt_core::geo::Tri, x: f32, z: f32) -> bool {
    let [a, b, c] = t.v;
    let cross = |p: [f32; 3], q: [f32; 3]| (q[2] - p[2]) * (x - q[0]) - (q[0] - p[0]) * (z - q[2]);
    let (c1, c2, c3) = (cross(a, b), cross(b, c), cross(c, a));
    (c1 >= 0.0 && c2 >= 0.0 && c3 >= 0.0) || (c1 <= 0.0 && c2 <= 0.0 && c3 <= 0.0)
}
fn plane_y(t: &obrrt_core::geo::Tri, x: f32, z: f32) -> f32 {
    let a = t.v[0];
    let off = -(t.n[0] * a[0] + t.n[1] * a[1] + t.n[2] * a[2]);
    -(x * t.n[0] + t.n[2] * z + off) / t.n[1]
}
/// Lowest static ceiling with height >= y - 78 (find_ceil, roughly: the game takes the first in list order).
fn ceil_at(ceils: &[obrrt_core::geo::Tri], x: f32, y: f32, z: f32) -> f32 {
    let mut best = 20000.0f32;
    for t in ceils {
        if t.n[1] >= 0.0 || !inside(t, x, z) {
            continue;
        }
        let h = plane_y(t, x, z);
        if h >= y - 78.0 && h < best {
            best = h;
        }
    }
    best
}

pub struct Geo {
    floors: Vec<obrrt_core::geo::Tri>,
    ceils: Vec<obrrt_core::geo::Tri>,
}

/// The long jump arc from a press on LONG_JUMP_LAND at `pos` facing `yaw`
/// with speed `v_land`, stick held back, over static floors and ceilings
/// (walls and objects ignored): Some((air frames, landing y, speed at
/// landing)), or None if it doesn't land within `max` frames.
fn predict(g: &Geo, pos: [f32; 3], yaw: i16, v_land: f32, mag: f32, max: u32) -> Option<(u32, f32, f32)> {
    let mut v = v_land * 1.5;
    if v > 48.0 {
        v = 48.0;
    }
    let mut vy = 30.0f32;
    let mut p = pos;
    let mut cur_floor = pos[1];
    let (sy, cy) = { let r = yaw as f32 * std::f32::consts::TAU / 65536.0; (r.sin(), r.cos()) };
    for n in 1..=max {
        v = if v > 0.35 { v - 0.35 } else if v < -0.35 { v + 0.35 } else { 0.0 };
        v -= 1.5 * mag;
        if v > 48.0 { v -= 1.0; }
        if v < -16.0 { v += 2.0; }
        for _ in 0..4 {
            let next = [p[0] + v * sy / 4.0, p[1] + vy / 4.0, p[2] + v * cy / 4.0];
            match floor_at(&g.floors, next[0], next[1], next[2]) {
                None => {
                    if next[1] <= cur_floor { return Some((n, cur_floor, v)); }
                    p[1] = next[1];
                    continue;
                }
                Some((fh, _)) => {
                    if next[1] <= fh { return Some((n, fh, v)); }
                    let ch = ceil_at(&g.ceils, next[0], fh + 80.0, next[2]);
                    if next[1] + 160.0 > ch {
                        if vy >= 0.0 { vy = 0.0; continue; }
                        if next[1] <= cur_floor { return Some((n, cur_floor, v)); }
                        p[1] = next[1];
                        continue;
                    }
                    p = next;
                    cur_floor = fh;
                }
            }
        }
        vy = (vy - 2.0).max(-75.0);
    }
    None
}

/// The proposed graded BLJ progress metric (ref/blj-conditions.md).
fn in_lj(s: &St) -> bool {
    s.action == ACT_LONG_JUMP || s.action == ACT_LONG_JUMP_LAND
}
fn blj_jump(p: &St, s: &St) -> bool {
    p.action == ACT_LONG_JUMP_LAND && in_lj(s) && p.forward_vel < -1.0
        && (s.forward_vel - 1.5 * p.forward_vel).abs() <= 4.5
        && (s.action == ACT_LONG_JUMP || s.vel[1] > 20.0)
}
#[derive(Debug, Default, Clone, Copy)]
struct Metric {
    score: f64,
    best_peak: f32,
    best_jumps: u32,
    best_grown: u32,
    alive: bool,
}
fn metric(traj: &[St]) -> Metric {
    // Per chain: landing speeds v_j after each BLJ jump (fv on the first
    // LONG_JUMP_LAND frame after the jump; same-frame landings count at once).
    let mut best = 0.0f64;
    let mut out = Metric::default();
    let (mut on, mut pending, mut peak, mut jumps, mut grown, mut last_land) = (false, false, 0f32, 0u32, 0u32, 0f32);
    let score = |peak: f32, grown: u32, jumps: u32| {
        (peak.max(16.0) / 16.0).ln() as f64 / 1.5f64.ln() + 0.25 * grown.min(20) as f64 + 0.1 * jumps.min(10) as f64
    };
    for w in traj.windows(2) {
        let (p, s) = (&w[0], &w[1]);
        if !in_lj(s) {
            on = false;
            continue;
        }
        if blj_jump(p, s) {
            if !on {
                on = true;
                peak = 0.0; jumps = 0; grown = 0; last_land = p.forward_vel;
            }
            jumps += 1;
            pending = true;
        }
        if on && pending && s.action == ACT_LONG_JUMP_LAND {
            pending = false;
            let v = s.forward_vel;
            if v < last_land - 0.5 && v < -16.0 {
                grown += 1;
            }
            last_land = v;
            peak = peak.max(-v);
        }
        if on {
            let c = score(peak, grown, jumps);
            if c > best {
                best = c;
                out.best_peak = peak; out.best_jumps = jumps; out.best_grown = grown;
            }
        }
    }
    let last = traj.last().unwrap();
    out.alive = on && in_lj(last) && last.forward_vel < -16.0;
    out.score = best + if out.alive { 0.5 } else { 0.0 };
    out
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (_gen, mut b, pads) = boot(&a[1], &a[2], &a[3])?;
    let mode = a[4].as_str();
    match mode {
        "trace" => {
            let (from, to): (u32, u32) = (a[5].parse()?, a[6].parse()?);
            let mut f = b.frame;
            let mut traj = vec![read_state(&b.game, f, false)];
            let mut last_score = -1.0;
            let quiet = a.get(7).is_some_and(|x| x == "metric");
            while (f as usize) < pads.len() && f < to {
                let p = pads[f as usize];
                b.game.step_game(p)?;
                f += 1;
                let s = read_state(&b.game, f, false);
                if f >= from {
                    if !quiet { println!("{}", line(f, &s, &ext(&b.game), Some(p))); }
                    traj.push(s);
                    let m = metric(&traj);
                    if (m.score - last_score).abs() > 1e-9 {
                        println!("  metric f{f}: {:.2} (peak {:.0}, jumps {}, grown {}, alive {}) act {:#x} fv {:.1}", m.score, m.best_peak, m.best_jumps, m.best_grown, m.alive, traj.last().unwrap().action, traj.last().unwrap().forward_vel);
                        last_score = m.score;
                    }
                }
            }
        }
        "scan" => {
            let (x, z): (f32, f32) = (a[5].parse()?, a[6].parse()?);
            let tris = b.game.with_memory(|m| obrrt_core::geo::read_floors(m).0);
            for t in &tris {
                let _ = t;
            }
            for dy in [-2000.0f32, -500.0, 0.0, 200.0, 500.0, 1000.0, 3000.0] {
                println!("y {dy}: {:?}", floor_at(&tris, x, dy, z).map(|(h, i)| (h, tris[i].n, tris[i].kind)));
            }
        }
        "script" => {
            let n: u32 = a[5].parse()?;
            let mut o = Opts { smart: None, gnd_back: false, quiet: false, mag: 1.0 };
            let mut f = b.frame;
            for arg in &a[6..] {
                if let Some(v) = arg.strip_prefix("inject=") {
                    let p: Vec<f32> = v.split(',').map(|x| x.parse().unwrap()).collect();
                    inject(&mut b.game, &mut f, p[0], p[1], p[2], p[3] as i32 as i16, p[4])?;
                } else if let Some(v) = arg.strip_prefix("v0=") {
                    set_lj_land(&mut b.game, v.parse()?)?;
                } else if let Some(v) = arg.strip_prefix("smart=") {
                    o.smart = Some(v.parse()?);
                } else if arg == "gnd=back" {
                    o.gnd_back = true;
                } else if arg == "quiet" {
                    o.quiet = true;
                } else if let Some(v) = arg.strip_prefix("mag=") {
                    o.mag = v.parse()?;
                }
            }
            let s = read_state(&b.game, f, false);
            println!("start {}", line(f, &s, &ext(&b.game), None));
            let geo = b.game.with_memory(|m| Geo { floors: obrrt_core::geo::read_floors(m).0, ceils: read_ceils(m) });
            script(&mut b.game, f, n, &o, &geo)?;
        }
        "slopes" => {
            // Inject on static floors of the boot area, facing downhill,
            // sorted by steepness; one line each.
            let n: u32 = a[5].parse()?;
            let v0: f32 = a[6].parse()?;
            let limit: usize = a.get(7).map_or(60, |s| s.parse().unwrap());
            let smart: Option<u32> = a.get(8).map(|s| s.parse().unwrap());
            let geo = b.game.with_memory(|m| Geo { floors: obrrt_core::geo::read_floors(m).0, ceils: read_ceils(m) });
            let tris = &geo.floors;
            let snap = b.game.save_state();
            let mut cands: Vec<usize> = (0..tris.len())
                .filter(|&i| {
                    let t = &tris[i];
                    let [p, q, r] = t.v;
                    let area = 0.5 * ((q[0] - p[0]) * (r[2] - p[2]) - (r[0] - p[0]) * (q[2] - p[2])).abs();
                    t.n[1] > 0.3 && t.n[1] < 0.995 && area > 250_000.0
                })
                .collect();
            cands.sort_by(|&i, &j| tris[i].n[1].partial_cmp(&tris[j].n[1]).unwrap());
            // spread over the steepness range
            let step = (cands.len() as f32 / limit as f32).max(1.0);
            let picks: Vec<usize> = (0..limit).map(|k| (k as f32 * step) as usize).filter(|&k| k < cands.len()).map(|k| cands[k]).collect();
            println!("{} candidate slopes, testing {}", cands.len(), picks.len());
            for i in picks {
                let t = &tris[i];
                let c = [0, 1, 2].map(|k| (t.v[0][k] + t.v[1][k] + t.v[2][k]) / 3.0);
                // facing downhill: floorAngle = atan2s(nz, nx)
                let yaw = (t.n[0].atan2(t.n[2]) * 32768.0 / std::f32::consts::PI) as i32 as i16;
                b.game.load_state(&snap)?;
                let mut f = b.frame;
                inject(&mut b.game, &mut f, c[0], c[1] + 5.0, c[2], yaw, v0)?;
                let st = read_state(&b.game, f, false);
                let e = ext(&b.game);
                if (st.pos[1] - c[1]).abs() > 30.0 || (e.fny - t.n[1]).abs() > 0.01 {
                    println!("tri {i}: ny {:.3} type {:#x} at {:?}: inject failed (y {:.0}, floor ny {:.3})", t.n[1], t.kind, c.map(|v| v as i32), st.pos[1], e.fny);
                    continue;
                }
                let o = Opts { smart: smart, gnd_back: false, quiet: true, mag: 1.0 };
                println!("== tri {i}: ny {:.3} ({:.1} deg) type {:#x} at {:?} yaw {yaw}", t.n[1], t.n[1].acos().to_degrees(), t.kind, c.map(|v| v as i32));
                script(&mut b.game, f, n, &o, &geo)?;
            }
        }
        _ => anyhow::bail!("unknown mode"),
    }
    Ok(())
}
