//! Riser BLJ lab (ref/blj-conditions.md section 6): the riser probe of an
//! external script (blj.py `blj_risers`) over the static walls / floors of a
//! zone, and its setup (`landing`) + a deterministic `mash_gaps` (fixed beam,
//! fixed press count, no wall-clock budget) from TAS states near each riser.
//!
//! riser_lab <rom> <movie.m64> list  [zones=6:1,6:3,7:1,9:1,24:1] [hmax=60] [k=15]
//! riser_lab <rom> <movie.m64> sweep [zones=...] [hmax=60] [k=6] [combos=24]
//!     [beam=24] [presses=16] [want=300] [reach=1200] [threads=4]
//!     [gapstick=script|neutral] [only=x,z] [onlyr=200]
//! riser_lab <rom> tree:<run_dir>:<node> sweep ...   (one start: that node)
//! riser_lab <rom> <movie.m64> corners [zones=...] [reach=1200] [variants=16]
//!     [pb=1] [want=300]   the wall-pinned BLJ (riser::corners) with the
//!     deterministic rule controller ([`pin_chain`]) over setup variants
//! riser_lab <rom> <movie.m64> ga [zones=...] [seeds=2] [maxstarts=40]
//!     [frames=f1,f2,..]   BljAny objectives (set OBRRT_BLJ_WALLS=1 and/or
//!     OBRRT_BLJ_RISERS=1) generated at TAS states near corners, each run as
//!     the tree runs it (one GA attempt, scale 3); per kind: attempts, tree
//!     successes, chains to <= -150 / -300, best speed
//!
//! `list`: every riser of each zone (the probe), with the known BLJ spots
//! of the doc's section 3.1 and whether a riser is within 150 of each.
//! `sweep`: per zone the `k` best-ranked risers that have a grounded TAS
//! frame within `reach` (plus any riser within 200 of a known spot); from
//! that frame the script's setup + mash for `combos` (phi, d0, k, along,
//! mode, psi) combos in its spread order; reports the best forward speed,
//! frames to <= -want, landing shapes accepted.

use std::collections::{BTreeMap, HashMap};

use anyhow::{Result, bail};
use obrrt_core::geo::Tri;
use obrrt_core::{GamePad, St, read_state};
use obrrt_objectives::extra::tech::bljany::riser::{self, Riser};
use sm64_binds::{SM64Game, SM64GameGenerator};

const ACT_IDLE: u32 = 0x0C40_0201;
const ACT_LONG_JUMP: u32 = 0x0300_0888;
const ACT_LONG_JUMP_LAND: u32 = 0x0000_0479;
const ACT_FLAG_AIR: u32 = 0x800;
const A: u16 = 0x8000;
const Z: u16 = 0x2000;

/// Known spots (ref/blj-conditions.md 3.1, blj.rs SPOTS): zone, point, name.
const KNOWN: &[((u16, u8), [f32; 3], &str)] = &[
    ((6, 1), [-2258.0, 0.0, 326.0], "lobby left small steps"),
    ((6, 1), [-2687.0, -409.0, -3600.0], "1F west seam (ramp)"),
    ((6, 1), [183.0, 0.0, 302.0], "lobby centre steps"),
    ((6, 1), [538.0, 77.0, -1842.0], "WF-room steps"),
    ((6, 1), [-2379.0, 51.0, 285.0], "lobby left 51/77"),
    ((6, 3), [3277.0, -1279.0, -2569.0], "basement steps"),
    ((7, 1), [-6725.0, 1312.0, -2640.0], "HMC lower steps"),
    ((24, 1), [4460.0, 256.0, 4097.0], "WF stairs"),
    ((9, 1), [1612.0, 427.0, 4466.0], "BoB elevator (object)"),
];

fn arg<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter().find_map(|a| a.strip_prefix(k).and_then(|r| r.strip_prefix('=')))
}
fn argf(args: &[String], k: &str, d: f32) -> f32 {
    arg(args, k).map_or(d, |v| v.parse().unwrap())
}

fn movie(path: &str) -> Result<Vec<GamePad>> {
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

fn rd_u32(m: &[u8], a: usize) -> u32 {
    u32::from_le_bytes(m[a..a + 4].try_into().unwrap())
}

/// The zone's static floors and walls if the partition is surely this
/// area's (Mario on a static floor that is in it), as geo::observe_floors.
fn snapshot(g: &SM64Game) -> Option<(Vec<Tri>, Vec<Tri>)> {
    g.with_memory(|m| {
        let mario = obrrt_core::layout::ADDR_gMarioStates as usize;
        let floor = rd_u32(m, mario + 0x68);
        if floor == 0 || rd_u32(m, floor as usize + 0x2C) != 0 {
            return None;
        }
        let (tris, seen) = obrrt_core::geo::read_floors(m);
        seen.contains(&floor).then(|| (tris, obrrt_core::geo::read_walls(m)))
    })
}

fn yaw_to(a: [f32; 3], b: [f32; 3]) -> i32 {
    ((b[0] - a[0]).atan2(b[2] - a[2]) * 32768.0 / std::f32::consts::PI) as i32
}
fn hd(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).hypot(a[2] - b[2])
}

/// The raw stick whose intendedYaw is `yaw` and intendedMag `mag` (0..32):
/// intendedMag = stickMag^2 / 128 of the adjusted stick (raw -+ 6, <8 -> 0,
/// clamped to 64; update_mario_joystick_inputs / adjust_analog_stick).
fn stick(cam_yaw: i16, yaw: i32, mag: f32) -> (i8, i8) {
    let rel = (yaw - cam_yaw as i32) as f32 * std::f32::consts::TAU / 65536.0;
    let t = if mag >= 32.0 { 80.0 } else { (128.0 * mag.max(0.0)).sqrt() };
    let raw = |v: f32| if v.abs() < 0.5 { 0 } else { (v + 6.0 * v.signum()).round().clamp(-127.0, 127.0) as i8 };
    (raw(t * rel.sin()), raw(-t * rel.cos()))
}

/// One emulator with its frame count and the current state.
struct Sim {
    g: SM64Game,
    frame: u32,
    s: St,
    frames_run: u64,
}

impl Sim {
    fn new(g: SM64Game, frame: u32) -> Sim {
        let s = read_state(&g, frame, false);
        Sim { g, frame, s, frames_run: 0 }
    }
    fn step(&mut self, p: GamePad) {
        self.g.step_game(p).unwrap();
        self.frame += 1;
        self.frames_run += 1;
        self.s = read_state(&self.g, self.frame, false);
    }
    /// builder_core `run.press(yaw|None, buttons, mag)`: returns the pad.
    fn press(&mut self, yaw: Option<i32>, buttons: u16, mag: f32) -> GamePad {
        let (x, y) = yaw.map_or((0, 0), |w| stick(self.s.cam_yaw, w, mag));
        let p = GamePad::new(buttons, x, y);
        self.step(p);
        p
    }
    fn save(&self) -> (sm64_binds::SaveState, u32, St) {
        (self.g.save_state(), self.frame, self.s.clone())
    }
    fn load(&mut self, st: &(sm64_binds::SaveState, u32, St)) {
        self.g.load_state(&st.0).unwrap();
        self.frame = st.1;
        self.s = st.2.clone();
    }
    fn idle(&self) -> bool {
        self.s.forward_vel == 0.0 && self.s.action == ACT_IDLE
    }
}

type Snap = (sm64_binds::SaveState, u32, St);

fn idle_after_release(m: &mut Sim) -> Option<[f32; 3]> {
    for _ in 0..60 {
        if m.idle() {
            return Some(m.s.pos);
        }
        m.press(None, 0, 0.0);
    }
    None
}

/// blj.py `stop_near` (predictive release, fine = 0).
fn stop_near(m: &mut Sim, inp: &mut Vec<GamePad>, pt: [f32; 3], release: f32) -> bool {
    let brake = |m: &mut Sim, inp: &mut Vec<GamePad>| {
        for _ in 0..150 {
            if m.idle() {
                return true;
            }
            inp.push(m.press(None, 0, 0.0));
        }
        false
    };
    for _ in 0..8 {
        if hd(m.s.pos, pt) <= release && m.idle() {
            return true;
        }
        let mut best: Option<(f32, usize)> = None;
        let s0 = m.save();
        let mut walked = Vec::new();
        for k in 0..400 {
            let d = hd(m.s.pos, pt);
            let s1 = m.save();
            let q = idle_after_release(m);
            m.load(&s1);
            if let Some(q) = q {
                let dq = hd(q, pt);
                if best.is_none_or(|b| dq < b.0) {
                    best = Some((dq, k));
                }
                if best.unwrap().0 <= release && dq > best.unwrap().0 + 20.0 {
                    break;
                }
            }
            if d <= release && m.s.forward_vel < 2.0 {
                break;
            }
            let mag = if d > 150.0 { 32.0 } else { (d / 5.0).clamp(1.5, 32.0) };
            walked.push(m.press(Some(yaw_to(m.s.pos, pt)), 0, mag));
            if d < 2.0 {
                break;
            }
        }
        m.load(&s0);
        let k_rel = best.map_or(walked.len(), |b| b.1);
        for p in &walked[..k_rel] {
            m.step(*p);
            inp.push(*p);
        }
        if !brake(m, inp) {
            return false;
        }
    }
    hd(m.s.pos, pt) <= 4.0 * release
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Prm {
    phi: i32,
    d0: f32,
    k: u32,
    along: f32,
    stair: bool,
    psi: i32,
}

struct Land {
    d: f32,
    rel: i32,
    ljl: bool,
    fv: f32,
}

/// blj.py `landing`: stop at the setup point, face `away + phi`, walk k,
/// Z, Z+A, then Z with the stick toward the riser (+psi) until landing.
fn landing(m: &mut Sim, s0: &Snap, r: &Riser, p: &Prm, inp: &mut Vec<GamePad>) -> Option<Land> {
    m.load(s0);
    let pt = r.point(p.along, p.d0);
    if !stop_near(m, inp, pt, 5.0) {
        return None;
    }
    let away = (r.away_yaw() + p.phi) & 0xFFFF;
    for _ in 0..p.k {
        inp.push(m.press(Some(away), 0, 32.0));
    }
    inp.push(m.press(Some(away), Z, 32.0));
    inp.push(m.press(Some(away), Z | A, 32.0));
    let toward = (away + 0x8000 + p.psi) & 0xFFFF;
    for _ in 0..70 {
        if m.s.action & ACT_FLAG_AIR == 0 {
            break;
        }
        inp.push(m.press(Some(toward), Z, 32.0));
    }
    let s = &m.s;
    let rel = ((s.yaw as i32 + 0x8000 - r.away_yaw()) & 0xFFFF) as i32;
    Some(Land { d: r.front(s.pos[0], s.pos[2]), rel, ljl: s.action == ACT_LONG_JUMP_LAND, fv: s.forward_vel })
}

fn wrap(a: i32) -> i32 {
    ((a + 0x8000).rem_euclid(0x10000)) - 0x8000
}

/// blj.py `_shape_ok`.
fn shape_ok(l: &Land, stair: bool) -> bool {
    if !l.ljl {
        return false;
    }
    if stair {
        let into = wrap(l.rel - 0x8000).abs();
        return (0.0..=60.0).contains(&l.d) && into < 0x1800;
    }
    let par = wrap(l.rel - 0x4000).abs().min(wrap(l.rel - 0xC000).abs());
    (-1.0..=8.0).contains(&l.d) && par < 0x500
}

struct MashCfg {
    beam: usize,
    presses: usize,
    want: f32,
    climb: f32,
    neutral_gaps: bool,
}

const STICKS: [i32; 7] = [0, 0x800, -0x800, 0x1000, -0x1000, 0x3000, -0x3000];

/// blj.py `_mash_gaps` with a fixed beam and press count: the best forward
/// speed and its inputs.
fn mash(m: &mut Sim, s0: &Snap, c: &MashCfg) -> (f32, Vec<GamePad>) {
    m.load(s0);
    let y0 = m.s.pos[1];
    let mut beam: Vec<(Vec<GamePad>, Snap)> = vec![(Vec::new(), s0.clone())];
    let mut best = (0.0f32, Vec::new());
    for np in 0..c.presses {
        // (fv, parent beam index, its new frames, rounded position)
        let mut kids: Vec<(f32, usize, Vec<GamePad>, (i32, i32, i32))> = Vec::new();
        let gaps: Vec<usize> = if np == 0 { (0..13).collect() } else { vec![1, 2, 3, 4] };
        for (bi, (inp, sv)) in beam.iter().enumerate() {
            for &gp in &gaps {
                for &so in &STICKS {
                    m.load(sv);
                    let mut out = Vec::new();
                    let mut ok = true;
                    for k in 0..=gp {
                        let yaw = (m.s.yaw as i32 + 0x8000 + so) & 0xFFFF;
                        let press = k == gp;
                        let p = if press || !c.neutral_gaps { m.press(Some(yaw), if press { A | Z } else { Z }, 32.0) } else { m.press(None, Z, 0.0) };
                        out.push(p);
                        if m.s.pos[1] - y0 > c.climb || !matches!(m.s.action, ACT_LONG_JUMP | ACT_LONG_JUMP_LAND) {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        continue;
                    }
                    let fv = m.s.forward_vel;
                    let pk = m.s.pos;
                    if fv < best.0 {
                        let sx = m.save();
                        let f2 = m.press(None, Z, 0.0);
                        let full = || inp.iter().chain(out.iter()).copied().collect::<Vec<_>>();
                        if m.s.action & ACT_FLAG_AIR == 0 && m.s.action == ACT_LONG_JUMP_LAND && m.s.forward_vel < best.0 {
                            let mut f = full();
                            f.push(f2);
                            best = (m.s.forward_vel, f);
                        } else if m.s.action & ACT_FLAG_AIR != 0 {
                            m.load(&sx);
                            let mut pa = m.s.pos;
                            let mut free = true;
                            for _ in 0..3 {
                                m.press(None, 0, 0.0);
                                let pb = m.s.pos;
                                if hd(pa, pb) < 0.5 * fv.abs() {
                                    free = false;
                                    break;
                                }
                                pa = pb;
                            }
                            if free {
                                best = (fv, full());
                            }
                        }
                    }
                    kids.push((fv, bi, out, (pk[0].round() as i32, pk[1].round() as i32, pk[2].round() as i32)));
                }
            }
        }
        if best.0 <= -c.want || kids.is_empty() {
            break;
        }
        kids.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut keys = std::collections::HashSet::new();
        let mut next = Vec::new();
        for (_, bi, out, key) in kids {
            if !keys.insert(key) {
                continue;
            }
            m.load(&beam[bi].1);
            for p in &out {
                m.step(*p);
            }
            let mut full = beam[bi].0.clone();
            full.extend(out);
            next.push((full, m.save()));
            if next.len() >= c.beam {
                break;
            }
        }
        beam = next;
    }
    m.load(s0);
    best
}

fn vdc(mut i: usize) -> f64 {
    let (mut f, mut r) = (1.0, 0.0);
    while i > 0 {
        f /= 2.0;
        r += f * (i % 2) as f64;
        i /= 2;
    }
    r
}

/// blj.py `riser_sweep`'s combo list in its spread order (offset 0).
fn combos() -> Vec<Prm> {
    let mut side = Vec::new();
    for al in [0.5, 0.33, 0.67] {
        for off in (0..=0x100).step_by(8) {
            for sgn in [1, -1] {
                for s in [0x4000, 0xC000] {
                    for d0 in (4..65).step_by(3) {
                        for k in [3, 5] {
                            side.push(Prm { phi: s + sgn * off, d0: d0 as f32, k, along: al, stair: false, psi: 0 });
                        }
                    }
                }
            }
        }
    }
    let mut drift = Vec::new();
    for al in [0.5, 0.7, 0.3] {
        for psi in [0x1000, -0x1000, 0x2000, -0x2000, 0x800, -0x800, 0x3000, -0x3000] {
            for off in (-0x100..=0x100).step_by(0x20) {
                for s in [0x4000, 0xC000] {
                    for d0 in [77.0, 60.0, 90.0, 45.0] {
                        for k in [4, 2, 6] {
                            drift.push(Prm { phi: s + off, d0, k, along: al, stair: false, psi });
                        }
                    }
                }
            }
        }
    }
    let mut stair = Vec::new();
    for al in [0.5, 0.33, 0.67] {
        for off in (-0x400..=0x400).step_by(0x40) {
            for d0 in [60.0, 90.0, 120.0, 40.0, 150.0] {
                for k in [2, 4, 6] {
                    stair.push(Prm { phi: off, d0, k, along: al, stair: true, psi: 0 });
                }
            }
        }
    }
    let mut all: Vec<Prm> = Vec::new();
    for i in 0..side.len().max(drift.len()).max(stair.len()) {
        for l in [&side, &drift, &stair] {
            if let Some(p) = l.get(i) {
                if !all.contains(p) {
                    all.push(*p);
                }
            }
        }
    }
    let n = all.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| vdc(a + 1).total_cmp(&vdc(b + 1)).then(a.cmp(&b)));
    order.into_iter().map(|i| all[i]).collect()
}

struct Outcome {
    best: f32,
    hit: Option<(usize, usize, Prm)>,
    shapes: usize,
    tried: usize,
    setup_fail: usize,
    best_prm: Option<Prm>,
    frames_run: u64,
}

fn sweep_one(g: SM64Game, start: &Snap, r: &Riser, cfg: &MashCfg, ncomb: usize, list: &[Prm]) -> Outcome {
    let mut m = Sim::new(g, start.1);
    m.load(start);
    let mut o = Outcome { best: 0.0, hit: None, shapes: 0, tried: 0, setup_fail: 0, best_prm: None, frames_run: 0 };
    for (ci, p) in list.iter().take(ncomb).enumerate() {
        o.tried += 1;
        let mut inp = Vec::new();
        let Some(l) = landing(&mut m, start, r, p, &mut inp) else {
            o.setup_fail += 1;
            continue;
        };
        let _ = l.fv;
        if !shape_ok(&l, p.stair) {
            continue;
        }
        o.shapes += 1;
        let s1 = m.save();
        let c = MashCfg { climb: if p.stair { 1200.0 } else { cfg.climb }, ..*cfg };
        let (fv, mm) = mash(&mut m, &s1, &c);
        if fv < o.best {
            o.best = fv;
            o.best_prm = Some(*p);
        }
        if fv <= -cfg.want || (trace_all() && fv < -50.0) {
            trace(&mut m, &s1, &mm, r, fv);
        }
        if fv <= -cfg.want {
            o.hit = Some((ci, inp.len() + mm.len(), *p));
            break;
        }
    }
    o.frames_run = m.frames_run;
    o
}

impl Clone for MashCfg {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for MashCfg {}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        bail!("usage: riser_lab <rom> <movie.m64|tree:dir:node> <list|sweep> [opts]");
    }
    let gen_ = SM64GameGenerator::from_file(&args[1])?;
    let mode = args[3].as_str();
    let hmax = argf(&args, "hmax", riser::H_MAX);
    let zones: Vec<(u16, u8)> = arg(&args, "zones")
        .unwrap_or("6:1,6:3,7:1,9:1,24:1")
        .split(',')
        .map(|z| {
            let (l, a) = z.split_once(':').unwrap();
            (l.parse().unwrap(), a.parse().unwrap())
        })
        .collect();
    let t0 = std::time::Instant::now();

    // Pass 1: replay; per zone the geometry and the grounded TAS frames.
    let mut geo: BTreeMap<(u16, u8), (Vec<Tri>, Vec<Tri>)> = BTreeMap::new();
    let mut frames: HashMap<(u16, u8), Vec<(u32, [f32; 3])>> = HashMap::new();
    let tree_start = args[2].strip_prefix("tree:").map(str::to_string);
    let pads: Vec<GamePad> = match &tree_start {
        Some(rest) => {
            let (dir, node) = rest.rsplit_once(':').unwrap();
            let node: usize = node.parse()?;
            let records = obrrt_search::route::read_tree_bin(&format!("{dir}/tree.bin"))?;
            let mut chain = vec![node + 1];
            while let Some(&r) = chain.last() {
                if r == 0 {
                    break;
                }
                let parent = records[r].0;
                chain.push(if parent >= 0 { parent as usize + 1 } else { 0 });
            }
            chain.iter().rev().flat_map(|r| records[*r].1.clone()).collect()
        }
        None => movie(&args[2])?,
    };
    let mut g = gen_.create_game()?;
    let mut last_needed = pads.len();
    let mut frame = 0u32;
    for (i, p) in pads.iter().enumerate() {
        if i >= last_needed {
            break;
        }
        g.step_game(*p)?;
        frame += 1;
        let s = read_state(&g, frame, false);
        let z = s.zone();
        if !zones.contains(&z) || !s.controllable() {
            continue;
        }
        if !geo.contains_key(&z) {
            if let Some(x) = snapshot(&g) {
                eprintln!("zone {z:?}: geometry at frame {frame}: {} floors, {} walls", x.0.len(), x.1.len());
                geo.insert(z, x);
            }
        }
        if s.action & ACT_FLAG_AIR == 0 && tree_start.is_none() {
            frames.entry(z).or_default().push((frame, s.pos));
        }
        if mode == "list" && geo.len() == zones.len() {
            last_needed = i + 1;
        }
    }
    if let Some(_) = &tree_start {
        let s = read_state(&g, frame, false);
        frames.entry(s.zone()).or_default().push((frame, s.pos));
    }
    eprintln!("replayed {frame} frames in {:.0} s", t0.elapsed().as_secs_f64());

    let all: BTreeMap<(u16, u8), Vec<Riser>> = geo.iter().map(|(z, (f, w))| (*z, riser::find(f, w, hmax))).collect();
    if mode == "list" {
        let k: usize = argf(&args, "k", 15.0) as usize;
        for (z, rs) in &all {
            let low = rs.iter().filter(|r| r.h() <= riser::H_LOW).count();
            println!("\n== zone {z:?}: {} risers (h <= {hmax}), {low} at h <= 40, {} by a corner", rs.len(), rs.iter().filter(|r| r.corner).count());
            for (kz, p, name) in KNOWN.iter().filter(|k| k.0 == *z) {
                let near: Vec<&Riser> = rs.iter().filter(|r| (r.ylo - p[1]).abs() < 120.0 && seg_dist(r, *p) < 150.0).collect();
                let _ = kz;
                println!("  known {name} {p:?}: {} risers within 150", near.len());
                for r in near.iter().take(4) {
                    println!("    {}", fmt_r(r));
                }
            }
            let near = frames.get(z).and_then(|v| v.first()).map(|f| f.1).unwrap_or([0.0; 3]);
            println!("  top {k} by the script's rank (near the zone's first TAS ground frame {near:.0?}):");
            for r in riser::rank(rs, near, 1e9, None, k) {
                println!("    {}", fmt_r(&r));
            }
        }
        return Ok(());
    }
    if mode == "ga" {
        return ga_mode(&gen_, &geo, &all, &frames, &pads, &args);
    }
    if mode == "corners" {
        return corners_mode(&gen_, &geo, &all, &frames, &pads, &args);
    }
    if mode != "sweep" {
        bail!("mode list|sweep|corners");
    }

    // Candidates: per zone, risers with a grounded TAS frame within reach.
    let k: usize = argf(&args, "k", 6.0) as usize;
    let reach = argf(&args, "reach", 1200.0);
    let only = arg(&args, "only").map(|v| {
        let p: Vec<f32> = v.split(',').map(|x| x.parse().unwrap()).collect();
        [p[0], 0.0, p[1]]
    });
    let onlyr = argf(&args, "onlyr", 200.0);
    let mut cands: Vec<((u16, u8), Riser, u32, f32, Option<&str>)> = Vec::new();
    for (z, rs) in &all {
        let Some(fr) = frames.get(z) else { continue };
        let mut v: Vec<(u32, f32, Riser, u32, Option<&str>)> = Vec::new();
        for r in rs {
            let c = r.mid();
            if only.is_some_and(|o| hd(o, c) > onlyr) {
                continue;
            }
            let Some(&(f, p)) = fr.iter().filter(|(_, p)| (p[1] - c[1]).abs() < 300.0).min_by(|a, b| hd(a.1, c).total_cmp(&hd(b.1, c))) else { continue };
            let d = hd(p, c);
            if d > reach {
                continue;
            }
            let known = KNOWN.iter().find(|kn| kn.0 == *z && (r.ylo - kn.1[1]).abs() < 120.0 && seg_dist(r, kn.1) < 200.0).map(|kn| kn.2);
            let key = if (r.ylo - p[1]).abs() <= 40.0 { 0 } else { 4 } + if r.h() <= riser::H_LOW { 0 } else { 2 } + if r.corner { 0 } else { 1 };
            v.push((key, d, *r, f, known));
        }
        v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let mut n = 0;
        for (_, d, r, f, known) in v {
            if n < k || known.is_some() || only.is_some() {
                cands.push((*z, r, f, d, known));
                n += 1;
            }
        }
    }
    eprintln!("{} candidates", cands.len());

    // Pass 2: the start states.
    let need: std::collections::BTreeSet<u32> = cands.iter().map(|c| c.2).collect();
    let mut starts: HashMap<u32, Snap> = HashMap::new();
    let mut g = gen_.create_game()?;
    let mut frame = 0u32;
    let maxf = *need.iter().max().unwrap_or(&0);
    for p in &pads {
        if frame >= maxf {
            break;
        }
        g.step_game(*p)?;
        frame += 1;
        if need.contains(&frame) {
            starts.insert(frame, (g.save_state(), frame, read_state(&g, frame, false)));
        }
    }
    let cfg = MashCfg {
        beam: argf(&args, "beam", 24.0) as usize,
        presses: argf(&args, "presses", 16.0) as usize,
        want: argf(&args, "want", 300.0),
        climb: 80.0,
        neutral_gaps: arg(&args, "gapstick") == Some("neutral"),
    };
    let ncomb = argf(&args, "combos", 24.0) as usize;
    let threads = argf(&args, "threads", 4.0) as usize;
    let list = combos();
    eprintln!("{} combos in the spread order; trying the first {ncomb}", list.len());
    let results: std::sync::Mutex<Vec<(usize, Outcome)>> = std::sync::Mutex::new(Vec::new());
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= cands.len() {
                    break;
                }
                let (_, r, f, _, _) = &cands[i];
                let g = gen_.create_game().unwrap();
                let o = sweep_one(g, &starts[f], r, &cfg, ncomb, &list);
                eprintln!("  done {i}: best {:.1}", o.best);
                results.lock().unwrap().push((i, o));
            });
        }
    });
    let mut res = results.into_inner().unwrap();
    res.sort_by_key(|r| r.0);
    println!("\n| zone | riser (mid, h, len, corner) | known | start f (dist) | tried | setup fail | shapes ok | best fv | <= -{} at combo / frames | params |", cfg.want);
    println!("|---|---|---|---|---|---|---|---|---|---|");
    let mut frames_total = 0;
    for (i, o) in &res {
        let (z, r, f, d, known) = &cands[*i];
        frames_total += o.frames_run;
        let c = r.mid();
        println!(
            "| {z:?} | ({:.0}, {:.0}, {:.0}) h {:.0} len {:.0}{} | {} | {f} ({d:.0}) | {} | {} | {} | {:.1} | {} | {} |",
            c[0],
            c[1],
            c[2],
            r.h(),
            r.len(),
            if r.corner { " corner" } else { "" },
            known.unwrap_or("-"),
            o.tried,
            o.setup_fail,
            o.shapes,
            o.best,
            o.hit.map_or("-".to_string(), |(ci, n, _)| format!("{ci} / {n}")),
            o.hit.map(|h| h.2).or(o.best_prm).map_or("-".to_string(), |p| format!(
                "{} phi {:#x} d0 {} k {} al {} psi {:#x}",
                if p.stair { "stair" } else { "side" },
                p.phi & 0xFFFF,
                p.d0,
                p.k,
                p.along,
                p.psi
            ))
        );
    }
    eprintln!("emulated {frames_total} frames in {:.0} s", t0.elapsed().as_secs_f64());
    Ok(())
}

/// Horizontal distance from `p` to the riser's segment.
fn seg_dist(r: &Riser, p: [f32; 3]) -> f32 {
    let (dx, dz) = (r.b[0] - r.a[0], r.b[1] - r.a[1]);
    let l2 = (dx * dx + dz * dz).max(1.0);
    let t = (((p[0] - r.a[0]) * dx + (p[2] - r.a[1]) * dz) / l2).clamp(0.0, 1.0);
    (p[0] - r.a[0] - t * dx).hypot(p[2] - r.a[1] - t * dz)
}

fn fmt_r(r: &Riser) -> String {
    let c = r.mid();
    format!(
        "mid ({:.0}, {:.0}, {:.0}) y {:.0}..{:.0} h {:.0} len {:.0} n ({:.2},{:.2}){}",
        c[0],
        c[1],
        c[2],
        r.ylo,
        r.yhi,
        r.h(),
        r.len(),
        r.n[0],
        r.n[1],
        if r.corner { " corner" } else { "" }
    )
}

fn trace_all() -> bool {
    std::env::args().any(|a| a == "traceall")
}

/// Replays a chain from its landing state, printing each frame: the pad
/// (buttons, the stick's offset from straight back), action, speed,
/// height, distance in front of the riser and along it.
fn trace(m: &mut Sim, s1: &Snap, mm: &[GamePad], r: &Riser, fv: f32) {
    m.load(s1);
    println!("trace (best {fv:.1}): riser mid {:.0?} h {:.0}", r.mid(), r.h());
    for (i, p) in mm.iter().enumerate() {
        let pre = m.s.clone();
        let off = if p.stick_x == 0 && p.stick_y == 0 {
            "   -   ".to_string()
        } else {
            let adj = |v: i8| if v.unsigned_abs() < 8 { 0.0 } else { v as f32 - 6.0 * (v as f32).signum() };
            let rel = (adj(p.stick_x).atan2(-adj(p.stick_y)) * 32768.0 / std::f32::consts::PI) as i32;
            format!("{:+#07x}", wrap(rel + pre.cam_yaw as i32 - (pre.yaw as i32 + 0x8000)))
        };
        m.step(*p);
        let s = &m.s;
        println!(
            "  {i:3} {}{} off {off} -> {:#010x} fv {:7.1} y {:6.1} d {:6.1} along {:6.1} vy {:5.1}",
            if p.button & A != 0 { "A" } else { "." },
            if p.button & Z != 0 { "Z" } else { "." },
            s.action,
            s.forward_vel,
            s.pos[1] - r.ylo,
            r.front(s.pos[0], s.pos[2]),
            r.along(s.pos[0], s.pos[2]),
            s.vel[1]
        );
    }
}

#[derive(Clone, Copy, Debug)]
struct PinVar {
    /// Pose: this far out of the pin along the wall's normal.
    x: f32,
    /// Walk frames before Z.
    k: u32,
    /// First flight: stick offset toward the step side.
    psi: i32,
    /// Chain press offset: 0 = the schedule (0x800 / 0x1000 / 0x3000 by
    /// speed), else this constant.
    off: i32,
    /// Gap frames: stick back + offset (true) or neutral.
    gap_stick: bool,
    /// Pose distance in front of the riser.
    d0: f32,
}

fn pin_vars() -> Vec<PinVar> {
    let mut v = Vec::new();
    for d0 in [3.0, 12.0] {
        for psi in [0, -0x300] {
            for k in [3, 4] {
                for x in [100.0, 60.0] {
                    v.push(PinVar { x, k, psi, off: 0, gap_stick: true, d0 });
                }
            }
        }
    }
    v
}

/// The stick yaw `back + |off|` turned toward the riser's step side (-n).
fn toward_step(c: &riser::Corner, back: i32, off: i32) -> i32 {
    let n = c.riser.n;
    let d = |y: i32| {
        let r = y as f32 * std::f32::consts::TAU / 65536.0;
        r.sin() * n[0] + r.cos() * n[1]
    };
    let (p, m) = (back + off.abs(), back - off.abs());
    let toward = if d(p) <= d(m) { p } else { m };
    if off >= 0 { toward } else if toward == p { m } else { p }
}

fn sched(fv: f32, off: i32) -> i32 {
    if off != 0 {
        return off;
    }
    if fv > -80.0 {
        0x800
    } else if fv > -180.0 {
        0x1000
    } else {
        0x3000
    }
}

/// Setup to the pinned pose, then the rule chain: press (A+Z, stick back
/// toward the step) whenever on the foot floor with A up; Z and the stick
/// otherwise. `pb`: pause-buffer a press whose last press landed at once
/// (objectives/src/extra/tech/pausebuf.rs). Returns (best fv, frames to
/// <= -want from the start, setup ok).
fn pin_chain(m: &mut Sim, start: &Snap, c: &riser::Corner, v: &PinVar, want: f32, pb: bool) -> (f32, Option<usize>, bool) {
    m.load(start);
    let mut inp = Vec::new();
    if !stop_near(m, &mut inp, c.out(v.x, v.d0), 5.0) {
        return (0.0, None, false);
    }
    let face = c.face_yaw() & 0xFFFF;
    for _ in 0..v.k {
        inp.push(m.press(Some(face), 0, 32.0));
    }
    inp.push(m.press(Some(face), Z, 32.0));
    inp.push(m.press(Some(face), Z | A, 32.0));
    if m.s.action != ACT_LONG_JUMP {
        return (0.0, None, false);
    }
    let mut n = 0;
    while m.s.action & ACT_FLAG_AIR != 0 && n < 70 {
        let back = toward_step(c, m.s.yaw as i32 + 0x8000, v.psi);
        inp.push(m.press(Some(back), Z, 32.0));
        n += 1;
    }
    let dbg = std::env::args().any(|a| a == "tracepin");
    let mut best = 0.0f32;
    let mut held_a = false;
    let mut last_press_at_once = false;
    if dbg {
        println!("variant {v:?}: setup {} frames", inp.len());
    }
    let mut tail = 0usize;
    for _ in 0..150 {
        let s = m.s.clone();
        if !matches!(s.action, ACT_LONG_JUMP | ACT_LONG_JUMP_LAND) {
            break;
        }
        let off = sched(s.forward_vel, v.off);
        let back = toward_step(c, s.yaw as i32 + 0x8000, off);
        let pad = if tail > 0 {
            let b = [Z, Z | 0x1000, Z][3 - tail];
            tail -= 1;
            m.step(GamePad::new(b, 0, 0));
            GamePad::new(b, 0, 0)
        } else if s.action == ACT_LONG_JUMP_LAND && !held_a && s.forward_vel < 0.0 && s.pos[1] <= c.riser.ylo + 2.0 {
            let buf = pb && last_press_at_once && s.play_mode == 0;
            // The press's sideways strain (10 sin(off) a frame) must
            // carry the jump frame across the riser plane: d + 1.
            let need = pin_off(c.riser.front(s.pos[0], s.pos[2]));
            let back = toward_step(c, s.yaw as i32 + 0x8000, need.max(sched(s.forward_vel, v.off)));
            let p = m.press(Some(back), A | Z | if buf { 0x1000 } else { 0 }, 32.0);
            last_press_at_once = m.s.action == ACT_LONG_JUMP_LAND;
            if buf {
                tail = 3;
            }
            p
        } else if v.gap_stick || s.action == ACT_LONG_JUMP {
            m.press(Some(back), Z, 32.0)
        } else {
            m.press(None, Z, 0.0)
        };
        held_a = pad.button & A != 0;
        inp.push(pad);
        if dbg {
            let q = &m.s;
            let out = (q.pos[0] - c.pin[0]) * c.wall_n[0] + (q.pos[2] - c.pin[2]) * c.wall_n[1];
            println!(
                "  {}{} -> {:#010x} fv {:7.1} y {:5.1} d {:6.2} out {:6.2} yaw-face {:+#06x} pos ({:.2}, {:.2}) vel ({:.2}, {:.2}) wall {:?}",
                if pad.button & A != 0 { "A" } else { "." },
                if pad.button & Z != 0 { "Z" } else { "." },
                q.action,
                q.forward_vel,
                q.pos[1] - c.riser.ylo,
                c.riser.front(q.pos[0], q.pos[2]),
                out,
                wrap(q.yaw as i32 - c.face_yaw()),
                q.pos[0],
                q.pos[2],
                q.vel[0],
                q.vel[2],
                q.wall_normal.map(|n| [(n[0] * 100.0).round() / 100.0, (n[2] * 100.0).round() / 100.0])
            );
        }
        if matches!(m.s.action, ACT_LONG_JUMP | ACT_LONG_JUMP_LAND) {
            best = best.min(m.s.forward_vel);
            if m.s.forward_vel <= -want {
                return (best, Some(inp.len()), true);
            }
        }
    }
    (best, None, true)
}

fn corners_mode(
    gen_: &SM64GameGenerator,
    geo: &BTreeMap<(u16, u8), (Vec<Tri>, Vec<Tri>)>,
    all: &BTreeMap<(u16, u8), Vec<Riser>>,
    frames: &HashMap<(u16, u8), Vec<(u32, [f32; 3])>>,
    pads: &[GamePad],
    args: &[String],
) -> Result<()> {
    let reach = argf(args, "reach", 1200.0);
    let want = argf(args, "want", 300.0);
    let nvar = argf(args, "variants", 16.0) as usize;
    let pb = arg(args, "pb") == Some("1");
    let mut cands = Vec::new();
    for (z, rs) in all {
        let cs = riser::corners(&geo[z].1, rs);
        println!("zone {z:?}: {} corners", cs.len());
        let Some(fr) = frames.get(z) else { continue };
        for c in cs {
            let pose = c.out(100.0, 3.0);
            let Some(&(f, p)) = fr.iter().filter(|(_, p)| (p[1] - pose[1]).abs() < 300.0).min_by(|a, b| hd(a.1, pose).total_cmp(&hd(b.1, pose))) else { continue };
            let only = arg(args, "only").map(|v| v.split(',').map(|x| x.parse::<f32>().unwrap()).collect::<Vec<_>>());
            if only.as_ref().is_some_and(|o| (c.pin[0] - o[0]).hypot(c.pin[2] - o[1]) > 30.0) {
                continue;
            }
            if hd(p, pose) <= reach {
                cands.push((*z, c, f, hd(p, pose)));
            }
        }
    }
    eprintln!("{} corners with a start", cands.len());
    let need: std::collections::BTreeSet<u32> = cands.iter().map(|c| c.2).collect();
    let mut starts: HashMap<u32, Snap> = HashMap::new();
    let mut g = gen_.create_game()?;
    let mut frame = 0u32;
    let maxf = need.iter().max().copied().unwrap_or(0);
    for p in pads {
        if frame >= maxf {
            break;
        }
        g.step_game(*p)?;
        frame += 1;
        if need.contains(&frame) {
            starts.insert(frame, (g.save_state(), frame, read_state(&g, frame, false)));
        }
    }
    let vars: Vec<PinVar> = pin_vars().into_iter().take(nvar).collect();
    let threads = argf(args, "threads", 4.0) as usize;
    let next = std::sync::atomic::AtomicUsize::new(0);
    let res = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= cands.len() {
                    break;
                }
                let (_, c, f, _) = &cands[i];
                let mut m = Sim::new(gen_.create_game().unwrap(), *f);
                let mut out = Vec::new();
                for v in &vars {
                    out.push(pin_chain(&mut m, &starts[f], c, v, want, pb));
                }
                res.lock().unwrap().push((i, out));
            });
        }
    });
    let mut res = res.into_inner().unwrap();
    res.sort_by_key(|r| r.0);
    println!("\n| zone | pin (riser h) | push off | start f (dist) | setup ok | hits / {} | first hit at variant | frames | best fv | per-variant best |", vars.len());
    println!("|---|---|---|---|---|---|---|---|---|---|");
    let mut by_budget = vec![0usize; vars.len()];
    for (i, out) in &res {
        let (z, c, f, d) = &cands[*i];
        let hits: Vec<usize> = out.iter().enumerate().filter(|(_, o)| o.1.is_some()).map(|(j, _)| j).collect();
        if let Some(&j) = hits.first() {
            for b in by_budget.iter_mut().skip(j) {
                *b += 1;
            }
        }
        println!(
            "| {z:?} | ({:.0}, {:.0}, {:.0}) h {:.0} | {} | {f} ({d:.0}) | {} | {} | {} | {} | {:.1} | {} |",
            c.pin[0],
            c.pin[1],
            c.pin[2],
            c.riser.h(),
            c.push_off,
            out.iter().filter(|o| o.2).count(),
            hits.len(),
            hits.first().map_or("-".into(), |j| j.to_string()),
            hits.first().map_or("-".into(), |&j| out[j].1.unwrap().to_string()),
            out.iter().map(|o| o.0).fold(0.0f32, f32::min),
            out.iter().map(|o| format!("{:.0}", o.0)).collect::<Vec<_>>().join(" ")
        );
    }
    // The geometric predicate against the rule's outcome (past -50).
    let (mut tp, mut fp, mut fneg, mut tn) = (0, 0, 0, 0);
    for (i, out) in &res {
        let works = out.iter().any(|o| o.0 <= -50.0);
        match (cands[*i].1.push_off, works) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fneg += 1,
            (false, false) => tn += 1,
        }
    }
    println!("\npush_off vs rule past -50: tp {tp} fp {fp} fn {fneg} tn {tn}");
    println!("\ncorners reaching -{want} within the first n variants: {}", by_budget.iter().enumerate().map(|(j, b)| format!("{}:{b}", j + 1)).collect::<Vec<_>>().join(" "));
    Ok(())
}

/// The press offset whose sideways strain (10 sin(off) a frame, all four
/// quarter steps) crosses `d` + 1 to the riser plane.
fn pin_off(d: f32) -> i32 {
    let s = ((d.max(0.0) + 1.0) / 10.0).min(1.0);
    (s.asin() * 32768.0 / std::f32::consts::PI) as i32
}

fn ga_mode(
    gen_: &SM64GameGenerator,
    geo: &BTreeMap<(u16, u8), (Vec<Tri>, Vec<Tri>)>,
    all: &BTreeMap<(u16, u8), Vec<Riser>>,
    frames: &HashMap<(u16, u8), Vec<(u32, [f32; 3])>>,
    pads: &[GamePad],
    args: &[String],
) -> Result<()> {
    use obrrt_core::Objective;
    use obrrt_search::ga::{Fitness, GaParams, Worker};
    use rand::SeedableRng;
    obrrt_objectives::moves::install();
    obrrt_core::geo::want_static_floors();
    let seeds = argf(args, "seeds", 2.0) as u64;
    let maxstarts = argf(args, "maxstarts", 40.0) as usize;
    // Starts: the TAS frame nearest each corner's pose (as `corners`).
    let mut need: Vec<u32> = match arg(args, "frames") {
        Some(v) => v.split(',').map(|x| x.parse().unwrap()).collect(),
        None => {
            let mut v = Vec::new();
            for (z, rs) in all {
                let Some(fr) = frames.get(z) else { continue };
                for c in riser::corners(&geo[z].1, rs) {
                    let pose = c.out(100.0, 3.0);
                    if let Some(&(f, p)) = fr.iter().filter(|(_, p)| (p[1] - pose[1]).abs() < 300.0).min_by(|a, b| hd(a.1, pose).total_cmp(&hd(b.1, pose))) {
                        if hd(p, pose) <= 1200.0 && !v.contains(&f) {
                            v.push(f);
                        }
                    }
                }
            }
            v
        }
    };
    need.sort();
    // Spread the cap over the list.
    if need.len() > maxstarts {
        let n = need.len();
        need = (0..maxstarts).map(|i| need[i * n / maxstarts]).collect();
    }
    let mut g = gen_.create_game()?;
    let mut frame = 0u32;
    let mut starts: Vec<(u32, sm64_binds::SaveState, St)> = Vec::new();
    let maxf = need.iter().max().copied().unwrap_or(0);
    for p in pads {
        if frame >= maxf {
            break;
        }
        g.step_game(*p)?;
        frame += 1;
        if need.contains(&frame) {
            // Settle first (the TAS may be mid-BLJ there): neutral until
            // idle, on a copy.
            let keep = g.save_state();
            let mut f2 = frame;
            for _ in 0..200 {
                let q = read_state(&g, f2, false);
                if q.controllable() && q.action == ACT_IDLE && q.forward_vel == 0.0 {
                    break;
                }
                g.step_game(GamePad::new(0, 0, 0))?;
                f2 += 1;
            }
            let st = read_state(&g, f2, true);
            eprintln!("start {frame} -> settled {f2} pos {:.0?} action {:#x}", st.pos, st.action);
            starts.push((f2, g.save_state(), st));
            g.load_state(&keep)?;
        }
    }
    obrrt_core::mover::publish();
    let params = GaParams { scale: 3.0, fitness: Fitness::Steps, new_ops: true, pink: 0.0, per_objective_budget: false };
    // (kind, attempts, successes, <= -150, <= -300, best)
    let mut tally: BTreeMap<String, (u32, u32, u32, u32, f32)> = BTreeMap::new();
    let mut w = Worker { base: gen_.create_game()?, tmp: gen_.create_game()? };
    let mut rows = Vec::new();
    for (f, save, st) in &starts {
        let mut objs: Vec<std::sync::Arc<dyn Objective>> = Vec::new();
        obrrt_objectives::extra::tech::bljany::BljAny::generate(st, &mut objs);
        for o in objs.iter().filter(|o| o.key().contains(" Wall (") || o.key().contains(" Riser (")) {
            let kind = if o.key().contains(" Wall (") { "Wall" } else if o.key().contains("Stair") { "Riser Stair" } else { "Riser Side" };
            for seed in 0..seeds {
                let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
                let at = w.attempt(&[o.as_ref()], st, save, &mut rng, params)?;
                let seg = &at.segments[0];
                // The kept segment's fastest long-jump speed.
                g.load_state(save)?;
                let mut best = 0.0f32;
                for (k, p) in seg.inputs.iter().enumerate() {
                    g.step_game(*p)?;
                    let q = read_state(&g, f + k as u32 + 1, false);
                    if matches!(q.action, ACT_LONG_JUMP | ACT_LONG_JUMP_LAND) {
                        best = best.min(q.forward_vel);
                    }
                }
                // Carry on past the kept end with the overlay alone (a
                // neutral genome) for 80 frames: how fast the chain gets.
                let mut cont = best;
                if let Some(ov) = o.pad_overlay() {
                    let mut traj = vec![st.clone()];
                    g.load_state(save)?;
                    let mut fr = *f;
                    let mut played: Vec<GamePad> = Vec::new();
                    for p in &seg.inputs {
                        g.step_game(*p)?;
                        fr += 1;
                        traj.push(read_state(&g, fr, false));
                        played.push(*p);
                    }
                    for _ in 0..80 {
                        let p = ov.pad(&traj, &played, GamePad::new(0, 0, 0));
                        g.step_game(p)?;
                        fr += 1;
                        let q = read_state(&g, fr, false);
                        if matches!(q.action, ACT_LONG_JUMP | ACT_LONG_JUMP_LAND) {
                            cont = cont.min(q.forward_vel);
                        }
                        traj.push(q);
                        played.push(p);
                    }
                }
                let best = cont;
                let t = tally.entry(kind.to_string()).or_default();
                t.0 += 1;
                t.1 += seg.success as u32;
                t.2 += (best <= -150.0) as u32;
                t.3 += (best <= -300.0) as u32;
                t.4 = t.4.min(best);
                rows.push(format!("| {f} | {} | {seed} | {} | {:.1} | {:.2} | {} |", o.key(), seg.success, best, seg.progress, at.frames));
                eprintln!("{}", rows.last().unwrap());
            }
        }
    }
    println!("| start f | objective | seed | success | best fv (kept + 80 frames of the overlay) | progress | attempt frames |\n|---|---|---|---|---|---|---|");
    for r in &rows {
        println!("{r}");
    }
    println!("\n| kind | attempts | successes | <= -150 | <= -300 | best |\n|---|---|---|---|---|---|");
    for (k, t) in &tally {
        println!("| {k} | {} | {} | {} | {} | {:.1} |", t.0, t.1, t.2, t.3, t.4);
    }
    Ok(())
}
