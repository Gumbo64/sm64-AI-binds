//! Problems: start states from real routes, split into training and held-out
//! cases (`gp/cases/<name>.txt`), and what is to be done from them.
//!
//! A case line: `train|test <movie> <frame> [task args]`: the state after
//! `frame` pads of the movie (the m64 format's, or raw).

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use obrrt_core::{Eta, GamePad, Generator, Mem, Objective, St, read_state};
use obrrt_objectives::Registry;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use sm64_binds::{SM64Game, SM64GameGenerator};

use crate::lang::{GenCfg, MAX_POINTS, RefNow};
use crate::rollout::{Case, End, PREFIX, Roll, Task};

/// The inputs of a Mupen64 movie or a raw dump.
pub fn read_movie(path: &str) -> Result<Vec<GamePad>> {
    let bytes = std::fs::read(path).with_context(|| path.to_string())?;
    let body = if bytes.len() >= 0x400 && &bytes[..4] == b"M64\x1a" {
        let samples = u32::from_le_bytes(bytes[0x18..0x1C].try_into().unwrap()) as usize;
        let data = &bytes[0x400..];
        &data[..(samples * 4).min(data.len() / 4 * 4)]
    } else {
        &bytes[..bytes.len() / 4 * 4]
    };
    Ok(body.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect())
}

/// A raw dump tasplay reads.
pub fn write_raw(path: &str, pads: &[GamePad]) -> Result<()> {
    let mut b = Vec::with_capacity(pads.len() * 4);
    for p in pads {
        b.extend_from_slice(&p.button.to_be_bytes());
        b.push(p.stick_x as u8);
        b.push(p.stick_y as u8);
    }
    std::fs::write(path, b)?;
    Ok(())
}

/// An objective of the Registry, with fixed points for the program.
pub struct ObjTask {
    pub obj: Arc<dyn Objective>,
    pub pts: [[f32; 3]; MAX_POINTS],
}

impl Task for ObjTask {
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        self.obj.eta(s, start, mem)
    }
    fn points(&self, _s: &St, _mem: &Mem) -> [[f32; 3]; MAX_POINTS] {
        self.pts
    }
    fn target_for_baseline(&self, s: &St) -> [f32; 3] {
        self.obj.steer_target(s).unwrap_or(self.pts[0])
    }
    fn overlay(&self) -> Option<obrrt_core::PadOverlay> {
        self.obj.pad_overlay()
    }
    fn objective(&self) -> Option<Arc<dyn Objective>> {
        Some(self.obj.clone())
    }
}

pub struct Problem {
    pub name: String,
    pub cases: Vec<Case>,
    pub tasks: Vec<Box<dyn Task>>,
    pub gen_cfg: GenCfg,
    pub roll: Roll,
    pub movies: Vec<(String, Vec<GamePad>)>,
}

const RAMP: [f32; 3] = [1600.0, 300.0, 4550.0];
const DOOR: [f32; 3] = [0.0, 803.0, -3155.0];
const RUN_C: [f32; 3] = [-79.4, 260.0, 4558.6];
const EXIT: [f32; 3] = [-38.0, 0.0, 3882.0];

fn objective(st: &St, key: &str) -> Result<Arc<dyn Objective>> {
    let mut rng = SmallRng::seed_from_u64(1);
    let all = Registry.generate(st, &mut rng);
    let found = all.into_iter().find(|o| o.key() == key);
    match found {
        Some(o) => Ok(o),
        None => bail!("objective {key} isn't generated at frame {} ({:?}, level {})", st.frame, st.pos, st.level),
    }
}

/// The world's point `d` along the yaw from `p`.
fn along(p: [f32; 3], yaw: i16, d: f32) -> [f32; 3] {
    let r = yaw as f32 * std::f32::consts::TAU / 65536.0;
    [p[0] + d * r.sin(), p[1], p[2] + d * r.cos()]
}

/// Builds a problem from its case file; `root` is objective-rrt-v2.
pub fn load(name: &str, root: &str, gen_gen: &SM64GameGenerator, skip_test: bool) -> Result<Problem> {
    // GP_CASES: another case file for this problem.
    let file = std::env::var("GP_CASES").unwrap_or_else(|_| format!("{root}/gp/cases/{name}.txt"));
    let text = std::fs::read_to_string(&file).with_context(|| file.clone())?;
    let mut movies: Vec<(String, Vec<GamePad>)> = Vec::new();
    let mut wanted: BTreeMap<usize, Vec<(usize, String, Vec<String>)>> = BTreeMap::new();
    for l in text.lines() {
        let l = l.split('#').next().unwrap_or("").trim();
        if l.is_empty() {
            continue;
        }
        let f: Vec<&str> = l.split_whitespace().collect();
        if skip_test && f[0] == "test" {
            continue;
        }
        let (split, m, frame): (String, &str, usize) = (f[0].to_string(), f[1], f[2].parse()?);
        let idx = match movies.iter().position(|(n, _)| n == m) {
            Some(i) => i,
            None => {
                movies.push((m.to_string(), read_movie(&format!("{root}/{m}"))?));
                movies.len() - 1
            }
        };
        if let Some(rm) = f[3..].iter().find_map(|a| a.strip_prefix("rmovie=")) {
            if !movies.iter().any(|(n, _)| n == rm) {
                movies.push((rm.to_string(), read_movie(&format!("{root}/{rm}"))?));
            }
        }
        wanted.entry(idx).or_default().push((frame, split, f[3..].iter().map(|s| s.to_string()).collect()));
    }
    let mut game = gen_gen.create_game()?;
    let power_on = game.save_state();
    let mut cases = Vec::new();
    let mut tasks: Vec<Box<dyn Task>> = Vec::new();
    // Grafts follow a trace other than the start's; states are kept only for those.
    let keep_traces = text.contains("rmovie=");
    let mut traces: BTreeMap<usize, Vec<St>> = BTreeMap::new();
    for (m, mut list) in wanted {
        list.sort_by_key(|x| x.0);
        let pads = movies[m].1.clone();
        // Follow cases: the movie's states, to take waypoints from.
        if name == "follow" && !traces.contains_key(&m) {
            let t = play_trace(&mut game, &power_on, &pads)?;
            traces.insert(m, t);
        }
        game.load_state(&power_on)?;
        let mut at = 0usize;
        let mut hist: Vec<St> = Vec::new();
        let mut last_save: Option<(usize, Arc<sm64_binds::SaveState>)> = None;
        for (frame, split, args) in list {
            step_to(&mut game, &pads, &mut at, frame, &mut hist)?;
            let opt = |k: &str| args.iter().find_map(|a| a.strip_prefix(&format!("{k}=")).map(String::from));
            // Noise: this many frames of a full stick at a random yaw, no
            // buttons, from the movie's state (a state on no route).
            let noise: Vec<GamePad> = match opt("noise") {
                Some(n) if opt("raw").is_some() => {
                    // noise=K raw=x:y (eval_follow.py's perturb arm).
                    let raw = opt("raw").unwrap();
                    let (x, y) = raw.split_once(':').unwrap();
                    vec![GamePad::new(0, x.parse()?, y.parse()?); n.parse()?]
                }
                Some(n) => {
                    let (k, seed) = n.split_once(':').unwrap_or((&n, "1"));
                    let (k, seed): (usize, u64) = (k.parse()?, seed.parse()?);
                    let mut r = SmallRng::seed_from_u64(seed ^ (frame as u64) << 16);
                    let yaw: f32 = rand::Rng::random_range(&mut r, 0.0..std::f32::consts::TAU);
                    let (x, y) = ((80.0 * yaw.sin()) as i8, (80.0 * yaw.cos()) as i8);
                    vec![GamePad::new(0, x, y); k]
                }
                None => Vec::new(),
            };
            // The noise is played again after loading (Case::noise), so one
            // save serves every case at this frame.
            let save = match &last_save {
                Some((f, s)) if *f == frame => s.clone(),
                _ => {
                    let s = Arc::new(game.save_state());
                    last_save = Some((frame, s.clone()));
                    s
                }
            };
            let (mut h2, mut p2) = (hist.clone(), pads[..frame].to_vec());
            for (k, p) in noise.iter().enumerate() {
                game.step_game(*p)?;
                p2.push(*p);
                h2.push(read_state(&game, (frame + k + 1) as u32, false));
            }
            if h2.len() > PREFIX {
                h2.drain(..h2.len() - PREFIX);
            }
            let f2 = frame + noise.len();
            let start = read_state(&game, f2 as u32, true);
            // Followers: the reference from past the noise, or another trace's (graft).
            let made = if let Some(rm) = opt("rmovie") {
                let idx = movies.iter().position(|(n, _)| *n == rm).ok_or_else(|| anyhow::anyhow!("rmovie {rm}"))?;
                if !traces.contains_key(&idx) {
                    let rp = movies[idx].1.clone();
                    let t = play_trace(&mut game, &power_on, &rp)?;
                    game.load_state(&save)?;
                    for p in &noise {
                        game.step_game(*p)?;
                    }
                    traces.insert(idx, t);
                }
                let rf: usize = opt("rframe").ok_or_else(|| anyhow::anyhow!("rframe"))?.parse()?;
                make_task(name, &start, &args, Some((&traces[&idx], &movies[idx].1)), rf)
            } else {
                make_task(name, &start, &args, traces.get(&m).map(|t| (t.as_slice(), pads.as_slice())), f2)
            };
            let task = match made {
                Ok(t) => Some(t),
                Err(e) => {
                    eprintln!("skip {} @{f2}: {e}", movies[m].0);
                    None
                }
            };
            if let Some(task) = task {
                let short = movies[m].0.rsplit('/').next().unwrap().trim_end_matches(".m64").to_string();
                let n = h2.len();
                let tag = args.iter().map(|a| format!(",{a}")).collect::<String>();
                cases.push(Case {
                    name: format!("{short}@{f2}{tag}"),
                    train: split == "train",
                    val: split == "val",
                    movie: m,
                    frame: f2,
                    noise: noise.clone(),
                    save: save.clone(),
                    pre_states: h2,
                    pre_pads: p2[f2 + 1 - n.min(PREFIX)..f2].to_vec(),
                    start: St { objects: None, ..start },
                    task: tasks.len(),
                    ref_pads: if noise.is_empty() { pads[frame..pads.len().min(frame + 1000)].to_vec() } else { Vec::new() },
                });
                tasks.push(task);
            }
            if !noise.is_empty() {
                game.load_state(&save)?;
            }
        }
        if !keep_traces {
            traces.clear();
        }
    }
    // A stable order whatever the file's.
    let (gen_cfg, roll) = match name {
        "speed" | "speed2" => (GenCfg { points: 1, use_pos: false, use_world: true, follow: false, use_ref: false, adf_p: 0, adf_b: 0 }, Roll { horizon: 300, ..Roll::default() }),
        "lakitu" | "lakitu_chain" => (GenCfg { points: 6, use_pos: true, use_world: true, follow: false, use_ref: false, adf_p: 0, adf_b: 0 }, Roll { horizon: 360, ..Roll::default() }),
        // Fitness over a whole segment: a slow discount, a bonus on the scale of its ETA.
        "follow" => (GenCfg { points: 3, use_pos: false, use_world: false, follow: true, use_ref: false, adf_p: 0, adf_b: 0 }, Roll { horizon: 600, stall: 120, gamma: 0.997, bonus: 100.0, death: 100.0 }),
        _ => bail!("unknown problem {name}"),
    };
    Ok(Problem { name: name.to_string(), cases, tasks, gen_cfg, roll, movies })
}

fn play_trace(game: &mut SM64Game, power_on: &sm64_binds::SaveState, pads: &[GamePad]) -> Result<Vec<St>> {
    game.load_state(power_on)?;
    let mut v = Vec::with_capacity(pads.len());
    for (i, p) in pads.iter().enumerate() {
        game.step_game(*p)?;
        v.push(read_state(game, i as u32 + 1, false));
    }
    Ok(v)
}

fn step_to(game: &mut SM64Game, pads: &[GamePad], at: &mut usize, frame: usize, hist: &mut Vec<St>) -> Result<()> {
    while *at < frame {
        game.step_game(pads[*at])?;
        *at += 1;
        if *at + PREFIX >= frame {
            hist.push(read_state(game, *at as u32, false));
            if hist.len() > PREFIX {
                hist.remove(0);
            }
        }
    }
    Ok(())
}

fn make_task(name: &str, st: &St, args: &[String], trace: Option<(&[St], &[GamePad])>, frame: usize) -> Result<Box<dyn Task>> {
    let mut pts = [[0.0f32; 3]; MAX_POINTS];
    Ok(match name {
        "speed" | "speed2" => {
            pts[0] = RAMP;
            Box::new(ObjTask { obj: objective(st, "Tech::SpeedChain bob-ramp")?, pts })
        }
        "lakitu" | "lakitu_chain" => {
            pts[0] = DOOR;
            // Landmarks: the run-up's crouch slide, the far end of its line, the
            // slide's exit (hill::C, RUN_YAW, EXIT).
            pts[1] = RUN_C;
            pts[2] = along(RUN_C, 23818, 3000.0);
            pts[3] = EXIT;
            // hill::WEST, and behind the crouch slide on its line (the approach).
            pts[4] = [-1400.0, 260.0, 4400.0];
            pts[5] = along(RUN_C, 23818, -500.0);
            Box::new(ObjTask { obj: objective(st, "Tech::LakituSkip hill")?, pts })
        }
        "follow" => {
            let get = |k: &str| args.iter().find_map(|a| a.strip_prefix(&format!("{k}=")).map(String::from));
            let h: usize = get("h").ok_or_else(|| anyhow::anyhow!("h="))?.parse()?;
            let sp: usize = get("s").ok_or_else(|| anyhow::anyhow!("s="))?.parse()?;
            let r: f32 = get("r").map_or(Ok(100.0), |v| v.parse())?;
            let (tr, pads) = trace.unwrap();
            Box::new(Follow::new(tr, pads, frame, sp, h, r)?)
        }
        n => bail!("task for {n}"),
    })
}

/// How many waypoints ahead count as reached (pyenv rl.rs `lookahead`, the
/// RL follower's eval passes 4).
const LOOKAHEAD: usize = 4;
/// follow_lib.py's MAX_WP.
const MAX_WP: usize = 140;

#[derive(Clone, Copy)]
struct Wp {
    pos: [f32; 3],
    zone: (u16, u8),
    frame: usize,
    pace: f32,
}

/// Exp. 3: the RL follower's task, rule for rule (pyenv/src/rl.rs trace_frame,
/// trace_eta; backbone/rlv2 follow_lib.py waypoints, train_follow.py limit,
/// eval_follow.py summary), so the numbers compare: waypoints every `spacing`
/// frames of a reference trace to `horizon`; a hit is the frame's movement
/// segment passing within `radius` (3D) of one of the next 5; a warp skips to the
/// first waypoint ahead in the new zone, or is lost; limit 2 × reference + 40.
/// `mem.flags`: the current waypoint; mem.f: the last position, mem.f[3]: hits.
pub struct Follow {
    way: Vec<Wp>,
    radius: f32,
    reference: usize,
    /// The reference from the start frame on (per frame, for `reference()`).
    refs: Vec<RefFrame>,
}

#[derive(Clone, Copy)]
struct RefFrame {
    pos: [f32; 3],
    now: RefNow,
}

const STARTED: u32 = 1 << 31;
/// `mem.flags`: the waypoint below this bit, the aligned reference frame above.
const I_BITS: u32 = 12;
const I_MASK: u32 = (1 << I_BITS) - 1;
/// How far ahead the alignment looks (train_follow.py's 48).
const ALIGN_AHEAD: usize = 48;

fn seg_dist2(a: [f32; 3], b: [f32; 3], p: [f32; 3]) -> f32 {
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let l = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let u = if l > 0.0 { (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1] + (p[2] - a[2]) * d[2]) / l).clamp(0.0, 1.0) } else { 0.0 };
    let q = [a[0] + u * d[0] - p[0], a[1] + u * d[1] - p[1], a[2] + u * d[2] - p[2]];
    q[0] * q[0] + q[1] * q[1] + q[2] * q[2]
}

impl Follow {
    /// `tr[i]` is the state after i + 1 pads; `frame` pads were played at the start.
    pub fn new(tr: &[St], pads: &[GamePad], frame: usize, spacing: usize, horizon: usize, radius: f32) -> Result<Follow> {
        // The trace's end: its last state before its last pad, as TracePool records.
        let end = tr.len() - 1;
        let e = end.min(frame + horizon);
        let mut idx: Vec<usize> = (frame + spacing..=e).step_by(spacing).collect();
        if idx.last() != Some(&e) {
            idx.push(e);
        }
        let control = |t: usize| {
            let s = &tr[t - 1];
            s.controllable() && !s.modal()
        };
        let mut keep: Vec<usize> = idx.iter().copied().filter(|&t| control(t)).collect();
        if keep.is_empty() {
            keep = vec![e];
        }
        if keep.len() > MAX_WP {
            keep.drain(..keep.len() - MAX_WP);
        }
        let mut way: Vec<Wp> = Vec::with_capacity(keep.len());
        for &t in &keep {
            let s = &tr[t - 1];
            let pace = match way.last() {
                Some(p) if p.zone == s.zone() && t > p.frame => (obrrt_core::dist(p.pos, s.pos) / (t - p.frame) as f32).max(obrrt_core::PACE as f32),
                _ => obrrt_core::PACE as f32,
            };
            way.push(Wp { pos: s.pos, zone: s.zone(), frame: t, pace });
        }
        let reference = way.last().unwrap().frame - frame;
        let refs = (frame..=e)
            .map(|t| {
                let s = &tr[t - 1];
                let p = pads[t];
                let adj = |v: i8| if v.unsigned_abs() < 8 { 0.0 } else { v as f32 - 6.0 * (v as f32).signum() };
                let (ax, ay) = (adj(p.stick_x), adj(p.stick_y));
                let m = ax.hypot(ay);
                let yaw = (ax.atan2(-ay) * 65536.0 / std::f32::consts::TAU) as i32 + s.cam_yaw as i32;
                RefFrame { pos: s.pos, now: RefNow { btn: p.button, yaw: yaw as i16, tilt: (m / 121.0).min(1.0), act: s.action, fv: s.forward_vel, dist: 0.0 } }
            })
            .collect();
        Ok(Follow { way, radius, reference, refs })
    }

    /// The waypoints, for the independent check.
    pub fn points_text(&self) -> String {
        self.way.iter().map(|w| format!("{} {} {} {} {}\n", w.pos[0], w.pos[1], w.pos[2], w.zone.0, w.zone.1)).collect()
    }

    fn cur(&self, mem: &Mem) -> usize {
        ((mem.flags & I_MASK) as usize).min(self.way.len() - 1)
    }

    fn align(&self, mem: &Mem) -> usize {
        ((mem.flags & !STARTED) >> I_BITS) as usize
    }

    /// The nearest reference frame in the window ahead of the last alignment.
    fn realign(&self, s: &St, mem: &mut Mem, i: usize) {
        let a = self.align(mem);
        let e = (a + ALIGN_AHEAD + 1).min(self.refs.len());
        let mut best = (f32::INFINITY, a);
        for k in a..e {
            let d = obrrt_core::dist(self.refs[k].pos, s.pos);
            if d < best.0 {
                best = (d, k);
            }
        }
        mem.flags = STARTED | ((best.1 as u32) << I_BITS) | i as u32;
    }

    fn eta_at(&self, s: &St, i: usize) -> f64 {
        let w = &self.way[i];
        (self.way.last().unwrap().frame - w.frame) as f64 + (obrrt_core::dist(s.pos, w.pos) / w.pace) as f64
    }
}

impl Task for Follow {
    fn eta(&self, s: &St, _start: &St, mem: &mut Mem) -> Eta {
        let i = self.cur(mem);
        if mem.flags & STARTED == 0 {
            self.realign(s, mem, i);
            mem.f[0] = s.pos[0] as f64;
            mem.f[1] = s.pos[1] as f64;
            mem.f[2] = s.pos[2] as f64;
            return Eta::Frames(self.eta_at(s, i));
        }
        let prev = [mem.f[0] as f32, mem.f[1] as f32, mem.f[2] as f32];
        mem.f[0] = s.pos[0] as f64;
        mem.f[1] = s.pos[1] as f64;
        mem.f[2] = s.pos[2] as f64;
        let z = s.zone();
        let live = s.controllable() && !s.modal();
        let mut advanced = None;
        if live && z != self.way[i].zone {
            match (i..self.way.len()).find(|&k| self.way[k].zone == z) {
                Some(k) => advanced = Some(k),
                None => return Eta::Lost,
            }
        } else {
            let end = (i + LOOKAHEAD + 1).min(self.way.len());
            let r2 = self.radius * self.radius;
            for k in (i..end).rev() {
                if self.way[k].zone == z && seg_dist2(prev, s.pos, self.way[k].pos) <= r2 {
                    advanced = Some(k + 1);
                    mem.f[3] += 1.0;
                    break;
                }
            }
        }
        let mut i = i;
        if let Some(k) = advanced {
            if k >= self.way.len() {
                return Eta::Done;
            }
            i = k;
        }
        self.realign(s, mem, i);
        Eta::Frames(self.eta_at(s, i))
    }
    fn points(&self, s: &St, mem: &Mem) -> [[f32; 3]; MAX_POINTS] {
        let i = self.cur(mem);
        let n = self.way.len() - 1;
        let mut p = [s.pos; MAX_POINTS];
        p[0] = self.way[i].pos;
        p[1] = self.way[(i + 1).min(n)].pos;
        p[2] = self.way[(i + 2).min(n)].pos;
        p
    }
    fn target_for_baseline(&self, s: &St) -> [f32; 3] {
        s.pos
    }
    fn horizon(&self) -> Option<usize> {
        Some(2 * self.reference + 40)
    }
    fn pace(&self, mem: &Mem) -> f32 {
        self.way[self.cur(mem)].pace
    }
    fn reference(&self, s: &St, mem: &Mem) -> RefNow {
        let f = &self.refs[self.align(mem).min(self.refs.len() - 1)];
        RefNow { dist: obrrt_core::dist(f.pos, s.pos), ..f.now }
    }
    fn report(&self, mem: &Mem, frames: usize, end: End) -> String {
        // eval_follow.py's summary fields: on finishing, i is the last waypoint.
        let done = end == End::Done;
        format!("ok={} i={} n={} hits={} frames={} ref={} died={}", done as u8, self.cur(mem), self.way.len(), mem.f[3] as usize + done as usize, frames, self.reference, matches!(end, End::Died | End::Lost) as u8)
    }
    fn way_text(&self) -> String {
        self.points_text()
    }
}
