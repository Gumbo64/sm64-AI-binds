//! The route mender: an RRT*-style rewire of one finished route. It keeps
//! making a route faster and never makes it slower.
//!
//!   mend <route.m64> --out DIR --budget GA_FRAMES [--goal-stars N] [--seed S]
//!        [--start-frame F] [--rom FILE] [--no-repair] [--no-cam]
//!   mend <route.m64> --out DIR --probe P   (survival of a one-frame shift)
//!
//! A route is inputs from power-on (raw or .m64: a `legs` route.m64, or
//! `nodepath` on a tree node). Its goal is the first frame with `goal-stars`
//! stars (default: the most the route reaches); it is cut there.
//!
//! Each step draws a window [a, b] of the route (where to look is a bandit
//! over route position, window length and gate tightness; see `Bandit`):
//!
//! 1. **Window GA.** The usual open GA from the route's state at a, seeded with
//!    the route's own inputs (kept in the elite, OBRRT_GA_KEEP_SEED), towards
//!    [`Window`]: be at the route's state at b (the gate) before the route is.
//!    The ETA follows the route's own path (the tube), not a straight line.
//!    The gate is a plane through the route's position at b: passing it within
//!    a radius, with the route's progress, action and enough speed. A plain
//!    radius would reward being a few frames behind on the same line, which
//!    "saves" frames that are all owed back downstream.
//! 2. **Carry.** Every met rollout (not only the winner; ga::set_met_sink), the
//!    soonest first: the rest of the route's inputs are replayed from its end,
//!    with the stick turned by the camera difference, since the stick is
//!    relative to a camera that may have moved differently.
//! 3. **Repair.** Where the replay leaves the route (objects and RNG see other
//!    timers), a smaller window GA from the last good state to a later gate,
//!    which must keep some of the lead; then the replay goes on.
//! 3b. **Trims** (`--trim-every F`, no GA): frames simply deleted at every
//!    route frame and carried to the goal. Cheap: a failed one costs only its
//!    survival (replay frames, counted apart).
//! 4. **Accept** only if the goal itself is reached sooner in that replay.
//!    Everything else is thrown away, so the route can only get faster.
//!
//! The step "get from this state through these waypoints" is one trait,
//! [`Follower`]; the seeded GA is its only implementation so far.
//! `--probe` and `--probe-repair` measure what a follower has to deliver.
//!
//! Outputs in DIR: route.m64 (raw), v/NNNN-FRAME.m64 (each accepted route),
//! curve.csv (GA frames, replay frames, goal frame at every accept and every
//! report), accepts.csv, stats.json. Off the `mend` binary, nothing runs: the
//! only code outside this file is the met-rollout sink in ga.rs, inert unset.

use std::sync::Mutex;

use anyhow::{Context, Result, anyhow, bail};
use obrrt_core::{Arrival, Eta, GaBudget, GamePad, Mem, Objective, Precision, St, read_state};
use obrrt_search::ga::{GaParams, Worker};
use obrrt_search::store::Delta;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde_json::json;
use sm64_binds::{SM64Game, SM64GameGenerator, SaveState};

/// As `legs`: the P arm's movement set, which a plain run gets unless set.
const DEFAULT_ENV: &[(&str, &str)] = &[
    ("OBRRT_MOVES", "1"),
    ("OBRRT_MOVE_GA", "1"),
    ("OBRRT_MOVE_GA_EXTRA", "1"),
    ("OBRRT_MOVES_MODES", "1"),
    ("OBRRT_MOVES_EXTRA", "1"),
    ("OBRRT_MOVES_SEEDED", "1"),
    ("OBRRT_CALM", "1"),
    ("OBRRT_LAKITU_ARC", "1"),
    ("OBRRT_PAINTING_FAST", "1"),
    ("OBRRT_GP_CATCH", "1"),
    ("OBRRT_SWIM_HOLD", "1"),
    ("OBRRT_TECH", "all"),
    // The route's inputs can't be bred out.
    ("OBRRT_GA_KEEP_SEED", "1"),
    // A window is up to hundreds of frames: 0.99 would hardly see its end.
    ("OBRRT_GA_GAMMA", "0.995"),
];

/// A route state kept for loading every this many frames (as a page delta).
const SAVE_EVERY: u32 = 48;
/// Window lengths (route frames) the bandit picks from.
const LENGTHS: [u32; 6] = [16, 32, 64, 128, 256, 512];
/// (radius, exact action, speed slack, max vertical speed difference): gate
/// tightness the bandit picks from.
const TIGHT: [(f32, bool, f32, f32); 3] = [(24.0, true, 0.97, 4.0), (64.0, true, 0.9, 12.0), (160.0, false, 0.75, 40.0)];
/// Rollouts kept per GA, and carried per GA at most.
const SINK: usize = 64;
const CARRY: usize = 8;
/// Repairs per candidate, gate distances tried per repair.
const MAX_REPAIRS: usize = 6;
const REPAIR_LENGTHS: [u32; 3] = [48, 120, 300];
/// The carry's limits: off the route by this much, for this many frames.
const OFF_DIST: f32 = 160.0;
const OFF_FRAMES: u32 = 6;
/// In another zone or dialog state than the route, this many frames.
const ZONE_FRAMES: u32 = 45;
/// Fewer stars than the route, this many frames.
const STAR_FRAMES: u32 = 60;
/// Route frames per bandit bucket.
const BUCKET: u32 = 24;
/// A GA is cut when even this fraction of the ETA misses the deadline.
const OPT: f64 = 0.5;

fn read_m64(path: &str) -> Result<Vec<GamePad>> {
    let b = std::fs::read(path).with_context(|| path.to_string())?;
    let body = if b.len() >= 0x400 && &b[..4] == b"M64\x1a" { &b[0x400..] } else { &b[..] };
    Ok(body.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect())
}

fn write_m64(path: &str, pads: &[GamePad]) -> Result<()> {
    let mut out = Vec::with_capacity(pads.len() * 4);
    for p in pads {
        out.extend_from_slice(&p.button.to_be_bytes());
        out.push(p.stick_x as u8);
        out.push(p.stick_y as u8);
    }
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, out)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    obrrt_core::dist(a, b)
}

fn hspeed(s: &St) -> f32 {
    (s.vel[0] * s.vel[0] + s.vel[2] * s.vel[2]).sqrt()
}

/// The stick `p` turned by the camera difference `d` (as legs' world_seed).
fn turned(p: GamePad, d: i32) -> GamePad {
    let d = d as i16;
    if d == 0 || (p.stick_x == 0 && p.stick_y == 0) {
        return p;
    }
    let (x, y) = (p.stick_x as f32, p.stick_y as f32);
    let a = x.atan2(-y) + d as f32 * std::f32::consts::TAU / 65536.0;
    let r = x.hypot(y);
    GamePad::new(p.button, (r * a.sin()).round().clamp(-127.0, 127.0) as i8, (-r * a.cos()).round().clamp(-127.0, 127.0) as i8)
}

// ---- the route ----

/// Save states as page deltas against one base per zone (8.6 MB -> ~20 KB).
#[derive(Default)]
struct Saves {
    bases: Vec<Vec<u8>>,
    base_of: std::collections::HashMap<(u16, u8), u16>,
    /// By frame, ascending.
    at: Vec<(u32, Delta)>,
}

impl Saves {
    fn push(&mut self, frame: u32, zone: (u16, u8), s: &SaveState) {
        let b = *self.base_of.entry(zone).or_insert_with(|| {
            self.bases.push(s.as_bytes().to_vec());
            self.bases.len() as u16 - 1
        });
        self.at.push((frame, Delta::new(b, &self.bases[b as usize], s)));
    }
    fn truncate(&mut self, frame: u32) {
        let n = self.at.partition_point(|x| x.0 <= frame);
        self.at.truncate(n);
    }
}

/// A route and everything derived from it.
struct Route {
    inputs: Vec<GamePad>,
    /// The state after `f` inputs (index f).
    trace: Vec<St>,
    goal_stars: i16,
    /// The first frame with `goal_stars` stars; inputs stop there.
    goal: u32,
    saves: Saves,
}

impl Route {
    /// Replays `inputs` from power-on, or from `base` (a route sharing them
    /// up to `keep`); the route is cut at the goal. None if no goal.
    fn build(game: &mut SM64Game, inputs: Vec<GamePad>, goal_stars: i16, base: Option<(&Route, u32)>) -> Result<Option<Route>> {
        let mut saves = Saves::default();
        let mut trace: Vec<St> = Vec::with_capacity(inputs.len() + 1);
        let mut from = 0u32;
        match base {
            Some((r, keep)) => {
                let i = r.saves.at.partition_point(|x| x.0 <= keep).checked_sub(1).ok_or_else(|| anyhow!("no save before {keep}"))?;
                let (f, d) = &r.saves.at[i];
                game.load_state(&d.apply(&r.saves.bases)?)?;
                from = *f;
                trace.extend_from_slice(&r.trace[..=from as usize]);
                saves.bases = r.saves.bases.clone();
                saves.base_of = r.saves.base_of.clone();
                saves.at = r.saves.at[..=i].to_vec();
            }
            None => trace.push(read_state(game, 0, false)),
        }
        let mut goal = None;
        for (i, p) in inputs.iter().enumerate().skip(from as usize) {
            game.step_game(*p)?;
            let f = i as u32 + 1;
            let s = read_state(game, f, false);
            if f % SAVE_EVERY == 0 && s.action != 0 {
                saves.push(f, s.zone(), &game.save_state());
            }
            let hit = s.stars >= goal_stars;
            trace.push(s);
            if hit {
                goal = Some(f);
                break;
            }
        }
        let Some(goal) = goal else { return Ok(None) };
        let mut inputs = inputs;
        inputs.truncate(goal as usize);
        saves.truncate(goal);
        Ok(Some(Route { inputs, trace, goal_stars, goal, saves }))
    }

    /// A route with no goal: all of `inputs`, which end the carry ("end").
    fn open(game: &mut SM64Game, inputs: Vec<GamePad>) -> Result<Route> {
        let mut saves = Saves::default();
        let mut trace = vec![read_state(game, 0, false)];
        for (i, p) in inputs.iter().enumerate() {
            game.step_game(*p)?;
            let f = i as u32 + 1;
            let s = read_state(game, f, false);
            if f % SAVE_EVERY == 0 && s.action != 0 {
                saves.push(f, s.zone(), &game.save_state());
            }
            trace.push(s);
        }
        let goal = inputs.len() as u32;
        Ok(Route { inputs, trace, goal_stars: i16::MAX, goal, saves })
    }

    /// The route's state at frame `f` loaded into `game`.
    fn load(&self, game: &mut SM64Game, f: u32) -> Result<()> {
        let i = self.saves.at.partition_point(|x| x.0 <= f);
        anyhow::ensure!(i > 0, "no save at or before frame {f}");
        let (g, d) = &self.saves.at[i - 1];
        game.load_state(&d.apply(&self.saves.bases)?)?;
        for p in &self.inputs[*g as usize..f as usize] {
            game.step_game(*p)?;
        }
        Ok(())
    }
}

fn gate_ok(s: &St) -> bool {
    s.controllable() && !s.modal() && s.alive()
}

// ---- the window objective ----

#[derive(Clone)]
struct TubePt {
    pos: [f32; 3],
    zone: (u16, u8),
    stars: i16,
    keys: u8,
    /// Route frames from here to the gate.
    left: f64,
    /// The route's own speed here (u/frame, at least 15).
    v: f64,
}

/// "Be at the route's state at frame b, before the route is."
struct Window {
    pts: Vec<TubePt>,
    gate: St,
    /// The route's direction at the gate; zero if it isn't moving.
    dir: [f32; 3],
    tight: (f32, bool, f32, f32),
    /// The goal itself: Done is the goal's stars.
    goal: Option<i16>,
    /// Done at or before this frame; past it, Lost.
    deadline: u32,
    /// The tube has dialog states (else a dialog is off the route).
    has_dialog: bool,
    budget: GaBudget,
}

impl Window {
    fn new(r: &Route, a: u32, b: u32, deadline: u32, tight: usize, pop: usize, gens: usize) -> Window {
        let t = &r.trace;
        let step = ((b - a) / 96).max(1) as usize;
        let speed = |k: usize| {
            let e = (k + 6).min(r.goal as usize);
            let (p, q) = (&t[k], &t[e]);
            if e > k && p.zone() == q.zone() { (dist(p.pos, q.pos) / (e - k) as f32) as f64 } else { 30.0 }
        };
        let mut pts: Vec<TubePt> = (a as usize..=b as usize)
            .step_by(step)
            .map(|k| TubePt { pos: t[k].pos, zone: t[k].zone(), stars: t[k].stars, keys: t[k].keys, left: (b as usize - k) as f64, v: speed(k).max(15.0) })
            .collect();
        if pts.last().is_none_or(|p| p.left > 0.0) {
            let k = b as usize;
            pts.push(TubePt { pos: t[k].pos, zone: t[k].zone(), stars: t[k].stars, keys: t[k].keys, left: 0.0, v: speed(k).max(15.0) });
        }
        let g = t[b as usize].clone();
        let e = &t[(b + 3).min(r.goal) as usize];
        let d = [e.pos[0] - g.pos[0], e.pos[1] - g.pos[1], e.pos[2] - g.pos[2]];
        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let dir = if n > 6.0 && e.zone() == g.zone() { [d[0] / n, d[1] / n, d[2] / n] } else { [0.0; 3] };
        let has_dialog = t[a as usize..=b as usize].iter().any(|s| s.in_dialog());
        let goal = (b == r.goal && r.goal_stars != i16::MAX).then_some(r.goal_stars);
        Window { pts, gate: g, dir, tight: TIGHT[tight], goal, deadline, has_dialog, budget: GaBudget { population: pop, generations: gens, horizon: (b - a) as usize + 30 } }
    }

    fn at_gate(&self, s: &St) -> bool {
        let g = &self.gate;
        if let Some(n) = self.goal {
            return s.stars >= n;
        }
        if s.zone() != g.zone() || s.stars != g.stars || s.keys != g.keys || s.holding != g.holding || !gate_ok(s) {
            return false;
        }
        let (r, exact, slack, dvy) = self.tight;
        let d = [s.pos[0] - g.pos[0], s.pos[1] - g.pos[1], s.pos[2] - g.pos[2]];
        if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > r * r {
            return false;
        }
        // Past the plane through the gate (not merely near it, behind it).
        if d[0] * self.dir[0] + d[1] * self.dir[1] + d[2] * self.dir[2] < 0.0 {
            return false;
        }
        if exact && s.action != g.action {
            return false;
        }
        if !exact && (s.action & 0x800) != (g.action & 0x800) {
            return false;
        }
        let v = hspeed(g);
        if hspeed(s) < v * slack - 1.0 {
            return false;
        }
        if g.action & 0x800 != 0 && (s.vel[1] - g.vel[1]).abs() > dvy {
            return false;
        }
        if v > 4.0 && (s.yaw.wrapping_sub(g.yaw) as i32).abs() > 0x2000 {
            return false;
        }
        true
    }

    /// The nearest tube point by ETA, and the ETA.
    fn nearest(&self, s: &St) -> Option<(usize, f64)> {
        let z = s.zone();
        let mut best: Option<(usize, f64)> = None;
        for (i, p) in self.pts.iter().enumerate() {
            if p.zone != z || p.stars != s.stars || p.keys != s.keys {
                continue;
            }
            let e = p.left + dist(s.pos, p.pos) as f64 / p.v;
            if best.is_none_or(|b| e < b.1) {
                best = Some((i, e));
            }
        }
        best
    }
}

impl Objective for Window {
    fn kind(&self) -> &'static str {
        "Mend::Window"
    }
    fn key(&self) -> String {
        format!("Mend::Window@{}", self.gate.frame)
    }
    fn obj_weight(&self, _: &St) -> f64 {
        1.0
    }
    fn base_criteria(&self, _: &St) -> bool {
        true
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.frame > self.deadline {
            return Eta::Lost;
        }
        if self.at_gate(s) {
            return Eta::Done;
        }
        if s.in_dialog() && !self.has_dialog {
            return Eta::Lost;
        }
        let Some((_, e)) = self.nearest(s) else { return Eta::Lost };
        let e = e.max(1.0);
        if s.frame as f64 + OPT * e > self.deadline as f64 {
            return Eta::Lost;
        }
        Eta::Frames(e)
    }
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(self.budget)
    }
    fn precision(&self) -> Precision {
        Precision::Precise
    }
    fn arrival(&self, _: &St) -> Arrival {
        Arrival::Pass
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let (i, e) = self.nearest(s)?;
        // Some frames ahead on the tube.
        let want = (self.pts[i].left - 15.0).max(0.0);
        let _ = e;
        self.pts[i..].iter().find(|p| p.left <= want).map(|p| p.pos).or(Some(self.gate.pos))
    }
}

// ---- carrying a saving down the route ----

enum Carry {
    /// The goal reached: the pads played from the start of the carry, the
    /// goal's frame.
    Goal { pads: Vec<GamePad>, frame: u32 },
    /// Off the route: at route frame `at` (last good route frame `good`, with
    /// `pads_good` pads played to it), `lost_j` route frames in.
    Off { pads_good: usize, good: u32, at: u32, why: &'static str, frames: u64 },
}

struct Opts {
    cam: bool,
}

/// From a state at frame `frame` (loaded in `game`), the route's inputs from
/// route frame `j` on. Output `pads` holds what was played.
fn carry(game: &mut SM64Game, r: &Route, st: &St, mut j: u32, opts: &Opts, pads: &mut Vec<GamePad>) -> Result<Carry> {
    let t = &r.trace;
    let start_pads = pads.len();
    let mut frame = st.frame;
    let mut cam = st.cam_yaw;
    let (mut off, mut zone_off, mut dlg_off, mut star_off) = (0u32, 0u32, 0u32, 0u32);
    let lives0 = st.lives;
    let (mut good, mut pads_good) = (j, pads.len());
    while j < r.goal {
        let p = r.inputs[j as usize];
        let q = if opts.cam { turned(p, t[j as usize].cam_yaw as i32 - cam as i32) } else { p };
        game.step_game(q)?;
        pads.push(q);
        frame += 1;
        j += 1;
        let s = read_state(game, frame, false);
        cam = s.cam_yaw;
        if s.stars >= r.goal_stars {
            return Ok(Carry::Goal { pads: std::mem::take(pads), frame });
        }
        let rf = &t[j as usize];
        let frames = (pads.len() - start_pads) as u64;
        if !s.alive() || s.lives < lives0 {
            return Ok(Carry::Off { pads_good, good, at: j, why: "died", frames });
        }
        zone_off = if s.zone() != rf.zone() { zone_off + 1 } else { 0 };
        if zone_off > ZONE_FRAMES {
            return Ok(Carry::Off { pads_good, good, at: j, why: "zone", frames });
        }
        dlg_off = if s.in_dialog() != rf.in_dialog() { dlg_off + 1 } else { 0 };
        if dlg_off > ZONE_FRAMES {
            return Ok(Carry::Off { pads_good, good, at: j, why: "dialog", frames });
        }
        star_off = if (s.stars, s.keys) < (rf.stars, rf.keys) { star_off + 1 } else { 0 };
        if star_off > STAR_FRAMES {
            return Ok(Carry::Off { pads_good, good, at: j, why: "progress", frames });
        }
        let d = if s.zone() == rf.zone() { dist(s.pos, rf.pos) } else { 0.0 };
        off = if s.zone() == rf.zone() && d > OFF_DIST { off + 1 } else { 0 };
        if off > OFF_FRAMES {
            return Ok(Carry::Off { pads_good, good, at: j, why: "off", frames });
        }
        if d <= 64.0 && gate_ok(&s) && gate_ok(rf) && s.zone() == rf.zone() {
            (good, pads_good) = (j, pads.len());
        }
    }
    let frames = (pads.len() - start_pads) as u64;
    Ok(Carry::Off { pads_good, good, at: j, why: "end", frames })
}

/// A route edit found by a thread: the inputs from `a`, reaching the goal at
/// `frame`.
struct Found {
    a: u32,
    b: u32,
    len: usize,
    tight: usize,
    /// The window's own lead (frames ahead at the gate) and repairs used.
    lead0: u32,
    repairs: u32,
    /// The window's own rollout (for rebasing onto a route changed meanwhile).
    window: Vec<GamePad>,
    /// The whole of the inputs from a to the goal.
    pads: Vec<GamePad>,
    frame: u32,
}

#[derive(Default, Clone, Copy, serde::Serialize)]
struct Stats {
    windows: u64,
    ga_frames: u64,
    replay_frames: u64,
    /// Windows with a rollout at the gate before the route.
    sooner: u64,
    sooner_frames: u64,
    /// Candidates carried; to the goal unrepaired, or carried off (per why).
    carried: u64,
    carried_goal: u64,
    off_died: u64,
    off_zone: u64,
    off_dialog: u64,
    off_progress: u64,
    off_off: u64,
    off_end: u64,
    /// Goal reached but not sooner.
    goal_not_sooner: u64,
    repair_gas: u64,
    repair_ga_frames: u64,
    repair_met: u64,
    repair_fail: u64,
    /// Windows that ended in a route that reached the goal sooner.
    wins: u64,
    wins_repaired: u64,
    errors: u64,
    /// Candidates' route frames survived after the gate, in total.
    survived: u64,
    /// Trims: tried, reached the goal sooner, frames replayed.
    trims: u64,
    trim_wins: u64,
    trim_frames: u64,
    /// A round's other wins, retried on the route the best one made.
    rebase_tried: u64,
    rebase_ok: u64,
    /// Trim passes skipped: the route hadn't changed, or trims over their share.
    trim_skipped: u64,
    /// Per REPAIR_LENGTHS: [tried, met].
    repair_by_len: [[u64; 2]; 3],
    /// Carried candidates by route frames survived after the gate:
    /// <30, <100, <300, <1000, more but off, reached the goal (sooner or not).
    survival: [u64; 6],
}

impl Stats {
    fn add(&mut self, o: &Stats) {
        self.windows += o.windows;
        self.ga_frames += o.ga_frames;
        self.replay_frames += o.replay_frames;
        self.sooner += o.sooner;
        self.sooner_frames += o.sooner_frames;
        self.carried += o.carried;
        self.carried_goal += o.carried_goal;
        self.off_died += o.off_died;
        self.off_zone += o.off_zone;
        self.off_dialog += o.off_dialog;
        self.off_progress += o.off_progress;
        self.off_off += o.off_off;
        self.off_end += o.off_end;
        self.goal_not_sooner += o.goal_not_sooner;
        self.repair_gas += o.repair_gas;
        self.repair_ga_frames += o.repair_ga_frames;
        self.repair_met += o.repair_met;
        self.repair_fail += o.repair_fail;
        self.wins += o.wins;
        self.wins_repaired += o.wins_repaired;
        self.errors += o.errors;
        self.survived += o.survived;
        self.trims += o.trims;
        self.trim_wins += o.trim_wins;
        self.trim_frames += o.trim_frames;
        self.rebase_tried += o.rebase_tried;
        self.rebase_ok += o.rebase_ok;
        self.trim_skipped += o.trim_skipped;
        for i in 0..3 {
            self.repair_by_len[i][0] += o.repair_by_len[i][0];
            self.repair_by_len[i][1] += o.repair_by_len[i][1];
        }
        for i in 0..6 {
            self.survival[i] += o.survival[i];
        }
    }
    fn off(&mut self, why: &str) {
        match why {
            "died" => self.off_died += 1,
            "zone" => self.off_zone += 1,
            "dialog" => self.off_dialog += 1,
            "progress" => self.off_progress += 1,
            "off" => self.off_off += 1,
            _ => self.off_end += 1,
        }
    }
}

/// One proposal of the bandit.
#[derive(Clone, Copy, Debug)]
struct Prop {
    a: u32,
    b: u32,
    len_i: usize,
    tight: usize,
    pop: usize,
    gens: usize,
}

struct Ctx<'a> {
    route: &'a Route,
    repair: bool,
    opts: Opts,
    effort_gens: usize,
    max_repairs: usize,
    /// The route's end counts as the goal (an open route: `--splice`'s).
    end_ok: bool,
    /// Gate tightness of repairs (`TIGHT`).
    repair_tight: usize,
}

// ---- the follower: the one step a controller has to provide ----

/// "From this state, follow the route through these waypoints, and be past
/// the last one by `deadline`." The mender's whole dependence on a controller
/// is this call; the seeded GA is today's answer ([`GaFollower`]). A learned
/// follower or an evolved GP controller implements the same trait (returning
/// one or a few rollouts, `frames` = what it simulated) and the window, carry,
/// repair and accept logic above and below it stay as they are.
struct FollowReq<'a> {
    route: &'a Route,
    start: &'a St,
    /// The save state of `start`.
    save: &'a SaveState,
    /// Route frame `start` stands for; `gate` the last waypoint. The
    /// waypoints are `route.trace[j..=gate]`, the route's own inputs
    /// `route.inputs[j..gate]`.
    j: u32,
    gate: u32,
    /// Latest game frame the gate may be passed at.
    deadline: u32,
    /// Gate tolerance (`TIGHT`).
    tight: usize,
    /// How hard to try (a GA's population and generations).
    effort: (usize, usize),
}

/// A rollout that passed the gate: its inputs, and the state at the gate.
type Followed = obrrt_search::ga::MetRollout;

struct FollowOut {
    /// Rollouts past the gate by the deadline, the soonest first. Each must
    /// replay exactly from `save` (they are replayed before being trusted).
    met: Vec<Followed>,
    /// Frames simulated.
    frames: u64,
}

trait Follower {
    fn follow(&mut self, game: &mut SM64Game, req: &FollowReq, rng: &mut SmallRng) -> Result<FollowOut>;
}

/// The seeded GA: the usual open GA towards [`Window`], with the route's
/// inputs, turned closed-loop for the camera, as its kept seed.
struct GaFollower<'a> {
    worker: &'a mut Worker,
    params: GaParams,
    cam: bool,
}

impl Follower for GaFollower<'_> {
    fn follow(&mut self, game: &mut SM64Game, q: &FollowReq, rng: &mut SmallRng) -> Result<FollowOut> {
        let r = q.route;
        let win = Window::new(r, q.j, q.gate, q.deadline, q.tight, q.effort.0, q.effort.1);
        game.load_state(q.save)?;
        let mut seed = Vec::with_capacity((q.gate - q.j) as usize);
        let mut cam = q.start.cam_yaw;
        let mut fr = q.start.frame;
        for k in q.j..q.gate {
            let p = r.inputs[k as usize];
            let pad = if self.cam { turned(p, r.trace[k as usize].cam_yaw as i32 - cam as i32) } else { p };
            game.step_game(pad)?;
            fr += 1;
            cam = read_state(game, fr, false).cam_yaw;
            seed.push(pad);
        }
        obrrt_search::ga::set_met_sink(SINK);
        let att = self.worker.attempt_full(&[&win as &dyn Objective], q.start, Some(q.save), rng, self.params, Some(&seed), None);
        let met = obrrt_search::ga::take_met_sink();
        let att = att?;
        Ok(FollowOut { met, frames: att.frames + (q.gate - q.j) as u64 })
    }
}

/// `f` from `start` through route frames j..=gate; rollouts strictly ahead of
/// the route at the gate, the soonest first.
#[allow(clippy::too_many_arguments)]
fn follow_counted(f: &mut dyn Follower, game: &mut SM64Game, x: &Ctx, start: &St, save: &SaveState, j: u32, gate: u32, deadline: u32, tight: usize, effort: (usize, usize), stats: &mut Stats, rng: &mut SmallRng, repairing: bool) -> Result<Vec<Followed>> {
    let out = f.follow(game, &FollowReq { route: x.route, start, save, j, gate, deadline, tight, effort }, rng)?;
    stats.ga_frames += out.frames;
    if repairing {
        stats.repair_ga_frames += out.frames;
    }
    let mut met = out.met;
    met.retain(|m| !m.genome.is_empty() && m.end.frame < gate);
    Ok(met)
}

/// Picks from the sink: up to CARRY, the soonest first, distinct by end cell.
fn pick_carry(met: Vec<obrrt_search::ga::MetRollout>) -> Vec<obrrt_search::ga::MetRollout> {
    let mut out: Vec<obrrt_search::ga::MetRollout> = Vec::with_capacity(CARRY);
    let cell = |s: &St| ((s.pos[0] / 16.0) as i32, (s.pos[1] / 16.0) as i32, (s.pos[2] / 16.0) as i32, s.frame);
    for m in met {
        if out.len() >= CARRY {
            break;
        }
        if !out.iter().any(|o| cell(&o.end) == cell(&m.end)) {
            out.push(m);
        }
    }
    out
}

/// One window: GA, carry, repair. A route reaching the goal sooner, if any.
fn try_window(f: &mut dyn Follower, game: &mut SM64Game, x: &Ctx, p: Prop, rng: &mut SmallRng, stats: &mut Stats) -> Result<Option<Found>> {
    let r = x.route;
    stats.windows += 1;
    r.load(game, p.a)?;
    let save = game.save_state();
    let start = read_state(game, p.a, false);
    let met = follow_counted(f, game, x, &start, &save, p.a, p.b, p.b, p.tight, (p.pop, p.gens), stats, rng, false)?;
    if met.is_empty() {
        return Ok(None);
    }
    stats.sooner += 1;
    stats.sooner_frames += (p.b - met[0].end.frame) as u64;
    let cands = pick_carry(met);
    let mut best: Option<Found> = None;
    // (route frames survived, candidate): the ones to repair.
    let mut offs: Vec<(u32, usize, u32, usize)> = Vec::new();
    for (ci, m) in cands.iter().enumerate() {
        // The window's end reached (the goal itself: nothing to carry).
        if m.end.stars >= r.goal_stars {
            if m.end.frame < r.goal && best.as_ref().is_none_or(|b| m.end.frame < b.frame) {
                best = Some(Found { a: p.a, b: p.b, len: p.len_i, tight: p.tight, lead0: p.b - m.end.frame, repairs: 0, window: m.genome.clone(), pads: m.genome.clone(), frame: m.end.frame });
            }
            continue;
        }
        stats.carried += 1;
        game.load_state(&save)?;
        for g in &m.genome {
            game.step_game(*g)?;
        }
        let end = read_state(game, m.end.frame, false);
        let mut pads = m.genome.clone();
        stats.replay_frames += m.genome.len() as u64;
        match carry(game, r, &end, p.b, &x.opts, &mut pads)? {
            Carry::Goal { pads, frame } => {
                stats.replay_frames += (pads.len() - m.genome.len()) as u64;
                stats.survived += (r.goal - p.b) as u64;
                stats.survival[5] += 1;
                if frame < r.goal {
                    stats.carried_goal += 1;
                    if best.as_ref().is_none_or(|b| frame < b.frame) {
                        best = Some(Found { a: p.a, b: p.b, len: p.len_i, tight: p.tight, lead0: p.b - m.end.frame, repairs: 0, window: m.genome.clone(), pads, frame });
                    }
                } else {
                    stats.goal_not_sooner += 1;
                }
            }
            Carry::Off { pads_good, good, at, why, frames } => {
                stats.replay_frames += frames;
                stats.survived += (at - p.b) as u64;
                let sv = at - p.b;
                stats.survival[if sv < 30 { 0 } else if sv < 100 { 1 } else if sv < 300 { 2 } else if sv < 1000 { 3 } else { 4 }] += 1;
                stats.off(why);
                let _ = pads_good;
                offs.push((at - p.b, ci, good, pads_good));
            }
        }
    }
    if best.is_some() || !x.repair {
        return Ok(best);
    }
    // Repair the candidate that got furthest (ties: soonest).
    offs.sort_by_key(|o| (std::cmp::Reverse(o.0), o.1));
    if let Some(&(_, ci, _, _)) = offs.first() {
        let m = &cands[ci];
        if let Some(f) = repair_chain(f, game, x, p, &save, m, rng, stats)? {
            stats.wins_repaired += 1;
            return Ok(Some(f));
        }
    }
    Ok(None)
}

/// Replays candidate `m` from the window's start; where it leaves the route,
/// a repair GA from the last good state; on, up to MAX_REPAIRS times.
#[allow(clippy::too_many_arguments)]
fn repair_chain(f: &mut dyn Follower, game: &mut SM64Game, x: &Ctx, p: Prop, save: &SaveState, m: &obrrt_search::ga::MetRollout, rng: &mut SmallRng, stats: &mut Stats) -> Result<Option<Found>> {
    let r = x.route;
    // `head`: the pads so far from a; (`at`, `st`): the route frame and the
    // state they lead to.
    let mut head: Vec<GamePad> = m.genome.clone();
    let mut j = p.b;
    let mut repairs = 0u32;
    'chain: loop {
        // Play the head, then the route on from j.
        game.load_state(save)?;
        for g in &head {
            game.step_game(*g)?;
        }
        let st = read_state(game, p.a + head.len() as u32, false);
        let mut pads = head.clone();
        let off = carry(game, r, &st, j, &x.opts, &mut pads)?;
        let (pads_good, good, at, why) = match off {
            Carry::Goal { pads, frame } => {
                if frame < r.goal {
                    return Ok(Some(Found { a: p.a, b: p.b, len: p.len_i, tight: p.tight, lead0: p.b.saturating_sub(m.end.frame), repairs, window: m.genome.clone(), pads, frame }));
                }
                stats.goal_not_sooner += 1;
                return Ok(None);
            }
            Carry::Off { pads_good, good, at, why, frames } => {
                stats.replay_frames += frames;
                (pads_good, good, at, why)
            }
        };
        if std::env::var_os("MEND_DEBUG").is_some() {
            println!("  chain: off at route frame {at} ({why}), last good {good} (frame {}), repairs {repairs}, route goal {}, GA {}", p.a + pads_good as u32, r.goal, stats.ga_frames);
        }
        if why == "end" && x.end_ok {
            let frame = p.a + pads.len() as u32;
            return Ok(Some(Found { a: p.a, b: p.b, len: p.len_i, tight: p.tight, lead0: p.b.saturating_sub(m.end.frame), repairs, window: m.genome.clone(), pads, frame }));
        }
        if repairs as usize >= x.max_repairs || why == "end" {
            return Ok(None);
        }
        repairs += 1;
        // The last good state.
        game.load_state(save)?;
        for g in &pads[..pads_good] {
            game.step_game(*g)?;
        }
        let frame = p.a + pads_good as u32;
        let from = read_state(game, frame, false);
        if frame >= good {
            return Ok(None);
        }
        let from_save = game.save_state();
        let lead = good - frame;
        for (li, &l) in REPAIR_LENGTHS.iter().enumerate() {
            let mut gate = (at.max(good + l / 2) + l / 2).min(r.goal);
            while gate < r.goal && !gate_ok(&r.trace[gate as usize]) {
                gate += 1;
            }
            let deadline = gate - 1;
            if frame >= deadline {
                continue;
            }
            // Loose gates: the lead pays for it; tight ones first.
            stats.repair_gas += 1;
            stats.repair_by_len[li][0] += 1;
            let met = follow_counted(f, game, x, &from, &from_save, good, gate, deadline, x.repair_tight, (p.pop, x.effort_gens.max(8)), stats, rng, true)?;
            if let Some(n) = pick_carry(met).into_iter().next() {
                stats.repair_met += 1;
                stats.repair_by_len[li][1] += 1;
                head = pads[..pads_good].to_vec();
                head.extend_from_slice(&n.genome);
                if n.end.stars >= r.goal_stars {
                    let frame = p.a + head.len() as u32;
                    if frame < r.goal {
                        return Ok(Some(Found { a: p.a, b: p.b, len: p.len_i, tight: p.tight, lead0: p.b.saturating_sub(m.end.frame), repairs, window: m.genome.clone(), pads: head, frame }));
                    }
                    return Ok(None);
                }
                j = gate;
                let _ = lead;
                continue 'chain;
            }
        }
        stats.repair_fail += 1;
        return Ok(None);
    }
}

// ---- where to look ----

/// Budget goes where savings are found: per route bucket and arm, frames
/// spent and frames saved. A bucket's weight is its saved share per spend,
/// plus a coverage term falling with its tries.
struct Bandit {
    tries: Vec<f64>,
    gain: Vec<f64>,
    len_spent: [f64; LENGTHS.len()],
    len_gain: [f64; LENGTHS.len()],
    tight_spent: [f64; TIGHT.len()],
    tight_gain: [f64; TIGHT.len()],
    /// `--slack K`: bucket weight x (1 + K x slack), see `slack_of`.
    slack: f64,
    /// Each bucket's slack, for the route with this goal.
    slacks: (u32, Vec<f64>),
    /// Rounds since the last accept.
    dry: u64,
    effort: f64,
}

fn yield_pick(rng: &mut SmallRng, spent: &[f64], gain: &[f64]) -> usize {
    // Every arm's estimate (gain + 1 frame) / (spent + 2 M frames), a floor
    // of a fifth of the mean, sampled.
    let w: Vec<f64> = spent.iter().zip(gain).map(|(s, g)| (g + 1.0) / (s + 2e6)).collect();
    let mean = w.iter().sum::<f64>() / w.len() as f64;
    let w: Vec<f64> = w.iter().map(|x| x.max(mean * 0.2)).collect();
    let mut x = rng.random::<f64>() * w.iter().sum::<f64>();
    for (i, v) in w.iter().enumerate() {
        if x < *v {
            return i;
        }
        x -= v;
    }
    w.len() - 1
}

impl Bandit {
    fn new(buckets: usize) -> Bandit {
        Bandit { tries: vec![0.0; buckets], gain: vec![0.0; buckets], len_spent: [0.0; LENGTHS.len()], len_gain: [0.0; LENGTHS.len()], tight_spent: [0.0; TIGHT.len()], tight_gain: [0.0; TIGHT.len()], slack: 0.0, slacks: (0, Vec::new()), dry: 0, effort: 1.0 }
    }

    fn propose(&mut self, r: &Route, first: u32, rng: &mut SmallRng) -> Option<Prop> {
        for _ in 0..200 {
            let len_i = yield_pick(rng, &self.len_spent, &self.len_gain);
            let tight = yield_pick(rng, &self.tight_spent, &self.tight_gain);
            if self.slack > 0.0 && self.slacks.0 != r.goal {
                self.slacks = (r.goal, slack_of(r));
            }
            let (sk, sl) = (self.slack, &self.slacks.1);
            let w: Vec<f64> = (first / BUCKET..=r.goal / BUCKET)
                .map(|b| {
                    let (t, g) = (self.tries.get(b as usize).copied().unwrap_or(0.0), self.gain.get(b as usize).copied().unwrap_or(0.0));
                    (1.0 + 2.0 * g) / (1.0 + t) * (1.0 + sk * sl.get(b as usize).copied().unwrap_or(0.0))
                })
                .collect();
            let mut x = rng.random::<f64>() * w.iter().sum::<f64>();
            let mut i = 0;
            for (k, v) in w.iter().enumerate() {
                i = k;
                if x < *v {
                    break;
                }
                x -= v;
            }
            let mut a = (first / BUCKET + i as u32) * BUCKET + rng.random_range(0..BUCKET);
            a = a.max(first);
            if a >= r.goal {
                continue;
            }
            // A start Mario controls, and not in a dialog.
            while a < r.goal && !gate_ok(&r.trace[a as usize]) {
                a += 1;
            }
            let mut b = a + LENGTHS[len_i];
            if b >= r.goal {
                b = r.goal;
            }
            while b < r.goal && !gate_ok(&r.trace[b as usize]) {
                b += 1;
            }
            if a + 4 > b {
                continue;
            }
            // More generations as a route gets harder, and for long windows.
            let gens = ((12.0 * self.effort).round() as usize).clamp(6, 240);
            return Some(Prop { a, b, len_i, tight, pop: 32, gens });
        }
        None
    }

    fn charge(&mut self, p: &Prop, frames: u64, saved: u32) {
        let (b0, b1) = ((p.a / BUCKET) as usize, (p.b / BUCKET) as usize);
        if self.tries.len() <= b1 {
            self.tries.resize(b1 + 1, 0.0);
            self.gain.resize(b1 + 1, 0.0);
        }
        // A try of a long window is about as many tries as it has buckets
        // only where it covers; a share each.
        let n = (b1 - b0 + 1) as f64;
        for b in b0..=b1 {
            self.tries[b] += 1.0 / n.sqrt();
            self.gain[b] += saved as f64 / n;
        }
        self.len_spent[p.len_i] += frames as f64;
        self.len_gain[p.len_i] += saved as f64;
        self.tight_spent[p.tight] += frames as f64;
        self.tight_gain[p.tight] += saved as f64;
    }

    /// A route changed: the frames after it moved, and its neighbourhood
    /// should be looked at again.
    fn changed(&mut self, a: u32, b: u32, saved: u32) {
        let (b0, b1) = ((a / BUCKET) as usize, (b / BUCKET) as usize);
        for t in self.tries.iter_mut().skip(b0.saturating_sub(4)).take(b1 - b0 + 9) {
            *t *= 0.25;
        }
        let drop = (saved / BUCKET) as usize;
        let at = b0.min(self.tries.len());
        let drop = drop.min(self.tries.len() - at);
        self.tries.drain(at..at + drop);
        self.gain.drain(at..at + drop);
    }

    fn line(&self) -> String {
        format!(
            "len yield/M {:?}, tight yield/M {:?}, effort {:.1}",
            LENGTHS.iter().zip(self.len_spent.iter().zip(&self.len_gain)).map(|(l, (s, g))| format!("{l}:{:.2}", g / (s / 1e6 + 1e-9))).collect::<Vec<_>>(),
            self.tight_spent.iter().zip(&self.tight_gain).map(|(s, g)| format!("{:.2}", g / (s / 1e6 + 1e-9))).collect::<Vec<_>>(),
            self.effort
        )
    }
}

/// Per bucket, how much the route seems to dawdle there: 1 at no net movement
/// over +-128 frames, 0 at SLACK_FULL u/frame or more; a zone change in that
/// span counts 0.5 (big accepts cluster at zone entries and at standstills,
/// RUNS.md). Frames Mario doesn't control count nothing (dialogs, dances).
const SLACK_FULL: f32 = 8.0;
fn slack_of(r: &Route) -> Vec<f64> {
    let n = (r.goal / BUCKET) as usize + 1;
    let (mut sum, mut cnt) = (vec![0.0f64; n], vec![0u32; n]);
    let g = r.goal as usize;
    for k in (0..=g).step_by(4) {
        let (a, b) = (&r.trace[k.saturating_sub(128)], &r.trace[(k + 128).min(g)]);
        let b_i = k / BUCKET as usize;
        cnt[b_i] += 1;
        if !gate_ok(&r.trace[k]) {
            continue;
        }
        sum[b_i] += if a.zone() != b.zone() { 0.5 } else { (1.0 - (dist(a.pos, b.pos) / 256.0 / SLACK_FULL).min(1.0)) as f64 };
    }
    sum.iter().zip(&cnt).map(|(s, c)| s / (*c).max(1) as f64).collect()
}

// ---- trims: frames simply taken out ----

/// Frames deleted per trim, tried in this order at each route frame.
const TRIMS: [u32; 5] = [1, 2, 3, 5, 8];
/// Route frames tried per batch of a trim pass.
const TRIM_CHUNK: u32 = 16;

/// A trim of `n` frames at `p` carried to the goal, if it gets there sooner.
fn try_trim(game: &mut SM64Game, r: &Route, p: u32, n: u32, opts: &Opts, frames: &mut u64) -> Result<Option<Vec<GamePad>>> {
    if p + n >= r.goal {
        return Ok(None);
    }
    r.load(game, p)?;
    let st = read_state(game, p, false);
    let mut pads = Vec::new();
    match carry(game, r, &st, p + n, opts, &mut pads)? {
        Carry::Goal { pads, frame } => {
            *frames += pads.len() as u64;
            Ok((frame < r.goal).then_some(pads))
        }
        Carry::Off { frames: f, .. } => {
            *frames += f;
            Ok(None)
        }
    }
}

// ---- the probe: how far a route survives a one-frame shift ----

fn probe(game: &mut SM64Game, r: &Route, first: u32, every: u32, out: &str) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{out}/probe.csv"))?);
    use std::io::Write;
    writeln!(f, "frame,zone,action,hspeed,kind,cam,survived,why,goal,lead_at_goal")?;
    let mut p = first;
    let mut totals: std::collections::BTreeMap<(String, bool), (u64, u64, u64)> = Default::default();
    while p + 2 < r.goal {
        if !gate_ok(&r.trace[p as usize]) {
            p += 1;
            continue;
        }
        for (kind, skip) in [("delete", 1u32), ("delay", 0)] {
            for cam in [false, true] {
                r.load(game, p)?;
                let mut fr = p;
                let mut pads = Vec::new();
                if kind == "delay" {
                    game.step_game(r.inputs[p as usize])?;
                    pads.push(r.inputs[p as usize]);
                    fr += 1;
                }
                let st = read_state(game, fr, false);
                let c = carry(game, r, &st, p + skip, &Opts { cam }, &mut pads)?;
                let (survived, why, goal) = match c {
                    Carry::Goal { frame, .. } => (r.goal - p, "goal".to_string(), Some(frame)),
                    Carry::Off { at, why, .. } => (at - p, why.to_string(), None),
                };
                let s = &r.trace[p as usize];
                writeln!(f, "{p},{}:{},{:#x},{:.1},{kind},{cam},{survived},{why},{},{}", s.level, s.area, s.action, hspeed(s), goal.map_or(String::new(), |g| g.to_string()), goal.map_or(String::new(), |g| (r.goal as i64 - g as i64).to_string()))?;
                let e = totals.entry((kind.to_string(), cam)).or_default();
                e.0 += 1;
                e.1 += survived as u64;
                e.2 += goal.is_some() as u64;
            }
        }
        p += every;
    }
    for ((k, c), (n, s, g)) in totals {
        println!("probe {k} cam {c}: n {n}, mean survival {:.0} route frames (route {} frames), goal reached {g}", s as f64 / n as f64, r.goal - first);
    }
    Ok(())
}

// ---- the probe: at what waypoint spacing does following work? ----

/// Waypoint spacings (route frames) measured by `--probe-repair`.
const SPACINGS: [u32; 6] = [12, 24, 48, 96, 192, 384];

/// What the mender needs from a follower, measured: from the route's state
/// with `n` frames taken out at p (a state a little off the route, ahead of
/// it), can the follower reach the waypoint `S` route frames on, keeping any
/// lead, at each tightness, for how many frames? One CSV row each.
fn probe_repair(slots: &[Mutex<(Worker, SM64Game)>], r: &Route, first: u32, every: u32, params: GaParams, cam: bool, seed: u64, out: &str) -> Result<()> {
    use std::io::Write;
    let mut ps = Vec::new();
    let mut p = first;
    while p + 8 < r.goal {
        if gate_ok(&r.trace[p as usize]) {
            ps.push(p);
            p += every;
        } else {
            p += 1;
        }
    }
    let mut tasks = Vec::new();
    for &p in &ps {
        for n in [1u32, 5] {
            for (si, &sp) in SPACINGS.iter().enumerate() {
                for tight in [1usize] {
                    tasks.push((p, n, si, sp, tight));
                }
            }
        }
    }
    let rows: Vec<String> = tasks
        .par_iter()
        .enumerate()
        .filter_map(|(i, &(p, n, _si, sp, tight))| {
            let mut guard = loop {
                if let Some(g) = slots.iter().find_map(|s| s.try_lock().ok()) {
                    break g;
                }
                std::thread::yield_now();
            };
            let (w, g) = &mut *guard;
            let mut gate = (p + n + sp).min(r.goal);
            while gate < r.goal && !gate_ok(&r.trace[gate as usize]) {
                gate += 1;
            }
            r.load(g, p).ok()?;
            let st = read_state(g, p, false);
            let save = g.save_state();
            let mut rng = SmallRng::seed_from_u64(seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut f = GaFollower { worker: w, params, cam };
            // The start stands for route frame p + n (it is n frames ahead).
            let out = f.follow(g, &FollowReq { route: r, start: &st, save: &save, j: p + n, gate, deadline: gate - 1, tight, effort: (32, 12) }, &mut rng).ok()?;
            let best = out.met.iter().filter(|m| !m.genome.is_empty() && m.end.frame < gate).map(|m| gate - m.end.frame).max();
            // The rollout's inputs as they stand, played on past the gate.
            let s = &r.trace[p as usize];
            Some(format!("{p},{}:{},{:#x},{:.1},{n},{sp},{tight},{},{},{}", s.level, s.area, s.action, hspeed(s), out.frames, best.is_some() as u8, best.unwrap_or(0)))
        })
        .collect();
    let mut f = std::fs::File::create(format!("{out}/probe_repair.csv"))?;
    writeln!(f, "frame,zone,action,hspeed,trim,spacing,tight,frames,met,lead")?;
    for l in rows {
        writeln!(f, "{l}")?;
    }
    Ok(())
}

// ---- splicing another route's section in ----

/// `--splice OTHER.m64`: another route's section (from the first controllable
/// state of the zone the routes' section starts in, to OTHER's end) in place
/// of this route's: the section played from this route's arrival there and
/// repaired against OTHER's trace, then this route's rest carried from the
/// nearest point of its own and repaired. Repeats with new seeds until a goal
/// sooner or the budget is out. The zone is the one OTHER ends in.
fn splice_route(f: &mut dyn Follower, game: &mut SM64Game, r: &Route, other: &str, seed: u64, stats: &mut Stats, cam: bool, budget: u64) -> Result<Option<Found>> {
    let inputs = read_m64(other)?;
    let gen_game = game;
    // OTHER's trace from power-on needs a fresh game.
    let rom = format!("{}/../../baserom.us.z64", env!("CARGO_MANIFEST_DIR"));
    let mut g0 = SM64GameGenerator::from_file(&rom)?.create_game()?;
    let p = Route::open(&mut g0, inputs)?;
    drop(g0);
    let zone = p.trace[p.goal as usize].zone();
    let enter = |t: &[St]| t.iter().position(|s| s.zone() == zone && gate_ok(s)).map(|x| x as u32);
    let (Some(ep), Some(er)) = (enter(&p.trace), enter(&r.trace)) else { bail!("no common zone") };
    println!("splice: zone {zone:?}, other enters at {ep} and ends at {}, route enters at {er}; goal {}", p.goal, r.goal);
    // How far OTHER's section survives from our arrival, after d waits: object
    // phases and the RNG depend on the frame; both are about free to test.
    let mut waits: Vec<(u32, u32, u32)> = Vec::new();
    for d in 0..=(ep.saturating_sub(er) + 20) {
        r.load(gen_game, er)?;
        let mut pads: Vec<GamePad> = (0..d).map(|_| GamePad::new(0, 0, 0)).collect();
        for q in &pads {
            gen_game.step_game(*q)?;
        }
        let st = read_state(gen_game, er + d, false);
        let c = carry(gen_game, &p, &st, ep, &Opts { cam }, &mut pads)?;
        let (at, good) = match c {
            Carry::Off { at, good, .. } => (at, good),
            Carry::Goal { .. } => (p.goal, p.goal),
        };
        waits.push((d, at, good));
    }
    println!("splice waits (d, off at, last good): {waits:?}");
    // The fewest waits that get about as far as any: each wait costs a frame.
    let cap = ep.saturating_sub(er);
    let far = waits.iter().filter(|w| w.0 < cap).map(|w| w.1).max().unwrap_or(0);
    let cands: Vec<u32> = waits.iter().filter(|w| w.0 < cap && w.1 + 10 >= far).map(|w| w.0).collect();
    let mut best_wait = cands.first().copied().unwrap_or(0);
    println!("splice: waits to try {cands:?}");
    let mut try_n = 0u64;
    let mut last_ga = u64::MAX;
    while stats.ga_frames < budget {
        // A try that simulated nothing would repeat for ever.
        if last_ga == stats.ga_frames {
            println!("splice: no GA ran (no lead left to spend), giving up");
            break;
        }
        last_ga = stats.ga_frames;
        try_n += 1;
        // Each try another wait (object phases), and rising effort.
        if !cands.is_empty() {
            best_wait = cands[(try_n as usize - 1) % cands.len()];
        }
        let rounds = (try_n as usize - 1) / cands.len().max(1);
        let gens = (40usize << rounds.min(3)).min(320);
        let pop = if rounds >= 2 { 128 } else { 64 };
        let mut rng = SmallRng::seed_from_u64(seed ^ try_n.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        r.load(gen_game, er)?;
        let save = gen_game.save_state();
        let st = read_state(gen_game, er, false);
        // The section, against OTHER.
        let px = Ctx { route: &p, repair: true, opts: Opts { cam }, effort_gens: gens, max_repairs: 30, end_ok: true, repair_tight: 2 };
        let prop = Prop { a: er, b: ep, len_i: 0, tight: 1, pop, gens };
        let mut wait_pads = Vec::new();
        let mut est = st.clone();
        if best_wait > 0 {
            gen_game.load_state(&save)?;
            wait_pads = vec![GamePad::new(0, 0, 0); best_wait as usize];
            for q in &wait_pads {
                gen_game.step_game(*q)?;
            }
            est = read_state(gen_game, er + best_wait, false);
        }
        let m0 = obrrt_search::ga::MetRollout { genome: wait_pads, end: est };
        let Some(sec) = repair_chain(f, gen_game, &px, prop, &save, &m0, &mut rng, stats)? else {
            println!("splice try {try_n}: the section failed, GA {}", stats.ga_frames);
            continue;
        };
        let at = er + sec.pads.len() as u32;
        gen_game.load_state(&save)?;
        for q in &sec.pads {
            gen_game.step_game(*q)?;
        }
        let end = read_state(gen_game, at, false);
        // The nearest point of this route, in the zone, not behind the arrival.
        let j = (er..r.goal.min(at + 300)).filter(|&k| r.trace[k as usize].zone() == end.zone()).min_by_key(|&k| dist(r.trace[k as usize].pos, end.pos) as u32).unwrap_or(at);
        println!("splice try {try_n}: section at frame {at} (other {}), {} repairs; route point {j}, {:.0} away, lead {} GA {}", p.goal, sec.repairs, dist(r.trace[j as usize].pos, end.pos), j as i64 - at as i64, stats.ga_frames);
        let rx = Ctx { route: r, repair: true, opts: Opts { cam }, effort_gens: gens, max_repairs: 30, end_ok: false, repair_tight: 2 };
        let m1 = obrrt_search::ga::MetRollout { genome: sec.pads.clone(), end };
        let prop = Prop { a: er, b: j, len_i: 0, tight: 1, pop, gens };
        if let Some(n) = repair_chain(f, gen_game, &rx, prop, &save, &m1, &mut rng, stats)? {
            println!("splice try {try_n}: GOAL at {} ({} repairs)", n.frame, n.repairs);
            return Ok(Some(n));
        }
        println!("splice try {try_n}: the rest failed, GA {} repairs met {} fail {}", stats.ga_frames, stats.repair_met, stats.repair_fail);
    }
    Ok(None)
}

// ---- main ----

/// The run's outputs.
struct Out {
    dir: String,
    curve: std::io::BufWriter<std::fs::File>,
    accepts: std::io::BufWriter<std::fs::File>,
    n: u32,
    goal0: u32,
    t0: std::time::Instant,
}

impl Out {
    fn point(&mut self, t: &Stats, goal: u32) -> Result<()> {
        use std::io::Write;
        writeln!(self.curve, "{},{},{},{:.0},{},{},{}", t.ga_frames, t.replay_frames, t.trim_frames, self.t0.elapsed().as_secs_f64(), goal, self.goal0 - goal, t.windows)?;
        self.curve.flush()?;
        Ok(())
    }
    fn stats(&self, t: &Stats, goal: u32) -> Result<()> {
        std::fs::write(format!("{}/stats.json", self.dir), serde_json::to_string_pretty(&json!({"goal0": self.goal0, "goal": goal, "accepts": self.n, "stats": t, "secs": self.t0.elapsed().as_secs_f64()}))?)?;
        Ok(())
    }
}

/// Moves the route onto `pads` from frame `a`, checks the replay agrees with
/// the carry, and writes it out. The frames saved.
#[allow(clippy::too_many_arguments)]
fn commit(game: &mut SM64Game, route: &mut Route, a: u32, pads: &[GamePad], want: u32, what: &str, lead0: u32, repairs: u32, t: &Stats, o: &mut Out) -> Result<u32> {
    use std::io::Write;
    let mut ni = route.inputs[..a as usize].to_vec();
    ni.extend_from_slice(pads);
    let Some(nr) = Route::build(game, ni, route.goal_stars, Some((&*route, a)))? else {
        bail!("the accepted route doesn't reach the goal (replay mismatch, wanted frame {want})");
    };
    if nr.goal != want {
        bail!("the accepted route's goal {} isn't the carried {want}", nr.goal);
    }
    let saved = route.goal - nr.goal;
    o.n += 1;
    println!("ACCEPT #{}: goal {} -> {} (-{saved}, total -{}), {what} lead {lead0} repairs {repairs}, GA {} replay {} trim {}", o.n, route.goal, nr.goal, o.goal0 - nr.goal, t.ga_frames, t.replay_frames, t.trim_frames);
    let z = route.trace[a as usize].zone();
    writeln!(o.accepts, "{},{},{},{},{},{saved},{what},{a},{lead0},{repairs},{}:{}", o.n, t.ga_frames, t.replay_frames, t.trim_frames, nr.goal, z.0, z.1)?;
    o.accepts.flush()?;
    *route = nr;
    write_m64(&format!("{}/route.m64", o.dir), &route.inputs)?;
    write_m64(&format!("{}/v/{:04}-{}.m64", o.dir, o.n, route.goal), &route.inputs)?;
    o.point(t, route.goal)?;
    Ok(saved)
}

/// One pass of trims over the whole route: at each frame, the trims of TRIMS
/// in parallel; any that reaches the goal sooner is taken at once.
#[allow(clippy::too_many_arguments)]
fn trim_pass(slots: &[Mutex<(Worker, SM64Game)>], game: &mut SM64Game, route: &mut Route, first: u32, opts: &Opts, t: &mut Stats, o: &mut Out, bandit: &mut Bandit, limit: u64) -> Result<u32> {
    let g0 = route.goal;
    let mut p = first;
    while p + 1 < route.goal {
        if t.trim_frames >= limit {
            break;
        }
        let tasks: Vec<(u32, u32)> = (p..(p + TRIM_CHUNK).min(route.goal)).flat_map(|q| TRIMS.iter().map(move |&n| (q, n))).collect();
        let rref = &*route;
        let res: Vec<(u32, u32, u64, Option<Vec<GamePad>>)> = tasks
            .par_iter()
            .map(|&(q, n)| {
                let mut guard = loop {
                    if let Some(g) = slots.iter().find_map(|s| s.try_lock().ok()) {
                        break g;
                    }
                    std::thread::yield_now();
                };
                let mut f = 0u64;
                let r = try_trim(&mut guard.1, rref, q, n, opts, &mut f).unwrap_or(None);
                (q, n, f, r)
            })
            .collect();
        let mut best: Option<(u32, u32, Vec<GamePad>)> = None;
        for (q, n, f, r) in res {
            t.trim_frames += f;
            t.trims += 1;
            if let Some(pads) = r {
                let fr = q + pads.len() as u32;
                if best.as_ref().is_none_or(|b| fr < b.0 + b.2.len() as u32) {
                    best = Some((q, n, pads));
                }
            }
        }
        match best {
            Some((q, n, pads)) => {
                t.trim_wins += 1;
                let want = q + pads.len() as u32;
                let saved = commit(game, route, q, &pads, want, &format!("trim {n}"), n, 0, t, o)?;
                bandit.changed(q, q + n, saved);
                p = q; // the same place again
            }
            None => p += TRIM_CHUNK,
        }
    }
    Ok(g0 - route.goal)
}

fn main() -> Result<()> {
    for (k, v) in DEFAULT_ENV {
        if std::env::var_os(k).is_none() {
            // SAFETY: single-threaded here, before any thread is started.
            unsafe { std::env::set_var(k, v) };
        }
    }
    obrrt_objectives::setup::from_env().map_err(anyhow::Error::msg)?;
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        bail!("usage: mend <route.m64> --out DIR --budget FRAMES [--goal-stars N] [--seed S] [--start-frame F] [--rom FILE] [--no-repair] [--no-cam] [--probe EVERY]");
    }
    let crate_dir = env!("CARGO_MANIFEST_DIR");
    let mut rom = format!("{crate_dir}/../../baserom.us.z64");
    let (mut out, mut budget, mut goal_stars, mut seed, mut start_frame, mut repair, mut cam, mut probe_every) = (String::new(), 10_000_000u64, None, 1u64, None, true, true, None);
    let mut report_every = 2_000_000u64;
    // --trim-every F: a trim pass at the start and each F GA frames (0: none).
    let mut trim_every = 0u64;
    let mut no_ga = false;
    let mut probe_rep = None;
    let mut slack = 0.0f64;
    let mut splice: Option<String> = None;
    let mut i = 2;
    while i < a.len() {
        let v = a.get(i + 1).cloned().unwrap_or_default();
        let mut one = false;
        match a[i].as_str() {
            "--out" => out = v,
            "--budget" => budget = v.parse()?,
            "--goal-stars" => goal_stars = Some(v.parse::<i16>()?),
            "--seed" => seed = v.parse()?,
            "--start-frame" => start_frame = Some(v.parse::<u32>()?),
            "--rom" => rom = v,
            "--report-every" => report_every = v.parse()?,
            "--probe" => probe_every = Some(v.parse::<u32>()?),
            "--trim-every" => trim_every = v.parse()?,
            "--probe-repair" => probe_rep = Some(v.parse::<u32>()?),
            "--slack" => slack = v.parse()?,
            "--splice" => splice = Some(v),
            "--no-ga" => {
                no_ga = true;
                one = true;
            }
            "--no-repair" => {
                repair = false;
                one = true;
            }
            "--no-cam" => {
                cam = false;
                one = true;
            }
            s => bail!("unknown argument {s}"),
        }
        i += if one { 1 } else { 2 };
    }
    if out.is_empty() {
        bail!("--out DIR");
    }
    std::fs::create_dir_all(format!("{out}/v"))?;
    let inputs = read_m64(&a[1])?;
    let gen_ = SM64GameGenerator::from_file(&rom)?;
    let mut game = gen_.create_game()?;
    // The stars to reach: as asked, or all the route reaches.
    let goal_stars = match goal_stars {
        Some(n) => n,
        None => {
            let mut most = 0;
            for (k, p) in inputs.iter().enumerate() {
                game.step_game(*p)?;
                most = most.max(read_state(&game, k as u32 + 1, false).stars);
            }
            game = gen_.create_game()?;
            most
        }
    };
    let Some(mut route) = Route::build(&mut game, inputs, goal_stars, None)? else {
        bail!("{}: never reaches {goal_stars} stars", a[1]);
    };
    // First frame Mario controls (default: skip menus).
    let first = start_frame.unwrap_or_else(|| route.trace.iter().position(gate_ok).unwrap_or(1) as u32).max(SAVE_EVERY);
    let goal0 = route.goal;
    let threads = rayon::current_num_threads();
    println!("mend {}: goal {goal_stars} stars at frame {goal0}, window starts from frame {first}, {threads} threads, budget {budget}, repair {repair}, cam {cam}, seed {seed}, {} saves", a[1], route.saves.at.len());
    println!("env: {}", DEFAULT_ENV.iter().map(|(k, _)| format!("{k}={}", std::env::var(k).unwrap_or_default())).collect::<Vec<_>>().join(" "));
    if let Some(every) = probe_every {
        probe(&mut game, &route, first, every, &out)?;
        return Ok(());
    }
    obrrt_search::movegenome::set_with_seed(std::env::var("OBRRT_MOVES_SEEDED").is_ok_and(|v| v == "1"));
    // A few policy rollouts per GA; the seed's variants are the point here.
    if std::env::var_os("OBRRT_MOVE_GA_N").is_none() {
        obrrt_search::movegenome::set_budget(6);
    }
    let params = GaParams { scale: 1.0, fitness: obrrt_search::ga::Fitness::Steps, new_ops: true, pink: 0.0, per_objective_budget: false };
    let slots: Vec<Mutex<(Worker, SM64Game)>> = (0..threads).map(|_| Ok(Mutex::new((Worker::new(&gen_)?, gen_.create_game()?)))).collect::<Result<_>>()?;
    if let Some(e) = probe_rep {
        return probe_repair(&slots, &route, first, e, params, cam, seed, &out);
    }
    use std::io::Write;
    let mut curve = std::io::BufWriter::new(std::fs::File::create(format!("{out}/curve.csv"))?);
    writeln!(curve, "ga_frames,replay_frames,trim_frames,secs,goal,saved,windows")?;
    let mut accepts = std::io::BufWriter::new(std::fs::File::create(format!("{out}/accepts.csv"))?);
    writeln!(accepts, "n,ga_frames,replay_frames,trim_frames,goal,saved,what,a,lead0,repairs,zone")?;
    write_m64(&format!("{out}/route.m64"), &route.inputs)?;
    let t0 = std::time::Instant::now();
    let mut o = Out { dir: out.clone(), curve, accepts, n: 0, goal0, t0 };
    let mut total = Stats::default();
    let mut bandit = Bandit::new((route.goal / BUCKET) as usize + 1);
    bandit.slack = slack;
    let mut plan_rng = SmallRng::seed_from_u64(seed ^ 0x6D65_6E64);
    if let Some(sp) = &splice {
        let mut guard = slots[0].lock().unwrap();
        let (w, g) = &mut *guard;
        let mut f = GaFollower { worker: w, params, cam };
        if let Some(n) = splice_route(&mut f, g, &route, sp, seed, &mut total, cam, budget)? {
            let what = format!("splice {sp}");
            commit(&mut game, &mut route, n.a, &n.pads, n.frame, &what, n.lead0, n.repairs, &total, &mut o)?;
        }
        println!("splice done: goal {}, GA {} frames", route.goal, total.ga_frames);
        drop(guard);
    }
    let (mut round, mut next_report) = (0u64, 0u64);
    o.point(&total, route.goal)?;
    let mut next_trim = 0u64;
    let mut trim_dry_at = u32::MAX;
    let opts_trim = Opts { cam };
    loop {
        if trim_every > 0 && total.ga_frames >= next_trim {
            next_trim = total.ga_frames + trim_every;
            // A pass on the route of the last dry pass can't find anything
            // (trims are deterministic). Trims may spend a fifth of the budget
            // up front plus half of what the GA has had: a long route's first
            // convergence can otherwise eat the whole budget.
            let limit = if no_ga { u64::MAX } else { budget / 5 + total.ga_frames / 2 };
            if route.goal == trim_dry_at || total.trim_frames >= limit {
                total.trim_skipped += 1;
            } else {
                loop {
                    let got = trim_pass(&slots, &mut game, &mut route, first, &opts_trim, &mut total, &mut o, &mut bandit, limit)?;
                    if got == 0 && total.trim_frames < limit {
                        trim_dry_at = route.goal;
                        break;
                    }
                    if total.trim_frames >= limit {
                        break;
                    }
                }
                println!("trim passes done: goal {} (-{}), {} trims, {} wins, {:.1} M frames", route.goal, goal0 - route.goal, total.trims, total.trim_wins, total.trim_frames as f64 / 1e6);
                o.point(&total, route.goal)?;
            }
        }
        if no_ga || total.ga_frames + total.replay_frames + total.trim_frames >= budget {
            break;
        }
        round += 1;
        let props: Vec<Prop> = (0..threads).filter_map(|_| bandit.propose(&route, first, &mut plan_rng)).collect();
        if props.is_empty() {
            break;
        }
        let rref = &route;
        let effort_gens = ((12.0 * bandit.effort).round() as usize).clamp(6, 240);
        let results: Vec<(Prop, Stats, Option<Found>)> = props
            .par_iter()
            .enumerate()
            .map(|(i, p)| {
                let mut stats = Stats::default();
                let mut guard = loop {
                    if let Some(g) = slots.iter().find_map(|s| s.try_lock().ok()) {
                        break g;
                    }
                    std::thread::yield_now();
                };
                let (w, g) = &mut *guard;
                let x = Ctx { route: rref, repair, opts: Opts { cam }, effort_gens, max_repairs: MAX_REPAIRS, end_ok: false, repair_tight: 1 };
                let mut f = GaFollower { worker: w, params, cam };
                let mut rng = SmallRng::seed_from_u64(seed ^ round.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i as u64 + 1).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
                let found = match try_window(&mut f, g, &x, *p, &mut rng, &mut stats) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("window {p:?}: {e}");
                        stats.errors += 1;
                        None
                    }
                };
                (*p, stats, found)
            })
            .collect();
        let mut founds: Vec<(Prop, Found)> = Vec::new();
        for (p, s, f) in results {
            total.add(&s);
            bandit.charge(&p, s.ga_frames, f.as_ref().map_or(0, |f| route.goal - f.frame));
            if let Some(f) = f {
                total.wins += 1;
                founds.push((p, f));
            }
        }
        founds.sort_by_key(|(p, f)| (f.frame, p.a));
        if let Some((_, f)) = founds.first() {
            // The best of the round; the others were against the old route.
            let what = format!("window {} tight {} b {}", LENGTHS[f.len], f.tight, f.b);
            let old_trace = route.trace.clone();
            let old_goal = route.goal;
            let saved = commit(&mut game, &mut route, f.a, &f.pads, f.frame, &what, f.lead0, f.repairs, &total, &mut o)?;
            bandit.changed(f.a, f.b, saved);
            let (a0, b0) = (f.a, f.b);
            // The round's other wins: their windows, if apart from this one,
            // tried again on the new route. A replay, so about free.
            for (_, g) in founds.iter().skip(1).filter(|(_, g)| g.b <= a0 || g.a >= b0) {
                total.rebase_tried += 1;
                let map = |x: u32| -> u32 {
                    if x <= a0 {
                        return x;
                    }
                    let want = &old_trace[x as usize];
                    let c = x.saturating_sub(old_goal - route.goal);
                    (c.saturating_sub(30)..=(c + 30).min(route.goal)).min_by_key(|&k| {
                        let s = &route.trace[k as usize];
                        if s.zone() == want.zone() { dist(s.pos, want.pos) as u32 } else { u32::MAX }
                    }).unwrap_or(c)
                };
                let (na, nb) = (map(g.a), map(g.b));
                if nb <= na || nb >= route.goal {
                    continue;
                }
                route.load(&mut game, na)?;
                for p in &g.window {
                    game.step_game(*p)?;
                }
                let st = read_state(&game, na + g.window.len() as u32, false);
                let mut pads = g.window.clone();
                total.replay_frames += pads.len() as u64;
                let lives_ok = st.alive() && nb > na + g.window.len() as u32;
                if !lives_ok {
                    continue;
                }
                if let Carry::Goal { pads, frame } = carry(&mut game, &route, &st, nb, &Opts { cam }, &mut pads)? {
                    total.replay_frames += pads.len() as u64;
                    if frame < route.goal {
                        total.rebase_ok += 1;
                        let what = format!("rebased {} tight {} b {}", LENGTHS[g.len], g.tight, nb);
                        let sv = commit(&mut game, &mut route, na, &pads, frame, &what, g.lead0, 0, &total, &mut o)?;
                        bandit.changed(na, nb, sv);
                    }
                }
            }
            bandit.effort = (bandit.effort / 2.0).max(1.0);
            bandit.dry = 0;
        } else {
            bandit.dry += 1;
            // About a sweep's worth of dry rounds: twice the generations.
            let sweep = ((route.goal - first) as u64 / (BUCKET as u64 * threads as u64)).max(4);
            if bandit.dry >= sweep {
                bandit.dry = 0;
                bandit.effort = (bandit.effort * 1.5).min(20.0);
            }
        }
        if total.ga_frames >= next_report {
            next_report = total.ga_frames + report_every;
            println!(
                "{:.0}s GA {:.1} M replay {:.1} M windows {} goal {} (-{}): sooner {} ({} f), carried {} goal {} notsooner {} off[died {} zone {} dialog {} prog {} off {} end {}] mean survived {:.0}; repairs {} met {} fail {} GA {:.1} M; wins {} ({} repaired); {}",
                t0.elapsed().as_secs_f64(), total.ga_frames as f64 / 1e6, total.replay_frames as f64 / 1e6, total.windows, route.goal, goal0 - route.goal,
                total.sooner, total.sooner_frames, total.carried, total.carried_goal, total.goal_not_sooner, total.off_died, total.off_zone, total.off_dialog, total.off_progress, total.off_off, total.off_end,
                total.survived as f64 / total.carried.max(1) as f64, total.repair_gas, total.repair_met, total.repair_fail, total.repair_ga_frames as f64 / 1e6, total.wins, total.wins_repaired, bandit.line()
            );
            o.point(&total, route.goal)?;
            o.stats(&total, route.goal)?;
        }
    }
    o.point(&total, route.goal)?;
    o.stats(&total, route.goal)?;
    println!("mend done: goal {goal0} -> {} (-{}), GA {} replay {} trim {}, {} accepts, {:.0} s; {}", route.goal, goal0 - route.goal, total.ga_frames, total.replay_frames, total.trim_frames, o.n, t0.elapsed().as_secs_f64(), bandit.line());
    Ok(())
}
