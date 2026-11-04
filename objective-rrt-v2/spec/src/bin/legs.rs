//! The landmark-route solver: solves a route given as an ordered list of
//! spec objectives ("legs", tools/spec/README.md) leg by leg with v2's open
//! GA (`Worker::attempt_full`: fixed horizon, γ-discounted ETA fitness, the
//! usual gen-0 seeding), from the previous leg's end, keeping a beam of
//! distinct end states.
//!
//! legs <route.json> [--out DIR] [--seed S] [--beam K] [--pop N] [--gens N]
//!      [--horizon-mult X] [--retry-mult X] [--rom FILE]
//!
//! Per leg and beam state: one attempt (budget from the leg, else the
//! route's, else 64 / 30 / ETA×2 + 120); the winner if it met the leg,
//! plus (`OBRRT_GA_QD=1`) the archive's met picks, become candidates. Each is
//! replayed from its parent, through the cutscene tail and any dialog or
//! menu (A on alternate frames) to the next controllable state. Candidates
//! are bucketed (zone, 128-unit cell, facing octant, action) keeping the
//! earliest Done, and the K earliest Done go on. A parent with no met
//! candidate retries once at 2× generations; a leg no parent meets stops the
//! route (reported with the best ETA left).
//!
//! Deterministic for a seed: every attempt's rng comes from (seed, leg,
//! parent, try); parents run in parallel (rayon) and are merged in order.
//!
//! Writes to DIR (default ../oss2/runs/legs-<name>-s<seed>, served by
//! oss2's serve.py): route.m64 (raw from power-on), route.json, paths.json,
//! regions.json, summary.json.

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail};
use obrrt_core::{Eta, GamePad, Mem, Objective, St, read_state};
use obrrt_search::ga::{GaParams, Worker};
use obrrt_spec::Spec;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use serde_json::{Value, json};
use sm64_binds::{SM64GameGenerator, SaveState};

const A: u16 = obrrt_core::buttons::A_BUTTON;
/// Defaults for unset switches: the P arm's movement and technique set
/// (RUNS.md) plus screen.sh's GP_CATCH / SWIM_HOLD.
const LEGS_DEFAULT_ENV: &[(&str, &str)] = &[
    ("OBRRT_MOVES", "1"),
    ("OBRRT_MOVE_GA", "1"),
    ("OBRRT_MOVE_GA_EXTRA", "1"),
    // Every movement policy, not only the library's own: off, the old
    // defaults had speed-kick and all the mode policies silently unplayed.
    ("OBRRT_MOVES_MODES", "1"),
    ("OBRRT_MOVES_EXTRA", "1"),
    // Legs seeded from a reference play the policies too.
    ("OBRRT_MOVES_SEEDED", "1"),
    ("OBRRT_CALM", "1"),
    ("OBRRT_LAKITU_ARC", "1"),
    ("OBRRT_PAINTING_FAST", "1"),
    ("OBRRT_GP_CATCH", "1"),
    ("OBRRT_SWIM_HOLD", "1"),
    ("OBRRT_TECH", "all"),
];
/// Frames a press-through (cutscene tail, dialogs, menus) may take.
const PRESS_LIMIT: usize = 1500;
/// Path samples every this many frames.
const SAMPLE: u32 = 3;
/// The TAS boot (run/src/main.rs TAS_BOOT_FRAMES): obrrt2's OBRRT_TAS_BOOT root.
const TAS_BOOT_FRAMES: usize = 1646;
const TAS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../ref/tas/full-120star-2026-U.m64");

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
    Ok(std::fs::write(path, out)?)
}

/// Inputs from power-on to an obrrt2 node (nodepath's walk).
fn node_inputs(run: &str, node: u32) -> Result<Vec<GamePad>> {
    let rec = obrrt_search::route::read_tree_bin(&format!("{run}/tree.bin"))?;
    let mut chain = vec![node];
    while let Some(&n) = chain.last() {
        let p = rec.get(n as usize + 1).ok_or_else(|| anyhow!("{run}: no node {n}"))?.0;
        if p < 0 {
            break;
        }
        chain.push(p as u32);
    }
    let mut out = rec[0].1.clone();
    for n in chain.iter().rev() {
        out.extend_from_slice(&rec[*n as usize + 1].1);
    }
    Ok(out)
}

/// `start` / `reference`: {"boot": "tas"}, {"run": dir, "node": id},
/// {"m64": file, "frames": n}. The inputs from power-on.
fn source_inputs(v: &Value, root: &str, what: &str) -> Result<(Vec<GamePad>, String)> {
    let path = |p: &str| if p.starts_with('/') { p.to_string() } else { format!("{root}/{p}") };
    if v.get("boot").is_some() {
        let mut t = read_m64(TAS)?;
        t.truncate(TAS_BOOT_FRAMES);
        return Ok((t, "boot: tas".into()));
    }
    if let Some(r) = v.get("run").and_then(|x| x.as_str()) {
        let node = v.get("node").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
        return Ok((node_inputs(&path(r), node)?, format!("run {r} node {node}")));
    }
    if let Some(m) = v.get("m64").and_then(|x| x.as_str()) {
        let mut t = read_m64(&path(m))?;
        if let Some(n) = v.get("frames").and_then(|x| x.as_u64()) {
            t.truncate(n as usize);
        }
        // `"from": frame`: a reference only: its fold starts there, for a leg
        // the movie also passes earlier (the fold reads it back from the desc).
        let from = v.get("from").and_then(|x| x.as_u64()).map_or(String::new(), |f| format!(" from {f}"));
        let d = format!("m64 {m} ({} frames){from}", t.len());
        return Ok((t, d));
    }
    bail!("{what}: expected {{\"boot\": \"tas\"}}, {{\"run\": dir, \"node\": id}} or {{\"m64\": file, \"frames\": n}}")
}

#[derive(Clone, Copy, Debug)]
struct Budget {
    population: usize,
    generations: usize,
    /// None: the leg's start ETA × horizon_mult + 120.
    horizon: Option<usize>,
}

fn parse_budget(v: Option<&Value>, base: Budget, path: &str) -> Result<Budget> {
    let Some(v) = v else { return Ok(base) };
    let m = v.as_object().ok_or_else(|| anyhow!("{path}: budget is {{population, generations, horizon}}"))?;
    let mut b = base;
    for (k, x) in m {
        let n = x.as_u64().ok_or_else(|| anyhow!("{path}.{k}: expected an integer"))? as usize;
        match k.as_str() {
            "population" => b.population = n.clamp(2, 512),
            "generations" => b.generations = n.clamp(1, 500),
            "horizon" => b.horizon = Some(n.clamp(10, 5000)),
            _ => bail!("{path}: unknown budget key {k} (population, generations, horizon)"),
        }
    }
    Ok(b)
}

struct Leg {
    /// The leg's spec; for a native leg a stand-in carrying its name and zones.
    spec: Spec,
    /// `"native": kind, "key_contains": s`: the Registry objective of that
    /// kind whose key contains s, generated from the leg's start state (a
    /// technique with its own overlays, e.g. Tech::LakituSkip hill).
    native: Option<(String, String)>,
    /// Index into the references: this leg's own (`"reference"` on the leg).
    reference: Option<usize>,
    /// Native objectives generated so far (any state: the reference folds'
    /// too), reused from states that may start them but don't generate them
    /// (the Registry makes objects' objectives only within 5000 of b; the
    /// tree attempts them from b's kin the same way).
    cache: Mutex<Vec<Arc<dyn Objective>>>,
    /// `"require_next": true`: a candidate counts only if the next leg's
    /// objective can start at its end (e.g. a BLJ that ends flying at a
    /// tree, so Tech::PoleRedirect can take it on).
    require_next: bool,
    /// `"done_requires_next": true`: the leg is Done only where the next
    /// leg's objective (one generated so far, e.g. by the reference fold) may
    /// start, so the GA optimises the pair (a BLJ ending flying at a tree).
    done_requires_next: bool,
    /// `"moves": "off"`: the move library's steering and the policy genomes
    /// off for this leg (OBRRT_MOVES / OBRRT_MOVE_GA as set for the others).
    moves_off: bool,
    /// `"moves": "kicks"`: generic steering, and of the move policies only
    /// the uphill and speed kicks (A+B jump kicks up steep slopes).
    moves_kicks: bool,
    /// `"moves": "policies"`: generic steering with every move policy (the
    /// set-up the first legs runs had: policies, no library steering).
    moves_policies: bool,
    /// `"stable_frames"` (default 20; 0 = off; the last leg: off): the
    /// stability probe's length for this leg's ground ends.
    stable_frames: u32,
    /// `"window_break": true`: a `--window` ends at this leg.
    window_break: bool,
    /// `"dialogs": "stop"`: a dialog opened after real progress ends the
    /// segment (pressed through, the leg goes on); default "avoid": no.
    dialog_stop: bool,
    budget: Budget,
    expect_frames: Option<u32>,
}

/// The native objective a leg names, from state `st`: of those generated
/// there with that kind and a key containing `key_contains` that may start
/// at `st`, the one with the lowest start ETA (ties: the first). None: none.
fn native_obj(kind: &str, key_contains: &str, st: &St) -> Option<Arc<dyn Objective>> {
    use obrrt_core::Generator;
    let mut rng = SmallRng::seed_from_u64(0x5eed);
    let mut best: Option<(f64, Arc<dyn Objective>)> = None;
    let debug = std::env::var("LEGS_DEBUG_NATIVE").is_ok();
    for o in obrrt_objectives::Registry.generate(st, &mut rng) {
        if debug && o.kind() == kind {
            eprintln!("    native {kind} at frame {}: {} base {} exists {} eta {:.0}", st.frame, o.key(), o.base_criteria(st), o.exists_in(st), obrrt_core::start_eta(o.as_ref(), st));
        }
        if o.kind() != kind || !o.key().contains(key_contains) || !o.base_criteria(st) || !o.exists_in(st) {
            continue;
        }
        let e = obrrt_core::start_eta(o.as_ref(), st);
        if e.is_finite() && best.as_ref().is_none_or(|b| e < b.0) {
            best = Some((e, o));
        }
    }
    best.map(|b| b.1)
}

impl Leg {
    /// The leg's objective from a start state, with budget `b` (natives keep
    /// their own unless the leg gives one).
    fn objective(&self, st: &St, b: Option<obrrt_core::GaBudget>) -> Option<Arc<dyn Objective>> {
        match &self.native {
            None => Some(Arc::new(match b {
                Some(b) => self.spec.clone_with_budget(b),
                None => self.spec.clone(),
            })),
            Some((k, c)) => {
                if let Some(o) = c.split('|').find_map(|c| native_obj(k, c, st)) {
                    let mut cache = self.cache.lock().unwrap();
                    if !cache.iter().any(|x| x.key() == o.key()) {
                        cache.push(o.clone());
                    }
                    return Some(o);
                }
                let cache = self.cache.lock().unwrap();
                c.split('|').find_map(|c| {
                    cache
                        .iter()
                        .filter(|o| o.key().contains(c) && o.base_criteria(st) && o.exists_in(st))
                        .map(|o| (obrrt_core::start_eta(o.as_ref(), st), o))
                        .filter(|x| x.0.is_finite())
                        .min_by(|a, b| a.0.total_cmp(&b.0))
                        .map(|x| x.1.clone())
                })
            }
        }
    }
}

/// A leg as the GA sees it: the spec, plus Done when a dialog opens right
/// after the start or (`"dialogs": "stop"`) at all: the dialog is pressed
/// through and the leg continues
/// from there (a level's intro dialog; Lakitu's on the way to the castle).
/// Without it the open GA's rollouts stall in the dialog.
struct LegObj<'a> {
    obj: &'a dyn Objective,
    /// `done_requires_next`: the next leg.
    next: Option<&'a Leg>,
    dialog_stop: bool,
    /// The leg's budget (None: the objective's own).
    budget: Option<obrrt_core::GaBudget>,
}

/// Whether a dialog opening at `s` ends the leg's segment: one right after
/// the leg's start (a level's intro dialog, unavoidable), or with
/// `"dialogs": "stop"` any (a boss's intro and after-hit dialogs, Lakitu's).
/// (A progress requirement, ETA down DIALOG_PROGRESS, stalled the King's
/// intro dialog, which opens a few units from the start.)
fn dialog_ends(stop: bool, s: &St, start: &St, _eta: f64, _eta0: f64) -> bool {
    s.frame <= start.frame + DIALOG_EARLY || stop
}

/// `done_requires_next`: whether the next leg can start from `s`.
fn next_can_start(next: &Leg, s: &St) -> bool {
    // In a cutscene, dialog or menu (a door, a painting, a star): the
    // press-through takes it to the next controllable state; no check.
    if !s.controllable() || s.modal() {
        return true;
    }
    if next.native.is_none() {
        // A spec leg: controllable, a finite start ETA, and not in the air
        // over a drop (a landmark reached mid-jump off the path).
        let over_drop = s.action & 0x800 != 0 && s.pos[1] - s.floor_y > DROP;
        return s.controllable() && !over_drop && obrrt_core::start_eta(&next.spec, s).is_finite();
    }
    let cached = next.cache.lock().unwrap().iter().any(|o| o.base_criteria(s) && o.exists_in(s) && obrrt_core::start_eta(o.as_ref(), s).is_finite());
    cached || next.objective(s, None).is_some_and(|o| obrrt_core::start_eta(o.as_ref(), s).is_finite())
}

/// Airborne this far above the floor below: over a drop.
const DROP: f32 = 400.0;

/// `done_requires_next`: the ETA while the leg is met but the next can't
/// start yet (about a re-aim).
const NEXT_FRAMES: f64 = 30.0;

/// A dialog this soon after a leg's start ends the segment in any case.
const DIALOG_EARLY: u32 = 30;
/// Dialog interruptions a leg may have.
const MAX_INTERRUPTS: usize = 8;

impl Objective for LegObj<'_> {
    fn kind(&self) -> &'static str { self.obj.kind() }
    fn key(&self) -> String { self.obj.key() }
    fn obj_weight(&self, b: &St) -> f64 { self.obj.obj_weight(b) }
    fn base_criteria(&self, _s: &St) -> bool { true }
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        let e = self.obj.eta(s, start, mem);
        if e.done() {
            if let Some(next) = self.next {
                // Not Done until the next leg can start here.
                if !next_can_start(next, s) {
                    return Eta::Frames(NEXT_FRAMES);
                }
            }
        }
        if !e.done() && s.in_dialog() && !start.in_dialog() && dialog_ends(self.dialog_stop, s, start, e.frames(), 0.0) {
            return Eta::Done;
        }
        e
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> { self.budget.or(self.obj.ga_budget()) }
    fn precision(&self) -> obrrt_core::Precision { self.obj.precision() }
    // done_requires_next generates the next leg's objectives: objects needed.
    fn needs_objects(&self) -> bool { self.obj.needs_objects() || self.next.is_some_and(|n| n.native.is_some() || n.spec.needs_objects()) }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> { self.obj.steer_target(s) }
    fn steer_button(&self, s: &St) -> u16 { self.obj.steer_button(s) }
    fn arrival(&self, s: &St) -> obrrt_core::Arrival { self.obj.arrival(s) }
    fn done_frames(&self, s: &St, m: &Mem) -> f64 { self.obj.done_frames(s, m) }
    fn pad_overlay(&self) -> Option<obrrt_core::PadOverlay> { self.obj.pad_overlay() }
    fn fixed(&self) -> Option<obrrt_core::Budget> { self.obj.fixed() }
    fn temperature(&self) -> f64 { self.obj.temperature() }
    fn kinship(&self) -> u32 { self.obj.kinship() }
    fn dense_keep(&self) -> obrrt_core::DenseKeep { self.obj.dense_keep() }
    fn success_value(&self) -> f64 { self.obj.success_value() }
    fn modal(&self) -> bool { self.obj.modal() }
}

/// A reference's per-leg (start, Done) frames, path samples and inputs.
struct Ref {
    desc: String,
    legs: Vec<Option<(u32, u32)>>,
    /// Per leg: the frame the fold started the leg's objective at (the
    /// first controllable, non-modal state after the leg's start), where
    /// its seed starts: our legs start after their own press-through, so a
    /// seed from the leg's start would replay the reference's dialog presses
    /// (A on alternate frames) in a controllable state: Mario jumping in place.
    seed_from: Vec<Option<u32>>,
    path: Vec<[f64; 6]>,
    inputs: Vec<GamePad>,
    top: bool,
    /// The camera yaw at each frame the fold read (index = frame).
    cam: Vec<Option<i16>>,
}

/// A reference's inputs replayed from another state keep their world
/// direction: the stick is relative to the camera, and the same state can
/// have another camera (BoB's spawn: about 50 degrees between the TAS's entry
/// and a route's). Closed loop: each frame's stick is turned by (reference's
/// camera - ours). Without a camera difference the inputs are unchanged.
fn world_seed(game: &mut sm64_binds::SM64Game, parent: &Cand, inputs: &[GamePad], cam: &[Option<i16>], from: usize) -> Result<Vec<GamePad>> {
    game.load_state(&parent.save)?;
    let mut out = Vec::with_capacity(inputs.len());
    let mut frame = parent.st.frame;
    let mut ours = parent.st.cam_yaw;
    for (k, p) in inputs.iter().enumerate() {
        let mut q = *p;
        if let Some(c) = cam.get(from + k).copied().flatten() {
            let d = (c as i32 - ours as i32) as i16;
            if d != 0 && (p.stick_x != 0 || p.stick_y != 0) {
                let (x, y) = (p.stick_x as f32, p.stick_y as f32);
                let a = x.atan2(-y) + d as f32 * std::f32::consts::TAU / 65536.0;
                let r = x.hypot(y);
                q = GamePad::new(p.button, (r * a.sin()).round().clamp(-127.0, 127.0) as i8, (-r * a.cos()).round().clamp(-127.0, 127.0) as i8);
            }
        }
        game.step_game(q)?;
        frame += 1;
        out.push(q);
        ours = read_state(game, frame, false).cam_yaw;
    }
    Ok(out)
}

/// A beam state.
struct Cand {
    /// Index of the previous leg's beam state this descends from.
    root: usize,
    /// Stopped by a dialog: the same leg goes on from here.
    interrupted: bool,
    /// `--window` > 1: the rollout's inputs after this leg's Done (the next
    /// legs' part), the next window's gen-0 seed. Empty otherwise.
    rest: Vec<GamePad>,
    /// Inputs from power-on.
    inputs: Vec<GamePad>,
    save: SaveState,
    st: St,
    /// Frame (from power-on) the leg was Done; the state after the
    /// press-through is `st`.
    done: u32,
    parent: Option<usize>,
    ga_frames: u64,
    gamma: Option<f64>,
    origin: &'static str,
    /// (frame, level, area, x, y, z) along this leg's part.
    path: Vec<[f64; 6]>,
}

fn sample(path: &mut Vec<[f64; 6]>, s: &St) {
    if s.frame % SAMPLE == 0 {
        path.push([s.frame as f64, s.level as f64, s.area as f64, s.pos[0].round() as f64, s.pos[1].round() as f64, s.pos[2].round() as f64]);
    }
}

/// Replays `genome` from (`save`, `st`), folding the leg's ETA; then the
/// cutscene tail and any dialog/menu until Mario is in control. None if
/// the leg isn't Done along it or Mario dies.
#[allow(clippy::too_many_arguments)]
fn replay(
    game: &mut sm64_binds::SM64Game,
    spec: &dyn Objective,
    next: Option<&Leg>,
    dialog_stop: bool,
    parent: &Cand,
    genome: &[GamePad],
    stable: u32,
    after: Option<&Leg>,
) -> Result<Option<(Vec<GamePad>, St, SaveState, u32, Vec<[f64; 6]>, bool)>> {
    game.load_state(&parent.save)?;
    let start = &parent.st;
    let eta0 = obrrt_core::start_eta(spec, start);
    let mut interrupted = false;
    let mut mem = Mem::default();
    let mut frame = start.frame;
    let mut played = Vec::new();
    let mut path = Vec::new();
    let mut done = None;
    let objects = spec.needs_objects();
    // As the GA's rollouts: the fold starts with the start state itself (a
    // native's latch can be set there, e.g. PoleRedirect's touch at the
    // trunk when the previous leg ended on it).
    let _ = spec.eta(start, start, &mut mem);
    for p in genome {
        game.step_game(*p)?;
        frame += 1;
        played.push(*p);
        let s = read_state(game, frame, objects);
        sample(&mut path, &s);
        if !s.alive() || s.lives < start.lives {
            return Ok(None);
        }
        // As LegObj: the spec's Done, or a dialog after real progress.
        let e = spec.eta(&s, start, &mut mem);
        if e.done() && next.is_none_or(|n| next_can_start(n, &read_state(game, frame, true))) {
            done = Some(frame);
            break;
        }
        if s.in_dialog() && !start.in_dialog() && dialog_ends(dialog_stop, &s, start, e.frames(), eta0) {
            done = Some(frame);
            interrupted = true;
            break;
        }
    }
    let Some(done) = done else {
        if std::env::var("LEGS_DEBUG_REPLAY").is_ok() {
            let s = read_state(game, frame, true);
            eprintln!("    replay: not Done after {} inputs: frame {} pos {:.0},{:.0},{:.0} action {:#x} eta {:?}", genome.len(), s.frame, s.pos[0], s.pos[1], s.pos[2], s.action, spec.eta(&s, start, &mut mem.clone()));
        }
        return Ok(None);
    };
    // Press through: A on alternate frames until controllable outside
    // dialogs and menus.
    let mut s = read_state(game, frame, false);
    let mut k = 0;
    while !(s.controllable() && !s.modal()) && k < PRESS_LIMIT {
        let pad = GamePad::new(if k % 2 == 0 { A } else { 0 }, 0, 0);
        game.step_game(pad)?;
        played.push(pad);
        frame += 1;
        k += 1;
        s = read_state(game, frame, false);
        sample(&mut path, &s);
    }
    if !(s.controllable() && !s.modal()) || !s.alive() {
        return Ok(None);
    }
    let s = read_state(game, frame, true);
    let save = game.save_state();
    // The stability probe: from an end on the ground, `stable` frames with a
    // neutral stick (restored afterwards; not part of the route).
    if stable > 0 && !interrupted && s.action & 0x800 == 0 && !unstable_probe(game, &s, stable, after)? {
        UNSTABLE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Ok(None);
    }
    game.load_state(&save)?;
    Ok(Some((played, s, save, done, path, interrupted)))
}

/// Candidates rejected by the probe in the current leg (route.json
/// `unstable_rejected`; output only).
static UNSTABLE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The stability probe of a leg end on the ground: `n` neutral frames from
/// `s` (the game in that state). Unstable (false) if Mario enters a butt or
/// stomach slide, a backward knockback or a freefall, drops more than 50, or
/// the next (spec) leg's start ETA rises by more than 5 frames.
fn unstable_probe(game: &mut sm64_binds::SM64Game, s: &St, n: u32, after: Option<&Leg>) -> Result<bool> {
    const BACK_KB: [u32; 3] = [0x00020460, 0x00020462, 0x00020464];
    const FREEFALL: u32 = 0x0100088C;
    let next_spec = after.filter(|l| l.native.is_none()).map(|l| &l.spec);
    let e0 = next_spec.map(|sp| obrrt_core::start_eta(sp, s));
    let mut frame = s.frame;
    for _ in 0..n {
        game.step_game(GamePad::new(0, 0, 0))?;
        frame += 1;
        let p = read_state(game, frame, false);
        if !p.controllable() || p.modal() {
            break;
        }
        let sliding = p.action & (1 << 18) != 0 || BACK_KB.contains(&p.action) || p.action == FREEFALL;
        // A drop counts only in the air (standing on something that sinks,
        // a pounded post or a platform, is not a fall).
        if sliding || (p.action & 0x800 != 0 && p.pos[1] < s.pos[1] - 50.0) || !p.alive() {
            return Ok(false);
        }
        if let (Some(sp), Some(e0)) = (next_spec, e0) {
            if obrrt_core::start_eta(sp, &p) > e0 + 5.0 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}


/// Live outputs (output only: nothing here feeds back into the search).
static LIVE_BASE: Mutex<Option<Value>> = Mutex::new(None);
static LIVE_PATHS: Mutex<Option<Value>> = Mutex::new(None);
static LIVE_RUN: Mutex<Option<Running>> = Mutex::new(None);
static LIVE_STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Seconds between live rewrites within a leg.
const LIVE_EVERY: u64 = 30;

/// The leg under way, for the live route.json.
struct Running {
    leg: usize,
    name: String,
    ga_frames: u64,
    best_left: f64,
    /// The earliest Done so far: (done frame, inputs from power-on, path,
    /// parent beam index).
    best: Option<(u32, Vec<GamePad>, Vec<[f64; 6]>, usize)>,
}

fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn write_atomic(path: &str, bytes: &[u8]) -> Result<()> {
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn live_note(frames: u64, left: f64, cand: Option<(u32, &[GamePad], &[[f64; 6]], usize)>) {
    let mut g = LIVE_RUN.lock().unwrap();
    let Some(r) = g.as_mut() else { return };
    r.ga_frames += frames;
    r.best_left = r.best_left.min(left);
    if let Some((d, inp, path, parent)) = cand {
        if r.best.as_ref().is_none_or(|b| d < b.0) {
            r.best = Some((d, inp.to_vec(), path.to_vec(), parent));
        }
    }
}

/// Every LIVE_EVERY s while a leg runs: route.json (the last full one plus
/// the running leg), paths.json (plus its best path so far) and route.m64
/// (that path, else the last full one's), written atomically.
fn live_writer(dir: &str) {
    let mut last = std::time::Instant::now();
    while !LIVE_STOP.load(std::sync::atomic::Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(500));
        if last.elapsed().as_secs() < LIVE_EVERY {
            continue;
        }
        last = std::time::Instant::now();
        let (Some(mut route), Some(r)) = (LIVE_BASE.lock().unwrap().clone(), LIVE_RUN.lock().unwrap().as_ref().map(|r| (r.leg, r.name.clone(), r.ga_frames, r.best_left, r.best.clone()))) else { continue };
        let (leg, name, ga, left, best) = r;
        route["live"] = json!(true);
        route["updated"] = json!(unix_now());
        let run = json!({"name": name, "index": leg, "status": "running", "ga_frames": ga,
            "best_eta_left": if left.is_finite() { json!(left.round()) } else { Value::Null },
            "best_done_frame": best.as_ref().map(|b| b.0)});
        if let Some(legs) = route["legs"].as_array_mut() {
            legs.truncate(leg);
            legs.push(run.clone());
        }
        route["running"] = run;
        let _ = write_atomic(&format!("{dir}/route.json"), serde_json::to_string_pretty(&route).unwrap_or_default().as_bytes());
        if let Some((_, inputs, path, parent)) = &best {
            if let Some(mut p) = LIVE_PATHS.lock().unwrap().clone() {
                // The running leg's entry: its best path so far.
                if let Some(legs) = p["legs"].as_array_mut() {
                    legs.truncate(leg);
                    legs.push(json!({"leg": leg, "candidates": [], "best_so_far": {"points": path, "parent": parent}}));
                }
                let _ = write_atomic(&format!("{dir}/paths.json"), serde_json::to_string(&p).unwrap_or_default().as_bytes());
            }
            let mut out = Vec::with_capacity(inputs.len() * 4);
            for p in inputs {
                out.extend_from_slice(&p.button.to_be_bytes());
                out.push(p.stick_x as u8);
                out.push(p.stick_y as u8);
            }
            let _ = write_atomic(&format!("{dir}/route.m64"), &out);
        }
    }
}

/// `--window-frames`: most counted legs in a window.
const WINDOW_MAX: usize = 4;

/// Whether a predicate is a zone change (a door, a painting, a warp).
fn changes_zone(p: &obrrt_spec::model::Pred) -> bool {
    use obrrt_spec::model::Pred;
    match p {
        Pred::All(v) | Pred::Any(v) => v.iter().any(changes_zone),
        Pred::InZone(_) | Pred::ZoneChanged(true) => true,
        _ => false,
    }
}

/// Slippery surface types (surface_terrains.h).
const SLIPPERY: [i16; 8] = [0x13, 0x14, 0x2A, 0x35, 0x36, 0x73, 0x74, 0x75];

/// Whether a predicate requires Mario grounded or controllable.
fn has_ground(p: &obrrt_spec::model::Pred) -> bool {
    use obrrt_spec::model::{Class, Pred};
    match p {
        Pred::All(v) => v.iter().any(has_ground),
        Pred::Any(v) => !v.is_empty() && v.iter().all(has_ground),
        Pred::ActionClass(c) => c.iter().all(|c| matches!(c, Class::Group(0x000) | Class::Group(0x040))),
        Pred::Controllable(true) => true,
        _ => false,
    }
}

/// A leg's landmark point (its last phase's target centre) and radius.
fn landmark(l: &Leg, st: &St) -> Option<([f32; 3], f32)> {
    use obrrt_spec::model::Shape;
    if l.native.is_some() {
        return None;
    }
    let t = l.spec.phases.last()?.target.as_ref()?;
    let r = match &t.shape {
        Shape::Sphere { r, .. } => *r,
        Shape::Box { min, max } => ((max[0] - min[0]).max(max[2] - min[2])) / 2.0,
        Shape::Point(_) => 150.0,
        Shape::Object(_) => return None,
    };
    obrrt_spec::target_dist(t, st, st.pos).map(|(_, c)| (c, r))
}

/// `--check`: per leg, lint against the tips (LANDMARK_TIPS.md numbers) and
/// the leg's reference; snap each landmark to stable flat ground on the
/// reference path; write the snapped route and a JSON report.
#[allow(clippy::too_many_arguments)]
fn check_route(doc: &Value, legs: &[Leg], refs: &[Ref], own: &[Option<usize>], st0: &St, game_gen: &SM64GameGenerator, out: &str) -> Result<()> {
    // One lite state per reference frame: (frame, zone, pos, action, ny,
    // floor type, controllable).
    type Lite = (u32, (u16, u8), [f32; 3], u32, f32, i16, bool);
    let mut issues: Vec<Vec<Value>> = vec![Vec::new(); legs.len()];
    let mut snapped: Vec<Option<[f32; 3]>> = vec![None; legs.len()];
    let mut stable_ref: Vec<Option<bool>> = vec![None; legs.len()];
    let mut add = |l: usize, tip: u32, level: &str, msg: String| {
        println!("  [{level}] leg {l} {}: {msg} (tip {tip})", legs[l].spec.name);
        issues[l].push(json!({"tip": tip, "level": level, "message": msg}));
    };
    for (l, leg) in legs.iter().enumerate() {
        // Tip 2: Done needs ground / control unless the leg ends mid-air.
        if leg.native.is_none() && leg.done_requires_next {
            if let Some(p) = leg.spec.phases.last() {
                let big = landmark(leg, st0).is_some_and(|x| x.1 > 150.0);
                if !has_ground(&p.done) {
                    add(l, 2, if big { "warning" } else { "note" }, format!("Done has no grounded/controllable condition{}", if big { " and a region wider than 150" } else { "" }));
                }
            }
        }
    }
    for (ri, r) in refs.iter().enumerate() {
        let mine: Vec<usize> = (0..legs.len()).filter(|&l| own[l] == Some(ri) && r.legs[l].is_some()).collect();
        if mine.is_empty() {
            continue;
        }
        let last = mine.iter().map(|&l| r.legs[l].unwrap().1).max().unwrap() + 40;
        let mut g = game_gen.create_game()?;
        let mut trace: Vec<Lite> = Vec::new();
        for (f, p) in r.inputs.iter().enumerate() {
            g.step_game(*p)?;
            let frame = f as u32 + 1;
            if frame > last {
                break;
            }
            if frame < st0.frame {
                continue;
            }
            let s = read_state(&g, frame, false);
            trace.push((frame, s.zone(), s.pos, s.action, s.floor_normal[1], s.floor_type, s.controllable()));
            for &l in mine.iter().filter(|&&l| r.legs[l].unwrap().1 == frame) {
                let leg = &legs[l];
                let st = read_state(&g, frame, true);
                // Tip 15: the Done state is stable in the reference.
                // (Native technique legs end in motion by design: no probe.)
                if leg.native.is_none() && leg.stable_frames > 0 && st.action & 0x800 == 0 && st.controllable() {
                    let save = g.save_state();
                    let ok = unstable_probe(&mut g, &st, 20, legs.get(l + 1))?;
                    g.load_state(&save)?;
                    stable_ref[l] = Some(ok);
                    if !ok {
                        add(l, 15, "error", format!("the reference's Done state at frame {frame} slides or falls within 20 frames"));
                    }
                }
                // Tip 20: a leg boundary on slippery or steep floor.
                if l + 1 < legs.len() && leg.native.is_none() && st.action & 0x800 == 0 && (SLIPPERY.contains(&st.floor_type) || st.floor_normal[1] < 0.9) {
                    add(l, 20, "warning", format!(
                        "the leg ends (in the reference, frame {frame}) on {} floor (type {:#x}, ny {:.2}): merge it and the next leg into one multi-phase leg (waypoints as phases) ending on stable ground",
                        if SLIPPERY.contains(&st.floor_type) { "slippery" } else { "steep" }, st.floor_type, st.floor_normal[1]));
                }
                let Some((t, _)) = landmark(leg, &st) else { continue };
                // Tip 1: the target's floor.
                let hit = g.with_memory(|m| obrrt_core::geo::floor_at(m, t[0], t[1] + 150.0, t[2]));
                match hit {
                    None => add(l, 1, "warning", format!("no floor under the target ({:.0}, {:.0}, {:.0})", t[0], t[1], t[2])),
                    Some(h) if h.normal[1] < 0.9 || SLIPPERY.contains(&h.kind) => add(l, 1, "warning", format!(
                        "target floor ny {:.2} type {:#x}{} at ({:.0}, {:.0}, {:.0})",
                        h.normal[1], h.kind, if SLIPPERY.contains(&h.kind) { " (slippery)" } else { " (steep)" }, t[0], h.y, t[2])),
                    _ => {}
                }
                // Tip 5: the straight segment from the previous landmark.
                let from = if l > 0 && legs[l - 1].spec.zones == leg.spec.zones { landmark(&legs[l - 1], &st).map(|x| x.0) } else { None }
                    .or_else(|| r.seed_from[l].and_then(|f0| trace.iter().find(|x| x.0 == f0).map(|x| x.2)));
                if let Some(a) = from {
                    let n = (obrrt_core::dist(a, t) / 100.0).ceil().max(1.0) as usize;
                    let mut bad: Vec<(f32, String)> = Vec::new();
                    for i in 1..n {
                        let u = i as f32 / n as f32;
                        let q: [f32; 3] = std::array::from_fn(|k| a[k] + (t[k] - a[k]) * u);
                        let hit = g.with_memory(|m| obrrt_core::geo::floor_at(m, q[0], q[1] + 300.0, q[2]));
                        let why = match hit {
                            None => Some("no floor".to_string()),
                            Some(h) if q[1] - h.y > 300.0 => Some(format!("a drop of {:.0}", q[1] - h.y)),
                            Some(h) if SLIPPERY.contains(&h.kind) => Some(format!("slippery floor {:#x}", h.kind)),
                            Some(h) if h.normal[1] < 0.7 => Some(format!("steep floor ny {:.2}", h.normal[1])),
                            _ => None,
                        };
                        if let Some(w) = why {
                            bad.push((u, w));
                        }
                    }
                    if !bad.is_empty() {
                        // Suggest the reference's grounded, flat point nearest
                        // the bad stretch's middle.
                        let mid = bad[bad.len() / 2].0;
                        let q: [f32; 3] = std::array::from_fn(|k| a[k] + (t[k] - a[k]) * mid);
                        let f0 = r.seed_from[l].unwrap_or(0);
                        let sug = trace
                            .iter()
                            .filter(|x| x.0 >= f0 && x.0 <= frame && x.3 & 0x800 == 0 && x.4 >= 0.9 && !SLIPPERY.contains(&x.5))
                            .min_by(|x, y| obrrt_core::dist(x.2, q).total_cmp(&obrrt_core::dist(y.2, q)))
                            .map(|x| x.2);
                        add(l, 5, "warning", format!(
                            "the straight line from ({:.0}, {:.0}, {:.0}) crosses {} of {} samples of bad terrain ({}){}",
                            a[0], a[1], a[2], bad.len(), n - 1, bad.iter().map(|b| b.1.clone()).collect::<Vec<_>>().join(", ").chars().take(160).collect::<String>(),
                            sug.map_or(String::new(), |p| format!("; add a landmark on the reference path at ({:.0}, {:.0}, {:.0})", p[0], p[1], p[2]))));
                    }
                }
            }
        }
        // Tips 1-2: snap each landmark to the reference's stable flat ground
        // near it (within 700), preferring higher points (the top of a climb).
        for &l in &mine {
            let Some((t, _)) = landmark(&legs[l], st0) else { continue };
            // Only arrival landmarks (default Done, or grounded Done); a door,
            // a star or a fight keeps its own Done.
            let ph = legs[l].spec.phases.last().unwrap();
            if ph.done_explicit && !has_ground(&ph.done) {
                continue;
            }
            let (f0, f1) = (r.seed_from[l].unwrap_or(0), r.legs[l].unwrap().1 + 30);
            let stable_at = |i: usize| -> bool {
                let y0 = trace[i].2[1];
                trace[i..(i + 20).min(trace.len())].iter().all(|x| x.3 & (1 << 18) == 0 && x.3 != 0x0100088C && ![0x00020460, 0x00020462, 0x00020464].contains(&x.3) && x.2[1] >= y0 - 50.0)
            };
            let best = (0..trace.len())
                .filter(|&i| {
                    let x = &trace[i];
                    x.0 >= f0 && x.0 <= f1 && x.6 && x.3 & 0x800 == 0 && (x.3 & 0x1C0 == 0 || x.3 & 0x1C0 == 0x40) && x.4 >= 0.9 && !SLIPPERY.contains(&x.5)
                        && obrrt_core::dist(x.2, t) <= 700.0 && legs[l].spec.zones.iter().any(|z| z.0 == x.1.0) && stable_at(i)
                })
                .max_by(|&i, &j| {
                    let sc = |k: usize| trace[k].2[1] - 0.3 * obrrt_core::dist(trace[k].2, t);
                    sc(i).total_cmp(&sc(j)).then(j.cmp(&i))
                });
            match best {
                Some(i) => snapped[l] = Some(trace[i].2),
                None => add(l, 1, "warning", "no stable flat grounded reference point within 700 to snap to".into()),
            }
        }
    }
    // The snapped route: targets moved, Done = grounded + controllable in 120.
    let mut new = doc.clone();
    let mut moves = Vec::new();
    if let Some(arr) = new["legs"].as_array_mut() {
        for (l, lv) in arr.iter_mut().enumerate() {
            let Some(p) = snapped.get(l).copied().flatten() else { continue };
            let old = landmark(&legs[l], st0).map(|x| x.0);
            let Some(ph) = lv["phases"].as_array_mut().and_then(|a| a.last_mut()) else { continue };
            ph["target"] = json!({"sphere": {"center": [p[0].round(), p[1].round(), p[2].round()], "r": 120}});
            ph["done"] = json!({"dist": {"le": 0}, "action_class": ["stationary", "moving"], "controllable": true});
            let d = old.map_or(0.0, |o| obrrt_core::dist(o, p));
            println!("  snap leg {l} {}: ({}) -> ({:.0}, {:.0}, {:.0}), moved {d:.0}", legs[l].spec.name, old.map_or("-".into(), |o| format!("{:.0}, {:.0}, {:.0}", o[0], o[1], o[2])), p[0], p[1], p[2]);
            moves.push(json!({"leg": l, "name": legs[l].spec.name, "from": old.map(|o| o.map(|x| x.round())), "to": p.map(|x| x.round()), "moved": d.round()}));
        }
        if let Some(d) = new.get_mut("doc") {
            *d = json!(format!("{} [Landmarks snapped by legs --check to stable flat ground on the references (LANDMARK_TIPS.md 1-2).]", d.as_str().unwrap_or("")));
        }
    }
    std::fs::write(out, serde_json::to_string_pretty(&new)?)?;
    let errors = issues.iter().flatten().filter(|i| i["level"] == "error").count();
    let report = json!({
        "route": doc.get("name"), "snapped_route": out, "errors": errors,
        "legs": legs.iter().enumerate().map(|(l, leg)| json!({"leg": l, "name": leg.spec.name, "issues": issues[l], "stable_in_reference": stable_ref[l], "snapped_to": snapped[l].map(|p| p.map(|x| x.round()))})).collect::<Vec<_>>(),
        "moves": moves,
    });
    let rp = format!("{}.check.json", out.trim_end_matches(".json"));
    std::fs::write(&rp, serde_json::to_string_pretty(&report)?)?;
    println!("check: {} issues ({errors} errors); snapped route {out}; report {rp}", issues.iter().flatten().count());
    if errors > 0 {
        bail!("{errors} landmark(s) not stable in their reference (tip 15)");
    }
    Ok(())
}

/// Distinct end-state bucket: zone, 128-unit cell, facing octant, action.
fn bucket(s: &St) -> (u16, u8, i32, i32, i32, u16, u32) {
    let c = |x: f32| (x / 128.0).floor() as i32;
    (s.level, s.area, c(s.pos[0]), c(s.pos[1]), c(s.pos[2]), (s.yaw as u16) >> 13, s.action)
}

struct LegOut {
    name: String,
    beam: Vec<Value>,
    frames: Option<u32>,
    ga_frames: u64,
    attempts: u64,
    retries: u64,
    met_bins: Vec<usize>,
    unstable: u64,
    done_by_gamma: std::collections::BTreeMap<String, usize>,
    failed: Option<f64>,
}

fn main() -> Result<()> {
    // The P config's movement switches unless set (a plain `legs` run has
    // the full move set), then obrrt2's own set-up (objectives::setup).
    for (k, v) in LEGS_DEFAULT_ENV {
        if std::env::var_os(k).is_none() {
            // SAFETY: single-threaded here, before any thread is started.
            unsafe { std::env::set_var(k, v) };
        }
    }
    let (tech, _) = obrrt_objectives::setup::from_env().map_err(anyhow::Error::msg)?;
    println!("env: {}; OBRRT_TECH={tech}", LEGS_DEFAULT_ENV.iter().map(|(k, _)| format!("{k}={}", std::env::var(k).unwrap_or_default())).collect::<Vec<_>>().join(" "));
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        bail!("usage: legs <route.json> [--out DIR] [--seed S] [--beam K] [--pop N] [--gens N] [--horizon-mult X] [--retry-mult X] [--rom FILE]");
    }
    let crate_dir = env!("CARGO_MANIFEST_DIR");
    let root = format!("{crate_dir}/..");
    let (mut out_dir, mut seed, mut beam_k) = (None, 1u64, 4usize);
    let (mut pop, mut gens, mut hmult, mut retry_mult) = (None, None, 2.0f64, 2.0f64);
    // --check OUT.json: lint and snap the route against its references
    // (tools/spec/LANDMARK_TIPS.md), write the snapped route to OUT.json and
    // the report next to it, and exit without solving.
    let mut check_out: Option<String> = None;
    // --window W (or the route's "window"): the GA optimises legs i..i+W-1
    // as chained stages and commits leg i only (1 = one leg at a time).
    let mut window_cli: Option<usize> = None;
    // --window-frames F: the window takes legs until their estimated frames
    // (reference frames, else 150) reach F: at least 2 counted legs, at most
    // WINDOW_MAX. Transition legs (zone changes; reference legs under 60
    // frames) ride along without counting, in both modes.
    let mut window_frames: Option<f64> = None;
    let mut rom = format!("{crate_dir}/../../baserom.us.z64");
    let mut i = 2;
    while i < a.len() {
        let v = a.get(i + 1).cloned().unwrap_or_default();
        match a[i].as_str() {
            "--out" => out_dir = Some(v),
            "--seed" => seed = v.parse()?,
            "--beam" => beam_k = v.parse()?,
            "--pop" => pop = Some(v.parse()?),
            "--gens" => gens = Some(v.parse()?),
            "--horizon-mult" => hmult = v.parse()?,
            "--retry-mult" => retry_mult = v.parse()?,
            "--check" => check_out = Some(v),
            "--window" => window_cli = Some(v.parse()?),
            "--window-frames" => window_frames = Some(v.parse()?),
            "--rom" => rom = v,
            s => bail!("unknown argument {s}"),
        }
        i += 2;
    }
    let text = std::fs::read_to_string(&a[1]).with_context(|| a[1].clone())?;
    let doc: Value = serde_json::from_str(&text).map_err(|e| anyhow!("{}: not JSON: {e}", a[1]))?;
    obrrt_spec::json::check_keys(doc.as_object().ok_or_else(|| anyhow!("$: expected an object"))?, "$", &["name", "start", "reference", "budget", "legs", "seed_ref", "window", "doc"])?;
    let name = doc.get("name").and_then(|x| x.as_str()).unwrap_or("route").to_string();
    let mut base = Budget { population: 64, generations: 30, horizon: None };
    base = parse_budget(doc.get("budget"), base, "$.budget")?;
    if let Some(p) = pop {
        base.population = p;
    }
    if let Some(g) = gens {
        base.generations = g;
    }
    let seed_ref = doc.get("seed_ref").and_then(|x| x.as_bool()).unwrap_or(false);
    let window = window_cli.or(doc.get("window").and_then(|x| x.as_u64()).map(|x| x as usize)).unwrap_or(1).max(1);
    let run_for_nodes = doc.get("reference").and_then(|r| r.get("run")).and_then(|x| x.as_str()).map(|r| format!("{root}/{r}"));
    let nodes = match &run_for_nodes {
        Some(r) => Some(obrrt_spec::node_positions(r)?),
        None => None,
    };
    let lookup = nodes.map(|n| move |id: u32| n.get(&id).copied());
    let nodes_fn: Option<&obrrt_spec::model::NodePos> = lookup.as_ref().map(|f| f as &obrrt_spec::model::NodePos);
    let mut legs = Vec::new();
    // (description, inputs, top-level): the references, top-level ones first.
    let mut ref_srcs: Vec<(String, Vec<GamePad>, bool)> = Vec::new();
    if let Some(rv) = doc.get("reference") {
        let list: Vec<&Value> = match rv {
            Value::Array(a) => a.iter().collect(),
            v => vec![v],
        };
        for r in list {
            let (inp, desc) = source_inputs(r, &root, "$.reference")?;
            ref_srcs.push((desc, inp, true));
        }
    }
    for (li, lv) in doc.get("legs").and_then(|x| x.as_array()).ok_or_else(|| anyhow!("$.legs: expected an array of legs"))?.iter().enumerate() {
        let path = format!("$.legs[{li}]");
        let mut m = lv.as_object().ok_or_else(|| anyhow!("{path}: expected an object"))?.clone();
        let budget = parse_budget(m.remove("budget").as_ref(), base, &format!("{path}.budget"))?;
        let expect_frames = m.remove("expect_frames").and_then(|x| x.as_u64()).map(|x| x as u32);
        let require_next = m.remove("require_next").and_then(|x| x.as_bool()).unwrap_or(false);
        let stable_frames = m.remove("stable_frames").and_then(|x| x.as_u64()).map_or(20, |x| x as u32);
        let window_break = m.remove("window_break").and_then(|x| x.as_bool()).unwrap_or(false);
        // Default on (but the last leg, below): Done only where the next leg
        // can start; `false` for a leg that must end mid-air.
        let done_requires_next = m.remove("done_requires_next").and_then(|x| x.as_bool()).unwrap_or(true);
        let (moves_off, moves_kicks, moves_policies) = match m.remove("moves").as_ref().map(|x| x.as_str()) {
            None | Some(Some("on")) => (false, false, false),
            Some(Some("off")) => (true, false, false),
            Some(Some("kicks")) => (false, true, false),
            Some(Some("policies")) => (false, false, true),
            _ => bail!("{path}.moves: \"on\" (default), \"off\", \"kicks\" or \"policies\""),
        };
        let dialog_stop = match m.remove("dialogs").as_ref().map(|x| x.as_str()) {
            None | Some(Some("avoid")) => false,
            Some(Some("stop")) => true,
            _ => bail!("{path}.dialogs: \"avoid\" (default) or \"stop\""),
        };
        if m.contains_key("variants") {
            bail!("{path}: legs take no variants");
        }
        // A per-leg reference (a "redirect": this leg's from another tree).
        let reference = match m.remove("reference") {
            Some(r) => {
                let (inp, desc) = source_inputs(&r, &root, &format!("{path}.reference"))?;
                let k = match ref_srcs.iter().position(|x: &(String, Vec<GamePad>, bool)| x.0 == desc) {
                    Some(k) => k,
                    None => {
                        ref_srcs.push((desc, inp, false));
                        ref_srcs.len() - 1
                    }
                };
                Some(k)
            }
            None => None,
        };
        let native = match m.remove("native") {
            Some(Value::String(k)) => {
                // One substring, or a list tried in order ("|"-joined here).
                let c = match m.remove("key_contains") {
                    Some(Value::String(c)) => c,
                    Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join("|"),
                    _ => String::new(),
                };
                Some((k, c))
            }
            Some(_) => bail!("{path}.native: an objective kind, e.g. \"Tech::LakituSkip\""),
            None => None,
        };
        let spec = if native.is_some() {
            // A stand-in spec: the leg's name and zones (for reports).
            let mut stand = serde_json::Map::new();
            for k in ["name", "zone", "zones", "doc"] {
                if let Some(v) = m.remove(k) {
                    stand.insert(k.into(), v);
                }
            }
            if let Some(k) = m.keys().next() {
                bail!("{path}: a native leg takes name, zone(s), native, key_contains, budget, reference, expect_frames, dialogs, require_next, done_requires_next, moves, stable_frames, window_break; not {k}");
            }
            stand.insert("phases".into(), json!([{"done": {"zone_changed": true}}]));
            Spec::parse(&Value::Object(stand), &path, nodes_fn)?.pop().unwrap()
        } else {
            Spec::parse(&Value::Object(m), &path, nodes_fn)?.pop().unwrap()
        };
        legs.push(Leg { spec, native, reference, cache: Mutex::new(Vec::new()), require_next, done_requires_next, moves_off, moves_kicks, moves_policies, stable_frames, window_break, dialog_stop, budget, expect_frames });
    }
    if legs.is_empty() {
        bail!("$.legs: no legs");
    }
    // The last leg has no next.
    if let Some(l) = legs.last_mut() {
        l.done_requires_next = false;
        l.stable_frames = 0;
    }
    let (start_inputs, start_desc) = source_inputs(doc.get("start").ok_or_else(|| anyhow!("$.start: missing"))?, &root, "$.start")?;
    let out_dir = out_dir.unwrap_or_else(|| format!("{root}/oss2/runs/legs-{name}-s{seed}"));
    std::fs::create_dir_all(&out_dir)?;

    let game_gen = SM64GameGenerator::from_file(&rom)?;
    let threads = rayon::current_num_threads();
    let workers: Vec<Mutex<Worker>> = (0..threads).map(|_| Ok(Mutex::new(Worker::new(&game_gen)?))).collect::<Result<_>>()?;
    let games: Vec<Mutex<sm64_binds::SM64Game>> = (0..threads).map(|_| Ok(Mutex::new(game_gen.create_game()?))).collect::<Result<_>>()?;
    let params = GaParams { scale: 1.0, fitness: obrrt_search::ga::Fitness::Steps, new_ops: true, pink: 0.0, per_objective_budget: false };

    // The start.
    let mut game = game_gen.create_game()?;
    for p in &start_inputs {
        game.step_game(*p)?;
    }
    let st0 = read_state(&game, start_inputs.len() as u32, true);
    if !st0.controllable() {
        bail!("start ({start_desc}) is not a controllable state (frame {})", st0.frame);
    }
    println!("start: {start_desc}, frame {} zone {}:{} pos {:.0},{:.0},{:.0}; {} legs, beam {beam_k}, window {window}, {threads} threads, QD {}",
        st0.frame, st0.level, st0.area, st0.pos[0], st0.pos[1], st0.pos[2], legs.len(), obrrt_search::qd::settings().map_or("off".to_string(), |q| format!("on (γ {:?})", q.gammas)));
    let _ = window_frames;

    // References: per reference, each leg's (start, Done) frames. A
    // top-level reference folds every leg in order from the start frame; a
    // per-leg one folds its legs, a run of them from the first controllable
    // state in the first one's zone (its entry frame is that leg's start).
    let mut refs: Vec<Ref> = Vec::new();
    for (ri, (desc, inp, top)) in ref_srcs.iter().enumerate() {
        let mine: Vec<usize> = (0..legs.len()).filter(|&l| *top || legs[l].reference == Some(ri)).collect();
        let ref_from: u32 = desc.rsplit_once(" from ").and_then(|(_, f)| f.parse().ok()).unwrap_or(0);
        let mut g = game_gen.create_game()?;
        let mut out: Vec<Option<(u32, u32)>> = vec![None; legs.len()];
        let mut seed_from: Vec<Option<u32>> = vec![None; legs.len()];
        let mut cam: Vec<Option<i16>> = vec![None; inp.len() + 2];
        let mut path = Vec::new();
        let need = true;
        let mut k = 0usize; // index into mine
        let mut start: Option<St> = None;
        let mut entry: Option<u32> = None;
        let mut obj: Option<Arc<dyn Objective>> = None;
        let mut mem = Mem::default();
        let mut last_done: Option<(usize, u32)> = None;
        for (f, p) in inp.iter().enumerate() {
            g.step_game(*p)?;
            let frame = f as u32 + 1;
            if frame < st0.frame.max(ref_from) || k >= mine.len() {
                if k >= mine.len() {
                    break;
                }
                continue;
            }
            let s = read_state(&g, frame, need);
            sample(&mut path, &s);
            cam[frame as usize] = Some(s.cam_yaw);
            let l = mine[k];
            // The run's leg start: the previous leg's Done if it was the one
            // before in this run, else (top: the start frame) the zone entry.
            if entry.is_none() {
                entry = match last_done {
                    Some((pl, d)) if pl + 1 == l => Some(d),
                    _ if *top && l == 0 => Some(st0.frame),
                    // A run's first leg: from the zone entry when it follows
                    // a zone change (its predecessor's zones differ); else
                    // its start is unknown (u32::MAX: no ref frames).
                    _ if obrrt_spec::model::zone_has(&legs[l].spec.zones, s.zone()) => {
                        let after_change = l == 0 || legs[l - 1].spec.zones != legs[l].spec.zones;
                        Some(if after_change { frame } else { u32::MAX })
                    }
                    _ => None,
                };
                if entry.is_none() {
                    continue;
                }
            }
            if start.is_none() {
                if s.controllable() && !s.modal() {
                    obj = legs[l].objective(&s, None);
                    if obj.is_none() {
                        continue;
                    }
                    start = Some(s.clone());
                    seed_from[l] = Some(frame);
                    mem = Mem::default();
                } else {
                    continue;
                }
            }
            let e = obj.as_ref().unwrap().eta(&s, start.as_ref().unwrap(), &mut mem);
            if e == Eta::Done {
                out[l] = Some((entry.unwrap(), frame));
                last_done = Some((l, frame));
                k += 1;
                start = None;
                entry = None;
            } else if e == Eta::Lost {
                start = None;
            }
        }
        println!("reference {desc}: leg (start, Done) {:?}", out);
        refs.push(Ref { desc: desc.clone(), legs: out, seed_from, path, inputs: inp.clone(), top: *top, cam });
    }
    // The reference a leg is compared with and seeded from: its own, else
    // the first top-level one.
    let leg_ref = |l: usize| legs[l].reference.or_else(|| refs.iter().position(|r| r.top));
    // Per leg: reference frames (if known) and whether it's a transition.
    let ref_frames: Vec<Option<f64>> = (0..legs.len())
        .map(|l| leg_ref(l).and_then(|ri| refs[ri].legs[l].and_then(|(a, b)| (a != u32::MAX).then(|| (b - a) as f64))))
        .collect();
    let transition: Vec<bool> = (0..legs.len())
        .map(|l| legs[l].native.is_none() && legs[l].spec.phases.last().is_some_and(|p| changes_zone(&p.done)) || ref_frames[l].is_some_and(|f| f < 60.0))
        .collect();
    // The legs after li in li's window.
    let window_list = |li: usize| -> Vec<usize> {
        let mut out = Vec::new();
        if legs[li].window_break || (window <= 1 && window_frames.is_none()) {
            return out;
        }
        let (mut counted, mut est) = (1usize, ref_frames[li].unwrap_or(150.0));
        for k in li + 1..legs.len() {
            let full = match window_frames {
                Some(f) => counted >= WINDOW_MAX || (counted >= 2 && est >= f),
                None => counted >= window,
            };
            if full && !transition[k] {
                break;
            }
            out.push(k);
            if !transition[k] {
                counted += 1;
            }
            est += ref_frames[k].unwrap_or(150.0);
            if legs[k].window_break {
                break;
            }
        }
        // A trailing transition leaves nothing to look ahead with.
        while out.last().is_some_and(|&k| transition[k]) {
            out.pop();
        }
        out
    };
    let window_desc = match window_frames {
        Some(f) => format!("frames {f} (max {WINDOW_MAX} legs, transitions free)"),
        None if window > 1 => format!("{window} legs (transitions free)"),
        None => "1".to_string(),
    };
    if window > 1 || window_frames.is_some() {
        println!("window: {window_desc}; per leg {:?}; transitions {:?}", (0..legs.len()).map(|l| window_list(l)).collect::<Vec<_>>(), (0..legs.len()).filter(|&l| transition[l]).collect::<Vec<_>>());
    }
    if let Some(out) = &check_out {
        let own: Vec<Option<usize>> = (0..legs.len()).map(leg_ref).collect();
        return check_route(&doc, &legs, &refs, &own, &st0, &game_gen, out);
    }

    // ---- the legs ----
    let mut beams: Vec<Vec<Cand>> = Vec::new();
    let st_root = st0.clone();
    beams.push(vec![Cand {
        root: 0,
        interrupted: false,
        rest: Vec::new(),
        inputs: start_inputs.clone(),
        save: game.save_state(),
        st: st0,
        done: st_root.frame,
        parent: None,
        ga_frames: 0,
        gamma: None,
        origin: "start",
        path: Vec::new(),
    }]);
    let mut leg_outs: Vec<LegOut> = Vec::new();
    // The outputs, rewritten after every leg (live) and at the end.
    macro_rules! write_outputs {
        ($live:expr) => {{
    // ---- outputs ----
    let solved = beams.len() - 1;
    let last = &beams[solved][0];
    // The best path's candidate per leg (parent pointers back).
    let mut chain = vec![0usize];
    for l in (1..=solved).rev() {
        let p = beams[l][*chain.last().unwrap()].parent.unwrap();
        chain.push(p);
    }
    chain.reverse(); // chain[l] = index in beams[l]
    write_m64(&format!("{out_dir}/route.m64"), &last.inputs)?;
    let mut legs_json = Vec::new();
    let mut regions = Vec::new();
    let colors = ["#60a5fa", "#f59e0b", "#a78bfa", "#34d399", "#f472b6", "#22d3ee"];
    for (li, lo) in leg_outs.iter().enumerate() {
        let best_frames = (li < solved).then(|| beams[li + 1][chain[li + 1]].done - beams[li][chain[li]].done);
        let refs_f: Vec<Value> = refs
            .iter()
            .map(|r| json!({"reference": r.desc, "frames": r.legs[li].and_then(|(a, b)| (a != u32::MAX).then(|| b - a)), "start_frame": r.legs[li].and_then(|x| (x.0 != u32::MAX).then_some(x.0)), "done_frame": r.legs[li].map(|x| x.1)}))
            .collect();
        let own = leg_ref(li);
        let ref0 = own.and_then(|ri| refs[ri].legs[li].and_then(|(a, b)| (a != u32::MAX).then(|| (b - a) as u64)));
        let expect = legs[li].expect_frames.map(|x| x as u64).or(ref0);
        let delta = match (best_frames, expect) {
            (Some(b), Some(e)) => Some(b as i64 - e as i64),
            _ => None,
        };
        legs_json.push(json!({
            "name": lo.name, "status": if lo.failed.is_some() { "failed" } else { "done" }, "from": if li == 0 { "start".to_string() } else { leg_outs[li - 1].name.clone() }, "to": lo.name,
            "frames": best_frames, "best_beam_frames": lo.frames, "ref_frames": refs_f, "reference": own.map(|ri| refs[ri].desc.clone()), "expect_frames": legs[li].expect_frames, "delta": delta,
            "ga_frames": lo.ga_frames, "unstable_rejected": lo.unstable, "attempts": lo.attempts, "retries": lo.retries,
            "distinct_done_bins": lo.met_bins.first().copied().unwrap_or(0), "done_bins_differing_by_gamma": lo.met_bins.iter().skip(1).copied().collect::<Vec<_>>(), "kept_by_gamma": lo.done_by_gamma,
            "failed": lo.failed.map(|b| json!({"best_eta_left": if b.is_finite() { json!(b.round()) } else { Value::Null }})),
            "budget": {"population": legs[li].budget.population, "generations": legs[li].budget.generations, "horizon": legs[li].budget.horizon},
            "beam": lo.beam,
        }));
        if li < solved {
            let (a, b) = (beams[li][chain[li]].done, beams[li + 1][chain[li + 1]].done);
            regions.push(json!({"from": a, "to": b, "label": lo.name, "note": format!("{} frames{}", b - a, delta.map_or(String::new(), |d| format!(" ({d:+} vs ref)"))), "color": colors[li % colors.len()]}));
        }
    }
    let landmarks: Vec<Value> = std::iter::once(json!({"name": "start", "zone": [st_root.level, st_root.area], "target": st_root.pos.map(|x| x.round())}))
        .chain(legs.iter().map(|l| {
            let p = l.spec.phases.last().unwrap();
            let target = p.target.as_ref().and_then(|t| obrrt_spec::target_dist(t, &st_root, st_root.pos)).map(|(_, c)| json!(c.map(|x| x.round())));
            json!({"name": l.spec.name, "zone": l.spec.zones.iter().map(|z| json!([z.0, z.1])).collect::<Vec<_>>(), "target": target})
        }))
        .collect();
    let total = (solved > 0).then(|| last.done - st_root.frame);
    // Per reference: its Done frame of the last leg it has (absolute game
    // frames) and, for top-level ones, the frames from the start. Plus the
    // stitched reference: each leg's own reference's frames, summed.
    let mut ref_total: Vec<Value> = refs
        .iter()
        .map(|r| {
            let last = r.legs.iter().enumerate().rev().find_map(|(l, x)| x.map(|x| (l, x.1)));
            json!({"reference": r.desc, "last_leg": last.map(|x| legs[x.0].spec.name.clone()), "done_frame": last.map(|x| x.1),
                "frames": if r.top { r.legs.get(solved.max(1) - 1).copied().flatten().map(|x| x.1 - st_root.frame) } else { None }})
        })
        .collect();
    let stitched: Option<u32> = (0..legs.len()).map(|l| leg_ref(l).and_then(|ri| refs[ri].legs[l].and_then(|(a, b)| (a != u32::MAX).then(|| b - a)))).sum();
    if refs.iter().any(|r| !r.top) {
        ref_total.push(json!({"reference": "stitched (each leg's own)", "frames": stitched}));
    }
    let route = json!({
        "name": name, "seed": seed, "beam": beam_k, "window": window_desc, "window_legs": (0..legs.len()).map(|l| window_list(l)).collect::<Vec<_>>(), "transition_legs": transition, "threads": threads, "start": start_desc, "start_frame": st_root.frame,
        "qd": obrrt_search::qd::settings().map(|q| json!({"gammas": q.gammas, "k": q.k, "grid": q.grid, "uniform": q.uniform, "fields": q.fields})),
        "landmarks": landmarks, "legs": legs_json,
        "totals": {"legs_solved": solved, "legs": legs.len(), "frames": total, "done_frame": (solved > 0).then_some(last.done), "end_frame": last.st.frame,
            "seconds": total.map(|t| (t as f64 / 30.0 * 100.0).round() / 100.0), "references": ref_total,
            "ga_frames": leg_outs.iter().map(|l| l.ga_frames).sum::<u64>()},
        "files": {"m64": "route.m64", "paths": "paths.json", "regions": "regions.json"},
    });
    let mut route = route;
    route["live"] = json!($live);
    route["updated"] = json!(unix_now());
    write_atomic(&format!("{out_dir}/route.json"), serde_json::to_string_pretty(&route)?.as_bytes())?;
    *LIVE_BASE.lock().unwrap() = Some(route.clone());
    write_atomic(&format!("{out_dir}/summary.json"), serde_json::to_string_pretty(&route["totals"])?.as_bytes())?;
    write_atomic(&format!("{out_dir}/regions.json"), serde_json::to_string_pretty(&regions)?.as_bytes())?;
    let paths = json!({
        "fields": ["frame", "level", "area", "x", "y", "z"], "every": SAMPLE,
        "legs": (1..=solved).map(|l| json!({"leg": l - 1, "candidates": beams[l].iter().map(|c| json!({"parent": c.parent, "points": c.path})).collect::<Vec<_>>()})).collect::<Vec<_>>(),
        "best": (1..=solved).map(|l| chain[l]).collect::<Vec<_>>(),
        "references": refs.iter().map(|r| json!({"reference": r.desc, "legs": r.legs, "points": r.path})).collect::<Vec<_>>(),
    });
    write_atomic(&format!("{out_dir}/paths.json"), serde_json::to_string(&paths)?.as_bytes())?;
    *LIVE_PATHS.lock().unwrap() = Some(paths);
    (route, legs_json, total, ref_total)
        }};
    }
    // The live writer: every LIVE_EVERY s, route.json with the leg under way.
    let live_thread = {
        let dir = out_dir.clone();
        std::thread::spawn(move || live_writer(&dir))
    };
    // Before the first leg too: without this a run shows nothing until its
    // first leg ends (the live writer has no base to add the running leg to).
    let _ = write_outputs!(true);
    let moves_env = std::env::var("OBRRT_MOVES").is_ok_and(|v| v == "1");
    obrrt_search::movegenome::set_with_seed(std::env::var("OBRRT_MOVES_SEEDED").is_ok_and(|v| v == "1"));
    for (li, leg) in legs.iter().enumerate() {
        // `"moves": "off"` legs: the library's steering and policy genomes off.
        obrrt_objectives::moves::set_steering(moves_env && !leg.moves_off && !leg.moves_kicks && !leg.moves_policies);
        obrrt_search::movegenome::set_forced_off(leg.moves_off);
        obrrt_search::movegenome::set_only(leg.moves_kicks.then(|| vec!["uphill-kick", "speed-kick", "jump-kick", "steep-climb"]));
        // The first generation plays every plausible policy alone, then
        // combinations (movegenome::gen0_plans), unless OBRRT_MOVE_GA_N says.
        if std::env::var_os("OBRRT_MOVE_GA_N").is_none() && std::env::var("OBRRT_MOVES_SEEDED").is_ok_and(|v| v == "1") {
            obrrt_search::movegenome::set_budget(leg.budget.population / 2);
        }
        UNSTABLE.store(0, std::sync::atomic::Ordering::Relaxed);
        *LIVE_RUN.lock().unwrap() = Some(Running { leg: li, name: leg.spec.name.clone(), ga_frames: 0, best_left: f64::INFINITY, best: None });
        let prev = beams.last().unwrap();
        let t0 = std::time::Instant::now();
        let mut finished: Vec<Cand> = Vec::new();
        let (mut ga_frames, mut attempts, mut retries) = (0, 0, 0);
        let mut met_bins: Vec<usize> = Vec::new();
        let mut best_left = f64::INFINITY;
        let mut interrupts = 0usize;
        let mut pending_own: Vec<Cand> = Vec::new();
        for round in 0..=MAX_INTERRUPTS {
        let cur: &Vec<Cand> = if round == 0 { prev } else { &pending_own };
        if cur.is_empty() {
            break;
        }
        let results: Vec<Result<(Vec<Cand>, u64, u64, u64, Vec<usize>, f64)>> = cur
            .par_iter()
            .enumerate()
            .map(|(pi, parent)| {
                let root = if round == 0 { pi } else { parent.root };
                let mut w = loop {
                    if let Some(w) = workers.iter().find_map(|w| w.try_lock().ok()) {
                        break w;
                    }
                    std::thread::yield_now();
                };
                let mut gm = loop {
                    if let Some(g) = games.iter().find_map(|g| g.try_lock().ok()) {
                        break g;
                    }
                    std::thread::yield_now();
                };
                let mut out = Vec::new();
                let wlist = window_list(li);
                let (mut ga_frames, mut attempts, mut retries) = (0u64, 0u64, 0u64);
                let mut met_bins = Vec::new();
                let mut best_left = f64::INFINITY;
                let Some(obj0) = leg.objective(&parent.st, None) else {
                    eprintln!("  leg {li} parent {pi}: no {:?} objective generated from frame {}", leg.native, parent.st.frame);
                    return Ok((out, 0, 0, 0, met_bins, best_left));
                };
                let eta0 = obrrt_core::start_eta(obj0.as_ref(), &parent.st);
                if !eta0.is_finite() {
                    return Ok((out, 0, 0, 0, met_bins, best_left));
                }
                // Already met where the parent is (e.g. a door leg after a
                // skip that ends inside): the parent itself, 0 frames.
                if obj0.eta(&parent.st, &parent.st, &mut Mem::default()).done() {
                    gm.load_state(&parent.save)?;
                    out.push(Cand {
                        root,
                        interrupted: false,
                        rest: Vec::new(),
                        inputs: parent.inputs.clone(),
                        save: gm.save_state(),
                        st: parent.st.clone(),
                        done: parent.st.frame,
                        parent: Some(root),
                        ga_frames: 0,
                        gamma: None,
                        origin: "met at start",
                        path: Vec::new(),
                    });
                    return Ok((out, 0, 0, 0, met_bins, best_left));
                }
                // The reference's inputs for this leg, as the seed genome.
                let seed_genome: Option<Vec<GamePad>> = if !parent.rest.is_empty() {
                    Some(parent.rest.clone())
                } else if seed_ref {
                    leg_ref(li).and_then(|ri| {
                        let (_, b) = refs[ri].legs[li]?;
                        let a = refs[ri].seed_from[li]?;
                        (a < b).then(|| {
                            let raw = &refs[ri].inputs[a as usize..b as usize];
                            world_seed(&mut gm, parent, raw, &refs[ri].cam, a as usize).unwrap_or_else(|_| raw.to_vec())
                        })
                    })
                } else {
                    None
                };
                // A parent still exactly on the reference: the reference's own
                // end of this leg is a candidate too, whatever the GA finds. A
                // seeded GA can tie it with a state a few units off, which
                // breaks a frame-exact trick in the next leg (tas-first-star:
                // 8 units at the slope, then 171 M GA frames without the warp).
                let mut ref_floor = None;
                if seed_ref && std::env::var("OBRRT_LEGS_REF_FLOOR").map_or(true, |v| v != "0") {
                    if let Some((ri, b)) = leg_ref(li).and_then(|ri| refs[ri].legs[li].map(|(_, b)| (ri, b as usize))) {
                        let at = parent.inputs.len();
                        if at < b && b <= refs[ri].inputs.len() && refs[ri].inputs[..at] == parent.inputs[..] {
                            gm.load_state(&parent.save)?;
                            let mut path = if round == 0 { Vec::new() } else { parent.path.clone() };
                            let mut st = parent.st.clone();
                            for (k, p) in refs[ri].inputs[at..b].iter().enumerate() {
                                gm.step_game(*p)?;
                                st = read_state(&gm, (at + k + 1) as u32, true);
                                sample(&mut path, &st);
                            }
                            let mut inputs = parent.inputs.clone();
                            inputs.extend_from_slice(&refs[ri].inputs[at..b]);
                            let done = st.frame;
                            ref_floor = Some(Cand { root, interrupted: false, rest: Vec::new(), inputs, save: gm.save_state(), st, done, parent: Some(root), ga_frames: 0, gamma: None, origin: "reference", path });
                        }
                    }
                }
                for t in 0..2u64 {
                    let mult = if t == 0 { 1.0 } else { retry_mult };
                    let horizon = leg.budget.horizon.unwrap_or(((eta0 * hmult) as usize + 120).clamp(60, 5000));
                    let budget = obrrt_core::GaBudget {
                        population: leg.budget.population,
                        generations: ((leg.budget.generations as f64 * mult).round() as usize).max(1),
                        horizon,
                    };
                    let obj_t = if leg.native.is_some() { obj0.clone() } else { leg.objective(&parent.st, Some(budget)).unwrap() };
                    let mut rng = SmallRng::seed_from_u64(
                        seed ^ (li as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (pi as u64 + 1).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ t.wrapping_mul(0x1656_67B1_9E37_79F9) ^ (round as u64).wrapping_mul(0x27D4_EB2F_1656_67C5),
                    );
                    let next_leg = legs.get(li + 1).filter(|_| leg.done_requires_next);
                    let leg_obj = LegObj { obj: obj_t.as_ref(), next: next_leg, dialog_stop: leg.dialog_stop, budget: Some(budget) };
                    // --window: the next legs as chained stages (each with its
                    // own leg's budget; a native one from what's been seen of
                    // it), up to a window_break, the route's end, or a leg
                    // with no objective yet.
                    let mut more: Vec<(Arc<dyn Objective>, &Leg, Option<&Leg>, obrrt_core::GaBudget)> = Vec::new();
                    {
                        for &k in &wlist {
                            let lk = &legs[k];
                            let bk = obrrt_core::GaBudget { population: lk.budget.population, generations: lk.budget.generations, horizon: lk.budget.horizon.unwrap_or(horizon) };
                            let o = match &lk.native {
                                None => lk.objective(&parent.st, Some(bk)),
                                Some(_) => lk.cache.lock().unwrap().first().cloned(),
                            };
                            let Some(o) = o else { break };
                            more.push((o, lk, legs.get(k + 1).filter(|_| lk.done_requires_next), bk));
                            if lk.window_break {
                                break;
                            }
                        }
                    }
                    let more_objs: Vec<LegObj> = more.iter().map(|(o, lk, nx, bk)| LegObj { obj: o.as_ref(), next: *nx, dialog_stop: lk.dialog_stop, budget: Some(*bk) }).collect();
                    let mut stages: Vec<&dyn Objective> = vec![&leg_obj];
                    stages.extend(more_objs.iter().map(|o| o as &dyn Objective));
                    let att = w.attempt_full(&stages, &parent.st, Some(&parent.save), &mut rng, params, seed_genome.as_deref(), None)?;
                    attempts += 1;
                    retries += t;
                    ga_frames += att.frames;
                    live_note(att.frames, att.segments.first().map_or(f64::INFINITY, |s| if s.success { 0.0 } else { -s.progress }), None);
                    let qd = obrrt_search::qd::take_picks();
                    let mut genomes: Vec<(Vec<GamePad>, Option<f64>, &'static str)> = Vec::new();
                    if let Some(seg) = att.segments.first() {
                        if seg.success {
                            // With a window: the whole rollout (leg i, then the
                            // next legs' part as the next window's seed).
                            let g = if stages.len() > 1 { att.segments.iter().flat_map(|s| s.inputs.iter().copied()).collect() } else { seg.inputs[..seg.played.min(seg.inputs.len())].to_vec() };
                            genomes.push((g, qd.as_ref().map(|_| obrrt_search::qd::settings().unwrap().gammas[0]), "winner"));
                        } else if seg.end.is_none() {
                            eprintln!("  leg {li} parent {pi} try {t}: unmet, no end state (died {}, {} inputs, progress {:.0})", att.died, seg.inputs.len(), -seg.progress);
                        } else if -seg.progress < best_left {
                            best_left = -seg.progress;
                            if let Some((e, _)) = &seg.end {
                                eprintln!("  leg {li} parent {pi} try {t}: unmet, ETA left {:.0} at frame {} zone {}:{} pos {:.0},{:.0},{:.0} action {:#x} dialog {} controllable {}",
                                    best_left, e.frame, e.level, e.area, e.pos[0], e.pos[1], e.pos[2], e.action, e.dialog, e.controllable());
                            }
                        }
                    }
                    if let Some(q) = &qd {
                        met_bins = q.met_bins.clone();
                        // With a window a pick may have met only leg i: the
                        // replay folds leg i alone, so take the picks that
                        // got anywhere (met or not) then.
                        for p in q.picks.iter().filter(|p| p.met || stages.len() > 1) {
                            if !genomes.iter().any(|g| g.0 == p.genome) {
                                genomes.push((p.genome.clone(), Some(p.gamma), "qd"));
                            }
                        }
                    }
                    for (g, gamma, origin) in genomes {
                        if let Some((played, st, save, done, path, interrupted)) = replay(&mut gm, obj_t.as_ref(), next_leg, leg.dialog_stop, parent, &g, if leg.native.is_some() { 0 } else { leg.stable_frames }, legs.get(li + 1))? {
                            // The remainder, if no press-through came between.
                            let di = (done - parent.st.frame) as usize;
                            let rest = if stages.len() > 1 && played.len() == di && g.len() > di { g[di..].to_vec() } else { Vec::new() };
                            let mut inputs = parent.inputs.clone();
                            inputs.extend_from_slice(&played);
                            let mut full = if round == 0 { Vec::new() } else { parent.path.clone() };
                            full.extend(path);
                            live_note(0, f64::INFINITY, Some((done, &inputs, &full, root)));
                            out.push(Cand { root, interrupted, rest, inputs, save, st, done, parent: Some(root), ga_frames: 0, gamma, origin, path: full });
                        }
                    }
                    if leg.require_next {
                        if let Some(nl) = legs.get(li + 1) {
                            let before = out.len();
                            out.retain(|c| c.interrupted || nl.objective(&c.st, None).is_some_and(|o| obrrt_core::start_eta(o.as_ref(), &c.st).is_finite()));
                            if before > out.len() {
                                eprintln!("  leg {li} parent {pi} try {t}: {} of {before} candidates dropped (the next leg can't start there)", before - out.len());
                            }
                        }
                    }
                    if !out.is_empty() {
                        break;
                    }
                }
                out.extend(ref_floor);
                let n = out.len().max(1) as u64;
                for c in &mut out {
                    c.ga_frames = ga_frames / n;
                }
                Ok((out, ga_frames, attempts, retries, met_bins, best_left))
            })
            .collect();
        let mut all = Vec::new();
        for r in results {
            let (c, g, at, rt, mb, bl) = r?;
            all.extend(c);
            ga_frames += g;
            attempts += at;
            retries += rt;
            if met_bins.len() < mb.len() {
                met_bins.resize(mb.len(), 0);
            }
            for (i, m) in mb.iter().enumerate() {
                met_bins[i] += m;
            }
            best_left = best_left.min(bl);
        }
        // Interrupted ones go round again (distinct, earliest first).
        let (mut stopped, done): (Vec<Cand>, Vec<Cand>) = all.into_iter().partition(|c| c.interrupted);
        finished.extend(done);
        stopped.sort_by_key(|c| (c.st.frame, c.root));
        let mut seen_i = std::collections::HashSet::new();
        stopped.retain(|c| seen_i.insert(bucket(&c.st)));
        stopped.truncate(beam_k.max(1));
        interrupts += stopped.len();
        if !stopped.is_empty() {
            println!("  leg {li} round {round}: {} stopped by a dialog at frames {:?}; continuing from there", stopped.len(), stopped.iter().map(|c| c.done).collect::<Vec<_>>());
        }
        pending_own = stopped;
        }
        let mut all = finished;
        let mut done_by_gamma = std::collections::BTreeMap::new();
        for c in &all {
            *done_by_gamma.entry(c.gamma.map_or("-".into(), |g| format!("{g}"))).or_insert(0usize) += 1;
        }
        // Distinct buckets, earliest Done; then the K earliest.
        all.sort_by_key(|c| (c.done, c.st.frame, c.parent));
        let mut seen = std::collections::HashSet::new();
        all.retain(|c| c.origin == "reference" || seen.insert(bucket(&c.st)));
        // The reference's end stays in the beam, last if it must.
        let floor = all.iter().position(|c| c.origin == "reference").filter(|&i| i >= beam_k.max(1)).map(|i| all.remove(i));
        all.truncate(beam_k.max(1) - floor.is_some() as usize);
        all.extend(floor);
        let name = legs[li].spec.name.clone();
        let failed = all.is_empty().then_some(best_left);
        let _ = interrupts;
        let frames = all.first().map(|c| c.done - prev[c.parent.unwrap()].done);
        let beam_json: Vec<Value> = all
            .iter()
            .map(|c| {
                let pd = prev[c.parent.unwrap()].done;
                json!({"done_frame": c.done, "end_frame": c.st.frame, "leg_frames": c.done - pd, "parent": c.parent, "origin": c.origin, "gamma": c.gamma,
                    "end": {"zone": [c.st.level, c.st.area], "pos": c.st.pos.map(|x| x.round()), "yaw": c.st.yaw, "speed": (c.st.forward_vel * 10.0).round() / 10.0, "action": format!("{:#x}", c.st.action)}})
            })
            .collect();
        println!(
            "leg {li} {name}: {} ({} candidates kept, {attempts} attempts, {retries} retries, {:.1} M GA frames, {:.0}s){}",
            frames.map_or("FAILED".into(), |f| format!("{f} frames")),
            all.len(),
            ga_frames as f64 / 1e6,
            t0.elapsed().as_secs_f64(),
            if met_bins.is_empty() { String::new() } else { format!("; distinct Done bins {} (γ 1.. differing from γ 0: {:?}), kept by γ {done_by_gamma:?}", met_bins[0], &met_bins[1..]) }
        );
        let unstable = UNSTABLE.load(std::sync::atomic::Ordering::Relaxed);
        if unstable > 0 {
            println!("  leg {li}: {unstable} candidate ends rejected by the stability probe");
        }
        leg_outs.push(LegOut { name, beam: beam_json, frames, ga_frames, attempts, retries, met_bins, unstable, done_by_gamma, failed });
        *LIVE_RUN.lock().unwrap() = None;
        if !all.is_empty() {
            beams.push(all);
            let _ = write_outputs!(true);
            continue;
        }
        if all.is_empty() {
            println!("  leg {li} failed: best ETA left {best_left:.0} frames; stopping");
            break;
        }
    }

    let (route, legs_json, total, ref_total) = write_outputs!(false);
    LIVE_STOP.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = live_thread.join();
    println!("\n{:<14} {:>7} {:>7} {:>7} {:>9}", "leg", "frames", "ref", "delta", "GA Mfr");
    for l in &legs_json {
        println!(
            "{:<14} {:>7} {:>7} {:>7} {:>9.2}",
            l["name"].as_str().unwrap_or(""),
            l["frames"].as_u64().map_or("-".into(), |x| x.to_string()),
            l["delta"].as_i64().zip(l["frames"].as_i64()).map_or("-".into(), |(d, f)| (f - d).to_string()),
            l["delta"].as_i64().map_or("-".into(), |x| format!("{x:+}")),
            l["ga_frames"].as_u64().unwrap_or(0) as f64 / 1e6
        );
    }
    println!("total: {} frames ({}), references {:?}", total.map_or("-".into(), |t| t.to_string()), route["totals"]["seconds"], ref_total);
    // Which policies the first generations played and which won (diagnostics).
    if obrrt_search::movegenome::enabled() {
        println!("{}", obrrt_search::movegenome::stats_line());
        write_atomic(&format!("{out_dir}/move_ga.json"), serde_json::to_string_pretty(&obrrt_search::movegenome::stats_json())?.as_bytes())?;
    }
    println!("out: {out_dir}");
    Ok(())
}
