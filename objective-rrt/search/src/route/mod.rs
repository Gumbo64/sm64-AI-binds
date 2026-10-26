//! Route optimisation bench: take a finished run's route to a goal (a star),
//! and make it faster, independently of the tree search.
//!
//! - The route's own states become **waypoints** (generic goals: same zone,
//!   within 150 units, at least the same progress, the same object progress),
//!   so any route can be optimised without per-star code.
//! - Inputs are a **stream** of the inputs for frames where Mario is
//!   controllable. Dialog/menu frames replay the original route's inputs for
//!   that dialog or menu ([`Policy`]), other uncontrollable frames (warps,
//!   star dance) press A every 4 frames. A route sped up early still lines up
//!   later: the stream only advances when Mario can act.
//! - A **method** ([`Method`], one file each in `methods/`) proposes trials:
//!   "re-solve waypoints `stages` starting from the best route's state at
//!   waypoint `from`". The [`Bench`] runs them on all workers (a GA over the
//!   stream, seeded with the best route's own stream: steps fitness, so
//!   sooner at every waypoint is better), splices any trial that met all its
//!   waypoints into the full route, and keeps it if replaying from spawn
//!   reaches the goal sooner.

pub mod methods;

use std::fs;
use std::io::Write as _;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Result, bail, ensure};
use obrrt_core::buttons::A_BUTTON;
use obrrt_core::{Budget, GamePad, ObjId, Objective, Progress, St, dist, read_state};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use sm64_binds::{SM64Game, SM64GameGenerator, SaveState};

use crate::ga::{Worker, random_segment_v2};
use crate::steer::steer;

/// Radius within which a waypoint's place counts as reached.
pub const RADIUS: f32 = 150.0;

/// A route state as a goal.
#[derive(Clone)]
pub struct Waypoint {
    /// Index of the route state it stands for (0 = the root, never a goal).
    pub index: usize,
    pub zone: (u16, u8),
    pub pos: [f32; 3],
    pub stars: i16,
    pub keys: u8,
    pub red_coins: i16,
    pub holding: bool,
    /// Objects whose health dropped by this state: their health must be at
    /// most this much (boss hits). Objects that disappeared by it (coins
    /// taken, enemies beaten, boxes broken) must be gone.
    pub health: Vec<(ObjId, i32)>,
    pub gone: Vec<ObjId>,
    /// The open dialog/menu (see [`modal_code`]), if this state is in one.
    pub modal: Option<i16>,
    /// Stars, keys or zone changed here, it's a dialog/menu, or it's the
    /// goal: a route can't skip it.
    pub essential: bool,
    /// The button the original route pressed most in the second before it
    /// (a hint for steering: B to grab, Z to pound).
    pub button: u16,
    /// For a zone change: where the original route last was in the old zone
    /// (the door or warp from outside). Steering aims there until the zone
    /// changes; `pos` is in the new zone's coordinates.
    pub approach: Option<([f32; 3], (u16, u8))>,
}

/// Which dialog or menu `s` is in: the dialog id, -2 the save menu, -3 the
/// act select screen.
pub fn modal_code(s: &St) -> Option<i16> {
    if s.in_dialog() {
        Some(s.dialog)
    } else if s.in_save_menu() {
        Some(-2)
    } else if s.act_select {
        Some(-3)
    } else {
        None
    }
}

impl Waypoint {
    pub fn met_by(&self, s: &St) -> bool {
        if let Some(m) = self.modal {
            return modal_code(s) == Some(m) && s.stars >= self.stars && s.keys >= self.keys;
        }
        if s.zone() != self.zone
            || s.stars < self.stars
            || s.keys < self.keys
            || s.red_coins < self.red_coins
            || (self.holding && !s.holding)
            || dist(s.pos, self.pos) > RADIUS
        {
            return false;
        }
        if self.health.is_empty() && self.gone.is_empty() {
            return true;
        }
        if s.objects.is_none() {
            return false;
        }
        self.gone.iter().all(|id| s.find(*id).is_none())
            && self.health.iter().all(|(id, hp)| s.find(*id).is_none_or(|o| o.health <= *hp))
    }

    fn uses_objects(&self) -> bool {
        !self.health.is_empty() || !self.gone.is_empty()
    }
}

impl Objective for Waypoint {
    fn kind(&self) -> &'static str {
        "Route::Waypoint"
    }
    fn key(&self) -> String {
        format!("Waypoint {}", self.index)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        1.0
    }
    fn base_criteria(&self, _s: &St) -> bool {
        true
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(self.pos) as f64 / 400.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| self.met_by(s))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let (at, d) = traj
            .iter()
            .enumerate()
            .filter(|(_, s)| s.zone() == self.zone)
            .map(|(i, s)| (i, dist(s.pos, self.pos)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, f32::MAX));
        Progress { score: -(d as f64), at }
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 6, horizon: 300 }
    }
    fn needs_objects(&self) -> bool {
        self.uses_objects()
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        match self.approach {
            Some((p, zone)) if s.zone() == zone => Some(p),
            _ => Some(self.pos),
        }
    }
    fn steer_button(&self, s: &St) -> u16 {
        if self.button != 0 && s.dist(self.pos) < 300.0 { self.button } else { 0 }
    }
}

/// Inputs for frames where Mario can't act: per dialog/menu, the inputs the
/// original route used in it (from its first frame); otherwise A every 4.
/// Every stretch of frames where Mario can't act is an episode, keyed by the
/// dialog/menu open at its start (see [`modal_code`]) or else by Mario's
/// action then (a door, a cutscene, walking up to an NPC); a new episode
/// starts when the dialog/menu changes. The original route's episodes are
/// kept in order: a replay's next episode takes the next entry with its key
/// (looking a few ahead, in case one was skipped), else presses A every 4.
#[derive(Clone, Default)]
pub struct Policy {
    pub episodes: Vec<(EpisodeKey, Vec<GamePad>)>,
}

/// How far ahead in the episode list a replay looks for its episode's key.
const EPISODE_LOOKAHEAD: usize = 6;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum EpisodeKey {
    Modal(i16),
    Action(u32),
}

pub fn episode_key(s: &St) -> EpisodeKey {
    match modal_code(s) {
        Some(m) => EpisodeKey::Modal(m),
        None => EpisodeKey::Action(s.action),
    }
}

impl Policy {
    pub fn dialogs_and_menus(&self) -> usize {
        self.episodes.iter().filter(|e| matches!(e.0, EpisodeKey::Modal(_))).count()
    }

    /// The entry a new episode with `key` takes, from list position `next`.
    fn find(&self, key: EpisodeKey, next: usize) -> Option<usize> {
        (next..(next + EPISODE_LOOKAHEAD).min(self.episodes.len())).find(|&j| self.episodes[j].0 == key)
    }
}

/// Plays episodes from the policy: the current entry and the next list position.
#[derive(Clone, Copy)]
struct EpisodeCursor {
    entry: Option<usize>,
    next: usize,
}

impl EpisodeCursor {
    fn pad(&mut self, policy: &Policy, key: EpisodeKey, offset: usize) -> GamePad {
        if offset == 0 {
            self.entry = policy.find(key, self.next);
            if let Some(j) = self.entry {
                self.next = j + 1;
            }
        }
        match self.entry.and_then(|j| policy.episodes[j].1.get(offset)) {
            Some(p) => *p,
            None => GamePad::new(if offset % 4 == 0 { A_BUTTON } else { 0 }, 0, 0),
        }
    }
}

/// Tracks the current episode while playing: returns its key and the offset
/// for this frame, or None when Mario can act.
#[derive(Default)]
struct EpisodeTracker {
    key: Option<EpisodeKey>,
    offset: usize,
}

impl EpisodeTracker {
    fn step(&mut self, before: &St) -> Option<(EpisodeKey, usize)> {
        if before.controllable() {
            self.key = None;
            return None;
        }
        let modal = modal_code(before);
        let restart = match self.key {
            None => true,
            // A dialog/menu opening or changing inside a stretch starts a new episode.
            Some(EpisodeKey::Modal(m)) => modal != Some(m),
            Some(EpisodeKey::Action(_)) => modal.is_some(),
        };
        if restart {
            self.key = Some(episode_key(before));
            self.offset = 0;
        }
        self.offset += 1;
        Some((self.key.unwrap(), self.offset - 1))
    }
}

/// The route being optimised, as loaded from a run.
pub struct Loaded {
    pub boot: Vec<GamePad>,
    pub root_save: SaveState,
    pub root: St,
    /// Route states: 0 the root, the last the goal.
    pub states: Vec<St>,
    pub waypoints: Vec<Waypoint>,
    /// The original route's stream and policy.
    pub stream: Vec<GamePad>,
    pub policy: Policy,
    pub original_frame: u32,
    pub node_ids: Vec<u32>,
    /// Debugging: the original route's inputs and per-frame (pos, action,
    /// controllable, modal) after boot.
    pub original_played: Vec<GamePad>,
    pub original_trace: Vec<([f32; 3], u32, bool, Option<i16>)>,
}

/// tree.bin: per record (parent, inputs); record 0 the boot, record i+1 node i.
pub fn read_tree_bin(path: &str) -> Result<Vec<(i32, Vec<GamePad>)>> {
    let data = fs::read(path)?;
    let u32_at = |p: usize| u32::from_le_bytes(data[p..p + 4].try_into().unwrap());
    let count = u32_at(0) as usize;
    let mut pos = 4;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let parent = u32_at(pos) as i32;
        let n = u32_at(pos + 4) as usize;
        pos += 8;
        let inputs = (0..n)
            .map(|k| {
                let b = &data[pos + 4 * k..pos + 4 * k + 4];
                GamePad::new(u16::from_be_bytes([b[0], b[1]]), b[2] as i8, b[3] as i8)
            })
            .collect();
        pos += 4 * n;
        out.push((parent, inputs));
    }
    Ok(out)
}

/// Loads the route root -> `node` (default: the earliest state with more
/// stars than the root) from `run_dir`, replaying it to read every state.
pub fn load(game_gen: &SM64GameGenerator, run_dir: &str, node: Option<u32>) -> Result<Loaded> {
    let records = read_tree_bin(&format!("{run_dir}/tree.bin"))?;
    ensure!(records.len() >= 2, "{run_dir}/tree.bin is empty");
    let boot = records[0].1.clone();
    // Node i's frame and stars, to pick the default goal.
    let csv = fs::read_to_string(format!("{run_dir}/nodes.csv"))?;
    let rows: Vec<Vec<&str>> = csv.lines().skip(1).map(|l| l.split(',').collect()).collect();
    let target = match node {
        Some(n) => n,
        None => {
            let root_stars: i16 = rows[0][8].parse()?;
            rows.iter()
                .filter(|r| r[8].parse::<i16>().unwrap_or(0) > root_stars)
                .min_by_key(|r| r[11].parse::<u32>().unwrap_or(u32::MAX))
                .map(|r| r[0].parse::<u32>().unwrap())
                .ok_or_else(|| anyhow::anyhow!("{run_dir} has no state with a star"))?
        }
    };
    let mut ids = vec![target];
    while let Some(&n) = ids.last() {
        let p = records[n as usize + 1].0;
        if p < 0 {
            break;
        }
        ids.push(p as u32);
    }
    ids.reverse();
    ensure!(ids[0] == 0, "route doesn't start at the root");

    let mut game = game_gen.create_game()?;
    for p in &boot {
        game.step_game(*p)?;
    }
    let root_save = game.save_state();
    let root = read_state(&game, boot.len() as u32, true);
    // Replay frame by frame: states at route nodes, the stream (inputs on
    // controllable frames) and the dialog/menu policy.
    let mut states = vec![root.clone()];
    let mut stream = Vec::new();
    let mut policy = Policy::default();
    let mut frame = boot.len() as u32;
    let mut before = root.clone();
    let mut tracker = EpisodeTracker::default();
    let mut episode: Option<(EpisodeKey, Vec<GamePad>)> = None;
    let mut button_hist: Vec<u16> = Vec::new();
    let mut buttons = Vec::new();
    // Per segment: the last position while still in the zone it started in.
    let mut approaches: Vec<([f32; 3], (u16, u8))> = Vec::new();
    let mut original_played = Vec::new();
    let mut original_trace = Vec::new();
    for &id in &ids[1..] {
        let seg_zone = before.zone();
        let mut last_in_zone = before.pos;
        for p in &records[id as usize + 1].1 {
            if before.zone() == seg_zone {
                last_in_zone = before.pos;
            }
            match tracker.step(&before) {
                Some((key, 0)) => {
                    if let Some(e) = episode.take() {
                        policy.episodes.push(e);
                    }
                    episode = Some((key, vec![*p]));
                }
                Some(_) => episode.as_mut().unwrap().1.push(*p),
                None => {
                    if let Some(e) = episode.take() {
                        policy.episodes.push(e);
                    }
                    stream.push(*p);
                }
            }
            button_hist.push(p.button);
            original_played.push(*p);
            original_trace.push((before.pos, before.action, before.controllable(), modal_code(&before)));
            game.step_game(*p)?;
            frame += 1;
            before = read_state(&game, frame, false);
        }
        // The button pressed most in the last 30 frames (A, B or Z).
        let recent = &button_hist[button_hist.len().saturating_sub(30)..];
        let count = |b: u16| recent.iter().filter(|x| *x & b != 0).count();
        let best = [obrrt_core::buttons::B_BUTTON, obrrt_core::buttons::Z_TRIG]
            .into_iter()
            .max_by_key(|b| count(*b))
            .filter(|b| count(*b) >= 3)
            .unwrap_or(0);
        buttons.push(best);
        approaches.push((last_in_zone, seg_zone));
        states.push(read_state(&game, frame, true));
        before = states.last().unwrap().clone();
    }
    if let Some(e) = episode.take() {
        policy.episodes.push(e);
    }
    let original_frame = frame;

    let mut waypoints = Vec::new();
    for i in 1..states.len() {
        let (prev, s) = (&states[i - 1], &states[i]);
        let same_zone = prev.zone() == s.zone();
        let mut health = Vec::new();
        let mut gone = Vec::new();
        if same_zone && prev.objects.is_some() && s.objects.is_some() {
            for o in prev.objects() {
                if o.interact_type == 0 {
                    continue;
                }
                match s.find(o.id) {
                    None => gone.push(o.id),
                    Some(n) if n.health < o.health && n.health >= 0 => health.push((o.id, n.health)),
                    _ => {}
                }
            }
        }
        // Earlier waypoints' requirements carry on (a coin once taken stays
        // taken), so each lists only what changed at it.
        let modal = modal_code(s);
        let essential = i == states.len() - 1
            || modal.is_some()
            || !same_zone
            || s.progress() != prev.progress();
        waypoints.push(Waypoint {
            index: i,
            zone: s.zone(),
            pos: s.pos,
            stars: s.stars,
            keys: s.keys,
            red_coins: s.red_coins,
            holding: s.holding,
            health,
            gone,
            modal,
            essential,
            button: buttons[i - 1],
            approach: (!same_zone).then(|| approaches[i - 1]),
        });
    }
    Ok(Loaded { boot, root_save, root, states, waypoints, stream, policy, original_frame, node_ids: ids, original_played, original_trace })
}

/// GA settings for one trial.
#[derive(Clone, Copy, Debug)]
pub struct GaSpec {
    pub population: usize,
    pub generations: usize,
    /// Frames allowed; default: the best route's frames for the trial's span
    /// × 1.2 + 300.
    pub horizon: Option<usize>,
}

impl Default for GaSpec {
    fn default() -> GaSpec {
        GaSpec { population: 16, generations: 8, horizon: None }
    }
}

/// A custom mutation a method can add (applied with probability 0.3 in place
/// of a built-in one).
pub type Mutation = fn(&mut Vec<GamePad>, &mut SmallRng);

/// A place to start a trial that isn't on the best route (a repair starts
/// where a faster trial ended): its save, state, frame after boot, the full
/// stream from the root to it, and the episode list position.
#[derive(Clone)]
pub struct StartPoint {
    pub save: SaveState,
    pub st: St,
    pub frame: usize,
    pub stream_prefix: Vec<GamePad>,
    pub episode: usize,
}

/// One unit of work: re-solve `stages` (waypoint indices, increasing, all
/// after `from`) starting from the best route's state at waypoint `from`
/// (0 = the root), seeded with the best route's stream over that span (or
/// `seed`). Build with `..Default::default()` for the optional fields.
#[derive(Clone)]
pub struct Trial {
    pub from: usize,
    pub stages: Vec<usize>,
    pub spec: GaSpec,
    pub seed: Option<Vec<GamePad>>,
    pub mutation: Option<Mutation>,
    /// Free text for the method's own bookkeeping (shown in the log).
    pub tag: String,
    /// Mutations only touch seed positions in this range (start, end).
    pub focus: Option<(usize, usize)>,
    /// Probability the custom mutation replaces a built-in one (default 0.3).
    pub custom_prob: f64,
    /// A whole candidate stream from the root: no GA, the bench just
    /// verifies it.
    pub whole: Option<Vec<GamePad>>,
    /// Start here instead of at the best route's state `from` (repairs).
    pub start: Option<StartPoint>,
}

impl Default for Trial {
    fn default() -> Trial {
        Trial {
            from: 0,
            stages: Vec::new(),
            spec: GaSpec::default(),
            seed: None,
            mutation: None,
            tag: String::new(),
            focus: None,
            custom_prob: 0.3,
            whole: None,
            start: None,
        }
    }
}

/// What came of a trial.
pub struct Outcome {
    /// The trial's id (in the order proposed, from 0).
    pub id: u64,
    pub trial: Trial,
    /// The trial's best stream up to its last met waypoint.
    pub stream: Option<Vec<GamePad>>,
    /// Why it didn't improve the route: "not all met", "not faster",
    /// "verify failed" (the spliced route missed an essential waypoint) or
    /// "goal not sooner"; None if it improved. A rejected faster trial is
    /// followed by an automatic repair (see Bench::run).
    pub reject: Option<&'static str>,
    /// Waypoints met (of `trial.stages`).
    pub met: usize,
    /// Frames to the last stage from the trial's start, if all were met, and
    /// the best route's frames over the same span.
    pub span_frames: Option<usize>,
    pub best_span_frames: Option<usize>,
    /// Whether it improved the whole route, and the route's goal frame after.
    pub improved: bool,
    pub goal_frame: u32,
    pub frames_used: u64,
}

/// The best route so far.
pub struct Best {
    pub stream: Vec<GamePad>,
    /// Inputs actually played after boot (stream plus automatic frames).
    pub played: Vec<GamePad>,
    /// Per route state (0 = the root): when the route reached its waypoint
    /// (frame after boot, stream position), None if it skipped it; and the
    /// save state and state there.
    pub arrival: Vec<Option<(usize, usize)>>,
    pub saves: Vec<Option<SaveState>>,
    pub states: Vec<Option<St>>,
    /// The episode list position at each arrival.
    pub episode: Vec<usize>,
    pub goal_frame: u32,
    /// Per frame after boot, up to the goal: zone, position, whether Mario
    /// could act (before the frame), and the stream position after it.
    pub trace: Vec<TracePoint>,
}

#[derive(Clone, Copy, Debug)]
pub struct TracePoint {
    pub zone: (u16, u8),
    pub pos: [f32; 3],
    pub controllable: bool,
    pub stream_pos: usize,
}

/// What methods see.
pub struct Ctx<'a> {
    pub waypoints: &'a [Waypoint],
    pub best: &'a Best,
    pub original_frame: u32,
    pub elapsed: f64,
    /// Trials run so far.
    pub trials: u64,
}

impl Ctx<'_> {
    /// Route states the best route reached (usable as `from`), in order.
    pub fn reached(&self) -> Vec<usize> {
        (0..self.best.arrival.len()).filter(|&i| self.best.arrival[i].is_some()).collect()
    }
    /// The goal (the last route state).
    pub fn goal(&self) -> usize {
        self.waypoints.len()
    }
    pub fn essential(&self, i: usize) -> bool {
        i == 0 || self.waypoints[i - 1].essential
    }
}

/// A route optimisation method: proposes trials, hears how they went.
pub trait Method: Send {
    fn name(&self) -> &'static str;
    fn propose(&mut self, ctx: &Ctx, rng: &mut SmallRng) -> Trial;
    fn feedback(&mut self, _outcome: &Outcome, _ctx: &Ctx) {}
}

struct RunResult {
    /// The episode list position after each frame (with keep_traj).
    episode_at: Vec<usize>,
    /// Stream consumed and inputs played.
    stream: Vec<GamePad>,
    played: Vec<GamePad>,
    /// Per stage met: (frame index after start, stream position).
    ends: Vec<(usize, usize)>,
    traj: Vec<St>,
    progress: f64,
}

impl RunResult {
    /// More waypoints met first; then sooner at the last one met (for a
    /// complete trial, the route's time); then progress on the next unmet
    /// one. (Pure steps order, sooner at waypoint 1 above all, prefers a
    /// trial one frame sooner at the first waypoint that then fails the rest
    /// over a complete one: wrong for whole routes.)
    fn better(&self, o: &RunResult) -> std::cmp::Ordering {
        self.ends
            .len()
            .cmp(&o.ends.len())
            .then_with(|| o.ends.last().map(|e| e.0).cmp(&self.ends.last().map(|e| e.0)))
            .then(self.progress.total_cmp(&o.progress))
    }
}

/// Plays `stream` from `start` (whose save is loaded in `base`), automatic
/// frames per `policy`, steering once the stream runs out, until all
/// `stages` are met or `horizon` frames.
fn rollout(
    w: &mut Worker,
    start: &St,
    stream: &[GamePad],
    stages: &[&Waypoint],
    policy: &Policy,
    episode_start: usize,
    horizon: usize,
    keep_traj: bool,
) -> Result<RunResult> {
    w.tmp.sync_from(&w.base)?;
    let objects = stages.iter().any(|s| s.uses_objects());
    let mut played = Vec::with_capacity(horizon);
    let mut ends = Vec::new();
    let mut traj = Vec::new();
    let mut cur = start.clone();
    let mut pos = 0;
    let mut taken: Vec<GamePad> = Vec::with_capacity(stream.len());
    let mut tracker = EpisodeTracker::default();
    let mut cursor = EpisodeCursor { entry: None, next: episode_start };
    let mut episode_at = Vec::new();
    let mut best_d = f32::MAX;
    for f in 0..horizon {
        if ends.len() == stages.len() {
            break;
        }
        // Controllable frames take the stream, then the steering controller
        // once it runs out; either way the input is part of the trial's
        // stream (`taken`), so a splice replays exactly what was played.
        let pad = if let Some((key, offset)) = tracker.step(&cur) {
            cursor.pad(policy, key, offset)
        } else {
            let p = if pos < stream.len() { stream[pos] } else { steer(stages[ends.len()], &cur, f, 4) };
            pos += 1;
            taken.push(p);
            p
        };
        w.tmp.step_game(pad)?;
        played.push(pad);
        cur = read_state(&w.tmp, start.frame + f as u32 + 1, objects);
        if !cur.alive() || cur.lives < start.lives {
            break;
        }
        while ends.len() < stages.len() && stages[ends.len()].met_by(&cur) {
            ends.push((f + 1, pos));
        }
        if let Some(st) = stages.get(ends.len()) {
            if cur.zone() == st.zone {
                best_d = best_d.min(dist(cur.pos, st.pos));
            }
        }
        if keep_traj {
            traj.push(cur.clone());
            episode_at.push(cursor.next);
        }
    }
    Ok(RunResult { episode_at, stream: taken, played, ends, traj, progress: -(best_d as f64) })
}

fn mutate(g: &mut Vec<GamePad>, rng: &mut SmallRng, custom: Option<Mutation>, custom_prob: f64, focus: Option<(usize, usize)>) {
    if g.is_empty() {
        let mut seg = Vec::new();
        random_segment_v2(rng, &mut seg, 10);
        g.extend(seg);
        return;
    }
    if let Some(m) = custom.filter(|_| rng.random_bool(custom_prob.clamp(0.0, 1.0))) {
        return m(g, rng);
    }
    let len = g.len();
    let a = match focus {
        Some((lo, hi)) if lo < len => rng.random_range(lo..hi.clamp(lo + 1, len)),
        _ => rng.random_range(0..len),
    };
    match rng.random_range(0..6) {
        // Replace a span with a random held segment.
        0 => {
            let b = (a + rng.random_range(3..40)).min(len);
            let mut seg = Vec::new();
            random_segment_v2(rng, &mut seg, b - a);
            g.splice(a..b, seg);
        }
        // Turn the stick over a span.
        1 => {
            let b = (a + rng.random_range(5..60)).min(len);
            let turn = rng.random_range(-40.0f32..40.0).to_radians();
            for p in &mut g[a..b] {
                let ang = (p.stick_y as f32).atan2(p.stick_x as f32) + turn;
                let mag = ((p.stick_x as f32).powi(2) + (p.stick_y as f32).powi(2)).sqrt().min(127.0);
                p.stick_x = (mag * ang.cos()) as i8;
                p.stick_y = (mag * ang.sin()) as i8;
            }
        }
        // Toggle A over a few frames.
        2 => {
            for p in &mut g[a..(a + rng.random_range(1..12)).min(len)] {
                p.button ^= A_BUTTON;
            }
        }
        // Press A, B or Z for a few frames.
        3 => {
            use obrrt_core::buttons::{B_BUTTON, Z_TRIG};
            let b = [A_BUTTON, A_BUTTON, Z_TRIG, Z_TRIG, B_BUTTON][rng.random_range(0..5)];
            for p in &mut g[a..(a + rng.random_range(1..5)).min(len)] {
                p.button |= b;
            }
        }
        // Delete a few frames (the route gets shorter).
        4 => {
            let b = (a + rng.random_range(1..15)).min(len);
            g.drain(a..b);
        }
        // Duplicate a few frames.
        _ => {
            let b = (a + rng.random_range(1..10)).min(len);
            let copy: Vec<GamePad> = g[a..b].to_vec();
            g.splice(a..a, copy);
        }
    }
}

/// The bench: holds the route, the best version so far, and the workers.
pub struct Bench {
    pub loaded: Loaded,
    pub best: Best,
    workers: Vec<Mutex<Worker>>,
    out_dir: String,
    started: Instant,
    trials: u64,
    frames: u64,
    curve: fs::File,
    log: fs::File,
    /// Fastest verified arrival (frames after boot) at route states, from
    /// prefix records.
    prefixes: std::collections::HashMap<usize, usize>,
}

impl Bench {
    pub fn new(game_gen: &SM64GameGenerator, loaded: Loaded, out_dir: &str) -> Result<Bench> {
        fs::create_dir_all(out_dir)?;
        let threads = rayon::current_num_threads();
        let workers = (0..threads)
            .map(|_| Ok(Mutex::new(Worker { base: game_gen.create_game()?, tmp: game_gen.create_game()? })))
            .collect::<Result<Vec<_>>>()?;
        let mut curve = fs::File::create(format!("{out_dir}/curve.csv"))?;
        writeln!(curve, "seconds,trials,frames,goal_frame")?;
        let log = fs::File::create(format!("{out_dir}/trials.jsonl"))?;
        let mut bench = Bench {
            best: Best {
                stream: Vec::new(),
                played: Vec::new(),
                arrival: Vec::new(),
                saves: Vec::new(),
                states: Vec::new(),
                episode: Vec::new(),
                goal_frame: u32::MAX,
                trace: Vec::new(),
            },
            loaded,
            workers,
            out_dir: out_dir.to_string(),
            started: Instant::now(),
            trials: 0,
            frames: 0,
            curve,
            log,
            prefixes: std::collections::HashMap::new(),
        };
        // The original route, through the bench's own replay (automatic
        // frames instead of the original's): the baseline.
        let stream = bench.loaded.stream.clone();
        match bench.verify(&stream)? {
            Some(best) => bench.best = best,
            None => bail!("the original route doesn't replay to its goal with the bench's automatic frames"),
        }
        writeln!(bench.curve, "0,0,0,{}", bench.best.goal_frame)?;
        Ok(bench)
    }

    /// Replays `stream` from the root with all waypoints as optional
    /// checkpoints: the route if it reaches the goal (with every waypoint's
    /// arrival, save and state), else None.
    fn verify(&mut self, stream: &[GamePad]) -> Result<Option<Best>> {
        let ld = &self.loaded;
        let w = self.workers[0].get_mut().unwrap();
        w.base.load_state(&ld.root_save)?;
        let goal = ld.waypoints.last().unwrap();
        let horizon = stream.len() * 3 + 3000;
        // Only the essential waypoints are required in order; the others are
        // recorded where first met after the previous arrival.
        let essential: Vec<&Waypoint> = ld.waypoints.iter().filter(|w| w.essential).collect();
        let r = rollout(w, &ld.root, stream, &essential, &ld.policy, 0, horizon, true)?;
        if r.ends.len() < essential.len() {
            if std::env::var("ROUTE_DEBUG").is_ok() {
                let last = r.traj.last();
                eprintln!(
                    "verify: met {}/{} essential; next {:?} at {:?} stars {} modal {:?}; end: zone {:?} pos {:?} action {:#x} modal {:?} controllable {} frames {} stream {}/{}",
                    r.ends.len(),
                    essential.len(),
                    essential[r.ends.len()].index,
                    essential[r.ends.len()].pos,
                    essential[r.ends.len()].stars,
                    essential[r.ends.len()].modal,
                    last.map(|s| s.zone()),
                    last.map(|s| s.pos),
                    last.map_or(0, |s| s.action),
                    last.and_then(modal_code),
                    last.is_some_and(|s| s.controllable()),
                    r.played.len(),
                    r.stream.len(),
                    stream.len()
                );
                for (k, e) in r.ends.iter().enumerate() {
                    eprintln!("  essential {} (state {}) met at frame {}", k, essential[k].index, e.0);
                }
                // First frame where the replay's input differs from the original's.
                let mut prev_pos = ld.root.pos;
                for (f, (p, o)) in r.played.iter().zip(&ld.original_played).enumerate() {
                    if p != o {
                        let (opos, oact, octl, omodal) = ld.original_trace[f];
                        eprintln!(
                            "  first input difference at frame {f}: bench {:?} original {:?}; original before: pos {:?} action {:#x} controllable {} modal {:?}; bench pos {:?}",
                            p, o, opos, oact, octl, omodal, prev_pos
                        );
                        break;
                    }
                    prev_pos = r.traj[f].pos;
                }
            }
            return Ok(None);
        }
        let n = ld.waypoints.len();
        let mut arrival = vec![None; n + 1];
        arrival[0] = Some((0usize, 0usize));
        let mut episode = vec![0usize; n + 1];
        // Stream position per trajectory index: count controllable frames.
        let mut pos_at = Vec::with_capacity(r.traj.len());
        let mut trace = Vec::with_capacity(r.traj.len());
        let mut pos = 0usize;
        let mut prev = ld.root.clone();
        for s in &r.traj {
            let ctl = prev.controllable();
            if ctl {
                pos += 1;
            }
            pos_at.push(pos.min(r.stream.len()));
            trace.push(TracePoint { zone: s.zone(), pos: s.pos, controllable: ctl, stream_pos: pos.min(r.stream.len()) });
            prev = s.clone();
        }
        let mut from = 0usize;
        for (k, wp) in ld.waypoints.iter().enumerate() {
            if let Some(i) = (from..r.traj.len()).find(|&i| wp.met_by(&r.traj[i])) {
                arrival[k + 1] = Some((i + 1, pos_at[i]));
                episode[k + 1] = r.episode_at[i];
                from = i;
            } else if wp.essential {
                return Ok(None);
            }
        }
        let goal_at = arrival[n].unwrap().0;
        // Saves at each arrival: replay the played inputs once more.
        let mut saves: Vec<Option<SaveState>> = vec![None; n + 1];
        let mut states: Vec<Option<St>> = vec![None; n + 1];
        saves[0] = Some(ld.root_save.clone());
        states[0] = Some(ld.root.clone());
        w.tmp.load_state(&ld.root_save)?;
        let mut want: Vec<(usize, usize)> = arrival.iter().enumerate().skip(1).filter_map(|(k, a)| a.map(|a| (a.0, k))).collect();
        want.sort();
        let mut wi = 0;
        for (f, p) in r.played[..goal_at].iter().enumerate() {
            w.tmp.step_game(*p)?;
            while wi < want.len() && want[wi].0 == f + 1 {
                let k = want[wi].1;
                saves[k] = Some(w.tmp.save_state());
                states[k] = Some(read_state(&w.tmp, ld.root.frame + f as u32 + 1, true));
                wi += 1;
            }
        }
        let _ = goal;
        Ok(Some(Best {
            stream: r.stream.clone(),
            played: r.played[..goal_at].to_vec(),
            arrival,
            saves,
            states,
            episode,
            goal_frame: ld.root.frame + goal_at as u32,
            trace: trace[..goal_at.min(trace.len())].to_vec(),
        }))
    }

    /// Runs one trial's GA on a worker.
    fn run_trial(&self, w: &mut Worker, t: &Trial, rng: &mut SmallRng) -> Result<(RunResult, u64)> {
        let ld = &self.loaded;
        let (save, start, from_frame, from_pos, episode) = match &t.start {
            Some(sp) => (&sp.save, &sp.st, sp.frame, usize::MAX, sp.episode),
            None => {
                let (Some(save), Some(start)) = (&self.best.saves[t.from], &self.best.states[t.from]) else {
                    bail!("trial starts at route state {} the best route didn't reach", t.from);
                };
                let (f, p) = self.best.arrival[t.from].unwrap();
                (save, start, f, p, self.best.episode[t.from])
            }
        };
        let last = *t.stages.last().unwrap();
        // The best route's span: to its last stage if it reached it, else to
        // the next state it did reach after it.
        let to = (last..self.best.arrival.len()).find_map(|k| self.best.arrival[k]).unwrap_or((self.best.played.len(), self.best.stream.len()));
        let seed = t.seed.clone().unwrap_or_else(|| {
            let p = from_pos.min(self.best.stream.len());
            self.best.stream[p..to.1.max(p)].to_vec()
        });
        let horizon = t.spec.horizon.unwrap_or(((to.0.saturating_sub(from_frame)) as f64 * 1.2) as usize + 300);
        let stages: Vec<&Waypoint> = t.stages.iter().map(|&i| &ld.waypoints[i - 1]).collect();
        w.base.load_state(save)?;
        let mut frames = 0u64;
        let mut pop: Vec<(Vec<GamePad>, RunResult)> = Vec::with_capacity(t.spec.population);
        for i in 0..t.spec.population.max(1) {
            let mut g = seed.clone();
            if i > 0 {
                for _ in 0..rng.random_range(1..4) {
                    mutate(&mut g, rng, t.mutation, t.custom_prob, t.focus);
                }
            }
            let r = rollout(w, start, &g, &stages, &ld.policy, episode, horizon, false)?;
            if i == 0 && std::env::var("ROUTE_DEBUG").is_ok() {
                let want: Vec<String> = t.stages.iter().map(|&k| format!("{}:{:?}", k, self.best.arrival[k].map(|a| a.0 - from_frame))).collect();
                eprintln!("seed: met {}/{} ends {:?}\n  best arrivals {:?}", r.ends.len(), stages.len(), r.ends.iter().map(|e| e.0).collect::<Vec<_>>(), want);
            }
            frames += r.played.len() as u64;
            pop.push((g, r));
        }
        for _ in 1..t.spec.generations {
            pop.sort_by(|a, b| b.1.better(&a.1));
            let elite = (pop.len() / 4).max(1);
            pop.truncate(elite);
            while pop.len() < t.spec.population.max(1) {
                let mut g = pop[rng.random_range(0..elite)].0.clone();
                mutate(&mut g, rng, t.mutation, t.custom_prob, t.focus);
                let r = rollout(w, start, &g, &stages, &ld.policy, episode, horizon, false)?;
                frames += r.played.len() as u64;
                pop.push((g, r));
            }
        }
        pop.sort_by(|a, b| b.1.better(&a.1));
        Ok((pop.swap_remove(0).1, frames))
    }

    /// Runs `method` for `minutes`, all workers in parallel; writes
    /// best.m64, curve.csv, trials.jsonl and summary.json.
    ///
    /// Each round: queued repairs (up to half the workers), then the
    /// method's proposals; all run in parallel against the best route as it
    /// was at the start of the round, and are spliced into that route (so a
    /// trial is never spliced into a route that changed under it). A trial
    /// faster over its span whose spliced route fails or isn't sooner is
    /// followed by a repair: from where that trial ended, re-solve up to the
    /// next essential waypoint, seeded with the old stream.
    pub fn run(&mut self, method: &mut dyn Method, minutes: f64, seed: u64) -> Result<()> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let deadline = minutes * 60.0;
        let mut improvements = 0u64;
        let mut next_id = 0u64;
        // (trial, the original's tag) waiting to run.
        let mut repairs: Vec<(Trial, String)> = Vec::new();
        let (mut repaired, mut repairs_tried) = (0u64, 0u64);
        while self.started.elapsed().as_secs_f64() < deadline {
            let snap_stream = self.best.stream.clone();
            let snap_arrival = self.best.arrival.clone();
            let mut trials: Vec<(Trial, Option<String>)> = Vec::new();
            let w = self.workers.len();
            while trials.len() < w / 2 && !repairs.is_empty() {
                let (t, tag) = repairs.remove(0);
                trials.push((t, Some(tag)));
            }
            {
                let ctx = Ctx {
                    waypoints: &self.loaded.waypoints,
                    best: &self.best,
                    original_frame: self.loaded.original_frame,
                    elapsed: self.started.elapsed().as_secs_f64(),
                    trials: self.trials,
                };
                while trials.len() < w {
                    trials.push((method.propose(&ctx, &mut rng), None));
                }
            }
            let seeds: Vec<u64> = trials.iter().map(|_| rng.random()).collect();
            let results: Vec<Result<Option<(RunResult, u64)>>> = trials
                .par_iter()
                .zip(seeds.par_iter())
                .map(|((t, _), s)| {
                    if t.whole.is_some() {
                        return Ok(None);
                    }
                    let mut w = loop {
                        if let Some(w) = self.workers.iter().find_map(|w| w.try_lock().ok()) {
                            break w;
                        }
                        std::thread::yield_now();
                    };
                    self.run_trial(&mut w, t, &mut SmallRng::seed_from_u64(*s)).map(Some)
                })
                .collect();
            for ((t, repair_of), r) in trials.into_iter().zip(results) {
                let r = r?;
                let id = next_id;
                next_id += 1;
                self.trials += 1;
                let frames = r.as_ref().map_or(0, |r| r.1);
                self.frames += frames;
                // The span in the round's snapshot: from (or the repair's
                // start) to the last stage (or the next state reached after it).
                let last = t.stages.last().copied().unwrap_or(0);
                let to = (last..snap_arrival.len()).find_map(|k| snap_arrival[k]);
                let (prefix, from_frame): (Vec<GamePad>, usize) = match &t.start {
                    Some(sp) => (sp.stream_prefix.clone(), sp.frame),
                    None => {
                        let (f, p) = snap_arrival[t.from].unwrap_or((0, 0));
                        (snap_stream[..p.min(snap_stream.len())].to_vec(), f)
                    }
                };
                let best_span = to.map(|a| a.0.saturating_sub(from_frame));
                let (met, span, trial_stream, candidate) = match (&t.whole, &r) {
                    (Some(whole), _) => (0, None, None, Some(whole.clone())),
                    (None, Some((r, _))) => {
                        let all = r.ends.len() == t.stages.len() && !t.stages.is_empty();
                        let span = all.then(|| r.ends.last().unwrap().0);
                        let cut = r.ends.last().map_or(0, |e| e.1.min(r.stream.len()));
                        let ts = r.stream[..cut].to_vec();
                        // Repairs are spliced whenever they meet their stages;
                        // others only when faster over the span.
                        let faster = span.is_some_and(|sp| repair_of.is_some() || best_span.is_none_or(|b| sp < b));
                        let cand = faster.then(|| {
                            let mut c = prefix.clone();
                            c.extend_from_slice(&ts);
                            c.extend_from_slice(&snap_stream[to.map_or(snap_stream.len(), |a| a.1).min(snap_stream.len())..]);
                            c
                        });
                        (r.ends.len(), span, Some(ts), cand)
                    }
                    (None, None) => (0, None, None, None),
                };
                let mut reject: Option<&'static str> = None;
                if let Some(c) = &candidate {
                    match self.verify(c)? {
                        Some(nb) if nb.goal_frame < self.best.goal_frame => {
                            self.best = nb;
                            improvements += 1;
                            if repair_of.is_some() {
                                repaired += 1;
                            }
                            writeln!(self.curve, "{:.1},{},{},{}", self.started.elapsed().as_secs_f64(), self.trials, self.frames, self.best.goal_frame)?;
                            println!(
                                "{:6.1}s  goal frame {} ({} saved) via {} [{}]",
                                self.started.elapsed().as_secs_f64(),
                                self.best.goal_frame,
                                self.loaded.original_frame as i64 - self.best.goal_frame as i64,
                                method.name(),
                                repair_of.as_ref().map_or(t.tag.clone(), |o| format!("repair of {o}"))
                            );
                        }
                        Some(_) => reject = Some("goal not sooner"),
                        None => reject = Some("verify failed"),
                    }
                    // Prefix record: even if the whole route didn't work out,
                    // the spliced route may reach the trial's last required
                    // waypoint sooner than ever: keep that prefix (a tree run
                    // can restart from it).
                    if reject.is_some() && t.whole.is_none() && span.is_some() {
                        self.prefix_record(c, last)?;
                    }
                } else {
                    reject = Some(if t.whole.is_some() { "verify failed" } else if span.is_none() { "not all met" } else { "not faster" });
                }
                // Repair: a faster trial that didn't make the route sooner gets
                // one follow-up from where it ended to the next essential
                // waypoint after its last stage.
                // Repairs cascade: a repair that met its goals but whose route
                // still failed (time shifted, so the old inputs after it no
                // longer work) is extended to the next essential waypoint,
                // until the goal.
                let depth = if repair_of.is_some() { t.tag.strip_prefix("repair").and_then(|r| r.split(' ').next()).and_then(|d| d.parse::<usize>().ok()).unwrap_or(1) } else { 0 };
                if candidate.is_some() && reject.is_some() && t.whole.is_none() && depth < 12 {
                    if let Some(ts) = &trial_stream {
                        let ess = ((last + 1)..=self.loaded.waypoints.len()).find(|&k| self.loaded.waypoints[k - 1].essential);
                        if let Some(ess) = ess {
                            // Same start as the trial; seed = its faster stream
                            // then the old stream up to the essential waypoint;
                            // mutations only on the old part (the joint).
                            let from_pos = to.map_or(snap_stream.len(), |a| a.1).min(snap_stream.len());
                            let ess_pos = snap_arrival[ess].map_or(snap_stream.len(), |a| a.1).min(snap_stream.len()).max(from_pos);
                            let mut seed = ts.clone();
                            seed.extend_from_slice(&snap_stream[from_pos..ess_pos]);
                            let mut stages = t.stages.clone();
                            stages.extend(((last + 1)..=ess).filter(|&k| snap_arrival[k].is_some() || k == ess));
                            let focus = Some((ts.len().saturating_sub(10), seed.len()));
                            repairs.push((
                                Trial {
                                    from: t.from,
                                    stages,
                                    spec: GaSpec { population: 16, generations: 6, horizon: None },
                                    seed: Some(seed),
                                    tag: format!("repair{} {last}->{ess}", depth + 1),
                                    focus,
                                    custom_prob: 0.0,
                                    start: t.start.clone(),
                                    ..Default::default()
                                },
                                repair_of.clone().unwrap_or_else(|| t.tag.clone()),
                            ));
                            repairs_tried += 1;
                        }
                    }
                }
                writeln!(
                    self.log,
                    "{}",
                    serde_json::json!({
                        "id": id, "t": self.started.elapsed().as_secs_f64(), "from": t.from, "stages": t.stages.len(),
                        "last": last, "met": met, "span": span, "best_span": best_span,
                        "improved": reject.is_none(), "reject": reject, "repair": repair_of.is_some(),
                        "goal_frame": self.best.goal_frame, "tag": t.tag,
                    })
                )?;
                if repair_of.is_some() {
                    continue;
                }
                let ctx = Ctx {
                    waypoints: &self.loaded.waypoints,
                    best: &self.best,
                    original_frame: self.loaded.original_frame,
                    elapsed: self.started.elapsed().as_secs_f64(),
                    trials: self.trials,
                };
                let outcome = Outcome {
                    id,
                    met,
                    span_frames: span,
                    best_span_frames: best_span,
                    improved: reject.is_none(),
                    reject,
                    stream: trial_stream,
                    goal_frame: self.best.goal_frame,
                    frames_used: frames,
                    trial: t,
                };
                method.feedback(&outcome, &ctx);
            }
        }
        println!("repairs: {repairs_tried} queued, {repaired} improved the route");
        self.write_outputs(method.name(), improvements)?;
        Ok(())
    }

    /// Replays `stream` from the root to route state `upto` (its essential
    /// waypoints and `upto` itself, in order); if it gets there sooner than
    /// any prefix so far (the best route's arrival included), writes
    /// prefix-<upto>.m64 (from power-on) and records it.
    fn prefix_record(&mut self, stream: &[GamePad], upto: usize) -> Result<()> {
        let ld = &self.loaded;
        if upto == 0 || upto > ld.waypoints.len() {
            return Ok(());
        }
        let stages: Vec<&Waypoint> = ld.waypoints[..upto].iter().filter(|w| w.essential || w.index == upto).collect();
        let w = self.workers[0].get_mut().unwrap();
        w.base.load_state(&ld.root_save)?;
        let r = rollout(w, &ld.root, stream, &stages, &ld.policy, 0, stream.len() * 3 + 3000, false)?;
        if r.ends.len() < stages.len() {
            return Ok(());
        }
        let at = r.ends.last().unwrap().0;
        let previous = self.prefixes.get(&upto).copied().or_else(|| self.best.arrival.get(upto).copied().flatten().map(|a| a.0));
        if previous.is_some_and(|p| at >= p) {
            return Ok(());
        }
        self.prefixes.insert(upto, at);
        let mut m64 = Vec::new();
        for p in ld.boot.iter().chain(&r.played[..at]) {
            m64.extend_from_slice(&p.button.to_be_bytes());
            m64.extend_from_slice(&[p.stick_x as u8, p.stick_y as u8]);
        }
        fs::write(format!("{}/prefix-{upto}.m64", self.out_dir), m64)?;
        let original = ld.states[upto].frame as i64 - ld.root.frame as i64;
        println!(
            "{:6.1}s  prefix record: route state {upto} reached at frame {} ({} sooner than the original route) -> prefix-{upto}.m64",
            self.started.elapsed().as_secs_f64(),
            ld.root.frame as usize + at,
            original - at as i64
        );
        Ok(())
    }

    fn write_outputs(&mut self, method: &str, improvements: u64) -> Result<()> {
        let mut m64 = Vec::with_capacity((self.loaded.boot.len() + self.best.played.len()) * 4);
        for p in self.loaded.boot.iter().chain(&self.best.played) {
            m64.extend_from_slice(&p.button.to_be_bytes());
            m64.extend_from_slice(&[p.stick_x as u8, p.stick_y as u8]);
        }
        fs::write(format!("{}/best.m64", self.out_dir), m64)?;
        let mut trace = String::from("f,zone,x,y,z,ctl,stream\n");
        for (i, t) in self.best.trace.iter().enumerate() {
            trace += &format!(
                "{},{}:{},{:.0},{:.0},{:.0},{},{}\n",
                self.loaded.root.frame as usize + i + 1,
                t.zone.0,
                t.zone.1,
                t.pos[0],
                t.pos[1],
                t.pos[2],
                t.controllable as u8,
                t.stream_pos
            );
        }
        fs::write(format!("{}/trace.csv", self.out_dir), trace)?;
        let summary = serde_json::json!({
            "method": method,
            "original_frame": self.loaded.original_frame,
            "best_frame": self.best.goal_frame,
            "saved": self.loaded.original_frame as i64 - self.best.goal_frame as i64,
            "trials": self.trials, "frames": self.frames, "improvements": improvements,
            "seconds": self.started.elapsed().as_secs_f64(),
            "waypoints": self.loaded.waypoints.len(),
            "essential": self.loaded.waypoints.iter().filter(|w| w.essential).count(),
        });
        fs::write(format!("{}/summary.json", self.out_dir), serde_json::to_string_pretty(&summary)?)?;
        println!("{summary}");
        Ok(())
    }
}

#[allow(dead_code)]
fn _unused(_: &SM64Game) {}
