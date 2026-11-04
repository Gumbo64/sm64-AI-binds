//! Speed mode: take an objective-rrt-v2 (obrrt2) run's best route and make it
//! faster. Every switch is off by default; with all of them off oss2 is
//! unchanged (same draws, same outputs).
//!
//! - **Piping** (`--pipe <obrrt run dir> [--pipe-node <id>]`): the route is
//!   the inputs from power-on to one node of the run's tree (default: the
//!   earliest node with the most stars), read from its tree.bin with
//!   obrrt-search's `route::read_tree_bin` and nodepath's walk to the root.
//!   tree.bin's record 0 is obrrt2's boot (fixed or `OBRRT_TAS_BOOT`), so the
//!   inputs replay from power-on as they are. The replay is checked against
//!   nodes.csv (frame and stars) and its every state is merged as a seed from
//!   the power-on cell, so the route's keys become cells holding their route
//!   frames ([`Route`]). Its milestones are recorded: each star and key
//!   count's first frame and each zone entry (zone, stars, keys).
//! - **Corridor** (`OSS_CORRIDOR=<units>`): a state is kept only if it is
//!   within that distance of a route state of the same phase: zone, stars,
//!   keys and act ([`Corridor`], a grid of the route's positions per phase).
//! - **Route selection** (`OSS_SELECT=route`): see [`Speed::select`].
//! - **Beat-the-record GA** (`OSS_GA_TARGET=route`): GA expansions aim at a
//!   route state Δ ∈ [30, 300] frames ahead of the base cell's place on the
//!   route ([`ReachRoute`], [`expand_route`]).
//! - **Propagation** (`OSS_PROP=<share of frames>`): when a cell is replaced
//!   by a faster state, the inputs of each child node of its old node are
//!   replayed from the faster state and merged (first-or-faster); a child
//!   cell reached sooner is replaced in turn, which queues its own children,
//!   so the saving cascades down the subtree (as obrrt's reroute grafts).

use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::{Context, Result, bail, ensure};
use obrrt_core::{Eta, GaBudget, Mem, Objective, Precision, St, frames_for, read_state};
use rand::Rng;
use rand::rngs::SmallRng;
use serde_json::{Value, json};
use sm64_binds::GamePad;

use crate::archive::{Archive, Cell, Entry};
use crate::config::Config;
use crate::ga::GaOut;
use crate::inputs::{Operator, plan};
use crate::objective::Tree;
use crate::search::{Worker, play_on};

/// Where on the route a state is: (zone, stars, keys, act).
pub type Phase = (u32, i16, u8, u8);

pub fn phase(st: &St) -> Phase {
    (crate::search::zone(st), st.stars, st.keys, st.act)
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// The speed switches (see the module doc and README.md).
#[derive(Clone, Copy, Debug, Default)]
pub struct SpeedConfig {
    /// `OSS_CORRIDOR=<units>`.
    pub corridor: Option<f32>,
    /// `OSS_SELECT=route`.
    pub select_route: bool,
    /// `OSS_GA_TARGET=route`.
    pub ga_route: bool,
    /// `OSS_GA_SEED=0|1` (1): seed the route GA's first generation with the
    /// route's own inputs from the anchor to the target (`attempt_seeded`:
    /// the seed, and copies with frames or a chunk dropped).
    pub ga_seed: bool,
    /// `OSS_PROP=<share>`: propagation's share of all frames (0 = off).
    pub prop: f64,
}

impl SpeedConfig {
    pub fn from_env() -> Result<SpeedConfig> {
        let mut c = SpeedConfig { ga_seed: true, ..Default::default() };
        if let Ok(v) = std::env::var("OSS_CORRIDOR") {
            let d: f32 = v.parse().context("OSS_CORRIDOR: a distance in units")?;
            ensure!(d > 0.0, "OSS_CORRIDOR: a positive distance");
            c.corridor = Some(d);
        }
        c.select_route = std::env::var("OSS_SELECT").is_ok_and(|v| v == "route");
        match std::env::var("OSS_GA_TARGET").as_deref() {
            Err(_) | Ok("registry") => {}
            Ok("route") => c.ga_route = true,
            Ok(v) => bail!("OSS_GA_TARGET: route or registry, not {v}"),
        }
        if let Ok(v) = std::env::var("OSS_GA_SEED") {
            c.ga_seed = v != "0";
        }
        if let Ok(v) = std::env::var("OSS_PROP") {
            let p: f64 = v.parse()?;
            ensure!((0.0..=1.0).contains(&p), "OSS_PROP: a share in 0..=1, not {v}");
            c.prop = p;
        }
        Ok(c)
    }

    /// Whether any switch needs a piped route.
    pub fn needs_route(&self) -> bool {
        self.corridor.is_some() || self.select_route || self.ga_route
    }
}

// ---------- the route ----------

/// A route state: the one after input `t`, at frame `t + 1`.
pub struct RoutePt {
    pub frame: u32,
    pub phase: Phase,
    pub pos: [f32; 3],
    pub key: Box<[Entry]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MKind {
    Stars(i16),
    Keys(u8),
    /// Entering zone with (stars, keys).
    Zone(u32, i16, u8),
}

pub struct Milestone {
    pub name: String,
    pub frame: u32,
    pub kind: MKind,
}

/// The piped route.
pub struct Route {
    /// The obrrt run dir and node.
    pub source: String,
    pub node: u32,
    pub inputs: Vec<GamePad>,
    pub pts: Vec<RoutePt>,
    /// The first controllable state (the boot before it is fixed): selection
    /// and targets start here.
    pub start: usize,
    /// First route state with each key.
    by_key: HashMap<Box<[Entry]>, u32>,
    /// Route states by (zone, stars, keys), for anchoring off-route cells.
    by_place: HashMap<(u32, i16, u8), Vec<u32>>,
    pub milestones: Vec<Milestone>,
}

/// The inputs from power-on to `node` of obrrt run `dir` (default: the
/// earliest node with the most stars), with that node's id, frame and stars
/// from nodes.csv.
pub fn load_pipe(dir: &str, node: Option<u32>) -> Result<(Vec<GamePad>, u32, u32, i16)> {
    let csv = std::fs::read_to_string(format!("{dir}/nodes.csv")).with_context(|| format!("reading {dir}/nodes.csv"))?;
    let mut lines = csv.lines();
    let head: Vec<&str> = lines.next().context("nodes.csv is empty")?.split(',').collect();
    let col = |n: &str| head.iter().position(|h| *h == n).with_context(|| format!("nodes.csv has no {n} column"));
    let (ci, cs, cf) = (col("id")?, col("stars")?, col("frame")?);
    let rows: Vec<(u32, i16, u32)> = lines
        .filter(|l| !l.is_empty())
        .map(|l| {
            let r: Vec<&str> = l.split(',').collect();
            Ok((r[ci].parse()?, r[cs].parse()?, r[cf].parse()?))
        })
        .collect::<Result<_>>()?;
    let pick = match node {
        Some(n) => *rows.iter().find(|r| r.0 == n).with_context(|| format!("node {n} is not in nodes.csv"))?,
        None => {
            let most = rows.iter().map(|r| r.1).max().context("nodes.csv has no nodes")?;
            *rows.iter().filter(|r| r.1 == most).min_by_key(|r| (r.2, r.0)).unwrap()
        }
    };
    // nodepath's walk: record 0 is the boot, node n is record n + 1.
    let records = obrrt_search::route::read_tree_bin(&format!("{dir}/tree.bin"))?;
    ensure!((pick.0 as usize) + 1 < records.len(), "node {} is not in tree.bin", pick.0);
    let mut chain = vec![pick.0 as usize + 1];
    let mut n = pick.0 as i32;
    loop {
        let parent = records[n as usize + 1].0;
        chain.push(if parent >= 0 { parent as usize + 1 } else { 0 });
        if parent < 0 {
            break;
        }
        n = parent;
    }
    let inputs: Vec<GamePad> = chain.iter().rev().flat_map(|&r| records[r].1.iter().copied()).collect();
    Ok((inputs, pick.0, pick.2, pick.1))
}

impl Route {
    /// Builds the route from its states, one per input (`sts[t]` after input t).
    pub fn new(source: String, node: u32, inputs: Vec<GamePad>, sts: &[St], tree: &Tree) -> Route {
        let start = sts.iter().position(|s| s.controllable()).unwrap_or(0);
        let mut pts = Vec::with_capacity(sts.len());
        let (mut by_key, mut by_place) = (HashMap::new(), HashMap::<_, Vec<u32>>::new());
        let mut milestones = Vec::new();
        let (mut stars, mut keys, mut zone) = (i16::MAX, u8::MAX, u32::MAX);
        let mut seen_zone = HashSet::new();
        for (t, s) in sts.iter().enumerate() {
            let key: Box<[Entry]> = tree.key(s).into();
            by_key.entry(key.clone()).or_insert(t as u32);
            let ph = phase(s);
            if t >= start {
                by_place.entry((ph.0, ph.1, ph.2)).or_default().push(t as u32);
                if stars == i16::MAX {
                    (stars, keys, zone) = (s.stars, s.keys, ph.0);
                }
                for n in stars + 1..=s.stars {
                    milestones.push(Milestone { name: format!("star {n}"), frame: s.frame, kind: MKind::Stars(n) });
                }
                for n in keys.saturating_add(1)..=s.keys {
                    milestones.push(Milestone { name: format!("key {n}"), frame: s.frame, kind: MKind::Keys(n) });
                }
                (stars, keys) = (stars.max(s.stars), keys.max(s.keys));
                // A zone entry is a milestone only the first time the route is
                // in that zone with those stars and keys at all (a star got
                // in a zone isn't an entry when the route comes back).
                let new = seen_zone.insert((ph.0, s.stars, s.keys));
                if ph.0 != zone && new {
                    let name = format!("zone {}.{} ({}*{})", s.level, s.area, s.stars, if s.keys > 0 { format!(" {}k", s.keys) } else { String::new() });
                    milestones.push(Milestone { name, frame: s.frame, kind: MKind::Zone(ph.0, s.stars, s.keys) });
                }
                zone = ph.0;
            }
            pts.push(RoutePt { frame: s.frame, phase: ph, pos: s.pos, key });
        }
        Route { source, node, inputs, pts, start, by_key, by_place, milestones }
    }

    /// The first route state with `key`.
    pub fn index_of(&self, key: &[Entry]) -> Option<usize> {
        self.by_key.get(key).map(|&t| t as usize)
    }

    /// Where cell `c` is on the route: its key's first route state, else the
    /// nearest route state in its zone with its stars and keys (the earliest
    /// of equals). None if neither (or before the route's start).
    pub fn anchor(&self, a: &Archive, c: &Cell) -> Option<usize> {
        if let Some(t) = self.index_of(&c.key).filter(|&t| t >= self.start) {
            return Some(t);
        }
        let (keys, stars) = progress(c);
        let zone = a.nodes[c.node as usize].zone;
        let pos = c.pos.map(|v| v as f32);
        let near = self.by_place.get(&(zone, stars?, keys?))?;
        near.iter().map(|&t| (dist(self.pts[t as usize].pos, pos), t)).min_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1))).map(|x| x.1 as usize)
    }
}

/// A cell's (keys, stars), from the Root objective's entries (id 0, fields
/// 1 and 2; the pipe checks that objective 0 is a Root).
pub fn progress(c: &Cell) -> (Option<u8>, Option<i16>) {
    let get = |f: u8| c.key.iter().find(|e| e.0 == 0 && e.1 == f).map(|e| e.2);
    (get(1).map(|v| v as u8), get(2).map(|v| v as i16))
}

// ---------- the corridor ----------

/// The route's positions per phase, in a grid of the corridor's width.
pub struct Corridor {
    pub width: f32,
    grid: HashMap<(Phase, [i32; 3]), Vec<[f32; 3]>>,
}

impl Corridor {
    pub fn new(width: f32, pts: &[RoutePt]) -> Corridor {
        let mut grid: HashMap<(Phase, [i32; 3]), Vec<[f32; 3]>> = HashMap::new();
        let thin = width / 8.0;
        for p in pts {
            let v = grid.entry((p.phase, Self::bucket(width, p.pos))).or_default();
            // Points closer than width / 8 to the last one kept add nothing.
            if v.last().is_none_or(|&q| dist(q, p.pos) > thin) {
                v.push(p.pos);
            }
        }
        Corridor { width, grid }
    }

    fn bucket(w: f32, p: [f32; 3]) -> [i32; 3] {
        p.map(|v| (v / w).floor() as i32)
    }

    /// Whether `st` is within the width of a route state of its phase.
    pub fn admits(&self, st: &St) -> bool {
        let ph = phase(st);
        let b = Self::bucket(self.width, st.pos);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(v) = self.grid.get(&(ph, [b[0] + dx, b[1] + dy, b[2] + dz]))
                        && v.iter().any(|&q| dist(q, st.pos) <= self.width)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }
}

// ---------- the beat-the-record objective ----------

/// Within this distance of the target in its phase is done, even if the key
/// differs (a field that doesn't matter here, e.g. a Near objective's).
pub const DONE_DIST: f32 = 150.0;
/// The key is compared only this close (cheap: most frames are further).
const KEY_DIST: f32 = 2000.0;
/// Optimism: a state is Lost once frame + OPT × ETA can't beat the record.
pub const OPT: f64 = 0.7;

/// Reach a route state sooner than the record for its key.
///
/// - ETA: in the target's phase, `frames_for(distance to it)`; in an earlier
///   phase of the route between anchor and target (another zone, before a
///   star or key), the distance to where the route left that phase plus the
///   route's frames from there to the target; any other phase is Lost.
/// - Done: in the target's phase, the schema key equal to the target's, or
///   within [`DONE_DIST`] of it (keys can differ in fields that don't matter
///   to the route, so equality alone is too strict).
/// - Lost once `s.frame ≥ record` or `s.frame + OPT × eta ≥ record`.
pub struct ReachRoute<'a> {
    tree: &'a Tree,
    pub frame: u32,
    pub record: u32,
    pos: [f32; 3],
    phase: Phase,
    key: &'a [Entry],
    /// Earlier phases: where the route leaves each, and when.
    legs: Vec<(Phase, [f32; 3], u32)>,
    budget: GaBudget,
}

impl<'a> ReachRoute<'a> {
    pub fn new(tree: &'a Tree, route: &'a Route, anchor: usize, target: usize, record: u32) -> ReachRoute<'a> {
        let t = &route.pts[target];
        let mut legs = Vec::new();
        for i in anchor..target {
            if route.pts[i].phase != route.pts[i + 1].phase {
                legs.push((route.pts[i].phase, route.pts[i].pos, route.pts[i].frame));
            }
        }
        let h = target - anchor + 60;
        ReachRoute { tree, frame: t.frame, record, pos: t.pos, phase: t.phase, key: &t.key, legs, budget: GaBudget { population: 16, generations: 5, horizon: h } }
    }

    /// The point to head for from `s` and the frames after it, if `s` is in
    /// a phase on the way.
    fn locate(&self, s: &St) -> Option<([f32; 3], f64)> {
        let ph = phase(s);
        if ph == self.phase {
            return Some((self.pos, frames_for(dist(s.pos, self.pos))));
        }
        (self.legs.iter())
            .filter(|l| l.0 == ph)
            .map(|l| (l.1, frames_for(dist(s.pos, l.1)) + (self.frame - l.2) as f64))
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }
}

impl Objective for ReachRoute<'_> {
    fn kind(&self) -> &'static str {
        "Speed::ReachRoute"
    }
    fn key(&self) -> String {
        format!("Speed::ReachRoute@{}", self.frame)
    }
    fn obj_weight(&self, _: &St) -> f64 {
        1.0
    }
    fn base_criteria(&self, _: &St) -> bool {
        true
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.frame >= self.record {
            return Eta::Lost;
        }
        if phase(s) == self.phase {
            let d = dist(s.pos, self.pos);
            if d <= DONE_DIST || (d <= KEY_DIST && *self.tree.key(s) == *self.key) {
                return Eta::Done;
            }
        }
        match self.locate(s) {
            Some((_, e)) if (s.frame as f64 + OPT * e) < self.record as f64 => Eta::Frames(e),
            _ => Eta::Lost,
        }
    }
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(self.budget)
    }
    fn precision(&self) -> Precision {
        Precision::Precise
    }
    fn needs_objects(&self) -> bool {
        self.tree.objects
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.locate(s).map(|l| l.0)
    }
}

/// One beat-the-record GA expansion of cell `src` (`OSS_GA_TARGET=route`):
/// 1. anchor the cell on the route ([`Route::anchor`]);
/// 2. draw Δ ∈ [30, 300] and take the route state Δ after the anchor (the
///    last one if that is past the end); the record is the fewer of its
///    route frame and its key's cell's frames; up to 4 draws for one the
///    cell can still beat (cell frames < record);
/// 3. run v2's GA (`attempt_full`, budget 16 / 5 / Δ + 60 at scale 1, so 5
///    generations; seeded with the route's inputs anchor → target unless
///    `OSS_GA_SEED=0`) towards [`ReachRoute`];
/// 4. replay the winner's inputs from the cell and merge every state
///    (first-or-faster), as a `winner` harvest.
///
/// A cell with no anchor, at the route's end, already behind every record
/// drawn, or in a modal state (where ReachRoute can't start) gets a pellet
/// instead (a GA `fallback`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn expand_route(w: &mut Worker, archive: &Archive, tree: &Tree, cfg: &Config, route: &Route, src: usize, mut rng: SmallRng) -> Result<GaOut> {
    let cell = &archive.cells[src];
    let p = cell.p();
    let last = route.pts.len() - 1;
    let mut pick = None;
    if let Some(a) = route.anchor(archive, cell).filter(|&a| a < last) {
        for _ in 0..4 {
            let t = (a + rng.random_range(30..=300)).min(last);
            let rp = &route.pts[t];
            let record = archive.find(&rp.key).map_or(rp.frame, |i| archive.cells[i].frames.min(rp.frame));
            if cell.frames < record {
                pick = Some((a, t, record));
                break;
            }
        }
    }
    let fallback = |w: &mut Worker, rng: &mut SmallRng| -> Result<GaOut> {
        let (op, inputs) = plan(archive, cfg, src, rng);
        let r = w.play(archive, tree, cfg, src, inputs, false)?;
        let frames = r.inputs.len() as u64;
        Ok(GaOut { src, p, op, parts: vec![r], kind: None, met: false, eligible: 0, frames })
    };
    let Some((a, t, record)) = pick else { return fallback(w, &mut rng) };
    archive.restore(&mut w.ga.base, cell.node as usize)?;
    let st = read_state(&w.ga.base, cell.frames, true);
    let obj = ReachRoute::new(tree, route, a, t, record);
    if st.modal() || !obj.eta(&st, &st, &mut Mem::default()).frames().is_finite() {
        return fallback(w, &mut rng);
    }
    let seed = cfg.speed.ga_seed.then(|| &route.inputs[a + 1..=t]);
    let params = obrrt_search::ga::GaParams { scale: 1.0, ..cfg.ga.params };
    let att = w.ga.attempt_full(&[&obj], &st, None, &mut rng, params, seed, None)?;
    let met = att.segments.first().is_some_and(|s| s.success);
    let mut frames = att.frames;
    let winner: Vec<GamePad> = att.segments.iter().flat_map(|s| s.inputs.iter().copied()).collect();
    let mut parts = Vec::new();
    if !winner.is_empty() {
        let g = &mut w.ga;
        g.tmp.sync_from(&g.base)?;
        frames += winner.len() as u64;
        parts.push(play_on(&mut g.tmp, archive, tree, cfg, src, winner, false, true)?);
    }
    Ok(GaOut { src, p, op: Operator::Ga, parts, kind: Some("Speed::ReachRoute"), met, eligible: 1, frames })
}

// ---------- selection, propagation, stats ----------

/// Route frames over which [`Speed::gap`] measures a lost lead.
const GAP: usize = 150;
/// Most recently improved cells kept for selection.
const RECENT: usize = 64;
/// Propagation queue cap: grafts past it are dropped (and forgotten, so the
/// dedupe set never holds more than the queue).
const PROP_CAP: usize = 1 << 16;

#[derive(Default, Clone, Copy, Debug)]
pub struct PropStats {
    pub queued: u64,
    pub dropped: u64,
    pub tried: u64,
    pub frames: u64,
    /// Cells the replays replaced by a faster state, and added.
    pub improved: u64,
    pub added: u64,
}

pub struct Speed {
    pub route: Route,
    /// Recently improved cells, newest last.
    recent: VecDeque<u32>,
    /// Grafts: (improved cell, child node of its old node).
    queue: VecDeque<(u32, u32)>,
    /// The queued pairs (only those: a pair is forgotten when taken or
    /// dropped, so this is bounded by the queue; cf. the reroute.rs leak).
    seen: HashSet<(u32, u32)>,
    pub prop: PropStats,
    /// Times each cell's best was replaced (by cell id).
    pub imp: Vec<u16>,
}

impl Speed {
    pub fn new(route: Route) -> Speed {
        Speed { route, recent: VecDeque::new(), queue: VecDeque::new(), seen: HashSet::new(), prop: PropStats::default(), imp: Vec::new() }
    }

    /// Cell `i` was replaced by a faster state.
    pub fn note(&mut self, i: u32) {
        if self.imp.len() <= i as usize {
            self.imp.resize(i as usize + 1, 0);
        }
        self.imp[i as usize] = self.imp[i as usize].saturating_add(1);
        self.recent.retain(|&c| c != i);
        if self.recent.len() == RECENT {
            self.recent.pop_front();
        }
        self.recent.push_back(i);
    }

    pub fn push(&mut self, cell: u32, child: u32) {
        if !self.seen.insert((cell, child)) {
            return;
        }
        if self.queue.len() >= PROP_CAP {
            self.seen.remove(&(cell, child));
            self.prop.dropped += 1;
            return;
        }
        self.queue.push_back((cell, child));
        self.prop.queued += 1;
    }

    pub fn pop(&mut self) -> Option<(u32, u32)> {
        let g = self.queue.pop_front()?;
        self.seen.remove(&g);
        Some(g)
    }

    pub fn queue_len(&self) -> (usize, usize) {
        (self.queue.len(), self.seen.len())
    }

    /// Frames the best cell with route state `t`'s key is ahead of the route
    /// there (0 without one).
    pub fn lead(&self, a: &Archive, t: usize) -> i64 {
        let r = &self.route;
        let first = r.index_of(&r.pts[t].key).unwrap_or(t);
        a.find(&r.pts[t].key).map_or(0, |i| (r.pts[first].frame as i64 - a.cells[i].frames as i64).max(0))
    }

    /// How much of the lead at route state `t` is lost by [`GAP`] frames
    /// later: where a saving stopped carrying downstream.
    pub fn gap(&self, a: &Archive, t: usize) -> i64 {
        self.lead(a, t) - self.lead(a, (t + GAP).min(self.route.pts.len() - 1))
    }

    /// `OSS_SELECT=route`: the base cell of an expansion.
    /// - 1/2: the route state with the largest [`Speed::gap`] of 4 drawn
    ///   uniformly from the route's start to its end (the first of equals),
    ///   and the cell holding its key: uniform while nothing is saved, then
    ///   biased to where a saving stops propagating (the time that could
    ///   still be saved downstream);
    /// - 1/4: one of the 64 most recently improved cells (an improvement
    ///   opens new savings right after it);
    /// - 1/4 (and when the above has no cell): any cell, uniformly (with a
    ///   corridor, every cell is a corridor cell).
    ///
    /// Deterministic: draws only from the run's rng.
    pub fn select(&self, a: &Archive, rng: &mut impl Rng) -> usize {
        let r: f64 = rng.random();
        if r < 0.5 {
            let mut best = (i64::MIN, 0);
            for _ in 0..4 {
                let t = rng.random_range(self.route.start..self.route.pts.len());
                let g = self.gap(a, t);
                if g > best.0 {
                    best = (g, t);
                }
            }
            if let Some(i) = a.find(&self.route.pts[best.1].key) {
                return i;
            }
        } else if r < 0.75 && !self.recent.is_empty() {
            return self.recent[rng.random_range(0..self.recent.len())] as usize;
        }
        rng.random_range(0..a.cells.len())
    }

    /// The fewest frames of a cell that reached milestone `m`, and that cell.
    pub fn best(&self, a: &Archive, m: &Milestone) -> Option<(u32, usize)> {
        (a.cells.iter().enumerate())
            .filter(|(_, c)| {
                let (keys, stars) = progress(c);
                match m.kind {
                    MKind::Stars(n) => stars.is_some_and(|s| s >= n),
                    MKind::Keys(n) => keys.is_some_and(|k| k >= n),
                    MKind::Zone(z, s, k) => a.nodes[c.node as usize].zone == z && stars == Some(s) && keys == Some(k),
                }
            })
            .map(|(i, c)| (c.frames, i))
            .min()
    }

    /// The milestone table and totals for stats.json.
    pub fn json(&self, a: &Archive, frames: u64, cfg: &crate::speed::SpeedConfig) -> Value {
        let r = &self.route;
        let ms: Vec<Value> = (r.milestones.iter())
            .map(|m| {
                let b = self.best(a, m).map(|b| b.0);
                json!({"name": m.name, "route_frame": m.frame, "best_frame": b, "saved_frames": b.map(|b| m.frame as i64 - b as i64)})
            })
            .collect();
        let fin = r.milestones.last().and_then(|m| Some((m.frame, self.best(a, m)?.0)));
        let saved_sum: i64 = (r.milestones.iter()).filter_map(|m| Some(m.frame as i64 - self.best(a, m)?.0 as i64)).sum();
        // Route keys whose cell now holds a faster state than the route's.
        let (mut n, mut faster, mut saved) = (0u64, 0u64, 0i64);
        for (k, &t) in &r.by_key {
            if (t as usize) < r.start {
                continue;
            }
            n += 1;
            if let Some(i) = a.find(k)
                && a.cells[i].frames < r.pts[t as usize].frame
            {
                faster += 1;
                saved += (r.pts[t as usize].frame - a.cells[i].frames) as i64;
            }
        }
        let p = &self.prop;
        json!({
            "pipe": r.source, "node": r.node, "route_frames": r.inputs.len(), "route_start": r.start,
            "corridor": cfg.corridor, "select": if cfg.select_route { "route" } else { "default" }, "ga_target": if cfg.ga_route { "route" } else { "registry" },
            "ga_seed": cfg.ga_seed, "prop_share": cfg.prop,
            "milestones": ms,
            "final": r.milestones.last().map(|m| m.name.clone()),
            "saved_frames": fin.map(|f| f.0 as i64 - f.1 as i64), "saved_s": fin.map(|f| (f.0 as f64 - f.1 as f64) / 30.0),
            "saved_s_sum": saved_sum as f64 / 30.0,
            "route_keys": n, "route_keys_faster": faster, "route_keys_saved_frames": saved,
            "improvements_per_mframe": if frames == 0 { 0.0 } else { a.replaced as f64 * 1e6 / frames as f64 },
            "prop": {"queued": p.queued, "dropped": p.dropped, "tried": p.tried, "frames": p.frames, "improved": p.improved, "added": p.added,
                     "queue": self.queue.len(), "seen": self.seen.len()},
        })
    }

    /// One log line: milestones' saved frames.
    pub fn line(&self, a: &Archive) -> String {
        (self.route.milestones.iter())
            .filter(|m| !matches!(m.kind, MKind::Zone(..)))
            .map(|m| format!("{}:{}", m.name, self.best(a, m).map_or("-".into(), |b| format!("{}({:+})", b.0, m.frame as i64 - b.0 as i64))))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Inputs as a raw .m64 (no header: 4 bytes a frame, buttons big-endian,
/// stick x, stick y), as obrrt2's writer and tree.bin store them; oss2's
/// `load_m64` and nodepath read it.
pub fn write_m64(path: &str, inputs: &[GamePad]) -> Result<()> {
    let mut b = Vec::with_capacity(inputs.len() * 4);
    for p in inputs {
        b.extend_from_slice(&p.button.to_be_bytes());
        b.extend_from_slice(&[p.stick_x as u8, p.stick_y as u8]);
    }
    std::fs::write(format!("{path}.tmp"), &b)?;
    std::fs::rename(format!("{path}.tmp"), path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::st;

    const ROOT: &str = r#"{"kind": "Root", "params": {"grid_log2": 10}}"#;

    /// Frame `f`, at x = `x` in castle grounds (16.1) or BoB (9.1), with `stars`.
    fn at(f: u32, x: f32, level: u16, stars: i16) -> St {
        let mut s = st([x, 0.0, 0.0], 0.0, level, 1, 0);
        (s.frame, s.stars) = (f, stars);
        s
    }

    /// Boot (2 frames, not controllable), 10 frames in the grounds, a BoB
    /// entry, a star at frame 20 in BoB.
    fn route(tree: &Tree) -> Route {
        let mut sts = Vec::new();
        for f in 1..=2 {
            let mut s = at(f, 0.0, 16, 0);
            s.action = 0;
            sts.push(s);
        }
        for f in 3..=12 {
            sts.push(at(f, 100.0 * f as f32, 16, 0));
        }
        for f in 13..=25 {
            sts.push(at(f, 100.0 * (f - 13) as f32, 9, if f >= 20 { 1 } else { 0 }));
        }
        let inputs = vec![GamePad::new(0, 0, 0); sts.len()];
        Route::new("test".into(), 0, inputs, &sts, tree)
    }

    #[test]
    fn route_milestones_start_at_control() {
        let tree = Tree::load(ROOT).unwrap();
        let r = route(&tree);
        assert_eq!(r.start, 2);
        let ms: Vec<(String, u32)> = r.milestones.iter().map(|m| (m.name.clone(), m.frame)).collect();
        assert_eq!(ms, vec![("zone 9.1 (0*)".to_string(), 13), ("star 1".to_string(), 20)]);
        assert_eq!(r.index_of(&tree.key(&at(3, 300.0, 16, 0))), Some(0), "the first state with a key");
    }

    #[test]
    fn a_zone_reentered_with_a_star_got_there_is_no_milestone() {
        let tree = Tree::load(ROOT).unwrap();
        // Grounds, BoB, a star in BoB, grounds, BoB again.
        let sts = [at(1, 0.0, 16, 0), at(2, 0.0, 9, 0), at(3, 0.0, 9, 1), at(4, 0.0, 16, 1), at(5, 0.0, 9, 1)];
        let r = Route::new("test".into(), 0, vec![GamePad::new(0, 0, 0); 5], &sts, &tree);
        let ms: Vec<&str> = r.milestones.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(ms, ["zone 9.1 (0*)", "star 1", "zone 16.1 (1*)"]);
    }

    #[test]
    fn corridor_is_distance_within_the_same_phase() {
        let tree = Tree::load(ROOT).unwrap();
        let r = route(&tree);
        let c = Corridor::new(600.0, &r.pts);
        assert!(c.admits(&at(0, 550.0, 16, 0)));
        let mut off = at(0, 500.0, 16, 0);
        off.pos[2] = 590.0;
        assert!(c.admits(&off));
        off.pos[2] = 700.0;
        assert!(!c.admits(&off), "too far from every route state");
        assert!(!c.admits(&at(0, 500.0, 16, 1)), "the route has no grounds state with a star");
        assert!(c.admits(&at(0, 500.0, 9, 1)));
        let mut act = at(0, 500.0, 9, 1);
        act.act = 2;
        assert!(!c.admits(&act), "another act is another phase");
    }

    #[test]
    fn reach_route_done_lost_and_legs() {
        let tree = Tree::load(ROOT).unwrap();
        let r = route(&tree);
        // Anchor: grounds frame 5 (index 4); target: BoB frame 18 (index 17, x 500).
        let o = ReachRoute::new(&tree, &r, 4, 17, 40);
        let m = &mut Mem::default();
        let s0 = at(5, 500.0, 16, 0);
        // In the grounds: to the route's exit (x 1200, frame 12), then 6 frames.
        let e = o.eta(&s0, &s0, m).frames();
        assert!((e - (700.0 / 30.0 + 6.0)).abs() < 1e-6, "{e}");
        // With a record of 18, frame 5 + 0.7 × 29.3 ≥ 18: Lost.
        assert_eq!(ReachRoute::new(&tree, &r, 4, 17, 18).eta(&s0, &s0, m), Eta::Lost);
        assert_eq!(o.eta(&at(14, 400.0, 9, 0), &s0, m), Eta::Done, "within 150 units");
        assert_eq!(o.eta(&at(14, 900.0, 9, 0), &s0, m), Eta::Done, "the same key (1024-unit cube)");
        assert!(matches!(o.eta(&at(14, 1100.0, 9, 0), &s0, m), Eta::Frames(_)));
        assert_eq!(o.eta(&at(40, 500.0, 9, 0), &s0, m), Eta::Lost, "not sooner than the record");
        assert_eq!(o.eta(&at(14, 500.0, 9, 1), &s0, m), Eta::Lost, "a phase not on the way");
        assert_eq!(o.steer_target(&s0), Some([1200.0, 0.0, 0.0]));
        assert_eq!(o.ga_budget().unwrap().horizon, 13 + 60);
    }

    #[test]
    fn prop_queue_forgets_dropped_and_taken_pairs() {
        let tree = Tree::load(ROOT).unwrap();
        let mut sp = Speed::new(route(&tree));
        for i in 0..PROP_CAP as u32 + 100 {
            sp.push(1, i);
        }
        sp.push(1, 0);
        assert_eq!(sp.queue_len(), (PROP_CAP, PROP_CAP));
        assert_eq!((sp.prop.queued, sp.prop.dropped), (PROP_CAP as u64, 100));
        assert_eq!(sp.pop(), Some((1, 0)));
        assert_eq!(sp.queue_len(), (PROP_CAP - 1, PROP_CAP - 1));
        sp.push(1, 0);
        assert_eq!(sp.queue_len(), (PROP_CAP, PROP_CAP), "a taken pair may be queued again");
    }

    #[test]
    fn recent_keeps_the_newest_improved_cells() {
        let tree = Tree::load(ROOT).unwrap();
        let mut sp = Speed::new(route(&tree));
        for i in 0..100 {
            sp.note(i % 70);
        }
        assert_eq!(sp.recent.len(), RECENT);
        assert_eq!(sp.recent.back(), Some(&29));
        assert_eq!(sp.imp[5], 2);
    }
}
