//! The experience roadmap (`OBRRT_ROAD=1`, or `OBRRT_ROAD=<dir>` to keep it
//! between runs): a graph of where the tree has been and what it cost to get
//! from one place to the next, used to rank and steer the goal draws
//! (`OBRRT_GOAL_FOCUS`).
//!
//! Why: a goal's ETA is a straight line, and Mario isn't holonomic. In the
//! zone-locked JRB tree of 2026-11-09 (camp-t1-0125) 7068 states had all 8
//! red coins and none collected the star 1000 units above them on the
//! ship's stern, in 623 attempts from the water under it; states with fewer
//! coins had stood at the star's spot (the path is dock, bow, deck). The
//! tree knew the way and couldn't use it.
//!
//! Cells are `OBRRT_ROAD_CELL` (256) unit cubes of a zone. Every tree edge
//! (parent state to child state, same zone, at most `OBRRT_ROAD_MAX_EDGE`
//! (400) frames) is a directed edge between their cells, costing the fewest
//! frames seen. A goal's cost from a cell is the cheapest of: straight there
//! ([`Road::direct`]: the horizontal distance at PACE, plus
//! `OBRRT_ROAD_CLIMB` (2) frames for each unit the target is above a jump's
//! reach, `OBRRT_ROAD_JUMP` (250); water up to its surface is free), or an
//! edge and the next cell's cost (Dijkstra from every cell at once). So a
//! state under the goal is far, and a state at the start of a path the tree
//! has walked is as near as the path is long.
//!
//! The states of a goal are ranked by this cost (never less than the ETA),
//! and a goal draw from a state whose cost comes through an edge attempts
//! that edge first: a move to the edge's end (`Road::Leg`), seeded with the
//! inputs that made the edge when they are in this tree.
//!
//! The graph ignores what a state holds (coins, stars, the act's objects),
//! so an edge may not be there for the state that asks: a leg that fails
//! wears its cell like any goal draw.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::io::Write;
use std::sync::{Arc, OnceLock};

use obrrt_core::{St, frames_for};

use crate::tree::Tree;

pub type C3 = (i32, i32, i32);
type Zone = (u16, u8);

#[derive(Clone, Copy)]
pub struct Settings {
    pub cell: f32,
    pub climb: f64,
    pub jump: f32,
    pub max_edge: u32,
    /// Rounds a goal's costs are kept before they are computed again.
    pub refresh: u64,
    /// Fields built per round at most (`OBRRT_ROAD_BUILDS`); past it a
    /// target with none uses the straight cost. A goal whose target moves
    /// with the state (an object's position) wanted one per state: the JRB
    /// tree of scout sc1 step 10 built 241,906 fields, each a search over
    /// 14k cells, and spent 2567 s of 3200 s ranking goals while the workers
    /// waited.
    pub builds: u64,
}

/// `OBRRT_ROAD`: None when off; Some(directory) to load and save.
pub fn setting() -> Option<Option<String>> {
    static S: OnceLock<Option<Option<String>>> = OnceLock::new();
    S.get_or_init(|| match obrrt_core::sys::env::var("OBRRT_ROAD") {
        Ok(v) if v.is_empty() || v == "0" => None,
        Ok(v) if v == "1" => Some(None),
        Ok(v) => Some(Some(v)),
        Err(_) => None,
    })
    .clone()
}

fn settings() -> Settings {
    static S: OnceLock<Settings> = OnceLock::new();
    *S.get_or_init(|| {
        let f = |k: &str, d: f64| obrrt_core::sys::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        Settings {
            cell: f("OBRRT_ROAD_CELL", 256.0) as f32,
            climb: f("OBRRT_ROAD_CLIMB", 2.0),
            jump: f("OBRRT_ROAD_JUMP", 250.0) as f32,
            max_edge: f("OBRRT_ROAD_MAX_EDGE", 400.0) as u32,
            refresh: f("OBRRT_ROAD_REFRESH", 20.0) as u64,
            builds: f("OBRRT_ROAD_BUILDS", 32.0) as u64,
        }
    })
}

/// The cheapest edge seen between two cells.
#[derive(Clone, Copy)]
pub struct Edge {
    pub frames: u32,
    pub count: u32,
    /// Where the edge's child state was.
    pub to: [f32; 3],
    /// The child node in this tree (its inputs make the edge); None for an
    /// edge loaded from a file.
    pub node: Option<u32>,
}

/// A cell's first state: its footing ([`St::footing`]) and the water surface
/// there. `old`: from a file written before footings (builds up to v3j),
/// where an airborne state's own height stood; the first state seen in the
/// cell replaces it.
#[derive(Clone, Copy)]
struct Rep {
    pos: [f32; 3],
    water: i16,
    old: bool,
}

struct Field {
    round: u64,
    cost: HashMap<C3, (f32, Option<C3>)>,
}

#[derive(Default)]
struct ZoneRoad {
    rep: HashMap<C3, Rep>,
    edges: HashMap<C3, HashMap<C3, Edge>>,
    n_edges: usize,
    fields: HashMap<C3, Arc<Field>>,
    loaded: bool,
}

#[derive(Default, Clone, serde::Serialize)]
pub struct Stats {
    pub cells: usize,
    pub edges: usize,
    pub loaded_edges: usize,
    pub fields: u64,
    /// Asks past the round's builds (`Settings::builds`): straight cost used.
    pub fields_capped: u64,
    pub legs: u64,
    /// `Road::Climb` attempts: far goals with no known way.
    pub climbs: u64,
    pub legs_seeded: u64,
    pub legs_ok: u64,
}

pub struct Road {
    s: Settings,
    dir: Option<String>,
    zones: HashMap<Zone, ZoneRoad>,
    seen: usize,
    round: u64,
    /// Fields built this round (see `Settings::builds`).
    built: u64,
    pub stats: Stats,
}

#[derive(PartialEq)]
struct Item(f32, C3);
impl Eq for Item {}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0).then_with(|| self.1.cmp(&o.1))
    }
}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// A leg to attempt: the edge's end, its frames, and its inputs' node.
pub struct Leg {
    pub to: [f32; 3],
    pub frames: u32,
    pub node: Option<u32>,
}

impl Road {
    pub fn new(dir: Option<String>) -> Road {
        Road { s: settings(), dir, zones: HashMap::new(), seen: 0, round: 0, built: 0, stats: Stats::default() }
    }

    fn cell(&self, p: [f32; 3]) -> C3 {
        let c = |v: f32| (v / self.s.cell).floor() as i32;
        (c(p[0]), c(p[1]), c(p[2]))
    }

    fn zone(&mut self, z: Zone) -> &mut ZoneRoad {
        let zr = self.zones.entry(z).or_default();
        if !zr.loaded {
            zr.loaded = true;
            if let Some(d) = &self.dir {
                if let Ok(text) = obrrt_core::sys::fs::read_to_string(format!("{d}/L{}A{}.road", z.0, z.1)) {
                    for line in text.lines() {
                        let mut it = line.split_whitespace();
                        let tag = it.next();
                        let v: Vec<f64> = it.filter_map(|x| x.parse().ok()).collect();
                        match tag {
                            Some("r") if v.len() >= 7 => {
                                zr.rep.entry((v[0] as i32, v[1] as i32, v[2] as i32)).or_insert(Rep { pos: [v[3] as f32, v[4] as f32, v[5] as f32], water: v[6] as i16, old: v.len() < 8 });
                            }
                            Some("e") if v.len() >= 11 => {
                                let (a, b) = ((v[0] as i32, v[1] as i32, v[2] as i32), (v[3] as i32, v[4] as i32, v[5] as i32));
                                let e = Edge { frames: v[6] as u32, count: v[7] as u32, to: [v[8] as f32, v[9] as f32, v[10] as f32], node: None };
                                if zr.edges.entry(a).or_default().insert(b, e).is_none() {
                                    zr.n_edges += 1;
                                    self.stats.loaded_edges += 1;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        zr
    }

    /// Takes in the tree's new nodes (each with its parent: an edge).
    pub fn observe(&mut self, tree: &Tree) {
        self.round += 1;
        self.built = 0;
        let max = self.s.max_edge;
        for i in self.seen..tree.nodes.len() {
            let n = &tree.nodes[i];
            let st = &n.st;
            if !st.alive() || st.modal() {
                continue;
            }
            let z = st.zone();
            let b = self.cell(st.pos);
            let rep = Rep { pos: st.footing(), water: st.water_level, old: false };
            let parent = n.parent.map(|p| &tree.nodes[p as usize].st).filter(|p| p.zone() == z && p.alive() && !p.modal());
            let a = parent.map(|p| (self.cell(p.pos), st.frame.saturating_sub(p.frame)));
            let zr = self.zone(z);
            // A cell from a file made without the water level (-32000) takes this state's.
            let r = zr.rep.entry(b).or_insert(rep);
            if r.water == -32000 || r.old {
                *r = rep;
            }
            let Some((a, dt)) = a else { continue };
            if a == b || dt == 0 || dt > max {
                continue;
            }
            let e = zr.edges.entry(a).or_default().entry(b);
            match e {
                std::collections::hash_map::Entry::Occupied(mut o) => {
                    let o = o.get_mut();
                    o.count += 1;
                    if dt < o.frames {
                        *o = Edge { frames: dt, count: o.count, to: st.pos, node: Some(i as u32) };
                    } else if o.node.is_none() {
                        // A loaded edge: this tree's own inputs for it.
                        o.node = Some(i as u32);
                        o.to = st.pos;
                    }
                }
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(Edge { frames: dt, count: 1, to: st.pos, node: Some(i as u32) });
                    zr.n_edges += 1;
                }
            }
        }
        self.seen = tree.nodes.len();
    }

    /// Frames straight from `p` to `t`: the horizontal distance at PACE, and
    /// `climb` frames a unit for the height beyond a jump (from the water's
    /// surface when `p` is under it).
    fn direct_s(s: &Settings, p: [f32; 3], water: i16, t: [f32; 3]) -> f64 {
        let h = ((p[0] - t[0]).powi(2) + (p[2] - t[2]).powi(2)).sqrt();
        let from = if p[1] < water as f32 { (water as f32).min(t[1]).max(p[1]) } else { p[1] };
        let up = (t[1] - from - s.jump).max(0.0);
        let down = (p[1] - t[1]).max(0.0);
        frames_for(h.max(down * 0.5)) + s.climb * up as f64
    }

    /// The climb cost per unit of height and the free jump height.
    pub fn climb(&self) -> (f64, f32) {
        (self.s.climb, self.s.jump)
    }

    pub fn direct(&self, st: &St, t: [f32; 3]) -> f64 {
        Self::direct_s(&self.s, st.footing(), st.water_level, t)
    }

    /// The goal's costs from every cell of the zone (kept `refresh` rounds).
    fn field(&mut self, z: Zone, t: [f32; 3]) -> Option<Arc<Field>> {
        let key = self.cell(t);
        let (s, round) = (self.s, self.round);
        if self.built >= s.builds {
            if let Some(f) = self.zones.get(&z).and_then(|zr| zr.fields.get(&key)).filter(|f| round < f.round + s.refresh) {
                return Some(f.clone());
            }
            self.stats.fields_capped += 1;
            return None;
        }
        let zr = self.zone(z);
        if let Some(f) = zr.fields.get(&key) {
            if round < f.round + s.refresh {
                return Some(f.clone());
            }
        }
        if zr.fields.len() > 96 {
            zr.fields.retain(|_, f| round < f.round + s.refresh);
            if zr.fields.len() > 96 {
                zr.fields.clear();
            }
        }
        let mut cost: HashMap<C3, (f32, Option<C3>)> = zr.rep.iter().map(|(&c, r)| (c, (Self::direct_s(&s, r.pos, r.water, t) as f32, None))).collect();
        let mut rev: HashMap<C3, Vec<(C3, u32)>> = HashMap::new();
        for (&a, m) in &zr.edges {
            for (&b, e) in m {
                rev.entry(b).or_default().push((a, e.frames));
            }
        }
        let mut heap: BinaryHeap<Item> = cost.iter().map(|(&c, &(d, _))| Item(d, c)).collect();
        while let Some(Item(d, c)) = heap.pop() {
            if cost.get(&c).is_some_and(|x| d > x.0) {
                continue;
            }
            for &(a, w) in rev.get(&c).map_or(&[][..], |v| v) {
                let nd = d + w as f32;
                if let Some(x) = cost.get_mut(&a) {
                    if nd < x.0 {
                        *x = (nd, Some(c));
                        heap.push(Item(nd, a));
                    }
                }
            }
        }
        let f = Arc::new(Field { round, cost });
        zr.fields.insert(key, f.clone());
        self.stats.fields += 1;
        self.built += 1;
        Some(f)
    }

    /// Frames from `st` to `t`: straight ([`Road::direct`]) or along the
    /// roadmap from its cell, whichever is less.
    pub fn eta(&mut self, st: &St, t: [f32; 3]) -> f64 {
        let d = self.direct(st, t);
        let c = self.cell(st.pos);
        let Some(f) = self.field(st.zone(), t) else { return d };
        f.cost.get(&c).map_or(d, |x| d.min(x.0 as f64))
    }

    /// [`Road::eta`] from a field already built and still fresh, or None
    /// (then `eta` builds it). Read-only, so the engine's goal ranking can
    /// run on every thread; the values are the same as `eta`'s.
    pub fn eta_cached(&self, st: &St, t: [f32; 3]) -> Option<f64> {
        let f = self.zones.get(&st.zone())?.fields.get(&self.cell(t))?;
        if self.round >= f.round + self.s.refresh {
            return None;
        }
        let d = self.direct(st, t);
        Some(f.cost.get(&self.cell(st.pos)).map_or(d, |x| d.min(x.0 as f64)))
    }

    /// The first edge of the roadmap's path from `st` to `t`, when the path
    /// beats going straight.
    pub fn way(&mut self, st: &St, t: [f32; 3]) -> Option<Leg> {
        let d = self.direct(st, t);
        let c = self.cell(st.pos);
        let z = st.zone();
        let f = self.field(z, t)?;
        let &(cost, next) = f.cost.get(&c)?;
        let next = next?;
        if cost as f64 > 0.9 * d {
            return None;
        }
        let e = *self.zones.get(&z)?.edges.get(&c)?.get(&next)?;
        Some(Leg { to: e.to, frames: e.frames, node: e.node })
    }

    pub fn summary(&mut self) -> Stats {
        self.stats.cells = self.zones.values().map(|z| z.rep.len()).sum();
        self.stats.edges = self.zones.values().map(|z| z.n_edges).sum();
        self.stats.clone()
    }

    /// Writes every zone's graph to the directory (`OBRRT_ROAD=<dir>`).
    pub fn save(&self) {
        let Some(d) = &self.dir else { return };
        let _ = obrrt_core::sys::fs::create_dir_all(d);
        for (z, zr) in &self.zones {
            if zr.n_edges == 0 {
                continue;
            }
            let path = format!("{d}/L{}A{}.road", z.0, z.1);
            let tmp = format!("{path}.tmp{}", std::process::id());
            let Ok(file) = obrrt_core::sys::fs::File::create(&tmp) else { continue };
            let mut w = std::io::BufWriter::new(file);
            for (c, r) in &zr.rep {
                if r.old {
                    let _ = writeln!(w, "r {} {} {} {:.0} {:.0} {:.0} {}", c.0, c.1, c.2, r.pos[0], r.pos[1], r.pos[2], r.water);
                } else {
                    let _ = writeln!(w, "r {} {} {} {:.0} {:.0} {:.0} {} 1", c.0, c.1, c.2, r.pos[0], r.pos[1], r.pos[2], r.water);
                }
            }
            for (a, m) in &zr.edges {
                for (b, e) in m {
                    let _ = writeln!(w, "e {} {} {} {} {} {} {} {} {:.0} {:.0} {:.0}", a.0, a.1, a.2, b.0, b.1, b.2, e.frames, e.count, e.to[0], e.to[1], e.to[2]);
                }
            }
            drop(w);
            let _ = obrrt_core::sys::fs::rename(&tmp, &path);
        }
    }
}
