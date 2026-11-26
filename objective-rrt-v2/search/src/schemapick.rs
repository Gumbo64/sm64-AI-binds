//! Start selection by schema cells (`OBRRT_PICK=schema|schema_kin`; unset =
//! the kinship softmax of `Engine::pick_start`).
//!
//! The tree's nodes are grouped by an objective-scattershot schema's key
//! (`OBRRT_PICK_SCHEMA`, default ../objective-scattershot/schemas/reroute.json;
//! each node keyed once, the first time a pick sees it). Like a scattershot
//! cell, each cell is represented by its fastest state: its earliest-frame
//! live member (kept incrementally: new nodes are compared with it, and only
//! a cell whose representative was retired is recomputed). A pick chooses a
//! CELL, not a state, from every cell whose representative is an eligible
//! start (the filters of `pick_start` other than kinship: alive, not
//! retired, modal as the objective, `base_criteria`, heuristic within the
//! kind's cutoff, and at least b's keys, stars and collected set unless kept
//! for its work under way), so a pick costs O(cells), not O(nodes). A cell's
//! weight is the softmax of its representative's straight-line heuristic at
//! the objective's temperature, times a novelty term 1 / sqrt(1 + times the
//! cell was picked) (scattershot's cell-level selection, without its
//! Thompson optimism). The start is the representative, or with probability
//! `OBRRT_PICK_RAND` (0.1) a uniformly random eligible member of the cell.
//!
//! `schema` uses no kinship at all. `schema_kin` multiplies the weight of
//! cells holding an eligible member of b's kin set by `OBRRT_PICK_KIN_W` (4)
//! and, in such a cell, takes the earliest eligible kin member instead.
//!
//! The per-objective cell table is cached until the tree changes (node count
//! or frames), so the several picks of one objective in a planning round scan
//! the cells once. Decisions use only the tree, the counts and the engine's
//! seeded RNG, so runs stay deterministic.

use std::collections::HashMap;
use std::sync::Arc as Rc;
use obrrt_core::sys::Instant;

use anyhow::{Result, bail};
use obrrt_core::{Objective, St};
use rand::Rng;
use rand::rngs::SmallRng;

use crate::tree::Tree;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Schema,
    SchemaKin,
}

/// One cached eligible cell: (cell id, heuristic, representative).
type Row = (u32, f64, u32);
/// Cache key: (hash of the objective's key, b's keys, b's stars, hash of
/// b's collected set). Not the pool index: `Pool::expire` shifts those.
type CacheKey = (u64, u8, i16, u64);

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct Stats {
    pub picks: u64,
    /// Picks with no eligible cell.
    pub none: u64,
    /// Picks that took a random member instead of the earliest.
    pub random: u64,
    /// schema_kin: picks whose cell held a kin member.
    pub kin_cells: u64,
    pub cache_hits: u64,
    pub scans: u64,
    /// Over picks: eligible cells and their members (summed; mean = / picks;
    /// members may include retired ones not yet dropped).
    pub sum_cells: u64,
    pub sum_members: u64,
    /// Wall seconds in picks (report only, never decisions).
    pub secs: f64,
}

pub struct SchemaPick {
    pub mode: Mode,
    pub path: String,
    schema: Box<objective_scattershot::objective::Tree>,
    rand_p: f64,
    kin_w: f64,
    /// Each node's cell (keyed up to `node_cell.len()`).
    node_cell: Vec<u32>,
    index: HashMap<u64, u32>,
    /// Members of each cell, ids ascending (retired ones dropped when the
    /// cell's representative is recomputed).
    cells: Vec<Vec<u32>>,
    /// Each cell's representative: its earliest live member (u32::MAX: none).
    rep: Vec<u32>,
    /// Times each cell was picked.
    picked: Vec<u32>,
    cache: HashMap<CacheKey, Rc<(Vec<Row>, u64)>>,
    epoch: (usize, u64),
    pub stats: Stats,
    /// Stats at the previous report (for the per-report window).
    last: Stats,
}

/// `OBRRT_PICK` and its settings; None when unset (the kinship softmax).
pub fn from_env() -> Result<Option<SchemaPick>> {
    let rc = &obrrt_core::config::current().research;
    let mode = match rc.pick.as_deref() {
        None | Some("") | Some("kin") => return Ok(None),
        Some("schema") => Mode::Schema,
        Some("schema_kin") => Mode::SchemaKin,
        Some(v) => bail!("OBRRT_PICK={v}: expected schema or schema_kin"),
    };
    let path = rc.pick_schema.clone().unwrap_or_else(|| "../objective-scattershot/schemas/reroute.json".into());
    let schema = crate::reroute::load_schema(&path)?;
    Ok(Some(SchemaPick {
        mode,
        path,
        schema,
        rand_p: rc.pick_rand,
        kin_w: rc.pick_kin_w,
        node_cell: Vec::new(),
        index: HashMap::new(),
        cells: Vec::new(),
        rep: Vec::new(),
        picked: Vec::new(),
        cache: HashMap::new(),
        epoch: (usize::MAX, 0),
        stats: Stats::default(),
        last: Stats::default(),
    }))
}

/// What a pick needs from the engine besides the tree.
pub struct Ctx<'a> {
    pub obj: &'a dyn Objective,
    /// The pool entry (diagnostics only; indices shift on expiry).
    pub entry: usize,
    pub b: u32,
    pub cutoff: f64,
    /// OBRRT_ETA_LEARN: the kind's ETA factor on the rank (the cutoff stays
    /// on the raw ETA); 1 when off.
    pub eta_c: f64,
    pub rrt_star: bool,
    /// b's kin set (schema_kin only).
    pub kin: Option<Vec<u32>>,
    /// Changes whenever the tree may have: (nodes, frames).
    pub epoch: (usize, u64),
}

/// The objective-dependent filters of `pick_start` (b's aside): the start
/// rank (c × ETA plus start bias) if `i` passes, the cutoff on the pure ETA.
fn ok_for(tree: &Tree, i: u32, obj: &dyn Objective, cutoff: f64, c: f64, rrt_star: bool) -> Option<f64> {
    let n = &tree.nodes[i as usize];
    let s = &n.st;
    if n.retired || (rrt_star && n.dominated) || !s.alive() || s.modal() != obj.modal() || !obj.base_criteria(s) {
        return None;
    }
    let h = obrrt_core::start_eta(obj, s);
    let b = obj.start_bias(s);
    (h.is_finite() && h <= cutoff).then(|| {
        let h = h * c;
        if b == 0.0 { h } else { h + b }
    })
}

/// b's: at least its keys, stars and (OBRRT_RETIRE_SUBSET) collected set,
/// unless the state is kept for its work under way.
fn ok_for_b(tree: &Tree, i: u32, b: &St) -> bool {
    if tree.is_wip(i) {
        return true;
    }
    let s = &tree.nodes[i as usize].st;
    s.keys >= b.keys && s.stars >= b.stars
}

impl SchemaPick {
    /// Keys the nodes added since the last pick.
    fn sync(&mut self, tree: &Tree) {
        for i in self.node_cell.len()..tree.nodes.len() {
            let k = crate::reroute::schema_hash(&self.schema, &tree.nodes[i].st);
            let next = self.cells.len() as u32;
            let c = *self.index.entry(k).or_insert(next);
            if c == next {
                self.cells.push(Vec::new());
                self.rep.push(u32::MAX);
                self.picked.push(0);
            }
            self.cells[c as usize].push(i as u32);
            self.node_cell.push(c);
            let r = self.rep[c as usize];
            if !tree.nodes[i].retired && (r == u32::MAX || tree.nodes[i].st.frame < tree.nodes[r as usize].st.frame) {
                self.rep[c as usize] = i as u32;
            }
        }
        // Cells whose representative was retired: drop their retired members
        // and take the earliest left.
        for c in 0..self.cells.len() {
            let r = self.rep[c];
            if r == u32::MAX || !tree.nodes[r as usize].retired {
                continue;
            }
            let m = &mut self.cells[c];
            m.retain(|&i| !tree.nodes[i as usize].retired);
            self.rep[c] = m.iter().copied().min_by_key(|&i| (tree.nodes[i as usize].st.frame, i)).unwrap_or(u32::MAX);
        }
    }

    pub fn pick(&mut self, tree: &Tree, cx: Ctx, rng: &mut SmallRng) -> Option<u32> {
        let t0 = Instant::now();
        let out = self.pick_inner(tree, cx, rng);
        self.stats.secs += t0.elapsed().as_secs_f64();
        out
    }

    fn pick_inner(&mut self, tree: &Tree, cx: Ctx, rng: &mut SmallRng) -> Option<u32> {
        if cx.epoch != self.epoch {
            self.sync(tree);
            self.cache.clear();
            self.epoch = cx.epoch;
        }
        self.stats.picks += 1;
        let obj = cx.obj;
        let b_st = &tree.nodes[cx.b as usize].st;
        let hash = |k: &dyn Fn(&mut std::collections::hash_map::DefaultHasher)| {
            use std::hash::Hasher;
            let mut h = std::collections::hash_map::DefaultHasher::new();
            k(&mut h);
            h.finish()
        };
        use std::hash::Hash;
        let set_hash = 0u64;
        let ck = (hash(&|h| obj.key().hash(h)), b_st.keys, b_st.stars, set_hash);
        let table = match self.cache.get(&ck) {
            Some(t) => {
                self.stats.cache_hits += 1;
                t.clone()
            }
            None => {
                self.stats.scans += 1;
                let mut rows = Vec::new();
                let mut members = 0u64;
                for (c, &r) in self.rep.iter().enumerate() {
                    if r == u32::MAX {
                        continue;
                    }
                    let Some(h) = ok_for(tree, r, obj, cx.cutoff, cx.eta_c, cx.rrt_star) else { continue };
                    if !ok_for_b(tree, r, b_st) {
                        continue;
                    }
                    members += self.cells[c].len() as u64;
                    rows.push((c as u32, h, r));
                }
                let t = Rc::new((rows, members));
                self.cache.insert(ck, t.clone());
                t
            }
        };
        let (rows, members) = (&table.0, table.1);
        self.stats.sum_cells += rows.len() as u64;
        self.stats.sum_members += members;
        if rows.is_empty() {
            self.stats.none += 1;
            return None;
        }
        // schema_kin: per cell, its earliest eligible kin member.
        let mut kin_best: HashMap<u32, u32> = HashMap::new();
        if let (Mode::SchemaKin, Some(kin)) = (self.mode, &cx.kin) {
            for &i in kin {
                if (i as usize) >= self.node_cell.len()
                    || ok_for(tree, i, obj, cx.cutoff, cx.eta_c, cx.rrt_star).is_none()
                    || !ok_for_b(tree, i, b_st)
                {
                    continue;
                }
                let c = self.node_cell[i as usize];
                let e = kin_best.entry(c).or_insert(i);
                if tree.nodes[i as usize].st.frame < tree.nodes[*e as usize].st.frame {
                    *e = i;
                }
            }
        }
        let hmin = rows.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
        let t = obj.temperature().max(1e-9);
        let weights: Vec<f64> = rows
            .iter()
            .map(|&(c, h, _)| {
                let kin = if kin_best.contains_key(&c) { self.kin_w } else { 1.0 };
                (-(h - hmin) / t).exp() / (1.0 + self.picked[c as usize] as f64).sqrt() * kin
            })
            .collect();
        let total: f64 = weights.iter().sum();
        let mut at = rows.len() - 1;
        if total > 0.0 {
            let mut x = rng.random_range(0.0..total);
            for (j, w) in weights.iter().enumerate() {
                if x < *w {
                    at = j;
                    break;
                }
                x -= w;
            }
        }
        let (c, _, earliest) = rows[at];
        self.picked[c as usize] += 1;
        let kin_member = kin_best.get(&c).copied();
        if kin_member.is_some() {
            self.stats.kin_cells += 1;
        }
        if self.rand_p > 0.0 && rng.random::<f64>() < self.rand_p {
            let elig: Vec<u32> = self.cells[c as usize]
                .iter()
                .copied()
                .filter(|&i| ok_for(tree, i, obj, cx.cutoff, cx.eta_c, cx.rrt_star).is_some() && ok_for_b(tree, i, b_st))
                .collect();
            if !elig.is_empty() {
                self.stats.random += 1;
                return Some(elig[rng.random_range(0..elig.len())]);
            }
        }
        Some(kin_member.unwrap_or(earliest))
    }

    fn picked_stats(&self) -> (usize, u32) {
        (self.picked.iter().filter(|&&p| p > 0).count(), self.picked.iter().copied().max().unwrap_or(0))
    }

    /// One log line: totals, and means per pick since the previous report.
    pub fn line(&mut self) -> String {
        let (s, l) = (&self.stats, &self.last);
        let dp = (s.picks - l.picks).max(1) as f64;
        let (distinct, maxp) = self.picked_stats();
        let line = format!(
            "pick {:?}: {} cells ({} nodes keyed), {} picked ever (max {}); {} picks ({} none, {} random, {} kin), {} scans {} cache hits, {:.1}s; this report: {} picks, mean {:.0} cells {:.0} members per pick",
            self.mode, self.cells.len(), self.node_cell.len(), distinct, maxp, s.picks, s.none, s.random, s.kin_cells, s.scans, s.cache_hits, s.secs,
            s.picks - l.picks, (s.sum_cells - l.sum_cells) as f64 / dp, (s.sum_members - l.sum_members) as f64 / dp,
        );
        self.last = self.stats.clone();
        line
    }

    pub fn summary(&self) -> serde_json::Value {
        let (distinct, maxp) = self.picked_stats();
        let s = &self.stats;
        let p = s.picks.max(1) as f64;
        let mut top: Vec<(u32, u32)> = self.picked.iter().enumerate().filter(|(_, p)| **p > 0).map(|(c, &p)| (p, c as u32)).collect();
        top.sort_unstable_by(|a, b| b.cmp(a));
        top.truncate(10);
        serde_json::json!({
            "mode": match self.mode { Mode::Schema => "schema", Mode::SchemaKin => "schema_kin" },
            "schema": self.path, "rand_p": self.rand_p, "kin_w": self.kin_w,
            "cells": self.cells.len(), "nodes_keyed": self.node_cell.len(),
            "members": self.cells.iter().map(|m| m.len()).sum::<usize>(),
            "cells_picked": distinct, "max_cell_picks": maxp,
            "top_cells": top.iter().map(|&(p, c)| serde_json::json!({"cell": c, "picks": p, "live": self.cells[c as usize].len(),
                "first": self.cells[c as usize].first()})).collect::<Vec<_>>(),
            "stats": s,
            "mean_cells_per_pick": s.sum_cells as f64 / p, "mean_members_per_pick": s.sum_members as f64 / p,
        })
    }
}
