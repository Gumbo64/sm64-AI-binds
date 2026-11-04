//! The archive: one cell per distinct key (the fastest state found with it),
//! the input tree that reaches every cell, the checkpoint saves the states are
//! rebuilt from, the stage groups cells are sampled in, and the Bayesian
//! estimate of each cell's chance to pay off.
//!
//! Storage, for millions of cells:
//! - a cell's key is stored once (the index maps a key's hash to cell ids);
//! - every node's inputs live in one flat arena;
//! - only checkpoint nodes hold a save (see `Config::replay_max`); the others
//!   are rebuilt from their nearest saved ancestor (`base`) by replaying.

use std::collections::HashMap;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{BufWriter, Write};

use anyhow::Result;
use rand::Rng;
use rand_distr::{Beta, Distribution};
use sm64_binds::{GamePad, SM64Game, SaveState};

use crate::store::Packer;

/// When a cell's best is replaced by a faster path, its evidence is scaled by
/// this and its prior re-derived from the new source: the posterior widens
/// towards the prior, so the improved cell is re-expanded soon and the speedup
/// cascades to the cells after it.
const RESET: f32 = 0.25;

/// A key entry: (objective id, field, value).
pub type Entry = (u16, u8, i32);

pub struct Cell {
    pub key: Box<[Entry]>,
    /// The node whose end is this cell's state.
    pub node: u32,
    /// Frames from power-on to that state.
    pub frames: u32,
    pub visits: f32,
    pub succ: f32,
    pub a0: f32,
    pub b0: f32,
    /// Its sampling group (an objective id; 0 is the root).
    pub group: u16,
    /// A door animation, dialog or menu: expanded with inputs that press through it.
    pub busy: bool,
    /// Mario's position in the state (for the viewer's spatial index).
    pub pos: [i16; 3],
}

impl Cell {
    /// Posterior mean of P(an expansion adds or improves another cell).
    pub fn p(&self) -> f32 {
        (self.a0 + self.succ) / (self.a0 + self.b0 + self.visits)
    }

    fn sample(&self, rng: &mut impl Rng) -> f32 {
        let a = self.a0 + self.succ;
        let b = self.b0 + self.visits - self.succ;
        Beta::new(a.max(1e-3), b.max(1e-3)).unwrap().sample(rng)
    }
}

/// A sampling group: the cells of one stage of the route. Cells compete only
/// within their group; groups compete by how often expanding from them reaches
/// new ground (a new cell in the group, or in a later one).
pub struct Group {
    pub cells: Vec<u32>,
    pub visits: f32,
    pub succ: f32,
    pub best_frames: u32,
}

impl Group {
    fn new() -> Group {
        Group { cells: Vec::new(), visits: 0.0, succ: 0.0, best_frames: u32::MAX }
    }

    /// Thompson draw of its progress rate, from a uniform Beta(1, 1) prior.
    fn sample(&self, rng: &mut impl Rng) -> f32 {
        Beta::new(1.0 + self.succ, (1.0 + self.visits - self.succ).max(1e-3)).unwrap().sample(rng)
    }
}

/// One record of tree.bin: the inputs after the parent's (in the arena at
/// `start..start + len`). `base` is the nearest ancestor-or-self holding a
/// save and `replay` the frames from it; `users` counts the nodes rebuilt from
/// this one's save. `zone` is level * 256 + area at its end; `entry` the node
/// where the path entered that zone.
pub struct Node {
    pub parent: i32,
    start: u64,
    len: u32,
    pub base: u32,
    pub replay: u32,
    pub users: u32,
    pub zone: u32,
    pub entry: u32,
}

/// A state worth keeping, `t + 1` frames into a rollout; `entered` if the zone
/// changed since the previous one (or the rollout's start).
pub struct Found {
    pub key: Vec<Entry>,
    pub t: usize,
    /// Present when this state is to be a checkpoint.
    pub save: Option<Box<[u8]>>,
    pub group: u16,
    pub busy: bool,
    pub zone: u32,
    pub entered: bool,
    /// Mario's position there.
    pub pos: [i16; 3],
}

/// Inputs played from cell `src`, whose state was node `node` at `frames`;
/// `p` is the cell's estimate when it was picked.
pub struct Rollout {
    pub src: usize,
    pub node: usize,
    pub frames: u32,
    pub p: f32,
    pub inputs: Vec<GamePad>,
    pub found: Vec<Found>,
}

/// What merging a rollout did.
#[derive(Default, Clone, Copy)]
pub struct Merged {
    /// Cells added or improved, other than the source.
    pub changed: usize,
    /// Cells added (not just improved), anywhere.
    pub added: usize,
    /// Cells whose best was replaced by a faster state.
    pub improved: usize,
    /// Reached new ground for the source's group (a new cell in it or a later one).
    pub advanced: bool,
}

impl Merged {
    pub fn add(&mut self, o: &Merged) {
        self.changed += o.changed;
        self.added += o.added;
        self.improved += o.improved;
        self.advanced |= o.advanced;
    }
}

enum Slot {
    One(u32),
    Many(Vec<u32>),
}

fn key_hash(key: &[Entry]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut h);
    h.finish()
}

pub struct Archive {
    pub cells: Vec<Cell>,
    index: HashMap<u64, Slot>,
    pub nodes: Vec<Node>,
    inputs: Vec<GamePad>,
    saves: HashMap<u32, Box<[u8]>>,
    save_bytes: usize,
    /// Nodes replaced during the current batch: their saves are freed once the
    /// batch is merged, if nothing is rebuilt from them.
    pending_free: Vec<u32>,
    /// Cells whose best was replaced by a faster path, in all.
    pub replaced: u64,
    /// Drop rollouts whose source cell was replaced earlier in the same batch.
    pub drop_stale: bool,
    /// Rollouts dropped that way.
    pub stale: u64,
    /// Sampling groups by objective id, and the route order of each.
    pub groups: Vec<Group>,
    rank: Vec<u16>,
    /// Sample a group first, then a cell in it (else all cells at once).
    pub grouped: bool,
    /// Fewest frames of a cell with an entry from each objective id.
    best: Vec<u32>,
    /// Zone entry nodes, by zone (for input sharing).
    entries: HashMap<u32, Vec<u32>>,
    packer: Packer,
    /// Speed mode (`OSS_CORRIDOR`): states outside it aren't kept.
    pub corridor: Option<crate::speed::Corridor>,
    /// Speed mode: each node's child nodes (for propagation; None = not kept).
    kids: Option<Vec<Vec<u32>>>,
    /// Speed mode: log replacements (cell, its old node) in `log`.
    pub track: bool,
    pub log: Vec<(u32, u32)>,
}

impl Archive {
    /// Node 0 and cell 0 are the power-on state, with no inputs. Saves are
    /// stored as their differences from `reference`.
    pub fn new(power_on: &SaveState, reference: &SaveState, key: Vec<Entry>, zone: u32) -> Archive {
        let mut a = Archive {
            cells: Vec::new(),
            index: HashMap::new(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            saves: HashMap::new(),
            save_bytes: 0,
            pending_free: Vec::new(),
            replaced: 0,
            drop_stale: std::env::var("OSS_KEEP_STALE").is_err(),
            stale: 0,
            groups: vec![Group::new()],
            rank: vec![0],
            grouped: true,
            best: Vec::new(),
            entries: HashMap::new(),
            packer: Packer::new(reference, zone),
            corridor: None,
            kids: None,
            track: false,
            log: Vec::new(),
        };
        let save = a.packer.pack(power_on, zone);
        a.put_save(0, save);
        a.nodes.push(Node { parent: -1, start: 0, len: 0, base: 0, replay: 0, users: 0, zone, entry: 0 });
        a.entries.entry(zone).or_default().push(0);
        a.add_cell(key.into(), 0, 0, 1.0, 1.0, 0, false, [0; 3]);
        a
    }

    /// Sets the sampling groups: `rank` is the route order by objective id (see
    /// `Tree::rank`; its length is the number of objectives), and `grouped`
    /// turns the two-level selection on. The power-on cell stays in group 0.
    pub fn with_groups(mut self, rank: Vec<u16>, grouped: bool) -> Archive {
        let n = rank.len().max(1);
        let first = std::mem::replace(&mut self.groups[0], Group::new());
        self.groups = (0..n).map(|_| Group::new()).collect();
        self.groups[0] = first;
        self.rank = if rank.is_empty() { vec![0] } else { rank };
        self.grouped = grouped;
        self.best = vec![u32::MAX; n];
        for i in 0..self.cells.len() {
            self.note_best(i);
        }
        self
    }

    // ---------- saves ----------

    /// `s` packed against the reference of its area `zone`.
    pub fn pack(&self, s: &SaveState, zone: u32) -> Box<[u8]> {
        self.packer.pack(s, zone)
    }

    pub fn unpack(&self, save: &[u8]) -> Result<SaveState> {
        self.packer.unpack(save)
    }

    fn put_save(&mut self, node: u32, save: Box<[u8]>) {
        self.save_bytes += save.len();
        self.saves.insert(node, save);
    }

    /// Nodes that hold a save.
    pub fn checkpoints(&self) -> usize {
        self.saves.len()
    }

    /// Bytes held in saves, including the per-area reference states.
    pub fn save_bytes(&self) -> usize {
        self.save_bytes + self.packer.references().1
    }

    /// Per-area reference states held: (count, bytes).
    pub fn references(&self) -> (usize, usize) {
        self.packer.references()
    }

    /// Frees the saves of the nodes replaced since the last call that nothing
    /// is rebuilt from. Called once a batch of rollouts is merged: a rollout of
    /// the batch may have started from such a node and rebuild its new nodes
    /// from it.
    pub fn free_pending(&mut self) {
        for node in std::mem::take(&mut self.pending_free) {
            if self.nodes[node as usize].users == 0
                && let Some(s) = self.saves.remove(&node)
            {
                self.save_bytes -= s.len();
            }
        }
    }

    /// Makes `game` the state at the end of `node`: its nearest saved
    /// ancestor, then the inputs after it.
    pub fn restore(&self, game: &mut SM64Game, node: usize) -> Result<()> {
        let base = self.nodes[node].base;
        let save = self.saves.get(&base).ok_or_else(|| anyhow::anyhow!("node {node}: its base {base} has no save"))?;
        game.load_state(&self.unpack(save)?)?;
        let mut chain = Vec::new();
        let mut i = node;
        while i != base as usize {
            chain.push(i);
            i = self.nodes[i].parent as usize;
        }
        for &n in chain.iter().rev() {
            for &pad in self.inputs(n) {
                game.step_game(pad)?;
            }
        }
        Ok(())
    }

    // ---------- the tree ----------

    /// The inputs of `node` (after its parent's).
    pub fn inputs(&self, node: usize) -> &[GamePad] {
        let n = &self.nodes[node];
        &self.inputs[n.start as usize..n.start as usize + n.len as usize]
    }

    /// The inputs from power-on to the end of `node`.
    pub fn path(&self, node: usize) -> Vec<GamePad> {
        self.since(node, usize::MAX, |_| false)
    }

    /// The last `n` or so frames of inputs leading to the end of `node` (whole
    /// nodes, so a few more), oldest first.
    pub fn tail(&self, node: usize, n: usize) -> Vec<GamePad> {
        self.since(node, n, |_| false)
    }

    /// The inputs from `node`'s zone entry to `node`.
    pub fn since_entry(&self, node: usize) -> Vec<GamePad> {
        let entry = self.nodes[node].entry as usize;
        self.since(node, usize::MAX, |i| i == entry)
    }

    /// Inputs walking up from `node`, stopping at a node `stop` says (excluded),
    /// at the root, or once `max` frames are collected.
    fn since(&self, mut node: usize, max: usize, stop: impl Fn(usize) -> bool) -> Vec<GamePad> {
        let mut parts = Vec::new();
        let mut len = 0;
        while !stop(node) {
            parts.push(self.inputs(node));
            len += parts.last().unwrap().len();
            if len >= max || self.nodes[node].parent < 0 {
                break;
            }
            node = self.nodes[node].parent as usize;
        }
        parts.into_iter().rev().flatten().copied().collect()
    }

    /// Keeps each node's children from now on (and for the nodes so far).
    pub fn keep_kids(&mut self) {
        let mut k = vec![Vec::new(); self.nodes.len()];
        for (i, n) in self.nodes.iter().enumerate() {
            if n.parent >= 0 {
                k[n.parent as usize].push(i as u32);
            }
        }
        self.kids = Some(k);
    }

    /// The child nodes of `node` (empty unless [`Archive::keep_kids`]).
    pub fn kids(&self, node: usize) -> &[u32] {
        self.kids.as_ref().and_then(|k| k.get(node)).map_or(&[], |v| v.as_slice())
    }

    /// Whether a state may be kept at all: inside the corridor, if any.
    pub fn admits(&self, st: &obrrt_core::St) -> bool {
        self.corridor.as_ref().is_none_or(|c| c.admits(st))
    }

    /// Zone entry nodes of `zone`.
    pub fn entries(&self, zone: u32) -> &[u32] {
        self.entries.get(&zone).map_or(&[], |v| v.as_slice())
    }

    // ---------- cells ----------

    /// The cell with `key`, if any.
    pub fn find(&self, key: &[Entry]) -> Option<usize> {
        match self.index.get(&key_hash(key))? {
            Slot::One(i) => (*self.cells[*i as usize].key == *key).then_some(*i as usize),
            Slot::Many(v) => v.iter().map(|&i| i as usize).find(|&i| *self.cells[i].key == *key),
        }
    }

    /// Whether a state with `key` at `frames` would be new or better.
    pub fn wanted(&self, key: &[Entry], frames: u32) -> bool {
        self.find(key).is_none_or(|i| frames < self.cells[i].frames)
    }

    /// The fewest frames of a cell with an entry from any objective in `ids`.
    pub fn best_in(&self, ids: std::ops::Range<usize>) -> Option<u32> {
        let b = self.best.get(ids)?.iter().copied().min()?;
        (b != u32::MAX).then_some(b)
    }

    fn note_best(&mut self, i: usize) {
        let (key, frames) = (&self.cells[i].key, self.cells[i].frames);
        for &(o, _, _) in key.iter() {
            if let Some(b) = self.best.get_mut(o as usize) {
                *b = (*b).min(frames);
            }
        }
    }

    fn add_cell(&mut self, key: Box<[Entry]>, node: u32, frames: u32, a0: f32, b0: f32, group: u16, busy: bool, pos: [i16; 3]) -> usize {
        let i = self.cells.len();
        let h = key_hash(&key);
        self.index
            .entry(h)
            .and_modify(|s| match s {
                Slot::One(j) => *s = Slot::Many(vec![*j, i as u32]),
                Slot::Many(v) => v.push(i as u32),
            })
            .or_insert(Slot::One(i as u32));
        let group = group.min(self.groups.len() as u16 - 1);
        self.cells.push(Cell { key, node, frames, visits: 0.0, succ: 0.0, a0, b0, group, busy, pos });
        let g = &mut self.groups[group as usize];
        g.cells.push(i as u32);
        g.best_frames = g.best_frames.min(frames);
        self.note_best(i);
        i
    }

    /// Adds a cell or replaces a slower one, as a new node under `parent` with
    /// `inputs`. Returns the cell's index if it changed, and whether it is new.
    pub fn insert(&mut self, f: Found, frames: u32, parent: usize, inputs: &[GamePad], src: usize, entered: bool) -> Option<(usize, bool)> {
        if !self.wanted(&f.key, frames) {
            return None;
        }
        let node = self.nodes.len() as u32;
        let entry = if entered { node } else { self.nodes[parent].entry };
        let (base, replay) = match f.save {
            Some(save) => {
                self.put_save(node, save);
                (node, 0)
            }
            None => {
                let (b, r) = (self.nodes[parent].base, self.nodes[parent].replay);
                self.nodes[b as usize].users += 1;
                (b, r + inputs.len() as u32)
            }
        };
        let start = self.inputs.len() as u64;
        self.inputs.extend_from_slice(inputs);
        self.nodes.push(Node { parent: parent as i32, start, len: inputs.len() as u32, base, replay, users: 0, zone: f.zone, entry });
        if let Some(k) = self.kids.as_mut() {
            k.push(Vec::new());
            k[parent].push(node);
        }
        if entered {
            self.entries.entry(f.zone).or_default().push(node);
        }
        let m = self.cells[src].p();
        let (a0, b0) = (2.0 * m, 2.0 * (1.0 - m));
        if let Some(i) = self.find(&f.key) {
            let c = &mut self.cells[i];
            let old = std::mem::replace(&mut c.node, node);
            (c.frames, c.a0, c.b0, c.pos) = (frames, a0, b0, f.pos);
            c.visits *= RESET;
            c.succ *= RESET;
            self.pending_free.push(old);
            self.replaced += 1;
            if self.track {
                self.log.push((i as u32, old));
            }
            let g = &mut self.groups[self.cells[i].group as usize];
            g.best_frames = g.best_frames.min(frames);
            self.note_best(i);
            return Some((i, false));
        }
        Some((self.add_cell(f.key.into(), node, frames, a0, b0, f.group, f.busy, f.pos), true))
    }

    /// Adds a rollout's states, each node chained under the previous one kept,
    /// and credits its source: the cell succeeds if the rollout added or
    /// improved any cell but itself; its group if it added a cell in the group
    /// or a later one (improvements don't count there).
    pub fn merge(&mut self, r: Rollout) -> Merged {
        let src = r.src;
        match self.merge_part(r) {
            Some(m) => {
                self.credit(src, &m);
                m
            }
            None => Merged::default(),
        }
    }

    /// [`Archive::merge`] without the credit, for an expansion made of several
    /// rollouts from one source (a GA attempt's winner and its last
    /// generation's ends): merge each, sum the results ([`Merged::add`]) and
    /// [`Archive::credit`] the source once. None if the rollout is stale
    /// (its source cell was replaced earlier in the batch; dropped, no credit).
    pub fn merge_part(&mut self, r: Rollout) -> Option<Merged> {
        // The source cell was replaced by a faster state earlier in this batch:
        // the rollout started from the slower one, so what it found is mostly
        // slower too. Drop it, with no credit either way.
        if self.drop_stale && self.cells[r.src].node as usize != r.node {
            self.stale += 1;
            return None;
        }
        let (mut parent, mut start, mut entered) = (r.node, 0, false);
        let src_group = self.cells[r.src].group as usize;
        let mut m = Merged::default();
        for f in r.found {
            let (frames, t) = (r.frames + f.t as u32 + 1, f.t);
            entered |= f.entered;
            if let Some((i, added)) = self.insert(f, frames, parent, &r.inputs[start..=t], r.src, entered) {
                (parent, start, entered) = (self.nodes.len() - 1, t + 1, false);
                m.changed += (i != r.src) as usize;
                if added {
                    m.added += 1;
                    let g = self.cells[i].group as usize;
                    m.advanced |= g == src_group || self.rank[g] > self.rank[src_group];
                } else {
                    m.improved += 1;
                }
            }
        }
        Some(m)
    }

    /// Credits an expansion of `src` that did `m`: one visit, a success if it
    /// added or improved another cell; its group a success if it advanced.
    pub fn credit(&mut self, src: usize, m: &Merged) {
        let src_group = self.cells[src].group as usize;
        let c = &mut self.cells[src];
        c.visits += 1.0;
        c.succ += (m.changed > 0) as u8 as f32;
        let g = &mut self.groups[src_group];
        g.visits += 1.0;
        g.succ += m.advanced as u8 as f32;
    }

    /// Thompson sampling: with groups, first over the groups that have cells,
    /// then over `k` random cells of the chosen one; without, over `k` random
    /// cells of all of them.
    pub fn select(&self, rng: &mut impl Rng, k: usize) -> usize {
        let mut best = (0, f32::NEG_INFINITY);
        let mut consider = |i: usize, rng: &mut _| {
            let s = self.cells[i].sample(rng);
            if s > best.1 {
                best = (i, s);
            }
        };
        if !self.grouped || self.groups.len() <= 1 {
            for _ in 0..k {
                let i = rng.random_range(0..self.cells.len());
                consider(i, rng);
            }
        } else {
            let g = (self.groups.iter().enumerate())
                .filter(|(_, g)| !g.cells.is_empty())
                .map(|(i, g)| (i, g.sample(rng)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
                .0;
            let cells = &self.groups[g].cells;
            for _ in 0..k {
                let i = cells[rng.random_range(0..cells.len())] as usize;
                consider(i, rng);
            }
        }
        best.0
    }

    /// Tests: files `ids` under `key`'s hash, as if they collided.
    #[cfg(test)]
    pub(crate) fn force_slot(&mut self, key: &[Entry], ids: Vec<u32>) {
        self.index.insert(key_hash(key), Slot::Many(ids));
    }

    /// Tests: whether `node` holds a save.
    #[cfg(test)]
    pub(crate) fn has_save(&self, node: usize) -> bool {
        self.saves.contains_key(&(node as u32))
    }

    // ---------- output ----------

    /// Writes, each to a temporary file then renamed into place:
    /// - `summary.json`: exact per-area cell counts and first frames, and the
    ///   area-to-area transitions (cells whose parent cell is in another area);
    /// - `cells.meta`: an index of `cells.jsonl` so a server can serve parts of
    ///   it without parsing it (see [`write_meta`]);
    /// - `cells.jsonl`: one line per cell, with its parent (the cell of its
    ///   nearest ancestor node that is one, -1 for none);
    /// - `tree.bin`: u32 count, then per node i32 parent, u32 n and n raw
    ///   4-byte inputs.
    ///
    /// `cells.meta` and `summary.json` carry the byte size of the
    /// `cells.jsonl` they describe, and are renamed in before it: a reader
    /// that sees a different size has caught a write in progress.
    pub fn write(&self, dir: &str) -> Result<()> {
        self.write_annotated(dir, None)
    }

    /// [`Archive::write`], with extra fields on some cells.jsonl lines
    /// (speed mode: route frame, frames saved, times improved).
    pub fn write_annotated(&self, dir: &str, extra: Option<&dyn Fn(usize, &Cell) -> Option<serde_json::Map<String, serde_json::Value>>>) -> Result<()> {
        let mut cell_at = vec![-1i64; self.nodes.len()];
        for (id, c) in self.cells.iter().enumerate() {
            cell_at[c.node as usize] = id as i64;
        }
        let mut anc = vec![-1i64; self.nodes.len()];
        for (i, n) in self.nodes.iter().enumerate() {
            anc[i] = if cell_at[i] >= 0 { cell_at[i] } else if n.parent >= 0 { anc[n.parent as usize] } else { -1 };
        }
        let parents: Vec<i32> = (self.cells.iter())
            .map(|c| match self.nodes[c.node as usize].parent {
                p if p >= 0 => anc[p as usize] as i32,
                _ => -1,
            })
            .collect();
        let zones: Vec<u32> = self.cells.iter().map(|c| self.nodes[c.node as usize].zone).collect();

        let tmp_cells = format!("{dir}/cells.jsonl.tmp");
        let mut f = BufWriter::new(File::create(&tmp_cells)?);
        let mut offsets = Vec::with_capacity(self.cells.len());
        let mut at = 0u64;
        for (id, c) in self.cells.iter().enumerate() {
            let key: Vec<[i32; 3]> = c.key.iter().map(|&(o, f, v)| [o as i32, f as i32, v]).collect();
            let mut line = serde_json::json!({"id": id, "key": key, "node": c.node, "frames": c.frames,
                "visits": c.visits, "succ": c.succ, "p": c.p(), "parent": parents[id]});
            if let Some(m) = extra.and_then(|f| f(id, c)) {
                line.as_object_mut().unwrap().extend(m);
            }
            let line = line.to_string();
            offsets.push(at);
            f.write_all(line.as_bytes())?;
            f.write_all(b"\n")?;
            at += line.len() as u64 + 1;
        }
        drop(f);

        // Exact per-area totals and transitions.
        let mut areas: std::collections::BTreeMap<u32, (u64, u32)> = Default::default();
        let mut trans: std::collections::BTreeMap<(u32, u32), (u64, u32)> = Default::default();
        for (id, c) in self.cells.iter().enumerate() {
            let a = areas.entry(zones[id]).or_insert((0, u32::MAX));
            (a.0, a.1) = (a.0 + 1, a.1.min(c.frames));
            if let Ok(p) = usize::try_from(parents[id])
                && zones[p] != zones[id]
            {
                let t = trans.entry((zones[p], zones[id])).or_insert((0, u32::MAX));
                (t.0, t.1) = (t.0 + 1, t.1.min(c.frames));
            }
        }
        let summary = serde_json::json!({
            "cells_jsonl_bytes": at, "cells": self.cells.len(),
            "areas": areas.iter().map(|(z, (n, first))| serde_json::json!({"zone": z, "n": n, "first": first})).collect::<Vec<_>>(),
            "transitions": trans.iter().map(|((a, b), (n, first))| serde_json::json!({"from": a, "to": b, "n": n, "first": first})).collect::<Vec<_>>(),
        });
        fs::write(format!("{dir}/summary.json.tmp"), summary.to_string())?;
        write_meta(&format!("{dir}/cells.meta.tmp"), at, &offsets, &parents, &zones, &self.cells)?;
        write_grid(&format!("{dir}/cells.grid.tmp"), at, &zones, &self.cells)?;
        fs::rename(format!("{dir}/cells.grid.tmp"), format!("{dir}/cells.grid"))?;
        fs::rename(format!("{dir}/cells.meta.tmp"), format!("{dir}/cells.meta"))?;
        fs::rename(format!("{dir}/summary.json.tmp"), format!("{dir}/summary.json"))?;
        fs::rename(&tmp_cells, format!("{dir}/cells.jsonl"))?;

        let tmp = format!("{dir}/tree.bin.tmp");
        let mut f = BufWriter::new(File::create(&tmp)?);
        f.write_all(&(self.nodes.len() as u32).to_le_bytes())?;
        for (i, n) in self.nodes.iter().enumerate() {
            f.write_all(&n.parent.to_le_bytes())?;
            f.write_all(&n.len.to_le_bytes())?;
            for p in self.inputs(i) {
                f.write_all(&p.button.to_be_bytes())?;
                f.write_all(&[p.stick_x as u8, p.stick_y as u8])?;
            }
        }
        drop(f);
        fs::rename(&tmp, format!("{dir}/tree.bin"))?;
        Ok(())
    }
}

/// `cells.meta`, an index of `cells.jsonl` for readers that want parts of it:
/// the magic `OSSM`, u32 version 2, u64 cell count n, u64 the byte size of the
/// `cells.jsonl` it describes; then, as little-endian arrays of n: u64 each
/// line's byte offset, i32 the parent cell id (-1 none), u32 the area (level *
/// 256 + area), u32 the frames, and i16 x, y, z of Mario's position (3n i16,
/// interleaved).
pub fn write_meta(path: &str, jsonl_bytes: u64, offsets: &[u64], parents: &[i32], zones: &[u32], cells: &[Cell]) -> Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(b"OSSM")?;
    f.write_all(&2u32.to_le_bytes())?;
    f.write_all(&(offsets.len() as u64).to_le_bytes())?;
    f.write_all(&jsonl_bytes.to_le_bytes())?;
    for o in offsets {
        f.write_all(&o.to_le_bytes())?;
    }
    for p in parents {
        f.write_all(&p.to_le_bytes())?;
    }
    for z in zones {
        f.write_all(&z.to_le_bytes())?;
    }
    for c in cells {
        f.write_all(&c.frames.to_le_bytes())?;
    }
    for c in cells {
        for v in c.pos {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

/// The side of a `cells.grid` bucket, in game units.
pub const GRID: i32 = 256;

/// `cells.grid`, the cells by area and position, for spatial queries: the magic
/// `OSSG`, u32 version 1, u32 bucket size, u32 bucket count b, u64 cell count
/// n, u64 the byte size of the `cells.jsonl` it describes; then b buckets of
/// (u32 area, i16 bx, by, bz, u16 pad, u32 start, u32 count), sorted by area then
/// (bx, by, bz), with bx = floor(x / size); then n u32 cell ids, each bucket's
/// at `start..start + count` in id order.
pub fn write_grid(path: &str, jsonl_bytes: u64, zones: &[u32], cells: &[Cell]) -> Result<()> {
    let b = |v: i16| (v as i32).div_euclid(GRID) as i16;
    let mut order: Vec<u32> = (0..cells.len() as u32).collect();
    let key = |i: u32| {
        let p = cells[i as usize].pos;
        (zones[i as usize], b(p[0]), b(p[1]), b(p[2]), i)
    };
    order.sort_unstable_by_key(|&i| key(i));
    let mut buckets: Vec<(u32, i16, i16, i16, u32, u32)> = Vec::new();
    for (at, &i) in order.iter().enumerate() {
        let (z, x, y, w, _) = key(i);
        match buckets.last_mut() {
            Some(last) if (last.0, last.1, last.2, last.3) == (z, x, y, w) => last.5 += 1,
            _ => buckets.push((z, x, y, w, at as u32, 1)),
        }
    }
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(b"OSSG")?;
    f.write_all(&1u32.to_le_bytes())?;
    f.write_all(&(GRID as u32).to_le_bytes())?;
    f.write_all(&(buckets.len() as u32).to_le_bytes())?;
    f.write_all(&(cells.len() as u64).to_le_bytes())?;
    f.write_all(&jsonl_bytes.to_le_bytes())?;
    for (z, x, y, w, start, count) in &buckets {
        f.write_all(&z.to_le_bytes())?;
        for v in [*x, *y, *w, 0] {
            f.write_all(&v.to_le_bytes())?;
        }
        f.write_all(&start.to_le_bytes())?;
        f.write_all(&count.to_le_bytes())?;
    }
    for i in &order {
        f.write_all(&i.to_le_bytes())?;
    }
    Ok(())
}
