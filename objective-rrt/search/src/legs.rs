//! Leg reuse (`OBRRT_LEG_REUSE=1|noact`, off by default; needs
//! `OBRRT_REROUTE=1`).
//!
//! A *spawn* is a state the game puts Mario into at a fixed point however he
//! got there. It is found from the tree alone, with no list of spawn kinds:
//!
//! * discontinuity: the edge into it changed zone (level, area), left a
//!   modal stretch (dialog, menu, act select), or moved Mario much further
//!   than his speed allows (a warp);
//! * convergence: its exact movement key (zone, act, 1-unit position,
//!   action, speed, facing, modal state) was reached by two nodes with other
//!   parents and other full keys (another progress, coins).
//!
//! A *leg* runs from a spawn to the next destination below it: a zone change
//! or a milestone (more stars or keys). The table keeps the fastest leg per
//! (spawn key, destination). When a node with a spawn key appears (any
//! progress) every known leg of that key is queued onto it, and a new or
//! faster leg is queued onto the copies of its spawn already in the tree. A
//! leg replays edge by edge as [`KIND_LEG`] grafts: each kept edge queues
//! the next one from its new node, so partial progress is kept, and the end
//! is a new zone-entry or milestone node the other grafts cascade from.
//!
//! The spawn key carries the act (objects differ by act); `noact` drops it.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};

use obrrt_core::St;

use crate::reroute::{move_key, Graft, Reroute, KIND_LEG};
use crate::tree::Tree;

/// Edges walked up from a destination looking for its spawn.
const LEG_MAX_EDGES: usize = 64;
/// Copies of a spawn a new or faster leg is queued onto at most.
const LEG_FANOUT: usize = 32;

#[derive(Clone, Copy, Debug)]
struct Leg {
    s: u32,
    d: u32,
    frames: u32,
}

#[derive(Default, Clone, Copy, Debug, serde::Serialize)]
pub struct LegStats {
    pub spawns_disc: u64,
    pub spawns_conv: u64,
    pub legs: u64,
    pub improved: u64,
    pub queued: u64,
    /// Leg replays that reached the leg's end (every edge kept).
    pub done: u64,
    /// Frames the leg's end was reached sooner than the original, summed
    /// over done chains whose copy was sooner.
    pub saved: u64,
    /// Done chains whose copy had more stars or keys than the original leg
    /// (a segment reused at a later star count).
    pub done_progress: u64,
}

pub struct Legs {
    act: bool,
    lkeys: Vec<u64>,
    by_key: HashMap<u64, Vec<u32>>,
    /// Per exact key: the first node's full key and parent (convergence).
    first: HashMap<u64, (u64, Option<u32>)>,
    spawns: HashMap<u64, u8>,
    table: HashMap<u64, BTreeMap<u64, Leg>>,
    pub stats: LegStats,
    /// By zone (level, area): spawns found and legs recorded (the spawn's
    /// zone), leg edges queued (the copy's zone), legs done (the end's zone).
    by_zone: BTreeMap<(u16, u8), [u64; 4]>,
}

fn hash<T: Hash>(t: &T) -> u64 {
    let mut h = DefaultHasher::new();
    t.hash(&mut h);
    h.finish()
}

impl Legs {
    pub fn from_env() -> Option<Legs> {
        let act = match std::env::var("OBRRT_LEG_REUSE").as_deref() {
            Ok("1") | Ok("act") => true,
            Ok("noact") => false,
            Ok("") | Ok("0") | Err(_) => return None,
            Ok(v) => panic!("OBRRT_LEG_REUSE={v}: expected 0, 1 or noact"),
        };
        Some(Legs {
            act,
            lkeys: Vec::new(),
            by_key: HashMap::new(),
            first: HashMap::new(),
            spawns: HashMap::new(),
            table: HashMap::new(),
            stats: LegStats::default(),
            by_zone: BTreeMap::new(),
        })
    }

    /// The exact movement key: zone, act (unless `noact`), 1-unit position,
    /// action id, speed, facing (1/64 turn), modal state. No progress.
    fn key(&self, st: &St) -> u64 {
        hash(&(
            st.level,
            st.area,
            if self.act { st.act } else { 0 },
            [st.pos[0].round() as i32, st.pos[1].round() as i32, st.pos[2].round() as i32],
            st.action & 0x1ff,
            st.forward_vel.round() as i32,
            (st.yaw as u16) >> 10,
            (st.dialog, st.menu_mode, st.act_select, st.transition != 0),
        ))
    }

    fn zone_mut(&mut self, z: (u16, u8)) -> &mut [u64; 4] {
        self.by_zone.entry(z).or_default()
    }

    pub fn summary(&self) -> serde_json::Value {
        let mut v = serde_json::to_value(self.stats).unwrap();
        v["act"] = self.act.into();
        v["spawn_keys"] = self.spawns.len().into();
        v["table"] = self.table.values().map(|m| m.len()).sum::<usize>().into();
        let z: serde_json::Map<String, serde_json::Value> =
            self.by_zone.iter().map(|(k, c)| (format!("{}:{}", k.0, k.1), serde_json::json!({"spawns": c[0], "legs": c[1], "queued": c[2], "done": c[3]}))).collect();
        v["by_zone"] = serde_json::Value::Object(z);
        v
    }
}

/// The node right below `top` on the path down to `d` (`d` below `top`).
fn below(tree: &Tree, top: u32, d: u32) -> Option<u32> {
    let mut at = d;
    loop {
        let p = tree.nodes[at as usize].parent?;
        if p == top {
            return Some(at);
        }
        at = p;
    }
}

fn live(tree: &Tree, f: u32) -> bool {
    f == 0 || !tree.nodes[f as usize].retired || (tree.sparse.is_some() && tree.can_anchor(f))
}

impl Reroute {
    /// Leg reuse for node `id`, just noted (see the module doc).
    pub(crate) fn legs_note(&mut self, id: u32, st: &St, tree: &Tree, hot: &HashSet<u32>, margin: u32, depth: u16) {
        let full = self.keys[id as usize];
        let parent = tree.nodes[id as usize].parent;
        let mut queue: Vec<(Graft, bool)> = Vec::new();
        {
            let l = self.legs.as_mut().unwrap();
            let lk = l.key(st);
            if l.lkeys.len() <= id as usize {
                l.lkeys.resize(id as usize + 1, u64::MAX);
            }
            l.lkeys[id as usize] = lk;
            l.by_key.entry(lk).or_default().push(id);
            // Spawn detection.
            let mut reason = 0u8;
            if let Some(p) = parent {
                let ps = &tree.nodes[p as usize].st;
                let frames = tree.nodes[id as usize].inputs.len().max(1) as f32;
                let d = ((ps.pos[0] - st.pos[0]).powi(2) + (ps.pos[1] - st.pos[1]).powi(2) + (ps.pos[2] - st.pos[2]).powi(2)).sqrt();
                let vmax = ps.forward_vel.abs().max(st.forward_vel.abs()) + 50.0;
                if ps.zone() != st.zone() || (ps.modal() && !st.modal()) || d > 2.0 * vmax * frames {
                    reason = 1;
                }
            }
            match l.first.get(&lk) {
                None => {
                    l.first.insert(lk, (full, parent));
                }
                Some(&(fk, fp)) => {
                    if reason == 0 && fk != full && fp != parent {
                        reason = 2;
                    }
                }
            }
            if reason > 0 && !l.spawns.contains_key(&lk) {
                l.spawns.insert(lk, reason);
                if reason == 1 {
                    l.stats.spawns_disc += 1;
                } else {
                    l.stats.spawns_conv += 1;
                }
                l.zone_mut(st.zone())[0] += 1;
            }
            // A copy of a spawn: every known leg from it.
            if l.spawns.contains_key(&lk) {
                if let Some(m) = l.table.get(&lk) {
                    for leg in m.values() {
                        // Only a copy the leg can give something new: other
                        // progress (full key), or sooner by the margin.
                        if leg.s == id || !(self.keys[leg.s as usize] != full || st.frame + margin <= tree.nodes[leg.s as usize].st.frame) {
                            continue;
                        }
                        if let Some(first) = below(tree, leg.s, leg.d) {
                            queue.push((Graft { from: id, child: first, depth, kind: KIND_LEG, leg: leg.d }, hot.contains(&id) || hot.contains(&leg.s)));
                        }
                    }
                }
            }
            // A destination: the leg from its spawn.
            if let Some(p) = parent {
                let ps = &tree.nodes[p as usize].st;
                let starred = st.stars > ps.stars;
                if ps.zone() != st.zone() || starred || st.keys > ps.keys {
                    let mut at = p;
                    let mut found = None;
                    for _ in 0..LEG_MAX_EDGES {
                        if l.lkeys.get(at as usize).is_some_and(|k| l.spawns.contains_key(k)) {
                            found = Some(at);
                            break;
                        }
                        match tree.nodes[at as usize].parent {
                            Some(q) => at = q,
                            None => break,
                        }
                    }
                    if let Some(s) = found {
                        let sst = &tree.nodes[s as usize].st;
                        let dest = hash(&(st.zone(), st.stars - sst.stars, st.keys as i16 - sst.keys as i16));
                        let frames = st.frame - sst.frame;
                        let sk = l.lkeys[s as usize];
                        let m = l.table.entry(sk).or_default();
                        let better = match m.get(&dest) {
                            None => {
                                l.stats.legs += 1;
                                true
                            }
                            Some(old) => {
                                let b = frames < old.frames;
                                l.stats.improved += b as u64;
                                b
                            }
                        };
                        if better {
                            m.insert(dest, Leg { s, d: id, frames });
                            if let Some(first) = below(tree, s, id) {
                                let (sfull, sframe) = (self.keys[s as usize], sst.frame);
                                let keys = &self.keys;
                                let copies: Vec<u32> = l.by_key.get(&sk).map_or(Vec::new(), |v| {
                                    v.iter()
                                        .copied()
                                        .filter(|&c| c != s && live(tree, c) && (keys[c as usize] != sfull || tree.nodes[c as usize].st.frame + margin <= sframe))
                                        .take(LEG_FANOUT)
                                        .collect()
                                });
                                for c in copies {
                                    queue.push((Graft { from: c, child: first, depth, kind: KIND_LEG, leg: id }, hot.contains(&c) || hot.contains(&s)));
                                }
                            }
                            let z = sst.zone();
                            l.zone_mut(z)[1] += 1;
                        }
                    }
                }
            }
        }
        for (g, h) in queue {
            let before = self.stats.queued;
            self.push(g, h);
            if self.stats.queued > before {
                let z = tree.nodes[g.from as usize].st.zone();
                let l = self.legs.as_mut().unwrap();
                l.stats.queued += 1;
                l.zone_mut(z)[2] += 1;
            }
        }
    }

    /// A kept leg edge `g` (now node `id`): queues the next edge, or counts
    /// the leg as done.
    pub(crate) fn leg_next(&mut self, g: Graft, id: u32, tree: &Tree, hot: &HashSet<u32>, depth: u16) {
        if g.child == g.leg {
            // Counted by the leg's end (`done` per zone is by destination).
            let (e, d) = (&tree.nodes[id as usize].st, &tree.nodes[g.leg as usize].st);
            let l = self.legs.as_mut().unwrap();
            l.stats.done += 1;
            l.stats.saved += d.frame.saturating_sub(e.frame) as u64;
            l.stats.done_progress += (e.stars > d.stars || e.keys > d.keys) as u64;
            l.zone_mut(d.zone())[3] += 1;
            return;
        }
        if let Some(next) = below(tree, g.child, g.leg) {
            let h = hot.contains(&g.child) || hot.contains(&next);
            self.push(Graft { from: id, child: next, depth, kind: KIND_LEG, leg: g.leg }, h);
        }
    }

    /// Whether leg edge `g`'s replayed end `st` is kept: the child's movement
    /// key, progress not below the start's, no more frames than the child
    /// took, and no state of its full key at or before it.
    pub(crate) fn leg_accept<C: Hash>(&mut self, g: Graft, st: &St, cell: &C, tree: &Tree) -> bool {
        let child = &tree.nodes[g.child as usize];
        if move_key(st) != move_key(&child.st) {
            self.stats.diverged += 1;
            return false;
        }
        let from = &tree.nodes[g.from as usize].st;
        if st.stars < from.stars || st.keys < from.keys {
            self.stats.diverged += 1;
            return false;
        }
        let parent = child.parent.map_or(0, |p| tree.nodes[p as usize].st.frame);
        if st.frame.saturating_sub(from.frame) > child.st.frame.saturating_sub(parent) {
            self.stats.not_faster += 1;
            return false;
        }
        let k = self.key_of(st, cell);
        if self.full_key_sooner(k, tree, st.frame) {
            self.stats.duplicate += 1;
            return false;
        }
        self.set_accepted(g);
        true
    }
}
