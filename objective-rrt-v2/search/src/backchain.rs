//! Backward chaining (`OBRRT_BACKCHAIN=1`, off by default): re-reach a
//! success sooner from further up its branch.
//!
//! A state N added as a success of objective O is queued (the higher O's
//! weight, a frontier advance or a zone-tier record for successes, the
//! sooner; the most recent first among equals). When served, an ancestor A
//! 2..K generations above N is picked whose path to N is at most `cap`
//! frames, and O is attempted from A with the GA seeded by the path's
//! recorded inputs: replayed, they reach O again, so the attempt can only
//! find O as soon or sooner (the GA ranks met rollouts by their frame).
//! Only O's own success test counts; the result is merged like any other
//! attempt (soft dedupe lets an earlier arrival at N's place win). Generic:
//! no game knowledge here.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, OnceLock};

use anyhow::Result;
use obrrt_core::{GamePad, Objective, St};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::ga::Attempt;
use crate::store::Delta;
use crate::tree::Tree;

#[derive(Debug, Clone, Copy)]
pub struct Settings {
    /// Share of a round's plans (`OBRRT_BACKCHAIN_SHARE`, 0.1).
    pub share: f64,
    /// Ancestor generations above the target (`OBRRT_BACKCHAIN_KMIN`/`_KMAX`, 2..6).
    pub kmin: u32,
    pub kmax: u32,
    /// Most frames on the path from the ancestor to the target (`OBRRT_BACKCHAIN_CAP`, 900).
    pub cap: usize,
    /// Attempts per target (`OBRRT_BACKCHAIN_PER_TARGET`, 3).
    pub per_target: u32,
    /// Only successes of objectives worth more than this (`OBRRT_BACKCHAIN_MINW`, 1).
    pub min_weight: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { share: 0.1, kmin: 2, kmax: 6, cap: 900, per_target: 3, min_weight: 1.0 }
    }
}

static SETTINGS: OnceLock<Option<Settings>> = OnceLock::new();

/// Turns backward chaining on (`Some`) before the engine is made.
pub fn configure(s: Option<Settings>) {
    let _ = SETTINGS.set(s);
}

fn settings() -> Option<Settings> {
    *SETTINGS.get_or_init(|| None)
}

const QUEUE: usize = 512;

/// Level, area, stars, keys and collected set: an earlier success counts
/// only if it ends here as the target did.
type Tier = (u16, u8, i16, u8, [u8; 26]);

fn tier(st: &St) -> Tier {
    (st.level, st.area, st.stars, st.keys, st.collected())
}

struct Target {
    obj: Arc<dyn Objective>,
    prio: u64,
    seq: u64,
    left: u32,
    tried: Vec<u32>,
}

/// What a planned backchain attempt came from.
struct Pending {
    seed: Vec<GamePad>,
    target: u32,
    start: u32,
}

/// What a backchain attempt did (taken before it is merged).
pub struct Outcome {
    seed: Option<(bool, u32)>,
    success: bool,
    end: Option<(u32, Tier)>,
}

pub fn outcome(att: &Attempt<Delta>) -> Outcome {
    let s = att.segments.first();
    Outcome {
        seed: att.ga.and_then(|g| g.seed),
        success: s.is_some_and(|s| s.success),
        end: s.and_then(|s| s.end.as_ref()).map(|(st, _)| (st.frame, tier(st))),
    }
}

#[derive(Default, serde::Serialize)]
struct Stats {
    queued: u64,
    dropped: u64,
    no_ancestor: u64,
    /// Ancestors (in kmin..kmax) passed over: beyond the frame cap,
    /// retired, modal or dead, a key or star gained on the way, O not
    /// startable there, already tried.
    rej_cap: u64,
    rej_retired: u64,
    rej_modal: u64,
    rej_progress: u64,
    rej_criteria: u64,
    rej_tried: u64,
    planned: u64,
    seed_met: u64,
    seed_missed: u64,
    successes: u64,
    /// Successes ending in the target's tier sooner than it.
    earlier: u64,
    /// ... and kept in the tree.
    earlier_added: u64,
    frames_saved: u64,
    /// Earlier and in the target's 512-unit cell.
    same_cell: u64,
}

pub struct Backchain {
    on: Option<Settings>,
    queue: BTreeSet<(u64, u64, u32)>,
    targets: HashMap<u32, Target>,
    deferred: Vec<u32>,
    tier_best: HashMap<Tier, u32>,
    seq: u64,
    /// Plans left this round, and plans made so far this round (counted by
    /// the engine), which index `pending`.
    pub budget: usize,
    pub made: usize,
    pending: HashMap<usize, Pending>,
    rng: SmallRng,
    stats: Stats,
    /// (new state, target, frames saved).
    improved: Vec<(u32, u32, u32)>,
    by_kind: std::collections::BTreeMap<&'static str, (u64, u64, u64)>,
}

impl Backchain {
    pub fn new(seed: u64) -> Backchain {
        Backchain {
            on: settings(),
            queue: BTreeSet::new(),
            targets: HashMap::new(),
            deferred: Vec::new(),
            tier_best: HashMap::new(),
            seq: 0,
            budget: 0,
            made: 0,
            pending: HashMap::new(),
            rng: SmallRng::seed_from_u64(seed ^ 0x6261_636b_6368_6e21),
            stats: Stats::default(),
            improved: Vec::new(),
            by_kind: Default::default(),
        }
    }

    pub fn on(&self) -> bool {
        self.on.is_some()
    }

    pub fn begin_round(&mut self, round_plans: usize) {
        let Some(s) = self.on else { return };
        self.budget = ((round_plans as f64 * s.share).ceil() as usize).max(1);
        self.made = 0;
        self.pending.clear();
        for id in std::mem::take(&mut self.deferred) {
            if let Some(t) = self.targets.get(&id) {
                self.queue.insert((t.prio, t.seq, id));
            }
        }
    }

    /// On adding `id` as a success of `obj` (worth `weight`).
    pub fn note_success(&mut self, tree: &Tree, id: u32, obj: &Arc<dyn Objective>, weight: f64, advanced: bool) {
        let Some(s) = self.on else { return };
        let st = &tree.nodes[id as usize].st;
        if weight <= s.min_weight || obj.fixed().is_some() || st.modal() {
            return;
        }
        let t = tier(st);
        let record = self.tier_best.get(&t).is_none_or(|&f| st.frame < f);
        if record {
            self.tier_best.insert(t, st.frame);
        }
        let prio = (weight * if advanced { 4.0 } else { 1.0 } * if record { 2.0 } else { 1.0 }).to_bits();
        self.seq += 1;
        self.queue.insert((prio, self.seq, id));
        self.targets.insert(id, Target { obj: obj.clone(), prio, seq: self.seq, left: s.per_target, tried: Vec::new() });
        self.stats.queued += 1;
        if self.queue.len() > QUEUE {
            if let Some((_, _, j)) = self.queue.pop_first() {
                self.targets.remove(&j);
                self.stats.dropped += 1;
            }
        }
    }

    /// The next backchain attempt: its start and objective. Its seed is
    /// kept for plan number `made` of this round.
    pub fn next(&mut self, tree: &Tree) -> Option<(u32, Arc<dyn Objective>)> {
        let s = self.on?;
        while self.budget > 0 {
            let (_, _, id) = self.queue.pop_last()?;
            let n = &tree.nodes[id as usize];
            if n.retired {
                self.targets.remove(&id);
                continue;
            }
            let t = self.targets.get(&id).unwrap();
            // Candidate ancestors: generation, node, path frames.
            let mut cands = Vec::new();
            let mut frames = n.inputs.len();
            let mut at = n.parent;
            let mut between: Vec<u32> = Vec::new();
            for g in 1..=s.kmax {
                let Some(a) = at else { break };
                let an = &tree.nodes[a as usize];
                if g >= s.kmin {
                    let st = &mut self.stats;
                    if frames > s.cap {
                        st.rej_cap += 1;
                    } else if an.retired {
                        st.rej_retired += 1;
                    } else if an.st.modal() || !an.st.alive() {
                        st.rej_modal += 1;
                    } else if t.tried.contains(&a) {
                        st.rej_tried += 1;
                    } else if !between.iter().all(|&x| tree.nodes[x as usize].st.progress() == an.st.progress()) {
                        st.rej_progress += 1;
                    } else if !(t.obj.exists_in(&an.st) && t.obj.base_criteria(&an.st)) {
                        st.rej_criteria += 1;
                    } else {
                        cands.push(a);
                    }
                }
                frames += an.inputs.len();
                if frames > s.cap {
                    break;
                }
                between.push(a);
                at = an.parent;
            }
            if cands.is_empty() {
                self.targets.remove(&id);
                self.stats.no_ancestor += 1;
                continue;
            }
            let a = cands[self.rng.random_range(0..cands.len())];
            // The recorded inputs from A to N.
            let mut chain = vec![id];
            let mut x = id;
            while let Some(p) = tree.nodes[x as usize].parent.filter(|&p| p != a) {
                chain.push(p);
                x = p;
            }
            let seed: Vec<GamePad> = chain.iter().rev().flat_map(|&i| tree.nodes[i as usize].inputs.iter().copied()).collect();
            let t = self.targets.get_mut(&id).unwrap();
            t.tried.push(a);
            t.left -= 1;
            let obj = t.obj.clone();
            if t.left == 0 {
                self.targets.remove(&id);
            } else {
                self.deferred.push(id);
            }
            self.budget -= 1;
            self.stats.planned += 1;
            self.pending.insert(self.made, Pending { seed, target: id, start: a });
            return Some((a, obj));
        }
        None
    }

    /// The seed of this round's plan number `i`, if it is a backchain one.
    pub fn seed(&self, i: usize) -> Option<&[GamePad]> {
        self.pending.get(&i).map(|p| p.seed.as_slice())
    }

    pub fn is_pending(&self, i: usize) -> bool {
        self.pending.contains_key(&i)
    }

    /// After plan `i` was merged (`before` = node count before): stats.
    pub fn merged(&mut self, i: usize, tree: &Tree, before: usize, out: Option<Outcome>) {
        let Some(p) = self.pending.get(&i) else { return };
        let Some(out) = out else { return };
        let n = &tree.nodes[p.target as usize];
        let kind = n.via;
        let e = self.by_kind.entry(kind).or_default();
        e.0 += 1;
        match out.seed {
            Some((true, _)) => self.stats.seed_met += 1,
            _ => self.stats.seed_missed += 1,
        }
        if !out.success {
            return;
        }
        self.stats.successes += 1;
        let Some((frame, t)) = out.end else { return };
        if frame >= n.st.frame || t != tier(&n.st) {
            return;
        }
        self.stats.earlier += 1;
        e.1 += 1;
        let saved = n.st.frame - frame;
        self.stats.frames_saved += saved as u64;
        e.2 += saved as u64;
        if tree.nodes.len() > before {
            let new = before as u32;
            let c = |s: &St| (s.pos.map(|v| (v / 512.0).floor() as i32), s.zone());
            if c(&tree.nodes[new as usize].st) == c(&n.st) {
                self.stats.same_cell += 1;
            }
            self.stats.earlier_added += 1;
            self.improved.push((new, p.target, saved));
        }
        let _ = p.start;
    }

    /// `backchain.json`: the counts, per target kind (attempts, earlier,
    /// frames saved), and each kept earlier state with what grew from it
    /// against what grew from the target.
    pub fn write(&self, dir: &str, tree: &Tree) -> Result<()> {
        if self.on.is_none() {
            return Ok(());
        }
        let sub = |root: u32| -> (usize, i16, u32) {
            // Subtree size, most stars and the deepest generation below.
            let (mut size, mut stars, mut depth) = (0, tree.nodes[root as usize].st.stars, 0);
            let mut level = vec![root];
            let mut g = 0;
            while !level.is_empty() {
                let mut next = Vec::new();
                for x in level {
                    size += 1;
                    stars = stars.max(tree.nodes[x as usize].st.stars);
                    depth = g;
                    next.extend_from_slice(&tree.children[x as usize]);
                }
                level = next;
                g += 1;
            }
            (size, stars, depth)
        };
        let improved: Vec<_> = self
            .improved
            .iter()
            .map(|&(new, target, saved)| {
                let (ns, nst, nd) = sub(new);
                let (ts, tst, td) = sub(target);
                serde_json::json!({
                    "new": new, "target": target, "via": tree.nodes[target as usize].via,
                    "frame": tree.nodes[new as usize].st.frame, "saved": saved,
                    "new_subtree": ns, "new_stars": nst, "new_depth": nd,
                    "target_subtree": ts, "target_stars": tst, "target_depth": td,
                })
            })
            .collect();
        let continued = self.improved.iter().filter(|&&(new, _, _)| !tree.children[new as usize].is_empty()).count();
        let by_kind: serde_json::Map<String, serde_json::Value> = self
            .by_kind
            .iter()
            .map(|(k, v)| (k.to_string(), serde_json::json!({"attempts": v.0, "earlier": v.1, "frames_saved": v.2})))
            .collect();
        let j = serde_json::json!({
            "settings": format!("{:?}", self.on.unwrap()),
            "stats": self.stats, "queue": self.queue.len(), "continued": continued,
            "by_kind": by_kind, "improved": improved,
        });
        obrrt_core::sys::fs::write(format!("{dir}/backchain.json"), serde_json::to_string_pretty(&j)?)?;
        let s = &self.stats;
        println!(
            "backchain: planned {} seed met {}/{} successes {} earlier {} (kept {}, continued {}) saved {} frames",
            s.planned,
            s.seed_met,
            s.seed_met + s.seed_missed,
            s.successes,
            s.earlier,
            s.earlier_added,
            continued,
            s.frames_saved
        );
        Ok(())
    }
}
