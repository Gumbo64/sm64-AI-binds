//! Human-played ("manual") nodes: the SM64Coin web miner's hybrid mode. A person plays from a node's
//! state in the visualiser; the inputs they played become a node of the tree like a searched one
//! (an edge of pads from its parent), with a boost that makes the search prefer to start from it
//! at first and fades as it is expanded. A child module of `engine` (it reaches the engine's
//! private state) so the additions stay out of engine.rs's own functions.
//!
//! * [`Engine::node_state`]: the game state (with the step hook's state, i.e. the block's
//!   perturbation) of a node, for the page to load into its visual game.
//! * [`Engine::insert_manual`] / [`Engine::insert_manual_at`]: adds the played segment, replayed
//!   headless through the same hooked game every search frame goes through, so what the verifier replays
//!   from the root is exactly what the page played (the proof check is the same function).
//! * the boost: [`Engine::manual_pick`] (the share of `b` draws that start from a boosted node) and
//!   [`Engine::manual_start_mult`] (its weight as a start); both fade as the node is used.
//! * [`Engine::set_pos_goal`]: the sphere bubble base states are drawn from (core/src/posgoal.rs).

use super::*;

/// The boosts of the manual nodes, and the segments already inserted (by hash).
#[derive(Default)]
pub(super) struct Manual {
    boost: HashMap<u32, f64>,
    seen: HashSet<u64>,
    /// Zones (level, area) some node of this tree is in, and how far `sync_zone_firsts` has scanned the nodes.
    zone_seen: HashSet<(u16, u8)>,
    zone_scan: usize,
}

/// What [`Engine::insert_manual`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManualInsert {
    /// The node's id.
    Added(u32),
    /// The same inputs were inserted before.
    Known,
    /// The state at the end is dead (Mario died, the game ended): nothing was added.
    Dead,
}

/// Share of `b` draws a manual node can take at most.
const MAX_SHARE: f64 = 0.75;
/// A boosted node's boost is multiplied by this each time it is drawn as `b`.
const DECAY: f64 = 0.97;
/// Below this boost a node is no longer special.
const FLOOR: f64 = 0.02;
/// How much a unit of boost multiplies a start's weight.
const START_GAIN: f64 = 3.0;

impl Engine {
    /// The game state of node `id` (with the hook's state).
    pub fn node_state(&mut self, id: u32) -> Result<crate::game::GameSave> {
        anyhow::ensure!((id as usize) < self.tree.nodes.len(), "no node {id}");
        let mut w = loop {
            if let Some(w) = self.workers.iter().find_map(|w| w.try_lock().ok()) {
                break w;
            }
            std::thread::yield_now();
        };
        let replayed = self.tree.load_into(id, &mut w.base, &self.bases)?;
        self.total_frames += replayed as u64;
        Ok(w.base.save_state())
    }

    /// The inputs from the root to node `id`.
    pub fn node_path(&self, id: u32) -> Vec<GamePad> {
        self.tree.path(id)
    }

    /// Adds `seg`, played from node `parent`'s state, as a node under `parent` (boosted by `weight`).
    pub fn insert_manual_at(&mut self, parent: u32, seg: &[GamePad], weight: f64) -> Result<ManualInsert> {
        anyhow::ensure!((parent as usize) < self.tree.nodes.len(), "no node {parent}");
        let mut pads = self.tree.path(parent);
        pads.extend_from_slice(seg);
        self.insert_manual(&pads, weight)
    }

    /// Adds the state reached by `pads` from the root (the page's played segment after the inputs of
    /// its parent's path), as `sync_import` adds another search's: the deepest node on the path is the
    /// parent, the rest replayed through the hooked game. Unlike a sync import it is added whatever
    /// its cell holds (a human's line is wanted even where the search got there sooner), tagged `manual`
    /// and boosted by `weight` (see [`Engine::manual_pick`]).
    pub fn insert_manual(&mut self, pads: &[GamePad], weight: f64) -> Result<ManualInsert> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for p in pads {
            (p.button, p.stick_x, p.stick_y).hash(&mut h);
        }
        if !self.manual.seen.insert(h.finish()) {
            return Ok(ManualInsert::Known);
        }
        let (mut at, mut off) = (0u32, 0usize);
        'down: loop {
            for &c in &self.tree.children[at as usize] {
                let e = &self.tree.nodes[c as usize].inputs;
                if !e.is_empty() && pads.len() - off >= e.len() && pads[off..off + e.len()].iter().zip(e).all(|(a, b)| (a.button, a.stick_x, a.stick_y) == (b.button, b.stick_x, b.stick_y)) {
                    (at, off) = (c, off + e.len());
                    continue 'down;
                }
            }
            break;
        }
        while at != 0 && (self.tree.nodes[at as usize].retired || !self.tree.has_save(at)) {
            let n = &self.tree.nodes[at as usize];
            off -= n.inputs.len();
            at = n.parent.unwrap_or(0);
        }
        if off == pads.len() {
            // Nothing beyond a node that exists: boost that one.
            self.manual.boost.insert(at, weight);
            return Ok(ManualInsert::Known);
        }
        let rest = &pads[off..];
        let frame = self.tree.nodes[at as usize].st.frame + rest.len() as u32;
        let (st, packed) = {
            let mut w = loop {
                if let Some(w) = self.workers.iter().find_map(|w| w.try_lock().ok()) {
                    break w;
                }
                std::thread::yield_now();
            };
            let base = self.tree.base_of(at);
            self.tree.load_into(at, &mut w.base, &self.bases)?;
            for p in rest {
                w.base.step_game(*p)?;
            }
            let st = read_state(&w.base, frame, true);
            (st, Delta::new(base, &self.bases[base as usize], &w.base.save_state()))
        };
        self.total_frames += rest.len() as u64;
        if !st.alive() {
            return Ok(ManualInsert::Dead);
        }
        let zone = st.zone();
        let base = match self.zone_base.get(&zone) {
            Some(&b) => b,
            None => {
                self.bases.push(crate::game::into_plain(packed.apply(&self.bases)?).into_bytes());
                let b = (self.bases.len() - 1) as u16;
                self.zone_base.insert(zone, b);
                b
            }
        };
        let delta = if packed.base == base { packed } else { Delta::new(base, &self.bases[base as usize], &packed.apply(&self.bases)?) };
        if self.zones.insert(zone) {
            println!("new zone {zone:?} via a manual node at node {}", self.tree.nodes.len());
        }
        let (modal, fr, progress) = (st.modal(), st.frame, st.progress());
        let c = cell(&st);
        let (id, adv) = self.tree.add(at, rest.to_vec(), st, delta, "manual");
        if self.cfg.new_eligible {
            self.ne_note(id, at);
        }
        self.note_best_at(id, progress, zone, fr, "manual")?;
        if modal {
            self.modal_queue.push(id);
        }
        self.cells.entry(c).or_default().push(id);
        self.sync_sent.insert(id);
        self.manual.boost.insert(id, weight);
        if adv {
            let (keys, stars) = self.tree.frontier_progress;
            let f = self.write_m64(id, &format!("frontier-k{keys}-s{stars}"))?;
            println!("FRONTIER keys {keys} stars {stars} at node {id} via a manual node: {f}");
        }
        Ok(ManualInsert::Added(id))
    }

    /// The boosted nodes and their boosts (for the page's display).
    pub fn manual_boosts(&self) -> Vec<(u32, f64)> {
        let mut v: Vec<(u32, f64)> = self.manual.boost.iter().map(|(&i, &w)| (i, w)).collect();
        v.sort_by_key(|x| x.0);
        v
    }

    /// With boosted nodes: a share of the `b` draws (`total / (total + 2)`, at most 75 %) starts from one,
    /// by boost; each draw multiplies the node's boost by 0.97, so it fades as the search expands from
    /// it. No random draw is made without boosted nodes, so a search without any is unchanged.
    pub(super) fn manual_pick(&mut self) -> Option<u32> {
        if self.manual.boost.is_empty() {
            return None;
        }
        let nodes = &self.tree.nodes;
        self.manual.boost.retain(|&i, w| *w >= FLOOR && !nodes[i as usize].retired);
        let mut items: Vec<(u32, f64)> = self.manual.boost.iter().map(|(&i, &w)| (i, w)).collect();
        if items.is_empty() {
            return None;
        }
        items.sort_by_key(|x| x.0);
        let total: f64 = items.iter().map(|x| x.1).sum();
        if !self.rng.random_bool((total / (total + 2.0)).min(MAX_SHARE)) {
            return None;
        }
        let b = weighted(&mut self.rng, &items)?;
        if let Some(w) = self.manual.boost.get_mut(&b) {
            *w *= DECAY;
        }
        Some(b)
    }

    /// A start's weight multiplier from its boost (1 for any other node).
    pub(super) fn manual_start_mult(&self, i: u32) -> f64 {
        1.0 + START_GAIN * self.manual.boost.get(&i).copied().unwrap_or(0.0)
    }

    /// The position goal's objectives in the pool: (entries, attempts, successes), from the `Mario::PosTime` entries made for the goal.
    pub fn pos_goal_stats(&self) -> (usize, u64, u64) {
        let mut t = (0usize, 0u64, 0u64);
        for e in self.pool.entries.iter().filter(|e| e.obj.kind() == "Mario::UserGoal") {
            t = (t.0 + 1, t.1 + e.attempts as u64, t.2 + e.successes as u64);
        }
        t
    }

    /// How strongly the goal pulls: the pool weight of its objective (a regular objective weighs 1) and the share of plans
    /// that start from the tree's state nearest the goal and run it (see `plan_user_goal`).
    pub fn set_pos_goal_strength(&mut self, _weight: f64, share: f64) {
        obrrt_core::posgoal::set_share(share);
    }

    /// The bubble's base state (OBRRT_POSGOAL_SHARE, the web miner's bubble strength): with probability `share` and a
    /// bubble holding live nodes, a b drawn like the ordinary frontier draw (a cell uniformly, then a state in it,
    /// soft-dedup weighted) but only from the nodes inside the bubble. No nodes inside, no bubble or no share: None,
    /// and no random draw is made when there is no bubble, so a search without one is unchanged.
    pub(super) fn plan_bubble(&mut self) -> Option<u32> {
        let g = obrrt_core::posgoal::get()?;
        let share = obrrt_core::posgoal::share();
        let nodes = &self.tree.nodes;
        let inside = |i: u32| {
            let n = &nodes[i as usize];
            n.st.zone() == g.zone && !n.retired && !n.st.modal() && n.st.alive() && obrrt_core::dist(n.st.pos, g.pos) <= g.radius
        };
        bubble_pick(&mut self.rng, &self.b_index, share, inside, |i| 0.5f64.powi(nodes[i as usize].stats.crowd as i32))
    }

    /// Tree sync's fast path: the first own node (not imported by sync) in each zone (level, area) this tree had not
    /// been in before, at most `k` per call (the rest next time; imported nodes mark their zone as known, so a worker
    /// does not offer what it was given). The caller offers them to the other workers at once instead of at the
    /// next round (a new area is worth more to the others than a better state of a place they all know).
    pub fn sync_zone_firsts(&mut self, k: usize) -> Vec<u32> {
        let mut out = Vec::new();
        while self.manual.zone_scan < self.tree.nodes.len() && out.len() < k {
            let i = self.manual.zone_scan;
            self.manual.zone_scan += 1;
            let n = &self.tree.nodes[i];
            if self.manual.zone_seen.insert(n.st.zone()) && i > 0 && n.via != "sync" && !n.retired && !self.sync_sent.contains(&(i as u32)) {
                out.push(i as u32);
            }
        }
        self.sync_sent.extend(out.iter().copied());
        out
    }

    /// The tree's state in the goal's zone nearest the goal: (distance, node id), or None without a goal or a state there.
    pub fn pos_goal_nearest(&self) -> Option<(f32, u32)> {
        let g = obrrt_core::posgoal::get()?;
        self.tree.nodes.iter().enumerate().filter(|(_, n)| n.st.zone() == g.zone && !n.retired).map(|(i, n)| (obrrt_core::dist(n.st.pos, g.pos), i as u32)).min_by(|a, b| a.0.total_cmp(&b.0))
    }

    /// The sphere bubble base states are drawn from (see `plan_bubble`): `zone` (level, area), centre `pos`, `radius`
    /// (clamped to 50..20000). None clears it.
    pub fn set_pos_goal(&mut self, goal: Option<((u16, u8), [f32; 3], f32)>) {
        use obrrt_core::posgoal::*;
        set(goal.map(|(zone, pos, r)| PosGoal { zone, pos, radius: if r.is_finite() { r.clamp(MIN_RADIUS, MAX_RADIUS) } else { DEFAULT_RADIUS } }));
    }
}

/// The bubble draw behind [`Engine::plan_bubble`]: with probability `share`, a b from the `b_index` cells' nodes for which `inside`
/// holds (a non-empty restricted cell uniformly, then a node in it by `weight`); None without a draw or without nodes inside.
fn bubble_pick(rng: &mut SmallRng, b_index: &[Vec<Vec<u32>>], share: f64, inside: impl Fn(u32) -> bool, weight: impl Fn(u32) -> f64) -> Option<u32> {
    if share <= 0.0 || !rng.random_bool(share.min(1.0)) {
        return None;
    }
    let cells: Vec<Vec<u32>> = b_index.iter().flatten().map(|c| c.iter().copied().filter(|&i| inside(i)).collect::<Vec<u32>>()).filter(|c| !c.is_empty()).collect();
    if cells.is_empty() {
        return None;
    }
    let cell = &cells[rng.random_range(0..cells.len())];
    let w: Vec<(u32, f64)> = cell.iter().map(|&i| (i, weight(i))).collect();
    Some(weighted(rng, &w).unwrap_or(cell[0]))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Nodes 0..12 in three cells; "inside" = ids 4..8 (cells 1 and part of 2).
    fn idx() -> Vec<Vec<Vec<u32>>> {
        vec![vec![vec![0, 1, 2, 3], vec![4, 5], vec![6, 7, 8, 9, 10, 11]]]
    }

    #[test]
    fn only_inside_at_full_share() {
        let mut rng = SmallRng::seed_from_u64(1);
        for _ in 0..2000 {
            let b = bubble_pick(&mut rng, &idx(), 1.0, |i| (4..8).contains(&i), |_| 1.0).unwrap();
            assert!((4..8).contains(&b), "{b}");
        }
    }

    #[test]
    fn share_is_the_proportion_drawn() {
        let mut rng = SmallRng::seed_from_u64(2);
        let n = 20000;
        let hits = (0..n).filter(|_| bubble_pick(&mut rng, &idx(), 0.5, |i| (4..8).contains(&i), |_| 1.0).is_some()).count();
        let f = hits as f64 / n as f64;
        assert!((f - 0.5).abs() < 0.02, "{f}");
    }

    #[test]
    fn empty_bubble_or_no_share_defers_to_the_normal_draw() {
        let mut rng = SmallRng::seed_from_u64(3);
        for _ in 0..500 {
            assert_eq!(bubble_pick(&mut rng, &idx(), 1.0, |_| false, |_| 1.0), None);
            assert_eq!(bubble_pick(&mut rng, &idx(), 0.0, |_| true, |_| 1.0), None);
        }
    }

    #[test]
    fn no_bubble_makes_no_draw() {
        // posgoal unset (the default): plan_bubble returns before touching the rng.
        assert_eq!(obrrt_core::posgoal::get(), None);
    }
}
