//! The search loop. Each iteration picks one cell per worker thread (Thompson
//! sampling, by stage group), expands each (pellets: inputs made by `inputs`;
//! GA: `ga::expand`), plays them in parallel from the cells' states, and
//! merges every state that reaches a key first or faster, in batch order (so a
//! seed and a thread count give one run).
//!
//! With `OSS_GA=<p>` a batch is a GA batch with probability p (or, with
//! `OSS_GA_BY=frames`, while the GA's share of frames is below p): every
//! expansion in it is a GA attempt (a busy cell still gets a press-through
//! pellet). Batches aren't mixed because a GA expansion plays ~200x the frames
//! of a pellet, and a batch waits for its slowest job.

use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

use anyhow::Result;
use obrrt_core::{St, check_layout, read_state};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use serde_json::{Value, json};
use objective_scattershot::util::skip_menu_pads;
use sm64_binds::{GamePad, SM64Game, SM64GameGenerator};

use crate::archive::{Archive, Found, Merged, Rollout};
use crate::config::Config;
use crate::ga::{GaOut, GaStats, KindStats};
use crate::inputs::{Operator, plan};
use crate::objective::Tree;

pub fn zone(st: &St) -> u32 {
    st.level as u32 * 256 + st.area as u32
}

/// One thread's games: `game` for pellets, `ga` (base / cur / tmp) for GA
/// attempts and for replaying their results.
pub(crate) struct Worker {
    pub(crate) game: SM64Game,
    pub(crate) ga: obrrt_search::ga::Worker,
}

/// Plays `inputs` on `game`, which holds cell `src`'s state, keeping each
/// key's first state that is new or faster. With `stay`, stops when Mario dies
/// or leaves the zone he started in. With `every` false only the last state is
/// keyed (the frames before are just played).
#[allow(clippy::too_many_arguments)]
pub(crate) fn play_on(game: &mut SM64Game, archive: &Archive, tree: &Tree, cfg: &Config, src: usize, mut inputs: Vec<GamePad>, stay: bool, every: bool) -> Result<Rollout> {
    let cell = &archive.cells[src];
    let node = cell.node as usize;
    let start_zone = archive.nodes[node].zone;
    let (mut found, mut seen, mut prev, mut entered) = (Vec::new(), HashSet::new(), start_zone, false);
    // Frames a state found now would need replayed from its saved ancestor.
    let (mut replay, mut chained) = (archive.nodes[node].replay, 0);
    let n = inputs.len();
    for t in 0..n {
        game.step_game(inputs[t])?;
        if !every && !stay && t + 1 < n {
            continue;
        }
        let frames = cell.frames + t as u32 + 1;
        let st = read_state(game, frames, tree.objects);
        let z = zone(&st);
        if stay && (z != start_zone || !st.alive()) {
            inputs.truncate(t);
            break;
        }
        // Without `every` the zone changes in between aren't seen: compare
        // with the start.
        entered |= z != prev;
        prev = z;
        let key = tree.key(&st);
        if archive.wanted(&key, frames) && archive.admits(&st) && seen.insert(key.clone()) {
            replay += (t + 1 - chained) as u32;
            chained = t + 1;
            let save = (replay > cfg.replay_max).then(|| archive.pack(&game.save_state(), z));
            if save.is_some() {
                replay = 0;
            }
            let pos = st.pos.map(|v| v.clamp(i16::MIN as f32, i16::MAX as f32) as i16);
            found.push(Found { key, t, save, group: tree.group(&st), busy: tree.busy(&st), zone: z, entered, pos });
            entered = false;
        }
    }
    Ok(Rollout { src, node, frames: cell.frames, p: cell.p(), inputs, found })
}

impl Worker {
    /// Plays `inputs` from cell `src`'s state (see [`play_on`]).
    pub(crate) fn play(&mut self, archive: &Archive, tree: &Tree, cfg: &Config, src: usize, inputs: Vec<GamePad>, stay: bool) -> Result<Rollout> {
        archive.restore(&mut self.game, archive.cells[src].node as usize)?;
        play_on(&mut self.game, archive, tree, cfg, src, inputs, stay, true)
    }
}

/// Rollouts (expansions) and successes (added or improved another cell) per
/// operator; cells added and improved (replaced by a faster state).
#[derive(Default, Clone, Copy)]
pub struct OpStats {
    pub rollouts: u64,
    pub succ: u64,
    pub added: u64,
    pub improved: u64,
    /// Frames emulated by this operator's expansions.
    pub frames: u64,
}

enum Job {
    Pellet { src: usize, op: Operator, inputs: Vec<GamePad> },
    Ga { src: usize, rng: SmallRng },
}

enum Done {
    Pellet(Operator, Rollout),
    Ga(GaOut),
}

pub struct Search {
    pub tree: Tree,
    pub archive: Archive,
    pub cfg: Config,
    workers: Vec<Worker>,
    rng: SmallRng,
    /// Frames played in expansions (pellets' inputs; GA attempts' every
    /// simulated frame and the replays of what they harvest).
    pub frames: u64,
    /// Expansions made (one cell each).
    pub rollouts: u64,
    pub operators: [OpStats; 4],
    /// Expansions and successes by predicted p, in deciles.
    pub calib: [(f32, f32); 10],
    pub ga: GaStats,
    /// Per objective kind, for `OSS_GA_LEARN` and the stats.
    pub kinds: BTreeMap<&'static str, KindStats>,
    /// Speed mode's piped route and state (`Search::pipe`).
    pub speed: Option<crate::speed::Speed>,
    started: Instant,
}

impl Search {
    /// A search whose only cell is power-on; give it a start with
    /// [`Search::seed`] (e.g. `skip_menu_pads()`).
    pub fn new(game_gen: &SM64GameGenerator, tree: Tree, cfg: Config, seed: u64) -> Result<Search> {
        let mut game = game_gen.create_game()?;
        let st = read_state(&game, 0, tree.objects);
        let power_on = game.save_state();
        for pad in skip_menu_pads() {
            game.step_game(pad)?;
        }
        check_layout(&game)?;
        let archive = Archive::new(&power_on, &game.save_state(), tree.key(&st), zone(&st)).with_groups(tree.rank(), cfg.grouped);
        let workers = (0..rayon::current_num_threads())
            .map(|_| Ok(Worker { game: game_gen.create_game()?, ga: obrrt_search::ga::Worker::new(game_gen)? }))
            .collect::<Result<_>>()?;
        Ok(Search {
            tree,
            archive,
            cfg,
            workers,
            rng: SmallRng::seed_from_u64(seed),
            frames: 0,
            rollouts: 0,
            operators: Default::default(),
            calib: [(0.0, 0.0); 10],
            ga: Default::default(),
            kinds: BTreeMap::new(),
            speed: None,
            started: Instant::now(),
        })
    }

    /// Plays `inputs` from cell `src` and merges what they reach, as an
    /// expansion of `src`. Returns the cells added or improved.
    pub fn play_from(&mut self, src: usize, inputs: Vec<GamePad>, stay: bool) -> Result<usize> {
        let r = self.workers[0].play(&self.archive, &self.tree, &self.cfg, src, inputs, stay)?;
        let m = self.archive.merge(r);
        self.archive.free_pending();
        Ok(m.changed)
    }

    /// Plays `inputs` from power-on (e.g. a replay from `load_m64`) and keeps
    /// every new or faster cell on the way.
    pub fn seed(&mut self, inputs: Vec<GamePad>) -> Result<usize> {
        self.play_from(0, inputs, false)
    }

    /// Speed mode: pipes in the route `inputs` (from power-on to node `node`
    /// of obrrt run `source`, expected to end at `frame` with `stars`):
    /// replays it, checks it, merges its every state from the power-on cell
    /// and records its milestones; then sets up the corridor and the
    /// replacement log. See `speed`.
    pub fn pipe(&mut self, source: &str, node: u32, inputs: Vec<GamePad>, frame: u32, stars: i16) -> Result<()> {
        use crate::speed::{Corridor, Route, Speed};
        anyhow::ensure!(self.tree.insts[0].kind == "Root", "speed mode needs a schema whose objective 0 is a Root (it keys level, keys and stars)");
        anyhow::ensure!(inputs.len() as u32 == frame, "{source} node {node}: {} inputs but nodes.csv says frame {frame}", inputs.len());
        let game = &mut self.workers[0].game;
        self.archive.restore(game, 0)?;
        let mut sts = Vec::with_capacity(inputs.len());
        for (t, &pad) in inputs.iter().enumerate() {
            game.step_game(pad)?;
            sts.push(read_state(game, t as u32 + 1, self.tree.objects));
        }
        let end = sts.last().map_or(0, |s| s.stars);
        anyhow::ensure!(end == stars, "{source} node {node}: the replay ends with {end} stars, nodes.csv says {stars} (another ROM or boot?)");
        let route = Route::new(source.to_string(), node, inputs.clone(), &sts, &self.tree);
        drop(sts);
        if self.cfg.speed.prop > 0.0 {
            self.archive.keep_kids();
        }
        let n = self.play_from(0, inputs, false)?;
        self.archive.log.clear();
        self.archive.track = true;
        if let Some(w) = self.cfg.speed.corridor {
            self.archive.corridor = Some(Corridor::new(w, &route.pts));
        }
        println!(
            "pipe: {source} node {node}: {} frames, {stars} stars, {n} cells; control from frame {}; milestones: {}",
            route.inputs.len(),
            route.pts[route.start].frame,
            route.milestones.iter().map(|m| format!("{} @{}", m.name, m.frame)).collect::<Vec<_>>().join(", ")
        );
        self.speed = Some(Speed::new(route));
        Ok(())
    }

    /// The base cell of the next expansion.
    fn select(&mut self) -> usize {
        match &self.speed {
            Some(sp) if self.cfg.speed.select_route => sp.select(&self.archive, &mut self.rng),
            _ => self.archive.select(&mut self.rng, self.cfg.candidates),
        }
    }

    /// Speed mode: takes the replacements logged since the last call; each
    /// improved cell becomes recent, and with `OSS_PROP` the child nodes of
    /// its old node are queued to replay from it.
    fn drain_log(&mut self) {
        let Some(sp) = self.speed.as_mut() else { return };
        let prop = self.cfg.speed.prop > 0.0;
        for (cell, old) in std::mem::take(&mut self.archive.log) {
            sp.note(cell);
            if prop {
                for &c in self.archive.kids(old as usize) {
                    sp.push(cell, c);
                }
            }
        }
    }

    /// `OSS_PROP`: replays queued grafts, in queue order, while propagation's
    /// frames are under its share of all frames. Each replays a child node's
    /// inputs from the improved cell's current state and merges every state
    /// (first-or-faster); a replacement queues its own children (cascade).
    fn propagate(&mut self) -> Result<()> {
        let share = self.cfg.speed.prop;
        if share <= 0.0 {
            return Ok(());
        }
        loop {
            let Some(sp) = self.speed.as_mut() else { break };
            if sp.prop.frames as f64 >= share * self.frames as f64 {
                break;
            }
            let Some((cell, child)) = sp.pop() else { break };
            let inputs = self.archive.inputs(child as usize).to_vec();
            if inputs.is_empty() {
                continue;
            }
            let r = self.workers[0].play(&self.archive, &self.tree, &self.cfg, cell as usize, inputs, false)?;
            let n = r.inputs.len() as u64;
            let m = self.archive.merge_part(r).unwrap_or_default();
            self.frames += n;
            let p = &mut self.speed.as_mut().unwrap().prop;
            p.tried += 1;
            p.frames += n;
            p.improved += m.improved as u64;
            p.added += m.added as u64;
            self.drain_log();
        }
        self.archive.free_pending();
        Ok(())
    }

    /// Whether the next batch is a GA batch. Draws from the run's rng only
    /// when the share is strictly between 0 and 1 (so `OSS_GA=0` is
    /// objective-scattershot's run exactly).
    fn ga_batch(&mut self) -> bool {
        let g = &self.cfg.ga;
        if g.share <= 0.0 {
            false
        } else if g.by_frames {
            (self.ga.frames as f64) < g.share * self.frames as f64 || self.frames == 0
        } else if g.share >= 1.0 {
            true
        } else {
            self.rng.random_bool(g.share)
        }
    }

    fn count(&mut self, op: Operator, p: f32, m: &Merged, frames: u64) {
        self.rollouts += 1;
        self.frames += frames;
        let s = &mut self.operators[op.index()];
        s.rollouts += 1;
        s.succ += (m.changed > 0) as u64;
        s.added += m.added as u64;
        s.improved += m.improved as u64;
        s.frames += frames;
        let b = &mut self.calib[((p * 10.0) as usize).min(9)];
        b.0 += 1.0;
        b.1 += (m.changed > 0) as u8 as f32;
    }

    /// One batch: an expansion per worker.
    pub fn iterate(&mut self) -> Result<()> {
        let ga = self.ga_batch();
        self.ga.batches += ga as u64;
        let jobs: Vec<Job> = (0..self.workers.len())
            .map(|_| {
                let src = self.select();
                let mut rng = SmallRng::seed_from_u64(self.rng.random());
                if ga && !self.archive.cells[src].busy {
                    return Job::Ga { src, rng };
                }
                let (op, inputs) = plan(&self.archive, &self.cfg, src, &mut rng);
                Job::Pellet { src, op, inputs }
            })
            .collect();
        let (archive, tree, cfg, kinds) = (&self.archive, &self.tree, &self.cfg, &self.kinds);
        let route = self.speed.as_ref().filter(|_| cfg.speed.ga_route).map(|s| &s.route);
        let done: Vec<Done> = (self.workers.par_iter_mut().zip(jobs))
            .map(|(w, job)| {
                Ok(match job {
                    Job::Pellet { src, op, inputs } => Done::Pellet(op, w.play(archive, tree, cfg, src, inputs, false)?),
                    Job::Ga { src, rng } => Done::Ga(match route {
                        Some(r) => crate::speed::expand_route(w, archive, tree, cfg, r, src, rng)?,
                        None => crate::ga::expand(w, archive, tree, cfg, kinds, src, rng)?,
                    }),
                })
            })
            .collect::<Result<_>>()?;
        for d in done {
            match d {
                Done::Pellet(op, r) => {
                    let (p, frames) = (r.p, r.inputs.len() as u64);
                    let m = self.archive.merge(r);
                    self.count(op, p, &m, frames);
                }
                Done::Ga(g) => self.merge_ga(g),
            }
            self.drain_log();
        }
        self.archive.free_pending();
        self.propagate()?;
        Ok(())
    }

    /// Merges a GA expansion's rollouts (or its fallback pellet) and credits
    /// its source once.
    fn merge_ga(&mut self, g: GaOut) {
        let (src, p) = (g.src, g.p);
        let mut m = Merged::default();
        let mut stale = false;
        for r in g.parts {
            match self.archive.merge_part(r) {
                Some(x) => m.add(&x),
                None => {
                    stale = true;
                    break;
                }
            }
        }
        if !stale {
            self.archive.credit(src, &m);
        }
        self.count(g.op, p, &m, g.frames);
        let s = &mut self.ga;
        if g.op != Operator::Ga {
            s.fallbacks += 1;
            return;
        }
        s.expansions += 1;
        s.frames += g.frames;
        s.eligible += g.eligible as u64;
        s.met += g.met as u64;
        s.succ += (m.changed > 0) as u64;
        s.added += m.added as u64;
        s.improved += m.improved as u64;
        s.stale += stale as u64;
        if let Some(kind) = g.kind {
            let k = self.kinds.entry(kind).or_default();
            k.tried += 1;
            k.met += g.met as u64;
            k.succ += (m.changed > 0) as u64;
            k.added += m.added as u64;
            k.improved += m.improved as u64;
            k.frames += g.frames;
        }
    }

    fn name(&self, i: usize) -> String {
        self.tree.insts[i].name.clone().unwrap_or_else(|| format!("{}#{i}", self.tree.insts[i].kind))
    }

    /// Per sampling group with cells: name, cells, expansions, those that
    /// reached new ground, and the fewest frames of a cell in it.
    pub fn group_stats(&self) -> Vec<Value> {
        (self.archive.groups.iter().enumerate())
            .filter(|(_, g)| !g.cells.is_empty())
            .map(|(i, g)| {
                let best = (g.best_frames != u32::MAX).then_some(g.best_frames);
                json!({"name": self.name(i), "cells": g.cells.len(), "visits": g.visits, "succ": g.succ, "best_frames": best})
            })
            .collect()
    }

    /// One line for the log: `Stage cells/expansions/progress%`.
    pub fn groups_line(&self) -> String {
        let pct = |s: f32, v: f32| if v > 0.0 { (100.0 * s / v).round() } else { 0.0 };
        (self.archive.groups.iter().enumerate())
            .filter(|(_, g)| !g.cells.is_empty())
            .map(|(i, g)| format!("{}:{}/{:.0}/{}%", self.name(i), g.cells.len(), g.visits, pct(g.succ, g.visits)))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// One line for the log: per operator expansions/added/improved, and the
    /// GA's objectives tried/met.
    pub fn operators_line(&self) -> String {
        let mut parts: Vec<String> = (Operator::ALL.iter())
            .filter(|&&op| self.operators[op.index()].rollouts > 0)
            .map(|&op| {
                let s = self.operators[op.index()];
                format!("{}:{}/+{}/^{}", op.name(), s.rollouts, s.added, s.improved)
            })
            .collect();
        if self.ga.expansions + self.ga.fallbacks > 0 {
            parts.push(format!("| ga tried {} met {} fallback {}", self.ga.expansions, self.ga.met, self.ga.fallbacks));
        }
        parts.join(" ")
    }

    /// The best (fewest) frames of a cell with an entry from each named
    /// objective's subtree.
    pub fn best(&self) -> Vec<(String, Option<u32>)> {
        (self.tree.insts.iter())
            .filter(|i| i.name.is_some())
            .map(|i| (i.name.clone().unwrap(), self.archive.best_in(i.id..i.end)))
            .collect()
    }

    /// Predicted vs realised success rate per decile of p.
    pub fn calibration(&self) -> String {
        let parts = self.calib.iter().enumerate().filter(|(_, b)| b.0 > 0.0);
        parts.map(|(d, b)| format!("{}:{:.2}({})", d, b.1 / b.0, b.0)).collect::<Vec<_>>().join(" ")
    }

    /// Cells whose best was replaced by a faster state, per million frames.
    pub fn improved_per_mframe(&self) -> f64 {
        if self.frames == 0 { 0.0 } else { self.archive.replaced as f64 * 1e6 / self.frames as f64 }
    }

    pub fn elapsed_s(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// Writes stats.json first (the GUI reloads when cells.jsonl changes):
    /// counters with the elapsed time, so it can show rates, the memory held
    /// (RSS: the whole process), groups and operators; then the archive.
    pub fn write(&self, dir: &str) -> Result<()> {
        let a = &self.archive;
        let rss = std::fs::read_to_string("/proc/self/statm").ok().and_then(|t| t.split_whitespace().nth(1)?.parse::<u64>().ok()).map_or(0, |p| p * 4096);
        let ops: serde_json::Map<String, Value> = (Operator::ALL.iter())
            .map(|&op| {
                let s = self.operators[op.index()];
                (op.name().to_string(), json!({"rollouts": s.rollouts, "succ": s.succ, "added": s.added, "improved": s.improved, "frames": s.frames}))
            })
            .collect();
        let kinds: serde_json::Map<String, Value> = self.kinds.iter().map(|(k, s)| (k.to_string(), s.json())).collect();
        let g = &self.ga;
        let ga = json!({"share": self.cfg.ga.share, "by_frames": self.cfg.ga.by_frames, "harvest": self.cfg.ga.harvest.name(), "learn": self.cfg.ga.learn,
            "batches": g.batches, "expansions": g.expansions, "fallbacks": g.fallbacks, "eligible": g.eligible, "met": g.met, "succ": g.succ,
            "added": g.added, "improved": g.improved, "stale": g.stale, "frames": g.frames, "kinds": kinds});
        let stats = json!({"elapsed_s": self.elapsed_s(), "frames": self.frames, "rollouts": self.rollouts,
            "cells": a.cells.len(), "nodes": a.nodes.len(), "stale_rollouts": a.stale, "checkpoints": a.checkpoints(), "save_bytes": a.save_bytes(), "references": a.references().0, "rss_bytes": rss,
            "replaced": a.replaced, "improved_per_mframe": self.improved_per_mframe(),
            "groups": self.group_stats(), "operators": ops, "ga": ga});
        let mut stats = stats;
        if let Some(sp) = &self.speed {
            stats["speed"] = sp.json(a, self.frames, &self.cfg.speed);
        }
        std::fs::write(format!("{dir}/stats.json.tmp"), stats.to_string())?;
        std::fs::rename(format!("{dir}/stats.json.tmp"), format!("{dir}/stats.json"))?;
        let Some(sp) = &self.speed else { return a.write(dir) };
        // The route and the fastest path to its final milestone, as .m64s
        // (from power-on); and per cell its route frame, frames saved and
        // times improved, for the viewer (`route`, `saved`, `imp`).
        crate::speed::write_m64(&format!("{dir}/route.m64"), &sp.route.inputs)?;
        if let Some(b) = sp.route.milestones.last().and_then(|m| sp.best(a, m)) {
            crate::speed::write_m64(&format!("{dir}/fastest.m64"), &a.path(a.cells[b.1].node as usize))?;
        }
        let extra = |i: usize, c: &crate::archive::Cell| {
            let mut m = serde_json::Map::new();
            if let Some(t) = sp.route.index_of(&c.key) {
                let rf = sp.route.pts[t].frame;
                m.insert("route".into(), rf.into());
                m.insert("saved".into(), (rf as i64 - c.frames as i64).into());
            }
            if let Some(&n) = sp.imp.get(i).filter(|&&n| n > 0) {
                m.insert("imp".into(), n.into());
            }
            (!m.is_empty()).then_some(m)
        };
        a.write_annotated(dir, Some(&extra))
    }
}
