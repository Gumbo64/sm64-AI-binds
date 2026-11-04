//! The search loop. Each iteration picks one cell per worker thread (Thompson
//! sampling, by stage group), makes inputs for each (see `inputs`), plays them
//! in parallel from the cells' states, and merges every state that reaches a
//! key first or faster, in batch order (so a seed and a thread count give one
//! run).

use std::collections::HashSet;
use std::time::Instant;

use anyhow::Result;
use obrrt_core::{St, check_layout, read_state};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use serde_json::{Value, json};
use sm64_ai::util::skip_menu_pads;
use sm64_binds::{GamePad, SM64Game, SM64GameGenerator};

use crate::archive::{Archive, Found, Merged, Rollout};
use crate::config::Config;
use crate::inputs::{Operator, plan};
use crate::objective::Tree;

pub fn zone(st: &St) -> u32 {
    st.level as u32 * 256 + st.area as u32
}

struct Worker {
    game: SM64Game,
}

impl Worker {
    /// Plays `inputs` from cell `src`'s state, keeping each key's first state
    /// that is new or faster. With `stay`, stops when Mario dies or leaves the
    /// zone he started in.
    fn play(&mut self, archive: &Archive, tree: &Tree, cfg: &Config, src: usize, mut inputs: Vec<GamePad>, stay: bool) -> Result<Rollout> {
        let cell = &archive.cells[src];
        let node = cell.node as usize;
        archive.restore(&mut self.game, node)?;
        let start_zone = archive.nodes[node].zone;
        let (mut found, mut seen, mut prev, mut entered) = (Vec::new(), HashSet::new(), start_zone, false);
        // Frames a state found now would need replayed from its saved ancestor.
        let (mut replay, mut chained) = (archive.nodes[node].replay, 0);
        for t in 0..inputs.len() {
            self.game.step_game(inputs[t])?;
            let frames = cell.frames + t as u32 + 1;
            let st = read_state(&self.game, frames, tree.objects);
            let z = zone(&st);
            if stay && (z != start_zone || !st.alive()) {
                inputs.truncate(t);
                break;
            }
            entered |= z != prev;
            prev = z;
            let key = tree.key(&st);
            if archive.wanted(&key, frames) && seen.insert(key.clone()) {
                replay += (t + 1 - chained) as u32;
                chained = t + 1;
                let save = (replay > cfg.replay_max).then(|| archive.pack(&self.game.save_state(), z));
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
}

/// Rollouts and successes (added or improved another cell) per operator.
#[derive(Default, Clone, Copy)]
pub struct OpStats {
    pub rollouts: u64,
    pub succ: u64,
    pub added: u64,
}

pub struct Search {
    pub tree: Tree,
    pub archive: Archive,
    pub cfg: Config,
    workers: Vec<Worker>,
    rng: SmallRng,
    /// Frames played in rollouts.
    pub frames: u64,
    /// Rollouts played (each expands one cell).
    pub rollouts: u64,
    pub operators: [OpStats; 3],
    /// Expansions and successes by predicted p, in deciles.
    pub calib: [(f32, f32); 10],
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
        let workers = (0..rayon::current_num_threads()).map(|_| Ok(Worker { game: game_gen.create_game()? })).collect::<Result<_>>()?;
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

    /// One batch: a rollout per worker.
    pub fn iterate(&mut self) -> Result<()> {
        let jobs: Vec<(usize, Operator, Vec<GamePad>)> = (0..self.workers.len())
            .map(|_| {
                let src = self.archive.select(&mut self.rng, self.cfg.candidates);
                let mut rng = SmallRng::seed_from_u64(self.rng.random());
                let (op, inputs) = plan(&self.archive, &self.cfg, src, &mut rng);
                (src, op, inputs)
            })
            .collect();
        let (archive, tree, cfg) = (&self.archive, &self.tree, &self.cfg);
        let rollouts: Vec<(Operator, Rollout)> = (self.workers.par_iter_mut().zip(jobs))
            .map(|(w, (src, op, inputs))| Ok((op, w.play(archive, tree, cfg, src, inputs, false)?)))
            .collect::<Result<_>>()?;
        for (op, r) in rollouts {
            self.frames += r.inputs.len() as u64;
            self.rollouts += 1;
            let p = r.p;
            let m: Merged = self.archive.merge(r);
            let s = &mut self.operators[op.index()];
            s.rollouts += 1;
            s.succ += (m.changed > 0) as u64;
            s.added += m.added as u64;
            let b = &mut self.calib[((p * 10.0) as usize).min(9)];
            b.0 += 1.0;
            b.1 += (m.changed > 0) as u8 as f32;
        }
        self.archive.free_pending();
        Ok(())
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
                (op.name().to_string(), json!({"rollouts": s.rollouts, "succ": s.succ, "added": s.added}))
            })
            .collect();
        let stats = json!({"elapsed_s": self.elapsed_s(), "frames": self.frames, "rollouts": self.rollouts,
            "cells": a.cells.len(), "nodes": a.nodes.len(), "stale_rollouts": a.stale, "checkpoints": a.checkpoints(), "save_bytes": a.save_bytes(), "references": a.references().0, "rss_bytes": rss,
            "groups": self.group_stats(), "operators": ops});
        std::fs::write(format!("{dir}/stats.json.tmp"), stats.to_string())?;
        std::fs::rename(format!("{dir}/stats.json.tmp"), format!("{dir}/stats.json"))?;
        a.write(dir)
    }
}
