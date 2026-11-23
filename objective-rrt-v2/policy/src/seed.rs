//! The policy inventory as an [`obrrt_search::ga::SeedSource`]: before a Worker
//! GA attempt, every active inventory policy is rolled out once from the
//! attempt's start (plus optional sampled lambda draws); the realised pads of
//! the best rollouts (finished ones first, by frames; then by closest ETA)
//! seed the GA. Emulator frames are counted (the whole bill is put on the
//! first returned plan).
//!
//! Enable in a binary: `obrrt_policy::seed::install(SeedCfg::from_env())`
//! and run with `OBRRT_POLICY_SEED=1` (or `obrrt_search::ga::set_policy_seed`).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use obrrt_core::{Eta, GamePad, Mem, Objective, St};
use obrrt_gp::lang::MAX_POINTS;
use obrrt_gp::rollout::{Case, Roll, Task};
use obrrt_search::ga::SeedSource;
use rand::rngs::SmallRng;
use sm64_binds::{SM64Game, SaveState};

use crate::eval::{self, Env};
use crate::hybrid::Library;
use crate::inv::{draw_instance, instance_of};

#[derive(Clone, Debug)]
pub struct SeedCfg {
    /// Inventory directory (`*.pol` at the top level are filtered by `index.json`'s status; `incoming/` and `incoming/done/` are all used).
    pub inventory: PathBuf,
    /// Sampled lambda draws per policy on top of its default instance.
    pub draws: usize,
    /// Return only rollouts that finish the objective.
    pub done_only: bool,
    /// Plans returned (best first).
    pub keep: usize,
    /// Rollout horizon in frames.
    pub horizon: usize,
    /// Points shown to policies when the objective has no `steer_target` (slot 0 first).
    pub fallback_pts: Vec<[f32; 3]>,
}

impl SeedCfg {
    /// `OBRRT_POLICY_INV` (policy/runs/inventory), `OBRRT_POLICY_DRAWS` (0), `OBRRT_POLICY_DONE_ONLY` (0), `OBRRT_POLICY_KEEP` (1 + `OBRRT_POLICY_SEED_EXTRA`), `OBRRT_POLICY_HORIZON` (300).
    pub fn from_env() -> SeedCfg {
        let n = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        SeedCfg {
            inventory: std::env::var("OBRRT_POLICY_INV").unwrap_or("policy/runs/inventory".into()).into(),
            draws: n("OBRRT_POLICY_DRAWS", 0),
            done_only: n("OBRRT_POLICY_DONE_ONLY", 0) == 1,
            keep: 1 + n("OBRRT_POLICY_SEED_EXTRA", 0),
            horizon: n("OBRRT_POLICY_HORIZON", 300),
            fallback_pts: Vec::new(),
        }
    }
}

/// An objective as the task a policy reads: ETA from the objective, point 0 its steering target (else the fallback).
pub(crate) struct ObjView<'a> {
    obj: &'a dyn Objective,
    fallback: Vec<[f32; 3]>,
}

impl<'a> ObjView<'a> {
    pub(crate) fn new(obj: &'a dyn Objective, fallback: Vec<[f32; 3]>) -> Self {
        ObjView { obj, fallback }
    }
}

impl Task for ObjView<'_> {
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        self.obj.eta(s, start, mem)
    }
    fn points(&self, s: &St, _mem: &Mem) -> [[f32; 3]; MAX_POINTS] {
        let mut p = [[0.0f32; 3]; MAX_POINTS];
        for (i, f) in self.fallback.iter().take(MAX_POINTS).enumerate() {
            p[i] = *f;
        }
        if let Some(t) = self.obj.steer_target(s) {
            p[0] = t;
        }
        p
    }
    fn target_for_baseline(&self, s: &St) -> [f32; 3] {
        self.points(s, &Mem::default())[0]
    }
}

pub struct PolicySeeds {
    lib: Library,
    env: Env,
    /// Active policy names (library keys), sorted.
    names: Vec<String>,
    cfg: SeedCfg,
    /// Rollouts are deterministic without lambda draws, so a (start, objective) is rolled out once per process
    /// (wall time only: the frames are charged to every attempt as if it had).
    cache: std::sync::Mutex<std::collections::HashMap<u64, Vec<(Vec<GamePad>, u64)>>>,
}

/// The library and the active names of an inventory directory: top-level policies whose index status is "active" (all of them if there is no index), and every policy under `incoming/`.
pub fn load_inventory(dir: &Path) -> (Library, Vec<String>) {
    let inc = dir.join("incoming");
    let done = inc.join("done");
    let lib = Library::load(&[dir, inc.as_path(), done.as_path()]);
    let active: Option<HashSet<String>> = std::fs::read_to_string(dir.join("index.json")).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()).map(|v| {
        v["entries"].as_array().into_iter().flatten().filter(|e| e["status"] == "active").filter_map(|e| e["id"].as_str().map(String::from)).collect()
    });
    let stems = |d: &Path| -> Vec<String> {
        std::fs::read_dir(d)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                (p.extension().is_some_and(|x| x == "pol")).then(|| p.file_stem().unwrap().to_string_lossy().to_string())
            })
            .collect()
    };
    let mut names: Vec<String> = stems(dir).into_iter().filter(|s| active.as_ref().is_none_or(|a| a.contains(s))).collect();
    names.extend(stems(&inc));
    names.extend(stems(&done));
    names.retain(|n| lib.bodies.contains_key(n));
    names.sort();
    names.dedup();
    (lib, names)
}

impl PolicySeeds {
    pub fn new(cfg: SeedCfg) -> PolicySeeds {
        let (lib, names) = load_inventory(&cfg.inventory);
        PolicySeeds { lib, env: Env::full(), names, cfg, cache: Default::default() }
    }
    pub fn len(&self) -> usize {
        self.names.len()
    }
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

impl SeedSource for PolicySeeds {
    fn seeds(&self, base: &mut SM64Game, start: &St, save: &SaveState, stage: &dyn Objective, rng: &mut SmallRng) -> Vec<(Vec<GamePad>, u64)> {
        let key = (self.cfg.draws == 0).then(|| {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            format!("{:?}", start).hash(&mut h);
            stage.key().hash(&mut h);
            h.finish()
        });
        if let Some(v) = key.and_then(|k| self.cache.lock().unwrap().get(&k).cloned()) {
            return v;
        }
        let case = Case {
            name: "deploy".into(),
            train: false,
            val: false,
            movie: 0,
            frame: start.frame as usize,
            noise: Vec::new(),
            save: Arc::new(save.clone()),
            pre_states: vec![start.clone()],
            pre_pads: Vec::new(),
            start: start.clone(),
            task: 0,
            ref_pads: Vec::new(),
        };
        let task = ObjView { obj: stage, fallback: self.cfg.fallback_pts.clone() };
        let roll = Roll { horizon: self.cfg.horizon, ..Roll::default() };
        let mut spent = 0u64;
        // (finished frames or None, best ETA, pads)
        let mut got: Vec<(Option<usize>, f64, Vec<GamePad>)> = Vec::new();
        for name in &self.names {
            let Some(pol) = self.lib.bodies.get(name) else { continue };
            for k in 0..=self.cfg.draws {
                let inst = if k == 0 { instance_of(pol) } else { draw_instance(pol, rng) };
                let Ok(pl) = eval::play_policy(base, &case, &task, &pol.body, &self.lib, &self.env, &roll, true, &inst) else { continue };
                spent += pl.out.frames;
                if pl.out.pads.is_empty() || (self.cfg.done_only && pl.out.done.is_none()) {
                    continue;
                }
                got.push((pl.out.done, pl.out.best_eta, pl.out.pads));
            }
        }
        got.sort_by(|a, b| match (a.0, b.0) {
            (Some(x), Some(y)) => x.cmp(&y),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal),
        });
        let mut out: Vec<(Vec<GamePad>, u64)> = Vec::new();
        for (_, _, pads) in got {
            if out.len() >= self.cfg.keep.max(1) {
                break;
            }
            if out.iter().any(|o| o.0 == pads) {
                continue;
            }
            out.push((pads, 0));
        }
        if out.is_empty() {
            // nothing usable: the bill still has to be reported
            out.push((Vec::new(), spent));
        } else {
            out[0].1 = spent;
        }
        if let Some(k) = key {
            self.cache.lock().unwrap().insert(k, out.clone());
        }
        out
    }
}

/// Installs the inventory as the process-wide seed source (it is consulted only when `OBRRT_POLICY_SEED=1` or `set_policy_seed(Some(true))`). Returns the number of policies.
pub fn install(cfg: SeedCfg) -> usize {
    let s = PolicySeeds::new(cfg);
    let n = s.len();
    obrrt_search::ga::set_seed_source(Some(Arc::new(s)));
    n
}
