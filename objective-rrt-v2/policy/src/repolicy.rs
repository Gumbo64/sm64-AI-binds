//! The standalone hybrid's `repolicy` operator as a [`obrrt_search::ga::BreedHook`]
//! for the production Worker GA: replay a parent's pads to a cut, let an inventory
//! policy (default instance or sampled lambda draw) play the rest of the horizon from
//! that mid-trajectory state, and hand back the realised pads. Frames: the replayed
//! prefix plus the policy's own (the GA then charges its re-evaluation of the child).
//!
//! Enable: `repolicy::install(SeedCfg::from_env())` and run with `OBRRT_POLICY_REPOLICY=p`.
//! Per attempt the session remembers how each policy did from this start (mean score of
//! its rollouts) and favours the best four 60 % of the time, as the hybrid did.

use std::collections::HashMap;
use std::sync::Arc;

use obrrt_core::{GamePad, Objective, St, read_state};
use obrrt_gp::rollout::{Case, PREFIX, Roll};
use obrrt_search::ga::{BreedHook, BreedSession};
use rand::Rng;
use rand::rngs::SmallRng;
use sm64_binds::{SM64Game, SaveState};

use crate::eval::{self, Env};
use crate::hybrid::Library;
use crate::inv::{draw_instance, instance_of};
use crate::seed::{ObjView, SeedCfg, load_inventory};

pub struct PolicyBreed {
    lib: Arc<Library>,
    env: Arc<Env>,
    names: Vec<String>,
    cfg: SeedCfg,
}

impl PolicyBreed {
    pub fn new(cfg: SeedCfg) -> PolicyBreed {
        let (lib, names) = load_inventory(&cfg.inventory);
        PolicyBreed { lib: Arc::new(lib), env: Arc::new(Env::full()), names, cfg }
    }
    pub fn len(&self) -> usize {
        self.names.len()
    }
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

impl BreedHook for PolicyBreed {
    fn session(&self) -> Box<dyn BreedSession> {
        Box::new(Session { lib: self.lib.clone(), env: self.env.clone(), names: self.names.clone(), cfg: self.cfg.clone(), stats: HashMap::new() })
    }
}

struct Session {
    lib: Arc<Library>,
    env: Arc<Env>,
    names: Vec<String>,
    cfg: SeedCfg,
    /// policy -> (sum of scores, n) of its rollouts in this attempt
    stats: HashMap<String, (f64, u32)>,
}

impl BreedSession for Session {
    fn breed(
        &mut self,
        game: &mut SM64Game,
        start: &St,
        save: &SaveState,
        stage: &dyn Objective,
        prefix: &[GamePad],
        states: &[St],
        horizon: usize,
        rng: &mut SmallRng,
    ) -> Option<(Vec<GamePad>, u64)> {
        if self.names.is_empty() {
            return None;
        }
        // policy: favour the ones that did well from this start
        let name = {
            let mut tried: Vec<(&String, f64)> = self.names.iter().filter_map(|n| self.stats.get(n).map(|(s, c)| (n, s / *c as f64))).collect();
            if tried.len() >= 4 && rng.random_bool(0.6) {
                tried.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                tried[rng.random_range(0..4)].0.clone()
            } else {
                self.names[rng.random_range(0..self.names.len())].clone()
            }
        };
        let pol = self.lib.bodies.get(&name)?;
        let inst = if rng.random_bool(0.5) { draw_instance(pol, rng) } else { instance_of(pol) };
        // replay the prefix to the cut state
        game.load_state(save).ok()?;
        for p in prefix {
            game.step_game(*p).ok()?;
        }
        let cut = prefix.len();
        let cur = states.last()?;
        let mid = game.save_state();
        let k = states.len().saturating_sub(PREFIX + 1);
        let pk = prefix.len().saturating_sub(PREFIX);
        let case = Case {
            name: "repolicy".into(),
            train: false,
            val: false,
            movie: 0,
            frame: cur.frame as usize,
            noise: Vec::new(),
            save: Arc::new(mid),
            pre_states: states[k..].to_vec(),
            pre_pads: prefix[pk..].to_vec(),
            start: start.clone(),
            task: 0,
            ref_pads: Vec::new(),
        };
        let _ = read_state; // (the cut state is the parent's own trajectory entry)
        let task = ObjView::new(stage, self.cfg.fallback_pts.clone());
        let roll = Roll { horizon: horizon.saturating_sub(cut).max(10), ..Roll::default() };
        let pl = eval::play_policy(game, &case, &task, &pol.body, &*self.lib, &self.env, &roll, true, &inst).ok()?;
        let score = match pl.out.done {
            Some(d) => 1e6 - (cut + d) as f64,
            None => -pl.out.best_eta,
        };
        let e = self.stats.entry(name).or_insert((0.0, 0));
        e.0 += score;
        e.1 += 1;
        let mut all = prefix.to_vec();
        all.extend_from_slice(&pl.out.pads);
        Some((all, cut as u64 + pl.out.frames))
    }
}

/// Installs the inventory as the process-wide breed hook (used only when `OBRRT_POLICY_REPOLICY` > 0). Returns the number of policies.
pub fn install(cfg: SeedCfg) -> usize {
    let h = PolicyBreed::new(cfg);
    let n = h.len();
    obrrt_search::ga::set_breed_hook(Some(Arc::new(h)));
    n
}
