//! The GA expansion (`OSS_GA`): objective-rrt-v2's open-loop GA, with its
//! discounted-ETA fitness, from a cell's state towards one objective that
//! obrrt-objectives' `Registry` generates there.
//!
//! 1. Restore the cell's state (in the GA worker's `base`) and read it with
//!    objects.
//! 2. Generate v2 objectives (`Registry::generate`) and keep those that may
//!    start here: `base_criteria`, `exists_in`, a finite `start_eta`, not
//!    modal. (A busy cell never gets here: it is a press-through pellet; a
//!    modal state or one with no objective falls back to a pellet.)
//! 3. Pick one with probability ∝ `obj_weight` × (with `OSS_GA_LEARN`, on by
//!    default) its kind's success rate in this run, (succ + 1) / (tried + 2),
//!    from the kind table as of the batch's start (so it is deterministic).
//! 4. Run `obrrt_search::ga::Worker::attempt_full` from that state with
//!    obrrt2's `GaParams` (budgets from the objective's `ga_budget`, else
//!    16 / 5 / 150 with 3x generations: up to 36 000 simulated frames).
//! 5. Harvest (`OSS_GA_HARVEST`): replay the winner's committed inputs (the
//!    attempt's segments: up to its fitness maximum, plus the automatic tail)
//!    from the cell with the pellet's own key/merge code, keying every state;
//!    with `ends` also each last-generation rollout's end state, with `all`
//!    every state along those rollouts. The last generation is taken through
//!    ga.rs's dense-add collector (`last_gen`, with a "new" test that accepts
//!    everything and no saves): its genomes come back as candidates' inputs.
//!    Every rollout's states go through the archive's merge exactly as a
//!    pellet's (key, first-or-faster, sparse saves by `replay_max`); the cell
//!    is credited once for the whole expansion.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Result;
use obrrt_core::{Generator, Objective, St, read_state, start_eta};
use obrrt_search::ga::DenseReq;
use obrrt_search::store::Delta;
use rand::Rng;
use rand::rngs::SmallRng;
use serde_json::{Value, json};
use sm64_binds::{GamePad, SaveState};

use crate::archive::{Archive, Rollout};
use crate::config::{Config, Harvest};
use crate::inputs::{Operator, plan};
use crate::objective::Tree;
use crate::search::{Worker, play_on};

/// Per objective kind: GA expansions towards it, those that met it, those
/// that added or improved another cell, cells added and improved, frames.
#[derive(Default, Clone, Copy, Debug)]
pub struct KindStats {
    pub tried: u64,
    pub met: u64,
    pub succ: u64,
    pub added: u64,
    pub improved: u64,
    pub frames: u64,
}

impl KindStats {
    /// The learned weight multiplier: the posterior mean success rate under a
    /// uniform prior.
    pub fn rate(&self) -> f64 {
        (self.succ as f64 + 1.0) / (self.tried as f64 + 2.0)
    }

    pub fn json(&self) -> Value {
        json!({"tried": self.tried, "met": self.met, "succ": self.succ, "added": self.added, "improved": self.improved, "frames": self.frames, "rate": self.rate()})
    }
}

/// The GA's totals.
#[derive(Default, Clone, Copy, Debug)]
pub struct GaStats {
    /// GA batches.
    pub batches: u64,
    /// GA expansions run (one objective tried each), and pellets made instead
    /// (modal state, or no objective could start there).
    pub expansions: u64,
    pub fallbacks: u64,
    /// Objectives that could start, summed over expansions.
    pub eligible: u64,
    pub met: u64,
    /// Expansions that added or improved another cell.
    pub succ: u64,
    pub added: u64,
    pub improved: u64,
    /// Expansions dropped because their source was replaced earlier in the batch.
    pub stale: u64,
    pub frames: u64,
}

/// A GA expansion's result: its rollouts (all from `src`) for the archive,
/// or a fallback pellet (`op` not `Ga`).
pub struct GaOut {
    pub src: usize,
    pub p: f32,
    pub op: Operator,
    pub parts: Vec<Rollout>,
    pub kind: Option<&'static str>,
    pub met: bool,
    pub eligible: usize,
    /// Every frame emulated: the attempt's and the harvest replays.
    pub frames: u64,
}

/// The objectives that may start from `st`, with their pick weights.
pub fn eligible(st: &St, rng: &mut SmallRng, kinds: Option<&BTreeMap<&'static str, KindStats>>) -> Vec<(Arc<dyn Objective>, f64)> {
    if st.modal() {
        return Vec::new();
    }
    // OBRRT_SPEC: plus the declarative objectives (none when unset).
    (obrrt_objectives::Registry.generate(st, rng).into_iter().chain(obrrt_spec::extra(st)))
        .filter(|o| !o.modal() && o.base_criteria(st) && o.exists_in(st) && start_eta(o.as_ref(), st).is_finite())
        .map(|o| {
            let m = kinds.map_or(1.0, |k| k.get(o.kind()).copied().unwrap_or_default().rate());
            let w = o.obj_weight(st) * m;
            (o, w)
        })
        .filter(|(_, w)| w.is_finite() && *w > 0.0)
        .collect()
}

/// Index drawn with probability ∝ weight.
fn pick(weights: impl Iterator<Item = f64> + Clone, rng: &mut SmallRng) -> usize {
    let total: f64 = weights.clone().sum();
    let mut x = rng.random::<f64>() * total;
    let mut last = 0;
    for (i, w) in weights.enumerate() {
        last = i;
        if x < w {
            return i;
        }
        x -= w;
    }
    last
}

/// One GA expansion of cell `src` (see the module doc).
pub(crate) fn expand(w: &mut Worker, archive: &Archive, tree: &Tree, cfg: &Config, kinds: &BTreeMap<&'static str, KindStats>, src: usize, mut rng: SmallRng) -> Result<GaOut> {
    let cell = &archive.cells[src];
    let p = cell.p();
    archive.restore(&mut w.ga.base, cell.node as usize)?;
    let st = read_state(&w.ga.base, cell.frames, true);
    let cands = eligible(&st, &mut rng, cfg.ga.learn.then_some(kinds));
    if cands.is_empty() {
        let (op, inputs) = plan(archive, cfg, src, &mut rng);
        let r = w.play(archive, tree, cfg, src, inputs, false)?;
        let frames = r.inputs.len() as u64;
        return Ok(GaOut { src, p, op, parts: vec![r], kind: None, met: false, eligible: 0, frames });
    }
    let (obj, _) = &cands[pick(cands.iter().map(|c| c.1), &mut rng)];
    let obj = obj.clone();

    let novel = |_: u64, _: u32| true;
    let pack = |_: &SaveState| Delta::none();
    let req = DenseReq {
        novel: &novel,
        // No saves: the harvest replays the inputs itself.
        replay_max: u32::MAX,
        start_dist: 0,
        pack: &pack,
    };
    let dense = (cfg.ga.harvest != Harvest::Winner).then_some(&req);
    let att = w.ga.attempt_full(&[obj.as_ref()], &st, None, &mut rng, cfg.ga.params, None, dense)?;
    let met = att.segments.first().is_some_and(|s| s.success);
    let mut frames = att.frames;

    // The rollouts to merge: the winner's committed inputs, then (ends / all)
    // each last-generation rollout's.
    let mut runs: Vec<(Vec<GamePad>, bool)> = Vec::new();
    let winner: Vec<GamePad> = att.segments.iter().flat_map(|s| s.inputs.iter().copied()).collect();
    if !winner.is_empty() {
        runs.push((winner, true));
    }
    if let Some(d) = &att.dense {
        for c in &d.cands {
            if !c.inputs.is_empty() {
                runs.push((c.inputs.clone(), cfg.ga.harvest == Harvest::All));
            }
        }
    }
    let mut parts = Vec::with_capacity(runs.len());
    let g = &mut w.ga;
    for (inputs, every) in runs {
        g.tmp.sync_from(&g.base)?;
        frames += inputs.len() as u64;
        parts.push(play_on(&mut g.tmp, archive, tree, cfg, src, inputs, false, every)?);
    }
    Ok(GaOut { src, p, op: Operator::Ga, parts, kind: Some(obj.kind()), met, eligible: cands.len(), frames })
}
