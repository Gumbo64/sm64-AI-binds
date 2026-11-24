//! Policy-seeded genomes for the GA (`OBRRT_MOVE_GA=1`, off by default).
//!
//! The registered mover ([`obrrt_core::mover::Mover`]) may offer forcible
//! policies (one per move of a movement library). Instead of one
//! classifier choosing a move per frame, the GA's first generation then
//! simulates closed-loop rollouts of single policies and of combinations
//! ("policy A for n frames, then B, then C") and keeps the pads actually
//! played as genomes, which later generations mutate as usual. The
//! `repolicy` mutation re-simulates a span of a genome under another
//! policy. Which moves matter is learned from [`Stats`] (summary.json's
//! `move_ga`, the log).
//!
//! Nothing here knows what a policy does: indices, names, a plausibility
//! pre-filter and the per-frame pad all come from the mover.
//!
//! Determinism: plans are drawn from the attempt's RNG (only when on) and
//! the mover's functions are pure; the stats are diagnostics only and
//! never read back by the search.

use std::collections::{BTreeMap, HashSet};
use std::sync::{Mutex, OnceLock};

use obrrt_core::{GamePad, Objective, St};
use rand::Rng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

/// Segment lengths a combination's non-final segments take (frames).
pub const LENS: [usize; 4] = [12, 24, 32, 48];

/// `OBRRT_MOVE_GA=1`.
/// Turns the policy-seeded genomes off for a while (spec's legs: a leg
/// with `"moves": "off"`); unset, `enabled` is OBRRT_MOVE_GA as before.
static FORCED_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_forced_off(off: bool) {
    FORCED_OFF.store(off, std::sync::atomic::Ordering::Relaxed);
}

/// Only these policies (by name) are offered for a while (spec's legs:
/// `"moves": "kicks"`); None = all, as before.
static ONLY: std::sync::Mutex<Option<Vec<&'static str>>> = std::sync::Mutex::new(None);

pub fn set_only(names: Option<Vec<&'static str>>) {
    *ONLY.lock().unwrap() = names;
}

pub fn enabled() -> bool {
    if FORCED_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| obrrt_core::sys::env::var("OBRRT_MOVE_GA").is_ok_and(|v| v == "1"))
}

/// `OBRRT_MOVE_GA_N` (default 8): most policy-seeded rollouts in a first
/// generation (in the [`slot`]s, at most half the population;
/// with [`extra`] on top of it; fewer when fewer plans survive the
/// plausibility filter).
pub fn budget_n() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| obrrt_core::sys::env::var("OBRRT_MOVE_GA_N").ok().and_then(|v| v.parse().ok()).unwrap_or(8))
}

/// Most policy-seeded rollouts for a while (spec's legs: half the leg's
/// population, unless `OBRRT_MOVE_GA_N` is set); 0: `OBRRT_MOVE_GA_N`.
static BUDGET: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_budget(n: usize) {
    BUDGET.store(n, std::sync::atomic::Ordering::Relaxed);
}

/// The first generation's budget given `plausible` policies: with
/// [`set_budget`], never less than one rollout for each plausible single
/// (`gen0_plans` plays all singles first, then combinations); else
/// [`budget_n`], whatever the policies.
pub fn gen0_budget(plausible: usize) -> usize {
    match BUDGET.load(std::sync::atomic::Ordering::Relaxed) {
        0 => budget_n(),
        n => n.max(plausible),
    }
}

/// Attempts with a seed genome also play the policies (spec's legs: a leg
/// seeded from its reference otherwise never plays one); off, as before, a
/// seed turns them off.
static WITH_SEED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_with_seed(on: bool) {
    WITH_SEED.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn with_seed() -> bool {
    WITH_SEED.load(std::sync::atomic::Ordering::Relaxed)
}

/// `OBRRT_MOVE_GA_EXTRA=1`: the policy-seeded rollouts are added on top
/// of the first generation's population (no random genome is replaced;
/// more frames per attempt). Default off: they take the odd (random) slots.
pub fn extra() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| obrrt_core::sys::env::var("OBRRT_MOVE_GA_EXTRA").is_ok_and(|v| v == "1"))
}

/// `OBRRT_MOVE_GA_SLOTS`: which first-generation slots policy rollouts
/// take without [`extra`]: `random` (the odd slots: random genomes),
/// `seeded` (even slots from 2: the controller rollouts with a noise
/// prefix, which policy rollouts most resemble), `mixed` (slots 4j+1 and
/// 4j+2), `auto` (default: `random` for a target to pass, `seeded` for
/// one to stop at or a door). In the lab, taking the random genomes' slots
/// cost door targets 20-40 frames (the random genomes' descendants find
/// the plain walk-in) and taking the seeded ones didn't; a BoB slope run
/// was the other way round (118 vs 150).
fn slots_mode() -> u8 {
    static M: OnceLock<u8> = OnceLock::new();
    *M.get_or_init(|| match obrrt_core::sys::env::var("OBRRT_MOVE_GA_SLOTS").as_deref() {
        Ok("random") => 0,
        Ok("seeded") => 1,
        Ok("mixed") => 2,
        _ => 3,
    })
}

/// The plan index first-generation slot `i` plays, if it is a policy slot
/// (`pass`: the target is reached by passing it).
pub fn slot(i: usize, pass: bool) -> Option<usize> {
    let mode = match slots_mode() {
        3 if pass => 0,
        3 => 1,
        m => m,
    };
    match mode {
        0 => (i % 2 == 1).then_some(i / 2),
        1 => (i % 2 == 0 && i >= 2).then_some(i / 2 - 1),
        _ => matches!(i % 4, 1 | 2).then_some(i / 4 * 2 + (i % 4 - 1)),
    }
}

/// How many policy slots a population of `n` has.
pub fn slots_in(n: usize, pass: bool) -> usize {
    (0..n).filter(|&i| slot(i, pass).is_some()).count()
}

/// `OBRRT_MOVE_GA_REPOLICY` (default 0.15): share of children made by the
/// `repolicy` mutation.
pub fn repolicy_p() -> f64 {
    static P: OnceLock<f64> = OnceLock::new();
    *P.get_or_init(|| obrrt_core::sys::env::var("OBRRT_MOVE_GA_REPOLICY").ok().and_then(|v| v.parse().ok()).unwrap_or(0.15))
}

/// `OBRRT_MOVE_GA_PRUNE` (default 1): the mover's plausibility filter.
fn prune() -> bool {
    static P: OnceLock<bool> = OnceLock::new();
    *P.get_or_init(|| !obrrt_core::sys::env::var("OBRRT_MOVE_GA_PRUNE").is_ok_and(|v| v == "0"))
}

/// Policies the mover offers (0 without one).
pub fn policies() -> usize {
    obrrt_core::mover::mover().map_or(0, |m| m.policies())
}

pub fn policy_name(p: usize) -> String {
    obrrt_core::mover::mover().map_or_else(|| "?".into(), |m| m.policy_name(p).to_string())
}

/// What a rollout plays after its prefix: policy segments in order (the
/// last one's length is ignored: it runs to the horizon unless `tail`
/// follows), then the `tail` pads, then the default steering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub segs: Vec<(usize, usize)>,
    pub tail: Vec<GamePad>,
}

/// One frame's source.
pub enum Step<'a> {
    Policy(usize),
    Pad(&'a GamePad),
    Steer,
}

impl Plan {
    pub fn single(p: usize) -> Plan {
        Plan { segs: vec![(p, usize::MAX)], tail: Vec::new() }
    }

    /// The source of the `k`th frame after the prefix.
    pub fn step(&self, k: usize) -> Step<'_> {
        let mut at = 0usize;
        let n = self.segs.len();
        for (i, &(p, len)) in self.segs.iter().enumerate() {
            let last = i + 1 == n && self.tail.is_empty();
            if last || k < at.saturating_add(len) {
                return Step::Policy(p);
            }
            at = at.saturating_add(len);
        }
        match self.tail.get(k - at) {
            Some(g) => Step::Pad(g),
            None => Step::Steer,
        }
    }

    /// E.g. `run24>long-jump`.
    pub fn label(&self) -> String {
        let n = self.segs.len();
        let mut s = String::new();
        for (i, &(p, len)) in self.segs.iter().enumerate() {
            if i > 0 {
                s.push('>');
            }
            s.push_str(&policy_name(p));
            if i + 1 < n || !self.tail.is_empty() {
                s.push_str(&len.to_string());
            }
        }
        if !self.tail.is_empty() {
            s.push_str(">pads");
        }
        s
    }

    fn policies(&self) -> impl Iterator<Item = usize> + '_ {
        self.segs.iter().map(|s| s.0)
    }
}

/// The frame's pad under `plan` (`k` frames after the prefix, `f` frames
/// into the rollout), towards `obj`'s target.
pub fn pad(plan: &Plan, k: usize, obj: &dyn Objective, st: &St, f: usize, every: usize) -> GamePad {
    match plan.step(k) {
        Step::Policy(p) => crate::steer::steer_policy(obj, st, f, every, Some(p)),
        Step::Pad(g) => *g,
        Step::Steer => crate::steer::steer(obj, st, f, every),
    }
}

/// Policies worth simulating from `start` towards `obj`'s target, with
/// the pruned ones counted in the stats.
pub fn plausible(obj: &dyn Objective, start: &St) -> Vec<usize> {
    let Some(m) = obrrt_core::mover::mover() else { return Vec::new() };
    let Some(target) = obj.steer_target(start) else { return Vec::new() };
    let mut out = Vec::new();
    let mut s = stats().lock().unwrap();
    let only = ONLY.lock().unwrap().clone();
    for p in 0..m.policies() {
        if only.as_ref().is_some_and(|o| !o.contains(&m.policy_name(p))) {
            continue;
        }
        let e = s.policy.entry(m.policy_name(p).to_string()).or_default();
        e.considered += 1;
        if !prune() || m.policy_plausible(p, start, target, obj.arrival(start)) {
            out.push(p);
        } else {
            e.pruned += 1;
        }
    }
    out
}

/// The first generation's plans (at most `n`): every plausible policy
/// alone (a random subset if fewer slots), then two- and three-segment
/// combinations of them with lengths from [`LENS`], all distinct.
pub fn gen0_plans(ok: &[usize], n: usize, rng: &mut SmallRng) -> Vec<Plan> {
    if ok.is_empty() || n == 0 {
        return Vec::new();
    }
    let mut singles: Vec<usize> = ok.to_vec();
    singles.shuffle(rng);
    let mut out: Vec<Plan> = singles.iter().take(n).map(|&p| Plan::single(p)).collect();
    if ok.len() < 2 {
        return out;
    }
    let mut seen: HashSet<Vec<(usize, usize)>> = out.iter().map(|p| p.segs.clone()).collect();
    let mut tries = 0;
    while out.len() < n && tries < 20 * n {
        tries += 1;
        let k = if rng.random_bool(1.0 / 3.0) { 3 } else { 2 };
        let mut segs: Vec<(usize, usize)> = Vec::with_capacity(k);
        for i in 0..k {
            let p = loop {
                let p = ok[rng.random_range(0..ok.len())];
                if segs.last().is_none_or(|s| s.0 != p) {
                    break p;
                }
            };
            let len = if i + 1 < k { LENS[rng.random_range(0..LENS.len())] } else { usize::MAX };
            segs.push((p, len));
        }
        if seen.insert(segs.clone()) {
            out.push(Plan { segs, tail: Vec::new() });
        }
    }
    out
}

/// A `repolicy` child of `genome`: the frames [i, j) re-simulated under a
/// random plausible policy (closed loop from the state at i), the
/// parent's frames from j on after it. Returns (prefix, plan).
pub fn repolicy(genome: &[GamePad], ok: &[usize], horizon: usize, rng: &mut SmallRng) -> Option<(Vec<GamePad>, Plan)> {
    if ok.is_empty() {
        return None;
    }
    let len = genome.len().min(horizon);
    let i = rng.random_range(0..len.max(1)).min(len);
    let n = LENS[rng.random_range(0..LENS.len())];
    let p = ok[rng.random_range(0..ok.len())];
    let j = (i + n).min(len);
    let tail = if j < len { genome[j..len].to_vec() } else { Vec::new() };
    Some((genome[..i].to_vec(), Plan { segs: vec![(p, n)], tail }))
}

// ---------------------------------------------------------------------------
// Stats (diagnostics only).

#[derive(Debug, Default, Clone)]
pub struct PolicyStat {
    /// First generations it was considered for, and pruned in.
    pub considered: u64,
    pub pruned: u64,
    /// Policy-seeded rollouts containing it.
    pub simulated: u64,
    /// The first generation's best was a plan containing it.
    pub gen0_best: u64,
    /// The attempt's winner was such a rollout itself / descends from one.
    pub wins: u64,
    pub lineage_wins: u64,
    /// `repolicy` children under it, and attempt winners among them.
    pub repolicy: u64,
    pub repolicy_wins: u64,
}

#[derive(Debug, Default, Clone)]
pub struct ComboStat {
    pub simulated: u64,
    pub gen0_best: u64,
    pub wins: u64,
    pub lineage_wins: u64,
}

#[derive(Debug, Default)]
pub struct Stats {
    pub attempts: u64,
    pub policy: BTreeMap<String, PolicyStat>,
    pub combo: BTreeMap<String, ComboStat>,
}

pub fn stats() -> &'static Mutex<Stats> {
    static S: OnceLock<Mutex<Stats>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(Stats::default()))
}

/// How a GA attempt's policy rollouts did.
pub struct Outcome<'a> {
    /// The attempt's policy-seeded plans (simulated).
    pub plans: &'a [Plan],
    /// Policies of the `repolicy` children made.
    pub repolicies: &'a [usize],
    /// Index into `plans` of the first generation's best, if it was one.
    pub gen0_best: Option<usize>,
    /// The winner's lineage plan (index into `plans`), and whether the
    /// winner is that rollout itself.
    pub winner: Option<(usize, bool)>,
    /// The winner's policy if a `repolicy` child made it.
    pub winner_repolicy: Option<usize>,
}

pub fn record(o: &Outcome) {
    let mut s = stats().lock().unwrap();
    s.attempts += 1;
    let labels: Vec<String> = o.plans.iter().map(|p| p.label()).collect();
    for (pl, l) in o.plans.iter().zip(&labels) {
        s.combo.entry(l.clone()).or_default().simulated += 1;
        let mut ps: Vec<usize> = pl.policies().collect();
        ps.sort_unstable();
        ps.dedup();
        for p in ps {
            s.policy.entry(policy_name(p)).or_default().simulated += 1;
        }
    }
    let bump = |s: &mut Stats, i: usize, f: fn(&mut PolicyStat), g: fn(&mut ComboStat)| {
        g(s.combo.entry(labels[i].clone()).or_default());
        let mut ps: Vec<usize> = o.plans[i].policies().collect();
        ps.sort_unstable();
        ps.dedup();
        for p in ps {
            f(s.policy.entry(policy_name(p)).or_default());
        }
    };
    if let Some(i) = o.gen0_best {
        bump(&mut s, i, |e| e.gen0_best += 1, |e| e.gen0_best += 1);
    }
    if let Some((i, direct)) = o.winner {
        bump(&mut s, i, |e| e.lineage_wins += 1, |e| e.lineage_wins += 1);
        if direct {
            bump(&mut s, i, |e| e.wins += 1, |e| e.wins += 1);
        }
    }
    for &p in o.repolicies {
        s.policy.entry(policy_name(p)).or_default().repolicy += 1;
    }
    if let Some(p) = o.winner_repolicy {
        s.policy.entry(policy_name(p)).or_default().repolicy_wins += 1;
    }
}

/// The stats as JSON (summary.json's `move_ga`): per policy, and the 40
/// combinations with the most lineage wins.
pub fn stats_json() -> serde_json::Value {
    let s = stats().lock().unwrap();
    let policy: serde_json::Map<String, serde_json::Value> = s
        .policy
        .iter()
        .map(|(k, e)| {
            let rate = if e.simulated > 0 { e.lineage_wins as f64 / e.simulated as f64 } else { 0.0 };
            (
                k.clone(),
                serde_json::json!({
                    "considered": e.considered, "pruned": e.pruned, "simulated": e.simulated,
                    "gen0_best": e.gen0_best, "wins": e.wins, "lineage_wins": e.lineage_wins,
                    "win_rate": rate, "repolicy": e.repolicy, "repolicy_wins": e.repolicy_wins,
                }),
            )
        })
        .collect();
    let mut combos: Vec<(&String, &ComboStat)> = s.combo.iter().collect();
    combos.sort_by(|a, b| b.1.lineage_wins.cmp(&a.1.lineage_wins).then(b.1.gen0_best.cmp(&a.1.gen0_best)).then(a.0.cmp(b.0)));
    let combo: Vec<serde_json::Value> = combos
        .iter()
        .take(40)
        .map(|(k, e)| {
            serde_json::json!({"plan": k, "simulated": e.simulated, "gen0_best": e.gen0_best, "wins": e.wins, "lineage_wins": e.lineage_wins})
        })
        .collect();
    serde_json::json!({"attempts": s.attempts, "policy": policy, "combos": combo, "distinct_combos": s.combo.len()})
}

/// One log line: per policy `name sim/pruned g0best/wins/lineage`, by
/// lineage wins.
pub fn stats_line() -> String {
    let s = stats().lock().unwrap();
    let mut v: Vec<(&String, &PolicyStat)> = s.policy.iter().collect();
    v.sort_by(|a, b| b.1.lineage_wins.cmp(&a.1.lineage_wins).then(a.0.cmp(b.0)));
    let parts: Vec<String> = v
        .iter()
        .map(|(k, e)| format!("{k} {}/{}p {}g0/{}w/{}l rp{}/{}", e.simulated, e.pruned, e.gen0_best, e.wins, e.lineage_wins, e.repolicy, e.repolicy_wins))
        .collect();
    format!("move-ga: {} attempts; {}", s.attempts, parts.join(", "))
}
