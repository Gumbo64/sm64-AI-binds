//! `OBRRT_GA_QD=1`: a MAP-Elites archive inside the open GA (ga.rs,
//! `attempt_open_inner`), so it keeps alternative ways alive instead of
//! converging on one mode. Off by default; off, nothing here is read and
//! the GA makes the same draws as before.
//!
//! - **Descriptor** of a rollout (per γ, at that γ's cut: its argmax frame,
//!   or the last stage's Done): fields from `OBRRT_GA_QD_FIELDS` (default
//!   `pos,yaw,speed,action,done`; also `side`): the position cell
//!   (`OBRRT_GA_QD_GRID`, 128 units), the facing octant, |forward speed| / 16
//!   (capped at 7), the action group (ground, air, swim, slide, other), the
//!   met flag, and `side`: which side of the start → target line the
//!   rollout's midpoint is on.
//! - **Archive:** the best rollout per (γ, bin), by that γ's fitness, then
//!   the earlier Done (or cut) frame. BTreeMap: iteration order is the bins'.
//! - **Parents:** with probability `OBRRT_GA_QD_UNIFORM` (0.5) a uniform
//!   pick (a γ uniformly, then one of its bins uniformly), else the
//!   fitness-ranked elite as before. The elite truncation and population
//!   size are unchanged; the winner is still the best under the primary γ.
//! - **Several γ** (`OBRRT_GA_GAMMAS=0.98,0.99,...`): each rollout folds one
//!   discounted running sum per γ (same rewards), one fitness and argmax
//!   each; γ index 0 is the primary (`OBRRT_GA_GAMMA`) and is always present.
//! - **Picks:** the K (`OBRRT_GA_QD_K`, 4) best distinct-bin rollouts, met
//!   ones first (earliest Done), at least the best met one per γ, then by
//!   fitness. [`take_picks`] hands the last attempt's to the caller on this
//!   thread (`legs`' beam); `OBRRT_GA_QD_ADD=1` also offers them to the
//!   dense collector (obrrt2: extra dense candidates).

use std::collections::BTreeMap;
use std::sync::Mutex;

use obrrt_core::{GamePad, St};
use rand::Rng;
use rand::rngs::SmallRng;

#[derive(Debug, Clone)]
pub struct Qd {
    pub grid: f32,
    pub uniform: f64,
    pub k: usize,
    pub add: bool,
    pub fields: Vec<String>,
    /// γ 0 is the primary.
    pub gammas: Vec<f64>,
}

pub fn settings() -> Option<&'static Qd> {
    obrrt_core::per_config!(Option<Qd>, |cfg| {
        if !cfg.research.qd {
            return None;
        }
        let f = |k: &str, d: f64| obrrt_core::sys::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let primary = cfg.ga.gamma;
        let mut gammas = vec![primary];
        if let Some(v) = &cfg.research.qd_gammas {
            for g in v.split(',').filter_map(|x| x.trim().parse::<f64>().ok()) {
                if !gammas.iter().any(|x| (x - g).abs() < 1e-12) {
                    gammas.push(g);
                }
            }
        }
        let fields = obrrt_core::sys::env::var("OBRRT_GA_QD_FIELDS")
            .unwrap_or_else(|_| "pos,yaw,speed,action,done".into())
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        Some(Qd {
            grid: f("OBRRT_GA_QD_GRID", 128.0) as f32,
            uniform: f("OBRRT_GA_QD_UNIFORM", 0.5),
            k: f("OBRRT_GA_QD_K", 4.0) as usize,
            add: obrrt_core::sys::env::var("OBRRT_GA_QD_ADD").is_ok_and(|v| v == "1"),
            fields,
            gammas,
        })
    })
    .as_ref()
}

/// The extra γ's (beyond the primary) rollouts fold, empty when QD is off
/// or has one γ.
pub fn extra_gammas() -> &'static [f64] {
    settings().map_or(&[], |q| &q.gammas[1..])
}

/// A bin: (γ index, descriptor fields).
pub type Bin = (u8, [i32; 7]);

fn action_group(a: u32) -> i32 {
    // sm64.h: ACT_FLAG_AIR, _SWIMMING, _BUTT_OR_STOMACH_SLIDE / DIVING; groups.
    if a & 0x2000 != 0 || a & 0x1C0 == 0x0C0 {
        2
    } else if a & 0x0004_0000 != 0 || a & 0x0008_0000 != 0 && a & 0x800 == 0 {
        3
    } else if a & 0x800 != 0 {
        1
    } else if a & 0x1C0 <= 0x040 {
        0
    } else {
        4
    }
}

/// The descriptor of a rollout ending at `end` (met = all stages met),
/// `mid` its midpoint state, `target` the first stage's steer target.
pub fn descriptor(q: &Qd, g: u8, end: &St, met: bool, start: &St, mid: &St, target: Option<[f32; 3]>) -> Bin {
    let mut d = [i32::MIN; 7];
    for f in &q.fields {
        match f.as_str() {
            "pos" => {
                d[0] = (end.pos[0] / q.grid).floor() as i32;
                d[1] = (end.pos[1] / q.grid).floor() as i32;
                d[2] = (end.pos[2] / q.grid).floor() as i32;
            }
            "yaw" => d[3] = ((end.yaw as u16) >> 13) as i32,
            "speed" => d[4] = ((end.forward_vel.abs() / 16.0) as i32).min(7),
            "action" => d[5] = action_group(end.action),
            "done" => d[6] = met as i32,
            "side" => {
                if let Some(t) = target {
                    let (ax, az) = (t[0] - start.pos[0], t[2] - start.pos[2]);
                    let (bx, bz) = (mid.pos[0] - start.pos[0], mid.pos[2] - start.pos[2]);
                    let c = ax * bz - az * bx;
                    // Packed into the action slot's high bits.
                    d[5] = d[5].max(0) + if c > 0.0 { 100 } else { 200 };
                }
            }
            _ => {}
        }
    }
    d.iter_mut().for_each(|x| if *x == i32::MIN { *x = 0 });
    (g, d)
}

/// One archived rollout.
#[derive(Clone)]
pub struct Entry {
    pub genome: Vec<GamePad>,
    pub plan: u32,
    pub key: Vec<usize>,
    pub fitness: f64,
    /// Inputs to its end (the Done frame when met, else the γ's cut).
    pub end_t: usize,
    pub met: bool,
    pub end: St,
}

impl Entry {
    /// Greater is better: fitness, then earlier end.
    fn better(&self, o: &Entry) -> std::cmp::Ordering {
        self.fitness.total_cmp(&o.fitness).then(o.end_t.cmp(&self.end_t))
    }
}

#[derive(Default)]
pub struct Archive {
    pub bins: BTreeMap<Bin, Entry>,
}

impl Archive {
    pub fn insert(&mut self, bin: Bin, e: Entry) {
        match self.bins.get(&bin) {
            Some(old) if old.better(&e).is_ge() => {}
            _ => {
                self.bins.insert(bin, e);
            }
        }
    }

    /// A γ uniformly, then one of its bins uniformly.
    pub fn pick(&self, rng: &mut SmallRng, n_gammas: usize) -> Option<&Entry> {
        if self.bins.is_empty() {
            return None;
        }
        let g = rng.random_range(0..n_gammas.max(1)) as u8;
        let of_g: Vec<&Entry> = self.bins.range((g, [i32::MIN; 7])..=(g, [i32::MAX; 7])).map(|x| x.1).collect();
        if of_g.is_empty() {
            let i = rng.random_range(0..self.bins.len());
            return self.bins.values().nth(i);
        }
        Some(of_g[rng.random_range(0..of_g.len())])
    }

    /// The K best distinct-bin picks (descriptor bins, ignoring γ): met ones
    /// first, at least the best met one per γ, then by fitness.
    pub fn picks(&self, k: usize) -> Vec<Pick> {
        let mut seen: std::collections::BTreeSet<[i32; 7]> = Default::default();
        let mut out: Vec<Pick> = Vec::new();
        let mk = |b: &Bin, e: &Entry, gammas: &[f64]| Pick {
            genome: e.genome[..e.end_t.min(e.genome.len())].to_vec(),
            met: e.met,
            gamma: gammas.get(b.0 as usize).copied().unwrap_or(0.0),
            gamma_index: b.0,
            end_t: e.end_t,
            fitness: e.fitness,
            bin: b.1,
            end: e.end.clone(),
        };
        let gammas = settings().map_or(vec![], |q| q.gammas.clone());
        // The best met one per γ.
        for g in 0..gammas.len().max(1) as u8 {
            let best = self
                .bins
                .iter()
                .filter(|(b, e)| b.0 == g && e.met)
                .min_by(|a, b| a.1.end_t.cmp(&b.1.end_t).then(b.1.fitness.total_cmp(&a.1.fitness)).then(a.0.cmp(b.0)));
            if let Some((b, e)) = best {
                if seen.insert(b.1) {
                    out.push(mk(b, e, &gammas));
                }
            }
        }
        // Then met ones by end, then the rest by fitness.
        let mut rest: Vec<(&Bin, &Entry)> = self.bins.iter().collect();
        rest.sort_by(|a, b| {
            b.1.met.cmp(&a.1.met).then_with(|| if a.1.met { a.1.end_t.cmp(&b.1.end_t) } else { b.1.fitness.total_cmp(&a.1.fitness) }).then(a.0.cmp(b.0))
        });
        for (b, e) in rest {
            if out.len() >= k {
                break;
            }
            if seen.insert(b.1) {
                out.push(mk(b, e, &gammas));
            }
        }
        out
    }
}

/// One of an attempt's distinct outcomes.
#[derive(Clone)]
pub struct Pick {
    /// From the attempt's start to its end.
    pub genome: Vec<GamePad>,
    pub met: bool,
    pub gamma: f64,
    pub gamma_index: u8,
    pub end_t: usize,
    pub fitness: f64,
    pub bin: [i32; 7],
    /// The end state (without objects).
    pub end: St,
}

/// What the last QD attempt on a thread left.
pub struct Out {
    pub picks: Vec<Pick>,
    pub bins: usize,
    /// [0]: distinct descriptor bins with a met rollout; [g > 0]: those of
    /// them where γ g's best rollout differs from γ 0's.
    pub met_bins: Vec<usize>,
}

thread_local! {
    static LAST: std::cell::RefCell<Option<Out>> = const { std::cell::RefCell::new(None) };
}

pub fn set_last(o: Out) {
    LAST.with(|l| *l.borrow_mut() = Some(o));
}

/// The last QD attempt's picks on this thread (taken).
pub fn take_picks() -> Option<Out> {
    LAST.with(|l| l.borrow_mut().take())
}

#[derive(Default)]
struct Stats {
    attempts: u64,
    bins: u64,
    picks: u64,
    met_picks: u64,
    uniform_parents: u64,
    elite_parents: u64,
    added: u64,
    met_bins_by_gamma: Vec<u64>,
}

/// This thread's runtime's tallies (the engine's own inside its threads).
fn stats() -> std::sync::Arc<Mutex<Stats>> {
    obrrt_core::runtime::current().ext::<Mutex<Stats>>()
}

pub fn note(out: &Out, uniform: u64, elite: u64, added: u64) {
    let stats = stats();
    let mut s = stats.lock().unwrap();
    s.attempts += 1;
    s.bins += out.bins as u64;
    s.picks += out.picks.len() as u64;
    s.met_picks += out.picks.iter().filter(|p| p.met).count() as u64;
    s.uniform_parents += uniform;
    s.elite_parents += elite;
    s.added += added;
    if s.met_bins_by_gamma.len() < out.met_bins.len() {
        s.met_bins_by_gamma.resize(out.met_bins.len(), 0);
    }
    for (i, m) in out.met_bins.iter().enumerate() {
        s.met_bins_by_gamma[i] += *m as u64;
    }
}

/// summary.json's `ga_qd` block (None when off).
pub fn summary() -> Option<serde_json::Value> {
    let q = settings()?;
    let stats = stats();
    let s = stats.lock().unwrap();
    let n = s.attempts.max(1) as f64;
    Some(serde_json::json!({
        "grid": q.grid, "uniform": q.uniform, "k": q.k, "add": q.add, "fields": q.fields, "gammas": q.gammas,
        "attempts": s.attempts,
        "bins_per_attempt": s.bins as f64 / n,
        "picks_per_attempt": s.picks as f64 / n,
        "met_picks_per_attempt": s.met_picks as f64 / n,
        "met_bins_by_gamma": s.met_bins_by_gamma,
        "parents": {"uniform": s.uniform_parents, "elite": s.elite_parents},
        "dense_offered": s.added,
    }))
}
