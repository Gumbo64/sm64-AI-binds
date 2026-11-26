//! Speed-up switches for re-attempts of known milestones (all off by
//! default; with all off the search is byte-identical to before):
//!
//! * `OBRRT_FASTER_ONLY=1`: a success of a polish attempt, or of an attempt
//!   whose objective is a known milestone (some node already met that
//!   objective key from a start of the same progress context, [`ctx`]), counts (adds a node by the
//!   soft dedupe, counts as a success for decay, rate and kind learning)
//!   only if it meets it strictly sooner (game frame) than the best so far
//!   for that key (and for a polish job, than the target's best), or sets
//!   a new (keys, stars) best (`Diagnostics::note_best`'s key). Otherwise it
//!   is a failure and adds a node only if new by the hard dedupe (a cell no
//!   state at or before its frame has).
//! * `OBRRT_BEAT=1`: beat-the-record objectives, [`Beat`] (wrapping any
//!   non-modal, non-fixed objective O met at frame F), at most one live per
//!   O key: a sooner meet replaces it by a tighter one.
//! * `OBRRT_BNB=1`: branch and bound in the GA's rollouts: a stage stops
//!   once `s.frame + eta_opt(s)` reaches `OBRRT_BNB_SLACK` (0) past the best
//!   frame any node met its objective key at from the stage start's context
//!   (a snapshot taken when the round's plans are made).
//!
//! `OBRRT_BEAT_OPT` (0.7) is the optimism factor both use: ETAs aren't lower
//! bounds, so a state is written off only if even 0.7 of its ETA misses.
//!
//! Always on (output only): `best_improved` events in events.jsonl and the
//! `speed` block of summary.json (per-objective-key and per-(keys, stars)
//! best-time improvements, frames saved, by kind).

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use obrrt_core::{Arrival, Budget, DenseKeep, Eta, GaBudget, Mem, Objective, PadOverlay, Precision, St};

pub struct Settings {
    pub faster_only: bool,
    pub beat: bool,
    pub bnb: bool,
    /// `OBRRT_BEAT_OPT`: optimism factor on ETAs (Beat's `Lost`, BnB's bound).
    pub opt: f64,
    /// `OBRRT_BEAT_W`: a Beat's weight relative to its objective's.
    pub beat_w: f64,
    /// `OBRRT_BEAT_BUCKET`: frames per deadline bucket in a Beat's key.
    pub bucket: u32,
    /// `OBRRT_BNB_SLACK`: frames past the bound before BnB stops a stage.
    pub slack: f64,
}

pub fn settings() -> &'static Settings {
    static S: OnceLock<Settings> = OnceLock::new();
    S.get_or_init(|| {
        let sp = &obrrt_core::config::current().research.speed;
        let num = |k: &str, d: f64| obrrt_core::sys::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        Settings {
            faster_only: sp.faster_only,
            beat: sp.beat,
            bnb: sp.bnb,
            opt: num("OBRRT_BEAT_OPT", 0.7),
            beat_w: num("OBRRT_BEAT_W", 0.5),
            bucket: (num("OBRRT_BEAT_BUCKET", 10.0) as u32).max(1),
            slack: num("OBRRT_BNB_SLACK", 0.0),
        }
    })
}

/// A meet's progress context: the attempt start's keys, stars and (with
/// OBRRT_STARSET_KEY) a hash of its collected set (`tree::set_key`). Bests
/// (FASTER_ONLY's bar, BnB's bound, Beat deadlines) are per (objective key,
/// context): meeting an objective later in the route, with more progress,
/// is not bounded by its meets with less.
pub type Ctx = (u8, i16, u64);

pub fn ctx(st: &St) -> Ctx {
    let set = if crate::engine::STARSET.load(Ordering::Relaxed) {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        crate::tree::set_key(st).hash(&mut h);
        h.finish()
    } else {
        0
    };
    (st.keys, st.stars, set)
}

/// A `&'static str` for a kind name made at run time (kept forever; one per
/// distinct name).
pub fn intern(s: String) -> &'static str {
    static T: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let mut t = T.get_or_init(Default::default).lock().unwrap();
    if let Some(&k) = t.get(&s) {
        return k;
    }
    let k: &'static str = Box::leak(s.clone().into_boxed_str());
    t.insert(s, k);
    k
}

/// `Beat(O, F)`: meet O before game frame F. Kind `"Beat::<O kind>"`, key
/// O's key plus the context and F's bucket; O's eligibility (from starts of
/// the context F was met from only) and ETA, except `Lost` once
/// `s.frame + opt × eta(s) >= F` (and a `Done` at or after F is `Lost`).
pub struct Beat {
    pub inner: Arc<dyn Objective>,
    pub deadline: u32,
    pub ctx: Ctx,
    kind: &'static str,
    key: String,
    opt: f64,
    w: f64,
}

pub fn beat_key(okey: &str, c: Ctx, deadline: u32) -> String {
    format!("{okey}#beat:k{}s{}c{:x}:{}", c.0, c.1, c.2, deadline / settings().bucket)
}

impl Beat {
    pub fn new(inner: Arc<dyn Objective>, c: Ctx, deadline: u32) -> Beat {
        Beat::with_opt(inner, c, deadline, settings().opt)
    }

    /// With optimism factor `opt` instead of `OBRRT_BEAT_OPT` (OBRRT_ETA_LEARN:
    /// `min(c, 1) × OBRRT_BEAT_OPT` for the inner kind, see etalearn.rs).
    pub fn with_opt(inner: Arc<dyn Objective>, c: Ctx, deadline: u32, opt: f64) -> Beat {
        let s = settings();
        let kind = intern(format!("Beat::{}", inner.kind()));
        let key = beat_key(&inner.key(), c, deadline);
        Beat { inner, deadline, ctx: c, kind, key, opt, w: s.beat_w }
    }
}

impl Objective for Beat {
    fn kind(&self) -> &'static str {
        self.kind
    }
    fn key(&self) -> String {
        self.key.clone()
    }
    fn obj_weight(&self, b: &St) -> f64 {
        // The pool multiplies by the Beat kind's own table weight (1 unless
        // set); O's kind weight comes along so W is relative to O's entry.
        self.inner.obj_weight(b) * self.w * crate::pool::kind_weight(self.inner.kind())
    }
    fn base_criteria(&self, s: &St) -> bool {
        ctx(s) == self.ctx && self.inner.base_criteria(s)
    }
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        let f = s.frame as f64;
        let dl = self.deadline as f64;
        match self.inner.eta(s, start, mem) {
            Eta::Done if s.frame < self.deadline => Eta::Done,
            Eta::Done | Eta::Lost => Eta::Lost,
            Eta::Frames(x) => {
                if f + self.opt * x.max(0.0) >= dl {
                    Eta::Lost
                } else {
                    Eta::Frames(x)
                }
            }
        }
    }
    fn fixed(&self) -> Option<Budget> {
        self.inner.fixed()
    }
    fn ga_budget(&self) -> Option<GaBudget> {
        self.inner.ga_budget()
    }
    fn precision(&self) -> Precision {
        self.inner.precision()
    }
    fn kinship(&self) -> u32 {
        self.inner.kinship()
    }
    fn dense_keep(&self) -> DenseKeep {
        self.inner.dense_keep()
    }
    fn start_bias(&self, s: &St) -> f64 {
        self.inner.start_bias(s)
    }
    fn done_frames(&self, s: &St, mem: &Mem) -> f64 {
        self.inner.done_frames(s, mem)
    }
    fn temperature(&self) -> f64 {
        self.inner.temperature()
    }
    fn needs_objects(&self) -> bool {
        self.inner.needs_objects()
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.inner.steer_target(s)
    }
    fn success_value(&self) -> f64 {
        self.inner.success_value()
    }
    fn steer_button(&self, s: &St) -> u16 {
        self.inner.steer_button(s)
    }
    fn arrival(&self, s: &St) -> Arrival {
        self.inner.arrival(s)
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        self.inner.pad_overlay()
    }
    fn modal(&self) -> bool {
        self.inner.modal()
    }
    fn exists_in(&self, s: &St) -> bool {
        self.inner.exists_in(s)
    }
}

// ---- branch and bound (GA rollouts) ----

/// An attempt's bounds: per stage its objective key (None: no bound), and
/// the round-start snapshot of the best meet per (key, context). A stage's
/// bound is looked up with the context of the state the stage starts from.
pub struct BoundsIn {
    pub keys: Vec<Option<String>>,
    pub best: Arc<HashMap<(String, Ctx), u32>>,
    /// Per stage, the optimism factor on its ETA: `OBRRT_BEAT_OPT`, or with
    /// OBRRT_ETA_LEARN `min(c_kind, 1) ×` that (round-start snapshot).
    pub opt: Vec<f64>,
}

impl BoundsIn {
    pub fn get(&self, k: usize, from: &St) -> Option<u32> {
        let key = self.keys.get(k)?.as_ref()?;
        self.best.get(&(key.clone(), ctx(from))).copied()
    }

    /// Stage `k`'s optimism factor.
    pub fn opt(&self, k: usize) -> f64 {
        self.opt.get(k).copied().unwrap_or(settings().opt)
    }
}

pub type Bounds = Arc<BoundsIn>;

thread_local! {
    static BOUNDS: RefCell<Option<Bounds>> = const { RefCell::new(None) };
}

/// Sets (Some) or clears (None) this thread's bounds for the attempt about
/// to run on it.
pub fn set_bounds(b: Option<Bounds>) {
    BOUNDS.with(|x| *x.borrow_mut() = b);
}

pub(crate) fn bounds() -> Option<Bounds> {
    BOUNDS.with(|x| x.borrow().clone())
}

/// Rollouts stopped by the bound, and the frames of their planned length
/// left unplayed (an upper bound on the frames saved: some would have ended
/// sooner anyway). Sums, so the same whatever the thread interleaving.
pub static BNB_STOPS: AtomicU64 = AtomicU64::new(0);
pub static BNB_SAVED: AtomicU64 = AtomicU64::new(0);
/// Rollouts played with a bound in force.
pub static BNB_ROLLOUTS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn bnb_note(stopped: bool, saved: u64) {
    BNB_ROLLOUTS.fetch_add(1, Ordering::Relaxed);
    if stopped {
        BNB_STOPS.fetch_add(1, Ordering::Relaxed);
        BNB_SAVED.fetch_add(saved, Ordering::Relaxed);
    }
}

// ---- engine-side bookkeeping ----

#[derive(Default, Clone, Debug, serde::Serialize)]
pub struct Imp {
    pub improvements: u64,
    pub frames_saved: u64,
}

impl Imp {
    fn add(&mut self, saved: u64) {
        self.improvements += 1;
        self.frames_saved += saved;
    }
}

#[derive(Default, Clone, Debug, serde::Serialize)]
pub struct FasterOnly {
    /// Successes judged (polish, or a known objective key).
    pub judged: u64,
    /// Accepted as strictly sooner than the key's (or polish target's) best.
    pub sooner: u64,
    pub sooner_frames: u64,
    /// Accepted for a new (keys, stars) best only.
    pub new_progress: u64,
    /// Met but not sooner: counted as failures.
    pub rejected: u64,
    /// Of the rejected, still added (new by the hard dedupe).
    pub rejected_added: u64,
    pub by_kind: BTreeMap<&'static str, (u64, u64, u64)>,
}

#[derive(Default, Clone, Debug, serde::Serialize)]
pub struct BeatStats {
    pub spawned: u64,
    pub replaced: u64,
    /// Tighter deadline in the live Beat's bucket: not respawned.
    pub same_bucket: u64,
    /// Successes of Beat attempts.
    pub successes: u64,
}

#[derive(Default)]
pub struct Speed {
    /// Best game frame at which a node met each objective key from each
    /// start context (a Beat's meets count for its objective's key). Shared
    /// with BnB's rollouts as a round-start snapshot.
    pub met_best: Arc<HashMap<(String, Ctx), u32>>,
    /// Beat key -> its objective.
    pub beat_inner: HashMap<String, Arc<dyn Objective>>,
    /// (Objective key, context) -> its live Beat's (deadline, key).
    pub beat_live: HashMap<(String, Ctx), (u32, String)>,
    /// Always on: objective-key best-time improvements (sooner meets of a
    /// key met before), total and by the kind that made them.
    pub met: Imp,
    pub met_by_kind: BTreeMap<&'static str, Imp>,
    /// Always on: (keys, stars) best-time improvements (note_best).
    pub progress: Imp,
    pub progress_by_kind: BTreeMap<&'static str, Imp>,
    pub fo: FasterOnly,
    pub beat: BeatStats,
}

impl Speed {
    pub fn note_met_imp(&mut self, via: &'static str, saved: u64) {
        self.met.add(saved);
        self.met_by_kind.entry(via).or_default().add(saved);
    }

    pub fn note_progress_imp(&mut self, via: &'static str, saved: u64) {
        self.progress.add(saved);
        self.progress_by_kind.entry(via).or_default().add(saved);
    }

    pub fn summary(&self) -> serde_json::Value {
        let s = settings();
        let kinds = |m: &BTreeMap<&'static str, Imp>| serde_json::to_value(m).unwrap_or_default();
        let mut v = serde_json::json!({
            "improvements": self.met.improvements,
            "frames_saved": self.met.frames_saved,
            "by_kind": kinds(&self.met_by_kind),
            "keys_met": self.met_best.len(),
            "progress": {"improvements": self.progress.improvements, "frames_saved": self.progress.frames_saved, "by_kind": kinds(&self.progress_by_kind)},
        });
        if s.faster_only {
            let mut fo = serde_json::to_value(&self.fo).unwrap_or_default();
            fo["by_kind"] = self
                .fo
                .by_kind
                .iter()
                .map(|(k, (a, f, r))| (k.to_string(), serde_json::json!({"sooner": a, "sooner_frames": f, "rejected": r})))
                .collect::<serde_json::Map<_, _>>()
                .into();
            v["faster_only"] = fo;
        }
        if s.beat {
            v["beat"] = serde_json::to_value(&self.beat).unwrap_or_default();
            v["beat"]["live"] = serde_json::json!(self.beat_live.len());
            v["beat"]["opt"] = serde_json::json!(s.opt);
            v["beat"]["w"] = serde_json::json!(s.beat_w);
        }
        if s.bnb {
            v["bnb"] = serde_json::json!({
                "rollouts": BNB_ROLLOUTS.load(Ordering::Relaxed),
                "stops": BNB_STOPS.load(Ordering::Relaxed),
                "frames_saved": BNB_SAVED.load(Ordering::Relaxed),
                "opt": s.opt, "slack": s.slack,
            });
        }
        v
    }
}
