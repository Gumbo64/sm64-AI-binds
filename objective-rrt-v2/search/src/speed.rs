//! Best meets and best-time improvements (output only): `best_improved`
//! events in events.jsonl and the `speed` block of summary.json
//! (per-objective-key and per-(keys, stars) improvements, frames saved, by
//! kind).

use std::collections::{BTreeMap, HashMap};

use obrrt_core::St;

/// A meet's progress context: the attempt start's keys, stars and a hash of
/// its collected set (`tree::set_key`). Bests are per (objective key,
/// context): meeting an objective later in the route, with more progress, is
/// not bounded by its meets with less.
pub type Ctx = (u8, i16, u64);

pub fn ctx(st: &St) -> Ctx {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    crate::tree::set_key(st).hash(&mut h);
    (st.keys, st.stars, h.finish())
}

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

#[derive(Default)]
pub struct Speed {
    /// Best game frame at which a node met each objective key from each
    /// start context.
    pub met_best: HashMap<(String, Ctx), u32>,
    /// Objective-key best-time improvements (sooner meets of a key met
    /// before), total and by the kind that made them.
    pub met: Imp,
    pub met_by_kind: BTreeMap<&'static str, Imp>,
    /// (keys, stars) best-time improvements (note_best).
    pub progress: Imp,
    pub progress_by_kind: BTreeMap<&'static str, Imp>,
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
        let kinds = |m: &BTreeMap<&'static str, Imp>| serde_json::to_value(m).unwrap_or_default();
        serde_json::json!({
            "improvements": self.met.improvements,
            "frames_saved": self.met.frames_saved,
            "by_kind": kinds(&self.met_by_kind),
            "keys_met": self.met_best.len(),
            "progress": {"improvements": self.progress.improvements, "frames_saved": self.progress.frames_saved, "by_kind": kinds(&self.progress_by_kind)},
        })
    }
}
