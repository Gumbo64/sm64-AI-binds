//! `OBRRT_GAMMA_SET=0.98,0.99,0.995` (obrrt2, off by default): each attempt
//! draws its GA's γ from the set, by a per-kind bandit kept like the kind
//! rates: each (kind, γ) has (paid, tried) counts, a γ is drawn with
//! probability ∝ (paid + 1) / (tried + 2), and "paid" is the kind bandit's
//! signal (a success that added a state, or an improvement). The draw uses
//! the attempt's own seed and the counts as of the round's start (merges
//! come after the round), so runs stay deterministic. Off: nothing here is
//! read and no draws are made. summary.json `gamma_set` reports the shares.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub fn set() -> Option<&'static [f64]> {
    static S: OnceLock<Option<Vec<f64>>> = OnceLock::new();
    S.get_or_init(|| {
        let v: Vec<f64> = obrrt_core::sys::env::var("OBRRT_GAMMA_SET").ok()?.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        (!v.is_empty()).then_some(v)
    })
    .as_deref()
}

#[derive(Default)]
struct State {
    /// kind -> per γ (paid, tried).
    kinds: BTreeMap<String, Vec<(u64, u64)>>,
}

fn state() -> &'static Mutex<State> {
    static S: OnceLock<Mutex<State>> = OnceLock::new();
    S.get_or_init(Default::default)
}

/// The γ index for an attempt of `kind` with plan seed `seed`.
pub fn choose(kind: &str, seed: u64) -> Option<u8> {
    let g = set()?;
    let st = state().lock().unwrap();
    let w: Vec<f64> = match st.kinds.get(kind) {
        Some(c) => c.iter().map(|&(p, t)| (p as f64 + 1.0) / (t as f64 + 2.0)).collect(),
        None => vec![1.0; g.len()],
    };
    let mut rng = SmallRng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15);
    let mut x = rng.random::<f64>() * w.iter().sum::<f64>();
    for (i, wi) in w.iter().enumerate() {
        if x < *wi {
            return Some(i as u8);
        }
        x -= wi;
    }
    Some((g.len() - 1) as u8)
}

pub fn gamma(i: u8) -> f64 {
    set().map_or(0.99, |g| g[i as usize])
}

thread_local! {
    static CUR: std::cell::Cell<Option<u8>> = const { std::cell::Cell::new(None) };
}

/// The γ index of the attempt being run or merged on this thread.
pub fn set_current(i: Option<u8>) {
    CUR.with(|c| c.set(i));
}

pub fn current() -> Option<u8> {
    CUR.with(|c| c.get())
}

/// Counts the current attempt's outcome for `kind`.
pub fn observe(kind: &str, paid: bool) {
    let (Some(g), Some(i)) = (set(), current()) else { return };
    let mut st = state().lock().unwrap();
    let c = st.kinds.entry(kind.to_string()).or_insert_with(|| vec![(0, 0); g.len()]);
    c[i as usize].1 += 1;
    c[i as usize].0 += paid as u64;
}

/// summary.json's `gamma_set` (None when off): per kind and γ the counts,
/// rate and the share it would be drawn with now.
pub fn summary() -> Option<serde_json::Value> {
    let g = set()?;
    let st = state().lock().unwrap();
    let mut kinds = serde_json::Map::new();
    let mut tot = vec![(0u64, 0u64); g.len()];
    for (k, c) in &st.kinds {
        let w: Vec<f64> = c.iter().map(|&(p, t)| (p as f64 + 1.0) / (t as f64 + 2.0)).collect();
        let sum: f64 = w.iter().sum();
        kinds.insert(
            k.clone(),
            serde_json::json!(c.iter().zip(&w).enumerate().map(|(i, (&(p, t), wi))| {
                tot[i].0 += p;
                tot[i].1 += t;
                serde_json::json!({"gamma": g[i], "paid": p, "tried": t, "share": wi / sum})
            }).collect::<Vec<_>>()),
        );
    }
    Some(serde_json::json!({
        "gammas": g,
        "total": tot.iter().enumerate().map(|(i, &(p, t))| serde_json::json!({"gamma": g[i], "paid": p, "tried": t})).collect::<Vec<_>>(),
        "kinds": kinds,
    }))
}
