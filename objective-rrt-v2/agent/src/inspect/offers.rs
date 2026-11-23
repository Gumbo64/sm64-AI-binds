//! `objectives`: what the registry offers at a state.

use obrrt_core::{Generator, Objective, Precision, start_eta};
use obrrt_objectives::Registry;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use serde_json::{Value, json};

use super::{r2, v3};
use obrrt_core::St;

/// Call `obrrt_objectives::setup::from_env()` once before this.
pub fn objectives(st: &St, kind: Option<&str>) -> Value {
    let mut rng = SmallRng::seed_from_u64(0x5eed);
    let all = Registry.generate(st, &mut rng);
    let total = all.len();
    let pat = kind.map(str::to_lowercase);
    let mut rows: Vec<(f64, Value)> = all
        .iter()
        .filter(|o| pat.as_ref().is_none_or(|p| o.kind().to_lowercase().contains(p.as_str())))
        .map(|o| {
            let eta = start_eta(o.as_ref() as &dyn Objective, st);
            let b = o.ga_budget();
            (eta, json!({
                "kind": o.kind(), "key": o.key(),
                "base_criteria": o.base_criteria(st), "exists_in": o.exists_in(st),
                "eta": eta.is_finite().then(|| r2(eta as f32)), "lost": !eta.is_finite(),
                "steer_target": o.steer_target(st).map(v3),
                "ga_budget": b.map(|b| json!({"population": b.population, "generations": b.generations, "horizon": b.horizon})),
                "precision": if o.precision() == Precision::Precise { "precise" } else { "explore" },
                "modal": o.modal(), "weight": r2(o.obj_weight(st) as f32),
            }))
        })
        .collect();
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    json!({"frame": st.frame, "total_offered": total, "shown": rows.len(), "kind_filter": kind, "objectives": rows.into_iter().map(|r| r.1).collect::<Vec<_>>()})
}
