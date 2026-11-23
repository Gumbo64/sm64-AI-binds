//! OBRRT_ATTEMPT_MODEL=<model.json>: the attempt-value model
//! (backbone/attempts/train.py), a small network trained on every attempt
//! logged by past runs. From what the search knows when it plans an attempt
//! (the objective's kind and start ETA, the start state, counts of the run
//! so far) it predicts: the chance the objective is met; the chance the
//! attempt pays (the subtree under the state it adds makes progress: a star,
//! a key, a red coin, an exit); the chance it leads to a star or key; and
//! quantiles of the emulator frames the GA will use and of the segment's
//! length. Held-out runs (2026-11-09, 950 k attempts): success AUC 0.96,
//! pays AP 0.50 (objective kind alone 0.25), cost within 0.12 dex.
//!
//! The engine uses it in `plan` (OBRRT_ATTEMPT_PROPOSE=K, default 4): the
//! usual sampler proposes K (objective, start) pairs for the chosen b, and
//! one is drawn by (chance it pays / predicted frames) ^ OBRRT_ATTEMPT_BETA
//! (default 1). The sampler stays the proposal, so nothing is ruled out.
//! The forward pass is done here, on the CPU: tens of microseconds each.

use std::collections::HashMap;
use std::sync::OnceLock;

pub struct Model {
    cats: Vec<String>,
    vocab: Vec<HashMap<String, usize>>,
    emb: Vec<Vec<Vec<f32>>>,
    num: Vec<(String, bool)>,
    mu: Vec<f32>,
    sd: Vec<f32>,
    lin: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
    head: (Vec<Vec<f32>>, Vec<f32>),
}

/// What the model sees of one planned attempt (the columns of build.py).
pub struct Input<'a> {
    pub kind: &'a str,
    pub level: u16,
    pub action: u32,
    pub start_eta: Option<f64>,
    pub stage: f64,
    pub plan_len: f64,
    /// Attempts and successes of this kind so far in the run.
    pub kind_n: f64,
    pub kind_s: f64,
    /// Attempts of this kind, and of any kind, from this state so far.
    pub kn_n: f64,
    pub node_n: f64,
    pub frames_before: f64,
    pub y: f64,
    pub hspeed: f64,
    pub fwd_vel: f64,
    pub stars: f64,
    pub coins: f64,
    pub red: f64,
    pub route_frame: f64,
    pub depth: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Out {
    pub success: f64,
    pub pays: f64,
    pub star: f64,
    /// log10 emulator frames, quantiles 0.1, 0.5, 0.9.
    pub cost: [f64; 3],
    /// log10 segment frames when it succeeds, same quantiles.
    pub seg: [f64; 3],
    /// How soon a star or key follows: 0.999 ^ route frames from the start
    /// state to the first one under the added state, 0 for none (the pays
    /// chance when the model has no such head).
    pub value: f64,
}

impl Out {
    /// Chance it pays per predicted emulator frame.
    pub fn pays_per_frame(&self) -> f64 {
        self.pays.max(1e-4) / 10f64.powf(self.cost[1].clamp(2.0, 7.0))
    }
}

fn matrix(v: &serde_json::Value) -> Vec<Vec<f32>> {
    v.as_array().expect("matrix").iter().map(vector).collect()
}

fn vector(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().expect("vector").iter().map(|x| x.as_f64().expect("number") as f32).collect()
}

impl Model {
    pub fn load(path: &str) -> Model {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("OBRRT_ATTEMPT_MODEL {path}: {e}"));
        let j: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("OBRRT_ATTEMPT_MODEL {path}: {e}"));
        let cats: Vec<String> = j["cats"].as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect();
        let vocab = cats.iter().map(|c| j["vocab"][c].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap() as usize)).collect()).collect();
        let w = &j["weights"];
        let emb = (0..cats.len()).map(|i| matrix(&w[format!("emb.{i}.weight")])).collect();
        let num = j["num"].as_array().unwrap().iter().map(|n| (n[0].as_str().unwrap().to_string(), n[1].as_str().unwrap() == "log1p")).collect();
        let layers = j["layers"].as_u64().unwrap() as usize;
        let lin = (0..layers).map(|i| (matrix(&w[format!("trunk.{}.weight", 2 * i)]), vector(&w[format!("trunk.{}.bias", 2 * i)]))).collect();
        Model { cats, vocab, emb, num, mu: vector(&j["mu"]), sd: vector(&j["sd"]), lin, head: (matrix(&w["head.weight"]), vector(&w["head.bias"])) }
    }

    pub fn eval(&self, x: &Input) -> Out {
        let mut h: Vec<f32> = Vec::with_capacity(self.num.len() + 16 * self.cats.len());
        for (i, (name, log)) in self.num.iter().enumerate() {
            let v = match name.as_str() {
                "start_eta" => x.start_eta.unwrap_or(0.0).clamp(0.0, 5000.0),
                "eta_missing" => x.start_eta.is_none() as u8 as f64,
                "stage" => x.stage,
                "plan_len" => x.plan_len,
                "kind_n" => x.kind_n,
                "kind_rate" => (x.kind_s + 1.0) / (x.kind_n + 2.0),
                "kn_n" => x.kn_n,
                "node_n" => x.node_n,
                "frames_before" => x.frames_before,
                "y" => x.y,
                "hspeed" => x.hspeed,
                "fwd_vel" => x.fwd_vel,
                "stars" => x.stars,
                "coins" => x.coins,
                "red" => x.red,
                "route_frame" => x.route_frame.max(0.0),
                "depth" => x.depth,
                other => panic!("attempt model: unknown input {other}"),
            };
            let v = if *log { v.ln_1p() } else { v };
            h.push((v as f32 - self.mu[i]) / self.sd[i]);
        }
        for (i, c) in self.cats.iter().enumerate() {
            let key = match c.as_str() {
                "kind" => x.kind.to_string(),
                "level" => x.level.to_string(),
                "action" => format!("{:#x}", x.action),
                other => panic!("attempt model: unknown category {other}"),
            };
            h.extend_from_slice(&self.emb[i][self.vocab[i].get(&key).copied().unwrap_or(0)]);
        }
        let apply = |w: &Vec<Vec<f32>>, b: &Vec<f32>, h: &[f32]| -> Vec<f32> { w.iter().zip(b).map(|(row, b)| row.iter().zip(h).map(|(a, x)| a * x).sum::<f32>() + b).collect() };
        for (w, b) in &self.lin {
            h = apply(w, b, &h);
            for v in &mut h {
                *v = v.max(0.0);
            }
        }
        let o = apply(&self.head.0, &self.head.1, &h);
        let sg = |v: f32| 1.0 / (1.0 + (-v as f64).exp());
        Out { success: sg(o[0]), pays: sg(o[1]), star: sg(o[2]), cost: [o[3] as f64, o[4] as f64, o[5] as f64], seg: [o[6] as f64, o[7] as f64, o[8] as f64], value: o.get(9).map_or(sg(o[1]), |&v| sg(v)) }
    }
}

/// The model of OBRRT_ATTEMPT_MODEL, if set.
pub fn model() -> Option<&'static Model> {
    static M: OnceLock<Option<Model>> = OnceLock::new();
    M.get_or_init(|| std::env::var("OBRRT_ATTEMPT_MODEL").ok().map(|p| Model::load(&p))).as_ref()
}

/// OBRRT_ATTEMPT_SCORE: what a proposal is drawn by, per frame of the run's
/// mean cost of its kind: `pays` (default), `star`, or `value`.
pub fn score(o: &Out) -> f64 {
    static S: OnceLock<u8> = OnceLock::new();
    match *S.get_or_init(|| match std::env::var("OBRRT_ATTEMPT_SCORE").as_deref() {
        Ok("star") => 1,
        Ok("value") => 2,
        _ => 0,
    }) {
        1 => o.star,
        2 => o.value,
        _ => o.pays,
    }
}

/// (OBRRT_ATTEMPT_PROPOSE, OBRRT_ATTEMPT_BETA).
pub fn settings() -> (usize, f64) {
    static S: OnceLock<(usize, f64)> = OnceLock::new();
    *S.get_or_init(|| {
        let f = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        ((f("OBRRT_ATTEMPT_PROPOSE", 4.0) as usize).max(1), f("OBRRT_ATTEMPT_BETA", 1.0))
    })
}
