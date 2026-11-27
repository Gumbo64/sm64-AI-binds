//! Benchmarks: the per-start GA baselines (pink-noise and overlay GA, one
//! search per start) and shared table printing. Policy evaluation lives in
//! `policy.rs`; the commands in `bin/policy.rs`.

use anyhow::Result;
use obrrt_gp::problems::Problem;

pub fn split(c: &obrrt_gp::rollout::Case) -> &'static str {
    if c.train {
        "tr"
    } else if c.val {
        "VA"
    } else {
        "TE"
    }
}

/// One start's result: frames to Done (None = failed) and emulator frames used.
#[derive(Clone, Copy, Debug)]
pub struct Res {
    pub done: Option<usize>,
    pub emu: u64,
}

/// Print the train / val / test summary of per-case results, return it as JSON.
pub fn summarize(pb: &Problem, label: &str, rs: &[Res]) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for (sp, tag) in [("tr", "train"), ("VA", "val"), ("TE", "test")] {
        let rows: Vec<&Res> = pb.cases.iter().zip(rs).filter(|(c, _)| split(c) == sp).map(|(_, r)| r).collect();
        if rows.is_empty() {
            continue;
        }
        let succ = rows.iter().filter(|r| r.done.is_some()).count();
        let fr = rows.iter().filter_map(|r| r.done).sum::<usize>() as f64 / succ.max(1) as f64;
        let emu = rows.iter().map(|r| r.emu).sum::<u64>() as f64 / rows.len() as f64;
        println!("{label}: {tag:<5} success {succ}/{} mean frames-to-Done {fr:.1} mean emu/start {:.2} M", rows.len(), emu / 1e6);
        out.insert(tag.to_string(), serde_json::json!({"n": rows.len(), "success": succ, "mean_frames": fr, "emu_per_start": emu}));
    }
    serde_json::Value::Object(out)
}

#[cfg(feature = "ga")]
pub mod ga {
    use super::*;
    use obrrt_search::ga::{GaParams, Worker};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use sm64_binds::SM64GameGenerator;

    /// One GA search per (case, seed): `scale` multiplies the generations,
    /// `pink` = 1 is the pure pink-noise GA, 0 the overlay GA (gabase).
    pub fn baseline(gg: &SM64GameGenerator, pb: &Problem, threads: usize, seeds: u64, scale: f64, pink: f64) -> Result<Vec<Vec<Res>>> {
        let params = GaParams { scale, pink };
        let n = pb.cases.len() * seeds as usize;
        let next = std::sync::atomic::AtomicUsize::new(0);
        let out: Vec<std::sync::Mutex<Option<Res>>> = (0..n).map(|_| std::sync::Mutex::new(None)).collect();
        std::thread::scope(|sc| {
            for _ in 0..threads {
                sc.spawn(|| {
                    let mut w = Worker::new(gg).expect("worker");
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        let (ci, seed) = (i % pb.cases.len(), (i / pb.cases.len()) as u64 + 1);
                        let c = &pb.cases[ci];
                        let obj = pb.tasks[c.task].objective().expect("objective");
                        w.base.load_state(&c.save).expect("load");
                        let st = obrrt_core::read_state(&w.base, c.frame as u32, true);
                        let mut rng = SmallRng::seed_from_u64(seed * 1000 + ci as u64);
                        let att = w.attempt(&[obj.as_ref()], &st, &c.save, &mut rng, params).expect("attempt");
                        let done = att.segments.first().filter(|s| s.success).map(|s| s.played);
                        *out[i].lock().unwrap() = Some(Res { done, emu: att.frames });
                    }
                });
            }
        });
        let flat: Vec<Res> = out.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect();
        Ok(flat.chunks(pb.cases.len()).map(|c| c.to_vec()).collect())
    }
}
