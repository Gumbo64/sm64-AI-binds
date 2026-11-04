//! Benchmarks the rolling-horizon controller on objectives from a run's
//! states (a v1 or v2 run: only its tree.bin is read).
//!
//! cargo run --release --bin goalbench -- <rom> <run dir> <kinds|all> [n=20] [seed=1] [candidates=200]
//!
//! Replays `candidates` nodes of the tree (drawn with the seed) from
//! power-on, generates v2 objectives at each with the `Registry`, keeps the
//! (node, objective) pairs whose base criteria hold there, draws `n` per kind
//! and attempts each once with the current settings (the open GA, or with
//! `OBRRT_RH=1` the rolling-horizon controller). Prints per
//! kind: attempts, success rate, mean committed frames to Done, simulated
//! frames per success, and the mean start ETA. Deterministic for a seed
//! (whatever the thread count, `RAYON_NUM_THREADS`).

use std::collections::BTreeMap;

use anyhow::{Result, ensure};
use obrrt_core::{GamePad, Generator, Objective, St, read_state, start_eta};
use obrrt_objectives::Registry;
use obrrt_search::ga::{Fitness, GaParams, Worker};
use obrrt_search::route::read_tree_bin;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use sm64_binds::{SM64Game, SM64GameGenerator, SaveState};

/// `f` over `items` on `RAYON_NUM_THREADS` threads (default: all), each
/// with its own `init()` state; results in order (so thread count doesn't
/// change them).
fn par_map<T: Sync, S, R: Send>(items: &[T], init: impl Fn() -> S + Sync, f: impl Fn(&mut S, &T) -> Result<R> + Sync) -> Result<Vec<R>> {
    let threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        .clamp(1, items.len().max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Option<Result<R>>>> = items.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let mut st = init();
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= items.len() {
                        break;
                    }
                    *out[i].lock().unwrap() = Some(f(&mut st, &items[i]));
                }
            });
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
}

/// Node `id`'s inputs from power-on (record 0 the boot, node i record i+1).
fn path(recs: &[(i32, Vec<GamePad>)], id: usize) -> Vec<GamePad> {
    let mut chain = vec![id + 1];
    let mut node = id as i32;
    while node >= 0 {
        let parent = recs[node as usize + 1].0;
        chain.push(if parent >= 0 { parent as usize + 1 } else { 0 });
        node = parent;
    }
    chain.iter().rev().flat_map(|&r| recs[r].1.iter().copied()).collect()
}

/// Node `id`'s state (with objects) and save, played from power-on.
fn replay(game: &mut SM64Game, power_on: &SaveState, recs: &[(i32, Vec<GamePad>)], id: usize) -> Result<(St, SaveState)> {
    game.load_state(power_on)?;
    let pads = path(recs, id);
    for p in &pads {
        game.step_game(*p)?;
    }
    Ok((read_state(game, pads.len() as u32, true), game.save_state()))
}

/// The objectives generated at `st` (rng seeded by the node) of the wanted
/// kinds whose base criteria hold there.
fn eligible(st: &St, id: usize, seed: u64, kinds: &Option<Vec<String>>) -> Vec<std::sync::Arc<dyn Objective>> {
    let mut rng = SmallRng::seed_from_u64(seed ^ (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    Registry
        .generate(st, &mut rng)
        .into_iter()
        .filter(|o| kinds.as_ref().is_none_or(|k| k.iter().any(|k| k == o.kind())))
        .filter(|o| o.exists_in(st) && o.base_criteria(st) && o.modal() == st.modal())
        .collect()
}

#[derive(Default)]
struct Row {
    attempts: u64,
    successes: u64,
    done_frames: u64,
    frames: u64,
    start_eta: f64,
}

fn main() -> Result<()> {
    obrrt_objectives::moves::install_controller();
    let tech = std::env::var("OBRRT_TECH").unwrap_or_else(|_| "all".into());
    obrrt_objectives::extra::tech::configure(&tech).map_err(anyhow::Error::msg)?;
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() >= 4, "usage: goalbench <rom> <run dir> <kinds|all> [n=20] [seed=1] [candidates=200]");
    let dir = args[2].trim_end_matches('/');
    let kinds: Option<Vec<String>> = (args[3] != "all").then(|| args[3].split(',').map(String::from).collect());
    let n: usize = args.get(4).map_or(Ok(20), |s| s.parse())?;
    let seed: u64 = args.get(5).map_or(Ok(1), |s| s.parse())?;
    let n_cands: usize = args.get(6).map_or(Ok(200), |s| s.parse())?;
    let recs = read_tree_bin(&format!("{dir}/tree.bin"))?;
    ensure!(recs.len() >= 2, "{dir}/tree.bin has no nodes");
    let game_gen = SM64GameGenerator::from_file(&args[1])?;
    let power_on = game_gen.create_game()?.save_state();
    let opt = if obrrt_search::ga::rh_on() { format!("rolling horizon {:?}", obrrt_search::ga::rh()) } else { "open-loop GA".into() };
    println!("{dir}: {} nodes; {opt}", recs.len() - 1);

    // Candidate nodes, then their eligible objectives (keys only: the states
    // are replayed again for the attempts, saves aren't kept).
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut ids: Vec<usize> = (0..recs.len() - 1).collect();
    ids.shuffle(&mut rng);
    ids.truncate(n_cands);
    ids.sort_unstable();
    let found: Vec<Vec<(String, String)>> = par_map(
        &ids,
        || game_gen.create_game().expect("game"),
        |game, &id| {
            let (st, _) = replay(game, &power_on, &recs, id)?;
            Ok(eligible(&st, id, seed, &kinds).iter().map(|o| (o.kind().to_string(), o.key())).collect())
        },
    )?;
    let mut by_kind: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();
    for (&id, objs) in ids.iter().zip(&found) {
        for (kind, key) in objs {
            by_kind.entry(kind.clone()).or_default().push((id, key.clone()));
        }
    }
    let mut picks: Vec<(String, usize, String, u64)> = Vec::new();
    for (kind, mut pairs) in by_kind {
        pairs.shuffle(&mut rng);
        pairs.truncate(n);
        for (id, key) in pairs {
            picks.push((kind.clone(), id, key, rng.random()));
        }
    }
    // The run's defaults (Config::default, run/src/main.rs's switches).
    let env = |k: &str| std::env::var(k).ok();
    let params = GaParams {
        scale: env("OBRRT_GA_SCALE").and_then(|v| v.parse().ok()).unwrap_or(3.0),
        fitness: Fitness::Steps,
        new_ops: env("OBRRT_GA_NEW").is_none_or(|v| v == "1"),
        pink: env("OBRRT_PINK").and_then(|v| v.parse().ok()).unwrap_or(0.0),
        per_objective_budget: env("OBRRT_GA_BUDGET").is_some_and(|v| v == "1"),
    };
    // Each attempt: (kind, success, committed frames to Done, frames, start ETA).
    let results: Vec<(String, bool, usize, u64, f64)> = par_map(
        &picks,
        || Worker::new(&game_gen).expect("worker"),
        |w, (kind, id, key, s)| {
            let (st, save) = replay(&mut w.base, &power_on, &recs, *id)?;
            let obj = eligible(&st, *id, seed, &kinds).into_iter().find(|o| &o.key() == key).expect("objective regenerates");
            let mut rng = SmallRng::seed_from_u64(*s);
            let att = w.attempt(&[obj.as_ref()], &st, &save, &mut rng, params)?;
            let seg = att.segments.first();
            let ok = seg.is_some_and(|s| s.success);
            Ok((kind.clone(), ok, seg.map_or(0, |s| s.played), att.frames, start_eta(obj.as_ref(), &st)))
        },
    )?;
    let mut rows: BTreeMap<String, Row> = BTreeMap::new();
    for (kind, ok, played, frames, eta) in results {
        let r = rows.entry(kind).or_default();
        r.attempts += 1;
        r.frames += frames;
        if eta.is_finite() {
            r.start_eta += eta;
        }
        if ok {
            r.successes += 1;
            r.done_frames += played as u64;
        }
    }
    println!("{:<32} {:>5} {:>6} {:>10} {:>12} {:>9}", "kind", "n", "succ%", "frames2done", "sim/success", "start_eta");
    let mut total = Row::default();
    for (kind, r) in &rows {
        let per = |a: u64, b: u64| if b > 0 { format!("{:.0}", a as f64 / b as f64) } else { "-".into() };
        println!(
            "{kind:<32} {:>5} {:>6.1} {:>10} {:>12} {:>9.0}",
            r.attempts,
            100.0 * r.successes as f64 / r.attempts.max(1) as f64,
            per(r.done_frames, r.successes),
            per(r.frames, r.successes),
            r.start_eta / r.attempts.max(1) as f64
        );
        total.attempts += r.attempts;
        total.successes += r.successes;
        total.frames += r.frames;
    }
    println!(
        "total: {} attempts, {} successes ({:.1}%), {} frames simulated",
        total.attempts,
        total.successes,
        100.0 * total.successes as f64 / total.attempts.max(1) as f64,
        total.frames
    );
    println!("population makeup: {}", obrrt_search::ga::makeup_line());
    Ok(())
}
