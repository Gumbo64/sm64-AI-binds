//! Route optimisation bench (see obrrt_search::route):
//!
//!     obrrt-route <rom> <run dir> [method=full] [minutes=5] [seed=1] [node=earliest star] [out dir]
//!
//! Loads the run's route to `node`, verifies it replays, then runs `method`
//! on all threads for `minutes`. Writes <out dir> (default
//! runs/route-<run>-<method>-s<seed>): best.m64 (from power-on),
//! curve.csv, trials.jsonl, summary.json.

use anyhow::{Result, bail};
use obrrt_search::route::{self, Bench, methods};
use sm64_binds::SM64GameGenerator;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: obrrt-route <rom> <run dir> [method=full] [minutes=5] [seed=1] [node] [out dir]\nmethods: {}", methods::NAMES.join(", "));
    }
    let run_dir = args[2].trim_end_matches('/');
    let method_name = args.get(3).map_or("full", String::as_str);
    let minutes: f64 = args.get(4).map_or(Ok(5.0), |m| m.parse())?;
    let seed: u64 = args.get(5).map_or(Ok(1), |s| s.parse())?;
    let node: Option<u32> = args.get(6).filter(|s| *s != "star").map(|s| s.parse()).transpose()?;
    let run_name = run_dir.rsplit('/').next().unwrap();
    let out = args.get(7).cloned().unwrap_or(format!("runs/route-{run_name}-{method_name}-s{seed}"));
    let Some(mut method) = methods::make(method_name) else {
        bail!("unknown method {method_name}; methods: {}", methods::NAMES.join(", "));
    };
    let game_gen = SM64GameGenerator::from_file(&args[1])?;
    let loaded = route::load(&game_gen, run_dir, node)?;
    println!(
        "route: {} states ({} essential), original goal frame {}, stream {} frames, {} dialogs/menus",
        loaded.waypoints.len(),
        loaded.waypoints.iter().filter(|w| w.essential).count(),
        loaded.original_frame,
        loaded.stream.len(),
        loaded.policy.dialogs_and_menus()
    );
    let mut bench = Bench::new(&game_gen, loaded, &out)?;
    println!("baseline (bench replay): goal frame {}", bench.best.goal_frame);
    bench.run(method.as_mut(), minutes, seed)
}
