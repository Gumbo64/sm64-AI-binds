//! Runs Objective-based RRT from power-on.
//!
//! cargo run --release -- <rom> [out_dir=runs/obrrt] [seed=1] [minutes=30] [<start.m64>]
//!
//! Boots with the menu-skipping inputs (or plays `<start.m64>`), waits for
//! Mario to stand still, checks the memory layout, then searches. Writes
//! `<out_dir>/`: frontier-*.m64 and first-*.m64 replays (from power-on),
//! nodes.csv, summary.json and events.jsonl.

use std::sync::Arc;

use anyhow::{Result, bail};
use obrrt_core::buttons::{A_BUTTON, START_BUTTON};
use obrrt_core::{GamePad, check_layout, read_state};
use obrrt_objectives::Registry;
use obrrt_search::{Config, Engine};
use sm64_binds::SM64GameGenerator;

fn boot_inputs() -> Vec<GamePad> {
    (0..1680)
        .map(|i| {
            let mut button = 0;
            if (150..160).contains(&i) || (200..300).contains(&i) {
                button = START_BUTTON;
            }
            if i % 2 == 0 {
                button = A_BUTTON;
            }
            GamePad::new(button, 0, 0)
        })
        .collect()
}

/// OBRRT_TAS_BOOT=1: the 2026 120-star TAS's inputs up to its first
/// controllable frame (1646; Mario idle at the same spot as the fixed boot's
/// root, 44 frames earlier). The root is taken there, without the still wait.
const TAS_BOOT_FRAMES: usize = 1646;

fn tas_boot_inputs() -> Result<Vec<GamePad>> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../ref/tas/full-120star-2026-U.m64");
    let bytes = std::fs::read(path)?;
    Ok(bytes[0x400..]
        .chunks_exact(4)
        .take(TAS_BOOT_FRAMES)
        .map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8))
        .collect())
}

fn read_m64(path: &str) -> Result<Vec<GamePad>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8))
        .collect())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        bail!("usage: obrrt <rom> [out_dir=runs/obrrt] [seed=1] [minutes=30] [<start.m64>]");
    }
    let mut cfg = Config::default();
    // Experiment knobs, from the environment.
    if let Ok(v) = std::env::var("OBRRT_GA_SCALE") {
        cfg.ga_scale = v.parse()?;
    }
    // Defaults: steps fitness, 3x generations, new input operators (see
    // the y-* experiments). These switch back to the old GA for comparison.
    match std::env::var("OBRRT_FITNESS").as_deref() {
        Ok("met") => cfg.fitness = obrrt_search::ga::Fitness::Met,
        Ok("steps") => cfg.fitness = obrrt_search::ga::Fitness::Steps,
        _ => {}
    }
    if let Ok(v) = std::env::var("OBRRT_GA_NEW") {
        cfg.ga_new_ops = v == "1";
    }
    // OBRRT_GA_BUDGET=1: per-objective GA budgets (Explore-only attempts
    // get 1x generations and the early stop).
    if std::env::var("OBRRT_GA_BUDGET").is_ok_and(|v| v == "1") {
        cfg.ga_per_objective_budget = true;
    }
    if std::env::var("OBRRT_SETTLE_RANK").is_ok_and(|v| v == "1") {
        obrrt_search::engine::SETTLE_RANK.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if std::env::var("OBRRT_FINGERPRINT").is_ok_and(|v| v == "0") {
        obrrt_search::engine::FINGERPRINT.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    // OBRRT_STARSET_KEY=1: dedupe and b cells by the collected set
    // (St::progress_set), not only the star/key counts.
    if std::env::var("OBRRT_STARSET_KEY").is_ok_and(|v| v == "1") {
        obrrt_search::engine::STARSET.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    // OBRRT_RETIRE_SUBSET=1: retire a state only when a frontier set has
    // everything it collected (OBRRT_RETIRE_SUBSET_CAP sets per count, 6).
    if std::env::var("OBRRT_RETIRE_SUBSET").is_ok_and(|v| v == "1") {
        obrrt_search::tree::RETIRE_SUBSET.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if let Ok(v) = std::env::var("OBRRT_RETIRE_SUBSET_CAP") {
        obrrt_search::tree::RETIRE_SUBSET_CAP.store(v.parse()?, std::sync::atomic::Ordering::Relaxed);
    }
    // OBRRT_KEEP_WIP=1: don't retire a state whose work under way in its
    // level (St::wip: boss damage, red coins, posts) every frontier state
    // there lacks; OBRRT_KEEP_WIP_N per level (64), OBRRT_KEEP_WIP_FRAMES
    // before the frontier's earliest arrival at most (3000).
    if std::env::var("OBRRT_KEEP_WIP").is_ok_and(|v| v == "1") {
        obrrt_search::tree::KEEP_WIP.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if let Ok(v) = std::env::var("OBRRT_KEEP_WIP_N") {
        obrrt_search::tree::KEEP_WIP_N.store(v.parse()?, std::sync::atomic::Ordering::Relaxed);
    }
    if let Ok(v) = std::env::var("OBRRT_KEEP_WIP_FRAMES") {
        obrrt_search::tree::KEEP_WIP_FRAMES.store(v.parse()?, std::sync::atomic::Ordering::Relaxed);
    }
    // OBRRT_FAST_MODAL=0: the old dialog/menu macros (A every 4th frame)
    // instead of the frame-optimal policies (objectives/src/menu.rs).
    if let Ok(v) = std::env::var("OBRRT_FAST_MODAL") {
        obrrt_objectives::set_fast_modal(v != "0");
    }
    // OBRRT_BACKCHAIN=1: re-attempt successes from 2..K generations up
    // their branch, the GA seeded with the recorded inputs (backchain.rs);
    // OBRRT_BACKCHAIN_SHARE (0.1), _KMIN (2), _KMAX (6), _CAP (900 frames),
    // _PER_TARGET (3), _MINW (1).
    if std::env::var("OBRRT_BACKCHAIN").is_ok_and(|v| v == "1") {
        let mut s = obrrt_search::backchain::Settings::default();
        let get = |k: &str| std::env::var(format!("OBRRT_BACKCHAIN_{k}")).ok();
        if let Some(v) = get("SHARE") {
            s.share = v.parse()?;
        }
        if let Some(v) = get("KMIN") {
            s.kmin = v.parse()?;
        }
        if let Some(v) = get("KMAX") {
            s.kmax = v.parse()?;
        }
        if let Some(v) = get("CAP") {
            s.cap = v.parse()?;
        }
        if let Some(v) = get("PER_TARGET") {
            s.per_target = v.parse()?;
        }
        if let Some(v) = get("MINW") {
            s.min_weight = v.parse()?;
        }
        obrrt_search::backchain::configure(Some(s));
    }
    if let Ok(v) = std::env::var("OBRRT_PICK_BY_B") {
        cfg.pick_by_b = v == "1";
    }
    if let Ok(v) = std::env::var("OBRRT_MAX_FRAMES") {
        cfg.max_frames = Some(v.parse()?);
    }
    if let Ok(v) = std::env::var("OBRRT_ROUND") {
        cfg.round_plans = v.parse()?;
    }
    if let Ok(v) = std::env::var("OBRRT_KIND_CAP") {
        cfg.kind_cap = v.parse()?;
    }
    if std::env::var("OBRRT_ZONE_BANDIT").is_ok_and(|v| v == "1") {
        cfg.zone_bandit = true;
    }
    if std::env::var("OBRRT_SOFT_DEDUP").is_ok_and(|v| v == "1") {
        cfg.soft_dedup = true;
    }
    if let Ok(v) = std::env::var("OBRRT_START_WEAR") {
        cfg.start_wear = v == "1";
    }
    if let Ok(v) = std::env::var("OBRRT_DECAY_ROUTE") {
        cfg.decay_route = v == "1";
    }
    if let Ok(v) = std::env::var("OBRRT_ENTRY_BANDIT") {
        cfg.entry_bandit = v == "1";
    }
    if let Ok(v) = std::env::var("OBRRT_KIND_BANDIT") {
        cfg.kind_bandit = v == "1";
    }
    if let Ok(v) = std::env::var("OBRRT_KIND_GAMMA") {
        cfg.kind_gamma = v.parse()?;
    }
    if let Ok(v) = std::env::var("OBRRT_TUNE_FRAMES") {
        cfg.tune_frames = v.parse()?;
    }
    if let Ok(v) = std::env::var("OBRRT_MEM_MB") {
        cfg.mem_budget = v.parse::<usize>()? * 1_000_000;
    }
    if let Ok(v) = std::env::var("OBRRT_ROUTE") {
        cfg.route_share = v.parse()?;
    }
    if let Ok(v) = std::env::var("OBRRT_VALUE_ZONES") {
        cfg.value_zones = v == "1";
    }
    if std::env::var("OBRRT_STAR").is_ok_and(|v| v == "1") {
        cfg.rrt_star = true;
    }
    // OBRRT_ARRIVAL=1: rank starts (and judge improvement) by estimated
    // arrival, for distance objectives; pace (u/frame, default 20), temperature (s), band (h units).
    if std::env::var("OBRRT_ARRIVAL").is_ok_and(|v| v == "1") {
        cfg.arrival = true;
    }
    if let Ok(v) = std::env::var("OBRRT_ARRIVAL_PACE") {
        cfg.arrival_pace = v.parse()?;
    }
    if let Ok(v) = std::env::var("OBRRT_ARRIVAL_T") {
        cfg.arrival_t = v.parse()?;
    }
    if let Ok(v) = std::env::var("OBRRT_ARRIVAL_BAND") {
        cfg.arrival_band = Some(v.parse()?);
    }
    // OBRRT_ARRIVAL_DECAY=either (default) | arrival: what counts as an
    // improvement under arrival ranking; OBRRT_ARRIVAL_MIX: share of start
    // picks ranked by arrival.
    match std::env::var("OBRRT_ARRIVAL_DECAY").as_deref() {
        Ok("arrival") => cfg.arrival_decay_either = false,
        Ok("either") | Err(_) => cfg.arrival_decay_either = true,
        Ok(v) => bail!("OBRRT_ARRIVAL_DECAY: either or arrival, not {v}"),
    }
    // OBRRT_EARLIEST=1: queue each new earliest arrival in its zone tier
    // (or b cell, by 30 frames) for up to OBRRT_EARLIEST_CAP (12) attempts
    // from itself, at most OBRRT_EARLIEST_SHARE (0.15) of a round's plans;
    // OBRRT_EARLIEST_CELL=0 queues zone-tier records only.
    if std::env::var("OBRRT_EARLIEST").is_ok_and(|v| v == "1") {
        cfg.earliest = true;
    }
    if let Ok(v) = std::env::var("OBRRT_EARLIEST_CAP") {
        cfg.earliest_cap = v.parse()?;
    }
    if let Ok(v) = std::env::var("OBRRT_EARLIEST_SHARE") {
        cfg.earliest_share = v.parse()?;
    }
    if let Ok(v) = std::env::var("OBRRT_EARLIEST_CELL") {
        cfg.earliest_cell = v != "0";
    }
    if let Ok(v) = std::env::var("OBRRT_ARRIVAL_MIX") {
        cfg.arrival_mix = v.parse()?;
    }
    if let Ok(v) = std::env::var("OBRRT_PINK") {
        cfg.ga_pink = v.parse()?;
    }
    if let Ok(v) = std::env::var("OBRRT_CHAIN_PROB") {
        cfg.chain_fixed = Some(v.parse()?);
    }
    // OBRRT_TECH: which speedrun-technique objectives are on ("all" by
    // default; see objectives/src/extra/tech/mod.rs for names and tags).
    let tech = std::env::var("OBRRT_TECH").unwrap_or_else(|_| "all".into());
    obrrt_objectives::extra::tech::configure(&tech).map_err(anyhow::Error::msg)?;
    println!("techniques ({tech}): {:?}", obrrt_objectives::extra::tech::enabled_names());
    if let Some(d) = args.get(2) {
        cfg.out_dir = d.clone();
    }
    if let Some(s) = args.get(3) {
        cfg.seed = s.parse()?;
    }
    if let Some(m) = args.get(4) {
        cfg.minutes = m.parse()?;
    }
    let game_gen = SM64GameGenerator::from_file(&args[1])?;
    let mut game = game_gen.create_game()?;
    let tas_boot = std::env::var("OBRRT_TAS_BOOT").is_ok_and(|v| v == "1") && args.len() <= 5;
    let mut boot = match args.get(5) {
        Some(p) => read_m64(p)?,
        None if tas_boot => tas_boot_inputs()?,
        None => boot_inputs(),
    };
    for p in &boot {
        game.step_game(*p)?;
    }
    let mut st = read_state(&game, boot.len() as u32, false);
    // OBRRT_BOOT_NOWAIT=1 (tests from a TAS cut): start where `<start.m64>`
    // ends if Mario is in control there, mid-air or moving, instead of
    // idling until he stands still.
    let nowait = std::env::var("OBRRT_BOOT_NOWAIT").is_ok_and(|v| v == "1") && args.len() > 5;
    let mut still = if (nowait || tas_boot) && st.controllable() { 10 } else { 0 };
    // The wait limit counts frames after `<start.m64>`, so long TAS cuts boot.
    let prefix_len = boot.len();
    while still < 10 || !st.controllable() {
        let before = st.pos;
        let idle = GamePad::new(0, 0, 0);
        game.step_game(idle)?;
        boot.push(idle);
        st = read_state(&game, boot.len() as u32, false);
        still = if st.pos == before { still + 1 } else { 0 };
        if boot.len() - prefix_len > 30_000 {
            bail!("Mario didn't stand still after booting");
        }
    }
    check_layout(&game)?;
    let root = read_state(&game, boot.len() as u32, true);
    println!(
        "root: frame {} level {} area {} pos {:?} stars {} keys {} objects {}; {} workers",
        root.frame,
        root.level,
        root.area,
        root.pos,
        root.stars,
        root.keys,
        root.objects().len(),
        rayon_threads()
    );
    let save = game.save_state();
    let mut engine = Engine::new(cfg, &game_gen, boot, save, root, Arc::new(Registry))?;
    // OBRRT_RESUME=<run dir>: continue that run's tree.
    if let Ok(dir) = std::env::var("OBRRT_RESUME") {
        engine.resume(&dir)?;
    }
    engine.run()
}

fn rayon_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}
