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
    // The steering controller, OBRRT_MOVES / OBRRT_MOVE_GA's movement
    // library, OBRRT_FAST_MODAL and OBRRT_TECH (objectives/src/setup.rs,
    // shared with spec's legs).
    let (tech, tech_names) = obrrt_objectives::setup::from_env().map_err(anyhow::Error::msg)?;
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        bail!("usage: obrrt2 <rom> [out_dir=runs/obrrt] [seed=1] [minutes=30] [<start.m64>]");
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
    // OBRRT_STAR_WINDOW=W: the frontier spans the top W star counts.
    if let Some(w) = std::env::var("OBRRT_STAR_WINDOW").ok().and_then(|v| v.parse::<i16>().ok()) {
        obrrt_search::tree::STAR_WINDOW.store(w.max(1), std::sync::atomic::Ordering::Relaxed);
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
    // Dense adds and sparse saves (search/src/dense.rs, tree.rs; all off by
    // default). OBRRT_ADD=winner|winner_all|last_gen|all (comma-separated):
    // which sampled states become candidate nodes; OBRRT_DENSE_GRID (64)
    // units of the dense key, OBRRT_DENSE_CAP (256 with all, else 0 = none) candidates per
    // attempt, OBRRT_DENSE_NE=0 keeps dense nodes out of the new-eligible
    // queue, OBRRT_DENSE_MEMOBJ=0 leaves object lists out of the memory
    // budget. OBRRT_DENSE_EVERY=<n> (1): winner_all/all candidates only every
    // n frames along a rollout (plus its end); OBRRT_DENSE_AIR=0: no mid-air
    // ones (take-off frames and ends still count). OBRRT_DENSE_EVERY=obj[:n]:
    // each objective chooses (Objective::dense_keep; n, 15, for the rest). OBRRT_REPLAY_MAX=<frames>: a node keeps a save only beyond
    // that many frames from its saved ancestor (150 with dense adds on,
    // off otherwise; "off" forces every save). OBRRT_SPARSE_CHECK=<n>:
    // rebuild n save-less nodes at the end and compare.
    // OBRRT_KINSHIP_MULT (1): kinship × this (OBRRT_KIN_UNIT below). OBRRT_NEAR_SEED=1: seed the
    // GA from a nearby start's attempt of the objective (OBRRT_NEAR_R 300
    // units, OBRRT_NEAR_P 0.5, OBRRT_NEAR_KEEP 32 records per objective).
    {
        let get = |k: &str| std::env::var(k).ok();
        let modes = match get("OBRRT_ADD") {
            Some(v) => obrrt_search::dense::Modes::parse(&v).map_err(anyhow::Error::msg)?,
            None => Default::default(),
        };
        let replay = match get("OBRRT_REPLAY_MAX").as_deref() {
            Some("off") => None,
            Some(v) => Some(v.parse::<u32>()?),
            None if modes.any() => Some(150),
            None => None,
        };
        if let Some(r) = replay {
            obrrt_search::tree::REPLAY_MAX.store(r, std::sync::atomic::Ordering::Relaxed);
        }
        if modes.any() {
            anyhow::ensure!(replay.is_some(), "OBRRT_ADD needs sparse saves (OBRRT_REPLAY_MAX=0 saves every node)");
            let grid: u32 = get("OBRRT_DENSE_GRID").map_or(Ok(64), |v| v.parse())?;
            anyhow::ensure!(grid >= 64 && grid % 64 == 0, "OBRRT_DENSE_GRID: a multiple of 64");
            obrrt_search::tree::DENSE_GRID.store(grid, std::sync::atomic::Ordering::Relaxed);
            obrrt_search::dense::configure(Some(obrrt_search::dense::Settings {
                modes,
                grid: grid as f32,
                cap: get("OBRRT_DENSE_CAP").map_or(Ok(if modes.all { 256 } else { 0 }), |v| v.parse())?,
                ne: get("OBRRT_DENSE_NE").is_none_or(|v| v != "0"),
                mem_objects: get("OBRRT_DENSE_MEMOBJ").is_none_or(|v| v != "0"),
                every: match get("OBRRT_DENSE_EVERY").as_deref() {
                    None => 1,
                    Some("obj") => 15,
                    Some(v) => v.strip_prefix("obj:").unwrap_or(v).parse()?,
                },
                per_obj: get("OBRRT_DENSE_EVERY").is_some_and(|v| v == "obj" || v.starts_with("obj:")),
                air: get("OBRRT_DENSE_AIR").is_none_or(|v| v != "0"),
            }));
        }
        if let Some(v) = get("OBRRT_KINSHIP_MULT") {
            cfg.kinship_mult = v.parse()?;
        }
        // OBRRT_KIN_UNIT=attempt: kinship generations in attempts (dense
        // intermediates and replay-chain nodes cost none; tree.rs
        // KIN_ATTEMPT); unset or "node": every node is one; "frames": tree-path
        // game frames / OBRRT_KIN_FRAMES (150) per generation.
        // OBRRT_KIN_SOFT=λ: continuous kinship, start weight × λ^distance
        // (in the unit) instead of the hard set.
        // OBRRT_KIN_CAP (0 = none): most kin-set members, nearest first.
        match get("OBRRT_KIN_UNIT").as_deref() {
            None | Some("") | Some("node") => {}
            Some("attempt") => obrrt_search::tree::KIN_ATTEMPT.store(true, std::sync::atomic::Ordering::Relaxed),
            Some("frames") => obrrt_search::tree::KIN_FRAMES_UNIT.store(true, std::sync::atomic::Ordering::Relaxed),
            Some(v) => anyhow::bail!("OBRRT_KIN_UNIT={v}: expected node, attempt or frames"),
        }
        if let Some(v) = get("OBRRT_KIN_FRAMES") {
            obrrt_search::tree::KIN_FRAMES.store(v.parse()?, std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(v) = get("OBRRT_KIN_SOFT") {
            let l: f64 = v.parse()?;
            anyhow::ensure!(l > 0.0 && l <= 1.0, "OBRRT_KIN_SOFT: 0 < λ <= 1");
            obrrt_search::tree::KIN_SOFT.store(l.to_bits(), std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(v) = get("OBRRT_KIN_CAP") {
            obrrt_search::tree::KIN_CAP.store(v.parse()?, std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(v) = get("OBRRT_SPARSE_CHECK") {
            cfg.rebuild_check = v.parse()?;
        }
        if get("OBRRT_NEAR_SEED").is_some_and(|v| v == "1") {
            obrrt_search::near::configure(Some(obrrt_search::near::Settings {
                radius: get("OBRRT_NEAR_R").map_or(Ok(300.0), |v| v.parse())?,
                p: get("OBRRT_NEAR_P").map_or(Ok(0.5), |v| v.parse())?,
                per_key: get("OBRRT_NEAR_KEEP").map_or(Ok(32), |v| v.parse())?,
            }));
        }
    }
    // OBRRT_FAST_MODAL: objectives::setup::from_env (above).
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
    // OBRRT_REROUTE=1: graft slower duplicates' subtrees onto faster ones by
    // replaying their inputs; OBRRT_REROUTE_SCHEMA=<objective-scattershot
    // schema> keys "the same state" by its objectives (default: the duplicate
    // cell); OBRRT_REROUTE_SHARE (0.25) of a round's slots at most;
    // OBRRT_REROUTE_MARGIN (10) frames saved at least.
    if std::env::var("OBRRT_REROUTE").is_ok_and(|v| v == "1") {
        cfg.reroute = true;
    }
    if let Ok(v) = std::env::var("OBRRT_REROUTE_SCHEMA") {
        cfg.reroute_schema = Some(v);
    }
    // OBRRT_POLISH=<share>: re-attempt the best route's earlier milestones
    // and graft the rest of the route onto sooner results (search/src/polish.rs).
    if let Ok(v) = std::env::var("OBRRT_POLISH") {
        cfg.polish = v.parse()?;
    }
    if let Ok(v) = std::env::var("OBRRT_REROUTE_SHARE") {
        cfg.reroute_share = v.parse()?;
    }
    if let Ok(v) = std::env::var("OBRRT_REROUTE_MARGIN") {
        cfg.reroute_margin = v.parse()?;
    }
    // OBRRT_ZONE_SHARE=1: replay the inputs states took since entering
    // their zone from other entries of it (search/src/share.rs);
    // OBRRT_ZONE_SHARE_K (4) top donors per entry, OBRRT_ZONE_SHARE_CAP
    // (3000) frames per path, OBRRT_ZONE_SHARE_SHARE (0.2) of a round's
    // slots at most, OBRRT_ZONE_SHARE_BG (2) background jobs per round.
    if std::env::var("OBRRT_ZONE_SHARE").is_ok_and(|v| v == "1") {
        cfg.zone_share = true;
    }
    if let Ok(v) = std::env::var("OBRRT_ZONE_SHARE_K") {
        cfg.zone_share_k = v.parse()?;
    }
    if let Ok(v) = std::env::var("OBRRT_ZONE_SHARE_CAP") {
        cfg.zone_share_cap = v.parse()?;
    }
    if let Ok(v) = std::env::var("OBRRT_ZONE_SHARE_SHARE") {
        cfg.zone_share_share = v.parse()?;
    }
    if let Ok(v) = std::env::var("OBRRT_ZONE_SHARE_BG") {
        cfg.zone_share_bg = v.parse()?;
    }
    // OBRRT_REPLAY_BUDGET=<fraction> (e.g. 0.1): replay jobs (reroute,
    // zone share) take no plan slots; after each round's plans they run
    // until their frames reach this fraction of the plans' (search/src/replay.rs).
    // Unset or 0: off.
    if let Ok(v) = std::env::var("OBRRT_REPLAY_BUDGET") {
        let f: f64 = v.parse()?;
        if f > 0.0 {
            cfg.replay_budget = Some(f);
        }
    }
    if std::env::var("OBRRT_STAR").is_ok_and(|v| v == "1") {
        cfg.rrt_star = true;
    }
    // OBRRT_MOVES=1 (the movement library) and OBRRT_MOVE_GA=1 (the GA's
    // first generation also simulates the library's moves as forced policies,
    // search/src/movegenome.rs): objectives::setup::from_env (above).
    let on = |k: &str| std::env::var(k).is_ok_and(|v| v == "1");
    // OBRRT_CHAIN=route: one objective per attempt, a follow-on only along
    // the best route (OBRRT_CHAIN_FOLLOW, 0.5). OBRRT_SAMPLER=rate: entry and
    // start weights from improvement per frame (OBRRT_RATE_FLOOR 0.15, _CAP
    // 4, _TAU 5e7 frames, _PRIOR 2e5 frames, _NOSTART 5000 frames, _NODE_PRIOR
    // 3e5 frames, _NODE_CAP 2) instead of the failure decay.
    match std::env::var("OBRRT_CHAIN").as_deref() {
        Ok("route") => cfg.chain_route = true,
        Ok("stack") | Err(_) => {}
        Ok(v) => bail!("OBRRT_CHAIN: route or stack, not {v}"),
    }
    if let Ok(v) = std::env::var("OBRRT_CHAIN_FOLLOW") {
        cfg.chain_follow = v.parse()?;
    }
    match std::env::var("OBRRT_SAMPLER").as_deref() {
        Ok("rate") => cfg.sampler_rate = true,
        Ok("decay") | Err(_) => {}
        Ok(v) => bail!("OBRRT_SAMPLER: rate or decay, not {v}"),
    }
    for (k, f) in [
        ("FLOOR", &mut cfg.rate.floor),
        ("CAP", &mut cfg.rate.cap),
        ("TAU", &mut cfg.rate.tau),
        ("PRIOR", &mut cfg.rate.prior_frames),
        ("NOSTART", &mut cfg.rate.nostart_frames),
        ("NODE_PRIOR", &mut cfg.node_prior),
        ("NODE_CAP", &mut cfg.node_cap),
        ("KIND_SCALE", &mut cfg.rate.kind_scale),
        ("ROUTE", &mut cfg.rate.route),
        ("NOVEL", &mut cfg.rate.novel),
    ] {
        if let Ok(v) = std::env::var(format!("OBRRT_RATE_{k}")) {
            *f = v.parse()?;
        }
    }
    // OBRRT_RATE_MODE=hier: hierarchical shrinkage (entry rate toward its
    // kind's, prior OBRRT_RATE_PRIOR frames; kind rate toward the pool's,
    // prior OBRRT_RATE_KIND_SCALE (10) × that). OBRRT_RATE_NOSTART_SKIP=1:
    // entries no live state could start are skipped, not charged
    // (re-checked against new states, OBRRT_RATE_SKIP_BUDGET 2e6 a round).
    match std::env::var("OBRRT_RATE_MODE").as_deref() {
        Ok("hier") => cfg.rate.hier = true,
        Ok("flat") | Err(_) => {}
        Ok(v) => bail!("OBRRT_RATE_MODE: hier or flat, not {v}"),
    }
    cfg.rate.nostart_skip = on("OBRRT_RATE_NOSTART_SKIP");
    if let Ok(v) = std::env::var("OBRRT_RATE_SKIP_BUDGET") {
        cfg.rate.skip_budget = v.parse()?;
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
    // (configured in objectives::setup::from_env, above).
    println!("techniques ({tech}): {tech_names:?}");
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
        // OBRRT_BOOT_NOWAIT: a cut that ends before control (a star's
        // kick-out) starts as soon as Mario is controllable.
        if nowait && st.controllable() {
            break;
        }
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
    // OBRRT_GEO=1: walkable distances over the level's floors (core/src/geo.rs),
    // for the goal focus's ranking and its approach moves (search/src/engine.rs).
    if std::env::var("OBRRT_GEO").is_ok_and(|v| v == "1") {
        obrrt_core::geo::set_enabled(true);
    }
    // OBRRT_ZONE_LOCK=1: the tree stays in the root's level (search/src/tree.rs).
    // Every star count stays open unless OBRRT_STAR_WINDOW says otherwise: a
    // star that doesn't end the visit (100 coins) mustn't close the others.
    if std::env::var("OBRRT_ZONE_LOCK").is_ok_and(|v| v == "1") {
        obrrt_search::tree::ZONE_LOCK.store(root.level as i32, std::sync::atomic::Ordering::Relaxed);
        if std::env::var_os("OBRRT_STAR_WINDOW").is_none() {
            obrrt_search::tree::STAR_WINDOW.store(99, std::sync::atomic::Ordering::Relaxed);
        }
        println!("zone lock: level {}", root.level);
    }
    let save = game.save_state();
    // OBRRT_SPEC=<file.json or dir>: declarative objectives (spec/) added to
    // the Registry's; unset, the Registry itself.
    let generator = obrrt_spec::wrap(Arc::new(Registry))?;
    let mut engine = Engine::new(cfg, &game_gen, boot, save, root, generator)?;
    // OBRRT_RESUME=<run dir>: continue that run's tree.
    if let Ok(dir) = std::env::var("OBRRT_RESUME") {
        engine.resume(&dir)?;
    }
    engine.run()
}

fn rayon_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}
