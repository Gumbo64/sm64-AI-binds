//! Runs one Objective-RRT tree from a single JSON config, with everything
//! configurable: run settings, every engine/GA `Config` field, the search's
//! global switches, objective-side switches (env), techniques, and
//! per-kind objective weights.
//!
//! cargo run --release --example tree -- run/examples/tree.json
//!
//! Every key is optional; anything left out keeps the `obrrt` default. An
//! unknown key is an error, so typos don't pass silently. The resolved
//! settings are written to `<out_dir>/config.json` next to the run's outputs.
//! See `objectives/OBJECTIVES.md` for objective kinds and switches, and
//! README.md for the outputs.
#![recursion_limit = "256"]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;

use anyhow::{Context, Result, anyhow, bail};
use obrrt_core::buttons::{A_BUTTON, START_BUTTON};
use obrrt_core::{GamePad, check_layout, read_state};
use obrrt_objectives::Registry;
use obrrt_search::{Config, Engine};
use serde_json::{Map, Value};
use sm64_binds::SM64GameGenerator;

fn main() -> Result<()> {
    // The steering controller (objectives/src/moves; move library off).
    obrrt_objectives::moves::install_controller();
    let path = std::env::args().nth(1).ok_or_else(|| anyhow!("usage: tree <config.json>"))?;
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading {path}"))?;
    let root: Map<String, Value> = serde_json::from_str(&text).with_context(|| format!("parsing {path}"))?;
    check_keys(&root, &["run", "env", "kind_weights", "techniques", "switches", "backchain", "engine"], "top level")?;
    let run = obj(&root, "run")?;
    check_keys(&run, &["rom", "out_dir", "seed", "minutes", "max_frames", "threads", "boot", "nowait", "resume"], "run")?;
    let out_dir = str_or(&run, "out_dir", "runs/tree")?;
    std::fs::create_dir_all(&out_dir)?;

    // 1. Environment first: objective-side switches (OBRRT_MOVE_WEIGHT,
    // OBRRT_SWIM_HOLD, OBRRT_GP_CATCH, OBRRT_TALK_NIL, ...) are read lazily
    // from it, and so are the thread count and the kind-weight table.
    let mut env: BTreeMap<String, String> = BTreeMap::new();
    for (k, v) in obj(&root, "env")?.into_iter().filter(|(k, _)| !k.starts_with('_')) {
        env.insert(k, match v {
            Value::String(s) => s,
            other => other.to_string(),
        });
    }
    if let Some(t) = run.get("threads") {
        env.insert("RAYON_NUM_THREADS".into(), t.to_string());
    }
    let mut weights = obj(&root, "kind_weights")?;
    weights.retain(|k, _| !k.starts_with('_'));
    if let Some((k, v)) = weights.iter().find(|(_, v)| !v.is_number()) {
        bail!("kind_weights.{k}: a number, not {v}");
    }
    if !weights.is_empty() {
        // Per-kind objective weight multipliers (e.g. {"Npc::Talk": 0.25}),
        // applied where each objective's weight is set (search/src/pool.rs).
        let file = format!("{out_dir}/kind_weights.json");
        std::fs::write(&file, serde_json::to_string_pretty(&Value::Object(weights))?)?;
        env.insert("OBRRT_KIND_WEIGHTS".into(), std::fs::canonicalize(&file)?.display().to_string());
    }
    for (k, v) in &env {
        // SAFETY: single-threaded here; nothing has read the environment yet.
        unsafe { std::env::set_var(k, v) };
    }

    // 2. Techniques: "all", "none", a comma list, "-name" or a tag.
    let tech = match root.get("techniques") {
        None => "all".to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) => a.iter().map(|v| v.as_str().unwrap_or_default()).collect::<Vec<_>>().join(","),
        Some(v) => bail!("techniques: a string or list, not {v}"),
    };
    obrrt_objectives::extra::tech::configure(&tech).map_err(anyhow::Error::msg)?;

    // 3. Global search switches.
    let sw = obj(&root, "switches")?;
    check_keys(
        &sw,
        &["starset_key", "keep_wip", "fast_modal"],
        "switches",
    )?;
    obrrt_search::engine::STARSET.store(bool_or(&sw, "starset_key", false)?, Relaxed);
    obrrt_search::tree::KEEP_WIP.store(bool_or(&sw, "keep_wip", false)?, Relaxed);
    if let Some(v) = sw.get("fast_modal") {
        obrrt_objectives::set_fast_modal(v.as_bool().ok_or_else(|| anyhow!("switches.fast_modal: bool"))?);
    }

    // 5. Engine and GA settings: every `Config` field.
    let mut cfg = Config::default();
    cfg.out_dir = out_dir.clone();
    cfg.seed = num_or(&run, "seed", 1.0)? as u64;
    cfg.minutes = num_or(&run, "minutes", 30.0)?;
    if let Some(v) = run.get("max_frames") {
        cfg.max_frames = Some(v.as_f64().ok_or_else(|| anyhow!("run.max_frames: a number"))? as u64);
    }
    apply_engine(&mut cfg, &obj(&root, "engine")?)?;

    // 6. Boot: the fixed menu-skip inputs, the 2026 TAS's first 1646 frames, or an m64.
    let rom = str_or(&run, "rom", "../baserom.us.z64")?;
    let game_gen = SM64GameGenerator::from_file(&rom)?;
    let mut game = game_gen.create_game()?;
    let boot_kind = str_or(&run, "boot", "default")?;
    let mut boot = match boot_kind.as_str() {
        "default" => boot_inputs(),
        "tas" => tas_boot_inputs()?,
        path => read_m64(path)?,
    };
    for p in &boot {
        game.step_game(*p)?;
    }
    let mut st = read_state(&game, boot.len() as u32, false);
    let nowait = bool_or(&run, "nowait", boot_kind == "tas")?;
    let mut still = if nowait && st.controllable() { 10 } else { 0 };
    let prefix = boot.len();
    while still < 10 || !st.controllable() {
        let before = st.pos;
        let idle = GamePad::new(0, 0, 0);
        game.step_game(idle)?;
        boot.push(idle);
        st = read_state(&game, boot.len() as u32, false);
        still = if st.pos == before { still + 1 } else { 0 };
        if boot.len() - prefix > 30_000 {
            bail!("Mario didn't stand still after booting");
        }
    }
    check_layout(&game)?;
    let root_st = read_state(&game, boot.len() as u32, true);

    // Record what this run used.
    let resolved = serde_json::json!({
        "config_file": path, "rom": rom, "boot": boot_kind, "nowait": nowait,
        "root_frame": root_st.frame, "techniques": obrrt_objectives::extra::tech::enabled_names(),
        "env": env, "engine": describe(&cfg), "file": Value::Object(root.clone()),
    });
    std::fs::write(format!("{out_dir}/config.json"), serde_json::to_string_pretty(&resolved)?)?;
    println!(
        "root: frame {} level {} area {} stars {}; techniques {:?}; config -> {out_dir}/config.json",
        root_st.frame,
        root_st.level,
        root_st.area,
        root_st.stars,
        obrrt_objectives::extra::tech::enabled_names()
    );

    let save = game.save_state();
    let mut engine = Engine::new(cfg, &game_gen, boot, save, root_st, Arc::new(Registry))?;
    if let Some(dir) = run.get("resume") {
        engine.resume(dir.as_str().ok_or_else(|| anyhow!("run.resume: a run dir"))?)?;
    }
    engine.run()
}

/// Sets `Config` fields from the "engine" object by name.
fn apply_engine(cfg: &mut Config, e: &Map<String, Value>) -> Result<()> {
    macro_rules! fields {
        ($($name:ident: $kind:tt),* $(,)?) => {{
            let known = [$(stringify!($name)),*, "mem_mb", "report_every_secs", "fitness"];
            check_keys(e, &known, "engine")?;
            $( if let Some(v) = e.get(stringify!($name)) { fields!(@set cfg.$name, v, $kind, stringify!($name)); } )*
        }};
        (@set $f:expr, $v:expr, f64, $n:expr) => { $f = $v.as_f64().ok_or_else(|| anyhow!("engine.{}: a number", $n))? };
        (@set $f:expr, $v:expr, f32, $n:expr) => { $f = $v.as_f64().ok_or_else(|| anyhow!("engine.{}: a number", $n))? as f32 };
        (@set $f:expr, $v:expr, u32, $n:expr) => { $f = $v.as_u64().ok_or_else(|| anyhow!("engine.{}: an integer", $n))? as u32 };
        (@set $f:expr, $v:expr, u64, $n:expr) => { $f = $v.as_u64().ok_or_else(|| anyhow!("engine.{}: an integer", $n))? };
        (@set $f:expr, $v:expr, usize, $n:expr) => { $f = $v.as_u64().ok_or_else(|| anyhow!("engine.{}: an integer", $n))? as usize };
        (@set $f:expr, $v:expr, bool, $n:expr) => { $f = $v.as_bool().ok_or_else(|| anyhow!("engine.{}: true/false", $n))? };
        (@set $f:expr, $v:expr, opt_f64, $n:expr) => { $f = if $v.is_null() { None } else { Some($v.as_f64().ok_or_else(|| anyhow!("engine.{}: a number or null", $n))?) } };
    }
    fields! {
        // Objective weight decay and start back-off.
        decay: f64, state_decay: f64, alpha: f64, gamma: f64, expire_rounds: u32, kind_floor: f64,
        // Chains of objectives per attempt.
        chain_prob: f64, max_chain: usize, chain_reach: f32,
        // GA.
        ga_scale: f64, ga_new_ops: bool, ga_pink: f64, ga_per_objective_budget: bool,
        // Start selection and speed.
        rrt_star: bool, arrival: bool, arrival_pace: f64, arrival_margin: f64,
        arrival_band: opt_f64, time_cost: f64,
        rewire_share: f64, rewire_margin: u32, route_share: f64, pick_by_b: bool,
        // Rounds, dedupe, bandits.
        round_plans: usize, soft_dedup: bool,
        // Boosts and tuning.
        new_eligible: bool,
    }
    if let Some(v) = e.get("mem_mb") {
        cfg.mem_budget = v.as_u64().ok_or_else(|| anyhow!("engine.mem_mb: an integer"))? as usize * 1_000_000;
    }
    if let Some(v) = e.get("report_every_secs") {
        cfg.report_every = std::time::Duration::from_secs_f64(v.as_f64().ok_or_else(|| anyhow!("engine.report_every_secs: a number"))?);
    }
    match e.get("fitness").map(|v| v.as_str()) {
        None => {}
        Some(Some("steps")) => cfg.fitness = obrrt_search::ga::Fitness::Steps,
        Some(Some("met")) => cfg.fitness = obrrt_search::ga::Fitness::Met,
        Some(_) => bail!("engine.fitness: \"steps\" or \"met\""),
    }
    Ok(())
}

/// The engine settings actually used, for `config.json`.
fn describe(c: &Config) -> Value {
    serde_json::json!({
        "seed": c.seed, "minutes": c.minutes, "max_frames": c.max_frames, "decay": c.decay,
        "state_decay": c.state_decay, "alpha": c.alpha, "gamma": c.gamma, "expire_rounds": c.expire_rounds,
        "kind_floor": c.kind_floor, "chain_prob": c.chain_prob, "max_chain": c.max_chain,
        "chain_reach": c.chain_reach, "ga_scale": c.ga_scale,
        "fitness": format!("{:?}", c.fitness), "ga_new_ops": c.ga_new_ops, "ga_pink": c.ga_pink,
        "ga_per_objective_budget": c.ga_per_objective_budget, "rrt_star": c.rrt_star, "arrival": c.arrival,
        "arrival_pace": c.arrival_pace, "arrival_margin": c.arrival_margin,
        "arrival_band": c.arrival_band, "time_cost": c.time_cost, "rewire_share": c.rewire_share,
        "rewire_margin": c.rewire_margin, "route_share": c.route_share,
        "pick_by_b": c.pick_by_b, "round_plans": c.round_plans,
        "soft_dedup": c.soft_dedup, "mem_mb": c.mem_budget / 1_000_000,
        "new_eligible": c.new_eligible,
    })
}

fn obj(m: &Map<String, Value>, k: &str) -> Result<Map<String, Value>> {
    match m.get(k) {
        None => Ok(Map::new()),
        Some(Value::Object(o)) => Ok(o.clone()),
        Some(v) => bail!("{k}: an object, not {v}"),
    }
}

fn check_keys(m: &Map<String, Value>, known: &[&str], section: &str) -> Result<()> {
    for k in m.keys() {
        if !k.starts_with('_') && !known.contains(&k.as_str()) {
            bail!("unknown key {section}.{k} (known: {})", known.join(", "));
        }
    }
    Ok(())
}

fn str_or(m: &Map<String, Value>, k: &str, d: &str) -> Result<String> {
    m.get(k).map_or(Ok(d.to_string()), |v| v.as_str().map(str::to_string).ok_or_else(|| anyhow!("{k}: a string")))
}

fn num_or(m: &Map<String, Value>, k: &str, d: f64) -> Result<f64> {
    m.get(k).map_or(Ok(d), |v| v.as_f64().ok_or_else(|| anyhow!("{k}: a number")))
}

fn bool_or(m: &Map<String, Value>, k: &str, d: bool) -> Result<bool> {
    m.get(k).map_or(Ok(d), |v| v.as_bool().ok_or_else(|| anyhow!("{k}: true/false")))
}

/// The menu-skipping power-on inputs (as `obrrt`).
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

/// The 2026 120-star TAS up to first control (frame 1646).
fn tas_boot_inputs() -> Result<Vec<GamePad>> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../ref/tas/full-120star-2026-U.m64");
    let bytes = std::fs::read(path)?;
    Ok(bytes[0x400..].chunks_exact(4).take(1646).map(pad).collect())
}

/// A headerless input movie (4 bytes a frame), as `obrrt <start.m64>`.
fn read_m64(path: &str) -> Result<Vec<GamePad>> {
    Ok(std::fs::read(path).with_context(|| format!("boot movie {path}"))?.chunks_exact(4).map(pad).collect())
}

fn pad(c: &[u8]) -> GamePad {
    GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)
}
