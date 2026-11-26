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
    obrrt_search::envconfig::from_env(&mut cfg)?;
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
    let nowait = std::env::var("OBRRT_BOOT_NOWAIT").is_ok_and(|v| v != "0") && args.len() > 5;
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
    obrrt_search::envconfig::after_root(&root);
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
