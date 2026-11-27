//! One process runs two searches with different configs and each makes the tree it makes alone.
//!
//! The configs differ in what the tree, the engine and the objectives each read (zone lock, dense adds,
//! kinship, technique set, the rolling-horizon controller's parameters), and the engines differ in the run-time
//! state each owns (a position goal; what the mover has published; the door and grid caches), so a setting or
//! state that leaked between engines (a process-wide static, the first engine's thread pool, one config read
//! for the other, one engine's publish epoch tagging the other's caches) changes a tree. Not covered: the static
//! floor snapshots of `obrrt_core::geo`, which are a function of the ROM and the zone (see `obrrt_core::runtime`).
//! Each config's reference tree comes from a fresh process (A's from this one, run first; B's from a child process
//! running only it), so a leak either way shows. Needs the game ROM: SM64_ROM names it; without it the test says so and passes.

use std::path::Path;
use std::sync::Arc;

use obrrt_core::buttons::{A_BUTTON, START_BUTTON};
use obrrt_core::config::{self, SearchConfig};
use obrrt_core::runtime::Runtime;
use obrrt_core::{GamePad, check_layout, read_state};
use obrrt_objectives::Registry;
use obrrt_search::{Config, Engine};
use sm64_binds::SM64GameGenerator;

/// The fixed boot of obrrt2 (run/src/main.rs), then idle until Mario stands still.
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

/// A position goal (zone, centre, radius) and the share of plans drawn from inside its bubble.
type Goal = Option<(((u16, u8), [f32; 3], f32), f64)>;

/// Runs one search under `sc` (its own run-time state, with `goal` set) for `frames` emulated frames and returns its tree.bin.
fn tree_of(rom: &str, sc: &SearchConfig, goal: Goal, out: &str, frames: u64) -> Vec<u8> {
    let sc = config::leak(sc.clone());
    // What the process's drivers do once from the environment: the steering controller, the movement library.
    obrrt_objectives::setup::from_config(sc).unwrap();
    let mut cfg = Config::default();
    obrrt_search::envconfig::apply_globals(sc, &mut cfg).unwrap();
    cfg.out_dir = out.into();
    cfg.seed = 1;
    cfg.minutes = 1e9;
    cfg.max_frames = Some(frames);
    cfg.runtime = Some(Runtime::new());
    let game_gen = SM64GameGenerator::from_file(rom).unwrap();
    let mut game = game_gen.create_game().unwrap();
    let mut boot = boot_inputs();
    for p in &boot {
        game.step_game(*p).unwrap();
    }
    let mut st = read_state(&game, boot.len() as u32, false);
    let mut still = 0;
    while still < 10 || !st.controllable() {
        let before = st.pos;
        let idle = GamePad::new(0, 0, 0);
        game.step_game(idle).unwrap();
        boot.push(idle);
        st = read_state(&game, boot.len() as u32, false);
        still = if st.pos == before { still + 1 } else { 0 };
        assert!(boot.len() < 30_000, "Mario didn't stand still after booting");
    }
    check_layout(&game).unwrap();
    let root = read_state(&game, boot.len() as u32, true);
    let save = game.save_state();
    let generator: Arc<dyn obrrt_core::Generator> = Arc::new(Registry);
    let mut engine = Engine::new(cfg, &game_gen, boot, save, root, generator).unwrap();
    if let Some((g, share)) = goal {
        engine.set_pos_goal(Some(g));
        engine.set_pos_goal_strength(0.0, share);
    }
    engine.run().unwrap();
    std::fs::read(format!("{out}/tree.bin")).unwrap()
}

/// A: the castle's zone-locked tree with dense adds and a position goal over the castle grounds' bridge; B: free, no
/// dense adds, no kinship stretch, movement techniques only, the rolling-horizon controller with a small population
/// (its parameters are derived from the config once per config), no goal.
fn configs() -> (SearchConfig, Goal, SearchConfig) {
    let mut a = SearchConfig::default();
    a.zone.lock = true;
    a.dedup.dense_adds = true;
    let mut b = SearchConfig::default();
    b.dedup.dense_adds = false;
    b.kin.mult = 1.0;
    b.objectives.tech = "movement".into();
    b.research.rh.on = true;
    b.research.rh.pop = 16;
    (a, Some((((16, 1), [0.0, 811.0, -100.0], 4000.0), 0.4)), b)
}

const FRAMES: u64 = 250_000;

/// Set in the child process that makes B's tree first thing (into this directory).
const CHILD: &str = "TWO_CONFIGS_B_FRESH";

#[test]
fn two_engines_with_different_configs_in_one_process() {
    let Some(rom) = std::env::var("SM64_ROM").ok().filter(|p| Path::new(p).exists()) else {
        eprintln!("two_configs: SM64_ROM not set (or missing); skipped");
        return;
    };
    let (a, goal_a, b) = configs();
    if let Ok(out) = std::env::var(CHILD) {
        tree_of(&rom, &b, None, &out, FRAMES);
        return;
    }
    let frames = FRAMES;
    let dir = std::env::temp_dir().join(format!("obrrt-two-configs-{}", std::process::id()));
    let d = |n: &str| dir.join(n).to_string_lossy().into_owned();

    // Each config's tree from a fresh process: A here (first), B in a child process running only it.
    let a_alone = tree_of(&rom, &a, goal_a, &d("a-alone"), frames);
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["two_engines_with_different_configs_in_one_process", "--exact", "--test-threads=1", "--nocapture"])
        .env(CHILD, d("b-alone"))
        .status()
        .unwrap();
    assert!(child.success(), "B's child process failed");
    let b_alone = std::fs::read(format!("{}/tree.bin", d("b-alone"))).unwrap();
    assert_ne!(a_alone, b_alone, "the two configs must make different trees, or the test shows nothing");

    // Back to back, then at the same time, in this process.
    assert_eq!(tree_of(&rom, &b, None, &d("b-after-a"), frames), b_alone, "B after A");
    assert_eq!(tree_of(&rom, &a, goal_a, &d("a-after-b"), frames), a_alone, "A after B");
    let (a_both, b_both) = std::thread::scope(|s| {
        let ta = s.spawn(|| tree_of(&rom, &a, goal_a, &d("a-together"), frames));
        let tb = s.spawn(|| tree_of(&rom, &b, None, &d("b-together"), frames));
        (ta.join().unwrap(), tb.join().unwrap())
    });
    assert_eq!(a_both, a_alone, "A alongside B");
    assert_eq!(b_both, b_alone, "B alongside A");
    let _ = std::fs::remove_dir_all(&dir);
}
