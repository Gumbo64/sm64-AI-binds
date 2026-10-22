//! Plain RRT over the real game, with a "dumb" steering controller: hold the
//! stick towards the sampled point, and while the point is above Mario,
//! press/release A on alternate frames.
//!
//! cargo run --release --example rrt -- <rom> [out_dir=runs/rrt] [seed=1]
//!     [minutes=30] [segment=20] [<start.m64>]
//!
//! Each round, every worker:
//!
//! 1. Picks a zone (level, area) the tree has reached, uniformly, and samples
//!    a point uniformly in [-8192, 8192]^3.
//! 2. Finds the tree node in that zone nearest the point (3D Euclidean).
//! 3. Restores that node and plays `segment` frames of the controller towards
//!    the point.
//! 4. Returns the end state, which joins the tree unless Mario died, lost a
//!    life, or moved less than a unit.
//!
//! Every node keeps its save state as the 4 KiB pages that differ from the
//! root's, XORed with them and LZ4-compressed (a full state is ~8.6 MB). Whenever a
//! node has more stars than the root, its full input file (from power-on) is
//! written to `<out_dir>/star<n>-frame<f>.m64`. The tree's nodes go to
//! `<out_dir>/nodes.csv` every 30 seconds, and at the end the deepest node's
//! input file to `<out_dir>/deepest.m64`.

use std::collections::HashMap;
use std::f32::consts::TAU;
use std::fs;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use sm64_ai::util::{read_gamepads_from_file, skip_menu_pads, write_gamepads_to_file};
use sm64_binds::buttons::A_BUTTON;
use sm64_binds::{GamePad, GameState, SM64Game, SM64GameGenerator, SaveState};

/// Every level fits in this cube.
const WORLD: f32 = 8192.0;
/// Full stick deflection.
const STICK: f32 = 127.0;
/// Below this, Mario has no health left.
const DEAD_HEALTH: i32 = 0x100;
/// Save states are diffed against the root's in pages of this size.
const PAGE: usize = 4096;
/// Extends per worker per round.
const PER_WORKER: usize = 4;

type Zone = (i32, i32);

struct Node {
    parent: Option<u32>,
    depth: u32,
    /// Inputs from the parent to this node.
    inputs: Vec<GamePad>,
    st: GameState,
    save: Delta,
}

/// A save state as the pages where it differs from the root's, each XORed
/// with the root's page (mostly zeros) and LZ4-compressed together.
#[derive(Default)]
struct Delta {
    pages: Vec<u32>,
    xor: Vec<u8>,
}

impl Delta {
    fn new(root: &[u8], state: &SaveState) -> Delta {
        let bytes = state.as_bytes();
        let mut pages = Vec::new();
        let mut xor = Vec::new();
        for i in 0..bytes.len().div_ceil(PAGE) {
            let r = i * PAGE..((i + 1) * PAGE).min(bytes.len());
            if bytes[r.clone()] != root[r.clone()] {
                pages.push(i as u32);
                xor.extend(bytes[r.clone()].iter().zip(&root[r]).map(|(a, b)| a ^ b));
            }
        }
        Delta { pages, xor: lz4_flex::compress_prepend_size(&xor) }
    }

    fn bytes(&self) -> usize {
        self.xor.len() + 4 * self.pages.len()
    }

    /// The full save state: the root's bytes with these pages XORed back in.
    fn apply(&self, root: &[u8]) -> Result<SaveState> {
        let mut bytes = root.to_vec();
        if !self.pages.is_empty() {
            let xor = lz4_flex::decompress_size_prepended(&self.xor)?;
            let mut chunks = xor.as_slice();
            for &i in &self.pages {
                let at = i as usize * PAGE;
                let len = PAGE.min(bytes.len() - at);
                for (b, x) in bytes[at..at + len].iter_mut().zip(&chunks[..len]) {
                    *b ^= x;
                }
                chunks = &chunks[len..];
            }
        }
        SaveState::from_bytes(bytes)
    }
}

fn zone(s: &GameState) -> Zone {
    (s.level_num, s.area_index)
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// The stick that moves Mario towards `target` (camera-relative: SM64's
/// intended yaw is `atan2s(-stick_y, stick_x) + camera yaw`, with yaw 0 facing
/// +z), and A on alternate frames while `target` is above him.
fn steer(st: &GameState, target: [f32; 3], frame: usize) -> GamePad {
    let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
    let desired = dx.atan2(dz);
    let camera = st.lakitu_yaw as i16 as f32 * TAU / 65536.0;
    let rel = desired - camera;
    let stick_x = (STICK * rel.sin()).round() as i8;
    let stick_y = (-STICK * rel.cos()).round() as i8;
    let button = if target[1] > st.pos[1] && frame % 2 == 0 { A_BUTTON } else { 0 };
    GamePad::new(button, stick_x, stick_y)
}

/// The inputs from power-on to `nodes[id]`.
fn full_path(boot: &[GamePad], nodes: &[Node], id: u32) -> Vec<GamePad> {
    let mut chain = Vec::new();
    let mut at = Some(id);
    while let Some(n) = at {
        chain.push(n);
        at = nodes[n as usize].parent;
    }
    let mut path = boot.to_vec();
    for &n in chain.iter().rev() {
        path.extend(&nodes[n as usize].inputs);
    }
    path
}

fn write_nodes_csv(nodes: &[Node], path: &str) -> Result<()> {
    let mut out = String::from("id,parent,depth,x,y,z,level,area,stars\n");
    for (i, n) in nodes.iter().enumerate() {
        let parent = n.parent.map_or(-1, |p| p as i64);
        out += &format!(
            "{i},{parent},{},{},{},{},{},{},{}\n",
            n.depth, n.st.pos[0], n.st.pos[1], n.st.pos[2], n.st.level_num, n.st.area_index, n.st.num_stars
        );
    }
    fs::write(path, out)?;
    Ok(())
}

struct Candidate {
    parent: u32,
    inputs: Vec<GamePad>,
    st: GameState,
    save: Delta,
}

fn extend(
    root: &[u8],
    game: &mut SM64Game,
    nodes: &[Node],
    by_zone: &HashMap<Zone, Vec<u32>>,
    zones: &[Zone],
    segment: usize,
    rng: &mut SmallRng,
) -> Result<Option<Candidate>> {
    let z = zones[rng.random_range(0..zones.len())];
    let target = [
        rng.random_range(-WORLD..WORLD),
        rng.random_range(-WORLD..WORLD),
        rng.random_range(-WORLD..WORLD),
    ];
    let near = *by_zone[&z]
        .iter()
        .min_by(|&&a, &&b| {
            let da = dist2(nodes[a as usize].st.pos, target);
            let db = dist2(nodes[b as usize].st.pos, target);
            da.total_cmp(&db)
        })
        .unwrap();
    let start = &nodes[near as usize].st;

    game.load_state(&nodes[near as usize].save.apply(root)?)?;
    let mut st = start.clone();
    let mut inputs = Vec::with_capacity(segment);
    for f in 0..segment {
        let pad = steer(&st, target, f);
        game.step_game(pad)?;
        inputs.push(pad);
        st = game.get_game_state()?;
        if st.health < DEAD_HEALTH || st.lives < start.lives || st.in_credits != 0 {
            return Ok(None);
        }
    }
    if zone(&st) == zone(start) && dist2(st.pos, start.pos) < 1.0 {
        return Ok(None);
    }
    let save = Delta::new(root, &game.save_state());
    Ok(Some(Candidate { parent: near, inputs, st, save }))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        bail!("usage: rrt <rom> [out_dir=runs/rrt] [seed=1] [minutes=30] [segment=20] [<start.m64>]");
    }
    let rom = &args[1];
    let out_dir = args.get(2).cloned().unwrap_or("runs/rrt".into());
    let seed: u64 = args.get(3).map_or(Ok(1), |s| s.parse())?;
    let minutes: f64 = args.get(4).map_or(Ok(30.0), |s| s.parse())?;
    let segment: usize = args.get(5).map_or(Ok(20), |s| s.parse())?;
    let start_m64 = args.get(6);
    fs::create_dir_all(&out_dir)?;

    let generator = SM64GameGenerator::from_file(rom)?;
    let threads = rayon::current_num_threads();
    let mut games: Vec<SM64Game> = (0..threads).map(|_| generator.create_game()).collect::<Result<_>>()?;

    // Boot, then wait for Mario to stand still.
    let mut boot = match start_m64 {
        Some(p) => read_gamepads_from_file(p)?,
        None => skip_menu_pads(),
    };
    let root_game = &mut games[0];
    for pad in &boot {
        root_game.step_game(*pad)?;
    }
    let mut st = root_game.get_game_state()?;
    let mut still = 0;
    while still < 10 {
        let before = st.pos;
        let idle = GamePad::new(0, 0, 0);
        root_game.step_game(idle)?;
        boot.push(idle);
        st = root_game.get_game_state()?;
        still = if st.pos == before { still + 1 } else { 0 };
        if boot.len() > 20_000 {
            bail!("Mario didn't stand still");
        }
    }
    let root_save = root_game.save_state();
    println!(
        "root: frame {} level {} area {} pos {:?} stars {}; save state {:.1} MB; {threads} workers",
        boot.len(),
        st.level_num,
        st.area_index,
        st.pos,
        st.num_stars,
        root_save.as_bytes().len() as f64 / 1e6
    );
    let root_stars = st.num_stars;

    let root = root_save.into_bytes();
    let mut nodes = vec![Node { parent: None, depth: 0, inputs: vec![], st: st.clone(), save: Delta::default() }];
    let mut delta_bytes = 0usize;
    let mut by_zone: HashMap<Zone, Vec<u32>> = HashMap::from([(zone(&st), vec![0])]);
    let mut zones = vec![zone(&st)];
    let mut best_stars = root_stars;
    let mut rngs: Vec<SmallRng> = (0..threads).map(|i| SmallRng::seed_from_u64(seed * 1000 + i as u64)).collect();

    let begin = Instant::now();
    let mut last_report = Instant::now();
    let mut last_csv = Instant::now();
    let (mut tried, mut frames) = (0u64, 0u64);
    let csv_path = format!("{out_dir}/nodes.csv");

    while begin.elapsed() < Duration::from_secs_f64(minutes * 60.0) {
        let results: Vec<Result<Vec<Candidate>>> = games
            .par_iter_mut()
            .zip(rngs.par_iter_mut())
            .map(|(game, rng)| {
                let mut out = Vec::new();
                for _ in 0..PER_WORKER {
                    if let Some(c) = extend(&root, game, &nodes, &by_zone, &zones, segment, rng)? {
                        out.push(c);
                    }
                }
                Ok(out)
            })
            .collect();
        tried += (threads * PER_WORKER) as u64;
        frames += (threads * PER_WORKER * segment) as u64;

        for c in results.into_iter().collect::<Result<Vec<_>>>()?.into_iter().flatten() {
            let id = nodes.len() as u32;
            let z = zone(&c.st);
            if !by_zone.contains_key(&z) {
                println!("new zone {z:?} at node {id} pos {:?}", c.st.pos);
                zones.push(z);
            }
            by_zone.entry(z).or_default().push(id);
            let depth = nodes[c.parent as usize].depth + 1;
            let stars = c.st.num_stars;
            delta_bytes += c.save.bytes();
            nodes.push(Node { parent: Some(c.parent), depth, inputs: c.inputs, st: c.st, save: c.save });

            if stars > best_stars {
                best_stars = stars;
                let path = full_path(&boot, &nodes, id);
                let file = format!("{out_dir}/star{stars}-frame{}.m64", path.len());
                write_gamepads_to_file(&path, &file)?;
                println!("STAR {stars} at node {id}: wrote {file}");
            }
        }

        if last_report.elapsed() > Duration::from_secs(10) {
            last_report = Instant::now();
            let secs = begin.elapsed().as_secs_f64();
            let top = nodes.iter().map(|n| n.st.pos[1]).fold(f32::MIN, f32::max);
            println!(
                "{:>6.0}s  nodes {:>7}  accepted {:>5.1}%  zones {}  stars {}  max y {:.0}  {:.0} frames/s  states {:.0} MB",
                secs,
                nodes.len(),
                100.0 * (nodes.len() - 1) as f64 / tried as f64,
                zones.len(),
                best_stars,
                top,
                frames as f64 / secs,
                delta_bytes as f64 / 1e6
            );
        }
        if last_csv.elapsed() > Duration::from_secs(30) {
            last_csv = Instant::now();
            write_nodes_csv(&nodes, &csv_path)?;
        }
    }
    write_nodes_csv(&nodes, &csv_path)?;
    let deepest = (0..nodes.len()).max_by_key(|&i| nodes[i].depth).unwrap();
    let path = full_path(&boot, &nodes, deepest as u32);
    let file = format!("{out_dir}/deepest.m64");
    write_gamepads_to_file(&path, &file)?;
    println!(
        "done: {} nodes, wrote {csv_path}; deepest node {deepest} (depth {}, frame {}) at {:?}, wrote {file}",
        nodes.len(),
        nodes[deepest].depth,
        path.len(),
        nodes[deepest].st.pos
    );
    Ok(())
}
