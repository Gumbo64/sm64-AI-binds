//! Swimming lab: steers with search/src/steer.rs from a start state to a
//! target (normally underwater) and logs each frame's inputs, height, pitch
//! and action, then frames to get within 150 units and the deepest point.
//!
//! cargo run --release --example swim_lab -- <rom> <start> X,Y,Z N [trace]
//!
//! start: `tree:<run_dir>:<node>` replays a run's tree.bin to that node.
//! `X,Y,Z` with a leading `+` is relative to the start. The controller's
//! switches (OBRRT_SWIM_HOLD) are read from the environment.
//!
//! `swim_lab <rom> <start> sample K N [seed]`: K points from
//! `mario::underwater_point` (OBRRT_SWIM_TARGETS=1's PosDist targets), each
//! steered to for N frames from the start; reports how many are reached
//! within 90 frames (PosDist's horizon) and N, and the depths.

use anyhow::{Result, bail};
use obrrt_core::{Objective, St, check_layout, dist, read_state};
use obrrt_search::route::read_tree_bin;
use obrrt_search::steer::steer;
use sm64_binds::{SM64Game, SM64GameGenerator};

struct Point([f32; 3]);

impl Objective for Point {
    fn kind(&self) -> &'static str {
        "Swim::Lab"
    }
    fn key(&self) -> String {
        format!("SwimLab {:?}", self.0)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        0.0
    }
    fn base_criteria(&self, _s: &St) -> bool {
        false
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut obrrt_core::Mem) -> obrrt_core::Eta {
        obrrt_core::Eta::Frames(obrrt_core::frames_for(s.dist(self.0)))
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.0)
    }
}

fn start(game: &mut SM64Game, spec: &str) -> Result<u32> {
    let Some(rest) = spec.strip_prefix("tree:") else { bail!("start: tree:<run_dir>:<node>") };
    let (dir, node) = rest.rsplit_once(':').ok_or_else(|| anyhow::anyhow!("tree:<run_dir>:<node>"))?;
    let node: usize = node.parse()?;
    let records = read_tree_bin(&format!("{dir}/tree.bin"))?;
    let mut chain = vec![node + 1];
    while let Some(&r) = chain.last() {
        if r == 0 {
            break;
        }
        let parent = records[r].0;
        chain.push(if parent >= 0 { parent as usize + 1 } else { 0 });
    }
    let mut frame = 0u32;
    for r in chain.iter().rev() {
        for p in &records[*r].1 {
            game.step_game(*p)?;
            frame += 1;
        }
    }
    check_layout(game)?;
    Ok(frame)
}

fn main() -> Result<()> {
    // The steering controller (objectives/src/moves; move library off).
    obrrt_objectives::moves::install_controller();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        bail!("usage: swim_lab <rom> <start> X,Y,Z N [trace]");
    }
    let gen_ = SM64GameGenerator::from_file(&args[1])?;
    let mut game = gen_.create_game()?;
    let mut frame = start(&mut game, &args[2])?;
    if args[3] == "sample" {
        return sample(&mut game, frame, &args);
    }
    let s0 = read_state(&game, frame, false);
    let rel = args[3].starts_with('+');
    let v: Vec<f32> = args[3].trim_start_matches('+').split(',').map(|x| x.parse()).collect::<Result<_, _>>()?;
    let target = if rel { [s0.pos[0] + v[0], s0.pos[1] + v[1], s0.pos[2] + v[2]] } else { [v[0], v[1], v[2]] };
    let n: usize = args[4].parse()?;
    let trace = args.iter().any(|a| a == "trace");
    println!(
        "start: frame {frame} level {} area {} pos {:.0?} action {:#010x} water_level {} pitch {} target {:.0?} d {:.0}",
        s0.level, s0.area, s0.pos, s0.action, s0.water_level, s0.pitch, target, dist(s0.pos, target)
    );
    let obj = Point(target);
    let mut s = s0.clone();
    let (mut closest, mut min_y, mut within) = ((dist(s.pos, target), 0), (s.pos[1], 0), None);
    let mut submerged = None;
    for k in 0..n {
        let pad = steer(&obj, &s, k, 8);
        game.step_game(pad)?;
        frame += 1;
        s = read_state(&game, frame, false);
        let d = dist(s.pos, target);
        if trace {
            println!(
                "{:4} A{} {:4} {:4} | pos {:6.0} {:6.0} {:6.0} depth {:5.0} pitch {:6} yaw {:6} fv {:5.1} act {:#010x} t {:3} d {:5.0}",
                k + 1,
                (pad.button & obrrt_core::buttons::A_BUTTON != 0) as u8,
                pad.stick_x,
                pad.stick_y,
                s.pos[0],
                s.pos[1],
                s.pos[2],
                s.water_level as f32 - s.pos[1],
                s.pitch,
                s.yaw,
                s.forward_vel,
                s.action,
                s.action_timer,
                d
            );
        }
        if d < closest.0 {
            closest = (d, k + 1);
        }
        if s.pos[1] < min_y.0 {
            min_y = (s.pos[1], k + 1);
        }
        if submerged.is_none() && (s.water_level as f32 - s.pos[1]) > 200.0 {
            submerged = Some(k + 1);
        }
        if d < 150.0 {
            within = Some(k + 1);
            break;
        }
    }
    println!(
        "result: within150 {:?} closest {:.0} at {} min_y {:.0} at {} (depth {:.0}) submerged(>200) {:?} end {:.0?} act {:#010x}",
        within,
        closest.0,
        closest.1,
        min_y.0,
        min_y.1,
        s0.water_level as f32 - min_y.0,
        submerged,
        s.pos,
        s.action
    );
    Ok(())
}

fn sample(game: &mut SM64Game, frame0: u32, args: &[String]) -> Result<()> {
    use rand::SeedableRng;
    let k: usize = args[4].parse()?;
    let n: usize = args.get(5).map_or(Ok(300), |v| v.parse())?;
    let seed: u64 = args.get(6).map_or(Ok(1), |v| v.parse())?;
    if !obrrt_objectives::mario::swim_targets() {
        bail!("sample needs OBRRT_SWIM_TARGETS=1");
    }
    // With objects: snapshots the zone's static floors.
    let s0 = read_state(game, frame0, true);
    let wl = s0.water_level as f32;
    println!("start: level {} area {} pos {:.0?} action {:#010x} water_level {wl}", s0.level, s0.area, s0.pos, s0.action);
    println!("static floors known: {}", obrrt_core::geo::static_floors(s0.zone()).is_some());
    let save = game.save_state();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let (mut tried, mut in90, mut in_n, mut frames) = (0, 0, 0, Vec::new());
    for _ in 0..k * 4 {
        if tried == k {
            break;
        }
        let Some(t) = obrrt_objectives::mario::underwater_point(&s0, &mut rng) else { continue };
        tried += 1;
        game.load_state(&save)?;
        let obj = Point(t);
        let mut s = s0.clone();
        let mut frame = frame0;
        let (mut hit, mut min_y, mut closest) = (None, s.pos[1], dist(s.pos, t));
        for f in 0..n {
            game.step_game(steer(&obj, &s, f, 8))?;
            frame += 1;
            s = read_state(game, frame, false);
            min_y = min_y.min(s.pos[1]);
            closest = closest.min(dist(s.pos, t));
            if dist(s.pos, t) < 150.0 {
                hit = Some(f + 1);
                break;
            }
        }
        if hit.is_some_and(|h| h <= 90) {
            in90 += 1;
        }
        if let Some(h) = hit {
            in_n += 1;
            frames.push(h);
        }
        println!(
            "target {:.0?} depth {:5.0} d0 {:5.0}: reached {:?} closest {:4.0} max depth {:5.0}",
            t,
            wl - t[1],
            dist(s0.pos, t),
            hit,
            closest,
            wl - min_y
        );
    }
    frames.sort();
    println!(
        "summary: {tried} targets, reached <=90f {in90}, <={n}f {in_n}, median frames {:?}",
        frames.get(frames.len() / 2)
    );
    Ok(())
}
