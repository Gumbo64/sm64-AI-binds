//! Painting re-entry lab (`OBRRT_PAINTING_FAST`, objectives/src/painting.rs).
//!
//! painting_lab <rom> measure <run_dir> [node]
//!   replay the run's best route (most stars, earliest; or to `node`) frame
//!   by frame; for every star exit prints the first controllable castle
//!   frame after it, the frame Mario touched a painting warp floor (and
//!   which painting), the level-change frame, Mario's distance to the
//!   painting he entered, and the node whose end state is the first
//!   controllable one (a lab start).
//! painting_lab <rom> ga <tree:run_dir:node> [seeds=6] [paint=NAME] [steps=N]
//!   from that state, the generated `Painting::Enter` objective (the painting
//!   `paint`, else the nearest one in Mario's area with stars left): one GA
//!   attempt (the tree's settings) per seed with its budget; prints frames to
//!   the warp floor and to the level change. Then the plain steering
//!   controller (steer(), no GA) on its own for comparison. The switch comes
//!   from the environment (`OBRRT_PAINTING_FAST=1`), as do `OBRRT_MOVES`.
//! painting_lab <rom> export <run_dir> <node> <out>
//!   the inputs from power-on to that node, headerless (an obrrt
//!   `<start.m64>`, e.g. ref/starts/).
//!   `horizon=N` overrides the objective's GA horizon (both arms alike).
//!   `steps=N` first advances N frames with no input; then A on alternate
//!   frames until Mario is controllable (save menu, dialogs).

use anyhow::{Result, bail};
use obrrt_core::layout::LEVEL_CASTLE;
use std::sync::Arc;

use obrrt_core::{Budget, GamePad, Generator, Objective, Progress, St, check_layout, dist, read_state};
use obrrt_objectives::{Registry, moves};
use obrrt_search::ga::{Fitness, GaParams, Worker};
use obrrt_search::route::read_tree_bin;
use obrrt_search::steer::steer;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use sm64_binds::{SM64Game, SM64GameGenerator};

const ACT_DISAPPEARED: u32 = 0x1300;

/// Painting warp floors (surface_terrains.h: SURFACE_PAINTING_WARP_D3 0xD3 ..
/// SURFACE_PAINTING_WARP_FC; initiate_painting_warp).
fn on_warp_floor(s: &St) -> bool {
    (0xD3..=0xFC).contains(&(s.floor_type as i32 & 0xFFFF))
}

fn chain_of(records: &[(i64, Vec<GamePad>)], node: usize) -> Vec<usize> {
    let mut chain = vec![node + 1];
    while let Some(&r) = chain.last() {
        if r == 0 {
            break;
        }
        let parent = records[r].0;
        chain.push(if parent >= 0 { parent as usize + 1 } else { 0 });
    }
    chain.reverse();
    chain
}

fn start(game: &mut SM64Game, spec: &str) -> Result<u32> {
    let Some(rest) = spec.strip_prefix("tree:") else { bail!("start: tree:<run_dir>:<node>") };
    let (dir, node) = rest.rsplit_once(':').ok_or_else(|| anyhow::anyhow!("tree:<run_dir>:<node>"))?;
    let records = read_tree_bin(&format!("{dir}/tree.bin"))?;
    let records: Vec<(i64, Vec<GamePad>)> = records.into_iter().map(|r| (r.0 as i64, r.1)).collect();
    let mut frame = 0u32;
    for r in chain_of(&records, node.parse()?) {
        for p in &records[r].1 {
            game.step_game(*p)?;
            frame += 1;
        }
    }
    check_layout(game)?;
    Ok(frame)
}

/// The best node in nodes.csv: most stars, then earliest frame.
fn best_node(dir: &str) -> Result<usize> {
    let text = std::fs::read_to_string(format!("{dir}/nodes.csv"))?;
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap_or("").split(',').collect();
    let col = |n: &str| head.iter().position(|h| *h == n).unwrap();
    let (ci, cs, cf) = (col("id"), col("stars"), col("frame"));
    let mut best = (0usize, -1i64, 0i64);
    for l in lines {
        let v: Vec<&str> = l.split(',').collect();
        let (id, st, f): (usize, i64, i64) = (v[ci].parse()?, v[cs].parse()?, v[cf].parse()?);
        if st > best.1 || (st == best.1 && f < best.2) {
            best = (id, st, f);
        }
    }
    Ok(best.0)
}

fn painting_at(s: &St) -> Option<&'static obrrt_objectives::Painting> {
    obrrt_objectives::paintings().iter().filter(|p| p.area == s.area).min_by(|a, b| dist(a.centre(), s.pos).total_cmp(&dist(b.centre(), s.pos)))
}

fn measure(gg: &SM64GameGenerator, dir: &str, node: Option<usize>) -> Result<()> {
    let node = match node {
        Some(n) => n,
        None => best_node(dir)?,
    };
    let records = read_tree_bin(&format!("{dir}/tree.bin"))?;
    let records: Vec<(i64, Vec<GamePad>)> = records.into_iter().map(|r| (r.0 as i64, r.1)).collect();
    let mut game = gg.create_game()?;
    let mut frame = 0u32;
    // (frame, state, node whose inputs end at this frame)
    let mut prev = read_state(&game, 0, false);
    let mut exit: Option<(u32, i16)> = None; // star frame, stars
    let mut ctrl: Option<(u32, St, Option<usize>)> = None;
    let mut want_node = false;
    let paths = std::env::var("PATHS").is_ok();
    let mut path: Vec<String> = Vec::new();
    let mut trig: Option<(u32, St, St)> = None;
    println!("run {dir} node {node}");
    println!("star_frame,stars,ctrl_frame,ctrl_node,ctrl_pos,trigger_frame,painting,level_frame,dest_level,ctrl_to_trigger,ctrl_to_level,dist_ctrl,trigger_pos,trigger_action");
    for r in chain_of(&records, node) {
        let n = r as i64 - 1;
        for p in &records[r].1 {
            game.step_game(*p)?;
            frame += 1;
            let s = read_state(&game, frame, false);
            if s.stars > prev.stars && s.level as u32 != LEVEL_CASTLE {
                // stars counter rises in the course
            }
            if s.stars > prev.stars {
                exit = Some((frame, s.stars));
                ctrl = None;
                trig = None;
                path.clear();
            }
            if exit.is_some() && ctrl.is_none() && s.level as u32 == LEVEL_CASTLE && s.controllable() {
                ctrl = Some((frame, s.clone(), None));
                want_node = true;
            }
            if paths && ctrl.is_some() && trig.is_none() && frame % 10 == 0 {
                path.push(format!("{:.0},{:.0},{:.0}", s.pos[0], s.pos[1], s.pos[2]));
            }
            if ctrl.is_some() && trig.is_none() && s.level as u32 == LEVEL_CASTLE && s.action == ACT_DISAPPEARED && on_warp_floor(&prev) | on_warp_floor(&s) {
                trig = Some((frame, s.clone(), prev.clone()));
            }
            if let (Some((sf, stars)), Some((cf, cs, cn)), true) = (exit, &ctrl, prev.level as u32 == LEVEL_CASTLE && s.level as u32 != LEVEL_CASTLE) {
                let (tf, _tp, tprev) = trig.clone().map_or((0, s.clone(), s.clone()), |t| t);
                let pn = painting_at(&tprev).map_or("?", |p| p.name);
                let pd = painting_at(&tprev).map_or(0.0, |p| dist(cs.pos, p.centre()));
                println!(
                    "{sf},{stars},{cf},{},{:.0}/{:.0}/{:.0},{tf},{pn},{},{},{},{},{pd:.0},{:.0}/{:.0}/{:.0},{:#x}",
                    cn.map_or(-1, |x| x as i64),
                    cs.pos[0],
                    cs.pos[1],
                    cs.pos[2],
                    frame,
                    s.level,
                    tf as i64 - *cf as i64,
                    frame - cf,
                    tprev.pos[0],
                    tprev.pos[1],
                    tprev.pos[2],
                    tprev.action
                );
                if paths {
                    println!("  path: {}", path.join(" "));
                }
                path.clear();
                exit = None;
                ctrl = None;
            }
            prev = s;
        }
        if want_node {
            if let Some(c) = ctrl.as_mut() {
                c.2 = Some(n.max(0) as usize);
            }
            want_node = false;
        }
    }
    Ok(())
}

/// The objective with another GA horizon (`horizon=N`): everything else
/// delegated.
struct Hz(Arc<dyn Objective>, usize);

impl Objective for Hz {
    fn kind(&self) -> &'static str {
        self.0.kind()
    }
    fn key(&self) -> String {
        self.0.key()
    }
    fn obj_weight(&self, b: &St) -> f64 {
        self.0.obj_weight(b)
    }
    fn base_criteria(&self, s: &St) -> bool {
        self.0.base_criteria(s)
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        self.0.base_heuristic(s)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        self.0.success(traj)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        self.0.progress(traj)
    }
    fn budget(&self) -> Budget {
        match self.0.budget() {
            Budget::Ga { population, generations, .. } => Budget::Ga { population, generations, horizon: self.1 },
            b => b,
        }
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.0.steer_target(s)
    }
    fn arrival(&self, s: &St) -> obrrt_core::Arrival {
        self.0.arrival(s)
    }
    fn steer_button(&self, s: &St) -> u16 {
        self.0.steer_button(s)
    }
    fn pad_overlay(&self) -> Option<obrrt_core::PadOverlay> {
        self.0.pad_overlay()
    }
}

fn opt<T: std::str::FromStr>(args: &[String], k: &str, d: T) -> T {
    args.iter().find_map(|a| a.strip_prefix(k).and_then(|v| v.parse().ok())).unwrap_or(d)
}

fn ga(gg: &SM64GameGenerator, args: &[String]) -> Result<()> {
    let on = |k: &str| std::env::var(k).is_ok_and(|v| v == "1");
    let mut game = gg.create_game()?;
    let mut frame0 = start(&mut game, &args[3])?;
    for _ in 0..opt(args, "steps=", 0usize) {
        game.step_game(GamePad::new(0, 0, 0))?;
        frame0 += 1;
    }
    // Through any dialog / cutscene to the first controllable frame.
    let mut waited = 0;
    // (A on alternate frames: answers the save menu, Save & Continue, and
    // advances dialogs.)
    while !read_state(&game, frame0, false).controllable() && waited < 600 {
        let a = if waited % 2 == 0 { obrrt_core::buttons::A_BUTTON } else { 0 };
        game.step_game(GamePad::new(a, 0, 0))?;
        frame0 += 1;
        waited += 1;
    }
    let s0 = read_state(&game, frame0, true);
    if on("OBRRT_MOVES") {
        moves::install();
    } else if on("OBRRT_MOVE_GA") {
        moves::install_policies();
    }
    moves::observe(&s0);
    obrrt_core::mover::publish();
    let mut rng = SmallRng::seed_from_u64(1);
    let objs: Vec<_> = Registry.generate(&s0, &mut rng).into_iter().filter(|o| o.kind() == "Painting::Enter").collect();
    let want: Option<String> = args.iter().find_map(|a| a.strip_prefix("paint=").map(|v| format!("Painting {v}")));
    let obj = match &want {
        Some(k) => objs.iter().find(|o| o.key() == *k || o.key().starts_with(&format!("{k} "))),
        None => objs.iter().filter(|o| o.base_criteria(&s0)).min_by(|a, b| a.base_heuristic(&s0).total_cmp(&b.base_heuristic(&s0))),
    };
    let Some(obj) = obj.cloned() else { bail!("no Painting::Enter at {:?}", s0.pos) };
    let obj: Arc<dyn Objective> = match opt(args, "horizon=", 0usize) {
        0 => obj,
        h => Arc::new(Hz(obj, h)),
    };
    println!(
        "start {} frame {frame0} (waited {waited}) pos {:.0?} yaw {:#06x} action {:#010x} | {} eligible {} weight {:.1} target {:?} arrival {:?} budget {:?} | FAST {}",
        args[3],
        s0.pos,
        s0.yaw as u16,
        s0.action,
        obj.key(),
        obj.base_criteria(&s0),
        obj.obj_weight(&s0),
        obj.steer_target(&s0).map(|t| [t[0] as i32, t[1] as i32, t[2] as i32]),
        obj.arrival(&s0),
        obj.budget(),
        on("OBRRT_PAINTING_FAST")
    );
    let save = game.save_state();
    let mut w = Worker { base: gg.create_game()?, tmp: gg.create_game()? };
    let params = GaParams { scale: 3.0, fitness: Fitness::Steps, new_ops: true, pink: 0.0, per_objective_budget: false };
    let seeds: u64 = opt(args, "seeds=", 6);
    let mut ts = Vec::new();
    let mut spent = 0u64;
    for seed in 0..seeds {
        let mut rng = SmallRng::seed_from_u64(seed);
        let at = w.attempt(&[obj.as_ref()], &s0, &save, &mut rng, params)?;
        let seg = &at.segments[0];
        // Frames to the warp floor: replay the winner.
        game.load_state(&save)?;
        let mut trig = None;
        for (k, p) in seg.inputs.iter().enumerate() {
            game.step_game(*p)?;
            let s = read_state(&game, frame0 + k as u32 + 1, false);
            if trig.is_none() && s.action == ACT_DISAPPEARED {
                trig = Some(k + 1);
            }
        }
        spent += at.frames;
        println!(
            "seed {seed}: {} | trigger {} | attempt frames {} | winner gen {} origin {}",
            if seg.success { format!("level at {}", seg.inputs.len()) } else { format!("miss (progress {:.0})", seg.progress) },
            trig.map_or("-".into(), |t| t.to_string()),
            at.frames,
            at.ga.as_ref().map_or(0, |g| g.winner_gen),
            at.ga.as_ref().map_or("-", |g| g.winner_origin)
        );
        if seg.success {
            ts.push(seg.inputs.len());
        }
    }
    ts.sort_unstable();
    // The steering controller alone.
    game.load_state(&save)?;
    let mut s = s0.clone();
    let mut steer_level = None;
    let mut steer_trig = None;
    for k in 0..400usize {
        let pad = steer(obj.as_ref(), &s, k, 16);
        game.step_game(pad)?;
        s = read_state(&game, frame0 + k as u32 + 1, false);
        if steer_trig.is_none() && s.action == ACT_DISAPPEARED {
            steer_trig = Some(k + 1);
        }
        if s.level as u32 != LEVEL_CASTLE {
            steer_level = Some(k + 1);
            break;
        }
    }
    println!(
        "summary: hits {}/{seeds} | median level frames {} | best {} | mean {} | mean attempt frames {} | steer-only trigger {} level {}",
        ts.len(),
        ts.get(ts.len() / 2).map_or("-".into(), |t| t.to_string()),
        ts.first().map_or("-".into(), |t| t.to_string()),
        if ts.is_empty() { "-".into() } else { format!("{:.1}", ts.iter().sum::<usize>() as f64 / ts.len() as f64) },
        spent / seeds.max(1),
        steer_trig.map_or("-".into(), |t| t.to_string()),
        steer_level.map_or("-".into(), |t| t.to_string()),
    );
    Ok(())
}

/// `export <run_dir> <node> <out>`: the inputs from power-on to the node,
/// headerless (4 bytes a frame, as obrrt's `<start.m64>` argument reads).
fn export(args: &[String]) -> Result<()> {
    let records = read_tree_bin(&format!("{}/tree.bin", args[3]))?;
    let records: Vec<(i64, Vec<GamePad>)> = records.into_iter().map(|r| (r.0 as i64, r.1)).collect();
    let mut out = Vec::new();
    for r in chain_of(&records, args[4].parse()?) {
        for p in &records[r].1 {
            out.extend_from_slice(&p.button.to_be_bytes());
            out.push(p.stick_x as u8);
            out.push(p.stick_y as u8);
        }
    }
    std::fs::write(&args[5], &out)?;
    println!("{} frames -> {}", out.len() / 4, args[5]);
    Ok(())
}

fn main() -> Result<()> {
    obrrt_objectives::moves::install_controller();
    obrrt_core::geo::want_static_floors();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        bail!("usage: painting_lab <rom> measure <run_dir> [node] | ga <tree:run:node> [seeds=6] [paint=NAME]");
    }
    let gg = SM64GameGenerator::from_file(&args[1])?;
    match args[2].as_str() {
        "measure" => measure(&gg, &args[3], args.get(4).and_then(|v| v.parse().ok())),
        "ga" => ga(&gg, &args),
        "export" => export(&args),
        m => bail!("unknown mode {m}"),
    }
}
