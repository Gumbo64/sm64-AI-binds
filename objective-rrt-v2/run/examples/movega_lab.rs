//! Policy-seeded GA lab (`OBRRT_MOVE_GA`, search/src/movegenome.rs; v2: the
//! rolling-horizon controller, `OBRRT_RH_*`; pop, gens and N are moot): one
//! GA attempt (the tree's settings: Steps fitness, new operators, scale 3)
//! from a fixed start towards a fixed target, success within 100 of it
//! (or, with `door`, entering a door: Arrival::Door; else Pass, or Stop
//! with `arrival=stop`), for several RNG seeds; prints
//! frames to the target, the frames the attempt spent (and frames/s), the
//! winner's jump starts per 1000 frames and door overshoots (door targets:
//! the winner came within 150 of the door airborne). `OBRRT_CALM`,
//! `OBRRT_MOVE_GA_EXTRA` etc. come from the environment as in the tree.
//!
//! cargo run --release --example movega_lab -- <rom> <start> X,Y,Z N [seeds=8] [pop=16] [gens=5] [door] [pre=X,Y,Z,N] [prestick=X,Z,N] [health=W]
//!
//! `start` as in moves_lab.rs (`boot`, `tree:<run_dir>:<node>`), or
//! `raw:<file>` (a headerless input prefix from power-on, ref/starts/), or
//! `tas:<file.m64>:<N>` (an m64's first N frames); `N` is
//! the GA horizon. The arm comes from the environment: `OBRRT_MOVES=1`
//! (library steering) and/or `OBRRT_MOVE_GA=1` (policy-seeded rollouts;
//! without `OBRRT_MOVES` the library offers only its policies).
//! tools/movega_lab_cases.sh runs moves_lab's cases in both arms.

use anyhow::{Result, bail};
use obrrt_core::buttons::{A_BUTTON, START_BUTTON};
use obrrt_core::{Eta, GamePad, Mem, Objective, St, check_layout, dist, read_state};
use obrrt_objectives::moves;
use obrrt_search::ga::{Fitness, GaParams, Worker};
use obrrt_search::route::read_tree_bin;
use obrrt_search::steer::steer;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use sm64_binds::{SM64Game, SM64GameGenerator};

struct Reach {
    target: [f32; 3],
    door: bool,
    arrival: obrrt_core::Arrival,
    /// v1's GA budget; moot in v2 (`OBRRT_RH_*`).
    #[allow(dead_code)]
    pop: usize,
    #[allow(dead_code)]
    gens: usize,
    #[allow(dead_code)]
    horizon: usize,
}

const ACT_PULLING_DOOR: u32 = 0x1320;
const ACT_PUSHING_DOOR: u32 = 0x1321;

impl Reach {
    fn at(&self, s: &St) -> bool {
        if self.door { matches!(s.action, ACT_PULLING_DOOR | ACT_PUSHING_DOOR) } else { dist(s.pos, self.target) < 100.0 }
    }
}

impl Objective for Reach {
    fn kind(&self) -> &'static str {
        "MoveGa::Lab"
    }
    fn key(&self) -> String {
        format!("MoveGaLab {:?}", self.target)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        0.0
    }
    fn base_criteria(&self, _s: &St) -> bool {
        false
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if self.at(s) { Eta::Done } else { Eta::Frames(obrrt_core::frames_for(dist(s.pos, self.target))) }
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.target)
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        self.arrival
    }
}

/// run/src/main.rs's fixed boot.
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

/// As moves_lab.rs.
fn start(game: &mut SM64Game, spec: &str) -> Result<u32> {
    let mut frame = 0u32;
    if spec == "boot" {
        for p in boot_inputs() {
            game.step_game(p)?;
            frame += 1;
        }
        let mut st = read_state(game, frame, false);
        let mut still = 0;
        while still < 10 || !st.controllable() {
            let before = st.pos;
            game.step_game(GamePad::new(0, 0, 0))?;
            frame += 1;
            st = read_state(game, frame, false);
            still = if st.pos == before { still + 1 } else { 0 };
        }
    } else if let Some(rest) = spec.strip_prefix("tree:") {
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
        for r in chain.iter().rev() {
            for p in &records[*r].1 {
                game.step_game(*p)?;
                frame += 1;
            }
        }
    } else if let Some(path) = spec.strip_prefix("raw:") {
        // A headerless input prefix from power-on (ref/starts/*.m64).
        let bytes = std::fs::read(path)?;
        for c in bytes.chunks_exact(4) {
            game.step_game(GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8))?;
            frame += 1;
        }
    } else if let Some(rest) = spec.strip_prefix("tas:") {
        // The first N frames of an m64 (0x400-byte header).
        let (path, n) = rest.rsplit_once(':').ok_or_else(|| anyhow::anyhow!("tas:<file.m64>:<N>"))?;
        let n: usize = n.parse()?;
        let bytes = std::fs::read(path)?;
        for c in bytes[0x400..].chunks_exact(4).take(n) {
            game.step_game(GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8))?;
            frame += 1;
        }
    } else {
        bail!("start: boot, tree:<run_dir>:<node>, raw:<inputs.m64> or tas:<file.m64>:<N>");
    }
    check_layout(game)?;
    Ok(frame)
}

struct Fixed([f32; 3]);
impl Objective for Fixed {
    fn kind(&self) -> &'static str {
        "MoveGa::Pre"
    }
    fn key(&self) -> String {
        String::new()
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        0.0
    }
    fn base_criteria(&self, _s: &St) -> bool {
        false
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        Eta::Frames(obrrt_core::frames_for(dist(s.pos, self.0)))
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.0)
    }
}

fn opt<T: std::str::FromStr>(args: &[String], k: &str, d: T) -> T {
    args.iter().find_map(|a| a.strip_prefix(k).and_then(|v| v.parse().ok())).unwrap_or(d)
}

fn main() -> Result<()> {
    // The steering controller (objectives/src/moves; move library off).
    obrrt_objectives::moves::install_controller();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        bail!("usage: movega_lab <rom> <start> X,Y,Z N [seeds=8] [pop=16] [gens=5] [door] [pre=..] [prestick=..] [health=W]");
    }
    let on = |k: &str| std::env::var(k).is_ok_and(|v| v == "1");
    obrrt_core::geo::want_static_floors();
    let gen_ = SM64GameGenerator::from_file(&args[1])?;
    let mut game = gen_.create_game()?;
    let mut frame0 = start(&mut game, &args[2])?;
    // Set-up steering: the generic controller (the mover is registered
    // only after it, as in moves_lab).
    for a in &args {
        if let Some(v) = a.strip_prefix("pre=") {
            let v: Vec<f32> = v.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?;
            let obj = Fixed([v[0], v[1], v[2]]);
            for k in 0..v[3] as usize {
                let s = read_state(&game, frame0, false);
                if dist(s.pos, obj.0) < 100.0 {
                    break;
                }
                game.step_game(steer(&obj, &s, k, 8))?;
                frame0 += 1;
            }
        }
        if let Some(v) = a.strip_prefix("prestick=") {
            let v: Vec<f32> = v.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?;
            for _ in 0..v[2] as usize {
                let st = read_state(&game, frame0, false);
                let (sx, sy) = moves::camera_stick(&st, [v[0], st.pos[1], v[1]]);
                game.step_game(GamePad::new(0, sx, sy))?;
                frame0 += 1;
            }
        }
        if let Some(v) = a.strip_prefix("health=") {
            let w: i16 = v.parse()?;
            let at = obrrt_core::layout::ADDR_gMarioStates as u64 + obrrt_core::layout::MARIO_health as u64;
            let (store, _, memory) = game.wasm_parts();
            memory.view(store).write(at, &((w << 8) | 0x80).to_le_bytes())?;
        }
    }
    let s0 = read_state(&game, frame0, true);
    if on("OBRRT_MOVES") {
        moves::install();
    } else if on("OBRRT_MOVE_GA") {
        moves::install_policies();
    }
    moves::observe(&s0);
    obrrt_core::mover::publish();
    let v: Vec<f32> = args[3].split(',').map(|x| x.parse()).collect::<Result<_, _>>()?;
    let target = [v[0], v[1], v[2]];
    let horizon: usize = args[4].parse()?;
    let door = args.iter().any(|a| a == "door");
    let arrival = if door {
        obrrt_core::Arrival::Door
    } else if args.iter().any(|a| a == "arrival=stop") {
        obrrt_core::Arrival::Stop
    } else {
        obrrt_core::Arrival::Pass
    };
    let obj = Reach { target, door, arrival, pop: opt(&args, "pop=", 16), gens: opt(&args, "gens=", 5), horizon };
    if let Some(m) = obrrt_core::mover::mover() {
        let v: Vec<String> = (0..m.policies()).map(|p| format!("{}{}", m.policy_name(p), if m.policy_plausible(p, &s0, target, arrival) { "" } else { "(pruned)" })).collect();
        println!("policies: {}; ceil {:.0}", v.join(" "), s0.ceil_y);
    }
    let seeds: u64 = opt(&args, "seeds=", 8);
    println!(
        "start frame {frame0} pos {:.0?} fv {:.1} action {:#010x}; target {target:.0?} ({:.0} away), horizon {horizon}, MOVES {} MOVE_GA {}",
        s0.pos,
        s0.forward_vel,
        s0.action,
        dist(s0.pos, target),
        on("OBRRT_MOVES"),
        on("OBRRT_MOVE_GA")
    );
    let save = game.save_state();
    let mut w = Worker::new(&gen_)?;
    let params = GaParams { scale: 3.0, fitness: Fitness::Steps, new_ops: true, pink: 0.0, per_objective_budget: false };
    let (mut hits, mut sum_t, mut sum_f) = (0, 0u64, 0u64);
    let mut ts = Vec::new();
    let (mut secs, mut sum_jumps, mut sum_win, mut overshoots) = (0.0f64, 0u64, 0u64, 0u64);
    for seed in 0..seeds {
        let mut rng = SmallRng::seed_from_u64(seed);
        let clock = std::time::Instant::now();
        let at = w.attempt(&[&obj], &s0, &save, &mut rng, params)?;
        secs += clock.elapsed().as_secs_f64();
        let seg = &at.segments[0];
        let t = seg.success.then_some(seg.inputs.len());
        let ga = at.ga.as_ref();
        // The winner replayed: jump starts from the ground, and (door
        // targets) whether it came within 150 of the door airborne.
        game.load_state(&save)?;
        let (mut prev, mut jumps, mut over) = (s0.clone(), 0, false);
        for (k, p) in seg.inputs.iter().enumerate() {
            game.step_game(*p)?;
            let cur = read_state(&game, frame0 + k as u32 + 1, false);
            jumps += obrrt_objectives::moves::policies::is_jump_start(&prev, &cur) as u64;
            over |= obj.door && cur.action & 0x800 != 0 && cur.hdist(target) < 150.0;
            prev = cur;
        }
        sum_jumps += jumps;
        sum_win += seg.inputs.len() as u64;
        overshoots += over as u64;
        println!(
            "seed {seed}: {} | attempt frames {} | best progress {:.0} | winner gen {} origin {} | jumps {jumps}{}",
            t.map_or("miss".to_string(), |t| format!("at {t}")),
            at.frames,
            seg.progress,
            ga.map_or(0, |g| g.winner_gen),
            ga.map_or("-", |g| g.winner_origin),
            if over { " | door overshoot" } else { "" }
        );
        sum_f += at.frames;
        if let Some(t) = t {
            hits += 1;
            sum_t += t as u64;
            ts.push(t);
        }
    }
    ts.sort_unstable();
    println!(
        "summary: hits {hits}/{seeds} | mean frames to target {} | median {} | mean attempt frames {} | {:.0} frames/s | jump starts/1000 winner frames {:.1} | door overshoots {overshoots}",
        if hits > 0 { format!("{:.1}", sum_t as f64 / hits as f64) } else { "-".into() },
        ts.get(ts.len() / 2).map_or("-".into(), |t| t.to_string()),
        sum_f / seeds.max(1),
        sum_f as f64 / secs.max(1e-9),
        1000.0 * sum_jumps as f64 / sum_win.max(1) as f64
    );
    if on("OBRRT_MOVE_GA") {
        println!("{}", obrrt_search::movegenome::stats_line());
    }
    Ok(())
}
