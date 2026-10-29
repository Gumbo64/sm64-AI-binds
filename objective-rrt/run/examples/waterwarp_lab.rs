//! Water-warp lab (objectives/src/extra/tech/waterwarp.rs, CATALOGUE G12):
//! plays a movie from power-on, snapshots every zone it visits (static
//! floors / ceilings / walls and the water boxes, from memory, the first
//! frame Mario is in control on a static floor of the area) and lists the
//! zone's warp spots; then reports every one-frame warp the movie makes
//! (waterwarp::warp, the objective's success test), whether the mechanism
//! explains it (Plunge: y = waterLevel - 100 from below it; downwarp: y =
//! waterLevel - 80 or ceiling - 160), and the nearest spot of its kind.
//! Also every other same-zone one-frame |dy| >= 300 (what the test leaves
//! out), to check for misses.
//!
//! waterwarp_lab <rom> <movie.m64|raw> [to=N] [top=K] [zones=L:A,...] [quiet]
//!   (scan; a zone whose water level moves, WDW, gets its spots again)
//! waterwarp_lab <rom> ga <start> [seeds=6] [key=SUBSTR] [steps=N]
//!   start = tree:<run_dir>:<node> | <movie|raw>[@N]: GA attempts (the
//!   tree's GA settings, each objective's budget) with every WaterWarp
//!   objective offered there; success, frames to the warp, frames spent
//! waterwarp_lab <rom> cut <movie> <N> <out> | export <run_dir> <node> <out>
//!   headerless start inputs (obrrt `<start.m64>`, OBRRT_BOOT_NOWAIT=1)
//!
//! Summary at the end: warps, rejected jumps, warps near a spot of their
//! kind (hit rate), and per zone how many spots a movie warp used.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use anyhow::{Result, bail};
use obrrt_core::geo::{EnvBox, Tri};
use obrrt_core::{GamePad, St, read_state};
use obrrt_objectives::extra::tech::waterwarp::{self, Kind, Spot};
use sm64_binds::{SM64Game, SM64GameGenerator};

fn arg<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter().find_map(|a| a.strip_prefix(k).and_then(|r| r.strip_prefix('=')))
}

fn movie(path: &str) -> Result<Vec<GamePad>> {
    let bytes = std::fs::read(path)?;
    let body = if bytes.len() >= 0x400 && &bytes[..4] == b"M64\x1a" {
        let n = u32::from_le_bytes(bytes[0x18..0x1C].try_into().unwrap()) as usize;
        let d = &bytes[0x400..];
        d[..(n * 4).min(d.len() / 4 * 4)].to_vec()
    } else {
        bytes[..bytes.len() / 4 * 4].to_vec()
    };
    Ok(body.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect())
}

fn rd_u32(m: &[u8], a: usize) -> u32 {
    u32::from_le_bytes(m[a..a + 4].try_into().unwrap())
}

type Snap = (Arc<[Tri]>, Arc<[Tri]>, Arc<[Tri]>, Vec<EnvBox>);

/// As geo::observe_floors: only when Mario's floor is a static one of the
/// partition read.
fn snapshot(g: &SM64Game) -> Option<Snap> {
    g.with_memory(|m| {
        let mario = obrrt_core::layout::ADDR_gMarioStates as usize;
        let floor = rd_u32(m, mario + 0x68);
        if floor == 0 || rd_u32(m, floor as usize + 0x2C) != 0 {
            return None;
        }
        let (tris, seen) = obrrt_core::geo::read_floors(m);
        seen.contains(&floor).then(|| {
            (tris.into(), obrrt_core::geo::read_ceils(m).into(), obrrt_core::geo::read_walls(m).into(), obrrt_core::geo::read_water(m))
        })
    })
}

fn name(z: (u16, u8)) -> String {
    let n = match z.0 {
        4 => "BBH",
        5 => "CCM",
        6 => "castle",
        7 => "HMC",
        8 => "SSL",
        9 => "BoB",
        10 => "SL",
        11 => "WDW",
        12 => "JRB",
        13 => "THI",
        14 => "TTC",
        15 => "RR",
        16 => "grounds",
        17 => "BitDW",
        18 => "VCUtM",
        19 => "BitFS",
        20 => "SA",
        21 => "BitS",
        22 => "LLL",
        23 => "DDD",
        24 => "WF",
        26 => "courtyard",
        27 => "PSS",
        28 => "CotMC",
        29 => "TotWC",
        31 => "WMotR",
        36 => "TTM",
        _ => "?",
    };
    format!("{n} {}:{}", z.0, z.1)
}

fn fmt(p: [f32; 3]) -> String {
    format!("({:.0}, {:.0}, {:.0})", p[0], p[1], p[2])
}

fn hd(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).hypot(a[2] - b[2])
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: waterwarp_lab <rom> <movie.m64|raw> [to=N] [top=K] [zones=L:A,...] [quiet] | ga <start> ... | cut <movie> <N> <out> | export <run_dir> <node> <out>");
    }
    match args[2].as_str() {
        "ga" => return ga(&args),
        "cut" => return cut(&args),
        "export" => return export(&args),
        _ => {}
    }
    scan(&args)
}

fn scan(args: &[String]) -> Result<()> {
    let args = args.to_vec();
    let pads = movie(&args[2])?;
    let to: usize = arg(&args, "to").map_or(pads.len(), |v| v.parse::<usize>().unwrap().min(pads.len()));
    let top: usize = arg(&args, "top").map_or(8, |v| v.parse().unwrap());
    let only: Option<Vec<(u16, u8)>> = arg(&args, "zones").map(|v| {
        v.split(',')
            .map(|z| {
                let (l, a) = z.split_once(':').unwrap();
                (l.parse().unwrap(), a.parse().unwrap())
            })
            .collect()
    });
    let quiet = args.iter().any(|a| a == "quiet");

    let gg = SM64GameGenerator::from_file(&args[1])?;
    let mut game = gg.create_game()?;
    let mut spots: BTreeMap<(u16, u8), Vec<Spot>> = BTreeMap::new();
    let mut boxes_of: HashMap<(u16, u8), Vec<EnvBox>> = HashMap::new();
    let mut geo_of: HashMap<(u16, u8), (Arc<[Tri]>, Arc<[Tri]>, Arc<[Tri]>)> = HashMap::new();
    let mut prev: Option<St> = None;
    let mut cur_held: Option<St> = None;
    // (frame, kind, prev, cur, nearest spot dist)
    let mut events: Vec<(usize, Kind, St, St, Option<f32>)> = Vec::new();
    let mut rejected: Vec<(usize, St, St)> = Vec::new();
    let mut others: Vec<(usize, St, St)> = Vec::new();
    for (k, p) in pads[..to].iter().enumerate() {
        game.step_game(*p)?;
        let f = k + 1;
        let s = read_state(&game, f as u32, false);
        let z = s.zone();
        if s.controllable() && !spots.contains_key(&z) && only.as_ref().is_none_or(|o| o.contains(&z)) {
            if let Some((fl, ce, wa, bx)) = snapshot(&game) {
                let t0 = std::time::Instant::now();
                let v = waterwarp::spots(fl.clone(), ce.clone(), wa.clone(), &bx);
                let water = bx.iter().filter(|b| b.is_water()).count();
                println!(
                    "zone {} (f{f}): {} floors, {} ceilings, {} walls, {} boxes ({} water); {} spots ({} plunge, {} clamp, {} ceiling) in {:.0} ms",
                    name(z),
                    fl.len(),
                    ce.len(),
                    wa.len(),
                    bx.len(),
                    water,
                    v.len(),
                    v.iter().filter(|s| s.kind == Kind::Plunge).count(),
                    v.iter().filter(|s| s.kind == Kind::Clamp).count(),
                    v.iter().filter(|s| s.kind == Kind::Ceiling).count(),
                    t0.elapsed().as_secs_f64() * 1e3
                );
                for (i, b) in bx.iter().enumerate() {
                    println!("  box {i}: id {} x {:.0}..{:.0} z {:.0}..{:.0} y {:.0}", b.id, b.lo[0], b.hi[0], b.lo[1], b.hi[1], b.y);
                }
                if !quiet {
                    let mut by = v.clone();
                    by.sort_by(|a, b| b.gain.abs().total_cmp(&a.gain.abs()));
                    for kind in [Kind::Plunge, Kind::Clamp, Kind::Ceiling] {
                        for s in by.iter().filter(|s| s.kind == kind).take(top) {
                            println!(
                                "  {:?} box {} from {} -> end {} gain {:+.0} step {:.0}{}",
                                s.kind,
                                s.water_box,
                                fmt(s.from),
                                fmt(s.end),
                                s.gain,
                                s.step,
                                if s.fall { " (fall)" } else { "" }
                            );
                        }
                    }
                }
                boxes_of.insert(z, bx);
                geo_of.insert(z, (fl, ce, wa));
                spots.insert(z, v);
            }
        } else if let Some(old) = boxes_of.get(&z)
            && s.controllable()
        {
            // The level's water moved 50+ (WDW; it also bobs a unit a frame):
            // new spots from the same geometry.
            let now = game.with_memory(obrrt_core::geo::read_water);
            if now.len() == old.len() && now.iter().zip(old).any(|(a, b)| (a.y - b.y).abs() >= 50.0) {
                let (fl, ce, wa) = geo_of[&z].clone();
                let v = waterwarp::spots(fl, ce, wa, &now);
                println!(
                    "zone {} (f{f}): water levels {:?} -> {:?}: {} spots ({} plunge, {} clamp, {} ceiling)",
                    name(z),
                    old.iter().map(|b| b.y as i32).collect::<Vec<_>>(),
                    now.iter().map(|b| b.y as i32).collect::<Vec<_>>(),
                    v.len(),
                    v.iter().filter(|s| s.kind == Kind::Plunge).count(),
                    v.iter().filter(|s| s.kind == Kind::Clamp).count(),
                    v.iter().filter(|s| s.kind == Kind::Ceiling).count()
                );
                boxes_of.insert(z, now);
                spots.insert(z, v);
            }
        }
        // The transition pv -> cur is judged with s as the next frame.
        if prev.is_some() && cur_held.is_some() {
            let (pv, cur) = (prev.take().unwrap(), cur_held.take().unwrap());
            let f = f - 1;
            let z = cur.zone();
            let dy = cur.pos[1] - pv.pos[1];
            let wet = pv.action & 0x1C0 == 0x0C0 || cur.action & 0x1C0 == 0x0C0;
            let raw = pv.zone() == z && dy.abs() >= waterwarp::GAIN && pv.alive() && cur.alive() && wet;
            if let Some(kind) = waterwarp::warp(&pv, &cur, Some(&s)) {
                let near = spots.get(&z).and_then(|v| {
                    v.iter()
                        .filter(|sp| match kind {
                            Kind::Plunge => sp.kind == Kind::Plunge,
                            _ => sp.kind != Kind::Plunge,
                        })
                        .map(|sp| hd(sp.from, pv.pos))
                        .min_by(f32::total_cmp)
                });
                println!(
                    "f{f} {} WARP {:?}: {} -> {} dy {:+.0} act {:#x} -> {:#x} water {} (next {}) next ceil {:.0} hs {:.1} vy {:.1}; nearest spot {}",
                    name(z),
                    kind,
                    fmt(pv.pos),
                    fmt(cur.pos),
                    dy,
                    pv.action,
                    cur.action,
                    cur.water_level,
                    s.water_level,
                    s.ceil_y,
                    pv.vel[0].hypot(pv.vel[2]),
                    cur.vel[1],
                    near.map_or("none (no spot of this kind in the zone)".to_string(), |d| format!("{d:.0}"))
                );
                events.push((f, kind, pv.clone(), cur.clone(), near));
            } else if raw {
                rejected.push((f, pv.clone(), cur.clone()));
            } else if pv.zone() == z && dy.abs() >= waterwarp::GAIN && pv.alive() && cur.alive() {
                others.push((f, pv.clone(), cur.clone()));
            }
            prev = Some(cur);
        } else if let Some(c) = cur_held.take() {
            prev = Some(c);
        }
        cur_held = Some(s);
    }
    println!("\nrejected: same-zone one-frame |dy| >= {} in or into water that isn't a warp (warp() false):", waterwarp::GAIN);
    for (f, pv, s) in &rejected {
        println!(
            "  f{f} {}: {} -> {} act {:#x} -> {:#x} water {} vy {:.1} hs {:.0}",
            name(s.zone()),
            fmt(pv.pos),
            fmt(s.pos),
            pv.action,
            s.action,
            s.water_level,
            s.vel[1],
            pv.vel[0].hypot(pv.vel[2])
        );
    }
    println!("\nother one-frame |dy| >= {} in a zone (not counted as water warps):", waterwarp::GAIN);
    for (f, pv, s) in &others {
        println!(
            "  f{f} {}: {} -> {} act {:#x} -> {:#x} water {} ctl {}/{}",
            name(s.zone()),
            fmt(pv.pos),
            fmt(s.pos),
            pv.action,
            s.action,
            s.water_level,
            pv.controllable(),
            s.controllable()
        );
    }
    let n = events.len();
    let hit = events.iter().filter(|e| e.4.is_some_and(|d| d < waterwarp::NEAR)).count();
    println!(
        "\nsummary: {n} warps ({} plunge, {} clamp, {} ceiling), {} rejected wet |dy| jumps; {hit}/{n} within {} of a spot of their kind",
        events.iter().filter(|e| e.1 == Kind::Plunge).count(),
        events.iter().filter(|e| e.1 == Kind::Clamp).count(),
        events.iter().filter(|e| e.1 == Kind::Ceiling).count(),
        rejected.len(),
        waterwarp::NEAR
    );
    let total: usize = spots.values().map(|v| v.len()).sum();
    println!("zones snapshotted: {}; spots in all: {total}", spots.len());
    for (z, v) in &spots {
        if v.is_empty() {
            continue;
        }
        let used = v.iter().filter(|sp| events.iter().any(|e| e.2.zone() == *z && hd(sp.from, e.2.pos) < waterwarp::NEAR)).count();
        println!("  {}: {} spots, {used} within {} of a movie warp", name(*z), v.len(), waterwarp::NEAR);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// GA attempts with the objective from a start.

/// Plays a start: `tree:<run_dir>:<node>`, or a movie / raw input file
/// (`file@N`: its first N frames). Returns the frame count.
fn play_start(game: &mut SM64Game, spec: &str) -> Result<u32> {
    let pads: Vec<GamePad> = if let Some(rest) = spec.strip_prefix("tree:") {
        let (dir, node) = rest.rsplit_once(':').ok_or_else(|| anyhow::anyhow!("tree:<run_dir>:<node>"))?;
        tree_inputs(dir, node.parse()?)?
    } else {
        let (file, n) = match spec.rsplit_once('@') {
            Some((f, n)) => (f, Some(n.parse::<usize>()?)),
            None => (spec, None),
        };
        let mut p = movie(file)?;
        if let Some(n) = n {
            p.truncate(n);
        }
        p
    };
    for p in &pads {
        game.step_game(*p)?;
    }
    obrrt_core::check_layout(game)?;
    Ok(pads.len() as u32)
}

fn tree_inputs(dir: &str, node: usize) -> Result<Vec<GamePad>> {
    let records = obrrt_search::route::read_tree_bin(&format!("{dir}/tree.bin"))?;
    let records: Vec<(i64, Vec<GamePad>)> = records.into_iter().map(|r| (r.0 as i64, r.1)).collect();
    let mut chain = vec![node + 1];
    while let Some(&r) = chain.last() {
        if r == 0 {
            break;
        }
        let parent = records[r].0;
        chain.push(if parent >= 0 { parent as usize + 1 } else { 0 });
    }
    chain.reverse();
    Ok(chain.into_iter().flat_map(|r| records[r].1.clone()).collect())
}

fn write_raw(pads: &[GamePad], out: &str) -> Result<()> {
    let mut v = Vec::with_capacity(pads.len() * 4);
    for p in pads {
        v.extend_from_slice(&p.button.to_be_bytes());
        v.push(p.stick_x as u8);
        v.push(p.stick_y as u8);
    }
    std::fs::write(out, v)?;
    println!("{} frames -> {out}", pads.len());
    Ok(())
}

/// `cut <movie> <N> <out>`: the movie's first N frames, headerless (an
/// obrrt `<start.m64>` with OBRRT_BOOT_NOWAIT=1, as tas_trace --save).
fn cut(args: &[String]) -> Result<()> {
    let mut p = movie(&args[3])?;
    p.truncate(args[4].parse()?);
    write_raw(&p, &args[5])
}

/// `export <run_dir> <node> <out>`: power-on to the node's state.
fn export(args: &[String]) -> Result<()> {
    write_raw(&tree_inputs(&args[3], args[4].parse()?)?, &args[5])
}

/// `ga <start> [seeds=6] [key=SUBSTR] [steps=N]`: from the start (after N
/// neutral frames, then A on alternate frames until Mario is controllable),
/// the WaterWarp objectives generate() offers (OBRRT_WATERWARP=1 is set
/// here), `seeds` GA attempts each with the objective's budget; prints the
/// success, frames to the warp and frames spent.
fn ga(args: &[String]) -> Result<()> {
    use obrrt_search::ga::{Fitness, GaParams, Worker};
    use rand::SeedableRng;
    // SAFETY: single-threaded, before anything reads the environment.
    unsafe { std::env::set_var("OBRRT_WATERWARP", "1") };
    obrrt_objectives::moves::install_controller();
    obrrt_core::geo::want_static_floors();
    let gg = SM64GameGenerator::from_file(&args[1])?;
    let mut game = gg.create_game()?;
    let mut frame0 = play_start(&mut game, &args[3])?;
    let steps: usize = arg(args, "steps").map_or(0, |v| v.parse().unwrap());
    for _ in 0..steps {
        game.step_game(GamePad::new(0, 0, 0))?;
        frame0 += 1;
    }
    let mut waited = 0;
    while !read_state(&game, frame0, false).controllable() && waited < 600 {
        let a = if waited % 2 == 0 { obrrt_core::buttons::A_BUTTON } else { 0 };
        game.step_game(GamePad::new(a, 0, 0))?;
        frame0 += 1;
        waited += 1;
    }
    let s0 = read_state(&game, frame0, true);
    if std::env::var("OBRRT_MOVES").is_ok_and(|v| v == "1") {
        obrrt_objectives::moves::install();
    }
    obrrt_objectives::moves::observe(&s0);
    obrrt_core::mover::publish();
    let mut objs = Vec::new();
    waterwarp::generate(&s0, &mut objs);
    if let Some(k) = arg(args, "key") {
        objs.retain(|o| o.key().contains(k));
    }
    let all = waterwarp::zone_spots(&s0).map_or(0, |v| v.len());
    println!(
        "start {} frame {frame0} (waited {waited}) {} pos {} action {:#x} water {} | {all} spots in the zone, {} offered",
        args[3],
        name(s0.zone()),
        fmt(s0.pos),
        s0.action,
        s0.water_level,
        objs.len()
    );
    if objs.is_empty() {
        if let Some(v) = waterwarp::zone_spots(&s0) {
            let mut v: Vec<_> = v.iter().collect();
            v.sort_by(|a, b| hd(a.from, s0.pos).total_cmp(&hd(b.from, s0.pos)));
            for sp in v.iter().take(3) {
                println!("  nearest: {:?} from {} ({:.0} away)", sp.kind, fmt(sp.from), hd(sp.from, s0.pos));
            }
        }
        return Ok(());
    }
    let save = game.save_state();
    let mut w = Worker { base: gg.create_game()?, tmp: gg.create_game()? };
    let params = GaParams { scale: 3.0, fitness: Fitness::Steps, new_ops: true, pink: 0.0, per_objective_budget: false };
    let seeds: u64 = arg(args, "seeds").map_or(6, |v| v.parse().unwrap());
    for obj in &objs {
        println!("objective {} eligible {} target {:?} budget {:?}", obj.key(), obj.base_criteria(&s0), obj.steer_target(&s0).map(|t| [t[0] as i32, t[1] as i32, t[2] as i32]), obj.budget());
        let (mut hits, mut ts, mut spent) = (0, Vec::new(), 0u64);
        for seed in 0..seeds {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
            let at = w.attempt(&[obj.as_ref()], &s0, &save, &mut rng, params)?;
            let seg = &at.segments[0];
            spent += at.frames;
            // Replay to find the warp frame and where it put him.
            game.load_state(&save)?;
            let mut traj = vec![s0.clone()];
            for (k, p) in seg.inputs.iter().enumerate() {
                game.step_game(*p)?;
                traj.push(read_state(&game, frame0 + k as u32 + 1, false));
            }
            let warp = (1..traj.len()).find(|&i| waterwarp::warp(&traj[i - 1], &traj[i], traj.get(i + 1)).is_some());
            println!(
                "  seed {seed}: {} | warp {} | attempt frames {} | winner gen {} origin {}",
                if seg.success { format!("SUCCESS at {}", seg.inputs.len()) } else { format!("miss (progress {:.0})", seg.progress) },
                warp.map_or("-".to_string(), |i| format!("f+{i} {} -> {}", fmt(traj[i - 1].pos), fmt(traj[i].pos))),
                at.frames,
                at.ga.as_ref().map_or(0, |g| g.winner_gen),
                at.ga.as_ref().map_or("-", |g| g.winner_origin)
            );
            if seg.success {
                hits += 1;
                ts.push(seg.inputs.len());
            }
        }
        ts.sort_unstable();
        println!(
            "  summary {}: hits {hits}/{seeds} | frames to warp median {} best {} | mean attempt frames {}",
            obj.key(),
            ts.get(ts.len() / 2).map_or("-".into(), |t| t.to_string()),
            ts.first().map_or("-".into(), |t| t.to_string()),
            spent / seeds.max(1)
        );
    }
    Ok(())
}
