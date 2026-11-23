//! Brute-force the one-step (and 3-step) outcome of every button set x stick cell from sampled
//! states, to find which presses and stick regions matter (menu table), and to validate
//! `menu::menu` (coverage of the pruned menu against the full outcome set).
//!
//!   menu_check collect <rom> <out.jsonl> [threads=4] [cap=25] [gap=20] [movie...]
//!   menu_check eval    <rom> <in.jsonl>   (needs menu.rs; replays the sampled states again)
//!
//! Pad layout per state: button set b in 0..8 (A=1, B=2, Z=4) x stick cell s in 0..121
//! (0 = neutral; 1 + mag_i * 24 + ang_i, angle = ang_i * 15 deg from Mario's facing,
//! magnitude MAGS[mag_i] = stickMag). Outcome of a pad = (action, speed class, vy class)
//! after 1 frame and after 3 frames of the same pad.

use std::collections::HashMap;
use std::io::Write;
use std::sync::Mutex;

use anyhow::Result;
use obrrt_core::{GamePad, St, read_state};
use obrrt_gp::problems::read_movie;
use sm64_binds::{SM64Game, SM64GameGenerator};

pub const MAGS: [f32; 5] = [16.0, 32.0, 46.0, 54.0, 64.0];
pub const NANG: usize = 24;
pub const NSTICK: usize = 1 + MAGS.len() * NANG;

const A: u16 = 0x8000;
const B: u16 = 0x4000;
const Z: u16 = 0x2000;

fn buttons(b: usize) -> u16 {
    (if b & 1 != 0 { A } else { 0 }) | (if b & 2 != 0 { B } else { 0 }) | (if b & 4 != 0 { Z } else { 0 })
}

/// The pad for button set `b`, stick cell `s`, from state `st`.
pub fn pad_of(st: &St, b: usize, s: usize) -> GamePad {
    if s == 0 {
        return GamePad::new(buttons(b), 0, 0);
    }
    let k = s - 1;
    let (mi, ai) = (k / NANG, k % NANG);
    let yaw = (st.yaw as i32 + (ai as i32) * 65536 / NANG as i32) as i16;
    let (x, y) = obrrt_gp::lang::stick(st.cam_yaw, yaw, MAGS[mi] / 121.0);
    GamePad::new(buttons(b), x, y)
}

fn fv_class(v0: f32, v1: f32) -> i8 {
    if v0.abs() < 5.0 {
        return 10 + (v1 / 8.0).floor().clamp(-5.0, 20.0) as i8;
    }
    if v1.abs() < 1.0 {
        return 0;
    }
    let r = v1 / v0;
    match r {
        r if r < 0.0 => -1,
        r if r < 0.7 => 1,
        r if r < 0.9 => 2,
        r if r < 0.97 => 3,
        r if r < 1.03 => 4,
        _ => 5,
    }
}

fn vy_class(vy: f32) -> i8 {
    if vy < -1.0 {
        0
    } else if vy <= 1.0 {
        1
    } else if vy <= 15.0 {
        2
    } else {
        3
    }
}

/// (action, speed class, vy class, in air).
pub type Out = (u32, i8, i8, bool);

pub fn outcome(v0: f32, s: &St) -> Out {
    (s.action, fv_class(v0, s.forward_vel), vy_class(s.vel[1]), s.action & 0x800 != 0)
}

/// All pads from the saved state; returns (outcome after 1 frame, after 3 frames) per pad index b * NSTICK + s.
fn brute(game: &mut SM64Game, save: &sm64_binds::SaveState, st: &St, frame: u32, pads: &[(usize, usize)]) -> Result<Vec<(Out, Out)>> {
    let mut out = Vec::with_capacity(pads.len());
    for &(b, s) in pads {
        game.load_state(save)?;
        let pad = pad_of(st, b, s);
        game.step_game(pad)?;
        let s1 = read_state(game, frame + 1, false);
        let o1 = outcome(st.forward_vel, &s1);
        game.step_game(pad)?;
        game.step_game(pad)?;
        let s3 = read_state(game, frame + 3, false);
        out.push((o1, outcome(st.forward_vel, &s3)));
    }
    Ok(out)
}

fn all_pads() -> Vec<(usize, usize)> {
    (0..8).flat_map(|b| (0..NSTICK).map(move |s| (b, s))).collect()
}

fn collect_movie(gg: &SM64GameGenerator, path: &str, cap: usize, gap: usize, out: &Mutex<std::fs::File>) -> Result<usize> {
    let pads = read_movie(path)?;
    let mut game = gg.create_game()?;
    let power_on = game.save_state();
    game.load_state(&power_on)?;
    let mut count: HashMap<u32, usize> = HashMap::new();
    let mut last: HashMap<u32, usize> = HashMap::new();
    let all = all_pads();
    let mut n = 0;
    let mut prev_pad = GamePad::new(0, 0, 0);
    for (t, p) in pads.iter().enumerate() {
        let st = read_state(&game, t as u32, false);
        let a = st.action;
        let c = count.get(&a).copied().unwrap_or(0);
        let due = last.get(&a).is_none_or(|&l| t >= l + gap);
        if c < cap && due && st.controllable() && st.alive() {
            let save = game.save_state();
            let res = brute(&mut game, &save, &st, t as u32, &all)?;
            game.load_state(&save)?;
            let mut dict: Vec<Out> = Vec::new();
            let mut idx: HashMap<Out, usize> = HashMap::new();
            let mut enc = |o: Out| *idx.entry(o).or_insert_with(|| { dict.push(o); dict.len() - 1 });
            let (o1, o3): (Vec<usize>, Vec<usize>) = res.iter().map(|(a, b)| (enc(*a), enc(*b))).unzip();
            let line = serde_json::json!({
                "mv": path, "t": t, "act": a, "fv": st.forward_vel, "vy": st.vel[1], "ny": st.floor_normal[1],
                "ftype": st.floor_type, "yaw": st.yaw, "cam": st.cam_yaw, "atimer": st.action_timer, "last": prev_pad.button,
                "lx": prev_pad.stick_x, "ly": prev_pad.stick_y, "hold": st.holding, "flags": st.mario_flags,
                "wall": st.wall_normal.is_some(), "h": st.pos[1] - st.floor_y,
                "dict": dict, "o1": o1, "o3": o3
            });
            writeln!(out.lock().unwrap(), "{line}")?;
            *count.entry(a).or_insert(0) += 1;
            last.insert(a, t);
            n += 1;
        }
        game.step_game(*p)?;
        prev_pad = *p;
    }
    Ok(n)
}

// ------------------------------------------------------------------ eval
use obrrt_gp::lang::{Ctx, MAX_POINTS, RefNow};
use obrrt_policy::menu;
use obrrt_policy::types::{Choice, Obs, Sense};
use std::collections::BTreeMap;

type Key = (Out, u8);

struct Rec {
    act: u32,
    full: std::collections::HashSet<Key>,
}

fn load_records(path: &str) -> Result<BTreeMap<String, BTreeMap<usize, Rec>>> {
    let mut m: BTreeMap<String, BTreeMap<usize, Rec>> = BTreeMap::new();
    for l in std::fs::read_to_string(path)?.lines() {
        let v: serde_json::Value = serde_json::from_str(l)?;
        let dict: Vec<Out> = v["dict"].as_array().unwrap().iter().map(|x| {
            let a = x.as_array().unwrap();
            (a[0].as_u64().unwrap() as u32, a[1].as_i64().unwrap() as i8, a[2].as_i64().unwrap() as i8, a[3].as_bool().unwrap())
        }).collect();
        let mut full = std::collections::HashSet::new();
        for i in v["o1"].as_array().unwrap() { full.insert((dict[i.as_u64().unwrap() as usize], 1u8)); }
        for i in v["o3"].as_array().unwrap() { full.insert((dict[i.as_u64().unwrap() as usize], 3u8)); }
        m.entry(v["mv"].as_str().unwrap().to_string()).or_default().insert(v["t"].as_u64().unwrap() as usize, Rec { act: v["act"].as_u64().unwrap() as u32, full });
    }
    Ok(m)
}

#[derive(Default, Clone)]
struct Agg {
    n: usize,
    cells: usize,
    pads: usize,
    full: usize,
    covered: usize,
    perfect: usize,
    /// Discrete outcomes only: (action, vy class, air, horizon), speed class ignored.
    full_a: usize,
    covered_a: usize,
    perfect_a: usize,
    missed_a: HashMap<(u32, i8, bool, u8), usize>,
    missed: HashMap<Key, usize>,
}

/// Fine positions sampled inside each class.
const FINE_ANG: [f32; 3] = [-0.8, 0.0, 0.8];
const FINE_MAG: [f32; 3] = [-1.0, 0.0, 1.0];

fn eval_movie(gg: &SM64GameGenerator, path: &str, recs: &BTreeMap<usize, Rec>, out: &Mutex<HashMap<u32, Agg>>) -> Result<()> {
    let pads = read_movie(path)?;
    let mut game = gg.create_game()?;
    let power_on = game.save_state();
    game.load_state(&power_on)?;
    let mut prev = GamePad::new(0, 0, 0);
    let last_t = *recs.keys().max().unwrap();
    let sense = Sense::default();
    for (t, p) in pads.iter().enumerate() {
        if t > last_t { break; }
        if let Some(rec) = recs.get(&t) {
            let st = read_state(&game, t as u32, false);
            let save = game.save_state();
            let mut pts = [[0.0f32; 3]; MAX_POINTS];
            // A target off to one side, so Ref::Target differs from Face.
            let ty = st.yaw as f32 * std::f32::consts::TAU / 65536.0 + 0.7;
            pts[0] = [st.pos[0] + 1500.0 * ty.sin(), st.pos[1], st.pos[2] + 1500.0 * ty.cos()];
            let ctx = Ctx { s: &st, last: prev, in_act: 1, pts, pace: 0.0, rf: RefNow::default() };
            let obs = Obs { ctx: &ctx, sense: &sense };
            let cells = menu::menu(&obs);
            let mut got: std::collections::HashSet<Key> = std::collections::HashSet::new();
            let mut npads = 0;
            for c in &cells {
                let neutral = c.stick.mag == obrrt_policy::types::Mag::Neutral;
                let fas: &[f32] = if neutral { &[0.0] } else { &FINE_ANG };
                let fms: &[f32] = if neutral { &[0.0] } else { &FINE_MAG };
                for &fa in fas {
                    for &fm in fms {
                        let ch = Choice { cell: *c, yaw: 0, fine_ang: fa, fine_mag: fm };
                        let pad = menu::realize(&ch, &obs);
                        game.load_state(&save)?;
                        game.step_game(pad)?;
                        let s1 = read_state(&game, t as u32 + 1, false);
                        got.insert((outcome(st.forward_vel, &s1), 1));
                        game.step_game(pad)?;
                        game.step_game(pad)?;
                        let s3 = read_state(&game, t as u32 + 3, false);
                        got.insert((outcome(st.forward_vel, &s3), 3));
                        npads += 1;
                    }
                }
            }
            game.load_state(&save)?;
            let covered = rec.full.iter().filter(|k| got.contains(*k)).count();
            let proj = |set: &std::collections::HashSet<Key>| -> std::collections::HashSet<(u32, i8, bool, u8)> { set.iter().map(|k| (k.0.0, k.0.2, k.0.3, k.1)).collect() };
            let (fa, ga) = (proj(&rec.full), proj(&got));
            let cov_a = fa.iter().filter(|k| ga.contains(*k)).count();
            let mut o = out.lock().unwrap();
            let a = o.entry(rec.act).or_default();
            a.n += 1;
            a.cells += cells.len();
            a.pads += npads;
            a.full += rec.full.len();
            a.covered += covered;
            a.perfect += (covered == rec.full.len()) as usize;
            a.full_a += fa.len();
            a.covered_a += cov_a;
            a.perfect_a += (cov_a == fa.len()) as usize;
            for k in fa.iter().filter(|k| !ga.contains(*k)) {
                *a.missed_a.entry(*k).or_default() += 1;
            }
            for k in rec.full.iter().filter(|k| !got.contains(*k)) {
                *a.missed.entry(*k).or_default() += 1;
            }
        }
        game.step_game(*p)?;
        prev = *p;
    }
    Ok(())
}

// ------------------------------------------------------------------ carry
#[derive(Default)]
struct CarryAgg {
    n: usize,
    diff: [usize; 8],
    tot: usize,
    cover: HashMap<Vec<u8>, usize>,
    /// Mean share of the touchdown outcomes reached by holding only these button sets: none; none+A; none,A,Z,A+Z; all eight.
    fam: [f64; 4],
}

/// Do button levels held through the air change what is reachable at touchdown? For sampled air
/// states: hold button set b1 (stick full ahead) until the action stops being airborne, then at that
/// first ground state try every b2 x {neutral, full ahead} for 1 and 3 frames; the union of action
/// outcomes is the signature of b1.
fn carry_movie(gg: &SM64GameGenerator, path: &str, recs: &BTreeMap<usize, Rec>, out: &Mutex<HashMap<u32, CarryAgg>>) -> Result<()> {
    let pads = read_movie(path)?;
    let mut game = gg.create_game()?;
    let power_on = game.save_state();
    game.load_state(&power_on)?;
    let last_t = *recs.keys().max().unwrap();
    for (t, p) in pads.iter().enumerate() {
        if t > last_t { break; }
        if let Some(rec) = recs.get(&t) {
            let st = read_state(&game, t as u32, false);
            if st.action & 0x800 != 0 && st.vel[1] < 30.0 {
                let save = game.save_state();
                let mut sigs: Vec<Option<std::collections::BTreeSet<(u32, u32)>>> = Vec::new();
                for b1 in 0..8usize {
                    game.load_state(&save)?;
                    let mut cur = st.clone();
                    let mut landed = None;
                    for k in 0..60 {
                        let pad = pad_of(&cur, b1, 1 + 4 * NANG); // mag index 4 (64), angle 0
                        game.step_game(pad)?;
                        cur = read_state(&game, t as u32 + 1 + k, false);
                        if cur.action & 0x800 == 0 { landed = Some(k); break; }
                    }
                    if landed.is_none() { sigs.push(None); continue; }
                    let ls = game.save_state();
                    let mut set = std::collections::BTreeSet::new();
                    for b2 in 0..8usize {
                        for s2 in [0usize, 1 + 4 * NANG] {
                            game.load_state(&ls)?;
                            let pad = pad_of(&cur, b2, s2);
                            game.step_game(pad)?;
                            let a1 = read_state(&game, 0, false).action;
                            game.step_game(pad)?;
                            game.step_game(pad)?;
                            let a3 = read_state(&game, 0, false).action;
                            set.insert((a1, a3));
                        }
                    }
                    sigs.push(Some(set));
                }
                game.load_state(&save)?;
                if sigs.iter().all(|x| x.is_some()) {
                    let mut o = out.lock().unwrap();
                    let e = o.entry(rec.act).or_default();
                    e.n += 1;
                    for b1 in 1..8 {
                        if sigs[b1] != sigs[0] { e.diff[b1] += 1; }
                    }
                    e.tot += sigs[0].as_ref().unwrap().len();
                    // Smallest set of held button sets whose touchdown outcomes cover the union (greedy).
                    let sets: Vec<&std::collections::BTreeSet<(u32, u32)>> = sigs.iter().map(|x| x.as_ref().unwrap()).collect();
                    let full: std::collections::BTreeSet<(u32, u32)> = sets.iter().flat_map(|s| s.iter().copied()).collect();
                    let mut got: std::collections::BTreeSet<(u32, u32)> = std::collections::BTreeSet::new();
                    let mut chosen: Vec<u8> = Vec::new();
                    while got.len() < full.len() {
                        let b = (0..8).max_by_key(|&b| sets[b].difference(&got).count()).unwrap();
                        got.extend(sets[b].iter().copied());
                        chosen.push(b as u8);
                    }
                    chosen.sort();
                    *e.cover.entry(chosen).or_default() += 1;
                    for (fi, fam) in [&[0usize][..], &[0, 1], &[0, 1, 4, 5], &[0, 1, 2, 3, 4, 5, 6, 7]].iter().enumerate() {
                        let u: std::collections::BTreeSet<(u32, u32)> = fam.iter().flat_map(|&b| sets[b].iter().copied()).collect();
                        e.fam[fi] += u.len() as f64 / full.len().max(1) as f64;
                    }
                }
            }
        }
        game.step_game(*p)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("collect") => {
            let gg = SM64GameGenerator::from_file(&args[2])?;
            let threads: usize = args.get(4).map_or(Ok(4), |s| s.parse())?;
            let cap: usize = args.get(5).map_or(Ok(25), |s| s.parse())?;
            let gap: usize = args.get(6).map_or(Ok(20), |s| s.parse())?;
            let movies: Vec<String> = if args.len() > 7 {
                args[7..].to_vec()
            } else {
                ["ref/tas/full-120star-2026-U.m64", "ref/tas/full-120star-2012-U.m64", "ref/tas/full-120star-rikku-U.m64", "ref/tas/full-70star-U.m64"].map(String::from).to_vec()
            };
            let f = Mutex::new(std::fs::File::create(&args[3])?);
            let next = std::sync::atomic::AtomicUsize::new(0);
            std::thread::scope(|sc| {
                for _ in 0..threads {
                    sc.spawn(|| loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= movies.len() {
                            break;
                        }
                        match collect_movie(&gg, &movies[i], cap, gap, &f) {
                            Ok(n) => eprintln!("{}: {n} states", movies[i]),
                            Err(e) => eprintln!("{}: {e:#}", movies[i]),
                        }
                    });
                }
            });
        }
        Some("carry") => {
            let gg = SM64GameGenerator::from_file(&args[2])?;
            let recs = load_records(&args[3])?;
            let threads: usize = args.get(4).map_or(Ok(4), |s| s.parse())?;
            let out: Mutex<HashMap<u32, CarryAgg>> = Mutex::new(HashMap::new());
            let movies: Vec<&String> = recs.keys().collect();
            let next = std::sync::atomic::AtomicUsize::new(0);
            std::thread::scope(|sc| {
                for _ in 0..threads {
                    sc.spawn(|| loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= movies.len() { break; }
                        if let Err(e) = carry_movie(&gg, movies[i], &recs[movies[i]], &out) { eprintln!("{}: {e:#}", movies[i]); }
                    });
                }
            });
            let names: HashMap<u32, String> = args.get(5).and_then(|p| std::fs::read_to_string(p).ok()).and_then(|t| serde_json::from_str::<HashMap<String, String>>(&t).ok())
                .map(|m| m.into_iter().filter_map(|(k, v)| k.parse().ok().map(|k| (k, v))).collect()).unwrap_or_default();
            let o = out.into_inner().unwrap();
            let mut rows: Vec<_> = o.iter().collect();
            rows.sort_by_key(|(_, v)| std::cmp::Reverse(v.n));
            println!("air action: n | fraction of states where holding button set b1 through the air changes the touchdown outcomes vs holding none | mean #outcome pairs");
            for (a, ag) in rows {
                let (n, d, tot) = (&ag.n, &ag.diff, &ag.tot);
                if *n < 5 { continue; }
                let mut cv: Vec<_> = ag.cover.iter().collect();
                cv.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
                let cvs: Vec<String> = cv.iter().take(4).map(|(k, c)| format!("{:?}x{}", k, c)).collect();
                print!("[cover by held-sets {{0}}/{{0,A}}/{{0,A,Z,AZ}}/all: {:.2}/{:.2}/{:.2}/{:.2}] [min covering held-sets: {}] ", ag.fam[0] / *n as f64, ag.fam[1] / *n as f64, ag.fam[2] / *n as f64, ag.fam[3] / *n as f64, cvs.join(" "));
                println!("{:22} n={:4} A={:.2} B={:.2} AB={:.2} Z={:.2} AZ={:.2} BZ={:.2} ABZ={:.2} | outcomes {:.1}", names.get(a).cloned().unwrap_or(format!("{a:#x}")), n, d[1] as f64 / *n as f64, d[2] as f64 / *n as f64, d[3] as f64 / *n as f64, d[4] as f64 / *n as f64, d[5] as f64 / *n as f64, d[6] as f64 / *n as f64, d[7] as f64 / *n as f64, *tot as f64 / *n as f64);
            }
        }
        Some("eval") => {
            let gg = SM64GameGenerator::from_file(&args[2])?;
            let recs = load_records(&args[3])?;
            let threads: usize = args.get(4).map_or(Ok(4), |s| s.parse())?;
            let out: Mutex<HashMap<u32, Agg>> = Mutex::new(HashMap::new());
            let movies: Vec<&String> = recs.keys().collect();
            let next = std::sync::atomic::AtomicUsize::new(0);
            std::thread::scope(|sc| {
                for _ in 0..threads {
                    sc.spawn(|| loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= movies.len() { break; }
                        if let Err(e) = eval_movie(&gg, movies[i], &recs[movies[i]], &out) { eprintln!("{}: {e:#}", movies[i]); }
                        eprintln!("done {}", movies[i]);
                    });
                }
            });
            let names: HashMap<u32, String> = args.get(5).and_then(|p| std::fs::read_to_string(p).ok()).and_then(|t| serde_json::from_str::<HashMap<String, String>>(&t).ok())
                .map(|m| m.into_iter().filter_map(|(k, v)| k.parse().ok().map(|k| (k, v))).collect()).unwrap_or_default();
            let o = out.into_inner().unwrap();
            let mut rows: Vec<(&u32, &Agg)> = o.iter().collect();
            rows.sort_by_key(|(_, a)| std::cmp::Reverse(a.n));
            let (mut tn, mut tf, mut tc, mut tp, mut tcells) = (0, 0, 0, 0, 0);
            let (mut tfa, mut tca, mut tpa) = (0, 0, 0);
            for (act, a) in &rows {
                let nm = names.get(act).cloned().unwrap_or(format!("{act:#x}"));
                let mut miss: Vec<_> = a.missed.iter().collect();
                miss.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
                let ms: Vec<String> = miss.iter().take(3).map(|(k, c)| format!("{}:{}/{}/{}{}@{}x{}", names.get(&k.0.0).cloned().unwrap_or(format!("{:#x}", k.0.0)), k.0.1, k.0.2, if k.0.3 { "air" } else { "gnd" }, "", k.1, c)).collect();
                let mut ma: Vec<_> = a.missed_a.iter().collect();
                ma.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
                let mas: Vec<String> = ma.iter().take(3).map(|(k, c)| format!("{}:vy{}/{}@{}x{}", names.get(&k.0).cloned().unwrap_or(format!("{:#x}", k.0)), k.1, if k.2 { "air" } else { "gnd" }, k.3, c)).collect();
                println!("{nm:26} n={:4} cells={:5.1} pads={:5.1} | act-cover={:.4} perfect={:.3} | full-cover={:.3} perfect={:.2} | miss(act): {} | miss(full): {}", a.n, a.cells as f64 / a.n as f64, a.pads as f64 / a.n as f64, a.covered_a as f64 / a.full_a.max(1) as f64, a.perfect_a as f64 / a.n as f64, a.covered as f64 / a.full.max(1) as f64, a.perfect as f64 / a.n as f64, mas.join(" "), ms.join("  "));
                tfa += a.full_a; tca += a.covered_a; tpa += a.perfect_a;
                tn += a.n; tf += a.full; tc += a.covered; tp += a.perfect; tcells += a.cells;
            }
            println!("TOTAL states={tn} mean cells={:.1} (full grid {}) | ACTION-level cover={:.4} perfect states={:.3} | full outcome (with speed class) cover={:.4} perfect={:.3}", tcells as f64 / tn as f64, 8 * NSTICK, tca as f64 / tfa as f64, tpa as f64 / tn as f64, tc as f64 / tf as f64, tp as f64 / tn as f64);
        }
        _ => anyhow::bail!("usage: menu_check collect <rom> <out.jsonl> [threads cap gap movies...]"),
    }
    Ok(())
}
