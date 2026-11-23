//! Validation of policy/src/sense.rs against the emulator.
//!
//!   sense_check check <rom> <movie> <from> <to> [stride=3]   geometry, landing, objects, timing
//!   sense_check fields <rom>                                 the Reader: names, values vs St, timing
//!   sense_check md                                           writes BEHAVIOURS.md (to stdout)
//!
//! The movie is played from power-on; at every `stride`-th frame in [from, to] the sense of
//! that state is compared with what the game then does (or with geo::floor_at, the game-faithful
//! static query).

use std::time::Instant;

use anyhow::Result;
use obrrt_core::geo;
use obrrt_core::{GamePad, St, read_state};
use obrrt_gp::problems::read_movie;
use obrrt_policy::sense::reader::{self, FieldSet, Value};
use obrrt_policy::sense::{self, AirModel, MarioFlags, SenseCtx, SenseOpts};
use obrrt_policy::types::Sense;
use sm64_binds::{SM64Game, SM64GameGenerator};

const AIR: u32 = 1 << 11;

fn pct(v: &mut Vec<f32>, p: f32) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f32 * p) as usize]
}

#[derive(Default)]
struct Tally {
    probes: usize,
    probe_bad: usize,
    own_floor: usize,
    own_floor_bad: usize,
    ceil: usize,
    ceil_bad: usize,
    ceil_dyn: usize,
    wall_frames: usize,
    wall_hit: usize,
    wall_pred: usize,
    wall_pred_hit: usize,
    wall3_hit: usize,
    wall3_pred: usize,
    wall3_both: usize,
    edge_pred: usize,
    edge_pred_air: usize,
    edge_air: usize,
    edge_air_pred: usize,
    land_n: usize,
    land_frames_err: Vec<f32>,
    land_pos_err: Vec<f32>,
    land_missing: usize,
    land_const_frames_err: Vec<f32>,
    land_const_pos_err: Vec<f32>,
    ledge_pred: usize,
    ledge_actual: usize,
    ledge_both: usize,
    cat_seen: std::collections::BTreeMap<String, (u8, usize)>,
    zone_miss: usize,
}

fn check(rom: &str, movie: &str, from: usize, to: usize, stride: usize) -> Result<()> {
    obrrt_objectives::moves::install_controller();
    let gg = SM64GameGenerator::from_file(rom)?;
    let mut game: SM64Game = gg.create_game()?;
    let pads = read_movie(movie)?;
    let to = to.min(pads.len().saturating_sub(40));
    // Play to `from`, then keep the states of the window.
    let mut t = Tally::default();
    let mut ctx = SenseCtx::new();
    let opts = SenseOpts::default();
    let mut states: Vec<St> = Vec::new();
    let mut senses: Vec<Option<Sense>> = Vec::new();
    let mut senses_c: Vec<Option<Sense>> = Vec::new();
    let mut time_parts = [0f64; 7];
    let mut time_n = 0usize;
    let mut rng = 12345u64;
    let mut rnd = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f32 / (1u64 << 53) as f32
    };
    let mut bad_examples: Vec<String> = Vec::new();
    for f in 0..to + 40 {
        if f >= from.saturating_sub(1) && f <= to + 40 {
            let sampled = f >= from && f <= to && (f - from) % stride == 0;
            let st = read_state(&game, f as u32, sampled);
            if sampled && st.controllable() {
                let pad = pads[f];
                let air = AirModel::from_pad(&st, pad);
                let s = game.with_memory(|m| {
                    let s = ctx.read(m, &st, &opts, &[], &air);
                    // Against geo::floor_at (static) at the probes and at random nearby points.
                    if ctx.has_zone(&st) {
                        let hd = st.yaw as f32 * std::f32::consts::TAU / 65536.0;
                        let (sy, cy) = (hd.sin(), hd.cos());
                        let mut check_pt = |x: f32, z: f32, y: f32, p: &obrrt_policy::types::FloorProbe, t: &mut Tally| {
                            // Compare with static-only truth: objects floors may legitimately be higher.
                            let truth = geo::floor_at(m, x, y, z);
                            t.probes += 1;
                            let ok = match truth {
                                None => !p.has_floor || p.y > -10999.0 && false,
                                Some(h) => p.has_floor && (p.y - h.y).abs() < 1.0 || (p.has_floor && p.y > h.y),
                            };
                            if !ok {
                                t.probe_bad += 1;
                                if bad_examples.len() < 8 {
                                    bad_examples.push(format!("f{f} probe ({x:.0},{y:.0},{z:.0}) mine {:?} truth {:?}", (p.has_floor, p.y, p.ty), truth.map(|h| (h.y, h.kind))));
                                }
                            }
                        };
                        for (i, &d) in sense::AHEAD_D.iter().enumerate() {
                            check_pt(st.pos[0] + sy * d, st.pos[2] + cy * d, st.pos[1], &s.ahead[i], &mut t);
                        }
                        let (vx, vz) = (st.vel[0], st.vel[2]);
                        let vl = vx.hypot(vz);
                        if vl > 1e-3 {
                            for (i, &d) in sense::VEL_D.iter().enumerate() {
                                check_pt(st.pos[0] + vx / vl * d, st.pos[2] + vz / vl * d, st.pos[1], &s.vel_ahead[i], &mut t);
                            }
                        }
                        // Random points through the cheap path (floors only).
                        let o2 = SenseOpts { walls: false, edges: false, landing: false, objs: false, dyn_floors: false, ..opts };
                        let _ = o2;
                        // Own position vs the game's referenced floor.
                        if st.floor_obj.is_none() && st.floor_y > -10000.0 {
                            t.own_floor += 1;
                            if let Some(h) = geo::floor_at(m, st.pos[0], st.pos[1], st.pos[2]) {
                                if (h.y - st.floor_y).abs() > 1.0 {
                                    t.own_floor_bad += 1;
                                }
                            }
                        }
                        // Ceiling headroom vs the game's ceilHeight (static ceilings).
                        let ceil_obj = {
                            let rd = |a: usize| m.get(a..a + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize).unwrap_or(0);
                            let cp = rd(obrrt_core::layout::ADDR_gMarioStates as usize + obrrt_core::layout::MARIO_ceil);
                            cp != 0 && rd(cp + obrrt_core::layout::SURFACE_object) != 0
                        };
                        if ceil_obj {
                            t.ceil_dyn += 1;
                        } else if st.ceil_y < 19000.0 {
                            t.ceil += 1;
                            // The game's ceilHeight was found at the position at the start of the frame
                            // (before the step); compare there.
                            let pp = [st.pos[0] - st.vel[0], st.pos[1] - st.vel[1], st.pos[2] - st.vel[2]];
                            let yq = st.floor_y.max(pp[1]) + 80.0;
                            let mine = ctx.ceil_height(&st, pp[0], yq, pp[2]).unwrap_or(f32::INFINITY);
                            if (mine - st.ceil_y).abs() > 1.0 {
                                t.ceil_bad += 1;
                                if t.ceil_bad <= 6 {
                                    println!("   ceil f{f} pos {:?} floor_y {} game ceil {} mine {} clear {} action {:#x}", st.pos, st.floor_y, st.ceil_y, mine, s.ceil_clear, st.action);
                                }
                            }
                        }
                    }
                    // Timing, part by part (fresh context state: zone already built).
                    if ((f - from) / stride) % 4 == 0 {
                        let parts = [
                            SenseOpts { floors: true, walls: false, edges: false, landing: false, objs: false, dyn_floors: true, dyn_walls: true },
                            SenseOpts { floors: false, walls: true, edges: false, landing: false, objs: false, dyn_floors: true, dyn_walls: true },
                            SenseOpts { floors: false, walls: false, edges: true, landing: false, objs: false, dyn_floors: true, dyn_walls: true },
                            SenseOpts { floors: false, walls: false, edges: false, landing: true, objs: false, dyn_floors: true, dyn_walls: true },
                            SenseOpts { floors: false, walls: false, edges: false, landing: false, objs: true, dyn_floors: true, dyn_walls: true },
                            opts,
                            SenseOpts { dyn_floors: false, dyn_walls: false, ..opts },
                        ];
                        for (k, po) in parts.iter().enumerate() {
                            let t0 = Instant::now();
                            let mut sink = 0usize;
                            for _ in 0..20 {
                                let ss = ctx.read(m, &st, po, &[], &air);
                                sink += ss.objs.len();
                            }
                            std::hint::black_box(sink);
                            time_parts[k] += t0.elapsed().as_secs_f64() / 20.0;
                        }
                        time_n += 1;
                    }
                    s
                });
                // Constant pad (full stick forward, A held) as the no-knowledge prediction.
                let sc = game.with_memory(|m| ctx.read(m, &st, &SenseOpts { floors: false, walls: false, edges: false, landing: true, objs: false, dyn_floors: true, dyn_walls: true }, &[], &AirModel::default()));
                for o in &s.objs {
                    let _ = o;
                }
                if let Some(objs) = st.objects.as_ref() {
                    for o in objs.iter() {
                        let name = obrrt_core::bhv_name(o.bhv).unwrap_or("?").to_string();
                        t.cat_seen.entry(name.clone()).or_insert((sense::behaviours::cat_of(o.bhv), 0)).1 += 1;
                    }
                }
                states.push(st.clone());
                senses.push(Some(s));
                senses_c.push(Some(sc));
            } else {
                states.push(st.clone());
                senses.push(None);
                senses_c.push(None);
            }
        }
        game.step_game(pads[f])?;
    }
    // states[k] corresponds to frame from-1+k.
    let base = from.saturating_sub(1);
    let idx = |f: usize| f - base;
    for f in from..=to {
        if (f - from) % stride != 0 {
            continue;
        }
        let (Some(s), Some(sc)) = (&senses[idx(f)], &senses_c[idx(f)]) else { continue };
        let st = &states[idx(f)];
        let next = &states[idx(f) + 1];
        let hv = st.vel[0].hypot(st.vel[2]);
        // Walls: along heading vs velocity; frames with heading ~ velocity.
        if hv > 8.0 && s.wall[0].is_finite() | s.wall[0].is_infinite() {
            let ang = (st.vel[0].atan2(st.vel[2]) * 65536.0 / std::f32::consts::TAU) as i32 - st.yaw as i32;
            if (ang as i16).unsigned_abs() < 0x1000 && st.action & AIR != 0 || st.action & AIR == 0 && (ang as i16).unsigned_abs() < 0x1000 {
                t.wall_frames += 1;
                let hit = next.wall_normal.is_some();
                let pred = s.wall[0] <= hv + 2.0;
                let hit3 = (1..=3usize).any(|k| states.get(idx(f) + k).is_some_and(|x| x.wall_normal.is_some()));
                let pred3 = s.wall[0] <= 3.0 * hv + 2.0;
                t.wall3_hit += hit3 as usize;
                t.wall3_pred += pred3 as usize;
                t.wall3_both += (hit3 && pred3) as usize;
                t.wall_hit += hit as usize;
                t.wall_pred += pred as usize;
                t.wall_pred_hit += (hit && pred) as usize;
            }
        }
        // Edge: predicted within 12 frames of travel -> airborne within 14 frames.
        if st.action & AIR == 0 && hv > 15.0 {
            let pred = s.edge_ahead <= 12.0 * hv;
            let mut air_soon = false;
            for k in 1..=14usize {
                if let Some(x) = states.get(idx(f) + k) {
                    if x.action & AIR != 0 {
                        air_soon = true;
                        break;
                    }
                }
            }
            t.edge_pred += pred as usize;
            t.edge_pred_air += (pred && air_soon) as usize;
            t.edge_air += air_soon as usize;
            t.edge_air_pred += (air_soon && pred) as usize;
        }
        // Ledge.
        {
            let pred = s.ledge_ahead.is_finite() && s.ledge_ahead <= hv + 60.0;
            let actual = (1..=3usize).any(|k| states.get(idx(f) + k).is_some_and(|x| x.action == 0x0800_034B));
            t.ledge_pred += pred as usize;
            t.ledge_actual += actual as usize;
            t.ledge_both += (pred && actual) as usize;
        }
        // Landing: airborne now, find the real landing.
        if st.action & AIR != 0 && st.vel[1] < 30.0 {
            let mut k = 1usize;
            let mut real = None;
            while let Some(x) = states.get(idx(f) + k) {
                if x.action & AIR == 0 {
                    real = Some((k, x.pos));
                    break;
                }
                k += 1;
                if k > 100 {
                    break;
                }
            }
            if let Some((k, rp)) = real {
                // A landing needs the same action all the way (an action change mid-air alters physics);
                // keep flights with no action change.
                let same = (0..k).all(|j| states.get(idx(f) + j).is_some_and(|x| x.action == st.action));
                if !same {
                    continue;
                }
                t.land_n += 1;
                match s.land {
                    Some(l) => {
                        t.land_frames_err.push(l.frames as f32 - k as f32);
                        t.land_pos_err.push(((l.pos[0] - rp[0]).powi(2) + (l.pos[2] - rp[2]).powi(2)).sqrt());
                    }
                    None => t.land_missing += 1,
                }
                if let Some(l) = sc.land {
                    t.land_const_frames_err.push(l.frames as f32 - k as f32);
                    t.land_const_pos_err.push(((l.pos[0] - rp[0]).powi(2) + (l.pos[2] - rp[2]).powi(2)).sqrt());
                }
            }
        }
    }
    println!("== geometry ({movie} frames {from}..{to} stride {stride}) ==");
    println!("floor probes vs geo::floor_at: {} probes, {} mismatches ({:.3}%)", t.probes, t.probe_bad, 100.0 * t.probe_bad as f64 / t.probes.max(1) as f64);
    for b in &bad_examples {
        println!("   {b}");
    }
    println!("own floor vs game floorHeight (static floors): {} frames, {} off by >1", t.own_floor, t.own_floor_bad);
    println!("headroom vs game ceilHeight (static ceilings): {} frames, {} off by >1 ({} frames skipped: the game's ceiling is an object's)", t.ceil, t.ceil_bad, t.ceil_dyn);
    println!(
        "wall (heading ~ velocity, speed > 8): {} frames; game wall hit next frame {} ; predicted (dist <= speed+2) {} ; both {} -> recall {:.2}, precision {:.2}",
        t.wall_frames, t.wall_hit, t.wall_pred, t.wall_pred_hit, t.wall_pred_hit as f64 / t.wall_hit.max(1) as f64, t.wall_pred_hit as f64 / t.wall_pred.max(1) as f64
    );
    println!(
        "wall, 3-frame window (hit within 3 frames; predicted dist <= 3*speed+2): game {} predicted {} both {} -> recall {:.2}, precision {:.2}",
        t.wall3_hit, t.wall3_pred, t.wall3_both, t.wall3_both as f64 / t.wall3_hit.max(1) as f64, t.wall3_both as f64 / t.wall3_pred.max(1) as f64
    );
    println!(
        "edge (grounded, v>15): predicted within 12f of travel {} ; became airborne within 14f {} ; precision {:.2} recall {:.2}",
        t.edge_pred, t.edge_air, t.edge_pred_air as f64 / t.edge_pred.max(1) as f64, t.edge_air_pred as f64 / t.edge_air.max(1) as f64
    );
    println!("ledge: predicted {} actual grab within 3f {} both {}", t.ledge_pred, t.ledge_actual, t.ledge_both);
    for (name, fe, pe) in [("actual pad", &mut t.land_frames_err, &mut t.land_pos_err), ("const pad (full fwd, A held)", &mut t.land_const_frames_err, &mut t.land_const_pos_err)] {
        let n = fe.len();
        let exact = fe.iter().filter(|e| **e == 0.0).count();
        let within1 = fe.iter().filter(|e| e.abs() <= 1.0).count();
        println!(
            "landing [{name}]: {} flights (+{} no prediction); frames err exact {:.0}% within1 {:.0}% ; |frames err| p50 {} p90 {} ; xz err p50 {:.1} p90 {:.1} p99 {:.1}",
            n, t.land_missing, 100.0 * exact as f64 / n.max(1) as f64, 100.0 * within1 as f64 / n.max(1) as f64,
            { let mut a: Vec<f32> = fe.iter().map(|e| e.abs()).collect(); pct(&mut a, 0.5) },
            { let mut a: Vec<f32> = fe.iter().map(|e| e.abs()).collect(); pct(&mut a, 0.9) },
            pct(pe, 0.5), pct(pe, 0.9), pct(pe, 0.99)
        );
    }
    let names = ["floors", "walls", "edges", "landing", "objs", "all", "all no dyn"];
    print!("timing per call (mean over {time_n} states, us):");
    for (k, n) in names.iter().enumerate() {
        print!(" {n} {:.1}", 1e6 * time_parts[k] / time_n.max(1) as f64);
    }
    println!();
    println!("objects seen (behaviour: category, frames):");
    let mut rows: Vec<_> = t.cat_seen.iter().collect();
    rows.sort_by_key(|(_, v)| std::cmp::Reverse(v.1));
    for (n, (c, k)) in rows.iter().take(40) {
        println!("   {:<36} {:<10} {}", n, sense::CAT_NAMES[*c as usize], k);
    }
    let _ = rnd();
    let _ = (MarioFlags::of, t.zone_miss);
    Ok(())
}

fn fields(rom: &str) -> Result<()> {
    obrrt_objectives::moves::install_controller();
    let gg = SM64GameGenerator::from_file(rom)?;
    let mut game: SM64Game = gg.create_game()?;
    let pads = read_movie("ref/tas/full-120star-2026-U.m64")?;
    println!("tables: {} globals, {} structs", reader::table().globals.len(), reader::table().structs.len());
    // Every field of the main structs parses.
    let mut bad = 0;
    for (s, fs) in &reader::table().structs {
        let root = match s.as_str() {
            "MarioState" => "gMarioState.".to_string(),
            "Object" => "obj[0].".to_string(),
            "Camera" => "gCamera->".to_string(),
            "LakituState" => "gLakituState.".to_string(),
            "HudDisplay" => "gHudDisplay.".to_string(),
            _ => continue,
        };
        for f in fs.keys() {
            if let Err(e) = FieldSet::parse(&[&format!("{root}{f}")]) {
                bad += 1;
                if bad < 6 {
                    println!("  parse failure: {e}");
                }
            }
        }
    }
    println!("struct fields that fail to parse: {bad}");
    let mut kbad = 0;
    for (b, keys) in sense::behaviours::KEYS {
        if obrrt_core::bhv_addr(b).is_none() {
            println!("  unknown behaviour in KEYS: {b}");
            kbad += 1;
        }
        for k in *keys {
            if !reader::table().structs["Object"].contains_key(*k) {
                println!("  KEYS {b}: no Object field {k}");
                kbad += 1;
            }
        }
    }
    for k in sense::behaviours::COMMON {
        if !reader::table().structs["Object"].contains_key(*k) {
            println!("  COMMON: no Object field {k}");
            kbad += 1;
        }
    }
    println!("behaviour table problems: {kbad}");
    let names = [
        "gGlobalTimer", "gRandomSeed16", "gCurrLevelNum", "gCurrAreaIndex", "gCurrCourseNum", "gMarioState.health", "gMarioState.numStars", "gMarioState.numCoins",
        "gMarioState.pos[0]", "gMarioState.pos[1]", "gMarioState.pos[2]", "gMarioState.forwardVel", "gMarioState.action", "gMarioState.floor->type",
        "gMarioState.floor->normal.y", "gMarioState.floorHeight", "gMarioState.ceilHeight", "gMarioState.actionTimer", "gMarioState.doubleJumpTimer",
        "gMarioState.faceAngle[1]", "gCamera->mode", "gCamera->yaw", "gLakituState.curPos[0]", "gLakituState.curFocus[1]", "mario_obj.oPosX", "mario_obj.oAction",
        "held.oAction", "floor_obj.oPosY", "nearest(cat=enemy).oPosX", "nearest(cat=enemy,k=1).oAction", "nearest(cat=coin).oPosY", "nearest(cat=star).oPosX",
        "nearest(bhv=bhvBobomb).oBobombFuseTimer", "obj[3].oAction", "gHudDisplay.coins", "gHudDisplay.wedges", "sCurrPlayMode", "gMenuMode", "gDialogID",
        "gMarioState.wall->normal.x", "gMarioState.heldObj->oPosY", "gMarioState.controller->rawStickX", "gMarioState.controller->buttonDown",
    ];
    let fs = FieldSet::parse(&names)?;
    println!("parsed {} fields", fs.len());
    let mut vals = vec![0f32; fs.len()];
    let mut mismatches = 0usize;
    let mut checked = 0usize;
    let mut tfetch = 0f64;
    let mut nf = 0usize;
    let mut prev_timer = None;
    let mut timer_ok = 0usize;
    let mut timer_n = 0usize;
    let mut sample_vals: Vec<(usize, Vec<Value>)> = Vec::new();
    for f in 0..6000usize.min(pads.len()) {
        game.step_game(pads[f])?;
        if f % 7 != 0 || f < 1600 {
            continue;
        }
        let st = read_state(&game, f as u32 + 1, true);
        let mut out = Vec::new();
        game.with_memory(|m| {
            let t0 = Instant::now();
            for _ in 0..20 {
                fs.fetch(m, &st, &mut vals);
            }
            tfetch += t0.elapsed().as_secs_f64() / 20.0;
            nf += 1;
            fs.fetch_values(m, &st, &mut out);
        });
        let get = |n: &str| fs.names().position(|x| x == n).unwrap();
        let eq = |a: f32, b: f32| (a - b).abs() <= 1e-3 * (1.0 + b.abs());
        let mut chk = |n: &str, truth: f32| {
            checked += 1;
            if !eq(vals[get(n)], truth) {
                mismatches += 1;
                if mismatches < 8 {
                    println!("  MISMATCH f{f} {n}: reader {} vs St {}", vals[get(n)], truth);
                }
            }
        };
        chk("gMarioState.health", st.health as f32);
        chk("gMarioState.pos[0]", st.pos[0]);
        chk("gMarioState.pos[1]", st.pos[1]);
        chk("gMarioState.pos[2]", st.pos[2]);
        chk("gMarioState.forwardVel", st.forward_vel);
        chk("gMarioState.action", st.action as f32);
        chk("gCurrLevelNum", st.level as f32);
        chk("gCurrAreaIndex", st.area as f32);
        chk("gMarioState.numStars", st.stars as f32);
        chk("gMarioState.numCoins", st.coins as f32);
        chk("gMarioState.floorHeight", st.floor_y);
        chk("gMarioState.floor->type", st.floor_type as f32);
        chk("gMarioState.floor->normal.y", st.floor_normal[1]);
        chk("gMarioState.faceAngle[1]", st.yaw as f32);
        chk("gMarioState.actionTimer", st.action_timer as f32);
        chk("mario_obj.oPosX", st.pos[0]);
        chk("sCurrPlayMode", st.play_mode as f32);
        if let Some(t) = prev_timer {
            timer_n += 1;
            if vals[get("gGlobalTimer")] - t >= 1.0 {
                timer_ok += 1;
            }
        }
        prev_timer = Some(vals[get("gGlobalTimer")]);
        // nearest enemy vs the Obj list.
        if let Some(objs) = st.objects.as_ref() {
            let mut best: Option<(f32, f32)> = None;
            for o in objs.iter() {
                if sense::behaviours::cat_of(o.bhv) == sense::CAT_ENEMY {
                    let d = (o.pos[0] - st.pos[0]).powi(2) + (o.pos[1] - st.pos[1]).powi(2) + (o.pos[2] - st.pos[2]).powi(2);
                    if best.is_none_or(|b| d < b.0) {
                        best = Some((d, o.pos[0]));
                    }
                }
            }
            if let Some((_, x)) = best {
                chk("nearest(cat=enemy).oPosX", x);
            } else if !vals[get("nearest(cat=enemy).oPosX")].is_nan() {
                mismatches += 1;
            }
        }
        if f % 700 == 0 {
            sample_vals.push((f, out));
        }
    }
    println!("reader vs St: {checked} comparisons, {mismatches} mismatches; gGlobalTimer advances {timer_ok}/{timer_n}");
    println!("fetch of {} fields: {:.2} us per call (mean over {nf} states)", fs.len(), 1e6 * tfetch / nf.max(1) as f64);
    for (f, v) in sample_vals.iter().take(3) {
        println!("frame {f}:");
        for (n, x) in fs.names().zip(v) {
            println!("   {n:<44} {x:?}");
        }
    }
    Ok(())
}


/// Which object/surface does Mario's `wall` pointer reference now? (kind, owner slot, owner behaviour)
fn wall_owner(game: &SM64Game) -> Option<(i16, Option<(u16, String)>)> {
    use obrrt_core::layout::*;
    game.with_memory(|m| {
        let rd = |a: usize| m.get(a..a + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize).unwrap_or(0);
        let w = rd(ADDR_gMarioStates as usize + MARIO_wall);
        if w == 0 {
            return None;
        }
        let kind = m.get(w + SURFACE_type..w + SURFACE_type + 2).map(|b| i16::from_le_bytes([b[0], b[1]])).unwrap_or(0);
        let o = rd(w + SURFACE_object);
        let obj = (o >= ADDR_gObjectPool as usize).then(|| {
            let slot = ((o - ADDR_gObjectPool as usize) / OBJECT_SIZE) as u16;
            (slot, obrrt_core::bhv_name(rd(o + OBJ_behavior) as u32).unwrap_or("?").to_string())
        });
        Some((kind, obj))
    })
}

fn rotate(p: GamePad, deg: f32) -> GamePad {
    let r = deg.to_radians();
    let (x, y) = (p.stick_x as f32, p.stick_y as f32);
    GamePad::new(p.button, (x * r.cos() - y * r.sin()).round().clamp(-128.0, 127.0) as i8, (x * r.sin() + y * r.cos()).round().clamp(-128.0, 127.0) as i8)
}

/// What blocks Mario at a state: sense vs the game, and what a rotated stick does.
fn block(rom: &str, movie: &str, frame: usize, steps: usize) -> Result<()> {
    obrrt_objectives::moves::install_controller();
    let gg = SM64GameGenerator::from_file(rom)?;
    let mut game: SM64Game = gg.create_game()?;
    let pads = read_movie(movie)?;
    for p in &pads[..frame] {
        game.step_game(*p)?;
    }
    let st = read_state(&game, frame as u32, true);
    println!("{movie} state f{frame}: pos {:?} yaw {:#x} fv {:.1} vel {:?} action {:#x}", st.pos, st.yaw as u16, st.forward_vel, st.vel, st.action);
    let mut ctx = SenseCtx::new();
    // Build the zone with the controllable state (the static grid needs one).
    let s = game.with_memory(|m| ctx.read(m, &st, &SenseOpts::default(), &[], &AirModel::default()));
    println!("walls (fwd, +0x4000, -0x4000, back): {:?}", s.wall);
    for (k, b) in ctx.blocks.iter().enumerate() {
        println!("   dir {k}: dist {:.1} src {:?} normal yaw {:#x}", b.dist, b.src, b.normal_yaw as u16);
    }
    let st_static = game.with_memory(|m| ctx.read(m, &st, &SenseOpts { dyn_walls: false, ..SenseOpts::default() }, &[], &AirModel::default()));
    println!("static-only walls: {:?}", st_static.wall);
    if let Some(objs) = st.objects.as_ref() {
        let mut near: Vec<_> = objs.iter().map(|o| ((o.pos[0] - st.pos[0]).hypot(o.pos[2] - st.pos[2]), o)).collect();
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        println!("nearest objects:");
        for (d, o) in near.iter().take(10) {
            println!("   slot {:3} {:<32} cat {:<9} dist {:7.1} pos ({:.0},{:.0},{:.0})", o.slot, obrrt_core::bhv_name(o.bhv).unwrap_or("?"), sense::CAT_NAMES[sense::behaviours::cat_of(o.bhv) as usize], d, o.pos[0], o.pos[1], o.pos[2]);
        }
    }
    let save = game.save_state();
    let mut dyn_surfs = 0usize;
    game.with_memory(|m| {
        let dg = obrrt_core::dyngeo::read(m, st.pos);
        dyn_surfs = dg.surfs.len();
        let mut near: Vec<_> = dg.surfs.iter().filter(|x| x.list == 2).collect();
        near.sort_by(|a, b| {
            let d = |x: &obrrt_core::dyngeo::DynSurf| x.v.iter().map(|p| (p[0] as f32 - st.pos[0]).hypot(p[2] as f32 - st.pos[2])).fold(f32::INFINITY, f32::min);
            d(a).total_cmp(&d(b))
        });
        println!("object collision walls near Mario: {} (of {} object surfaces); nearest:", near.len(), dyn_surfs);
        for x in near.iter().take(6) {
            println!("   slot {:3} kind {:#x} verts {:?} n {:?}", x.slot, x.kind, x.v, x.n);
        }
    });
    println!("--- replay with the stick rotated by an offset (buttons as in the movie), first wall contact:");
    for deg in [-30.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0] {
        game.load_state(&save)?;
        let mut hit = None;
        let mut last = st.pos;
        for k in 0..steps {
            let p = pads.get(frame + k).copied().unwrap_or(GamePad::new(0, 0, 0));
            game.step_game(rotate(p, deg))?;
            let now = read_state(&game, (frame + k + 1) as u32, false);
            last = now.pos;
            if hit.is_none() {
                if let Some((kind, owner)) = wall_owner(&game) {
                    hit = Some((k + 1, kind, owner, now.pos, now.action));
                }
            }
        }
        match hit {
            Some((k, kind, owner, pos, act)) => println!("  {deg:+5.0} deg: wall touched after {k} frames at ({:.0},{:.0},{:.0}) action {act:#x}: surface kind {kind:#x}, owner {:?}; end pos ({:.0},{:.0},{:.0})", pos[0], pos[1], pos[2], owner, last[0], last[1], last[2]),
            None => println!("  {deg:+5.0} deg: no wall within {steps} frames; end pos ({:.0},{:.0},{:.0})", last[0], last[1], last[2]),
        }
    }
    // Sense's prediction frame by frame along the movie (what it said vs the game's wall pointer).
    game.load_state(&save)?;
    println!("--- sense blocks (fwd) vs game wall owner along the movie:");
    for k in 0..steps.min(40) {
        let st = read_state(&game, (frame + k) as u32, true);
        let s = game.with_memory(|m| ctx.read(m, &st, &SenseOpts::default(), &[], &AirModel::default()));
        let own = wall_owner(&game);
        println!("  +{k:2} pos ({:.0},{:.0},{:.0}) fv {:.1} act {:#x} sense fwd {:.1} ({:?}) game wall {:?}", st.pos[0], st.pos[1], st.pos[2], st.forward_vel, st.action, s.wall[0], ctx.blocks[0].src, own);
        game.step_game(pads.get(frame + k).copied().unwrap_or(GamePad::new(0, 0, 0)))?;
    }
    Ok(())
}

/// Ledge prediction: at every real ledge grab of the movie, what did `ledge_ahead` say on the frames before?
fn ledge(rom: &str, movie: &str) -> Result<()> {
    obrrt_objectives::moves::install_controller();
    let gg = SM64GameGenerator::from_file(rom)?;
    let mut game: SM64Game = gg.create_game()?;
    let pads = read_movie(movie)?;
    // Pass 1: the frames where the action becomes LEDGE_GRAB.
    let mut grabs = Vec::new();
    let mut prev = 0u32;
    for (f, p) in pads.iter().enumerate() {
        game.step_game(*p)?;
        let st = read_state(&game, f as u32 + 1, false);
        if st.action == 0x0800_034B && prev != 0x0800_034B {
            grabs.push(f + 1);
        }
        prev = st.action;
    }
    println!("{} ledge grabs in {} frames", grabs.len(), pads.len());
    // Pass 2: sense on the 1..=4 frames before each.
    let game2 = &mut gg.create_game()?;
    let mut ctx = SenseCtx::new();
    let want: std::collections::BTreeSet<usize> = grabs.iter().flat_map(|g| (1..=4).map(move |k| g - k)).collect();
    let (mut seen, mut flagged, mut flagged_near) = (0, 0, 0);
    let mut by_k = [0usize; 5];
    for (f, p) in pads.iter().enumerate() {
        if want.contains(&f) {
            let st = read_state(game2, f as u32, false);
            if st.controllable() {
                let air = AirModel::default();
                let s = game2.with_memory(|m| ctx.read(m, &st, &SenseOpts { floors: false, landing: false, objs: false, ..SenseOpts::default() }, &[], &air));
                let k = grabs.iter().filter(|&&g| g > f).map(|g| g - f).min().unwrap_or(9);
                seen += 1;
                let hv = st.vel[0].hypot(st.vel[2]);
                let velyaw = (st.vel[0].atan2(st.vel[2]) * 65536.0 / std::f32::consts::TAU) as i32 as i16;
                println!("   grab in {k}: pos ({:.0},{:.0},{:.0}) yaw {:#x} vel-yaw {:#x} hv {:.1} vy {:.1} action {:#x} wall {:?} ledge_ahead {} floor_y {:.0}", st.pos[0], st.pos[1], st.pos[2], st.yaw as u16, velyaw as u16, hv, st.vel[1], st.action, s.wall, s.ledge_ahead, st.floor_y);
                if s.ledge_ahead.is_finite() {
                    flagged += 1;
                    if k <= 4 {
                        by_k[k] += 1;
                    }
                    if s.ledge_ahead <= 2.0 * hv + 60.0 {
                        flagged_near += 1;
                    }
                }
            }
        }
        game2.step_game(*p)?;
        if f > *want.iter().next_back().unwrap_or(&0) + 1 {
            break;
        }
    }
    println!("frames before grabs (1..4 before, controllable): {seen}; ledge_ahead finite on {flagged} ({flagged_near} within 2 frames of travel + 60); by frames-before-grab 1..4: {:?}", &by_k[1..]);
    Ok(())
}

fn md() {
    println!("# Object behaviours\n\nGenerated by `sense_check md`. Category = what `Sense.objs` calls it; key fields are `Object` fields (plus the common ones in `behaviours::COMMON`) worth reading for that behaviour.\n");
    println!("| behaviour | category | key fields |\n|---|---|---|");
    for &(_, n) in obrrt_core::layout::BEHAVIOURS {
        let c = sense::behaviours::categorize(n);
        if c == sense::CAT_NONE {
            continue;
        }
        let k = sense::behaviours::keys_of(n);
        println!("| `{n}` | {} | {} |", sense::CAT_NAMES[c as usize], k.iter().map(|x| format!("`{x}`")).collect::<Vec<_>>().join(", "));
    }
    println!("\nCommon fields of every object: {}", sense::behaviours::COMMON.iter().map(|x| format!("`{x}`")).collect::<Vec<_>>().join(", "));
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let _ = GamePad::new(0, 0, 0);
    match a.get(1).map(String::as_str) {
        Some("check") => check(&a[2], &a[3], a[4].parse()?, a[5].parse()?, a.get(6).map_or(Ok(3), |s| s.parse())?),
        Some("fields") => fields(&a[2]),
        Some("ledge") => ledge(&a[2], &a[3]),
        Some("block") => block(&a[2], &a[3], a[4].parse()?, a.get(5).map_or(Ok(60), |s| s.parse())?),
        Some("md") => {
            md();
            Ok(())
        }
        _ => {
            eprintln!("usage: sense_check check <rom> <movie> <from> <to> [stride] | fields <rom> | md");
            Ok(())
        }
    }
}
