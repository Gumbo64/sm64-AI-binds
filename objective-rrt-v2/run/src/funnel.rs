//! Funnel experiments (plan "new theoretical avenues", avenue 1 and 2): where
//! the game collapses state (a level entry, a star's exit), can one route's
//! suffix be played after another route's prefix?
//!
//! funnel census <rom> <out.csv> <movie>...
//!     One row per zone entry of each movie: the first controllable,
//!     non-modal frame after a (level, area) change, with everything that
//!     could differ between routes there (Mario, camera, RNG, global timer,
//!     a hash of the objects).
//! funnel transplant <rom> <A> <fa> <B> <fb> [cam=1] [limit]
//!     Plays A's first `fa` inputs, then B's inputs from index `fb` on (the
//!     stick turned by B's camera minus ours each frame when cam=1), and
//!     reports whether B's next event (a star or a zone change) arrives, when, and how far Mario strays
//!     from B's own path on the way.
//!
//! Movies: raw (4 bytes a frame from power-on) or Mupen .m64. Read only.

use anyhow::{Result, bail};
use obrrt_core::layout::{ADDR_gGlobalTimer, ADDR_gRandomSeed16};
use obrrt_core::{GamePad, St, read_state};
use sm64_binds::{SM64Game, SM64GameGenerator};
use std::hash::{Hash, Hasher};
use std::io::Write;

fn movie(path: &str) -> Result<Vec<GamePad>> {
    let bytes = std::fs::read(path)?;
    let body = if bytes.len() >= 0x400 && &bytes[..4] == b"M64\x1a" {
        let n = u32::from_le_bytes(bytes[0x18..0x1c].try_into().unwrap()) as usize;
        let d = &bytes[0x400..];
        &d[..(n * 4).min(d.len() / 4 * 4)]
    } else {
        &bytes[..bytes.len() / 4 * 4]
    };
    Ok(body.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect())
}

fn timers(game: &SM64Game) -> (u16, u32) {
    game.with_memory(|m| {
        let (r, g) = (ADDR_gRandomSeed16 as usize, ADDR_gGlobalTimer as usize);
        (u16::from_le_bytes([m[r], m[r + 1]]), u32::from_le_bytes(m[g..g + 4].try_into().unwrap()))
    })
}

/// (count, hash) of the loaded objects: behaviour, position to the unit, action.
fn objects(s: &St) -> (usize, u64) {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let Some(o) = s.objects.as_ref() else { return (0, 0) };
    let mut v: Vec<(u32, [i32; 3], i32)> = o.iter().map(|o| (o.bhv, o.pos.map(|x| x.round() as i32), o.action)).collect();
    v.sort();
    v.hash(&mut h);
    (v.len(), h.finish())
}

fn census(rom: &str, out: &str, movies: &[String]) -> Result<()> {
    let game_gen = SM64GameGenerator::from_file(rom)?;
    let mut f = std::io::BufWriter::new(std::fs::File::create(out)?);
    writeln!(f, "movie,frame,changed,from,zone,x,y,z,vx,vy,vz,fvel,yaw,action,cam_yaw,rng,gtimer,stars,star_set,coins,health,save_flags,nobj,objhash,course,star,keys,act")?;
    for path in movies {
        let pads = movie(path)?;
        let mut game = game_gen.create_game()?;
        let (mut zone, mut from, mut changed, mut pending) = ((0u16, 0u8), (0u16, 0u8), 0usize, false);
        let mut stars = 0;
        let mut had = [0u8; 26];
        for (i, p) in pads.iter().enumerate() {
            game.step_game(*p)?;
            let s = read_state(&game, i as u32 + 1, false);
            // A star: a row of its own (zone "star"), the set as it is after it.
            if s.stars > stars {
                stars = s.stars;
                let name = std::path::Path::new(path).file_name().unwrap().to_string_lossy();
                // Which star: the course (1-25; 26 = the castle's own) and its 1-based index.
                let now = s.collected();
                let (c, b) = (0..26).find_map(|c| (now[c] & !had[c] != 0).then(|| (c + 1, (now[c] & !had[c]).trailing_zeros() + 1))).unwrap_or((0, 0));
                had = now;
                writeln!(f, "{name},{},{},{}:{},star,{:.2},{:.2},{:.2},0,0,0,0,0,{:#x},0,0,0,{},{:x},{},{},{:#x},0,0,{c},{b},{},{}", i + 1, i + 1, zone.0, zone.1, s.pos[0], s.pos[1], s.pos[2], s.action, s.stars, s.progress_set(), s.coins, s.health, s.save_flags, s.keys, s.act)?;
            }
            if s.zone() != zone && s.level != 0 {
                (from, zone, changed, pending) = (zone, s.zone(), i + 1, true);
            }
            if pending && s.controllable() && !s.modal() && !s.in_dialog() {
                pending = false;
                let s = read_state(&game, i as u32 + 1, true);
                let (rng, gt) = timers(&game);
                let (n, h) = objects(&s);
                let name = std::path::Path::new(path).file_name().unwrap().to_string_lossy();
                writeln!(
                    f,
                    "{name},{},{changed},{}:{},{}:{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{:#x},{},{rng},{gt},{},{:x},{},{},{:#x},{n},{h:016x},0,0,{},{}",
                    i + 1, from.0, from.1, zone.0, zone.1, s.pos[0], s.pos[1], s.pos[2], s.vel[0], s.vel[1], s.vel[2], s.forward_vel, s.yaw, s.action,
                    s.cam_yaw, s.stars, s.progress_set(), s.coins, s.health, s.save_flags, s.keys, s.act
                )?;
            }
        }
    }
    Ok(())
}

fn transplant(rom: &str, a: &str, fa: usize, b: &str, fb: usize, cam: bool, limit: usize) -> Result<()> {
    let game_gen = SM64GameGenerator::from_file(rom)?;
    let (pa, pb) = (movie(a)?, movie(b)?);
    if fa > pa.len() || fb >= pb.len() {
        bail!("fa {fa} of {} or fb {fb} of {} out of range", pa.len(), pb.len());
    }
    // B alone: its camera before each input, its path, and its next star after fb.
    let mut game = game_gen.create_game()?;
    let mut ref_states: Vec<St> = Vec::with_capacity(pb.len() + 1);
    let mut s0 = read_state(&game, 0, false);
    ref_states.push(s0.clone());
    let mut target = None;
    for (i, p) in pb.iter().enumerate() {
        game.step_game(*p)?;
        s0 = read_state(&game, i as u32 + 1, false);
        if i + 1 > fb && target.is_none() && (s0.stars > ref_states[fb].stars || (s0.zone() != ref_states[fb].zone() && s0.level != 0)) {
            target = Some(i + 1);
        }
        ref_states.push(s0.clone());
        if target.is_some_and(|t| i + 1 >= t) {
            break;
        }
    }
    let end = target.unwrap_or(ref_states.len() - 1).min(fb + limit);
    // A's prefix, then B's inputs.
    let mut game = game_gen.create_game()?;
    for p in &pa[..fa] {
        game.step_game(*p)?;
    }
    let start = read_state(&game, fa as u32, false);
    let (rng_a, gt_a) = timers(&game);
    let r = &ref_states[fb];
    let d0 = obrrt_core::dist(start.pos, r.pos);
    let mut ours = start.cam_yaw;
    let (mut first_off, mut max_d, mut got, mut frames) = (None, 0f32, None, 0usize);
    // Up to 120 frames past B's own time, in case the star comes a little later.
    for k in fb..(end + 120).min(pb.len()) {
        let mut q = pb[k];
        if cam && k < ref_states.len() {
            let d = (ref_states[k].cam_yaw as i32 - ours as i32) as i16;
            if d != 0 && (q.stick_x != 0 || q.stick_y != 0) {
                let (x, y) = (q.stick_x as f32, q.stick_y as f32);
                let ang = x.atan2(-y) + d as f32 * std::f32::consts::TAU / 65536.0;
                let m = x.hypot(y);
                q = GamePad::new(q.button, (m * ang.sin()).round().clamp(-127.0, 127.0) as i8, (-m * ang.cos()).round().clamp(-127.0, 127.0) as i8);
            }
        }
        game.step_game(q)?;
        frames += 1;
        let s = read_state(&game, (fa + frames) as u32, false);
        ours = s.cam_yaw;
        if let Some(rs) = ref_states.get(k + 1) {
            let d = if s.zone() == rs.zone() { obrrt_core::dist(s.pos, rs.pos) } else { 1e9 };
            max_d = max_d.max(d);
            if d > 50.0 && first_off.is_none() {
                first_off = Some(frames);
            }
        }
        if s.stars > start.stars || (s.zone() != start.zone() && s.level != 0) {
            let e = &ref_states[end.min(ref_states.len() - 1)];
            got = Some((frames, s.zone() == e.zone() && s.stars - start.stars == e.stars - r.stars));
            break;
        }
    }
    println!(
        "{{\"a\":\"{a}\",\"fa\":{fa},\"b\":\"{b}\",\"fb\":{fb},\"cam\":{cam},\"start_dist\":{d0:.2},\"cam_diff\":{},\"rng_a\":{rng_a},\"gt_a\":{gt_a},\"ref_frames\":{},\"star\":{},\"frames\":{},\"same_set\":{},\"first_off\":{},\"max_dist\":{:.0}}}",
        (r.cam_yaw as i32 - start.cam_yaw as i32) as i16,
        target.map_or(-1, |t| (t - fb) as i64),
        got.is_some(),
        got.map_or(-1, |g| g.0 as i64),
        got.is_some_and(|g| g.1),
        first_off.map_or(-1, |x| x as i64),
        max_d.min(99999.0)
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("census") if args.len() >= 5 => census(&args[2], &args[3], &args[4..]),
        Some("transplant") if args.len() >= 7 => transplant(
            &args[2],
            &args[3],
            args[4].parse()?,
            &args[5],
            args[6].parse()?,
            args.get(7).is_none_or(|v| v != "0"),
            args.get(8).map_or(100_000, |v| v.parse().unwrap()),
        ),
        _ => bail!("usage: funnel census <rom> <out.csv> <movie>... | funnel transplant <rom> <A> <fa> <B> <fb> [cam=1] [limit]"),
    }
}
