//! Tries to play a savestate-start .m64 (an individual-level TAS) from states
//! of a power-on replay: plays `prefix` (an .m64 or raw replay) up to each
//! frame in `lo..hi`, then the movie's inputs, and reports the starts from
//! which the star count goes up. Mupen savestates themselves can't be loaded
//! into this emulator, so this looks for a start that behaves the same.
//!
//! cargo run --release --bin ilsync -- <rom> <prefix> <movie.m64> <lo> <hi> [out.csv]
//!
//! With `out.csv`, the best start's run (fewest frames to the star) is dumped
//! as the `tasplay` bin does, frames numbered from power-on.

use anyhow::{Result, bail};
use obrrt_core::{GamePad, read_state};
use sm64_binds::SM64GameGenerator;
use std::io::Write;

fn pads(path: &str) -> Result<Vec<GamePad>> {
    let bytes = std::fs::read(path)?;
    let body = if bytes.len() >= 0x400 && &bytes[..4] == b"M64\x1a" {
        let n = u32::from_le_bytes(bytes[0x18..0x1C].try_into().unwrap()) as usize;
        let data = &bytes[0x400..];
        &data[..(n * 4).min(data.len() / 4 * 4)]
    } else {
        &bytes[..bytes.len() / 4 * 4]
    };
    Ok(body.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        bail!("usage: ilsync <rom> <prefix> <movie.m64> <lo> <hi> [out.csv]");
    }
    let prefix = pads(&a[2])?;
    let movie = pads(&a[3])?;
    let (lo, hi): (usize, usize) = (a[4].parse()?, a[5].parse::<usize>()?.min(prefix.len()));
    let game_gen = SM64GameGenerator::from_file(&a[1])?;
    let mut game = game_gen.create_game()?;
    for p in &prefix[..lo] {
        game.step_game(*p)?;
    }
    let base_stars = read_state(&game, lo as u32, false).stars;
    let mut best: Option<(usize, usize)> = None;
    let mut hits = 0;
    for at in lo..hi {
        let save = game.save_state();
        let s0 = read_state(&game, at as u32, false);
        for (k, p) in movie.iter().enumerate() {
            game.step_game(*p)?;
            let s = read_state(&game, (at + k + 1) as u32, false);
            if s.stars > s0.stars {
                hits += 1;
                println!(
                    "start {at} (level {} area {} stars {}): star after {} movie frames at {:?}",
                    s0.level,
                    s0.area,
                    s0.stars,
                    k + 1,
                    s.pos
                );
                if best.is_none_or(|b| k + 1 < b.1) {
                    best = Some((at, k + 1));
                }
                break;
            }
        }
        game.load_state(&save)?;
        game.step_game(prefix[at])?;
    }
    println!("{hits} of {} starts reach a star (prefix stars {base_stars})", hi - lo);
    if let (Some((at, _)), Some(out)) = (best, a.get(6)) {
        let mut game = game_gen.create_game()?;
        let mut w = std::io::BufWriter::new(std::fs::File::create(out)?);
        writeln!(w, "frame,level,area,course,act,x,y,z,vy,fvel,hspeed,yaw,action,stars,keys,coins,health,lives,dialog,play_mode,controllable,held,buttons,sx,sy")?;
        let all: Vec<GamePad> = prefix[..at].iter().chain(movie.iter()).copied().collect();
        for (i, p) in all.iter().enumerate() {
            game.step_game(*p)?;
            let s = read_state(&game, i as u32 + 1, false);
            let hs = (s.vel[0] * s.vel[0] + s.vel[2] * s.vel[2]).sqrt();
            writeln!(
                w,
                "{},{},{},{},{},{:.1},{:.1},{:.1},{:.2},{:.2},{:.2},{},{:#x},{},{},{},{},{},{},{},{},{},0,0,0",
                i + 1, s.level, s.area, s.course, s.act, s.pos[0], s.pos[1], s.pos[2], s.vel[1], s.forward_vel, hs,
                s.yaw, s.action, s.stars, s.keys, s.coins, s.health, s.lives, s.dialog, s.play_mode,
                s.controllable() as u8, s.holding as u8
            )?;
        }
        println!("{out}: start {at}, {} frames", all.len());
    }
    Ok(())
}
