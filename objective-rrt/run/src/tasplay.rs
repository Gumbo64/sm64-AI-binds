//! Replays a Mupen64 .m64 (or one of our raw replays) and writes one CSV row
//! per input sample: where Mario is, what he's doing and the run's progress.
//!
//! cargo run --release --bin tasplay -- <rom> <movie.m64> <out.csv> [max samples]
//!
//! Input is treated as untrusted data. A file starting with `M64\x1a` is a
//! Mupen64 movie: a 0x400-byte header (sample count at 0x18, controller
//! count at 0x15, start type at 0x1C), then 4 bytes per controller poll
//! (big-endian buttons, stick x, stick y). Anything else is our raw format,
//! the same 4 bytes per frame with no header. One sample is one game frame
//! here: SM64 polls the controller once per game frame, and lag frames
//! without a poll have no sample.

use anyhow::{Result, bail};
use obrrt_core::{GamePad, check_layout, read_state};
use sm64_binds::SM64GameGenerator;
use std::io::Write;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        bail!("usage: tasplay <rom> <movie.m64> <out.csv> [max samples]");
    }
    let bytes = std::fs::read(&args[2])?;
    let body = if bytes.len() >= 0x400 && &bytes[..4] == b"M64\x1a" {
        let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let samples = u32_at(0x18) as usize;
        let controllers = bytes[0x15];
        let start = u16::from_le_bytes([bytes[0x1C], bytes[0x1D]]);
        let rom_name = String::from_utf8_lossy(&bytes[0xC4..0xE4]).trim_end_matches('\0').to_string();
        let country = bytes[0xE8] as char;
        eprintln!(
            "m64 v{} samples {samples} controllers {controllers} start {start} rom {rom_name:?} country {country:?} crc {:08x}",
            u32_at(4),
            u32_at(0xE4)
        );
        if start != 2 {
            bail!("movie does not start from power-on (start type {start})");
        }
        if controllers != 1 {
            eprintln!("warning: {controllers} controllers; reading controller 1 only is not supported, samples interleave");
        }
        let data = &bytes[0x400..];
        // Some movies carry input past the header's count; play only the count.
        &data[..(samples * 4).min(data.len() / 4 * 4)]
    } else {
        &bytes[..bytes.len() / 4 * 4]
    };
    let pads: Vec<(u16, i8, i8)> =
        body.chunks_exact(4).map(|c| (u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect();
    let n = args.get(4).map_or(pads.len(), |s| s.parse::<usize>().unwrap().min(pads.len()));

    let game_gen = SM64GameGenerator::from_file(&args[1])?;
    let mut game = game_gen.create_game()?;
    let mut checked = false;
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[3])?);
    writeln!(
        out,
        "frame,level,area,course,act,x,y,z,vy,fvel,hspeed,yaw,action,stars,keys,coins,health,lives,dialog,play_mode,controllable,held,buttons,sx,sy"
    )?;
    for (i, &(b, x, y)) in pads[..n].iter().enumerate() {
        game.step_game(GamePad::new(b, x, y))?;
        let s = read_state(&game, i as u32 + 1, false);
        // The layout check needs Mario's object, which exists once a level loads.
        if !checked && s.level != 0 && s.action != 0 {
            check_layout(&game)?;
            checked = true;
        }
        let hs = (s.vel[0] * s.vel[0] + s.vel[2] * s.vel[2]).sqrt();
        writeln!(
            out,
            "{},{},{},{},{},{:.1},{:.1},{:.1},{:.2},{:.2},{:.2},{},{:#x},{},{},{},{},{},{},{},{},{},{:#06x},{},{}",
            i + 1,
            s.level,
            s.area,
            s.course,
            s.act,
            s.pos[0],
            s.pos[1],
            s.pos[2],
            s.vel[1],
            s.forward_vel,
            hs,
            s.yaw,
            s.action,
            s.stars,
            s.keys,
            s.coins,
            s.health,
            s.lives,
            s.dialog,
            s.play_mode,
            s.controllable() as u8,
            s.holding as u8,
            b,
            x,
            y
        )?;
    }
    eprintln!("{}: {n} frames", args[3]);
    Ok(())
}
