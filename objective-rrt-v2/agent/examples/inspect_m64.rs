//! Test driver for the `inspect` module.
//!
//! inspect_m64 <m64> <frame> <cmd> [args...] [--json]
//!   cmds: state | progress | objects [near] [bhv] | geo [radius] | behavior <name> | level [all]
//!         | objectives [kind] | diff <frame2> | summary
//! ROM: $SM64_ROM or /workspaces/sm64-combined/sm64-AI-binds/baserom.us.z64.

#[path = "../src/inspect/mod.rs"]
#[allow(dead_code)]
mod inspect;

use anyhow::{Result, bail};
use inspect::View;
use obrrt_core::{GamePad, read_state};
use sm64_binds::SM64GameGenerator;

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let json = args.iter().any(|a| a == "--json");
    args.retain(|a| a != "--json");
    if args.len() < 3 {
        bail!("usage: inspect_m64 <m64> <frame> <cmd> [args...] [--json]");
    }
    let rom = std::env::var("SM64_ROM").unwrap_or_else(|_| "/workspaces/sm64-combined/sm64-AI-binds/baserom.us.z64".into());
    // `behavior` needs no game.
    if args[2] == "behavior" {
        let v = inspect::behavior(args.get(3).map_or("", |s| s))?;
        println!("{}", if json { serde_json::to_string_pretty(&v)? } else { inspect::render("behavior", &v) });
        return Ok(());
    }
    let mut game = SM64GameGenerator::from_file(&rom)?.create_game()?;
    let bytes = std::fs::read(&args[0])?;
    let bytes = if bytes.starts_with(b"M64\x1a") { &bytes[0x400..] } else { &bytes[..] };
    let pads: Vec<GamePad> = bytes.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect();
    let target: usize = args[1].parse()?;
    let frame2: Option<usize> = if args[2] == "diff" { Some(args.get(3).and_then(|s| s.parse().ok()).unwrap_or(target + 30)) } else { None };
    obrrt_core::check_layout(&game).ok();
    let _ = obrrt_objectives::setup::from_env();
    let mut snap = None;
    let end = frame2.unwrap_or(target).max(target);
    for (i, p) in pads.iter().take(end).enumerate() {
        if i == target && frame2.is_some() {
            snap = Some(read_state(&game, i as u32, true));
        }
        game.step_game(*p)?;
    }
    let st = read_state(&game, end as u32, true);
    let (a_st, b_st) = match (&snap, frame2) {
        (Some(a), Some(_)) => (a.clone(), st.clone()),
        _ => (st.clone(), st.clone()),
    };
    let v = View { game: &game, st: &st };
    let (cmd, rest) = (args[2].as_str(), &args[3..]);
    let value = match cmd {
        "state" => inspect::state(&v),
        "progress" => inspect::progress(&st),
        "objects" => inspect::objects(&v, rest.first().and_then(|s| s.parse().ok()), rest.get(1).map(|s| s.as_str())), // near "-" = none
        "geo" => inspect::geo(&v, rest.first().and_then(|s| s.parse().ok()).unwrap_or(500.0), None),
        "level" => inspect::level(&st, rest.first().is_some_and(|s| s == "all"))?,
        "objectives" => inspect::objectives(&st, rest.first().map(|s| s.as_str())),
        "diff" => inspect::diff(&View { game: &game, st: &a_st }, &View { game: &game, st: &b_st }),
        "summary" => {
            println!("{}", inspect::summary_line(&st));
            return Ok(());
        }
        other => bail!("unknown cmd {other}"),
    };
    println!("{}", if json { serde_json::to_string_pretty(&value)? } else { inspect::render(cmd, &value) });
    Ok(())
}
