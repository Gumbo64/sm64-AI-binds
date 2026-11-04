//! Move usage and King Bob-omb returns along a raw route .m64 (from power-on),
//! per region of a regions.json (legs' output) and in total.
//!
//! route-stats <route.m64> [regions.json] [--json OUT] [--rom FILE]
//!
//! Moves: frames in each of the actions below (search/src/ga.rs MOVES plus
//! the slope and kick moves of objectives/src/moves), and how many regions
//! use a long jump (compare.py's "winners with a long jump" per leg).
//! King returns: entries into King Bob-omb's act 5 (jumping back after a
//! throw off his hill, king_bobomb.inc.c).

use anyhow::{Context, Result, bail};
use obrrt_core::{GamePad, read_state};
use serde_json::{Value, json};
use sm64_binds::SM64GameGenerator;

const MOVES: &[(&str, u32)] = &[
    ("long_jump", 0x03000888),
    ("dive", 0x0188088A),
    ("dive_slide", 0x00880456),
    ("rollout", 0x010008A6),
    ("wall_kick", 0x03000886),
    ("steep_jump", 0x03000885),
    ("triple_jump", 0x01000882),
    ("side_flip", 0x01000887),
    ("backflip", 0x01000883),
    ("jump_kick", 0x018008AC),
    ("slide_kick", 0x018008AA),
    ("slide_kick_slide", 0x0080045A),
    ("butt_slide", 0x00840452),
    ("stomach_slide", 0x008C0453),
    ("crouch_slide", 0x04808459),
    ("ground_pound", 0x008008A9),
];

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        bail!("usage: route-stats <route.m64> [regions.json] [--json OUT] [--rom FILE]");
    }
    let mut rom = format!("{}/../../baserom.us.z64", env!("CARGO_MANIFEST_DIR"));
    let (mut regions_path, mut out) = (None, None);
    let mut i = 2;
    while i < a.len() {
        match a[i].as_str() {
            "--json" => {
                out = a.get(i + 1).cloned();
                i += 2;
            }
            "--rom" => {
                rom = a[i + 1].clone();
                i += 2;
            }
            p => {
                regions_path = Some(p.to_string());
                i += 1;
            }
        }
    }
    let b = std::fs::read(&a[1]).with_context(|| a[1].clone())?;
    let body = if b.len() >= 0x400 && &b[..4] == b"M64\x1a" { &b[0x400..] } else { &b[..] };
    let pads: Vec<GamePad> = body.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect();
    let regions: Vec<(u32, u32, String)> = match &regions_path {
        Some(p) => {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(p)?)?;
            v.as_array()
                .map(|r| r.iter().map(|x| (x["from"].as_u64().unwrap_or(0) as u32, x["to"].as_u64().unwrap_or(0) as u32, x["label"].as_str().unwrap_or("").to_string())).collect())
                .unwrap_or_default()
        }
        None => Vec::new(),
    };
    let king = obrrt_core::bhv_addr("bhvKingBobomb").unwrap_or(0);
    let game_gen = SM64GameGenerator::from_file(&rom)?;
    let mut g = game_gen.create_game()?;
    let n = MOVES.len();
    let mut total = vec![0u32; n];
    let mut per: Vec<Vec<u32>> = vec![vec![0; n]; regions.len()];
    let mut returns_per = vec![0u32; regions.len()];
    let (mut returns, mut prev_act) = (0u32, -1i32);
    let mut throws = 0u32;
    for (f, p) in pads.iter().enumerate() {
        g.step_game(*p)?;
        let frame = f as u32 + 1;
        let objs = frame > 0 && pads.len() > 0;
        let s = read_state(&g, frame, objs);
        let ri = regions.iter().position(|r| frame > r.0 && frame <= r.1);
        if let Some(k) = MOVES.iter().position(|m| m.1 == s.action) {
            total[k] += 1;
            if let Some(ri) = ri {
                per[ri][k] += 1;
            }
        }
        let act = s.objects().iter().find(|o| o.bhv == king).map_or(-1, |o| o.action);
        if act == 5 && prev_act != 5 {
            returns += 1;
            if let Some(ri) = ri {
                returns_per[ri] += 1;
            }
        }
        if act == 4 && prev_act != 4 {
            throws += 1;
        }
        prev_act = act;
    }
    let mv = |v: &[u32]| -> Value { json!(MOVES.iter().zip(v).filter(|(_, c)| **c > 0).map(|(m, c)| (m.0.to_string(), json!(c))).collect::<serde_json::Map<_, _>>()) };
    let with_lj = per.iter().filter(|v| v[0] > 0).count();
    let rep = json!({
        "m64": a[1], "frames": pads.len(), "moves": mv(&total), "king_throws": throws, "king_returns": returns,
        "regions_with_long_jump": format!("{with_lj}/{}", regions.len()),
        "regions": regions.iter().enumerate().map(|(i, r)| json!({"label": r.2, "from": r.0, "to": r.1, "moves": mv(&per[i]), "king_returns": returns_per[i]})).collect::<Vec<_>>(),
    });
    println!(
        "{}: {} frames; moves {}; King throws {throws}, returns {returns}; regions with a long jump {with_lj}/{}",
        a[1],
        pads.len(),
        rep["moves"],
        regions.len()
    );
    for r in rep["regions"].as_array().unwrap() {
        println!("  {:<12} {:>5}-{:<5} {} {}", r["label"].as_str().unwrap_or(""), r["from"], r["to"], r["moves"], if r["king_returns"].as_u64().unwrap_or(0) > 0 { format!("King returns {}", r["king_returns"]) } else { String::new() });
    }
    if let Some(o) = out {
        std::fs::write(o, serde_json::to_string_pretty(&rep)?)?;
    }
    Ok(())
}
