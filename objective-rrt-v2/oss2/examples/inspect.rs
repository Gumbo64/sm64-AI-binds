//! `cargo run --release --example inspect -- runs/<run> <cell id>...`: replays each
//! cell's path from tree.bin from power-on and prints Mario's state at the end
//! and over the last frames (position, speed, action, dialog, floor). With
//! `OSS_INSPECT_STRIDE=n` it prints every n-th frame of the whole path from
//! frame 1600 (after the menus) instead.

use anyhow::Result;
use obrrt_core::read_state;
use sm64_binds::{GamePad, SM64GameGenerator};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let bin = std::fs::read(format!("{dir}/tree.bin"))?;
    let u32_at = |i: usize| u32::from_le_bytes(bin[i..i + 4].try_into().unwrap());
    let mut records: Vec<(i32, Vec<GamePad>)> = Vec::new();
    let mut i = 4;
    for _ in 0..u32_at(0) {
        let (parent, len) = (u32_at(i) as i32, u32_at(i + 4) as usize);
        let pads = bin[i + 8..i + 8 + 4 * len].chunks(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8));
        records.push((parent, pads.collect()));
        i += 8 + 4 * len;
    }
    // cells.jsonl is large: read the lines wanted only.
    let wanted: Vec<usize> = args[2..].iter().map(|s| s.parse().unwrap()).collect();
    let text = std::fs::read_to_string(format!("{dir}/cells.jsonl"))?;
    let lines: Vec<&str> = text.lines().collect();
    let stride: usize = std::env::var("OSS_INSPECT_STRIDE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let game_gen = SM64GameGenerator::new(oss2::run::rom()?)?;
    let objects = true;
    for id in wanted {
        let cell: serde_json::Value = serde_json::from_str(lines[id])?;
        let mut node = cell["node"].as_i64().unwrap() as i32;
        let mut parts = Vec::new();
        while node >= 0 {
            parts.push(&records[node as usize].1);
            node = records[node as usize].0;
        }
        let path: Vec<GamePad> = parts.into_iter().rev().flatten().copied().collect();
        let mut game = game_gen.create_game()?;
        println!("cell {id}: {} frames", path.len());
        for (t, pad) in path.iter().enumerate() {
            game.step_game(*pad)?;
            let tail = t + 1 == path.len() || (t + 1 + 12 >= path.len() && (path.len() - t) % 4 == 0);
            if (stride == 0 && tail) || (stride > 0 && t + 1 >= 1600 && ((t + 1) % stride == 0 || t + 1 == path.len())) {
                let s = read_state(&game, t as u32 + 1, objects);
                println!(
                    "  f{:5} lvl {:2} pos ({:7.0},{:6.0},{:7.0}) fwd {:6.1} vy {:5.1} yaw {:6} act {:#010x} dialog {:3} floor_y {:6.0} lakitu {}",
                    t + 1, s.level, s.pos[0], s.pos[1], s.pos[2], s.forward_vel, s.vel[1], s.yaw, s.action, s.dialog, s.floor_y,
                    s.objects.as_deref().is_some_and(|o| o.iter().any(|x| x.bhv_name() == "bhvCameraLakitu"))
                );
            }
        }
    }
    Ok(())
}
