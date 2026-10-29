//! `cargo run --release --example verify -- runs/zero_star [n]`: replays n
//! cells' paths from tree.bin from power-on and checks their keys.

use anyhow::{Result, bail};
use objective_scattershot::objective::Tree;
use obrrt_core::read_state;
use sm64_binds::{GamePad, SM64GameGenerator};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let n: usize = args.get(2).map_or(Ok(3), |s| s.parse())?;
    let tree = Tree::load(&std::fs::read_to_string(format!("{dir}/schema.json"))?)?;

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

    let cells: Vec<serde_json::Value> = std::fs::read_to_string(format!("{dir}/cells.jsonl"))?.lines().map(serde_json::from_str).collect::<Result<_, _>>()?;
    let game_gen = SM64GameGenerator::new(std::fs::read("../baserom.us.z64")?)?;
    for k in 0..n {
        let cell = &cells[(k + 1) * cells.len() / (n + 1)];
        let mut node = cell["node"].as_i64().unwrap() as i32;
        let mut parts = Vec::new();
        while node >= 0 {
            parts.push(&records[node as usize].1);
            node = records[node as usize].0;
        }
        let path: Vec<GamePad> = parts.into_iter().rev().flatten().copied().collect();
        let mut game = game_gen.create_game()?;
        for pad in &path {
            game.step_game(*pad)?;
        }
        let key: Vec<[i64; 3]> = tree.key(&read_state(&game, path.len() as u32, tree.objects)).iter().map(|&(o, f, v)| [o as i64, f as i64, v as i64]).collect();
        let want: Vec<[i64; 3]> = serde_json::from_value(cell["key"].clone())?;
        println!("cell {} frames {} path {}: {}", cell["id"], cell["frames"], path.len(), if key == want { "ok" } else { "MISMATCH" });
        if key != want || path.len() as u64 != cell["frames"].as_u64().unwrap() {
            bail!("cell {}: replayed key {key:?}, archived {want:?}", cell["id"]);
        }
    }
    Ok(())
}
