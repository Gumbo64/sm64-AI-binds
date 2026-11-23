//! Small shared helpers: movies, start sources, JSON files.

use anyhow::{Context, Result, anyhow, bail};
use obrrt_core::GamePad;
use serde_json::Value;
use std::io::Write;
use std::path::Path;

/// The objective-rrt-v2 directory (relative paths in sessions resolve here).
pub const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
pub const DEFAULT_ROM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../baserom.us.z64");
/// obrrt2's OBRRT_TAS_BOOT root: the 2026 TAS's first 1646 frames (file
/// select and the intro, ending on the castle grounds in control).
const TAS_BOOT_FRAMES: usize = 1646;
const TAS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../ref/tas/full-120star-2026-U.m64");

pub fn read_m64(path: &str) -> Result<Vec<GamePad>> {
    let b = std::fs::read(path).with_context(|| path.to_string())?;
    let body = if b.len() >= 0x400 && &b[..4] == b"M64\x1a" { &b[0x400..] } else { &b[..] };
    Ok(pads_from_bytes(body))
}

pub fn pads_from_bytes(b: &[u8]) -> Vec<GamePad> {
    b.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect()
}

pub fn pads_to_bytes(pads: &[GamePad]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pads.len() * 4);
    for p in pads {
        out.extend_from_slice(&p.button.to_be_bytes());
        out.push(p.stick_x as u8);
        out.push(p.stick_y as u8);
    }
    out
}

/// Raw, headerless, 4 bytes a frame (legs' and nodepath's format; the viewer reads it).
pub fn write_m64(path: &str, pads: &[GamePad]) -> Result<()> {
    write_atomic(path, &pads_to_bytes(pads))
}

/// Inputs from power-on to an obrrt2 node (nodepath's walk).
pub fn node_inputs(run: &str, node: u32) -> Result<Vec<GamePad>> {
    let rec = obrrt_search::route::read_tree_bin(&format!("{run}/tree.bin"))?;
    let mut chain = vec![node];
    while let Some(&n) = chain.last() {
        let p = rec.get(n as usize + 1).ok_or_else(|| anyhow!("{run}: no node {n}"))?.0;
        if p < 0 {
            break;
        }
        chain.push(p as u32);
    }
    let mut out = rec[0].1.clone();
    for n in chain.iter().rev() {
        out.extend_from_slice(&rec[*n as usize + 1].1);
    }
    Ok(out)
}

pub fn resolve(p: &str) -> String {
    if p.starts_with('/') { p.to_string() } else { format!("{ROOT}/{p}") }
}

/// A start: {"boot": "tas"}, {"run": dir, "node": id}, {"m64": file, "frames": n}.
pub fn source_inputs(v: &Value) -> Result<(Vec<GamePad>, String)> {
    if v["boot"] == "own" || v.get("boot").is_none() && v.get("run").is_none() && v.get("m64").is_none() {
        return Ok((own_boot()?, "boot: own (obrrt2's fixed menu-skip inputs, then idle until Mario stands still; no replay inputs)".into()));
    }
    if v.get("boot").is_some() {
        let mut t = read_m64(TAS)?;
        t.truncate(TAS_BOOT_FRAMES);
        return Ok((t, "boot: tas (file select and intro only)".into()));
    }
    if let Some(r) = v.get("run").and_then(|x| x.as_str()) {
        let node = v.get("node").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
        return Ok((node_inputs(&resolve(r), node)?, format!("run {r} node {node}")));
    }
    if let Some(m) = v.get("m64").and_then(|x| x.as_str()) {
        let mut t = read_m64(&resolve(m))?;
        if let Some(n) = v.get("frames").and_then(|x| x.as_u64()) {
            t.truncate(n as usize);
        }
        return Ok((t.clone(), format!("m64 {m} ({} frames)", t.len())));
    }
    bail!("start: expected {{\"boot\": \"tas\"}}, {{\"run\": dir, \"node\": id}} or {{\"m64\": file, \"frames\": n}}")
}

/// obrrt2's fixed boot (run/src/main.rs `boot_inputs`): A on alternate
/// frames with Start at 150-160 and 200-300, 1680 frames; then idle frames
/// until Mario is controllable and has stood still for 10 frames.
pub fn own_boot() -> Result<Vec<GamePad>> {
    use obrrt_core::buttons::{A_BUTTON, START_BUTTON};
    let mut pads: Vec<GamePad> = (0..1680)
        .map(|i| {
            let mut b = 0;
            if (150..160).contains(&i) || (200..300).contains(&i) {
                b = START_BUTTON;
            }
            if i % 2 == 0 {
                b = A_BUTTON;
            }
            GamePad::new(b, 0, 0)
        })
        .collect();
    let mut game = sm64_binds::SM64GameGenerator::from_file(DEFAULT_ROM)?.create_game()?;
    for p in &pads {
        game.step_game(*p)?;
    }
    let mut st = obrrt_core::read_state(&game, pads.len() as u32, false);
    let mut still = 0;
    while still < 10 || !st.controllable() {
        let before = st.pos;
        let idle = GamePad::new(0, 0, 0);
        game.step_game(idle)?;
        pads.push(idle);
        st = obrrt_core::read_state(&game, pads.len() as u32, false);
        still = if st.pos == before { still + 1 } else { 0 };
        anyhow::ensure!(pads.len() < 32_000, "own boot: Mario didn't stand still");
    }
    Ok(pads)
}

pub fn write_atomic(path: impl AsRef<Path>, bytes: &[u8]) -> Result<()> {
    let path = path.as_ref();
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&tmp, bytes).with_context(|| tmp.display().to_string())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn write_json(path: impl AsRef<Path>, v: &Value) -> Result<()> {
    let mut s = serde_json::to_string_pretty(v)?;
    s.push('\n');
    write_atomic(path, s.as_bytes())
}

pub fn read_json(path: impl AsRef<Path>) -> Result<Value> {
    let path = path.as_ref();
    let s = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
    Ok(serde_json::from_str(&s).with_context(|| path.display().to_string())?)
}

pub fn append_jsonl(path: impl AsRef<Path>, v: &Value) -> Result<()> {
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path.as_ref())?;
    let mut s = serde_json::to_string(v)?;
    s.push('\n');
    f.write_all(s.as_bytes())?;
    Ok(())
}

pub fn read_jsonl(path: impl AsRef<Path>) -> Result<Vec<Value>> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(Vec::new());
    }
    let s = std::fs::read_to_string(path)?;
    s.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).with_context(|| format!("{}: {l}", path.display()))).collect()
}

pub fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

/// A process is alive (signal 0).
pub fn pid_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}
