//! `behavior`: a behaviour script from the decomp and where its native functions live.

use std::path::PathBuf;

use anyhow::{Result, bail};
use obrrt_core::bhv_addr;
use serde_json::{Value, json};

fn port() -> PathBuf {
    PathBuf::from(std::env::var("SM64_PORT").unwrap_or_else(|_| "/workspaces/sm64-combined/sm64-binds/sm64-port".into()))
}

/// The identifiers inside the call `KEY(...)` of a script line.
fn call_args<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let t = line.trim_start();
    let rest = t.strip_prefix(key)?.strip_prefix('(')?;
    Some(rest.split(')').next().unwrap_or(rest))
}

/// `file:line` of every definition of C function `f` (a line starting at column 0
/// that contains `f(` and doesn't end in `;`).
fn find_defs(files: &[(String, String)], f: &str) -> Vec<String> {
    let pat = format!("{f}(");
    let mut out = vec![];
    for (name, text) in files {
        for (i, line) in text.lines().enumerate() {
            if line.starts_with(|c: char| c.is_ascii_alphabetic()) && !line.trim_end().ends_with(';') {
                if let Some(pos) = line.find(&pat) {
                    let before = line[..pos].chars().last();
                    if before.is_some_and(|c| c == ' ' || c == '*') {
                        out.push(format!("{name}:{}", i + 1));
                    }
                }
            }
        }
    }
    out
}

pub fn behavior(name: &str) -> Result<Value> {
    let path = port().join("data/behavior_data.c");
    let text = std::fs::read_to_string(&path).map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
    let want = name.trim();
    let full = if want.to_lowercase().starts_with("bhv") { want.to_string() } else { format!("bhv{want}") };
    let lines: Vec<&str> = text.lines().collect();
    let header = |l: &str| l.strip_prefix("const BehaviorScript ").and_then(|r| r.split('[').next()).map(str::to_string);
    let found = lines.iter().position(|l| header(l).is_some_and(|n| n.eq_ignore_ascii_case(&full)));
    let Some(start) = found else {
        let key = want.to_lowercase();
        let key = key.strip_prefix("bhv").unwrap_or(&key);
        let near: Vec<String> = lines.iter().filter_map(|l| header(l)).filter(|n| n.to_lowercase().contains(key)).take(15).collect();
        bail!("no behaviour script named {full} in data/behavior_data.c{}", if near.is_empty() { String::new() } else { format!("; similar: {}", near.join(", ")) });
    };
    let end = (start..lines.len()).find(|&i| lines[i].starts_with("};")).unwrap_or(lines.len() - 1);
    let name = header(lines[start]).unwrap_or(full);
    let script = lines[start..=end].join("\n");
    let (mut natives, mut refs) = (vec![], vec![]);
    for l in &lines[start..=end] {
        if let Some(f) = call_args(l, "CALL_NATIVE") {
            natives.push(f.trim().to_string());
        }
        for k in ["CALL", "GOTO", "SPAWN_CHILD", "SPAWN_CHILD_WITH_PARAM", "SPAWN_OBJ", "BEGIN_REPEAT"] {
            if let Some(a) = call_args(l, k) {
                if let Some(b) = a.split(',').map(str::trim).find(|x| x.starts_with("bhv")) {
                    refs.push(format!("{k} {b}"));
                }
            }
        }
    }
    natives.dedup();
    let mut files = vec![];
    if !natives.is_empty() {
        for d in ["src/game/behaviors", "src/game", "src/engine"] {
            let Ok(rd) = std::fs::read_dir(port().join(d)) else { continue };
            let mut v: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "c")).collect();
            v.sort();
            for p in v {
                if let Ok(t) = std::fs::read_to_string(&p) {
                    files.push((p.strip_prefix(port()).unwrap_or(&p).display().to_string(), t));
                }
            }
        }
    }
    let natives_json: Vec<Value> = natives.iter().map(|f| json!({"fn": f, "defined": find_defs(&files, f)})).collect();
    Ok(json!({
        "name": name, "address": bhv_addr(&name).map(|a| format!("{a:#x}")), "file": "data/behavior_data.c", "line": [start + 1, end + 1],
        "script": script, "natives": natives_json, "refs": refs,
    }))
}
