//! Small JSON readers that carry the JSON path, so a bad spec says where
//! (`$.phases[1].done.dist: expected a number or a range, got "near"`).

use anyhow::{Result, anyhow, bail};
use serde_json::{Map, Value};

/// A value and where it is in the document.
#[derive(Clone, Copy)]
pub struct J<'a> {
    pub v: &'a Value,
    pub path: &'a str,
}

pub fn err<T>(path: &str, msg: impl std::fmt::Display) -> Result<T> {
    bail!("{path}: {msg}")
}

fn show(v: &Value) -> String {
    let s = v.to_string();
    if s.len() > 60 { format!("{}...", &s[..57]) } else { s }
}

impl<'a> J<'a> {
    pub fn new(v: &'a Value, path: &'a str) -> J<'a> {
        J { v, path }
    }

    pub fn obj(&self) -> Result<&'a Map<String, Value>> {
        self.v.as_object().ok_or_else(|| anyhow!("{}: expected an object, got {}", self.path, show(self.v)))
    }

    pub fn arr(&self) -> Result<&'a Vec<Value>> {
        self.v.as_array().ok_or_else(|| anyhow!("{}: expected an array, got {}", self.path, show(self.v)))
    }

    pub fn f64(&self) -> Result<f64> {
        if let Some(x) = self.v.as_f64() {
            return Ok(x);
        }
        // Hex strings for yaws, actions, fields.
        if let Some(s) = self.v.as_str() {
            if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                if let Ok(x) = i64::from_str_radix(h, 16) {
                    return Ok(x as f64);
                }
            }
        }
        err(self.path, format!("expected a number, got {}", show(self.v)))
    }

    pub fn f32(&self) -> Result<f32> {
        Ok(self.f64()? as f32)
    }

    pub fn u32(&self) -> Result<u32> {
        let x = self.f64()?;
        if x < 0.0 || x.fract() != 0.0 || x > u32::MAX as f64 {
            return err(self.path, format!("expected a non-negative integer, got {}", show(self.v)));
        }
        Ok(x as u32)
    }

    pub fn bool(&self) -> Result<bool> {
        self.v.as_bool().ok_or_else(|| anyhow!("{}: expected true or false, got {}", self.path, show(self.v)))
    }

    pub fn str(&self) -> Result<&'a str> {
        self.v.as_str().ok_or_else(|| anyhow!("{}: expected a string, got {}", self.path, show(self.v)))
    }

    pub fn vec3(&self) -> Result<[f32; 3]> {
        let a = self.arr()?;
        if a.len() != 3 {
            return err(self.path, format!("expected [x, y, z], got {}", show(self.v)));
        }
        let mut out = [0.0; 3];
        for (i, x) in a.iter().enumerate() {
            let p = format!("{}[{i}]", self.path);
            out[i] = J::new(x, &p).f32()?;
        }
        Ok(out)
    }
}

/// Checks that an object has only `allowed` keys (plus free-text `doc`,
/// `comment`, `notes`), naming the unknown one and the allowed ones.
pub fn check_keys(m: &Map<String, Value>, path: &str, allowed: &[&str]) -> Result<()> {
    for k in m.keys() {
        if matches!(k.as_str(), "doc" | "comment" | "notes") || allowed.contains(&k.as_str()) {
            continue;
        }
        let near: Vec<&&str> = allowed.iter().filter(|a| similar(a, k)).collect();
        let hint = if near.is_empty() { String::new() } else { format!(" (did you mean {}?)", near.iter().map(|s| format!("\"{s}\"")).collect::<Vec<_>>().join(" or ")) };
        bail!("{path}: unknown key \"{k}\"{hint}; allowed here: {}", allowed.join(", "));
    }
    Ok(())
}

/// Loose similarity for "did you mean": one is a prefix/substring of the
/// other, or they differ by at most 2 edits.
fn similar(a: &str, b: &str) -> bool {
    if a.contains(b) || b.contains(a) {
        return true;
    }
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + (a[i - 1] != b[j - 1]) as usize);
        }
        prev = cur;
    }
    prev[b.len()] <= 2
}

/// `parent.key` / `parent[i]` path strings.
pub fn key(path: &str, k: &str) -> String {
    format!("{path}.{k}")
}

pub fn idx(path: &str, i: usize) -> String {
    format!("{path}[{i}]")
}
