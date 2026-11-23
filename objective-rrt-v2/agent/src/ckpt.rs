//! Sparse checkpoints: save states kept as the 4 KiB pages that differ from
//! a per-zone base, XORed and LZ4-compressed (search/src/store.rs's scheme,
//! with a file format). A checkpoint is owned by the segment (commit id)
//! whose inputs reach it; it is valid on every path through that segment.

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sm64_binds::SaveState;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::util::{append_jsonl, read_jsonl, write_atomic};

const PAGE: usize = 4096;
const MAGIC: &[u8; 4] = b"ARCK";

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Entry {
    pub frame: u32,
    pub owner: String,
    pub file: String,
    pub zone: (u16, u8),
}

pub struct Store {
    dir: PathBuf,
    /// Zone -> base index; base bytes loaded lazily.
    zones: HashMap<(u16, u8), u16>,
    bases: HashMap<u16, Vec<u8>>,
    pub entries: Vec<Entry>,
}

impl Store {
    pub fn open(session: &std::path::Path) -> Result<Store> {
        let dir = session.join("ckpt");
        std::fs::create_dir_all(dir.join("bases"))?;
        let mut zones = HashMap::new();
        for v in read_jsonl(dir.join("bases.jsonl"))? {
            let z = (v["level"].as_u64().unwrap_or(0) as u16, v["area"].as_u64().unwrap_or(0) as u8);
            zones.insert(z, v["base"].as_u64().unwrap_or(0) as u16);
        }
        let index = read_jsonl(dir.join("index.jsonl"))?;
        // `{"revoke": file}` records withdraw a checkpoint (kept on disk).
        let revoked: std::collections::HashSet<String> = index.iter().filter_map(|v| v["revoke"].as_str().map(Into::into)).collect();
        let entries = index
            .into_iter()
            .filter(|v| v.get("revoke").is_none() && !revoked.contains(v["file"].as_str().unwrap_or("")))
            .map(|v| Entry {
                frame: v["frame"].as_u64().unwrap_or(0) as u32,
                owner: v["owner"].as_str().unwrap_or("").to_string(),
                file: v["file"].as_str().unwrap_or("").to_string(),
                zone: (v["level"].as_u64().unwrap_or(0) as u16, v["area"].as_u64().unwrap_or(0) as u8),
            })
            .collect();
        Ok(Store { dir, zones, bases: HashMap::new(), entries })
    }

    fn base(&mut self, i: u16) -> Result<&Vec<u8>> {
        if !self.bases.contains_key(&i) {
            let p = self.dir.join(format!("bases/{i}.lz4"));
            let b = std::fs::read(&p).with_context(|| p.display().to_string())?;
            self.bases.insert(i, lz4_flex::decompress_size_prepended(&b)?);
        }
        Ok(&self.bases[&i])
    }

    /// Saves `state` at `frame` (the state after that many inputs from
    /// power-on) for segment `owner`. Bytes written.
    pub fn put(&mut self, frame: u32, owner: &str, zone: (u16, u8), state: &SaveState) -> Result<usize> {
        if self.entries.iter().any(|e| e.frame == frame && e.owner == owner) {
            return Ok(0);
        }
        let bytes = state.as_bytes();
        let bi = match self.zones.get(&zone) {
            Some(&b) => b,
            None => {
                let b = self.zones.len() as u16;
                write_atomic(self.dir.join(format!("bases/{b}.lz4")), &lz4_flex::compress_prepend_size(bytes))?;
                append_jsonl(self.dir.join("bases.jsonl"), &json!({"base": b, "level": zone.0, "area": zone.1}))?;
                self.zones.insert(zone, b);
                self.bases.insert(b, bytes.to_vec());
                b
            }
        };
        let root = self.base(bi)?;
        anyhow::ensure!(root.len() == bytes.len(), "checkpoint: state size {} differs from its base's {}", bytes.len(), root.len());
        let mut pages: Vec<u32> = Vec::new();
        let mut xor = Vec::new();
        for i in 0..bytes.len().div_ceil(PAGE) {
            let r = i * PAGE..((i + 1) * PAGE).min(bytes.len());
            if bytes[r.clone()] != root[r.clone()] {
                pages.push(i as u32);
                xor.extend(bytes[r.clone()].iter().zip(&root[r]).map(|(a, b)| a ^ b));
            }
        }
        let mut out = Vec::with_capacity(16 + pages.len() * 4 + xor.len() / 4);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&bi.to_le_bytes());
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&(pages.len() as u32).to_le_bytes());
        for p in &pages {
            out.extend_from_slice(&p.to_le_bytes());
        }
        out.extend_from_slice(&lz4_flex::compress_prepend_size(&xor));
        let file = format!("{frame:08}-{owner}.ck");
        write_atomic(self.dir.join(&file), &out)?;
        append_jsonl(self.dir.join("index.jsonl"), &json!({"frame": frame, "owner": owner, "file": file, "level": zone.0, "area": zone.1}))?;
        self.entries.push(Entry { frame, owner: owner.to_string(), file, zone });
        Ok(out.len())
    }

    pub fn get(&mut self, e: &Entry) -> Result<SaveState> {
        let p = self.dir.join(&e.file);
        let b = std::fs::read(&p).with_context(|| p.display().to_string())?;
        if b.len() < 14 || &b[..4] != MAGIC {
            bail!("{}: not a checkpoint", p.display());
        }
        let bi = u16::from_le_bytes([b[4], b[5]]);
        let len = u32::from_le_bytes(b[6..10].try_into().unwrap()) as usize;
        let n = u32::from_le_bytes(b[10..14].try_into().unwrap()) as usize;
        let pages: Vec<usize> = (0..n).map(|i| u32::from_le_bytes(b[14 + 4 * i..18 + 4 * i].try_into().unwrap()) as usize).collect();
        let xor = lz4_flex::decompress_size_prepended(&b[14 + 4 * n..])?;
        let mut bytes = self.base(bi)?.clone();
        anyhow::ensure!(bytes.len() == len, "{}: base size differs", p.display());
        let mut rest = xor.as_slice();
        for i in pages {
            let at = i * PAGE;
            let l = PAGE.min(bytes.len() - at);
            for (x, y) in bytes[at..at + l].iter_mut().zip(&rest[..l]) {
                *x ^= y;
            }
            rest = &rest[l..];
        }
        SaveState::from_bytes(bytes)
    }

    /// The latest checkpoint at or before `frame` among those valid on a
    /// path: `valid(owner, frame)` says whether a checkpoint of that segment
    /// at that frame lies on it.
    pub fn best(&self, frame: u32, valid: impl Fn(&str, u32) -> bool) -> Option<Entry> {
        self.entries.iter().filter(|e| e.frame <= frame && valid(&e.owner, e.frame)).max_by_key(|e| e.frame).cloned()
    }

    /// Withdraws the checkpoints of `owner` at frames > `after` (written for a
    /// state that is not on that segment's path); the files stay on disk.
    pub fn revoke(&mut self, owner: &str, after: u32, why: &str) -> Result<usize> {
        let bad: Vec<Entry> = self.entries.iter().filter(|e| e.owner == owner && e.frame > after).cloned().collect();
        for e in &bad {
            append_jsonl(self.dir.join("index.jsonl"), &json!({"revoke": e.file, "why": why}))?;
        }
        self.entries.retain(|e| !(e.owner == owner && e.frame > after));
        Ok(bad.len())
    }

    pub fn summary(&self) -> Value {
        let bytes: u64 = self.entries.iter().filter_map(|e| std::fs::metadata(self.dir.join(&e.file)).ok()).map(|m| m.len()).sum();
        json!({"checkpoints": self.entries.len(), "bases": self.zones.len(), "bytes": bytes})
    }
}
