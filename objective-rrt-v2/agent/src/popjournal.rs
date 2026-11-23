//! The GA population journal: every generation's population after
//! evaluation, exactly reconstructible. One file per segment (commit id),
//! `pop/<segment>.pj`, of records:
//!
//! `[u32 len][u8 kind][u64 generation][u32 frame][lz4 body]`
//!
//! kind 0 is a keyframe (every plan in full), kind 1 a delta on the previous
//! record: each plan names the previous plan it matches best once that one
//! is shifted by the frames committed in between, and lists only the spans
//! that differ. `frame` is the committed state's frame the population was
//! evaluated from. The first record of each file is a keyframe, so files are
//! independent.

use anyhow::{Context, Result, bail};
use obrrt_core::GamePad;
use obrrt_search::ga::rhstep::PlanRec;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

pub struct Writer {
    file: std::fs::File,
    prev: Option<(u32, Vec<PlanRec>)>,
    since_key: usize,
    pub keyframe_every: usize,
    pub bytes: u64,
    pub records: u64,
}

fn put_pads(out: &mut Vec<u8>, pads: &[GamePad]) {
    for p in pads {
        out.extend_from_slice(&p.button.to_le_bytes());
        out.push(p.stick_x as u8);
        out.push(p.stick_y as u8);
    }
}

fn put_meta(out: &mut Vec<u8>, p: &PlanRec) {
    out.extend_from_slice(&p.raw.to_le_bytes());
    out.push(p.origin);
    out.extend_from_slice(&p.generation.to_le_bytes());
    out.extend_from_slice(&p.fitness.to_le_bytes());
    out.extend_from_slice(&p.at.to_le_bytes());
    out.extend_from_slice(&p.first_done.to_le_bytes());
    out.push(p.end);
    out.extend_from_slice(&p.final_eta.to_le_bytes());
}

struct Cur<'a> {
    b: &'a [u8],
    i: usize,
}

impl Cur<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        if self.i + n > self.b.len() {
            bail!("population journal: truncated record");
        }
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn pads(&mut self, n: usize) -> Result<Vec<GamePad>> {
        let b = self.take(4 * n)?;
        Ok(b.chunks_exact(4).map(|c| GamePad::new(u16::from_le_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect())
    }
    fn meta(&mut self, pads: Vec<GamePad>) -> Result<PlanRec> {
        let raw = self.u32()?;
        let origin = self.u8()?;
        let generation = self.u16()?;
        let fitness = f64::from_le_bytes(self.take(8)?.try_into()?);
        let at = self.u32()?;
        let first_done = i32::from_le_bytes(self.take(4)?.try_into()?);
        let end = self.u8()?;
        let final_eta = f32::from_le_bytes(self.take(4)?.try_into()?);
        Ok(PlanRec { pads, raw, origin, generation, fitness, at, first_done, end, final_eta })
    }
}

/// The previous plan `c` matches best when shifted by `shift` (index,
/// matching frames).
fn best_ref(c: &[GamePad], prev: &[PlanRec], shift: usize) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for (j, p) in prev.iter().enumerate() {
        let src = p.pads.get(shift..).unwrap_or(&[]);
        let m = c.iter().zip(src).filter(|(a, b)| a == b).count();
        if m > 0 && best.is_none_or(|b| m > b.1) {
            best = Some((j, m));
            if m == c.len() {
                break;
            }
        }
    }
    best
}

impl Writer {
    pub fn open(path: &Path, keyframe_every: usize) -> Result<Writer> {
        let file = std::fs::OpenOptions::new().create(true).append(true).open(path).with_context(|| path.display().to_string())?;
        let bytes = file.metadata()?.len();
        Ok(Writer { file, prev: None, since_key: 0, keyframe_every: keyframe_every.max(1), bytes, records: 0 })
    }

    /// Records a population evaluated from the state at `frame`.
    pub fn record(&mut self, generation: u64, frame: u32, pop: &[PlanRec]) -> Result<()> {
        let key = self.prev.is_none() || self.since_key + 1 >= self.keyframe_every || self.prev.as_ref().is_some_and(|(f, _)| frame < *f);
        let mut body = Vec::with_capacity(pop.len() * 64);
        body.extend_from_slice(&(pop.len() as u32).to_le_bytes());
        if key {
            for p in pop {
                body.extend_from_slice(&(p.pads.len() as u32).to_le_bytes());
                put_pads(&mut body, &p.pads);
                put_meta(&mut body, p);
            }
            self.since_key = 0;
        } else {
            let (pf, prev) = self.prev.as_ref().unwrap();
            let shift = (frame - pf) as usize;
            body.extend_from_slice(&(shift as u32).to_le_bytes());
            for p in pop {
                body.extend_from_slice(&(p.pads.len() as u32).to_le_bytes());
                match best_ref(&p.pads, prev, shift) {
                    None => {
                        body.extend_from_slice(&u32::MAX.to_le_bytes());
                        put_pads(&mut body, &p.pads);
                    }
                    Some((j, _)) => {
                        body.extend_from_slice(&(j as u32).to_le_bytes());
                        let src = prev[j].pads.get(shift..).unwrap_or(&[]);
                        // Runs where the plan differs from (or runs past) the reference.
                        let mut runs: Vec<(usize, usize)> = Vec::new();
                        let mut t = 0;
                        while t < p.pads.len() {
                            if src.get(t) == Some(&p.pads[t]) {
                                t += 1;
                                continue;
                            }
                            let a = t;
                            while t < p.pads.len() && src.get(t) != Some(&p.pads[t]) {
                                t += 1;
                            }
                            runs.push((a, t - a));
                        }
                        body.extend_from_slice(&(runs.len() as u32).to_le_bytes());
                        for (a, n) in runs {
                            body.extend_from_slice(&(a as u32).to_le_bytes());
                            body.extend_from_slice(&(n as u32).to_le_bytes());
                            put_pads(&mut body, &p.pads[a..a + n]);
                        }
                    }
                }
                put_meta(&mut body, p);
            }
            self.since_key += 1;
        }
        let z = lz4_flex::compress_prepend_size(&body);
        let mut rec = Vec::with_capacity(z.len() + 17);
        rec.extend_from_slice(&((z.len() + 13) as u32).to_le_bytes());
        rec.push(if key { 0 } else { 1 });
        rec.extend_from_slice(&generation.to_le_bytes());
        rec.extend_from_slice(&frame.to_le_bytes());
        rec.extend_from_slice(&z);
        self.file.write_all(&rec)?;
        self.bytes += rec.len() as u64;
        self.records += 1;
        self.prev = Some((frame, pop.to_vec()));
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.file.flush()?;
        Ok(())
    }
}

/// A record's header.
#[derive(Debug, Clone, Copy)]
pub struct Head {
    pub offset: u64,
    pub kind: u8,
    pub generation: u64,
    pub frame: u32,
}

/// The records of a journal file (headers only).
pub fn index(path: &Path) -> Result<Vec<Head>> {
    let mut f = std::fs::File::open(path).with_context(|| path.display().to_string())?;
    let len = f.metadata()?.len();
    let mut out = Vec::new();
    let mut off = 0u64;
    let mut h = [0u8; 17];
    while off + 17 <= len {
        f.seek(SeekFrom::Start(off))?;
        f.read_exact(&mut h)?;
        let n = u32::from_le_bytes(h[..4].try_into()?) as u64;
        if off + 4 + n > len {
            break; // a torn last record
        }
        out.push(Head { offset: off, kind: h[4], generation: u64::from_le_bytes(h[5..13].try_into()?), frame: u32::from_le_bytes(h[13..17].try_into()?) });
        off += 4 + n;
    }
    Ok(out)
}

fn body(f: &mut std::fs::File, h: &Head) -> Result<Vec<u8>> {
    f.seek(SeekFrom::Start(h.offset))?;
    let mut n = [0u8; 4];
    f.read_exact(&mut n)?;
    let n = u32::from_le_bytes(n) as usize;
    let mut b = vec![0u8; n];
    f.read_exact(&mut b)?;
    Ok(lz4_flex::decompress_size_prepended(&b[13..])?)
}

fn decode(b: &[u8], kind: u8, prev: Option<&[PlanRec]>) -> Result<Vec<PlanRec>> {
    let mut c = Cur { b, i: 0 };
    let n = c.u32()? as usize;
    let mut out = Vec::with_capacity(n);
    if kind == 0 {
        for _ in 0..n {
            let len = c.u32()? as usize;
            let pads = c.pads(len)?;
            out.push(c.meta(pads)?);
        }
        return Ok(out);
    }
    let prev = prev.context("population journal: a delta record without its predecessor")?;
    let shift = c.u32()? as usize;
    for _ in 0..n {
        let len = c.u32()? as usize;
        let r = c.u32()?;
        let pads = if r == u32::MAX {
            c.pads(len)?
        } else {
            let src = prev.get(r as usize).context("population journal: bad reference")?.pads.get(shift..).unwrap_or(&[]);
            let mut pads: Vec<GamePad> = (0..len).map(|t| src.get(t).copied().unwrap_or_default()).collect();
            let runs = c.u32()?;
            for _ in 0..runs {
                let a = c.u32()? as usize;
                let k = c.u32()? as usize;
                let seg = c.pads(k)?;
                pads[a..a + k].copy_from_slice(&seg);
            }
            pads
        };
        out.push(c.meta(pads)?);
    }
    Ok(out)
}

/// The population of record `i` (from its keyframe on).
pub fn read_at(path: &Path, heads: &[Head], i: usize) -> Result<Vec<PlanRec>> {
    let k = (0..=i).rev().find(|&j| heads[j].kind == 0).context("population journal: no keyframe")?;
    let mut f = std::fs::File::open(path)?;
    let mut pop: Option<Vec<PlanRec>> = None;
    for h in &heads[k..=i] {
        let b = body(&mut f, h)?;
        pop = Some(decode(&b, h.kind, pop.as_deref())?);
    }
    Ok(pop.unwrap())
}

/// The last record evaluated from the state at `frame`.
pub fn find_frame(heads: &[Head], frame: u32) -> Option<usize> {
    heads.iter().rposition(|h| h.frame == frame)
}
