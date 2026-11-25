//! Attempt results as bytes, and the executors that move them between workers.
//!
//! An [`AttemptExecutor`] runs a round's attempts. The default (none) is the engine's rayon path.
//! [`SharedExecutor`] is for several processes that each hold a copy of the same search (browser
//! Web Workers, one tree per worker, rebuilt identically from the same seed): the attempts of a
//! round are claimed one by one through [`AttemptPeers`], each result goes out as bytes
//! ([`encode`]) and every worker decodes all of them in attempt order, so every copy merges the
//! same results in the same order whichever worker ran which attempt.

use anyhow::{Result, anyhow, bail};
use obrrt_core::{St, dyngeo::DynGeo};
use serde::{Deserialize, Serialize};
use sm64_binds::GamePad;
use std::sync::Arc;

use crate::dense;
use crate::ga::{Attempt, GaInfo, ORIGINS, Segment};
use crate::store::{Delta, DeltaWire};

pub type AttemptResult = Result<Attempt<Delta>>;

/// A round's attempt `i` as a function. Shareable between threads natively; the browser's games are not
/// (one thread), so there it is not.
#[cfg(not(target_arch = "wasm32"))]
pub type OneAttempt<'a> = dyn Fn(usize) -> AttemptResult + Sync + 'a;
#[cfg(target_arch = "wasm32")]
pub type OneAttempt<'a> = dyn Fn(usize) -> AttemptResult + 'a;

/// How a round's attempts get run (see the module doc).
pub trait AttemptExecutor: Send + Sync {
    /// Runs attempts `0..n`, calling `one(i)` for each one this executor takes itself (on any
    /// thread), and returns all `n` results in index order.
    fn run(&self, n: usize, one: &OneAttempt<'_>) -> Vec<AttemptResult>;
}

#[derive(Serialize, Deserialize)]
struct GaW {
    generations: usize,
    winner_gen: u16,
    origin: u8,
    gen0: (usize, f64),
    best: (usize, f64),
    moves: [u16; 7],
    seed: Option<(bool, u32)>,
}

#[derive(Serialize, Deserialize)]
struct Head {
    frames: u64,
    died: bool,
    ga: Option<GaW>,
    segs: u32,
    dense: Option<(dense::Stats, u32)>,
}

#[derive(Serialize)]
struct SegHeadRef<'a> {
    inputs: &'a [GamePad],
    success: bool,
    progress: f64,
    played: usize,
    tail: &'a [GamePad],
    end: bool,
}
#[derive(Deserialize)]
struct SegHead {
    inputs: Vec<GamePad>,
    success: bool,
    progress: f64,
    played: usize,
    tail: Vec<GamePad>,
    end: bool,
}

#[derive(Serialize)]
struct CandHeadRef<'a> {
    chain: u32,
    inputs: &'a [GamePad],
    key: u64,
    save: bool,
}
#[derive(Deserialize)]
struct CandHead {
    chain: u32,
    inputs: Vec<GamePad>,
    key: u64,
    save: bool,
}

fn put<T: Serialize>(out: &mut Vec<u8>, v: &T) -> Result<()> {
    postcard::to_extend(v, std::mem::take(out)).map(|v| *out = v).map_err(|e| anyhow!("encode: {e}"))
}

fn take<'a, T: Deserialize<'a>>(b: &mut &'a [u8]) -> Result<T> {
    let (v, rest) = postcard::take_from_bytes(b).map_err(|e| anyhow!("decode: {e}"))?;
    *b = rest;
    Ok(v)
}

fn put_state(out: &mut Vec<u8>, st: &St) -> Result<()> {
    put(out, st)?;
    let geo = st.objects.as_ref().and_then(|_| obrrt_core::dyngeo::of(st));
    put(out, &geo.as_deref())
}

fn take_state(b: &mut &[u8]) -> Result<St> {
    let st: St = take(b)?;
    let geo: Option<DynGeo> = take(b)?;
    if let Some(g) = geo {
        obrrt_core::dyngeo::attach(&st, g);
    }
    Ok(st)
}

/// An attempt's result as bytes. The learned model's per-node scores (`OBRRT_MODEL`) are not carried:
/// an attempt with them is an error.
pub fn encode(r: &AttemptResult) -> Vec<u8> {
    let mut out = Vec::new();
    match r {
        Err(e) => {
            out.push(0);
            let _ = put(&mut out, &format!("{e:#}"));
        }
        Ok(a) => {
            out.push(1);
            if let Err(e) = encode_ok(&mut out, a) {
                out.clear();
                out.push(0);
                let _ = put(&mut out, &format!("encode: {e:#}"));
            }
        }
    }
    out
}

fn encode_ok(out: &mut Vec<u8>, a: &Attempt<Delta>) -> Result<()> {
    let ga = a.ga.map(|g| GaW {
        generations: g.generations,
        winner_gen: g.winner_gen,
        origin: ORIGINS.iter().position(|o| *o == g.winner_origin).unwrap_or(0) as u8,
        gen0: g.gen0,
        best: g.best,
        moves: g.moves,
        seed: g.seed,
    });
    let head = Head { frames: a.frames, died: a.died, ga, segs: a.segments.len() as u32, dense: a.dense.as_ref().map(|d| (d.stats, d.cands.len() as u32)) };
    put(out, &head)?;
    for s in &a.segments {
        if s.model.is_some() {
            bail!("a segment with model scores cannot cross workers");
        }
        put(out, &SegHeadRef { inputs: &s.inputs, success: s.success, progress: s.progress, played: s.played, tail: &s.tail, end: s.end.is_some() })?;
        if let Some((st, d)) = &s.end {
            put_state(out, st)?;
            put(out, &d.to_wire())?;
        }
    }
    if let Some(d) = &a.dense {
        for c in &d.cands {
            put(out, &CandHeadRef { chain: c.chain, inputs: &c.inputs, key: c.key, save: c.save.is_some() })?;
            put_state(out, &c.st)?;
            if let Some(s) = &c.save {
                put(out, &s.to_wire())?;
            }
        }
    }
    Ok(())
}

pub fn decode(bytes: &[u8]) -> AttemptResult {
    let (tag, mut b) = bytes.split_first().ok_or_else(|| anyhow!("empty attempt result"))?;
    if *tag == 0 {
        let msg: String = take(&mut b)?;
        return Err(anyhow!("{msg}"));
    }
    let head: Head = take(&mut b)?;
    let mut segments = Vec::with_capacity(head.segs as usize);
    for _ in 0..head.segs {
        let h: SegHead = take(&mut b)?;
        let end = if h.end {
            let st = take_state(&mut b)?;
            let w: DeltaWire = take(&mut b)?;
            Some((st, Delta::from_wire(w)?))
        } else {
            None
        };
        segments.push(Segment { inputs: h.inputs, end, success: h.success, progress: h.progress, model: None, tail: h.tail, played: h.played });
    }
    let dense = match head.dense {
        None => None,
        Some((stats, n)) => {
            let mut cands = Vec::with_capacity(n as usize);
            for _ in 0..n {
                let h: CandHead = take(&mut b)?;
                let st = take_state(&mut b)?;
                let save = if h.save { Some(Delta::from_wire(take(&mut b)?)?) } else { None };
                cands.push(dense::Cand { chain: h.chain, inputs: h.inputs, st, save, key: h.key });
            }
            Some(Box::new(dense::Out { cands, stats }))
        }
    };
    let ga = head.ga.map(|g| GaInfo {
        generations: g.generations,
        winner_gen: g.winner_gen,
        winner_origin: ORIGINS[(g.origin as usize).min(ORIGINS.len() - 1)],
        gen0: g.gen0,
        best: g.best,
        moves: g.moves,
        seed: g.seed,
    });
    Ok(Attempt { segments, frames: head.frames, died: head.died, ga, dense })
}

/// The channel between the workers of a [`SharedExecutor`]: every worker makes the same calls in the
/// same order, with the same `n` per round.
pub trait AttemptPeers: Send + Sync {
    /// The next unclaimed attempt of the round (None: all claimed). Each index goes to exactly one
    /// worker. Errs when the round is to be abandoned (stop, a peer gone).
    fn claim(&self, round: u64, n: usize) -> Result<Option<usize>>;
    /// Publishes the result bytes of attempt `i` of the round.
    fn put(&self, round: u64, i: usize, bytes: &[u8]) -> Result<()>;
    /// Blocks until all `n` results of the round are published and returns them in index order.
    fn collect(&self, round: u64, n: usize) -> Result<Vec<Vec<u8>>>;
}

/// Runs a round's attempts across the workers sharing `peers` (see the module doc).
pub struct SharedExecutor {
    pub peers: Arc<dyn AttemptPeers>,
    round: std::sync::atomic::AtomicU64,
    /// Where the per-25-rounds timing line goes (the browser build's log), if anywhere.
    pub log: Option<Arc<dyn Fn(String) + Send + Sync>>,
    stats: std::sync::Mutex<[f64; 6]>,
}

impl SharedExecutor {
    pub fn new(peers: Arc<dyn AttemptPeers>) -> SharedExecutor {
        SharedExecutor { peers, round: std::sync::atomic::AtomicU64::new(0), log: None, stats: std::sync::Mutex::new([0.0; 6]) }
    }

    /// Timing lines (every 20 rounds) go to `log`.
    pub fn with_log(mut self, log: Arc<dyn Fn(String) + Send + Sync>) -> SharedExecutor {
        self.log = Some(log);
        self
    }

    fn run_round(&self, round: u64, n: usize, one: &OneAttempt<'_>) -> Result<Vec<AttemptResult>> {
        use obrrt_core::sys::Instant;
        let t0 = Instant::now();
        let (mut run, mut mine, mut bytes) = (0.0, 0usize, 0usize);
        while let Some(i) = self.peers.claim(round, n)? {
            let t = Instant::now();
            let r = one(i);
            let enc = encode(&r);
            run += t.elapsed().as_secs_f64();
            mine += 1;
            bytes += enc.len();
            self.peers.put(round, i, &enc)?;
        }
        let t1 = Instant::now();
        let raw = self.peers.collect(round, n)?;
        let wait = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let out: Vec<AttemptResult> = raw.iter().map(|b| decode(b)).collect();
        let dec = t2.elapsed().as_secs_f64();
        if let Some(log) = &self.log {
            let mut st = self.stats.lock().unwrap();
            for (a, b) in st.iter_mut().zip([run, wait, dec, mine as f64, bytes as f64, t0.elapsed().as_secs_f64()]) {
                *a += b;
            }
            if round % 20 == 0 {
                log(format!("shared rounds {round}: per round run {:.2}s wait {:.2}s decode {:.2}s ({:.1} attempts, {:.0} KB made) of {:.2}s", st[0] / 20.0, st[1] / 20.0, st[2] / 20.0, st[3] / 20.0, st[4] / 20.0e3, st[5] / 20.0));
                *st = [0.0; 6];
            }
        }
        Ok(out)
    }
}

impl AttemptExecutor for SharedExecutor {
    fn run(&self, n: usize, one: &OneAttempt<'_>) -> Vec<AttemptResult> {
        let round = self.round.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        match self.run_round(round, n, one) {
            Ok(v) => v,
            // The round is abandoned (the page stopped the search, or a peer is gone): every attempt
            // fails, the engine counts traps and the step ends with the stop.
            Err(e) => (0..n).map(|_| Err(anyhow!("shared round {round}: {e:#}"))).collect(),
        }
    }
}
