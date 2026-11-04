//! Writes training data for the learned backbone (backbone/, sm64_AI_backbone.md):
//! raw state features ([`obrrt_core::features`]) along a run's tree edges or
//! a TAS movie, with the pads and event bits between samples.
//!
//! cargo run --release --bin dataset -- schema <out.json>
//! cargo run --release --bin dataset -- tree <rom> <run_dir> <select.bin> <out_dir> [threads=8]
//! cargo run --release --bin dataset -- tas <rom> <movie.m64> <out_dir>
//!
//! `select.bin`: little-endian u32s, `node << 2 | mode`, mode 1 = record the
//! node's whole edge, 2 = only its last 64 frames (history for a selected
//! child). Tree samples are taken on frames divisible by 4; TAS samples on
//! every frame. A sample's `prev` is the sample 4 frames earlier on the same
//! path (-1 if not recorded); `pads` are the 4 inputs between them, oldest
//! first, and `events` the OR of the per-frame event bits over those frames.
//!
//! Out dir: per shard `s<k>/` with gi.i32, gf.f32, oi.i32, of.f32, on.u16,
//! tri.f32, tn.u16, pads.u32, meta.i64 (node, frame, prev, events, pads
//! valid, source), ends.f32 (selected nodes' end states, to check the
//! replay against nodes.csv); and `zones/<level>_<area>.f32`: each zone's
//! static triangles. Input files are trusted to be well formed only as far
//! as bounds checks go.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::rc::Rc;
use std::sync::Mutex;

use anyhow::{Result, bail, ensure};
use obrrt_core::features::{self, Features};
use obrrt_core::{GamePad, St, read_state};
use sm64_binds::{SM64Game, SM64GameGenerator};

const HIST: usize = 4;
const TAIL: u32 = 64;

struct Out {
    gi: BufWriter<File>,
    gf: BufWriter<File>,
    oi: BufWriter<File>,
    of: BufWriter<File>,
    on: BufWriter<File>,
    tri: BufWriter<File>,
    tn: BufWriter<File>,
    pads: BufWriter<File>,
    meta: BufWriter<File>,
    ends: BufWriter<File>,
    n: i64,
}

fn w<T: Copy>(f: &mut BufWriter<File>, v: &[T]) -> Result<()> {
    // Plain-old-data slices (i32, f32, u16, u32, i64) as little-endian bytes.
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    f.write_all(bytes)?;
    Ok(())
}

impl Out {
    fn new(dir: &str) -> Result<Out> {
        fs::create_dir_all(dir)?;
        let f = |n: &str| -> Result<BufWriter<File>> { Ok(BufWriter::with_capacity(1 << 20, File::create(format!("{dir}/{n}"))?)) };
        Ok(Out {
            gi: f("gi.i32")?,
            gf: f("gf.f32")?,
            oi: f("oi.i32")?,
            of: f("of.f32")?,
            on: f("on.u16")?,
            tri: f("tri.f32")?,
            tn: f("tn.u16")?,
            pads: f("pads.u32")?,
            meta: f("meta.i64")?,
            ends: f("ends.f32")?,
            n: 0,
        })
    }

    fn push(&mut self, ft: &Features, hist: &Hist, meta: [i64; 3], source: i64) -> Result<i64> {
        w(&mut self.gi, &ft.ints)?;
        w(&mut self.gf, &ft.flts)?;
        w(&mut self.oi, &ft.obj_ints)?;
        w(&mut self.of, &ft.obj_flts)?;
        w(&mut self.on, &[ft.n_obj as u16])?;
        w(&mut self.tri, &ft.tris)?;
        w(&mut self.tn, &[ft.n_tri as u16])?;
        let (pads, ev, valid) = hist.window();
        w(&mut self.pads, &pads)?;
        w(&mut self.meta, &[meta[0], meta[1], meta[2], ev as i64, valid as i64, source])?;
        self.n += 1;
        Ok(self.n - 1)
    }

    fn flush(&mut self) -> Result<()> {
        for f in [&mut self.gi, &mut self.gf, &mut self.oi, &mut self.of, &mut self.on, &mut self.tri, &mut self.tn, &mut self.pads, &mut self.meta, &mut self.ends] {
            f.flush()?;
        }
        Ok(())
    }
}

/// The last HIST frames' pads and event bits along the current path.
#[derive(Clone, Default)]
struct Hist {
    pads: [u32; HIST],
    ev: [u32; HIST],
    len: usize,
}

impl Hist {
    fn step(&mut self, p: GamePad, ev: u32) {
        self.pads.rotate_left(1);
        self.ev.rotate_left(1);
        self.pads[HIST - 1] = (p.button as u32) << 16 | (p.stick_x as u8 as u32) << 8 | p.stick_y as u8 as u32;
        self.ev[HIST - 1] = ev;
        self.len = (self.len + 1).min(HIST);
    }
    fn window(&self) -> ([u32; HIST], u32, usize) {
        (self.pads, self.ev.iter().fold(0, |a, b| a | b), self.len)
    }
}

/// Zones whose static geometry is written (shared by the shards).
struct Zones {
    dir: String,
    done: Mutex<HashSet<(u16, u8)>>,
}

impl Zones {
    fn observe(&self, m: &[u8], st: &St) -> Result<()> {
        let z = st.zone();
        if self.done.lock().unwrap().contains(&z) {
            return Ok(());
        }
        if let Some(tris) = features::zone_geometry(m, st) {
            if self.done.lock().unwrap().insert(z) {
                let flat: Vec<f32> = tris.iter().flatten().copied().collect();
                fs::write(format!("{}/{}_{}.f32", self.dir, z.0, z.1), unsafe {
                    std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 4)
                })?;
            }
        }
        Ok(())
    }
}

fn record(game: &SM64Game, frame: u32, zones: &Zones) -> Result<(St, Features)> {
    let st = read_state(game, frame, true);
    let ft = game.with_memory(|m| -> Result<Features> {
        zones.observe(m, &st)?;
        Ok(features::read_features(m, &st))
    })?;
    Ok((st, ft))
}

fn read_tree(path: &str) -> Result<Vec<(i32, Vec<GamePad>)>> {
    let data = fs::read(path)?;
    let u32_at = |p: usize| -> Result<u32> {
        ensure!(p + 4 <= data.len(), "{path}: truncated");
        Ok(u32::from_le_bytes(data[p..p + 4].try_into().unwrap()))
    };
    let count = u32_at(0)? as usize;
    let mut pos = 4;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let parent = u32_at(pos)? as i32;
        let n = u32_at(pos + 4)? as usize;
        pos += 8;
        ensure!(pos + 4 * n <= data.len(), "{path}: truncated");
        let inputs = data[pos..pos + 4 * n]
            .chunks_exact(4)
            .map(|b| GamePad::new(u16::from_be_bytes([b[0], b[1]]), b[2] as i8, b[3] as i8))
            .collect();
        pos += 4 * n;
        out.push((parent, inputs));
    }
    Ok(out)
}

struct Entry {
    node: u32,
    save: Option<Rc<sm64_binds::SaveState>>,
    before: St,
    hist: Hist,
    frame: u32,
    /// Last recorded sample on the path: (frame, index).
    last: (u32, i64),
}

/// One shard: DFS over the union of the root paths of `selected`.
fn tree_shard(rom: &str, records: &[(i32, Vec<GamePad>)], selected: &[(u32, u8)], dir: &str, zones: &Zones) -> Result<()> {
    let parent = |n: u32| records[n as usize + 1].0;
    let mode: HashMap<u32, u8> = selected.iter().copied().collect();
    let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut needed: HashSet<u32> = HashSet::from([0]);
    for &(n, _) in selected {
        let mut c = n;
        while !needed.contains(&c) {
            needed.insert(c);
            let p = parent(c);
            ensure!(p >= 0, "node {c} has no parent but isn't the root");
            children.entry(p as u32).or_default().push(c);
            c = p as u32;
        }
    }
    for v in children.values_mut() {
        v.sort_unstable();
    }

    let game_gen = SM64GameGenerator::from_file(rom)?;
    let mut game = game_gen.create_game()?;
    let mut before = read_state(&game, 0, false);
    let mut frame = 0u32;
    for p in &records[0].1 {
        game.step_game(*p)?;
        frame += 1;
        before = read_state(&game, frame, false);
    }
    let mut out = Out::new(dir)?;
    let mut stack = vec![Entry { node: 0, save: None, before, hist: Hist::default(), frame, last: (u32::MAX, -1) }];
    let (mut edges, mut frames) = (0u64, 0u64);
    let t0 = std::time::Instant::now();
    while let Some(mut e) = stack.pop() {
        if let Some(s) = &e.save {
            game.load_state(s)?;
        }
        let inputs = &records[e.node as usize + 1].1;
        let m = mode.get(&e.node).copied().unwrap_or(0);
        let end = e.frame + inputs.len() as u32;
        for p in inputs {
            game.step_game(*p)?;
            e.frame += 1;
            let cur = read_state(&game, e.frame, false);
            e.hist.step(*p, features::events(&e.before, &cur));
            e.before = cur;
            let want = m == 1 || (m == 2 && e.frame + TAIL >= end);
            if want && e.frame % HIST as u32 == 0 {
                let prev = if e.last.0 != u32::MAX && e.last.0 + HIST as u32 == e.frame { e.last.1 } else { -1 };
                let (_, ft) = record(&game, e.frame, zones)?;
                let idx = out.push(&ft, &e.hist, [e.node as i64, e.frame as i64, prev], 0)?;
                e.last = (e.frame, idx);
            }
        }
        frames += inputs.len() as u64;
        edges += 1;
        if m != 0 {
            let s = &e.before;
            w(&mut out.ends, &[e.node as f32, e.frame as f32, s.pos[0], s.pos[1], s.pos[2]])?;
        }
        let kids = children.get(&e.node).map_or(&[][..], |v| v.as_slice());
        let save = (kids.len() > 1).then(|| Rc::new(game.save_state()));
        // Reverse so the lowest id runs first; a single child continues
        // from the current game state.
        for &k in kids.iter().rev() {
            stack.push(Entry { node: k, save: save.clone(), before: e.before.clone(), hist: e.hist.clone(), frame: e.frame, last: e.last });
        }
        if edges % 20_000 == 0 {
            eprintln!("{dir}: {edges} edges, {frames} frames, {} samples, {:.0} s", out.n, t0.elapsed().as_secs_f64());
        }
    }
    out.flush()?;
    eprintln!("{dir}: done: {edges} edges, {frames} frames, {} samples, {:.0} s", out.n, t0.elapsed().as_secs_f64());
    Ok(())
}

fn tas(rom: &str, movie: &str, dir: &str) -> Result<()> {
    let bytes = fs::read(movie)?;
    ensure!(bytes.len() >= 0x400 && &bytes[..4] == b"M64\x1a", "{movie}: not an .m64");
    let samples = u32::from_le_bytes(bytes[0x18..0x1C].try_into().unwrap()) as usize;
    ensure!(u16::from_le_bytes([bytes[0x1C], bytes[0x1D]]) == 2, "{movie}: doesn't start from power-on");
    let data = &bytes[0x400..];
    let body = &data[..(samples * 4).min(data.len() / 4 * 4)];
    let pads: Vec<GamePad> = body.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect();

    let zones = Zones { dir: format!("{dir}/zones"), done: Mutex::new(HashSet::new()) };
    fs::create_dir_all(&zones.dir)?;
    let game_gen = SM64GameGenerator::from_file(rom)?;
    let mut game = game_gen.create_game()?;
    let mut out = Out::new(&format!("{dir}/s0"))?;
    let mut hist = Hist::default();
    let mut before = read_state(&game, 0, false);
    // Index of the sample at each frame (every frame is recorded once
    // Mario exists), for prev links 4 frames back.
    let mut at: Vec<i64> = vec![-1; pads.len() + 1];
    for (i, p) in pads.iter().enumerate() {
        game.step_game(*p)?;
        let frame = i as u32 + 1;
        let cur = read_state(&game, frame, false);
        hist.step(*p, features::events(&before, &cur));
        let live = cur.level != 0 && cur.action != 0;
        before = cur;
        if live {
            let prev = if frame as usize >= HIST { at[frame as usize - HIST] } else { -1 };
            let (_, ft) = record(&game, frame, &zones)?;
            at[frame as usize] = out.push(&ft, &hist, [-1, frame as i64, prev], 1)?;
        }
    }
    out.flush()?;
    let s = &before;
    eprintln!("{movie}: {} frames, {} samples; end: level {} stars {} pos {:?}", pads.len(), out.n, s.level, s.stars, s.pos);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("schema") => {
            let j = serde_json::json!({
                "int_cols": features::INT_COLS, "flt_cols": features::FLT_COLS,
                "obj_int_cols": features::OBJ_INT_COLS, "obj_flt_cols": features::OBJ_FLT_COLS,
                "tri_cols": features::TRI_COLS, "events": features::EVENT_NAMES,
                "max_obj": features::MAX_OBJ, "max_dsurf": features::MAX_DSURF, "hist": HIST,
                "behaviours": obrrt_core::layout::BEHAVIOURS.iter().map(|b| b.1).collect::<Vec<_>>(),
            });
            fs::write(&args[2], serde_json::to_string_pretty(&j)?)?;
        }
        Some("tree") if args.len() >= 6 => {
            let (rom, run, sel, out) = (&args[2], &args[3], &args[4], &args[5]);
            let threads: usize = args.get(6).map_or(Ok(8), |s| s.parse())?;
            let records = read_tree(&format!("{run}/tree.bin"))?;
            let raw = fs::read(sel)?;
            let mut selected: Vec<(u32, u8)> =
                raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).map(|v| (v >> 2, (v & 3) as u8)).collect();
            ensure!(selected.iter().all(|&(n, _)| (n as usize) + 1 < records.len()), "{sel}: node id out of range");
            // Contiguous shards in the given order (callers sort by DFS
            // order so shards share little of their root paths).
            let zones = Zones { dir: format!("{out}/zones"), done: Mutex::new(HashSet::new()) };
            fs::create_dir_all(&zones.dir)?;
            let chunk = selected.len().div_ceil(threads.max(1));
            let shards: Vec<Vec<(u32, u8)>> = selected.drain(..).collect::<Vec<_>>().chunks(chunk.max(1)).map(|c| c.to_vec()).collect();
            std::thread::scope(|s| -> Result<()> {
                let hs: Vec<_> = shards
                    .iter()
                    .enumerate()
                    .map(|(k, sh)| {
                        let (records, zones) = (&records, &zones);
                        s.spawn(move || tree_shard(rom, records, sh, &format!("{out}/s{k}"), zones))
                    })
                    .collect();
                for h in hs {
                    h.join().map_err(|_| anyhow::anyhow!("shard panicked"))??;
                }
                Ok(())
            })?;
        }
        Some("tas") if args.len() >= 5 => tas(&args[2], &args[3], &args[4])?,
        _ => bail!("usage: dataset schema <out.json> | tree <rom> <run_dir> <select.bin> <out_dir> [threads] | tas <rom> <movie.m64> <out_dir>"),
    }
    Ok(())
}
