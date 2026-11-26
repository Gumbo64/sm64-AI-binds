//! The command-line run shared by the `oss` and `oss_share` examples:
//!
//! `<schema.json> [minutes] [seed] [--seed file.m64]... [--noise NAME] [--iterations N] [--frames N] [--out DIR] [--rom FILE]`
//!
//! Writes `runs/<schema stem>[-share]/{schema.json, objectives.json, stats.json,
//! cells.jsonl, tree.bin}` every 30 s and at the end, and logs every 5 s.
//! `--iterations N` stops after N batches instead of after `minutes` (for
//! reproducible comparisons: a seed, a thread count and N give one run);
//! `--frames N` once N frames are emulated (equal-frame comparisons; also
//! reproducible, it is checked between batches). The ROM is `--rom`, else
//! `OSS_ROM`, else `sm64-AI-binds/baserom.us.z64`.
//!
//! Key coarseness, applied to the schema before it is loaded (the
//! schema.json written to the run dir is the schema as used):
//! - `OSS_GRID_ADD=<k>`: every objective with a position grid (Root, Stage with
//!   `grid_log2`, Lens, Boss, ...: those whose `pos_grid()` is set, with their
//!   default if the schema leaves it out) gets `grid_log2 + k` (k > 0 coarser).
//! - `OSS_PARAMS='{"Lens": {"speed_step": 0, "yaw_sectors": 4}, ...}'`: these
//!   params set on every node of that kind (after `OSS_GRID_ADD`).

use std::time::Instant;

use anyhow::{Context, Result, bail};
use objective_scattershot::noise::Variant;
use objective_scattershot::util::skip_menu_pads;
use sm64_binds::SM64GameGenerator;

use crate::config::Config;
use crate::objective::Tree;
use crate::search::Search;
use crate::share::Sharer;
use crate::util::load_m64;

pub struct Args {
    pub schema: String,
    pub minutes: f64,
    pub seed: u64,
    pub seeds: Vec<String>,
    pub noise: String,
    pub iterations: Option<u64>,
    pub out: Option<String>,
    pub frames: Option<u64>,
    pub rom: Option<String>,
    /// `--pipe <obrrt run dir>`: speed mode's route (see `speed`).
    pub pipe: Option<String>,
    /// `--pipe-node <id>`: the route's end node (default: the earliest node
    /// with the most stars).
    pub pipe_node: Option<u32>,
}

impl Args {
    pub fn parse(args: impl Iterator<Item = String>) -> Result<Args> {
        let mut it = args.skip(1);
        let (mut pos, mut seeds, mut noise, mut iterations, mut out) = (Vec::new(), Vec::new(), "pink".to_string(), None, None);
        let (mut frames, mut rom, mut pipe, mut pipe_node) = (None, None, None, None);
        while let Some(a) = it.next() {
            let mut val = || it.next().with_context(|| format!("{a} needs a value"));
            match a.as_str() {
                "--seed" => seeds.push(val()?),
                "--noise" => noise = val()?,
                "--iterations" => iterations = Some(val()?.parse()?),
                "--out" => out = Some(val()?),
                "--frames" => frames = Some(val()?.parse()?),
                "--rom" => rom = Some(val()?),
                "--pipe" => pipe = Some(val()?),
                "--pipe-node" => pipe_node = Some(val()?.parse()?),
                s if s.starts_with("--") => bail!("unknown option {s}"),
                _ => pos.push(a),
            }
        }
        let Some(schema) = pos.first().cloned() else { bail!("usage: <schema.json> [minutes] [seed] [--seed file.m64]... [--noise NAME] [--iterations N] [--frames N] [--out DIR] [--rom FILE] [--pipe OBRRT_RUN_DIR [--pipe-node ID]]") };
        let minutes = pos.get(1).map_or(Ok(60.0), |s| s.parse())?;
        let seed = pos.get(2).map_or(Ok(0), |s| s.parse())?;
        Ok(Args { schema, minutes, seed, seeds, noise, iterations, out, frames, rom, pipe, pipe_node })
    }
}

/// The ROM's bytes: `path`, else `OSS_ROM`, else sm64-AI-binds/baserom.us.z64.
pub fn rom_at(path: Option<&str>) -> Result<Vec<u8>> {
    let p = path.map(str::to_string).or_else(|| std::env::var("OSS_ROM").ok()).unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../../baserom.us.z64").to_string());
    std::fs::read(&p).with_context(|| format!("reading the ROM {p}"))
}

/// [`rom_at`] with no path (`OSS_ROM` or the default).
pub fn rom() -> Result<Vec<u8>> {
    rom_at(None)
}

/// The schema with the coarseness switches applied (see the module doc).
pub fn coarsen(schema: &str) -> Result<String> {
    let add: i64 = std::env::var("OSS_GRID_ADD").ok().map_or(Ok(0), |v| v.parse())?;
    let params: Option<serde_json::Map<String, serde_json::Value>> = match std::env::var("OSS_PARAMS") {
        Ok(v) => Some(serde_json::from_str(&v).context("OSS_PARAMS: a JSON object {kind: {param: value}}")?),
        Err(_) => None,
    };
    if add == 0 && params.is_none() {
        return Ok(schema.to_string());
    }
    let mut spec: serde_json::Value = serde_json::from_str(schema)?;
    let grids: Vec<Option<u32>> = if add != 0 { Tree::load(schema)?.insts.iter().map(|i| i.obj.pos_grid()).collect() } else { Vec::new() };
    // Pre-order, as Tree::load numbers the objectives.
    fn walk(n: &mut serde_json::Value, id: &mut usize, grids: &[Option<u32>], add: i64, params: Option<&serde_json::Map<String, serde_json::Value>>) {
        if !n.get("params").is_some_and(|p| p.is_object()) {
            n["params"] = serde_json::json!({});
        }
        if let Some(Some(g)) = grids.get(*id) {
            n["params"]["grid_log2"] = (*g as i64 + add).clamp(0, 16).into();
        }
        if let Some(set) = n.get("kind").and_then(|k| k.as_str()).and_then(|k| params?.get(k)).and_then(|p| p.as_object()).cloned() {
            for (k, v) in set {
                n["params"][k] = v;
            }
        }
        *id += 1;
        if let Some(cs) = n.get_mut("children").and_then(|c| c.as_array_mut()) {
            for c in cs {
                walk(c, id, grids, add, params);
            }
        }
    }
    walk(&mut spec, &mut 0, &grids, add, params.as_ref());
    Ok(serde_json::to_string_pretty(&spec)?)
}

pub fn run(args: Args, share: bool) -> Result<()> {
    let stem = std::path::Path::new(&args.schema).file_stem().unwrap().to_string_lossy().to_string();
    let dir = args.out.clone().unwrap_or_else(|| format!("runs/{stem}{}", if share { "-share" } else { "" }));
    std::fs::create_dir_all(&dir)?;
    let schema = coarsen(&std::fs::read_to_string(&args.schema)?)?;
    std::fs::write(format!("{dir}/schema.json"), &schema)?;
    let tree = Tree::load(&schema)?;
    std::fs::write(format!("{dir}/objectives.json"), serde_json::to_string_pretty(&tree.insts)?)?;

    let variant = Variant::named(&args.noise).with_context(|| format!("unknown noise {}", args.noise))?;
    let game_gen = SM64GameGenerator::new(rom_at(args.rom.as_deref())?)?;
    let cfg = Config::from_env(variant)?;
    if cfg.speed.needs_route() && args.pipe.is_none() {
        bail!("OSS_CORRIDOR, OSS_SELECT=route and OSS_GA_TARGET=route need a route: --pipe <obrrt run dir>");
    }
    if args.pipe_node.is_some() && args.pipe.is_none() {
        bail!("--pipe-node needs --pipe");
    }
    if cfg.ga.share > 0.0 {
        // As obrrt2 sets them up: the steering controller, and the
        // technique objectives (OBRRT_TECH, "all" by default).
        obrrt_objectives::moves::install_controller();
        let tech = std::env::var("OBRRT_TECH").unwrap_or_else(|_| "all".into());
        obrrt_objectives::extra::tech::configure(&tech).map_err(anyhow::Error::msg)?;
    }
    println!(
        "ga: share {} by {} harvest {} learn {} params {:?}",
        cfg.ga.share,
        if cfg.ga.by_frames { "frames" } else { "expansions" },
        cfg.ga.harvest.name(),
        cfg.ga.learn,
        cfg.ga.params
    );
    if args.pipe.is_some() {
        println!("speed: {:?}", cfg.speed);
    }
    let mut search = Search::new(&game_gen, tree, cfg, args.seed)?;
    match &args.pipe {
        // Speed mode: the route replaces the menu-skip seed (it has its own boot).
        Some(dir) => {
            let (inputs, node, frame, stars) = crate::speed::load_pipe(dir, args.pipe_node)?;
            search.pipe(dir, node, inputs, frame, stars)?;
        }
        None => {
            search.seed(skip_menu_pads())?;
        }
    }
    for path in &args.seeds {
        println!("{path}: {} cells", search.seed(load_m64(path)?)?);
    }
    let mut sharer = share.then(|| Sharer::new(args.seed));

    let start = Instant::now();
    let (mut last_print, mut last_write) = (Instant::now(), Instant::now());
    let (mut last_cells, mut last_frames, mut last_replaced, mut last_shares) = (0, 0, 0, (0, 0));
    let mut iter = 0u64;
    let done = |iter: u64, frames: u64| match (args.iterations, args.frames) {
        (Some(n), _) if iter >= n => true,
        (_, Some(f)) if frames >= f => true,
        (None, None) => start.elapsed().as_secs_f64() >= args.minutes * 60.0,
        _ => false,
    };
    while !done(iter, search.frames) {
        search.iterate()?;
        iter += 1;
        if let Some(sh) = sharer.as_mut()
            && iter % 100 == 0
        {
            sh.share(&mut search)?;
        }
        let dt = last_print.elapsed().as_secs_f64();
        if dt > 5.0 {
            let a = &search.archive;
            let best: Vec<String> = search.best().iter().map(|(n, f)| format!("{n}:{}", f.map_or("-".into(), |f| f.to_string()))).collect();
            println!(
                "[{:.0}s] cells {} ({:.0}/s) frames {} ({:.0}/s) replaced {} ({:.0}/Mframe) saves {}MB ({} of {} nodes) | {}",
                start.elapsed().as_secs_f64(),
                a.cells.len(),
                (a.cells.len() - last_cells) as f64 / dt,
                search.frames,
                (search.frames - last_frames) as f64 / dt,
                a.replaced - last_replaced,
                search.improved_per_mframe(),
                a.save_bytes() >> 20,
                a.checkpoints(),
                a.nodes.len(),
                best.join(" ")
            );
            if let Some(sh) = &sharer {
                println!("  shares {} -> {} cells added/improved", sh.attempts - last_shares.0, sh.changed - last_shares.1);
                last_shares = (sh.attempts, sh.changed);
            }
            println!("  groups (cells/expansions/progress%): {}", search.groups_line());
            println!("  operators (expansions/+added/^improved): {}", search.operators_line());
            if let Some(sp) = &search.speed {
                let (q, seen) = sp.queue_len();
                let p = &sp.prop;
                println!("  speed (milestone:best(saved)): {} | prop tried {} ^{} frames {} queue {q} seen {seen}", sp.line(&search.archive), p.tried, p.improved, p.frames);
            }
            (last_cells, last_frames, last_replaced) = (a.cells.len(), search.frames, a.replaced);
            last_print = Instant::now();
        }
        if last_write.elapsed().as_secs() > 30 {
            println!("calibration (p decile: realised rate (n)): {}", search.calibration());
            search.write(&dir)?;
            last_write = Instant::now();
        }
    }
    search.write(&dir)?;
    println!("calibration: {}", search.calibration());
    println!("operators (expansions/+added/^improved): {}", search.operators_line());
    println!(
        "done: {iter} iterations, {} rollouts, {} cells, {} frames, {:.1}s ({:.0} frames/s), replaced {} ({:.1}/Mframe)",
        search.rollouts,
        search.archive.cells.len(),
        search.frames,
        start.elapsed().as_secs_f64(),
        search.frames as f64 / start.elapsed().as_secs_f64(),
        search.archive.replaced,
        search.improved_per_mframe()
    );
    Ok(())
}
