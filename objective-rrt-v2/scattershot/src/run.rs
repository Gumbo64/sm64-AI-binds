//! The command-line run shared by the `oss` and `oss_share` examples:
//!
//! `<schema.json> [minutes] [seed] [--seed file.m64]... [--noise NAME] [--iterations N] [--out DIR]`
//!
//! Writes `runs/<schema stem>[-share]/{schema.json, objectives.json, stats.json,
//! cells.jsonl, tree.bin}` every 30 s and at the end, and logs every 5 s.
//! `--iterations N` stops after N batches instead of after `minutes` (for
//! reproducible comparisons: a seed, a thread count and N give one run).

use obrrt_core::sys::Instant;

use anyhow::{Context, Result, bail};
use crate::noise::Variant;
use crate::util::skip_menu_pads;
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
}

impl Args {
    pub fn parse(args: impl Iterator<Item = String>) -> Result<Args> {
        let mut it = args.skip(1);
        let (mut pos, mut seeds, mut noise, mut iterations, mut out) = (Vec::new(), Vec::new(), "pink".to_string(), None, None);
        while let Some(a) = it.next() {
            let mut val = || it.next().with_context(|| format!("{a} needs a value"));
            match a.as_str() {
                "--seed" => seeds.push(val()?),
                "--noise" => noise = val()?,
                "--iterations" => iterations = Some(val()?.parse()?),
                "--out" => out = Some(val()?),
                s if s.starts_with("--") => bail!("unknown option {s}"),
                _ => pos.push(a),
            }
        }
        let Some(schema) = pos.first().cloned() else { bail!("usage: <schema.json> [minutes] [seed] [--seed file.m64]... [--noise NAME] [--iterations N] [--out DIR]") };
        let minutes = pos.get(1).map_or(Ok(60.0), |s| s.parse())?;
        let seed = pos.get(2).map_or(Ok(0), |s| s.parse())?;
        Ok(Args { schema, minutes, seed, seeds, noise, iterations, out })
    }
}

pub fn run(args: Args, share: bool) -> Result<()> {
    let stem = std::path::Path::new(&args.schema).file_stem().unwrap().to_string_lossy().to_string();
    let dir = args.out.clone().unwrap_or_else(|| format!("runs/{stem}{}", if share { "-share" } else { "" }));
    obrrt_core::sys::fs::create_dir_all(&dir)?;
    let schema = obrrt_core::sys::fs::read_to_string(&args.schema)?;
    obrrt_core::sys::fs::write(format!("{dir}/schema.json"), &schema)?;
    let tree = Tree::load(&schema)?;
    obrrt_core::sys::fs::write(format!("{dir}/objectives.json"), serde_json::to_string_pretty(&tree.insts)?)?;

    let variant = Variant::named(&args.noise).with_context(|| format!("unknown noise {}", args.noise))?;
    let game_gen = SM64GameGenerator::new(obrrt_core::sys::fs::read("../baserom.us.z64")?)?;
    let mut search = Search::new(&game_gen, tree, Config::from_env(variant), args.seed)?;
    search.seed(skip_menu_pads())?;
    for path in &args.seeds {
        println!("{path}: {} cells", search.seed(load_m64(path)?)?);
    }
    let mut sharer = share.then(|| Sharer::new(args.seed));

    let start = Instant::now();
    let (mut last_print, mut last_write) = (Instant::now(), Instant::now());
    let (mut last_cells, mut last_frames, mut last_replaced, mut last_shares) = (0, 0, 0, (0, 0));
    let mut iter = 0u64;
    let done = |iter: u64| match args.iterations {
        Some(n) => iter >= n,
        None => start.elapsed().as_secs_f64() >= args.minutes * 60.0,
    };
    while !done(iter) {
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
                "[{:.0}s] cells {} ({:.0}/s) frames/s {:.0} replaced {} saves {}MB ({} of {} nodes) | {}",
                start.elapsed().as_secs_f64(),
                a.cells.len(),
                (a.cells.len() - last_cells) as f64 / dt,
                (search.frames - last_frames) as f64 / dt,
                a.replaced - last_replaced,
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
    println!("done: {iter} iterations, {} rollouts, {} cells, {} frames", search.rollouts, search.archive.cells.len(), search.frames);
    Ok(())
}
