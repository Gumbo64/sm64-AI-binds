//! The curiosity run (`src/population.rs` with `CuriosityScoring`) with a
//! choice of input noise, seeded so runs can be repeated and compared. Same
//! parameters as `examples/curiosity.rs` (population 128, genome 600, 1 frame
//! committed per iteration, 10 hot-start iterations). Every genome is scored
//! in the game.
//!
//! cargo run --release --bin noise_bench -- <rom> <variant> [frames=54000]
//!     [out_dir=runs-bench/<variant>] [seed=1] [threads=16]
//!
//! Variants (`src/noise.rs`): `white` (the original), `pink`, `brown`,
//! `brown-stick`, `walk`, `pink-block`, `white-block`; `stats` as the sixth
//! argument prints the variant's sampler statistics instead of running.
//!
//! Writes `path.m64`, `frames.csv`, `visits.csv`, `buckets.csv` and
//! `summary.json` into `out_dir` (with `-<frame>` copies every 3000 frames).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::time::Instant;

use anyhow::{Result, bail};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use sm64_ai::curiosity_scoring::{BucketKey, CuriosityScoring};
use sm64_ai::noise::{Slot, Variant};
use sm64_ai::population::Population;
use sm64_ai::scoring::ScoringStrategy;
use sm64_ai::util::{skip_menu_pads, write_gamepads_to_file};
use sm64_binds::buttons::A_BUTTON;
use sm64_binds::{GamePad, GameState, SM64GameGenerator};

const POP: usize = 128;
const GENOME: usize = 600;
const HOT_START: usize = 10;
const CHECKPOINT: usize = 3000;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        bail!("usage: noise_bench <rom> <variant> [frames] [out_dir] [seed] [threads]");
    }
    let Some(variant) = Variant::named(&args[1]) else {
        bail!("unknown variant {}", args[1]);
    };
    let frames: usize = args
        .get(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(54_000);
    let out = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| format!("runs-bench/{}", args[1]));
    let seed: u64 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(1);
    if args.get(5).map(String::as_str) == Some("stats") {
        sampler_stats(variant, seed);
        return Ok(());
    }
    let threads: usize = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(16);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    fs::create_dir_all(&out)?;
    let t0 = Instant::now();

    let generator = SM64GameGenerator::from_file(&args[0])?;
    let path = skip_menu_pads();
    let mut main_game = generator.create_game()?;
    let mut scoring = CuriosityScoring::default();
    for p in &path {
        main_game.step_game(*p)?;
        scoring.add_state(main_game.get_game_state()?);
    }
    let mut pop = Population::with_options(POP, GENOME, scoring, path, &generator, variant, seed)?;
    println!(
        "setup {:.1}s ({threads} threads, {} backend)",
        t0.elapsed().as_secs_f64(),
        main_game.backend()
    );

    let mut genomes_scored = 0u64;
    let mut visits: HashMap<BucketKey, u32> = HashMap::new();
    let mut csv = fs::File::create(format!("{out}/frames.csv"))?;
    writeln!(
        csv,
        "frame,level,area,course,x,y,z,stars,coins,lives,health,best,genomes,wall"
    )?;
    let mut first_level: Vec<(i32, usize)> = Vec::new();
    let mut deaths = 0;
    let mut prev_lives = None;
    let mut max_coins = 0;

    let mut iteration = 0usize;
    let mut committed = 0usize;
    while committed < frames {
        let best = pop.iterate()?;
        genomes_scored += POP as u64;
        iteration += 1;
        if iteration <= HOT_START {
            continue;
        }

        // --- commit 1 frame ---
        pop.base_step_n(1)?;
        main_game.step_game(*pop.base_path.last().unwrap())?;
        let st = main_game.get_game_state()?;
        *visits.entry(pop.scoring().get_bucket_key(&st)).or_insert(0) += 1;
        committed += 1;
        if !first_level.iter().any(|&(l, _)| l == st.level_num) {
            first_level.push((st.level_num, committed));
        }
        if prev_lives.is_some_and(|p| st.lives < p) {
            deaths += 1;
        }
        prev_lives = Some(st.lives);
        max_coins = max_coins.max(st.coins);
        writeln!(
            csv,
            "{committed},{},{},{},{},{},{},{},{},{},{},{best},{genomes_scored},{:.1}",
            st.level_num,
            st.area_index,
            st.course_num,
            st.pos[0],
            st.pos[1],
            st.pos[2],
            st.num_stars,
            st.coins,
            st.lives,
            st.health,
            t0.elapsed().as_secs_f64()
        )?;
        if committed % 300 == 0 {
            println!(
                "frame {committed}: level {} stars {} coins {} | voxels {} | {:.1} fr/s",
                st.level_num,
                st.num_stars,
                st.coins,
                visits.len(),
                committed as f64 / t0.elapsed().as_secs_f64()
            );
        }
        if committed % CHECKPOINT == 0 || committed == frames {
            csv.flush()?;
            let tag = if committed == frames {
                String::new()
            } else {
                format!("-{committed}")
            };
            let run = Run {
                first_level: &first_level,
                deaths,
                max_coins,
                genomes_scored,
                committed,
                t0,
            };
            write_outputs(&out, &tag, &pop.base_path, &visits, pop.scoring(), &run, &st)?;
        }
    }
    println!(
        "done: {committed} frames, {} voxels, {genomes_scored} genomes ({:.0}s)",
        visits.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

struct Run<'a> {
    first_level: &'a [(i32, usize)],
    deaths: u32,
    max_coins: i32,
    genomes_scored: u64,
    committed: usize,
    t0: Instant,
}

fn write_outputs(
    out: &str,
    tag: &str,
    path: &[GamePad],
    visits: &HashMap<BucketKey, u32>,
    scoring: &CuriosityScoring,
    run: &Run,
    st: &GameState,
) -> Result<()> {
    write_gamepads_to_file(path, &format!("{out}/path{tag}.m64"))?;
    let mut s = String::from("level,x,y,z,visits\n");
    for ((l, x, y, z), c) in visits {
        writeln!(s, "{l},{x},{y},{z},{c}")?;
    }
    fs::write(format!("{out}/visits{tag}.csv"), s)?;
    let mut s = String::from("level,x,y,z,count\n");
    for ((l, x, y, z), c) in &scoring.bucket_counts {
        writeln!(s, "{l},{x},{y},{z},{c}")?;
    }
    fs::write(format!("{out}/buckets{tag}.csv"), s)?;
    let levels: Vec<String> = run
        .first_level
        .iter()
        .map(|(l, f)| format!("[{l},{f}]"))
        .collect();
    let json = format!(
        "{{\"frames\":{},\"stars\":{},\"coins\":{},\"max_coins\":{},\"lives\":{},\"deaths\":{},\
         \"level\":{},\"pos\":[{},{},{}],\"first_level\":[{}],\"voxels\":{},\"buckets\":{},\
         \"genomes\":{},\"wall_secs\":{:.1}}}\n",
        run.committed,
        st.num_stars,
        st.coins,
        run.max_coins,
        st.lives,
        run.deaths,
        st.level_num,
        st.pos[0],
        st.pos[1],
        st.pos[2],
        levels.join(","),
        visits.len(),
        scoring.bucket_counts.len(),
        run.genomes_scored,
        run.t0.elapsed().as_secs_f64()
    );
    fs::write(format!("{out}/summary{tag}.json"), json)?;
    Ok(())
}

/// Per-frame distribution and frame-to-frame correlation of a sampler.
fn sampler_stats(v: Variant, seed: u64) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n = 200_000;
    let mut slot = Slot::new(&mut rng, v);
    let pads: Vec<GamePad> = (0..n).map(|_| slot.next(&mut rng)).collect();
    let xs: Vec<f64> = pads.iter().map(|p| p.stick_x as f64).collect();
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let ac = |lag: usize| {
        xs.windows(lag + 1)
            .map(|w| (w[0] - mean) * (w[lag] - mean))
            .sum::<f64>()
            / ((n - lag) as f64 * var)
    };
    let duty = |b: u16| pads.iter().filter(|p| p.button & b != 0).count() as f64 / n as f64;
    let presses = pads
        .windows(2)
        .filter(|w| w[1].button & A_BUTTON != 0 && w[0].button & A_BUTTON == 0)
        .count();
    println!(
        "stick x: mean {mean:.1} sd {:.1} (uniform: -0.5, 73.9) | autocorr lag1 {:.3} lag10 {:.3} lag100 {:.3} | \
         A {:.2}% | A presses per 1000 frames {:.1}, mean hold {:.1} frames",
        var.sqrt(),
        ac(1),
        ac(10),
        ac(100),
        100.0 * duty(A_BUTTON),
        1000.0 * presses as f64 / n as f64,
        duty(A_BUTTON) * n as f64 / presses.max(1) as f64
    );
}
