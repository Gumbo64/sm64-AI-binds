use sm64_ai::population::Population;
use sm64_ai::util::skip_menu_pads;
use sm64_ai::scoring::ScoringStrategy;
use sm64_ai::curiosity_scoring::CuriosityScoring;
use anyhow::{Result, Error};
use std::time::Instant;

fn main() -> Result<(), Error> {
    // smaller than curiosity.rs (pop128, genlen600)
    let pop_size = 128;
    let genome_length = 60;
    let step_amount = 1;
    let setup_iters = 2;
    let bench_iters = 2000;

    let t = Instant::now();
    let base_path = skip_menu_pads();
    eprintln!("skip_menu_pads: {:.3?} ({} pads)", t.elapsed(), base_path.len());

    let scoring: CuriosityScoring = CuriosityScoring::default();

    eprintln!("Population::new ...");
    let t = Instant::now();
    let mut pop = Population::new(pop_size, genome_length, scoring, base_path)?;
    let setup = t.elapsed();
    eprintln!("Population::new: {:.3?}", setup);

    for _ in 0..setup_iters {
        pop.iterate()?;
        pop.base_step_n(step_amount)?;
    }

    let mut iterate_times = Vec::new();
    let mut step_times = Vec::new();
    let mut fitnesses = Vec::new();
    for _ in 0..bench_iters {
        let t = Instant::now();
        let best = pop.iterate()?;
        iterate_times.push(t.elapsed().as_nanos() as u128);

        let t = Instant::now();
        pop.base_step_n(step_amount)?;
        step_times.push(t.elapsed().as_nanos() as u128);

        fitnesses.push(best);
    }
    fitnesses.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut sorted = iterate_times.clone();
    sorted.sort_unstable();
    let mut step_sorted = step_times.clone();
    step_sorted.sort_unstable();
    println!(
        "iterate: p50={:.3}ms mean={:.3}ms (n={})",
        sorted[sorted.len() / 2] as f64 / 1e6,
        iterate_times.iter().sum::<u128>() as f64 / (bench_iters as f64 * 1e6),
        bench_iters
    );
    println!(
        "base_step_n ({} pads): p50={:.3}ms mean={:.3}ms",
        step_amount,
        step_sorted[step_sorted.len() / 2] as f64 / 1e6,
        step_times.iter().sum::<u128>() as f64 / (bench_iters as f64 * 1e6)
    );
    println!(
        "best_fitness: last={:.4} median={:.4}",
        fitnesses.last().unwrap(),
        fitnesses[fitnesses.len() / 2]
    );
    Ok(())
}
