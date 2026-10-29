//! `cargo run --release --example oss -- schemas/zero_star.json [minutes] [seed] [--seed file.m64]... [--noise white|pink|pink-block|white-block|brown|brown-stick|walk] [--iterations N] [--out DIR]`
//! (default pink). Writes runs/<schema stem>/{schema.json, objectives.json, stats.json, cells.jsonl, tree.bin}.
//! See `objective_scattershot::run`.

fn main() -> anyhow::Result<()> {
    objective_scattershot::run::run(objective_scattershot::run::Args::parse(std::env::args())?, false)
}
