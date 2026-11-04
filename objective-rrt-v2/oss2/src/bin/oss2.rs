//! `oss2 <schema.json> [minutes] [seed] [--seed file.m64]... [--noise NAME] [--iterations N] [--frames N] [--out DIR] [--rom FILE]`
//! (default pink noise). Writes runs/<schema stem>/{schema.json, objectives.json, stats.json, cells.jsonl, tree.bin, ...}.
//! See `oss2::run` and README.md (`OSS_GA` etc.).

fn main() -> anyhow::Result<()> {
    // OBRRT_SPEC: load (and check) the declarative objectives up front.
    obrrt_spec::from_env()?;
    oss2::run::run(oss2::run::Args::parse(std::env::args())?, false)
}
