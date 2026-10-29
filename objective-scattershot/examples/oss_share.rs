//! `cargo run --release --example oss_share -- schemas/zero_star.json [minutes] [seed] [--seed file.m64]... [--noise NAME] [--iterations N] [--out DIR]`
//! `oss`, plus input sharing between states that entered the same zone (every 100 batches).
//! Writes runs/<schema stem>-share/. See `objective_scattershot::run`.

fn main() -> anyhow::Result<()> {
    objective_scattershot::run::run(objective_scattershot::run::Args::parse(std::env::args())?, true)
}
