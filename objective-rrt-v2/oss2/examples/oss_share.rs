//! `cargo run --release --example oss_share -- schemas/zero_star.json [minutes] [seed] [--seed file.m64]... [--noise NAME] [--iterations N] [--out DIR]`
//! `oss`, plus input sharing between states that entered the same zone (every 100 batches).
//! Writes runs/<schema stem>-share/. See `oss2::run`.

fn main() -> anyhow::Result<()> {
    oss2::run::run(oss2::run::Args::parse(std::env::args())?, true)
}
