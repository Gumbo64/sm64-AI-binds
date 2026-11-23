//! author_eval <rom> <problem> <file.pol|id> [-v] [--dir D] [--threads n]: policy eval with object-aware named fields.
use anyhow::Result;
use obrrt_gp::problems::load;
use obrrt_policy::dsl;
use obrrt_policy::inv::{Eval, Inventory, instance_of};
use sm64_binds::SM64GameGenerator;
#[path = "../author_env.rs"]
mod author_env;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let gg = SM64GameGenerator::from_file(&args[1])?;
    let pb = load(&args[2], ".", &gg, false)?;
    let env = author_env::env();
    let ev = Eval::new(&pb, &env);
    let dir = args.iter().position(|a| a == "--dir").map(|i| args[i + 1].clone()).unwrap_or("policy/runs/inv-author".into());
    let threads: usize = args.iter().position(|a| a == "--threads").map_or(1, |i| args[i + 1].parse().unwrap());
    let inv = Inventory::open(std::path::Path::new(&dir))?;
    let pol = if args[3].ends_with(".pol") { dsl::parse(&std::fs::read_to_string(&args[3])?)? } else { inv.bodies.get(&args[3]).cloned().ok_or_else(|| anyhow::anyhow!("no policy"))? };
    let mut gs: Vec<_> = (0..threads).map(|_| gg.create_game()).collect::<Result<_, _>>()?;
    let idx: Vec<usize> = (0..pb.cases.len()).collect();
    let ps = ev.run(&mut gs, &inv.bodies, &pol.body, &idx, &instance_of(&pol), false)?;
    for sp in ["tr", "VA", "TE"] {
        let rows: Vec<_> = idx.iter().zip(&ps).filter(|(i, _)| obrrt_policy::bench::split(&pb.cases[**i]) == sp).collect();
        if rows.is_empty() { continue; }
        let succ = rows.iter().filter(|(_, p)| p.out.done.is_some()).count();
        let fr = rows.iter().filter_map(|(_, p)| p.out.done).sum::<usize>() as f64 / succ.max(1) as f64;
        let fit = rows.iter().map(|(_, p)| p.out.fitness).sum::<f64>() / rows.len() as f64;
        println!("{}: {sp} success {succ}/{} mean frames-to-Done {fr:.1} mean fitness {fit:.1}", pol.name, rows.len());
    }
    if args.iter().any(|a| a == "-v") {
        for (i, p) in idx.iter().zip(&ps) {
            let c = &pb.cases[*i];
            let mut top: Vec<_> = p.desc.fired.iter().collect();
            top.sort_by(|a, b| b.1.cmp(a.1));
            println!("  {:>2} {:<3} {:<36} {:>4} {:?} fit {:>5.0} end {} ({:.0},{:.0},{:.0}) {}", i, obrrt_policy::bench::split(c), c.name, p.out.done.map_or("-".to_string(), |d| d.to_string()), p.out.end, p.out.fitness, dsl::action_name(p.desc.end_action), p.desc.end_pos[0], p.desc.end_pos[1], p.desc.end_pos[2], top.iter().take(5).map(|(k, v)| format!("{k}x{v}")).collect::<Vec<_>>().join(","));
        }
    }
    Ok(())
}
