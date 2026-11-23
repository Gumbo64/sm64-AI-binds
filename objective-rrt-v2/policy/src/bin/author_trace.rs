//! author_trace <rom> <problem> <file.pol|--inv dir id> <case idx|all> [--dir D]: per-frame trace of a policy on a start (author tool).
use anyhow::Result;
use obrrt_core::read_state;
use obrrt_gp::problems::load;
use obrrt_policy::dsl;
#[path = "../author_env.rs"]
mod author_env;
use obrrt_policy::inv::{Eval, Inventory, instance_of};
use sm64_binds::SM64GameGenerator;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let gg = SM64GameGenerator::from_file(&args[1])?;
    let pb = load(&args[2], ".", &gg, false)?;
    let env = author_env::env();
    let ev = Eval::new(&pb, &env);
    let dir = args.iter().position(|a| a == "--dir").map(|i| args[i + 1].clone()).unwrap_or("policy/runs/inv-author".into());
    let inv = Inventory::open(std::path::Path::new(&dir))?;
    let pol = if args[3].ends_with(".pol") { dsl::parse(&std::fs::read_to_string(&args[3])?)? } else { inv.bodies.get(&args[3]).cloned().ok_or_else(|| anyhow::anyhow!("no policy"))? };
    let mut gs = vec![gg.create_game()?];
    let idx: Vec<usize> = if args[4] == "all" { (0..pb.cases.len()).collect() } else { vec![args[4].parse()?] };
    let ps = ev.run(&mut gs, &inv.bodies, &pol.body, &idx, &instance_of(&pol), true)?;
    for (k, p) in idx.iter().zip(ps.iter()) {
        let c = &pb.cases[*k];
        println!("== case {} {} done {:?} end {:?}", k, c.name, p.out.done, p.out.end);
        let g = &mut gs[0];
        g.load_state(&c.save)?;
        for q in &c.noise { g.step_game(*q)?; }
        let st = &c.start;
        println!("start act {} fv {:.1} pos ({:.0},{:.0},{:.0}) yaw {} cam {}", dsl::action_name(st.action), st.forward_vel, st.pos[0], st.pos[1], st.pos[2], st.yaw, st.cam_yaw);
        if args.len() > 5 && args[5] == "-q" { continue; }
        for (t, pad) in p.out.pads.iter().enumerate() {
            g.step_game(*pad)?;
            let s = read_state(g, c.frame as u32 + t as u32 + 1, false);
            if let Some(i) = args.iter().position(|a| a == "-o") {
                if args[i + 1].parse::<usize>().ok() == Some(t + 1) {
                    let so = read_state(g, c.frame as u32 + t as u32 + 1, true);
                    for o in so.objects() {
                        let d = ((o.pos[0] - s.pos[0]).powi(2) + (o.pos[2] - s.pos[2]).powi(2)).sqrt();
                        if d < 700.0 { println!("    obj {:<28} d {:>5.0} pos ({:.0},{:.0},{:.0}) act {} it {:#x}", o.bhv_name(), d, o.pos[0], o.pos[1], o.pos[2], o.action, o.interact_type); }
                    }
                }
            }
            println!("{:>3} {:<22} fv {:>6.1} vy {:>6.1} pos ({:>6.0},{:>5.0},{:>6.0}) ny {:.2} btn {:04x} st ({:>4},{:>4})", t + 1, dsl::action_name(s.action), s.forward_vel, s.vel[1], s.pos[0], s.pos[1], s.pos[2], s.floor_normal[1], pad.button, pad.stick_x, pad.stick_y);
        }
    }
    Ok(())
}
