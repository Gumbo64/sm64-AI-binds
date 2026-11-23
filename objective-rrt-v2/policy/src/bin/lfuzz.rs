//! Closed-loop fuzz of policies on the Lakitu skip, with the same start families as the hand-overlay fuzz
//! (scratchpad/overlay_fuzz.md, hill_fuzz.rs; same rng draws, so start i here is start i there):
//!   family A: the boot spawn, then `na` perturbed spawns (teleport dx/dz/yaw, 0..24 walking frames),
//!   family B: the hill-bottom snapshot (live Lakitu), then `nb` perturbed (pos +-60, speed x0.90..1.04, yaw +-0x500).
//! A policy is played with `play_policy` (the objective's own eta/Done/Lost) and then every play is replayed
//! independently from its start state: success = reached the castle / door circle with Mario never grounded in
//! Lakitu's box and Lakitu never talking.
//!   lfuzz <rom> <state-file> <na> <nb> <seed> <out.tsv> <policy.pol|id>[,more...] [--dir INVDIR] [--threads N]
use std::sync::Arc;

use anyhow::Result;
use obrrt_core::{GamePad, St, read_state};
use obrrt_gp::problems::load;
use obrrt_gp::rollout::Case;
use obrrt_policy::dsl;
use obrrt_policy::eval::Env;
use obrrt_policy::inv::{Eval, Inventory, instance_of};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sm64_binds::{SM64Game, SM64GameGenerator, SaveState};

const M: usize = obrrt_core::layout::ADDR_gMarioStates as usize;
const BOX_X: (f32, f32) = (-544.0, 545.0);
const BOX_Z: (f32, f32) = (-2000.0, -177.0);
const DOOR: [f32; 3] = [0.0, 803.0, -3155.0];

fn boot_inputs() -> Vec<GamePad> {
    use obrrt_core::buttons::{A_BUTTON, START_BUTTON};
    (0..1680)
        .map(|i| {
            let mut button = 0;
            if (150..160).contains(&i) || (200..300).contains(&i) {
                button = START_BUTTON;
            }
            if i % 2 == 0 {
                button = A_BUTTON;
            }
            GamePad::new(button, 0, 0)
        })
        .collect()
}

fn write(g: &mut SM64Game, addr: usize, bytes: &[u8]) -> Result<()> {
    let (store, _, memory) = g.wasm_parts();
    memory.view(store).write(addr as u64, bytes)?;
    Ok(())
}

fn opt<'a>(a: &'a [String], k: &str) -> Option<&'a str> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).map(|s| s.as_str())
}

fn mkcase(g: &SM64Game, name: String, s0: &St, tmpl: &Case) -> Case {
    Case {
        name,
        train: true,
        val: false,
        movie: tmpl.movie,
        frame: s0.frame as usize,
        noise: Vec::new(),
        save: Arc::new(g.save_state()),
        pre_states: vec![s0.clone()],
        pre_pads: Vec::new(),
        start: s0.clone(),
        task: tmpl.task,
        ref_pads: Vec::new(),
    }
}

/// Replay `pads` from the case's save and judge the play independently of the objective.
fn verify(g: &mut SM64Game, c: &Case, pads: &[GamePad]) -> Result<(bool, String)> {
    g.load_state(&c.save)?;
    let mut f = c.frame as u32;
    let mut sig: Vec<String> = Vec::new();
    let mut last_air_start: Option<(String, f32, f32)> = None;
    let mut prev_act = c.start.action;
    for (k, p) in pads.iter().enumerate() {
        g.step_game(*p)?;
        f += 1;
        let n = read_state(g, f, false);
        if n.action != prev_act {
            if n.forward_vel > 40.0 && sig.last().map_or(true, |l| *l != dsl::action_name(n.action)) {
                sig.push(dsl::action_name(n.action).to_string());
            }
            if n.action & 0x800 != 0 {
                last_air_start = Some((dsl::action_name(n.action).to_string(), n.pos[2], n.forward_vel));
            }
            prev_act = n.action;
        }
        let sigs = || -> String {
            let tail: Vec<_> = sig.iter().rev().take(5).rev().cloned().collect();
            format!("{}|{}", tail.join(">"), last_air_start.as_ref().map_or(String::new(), |(a, z, v)| format!("{a}@z{z:.0}v{v:.0}")))
        };
        let in_box = n.pos[0] > BOX_X.0 && n.pos[0] < BOX_X.1 && n.pos[1] > 800.0 && n.pos[2] > BOX_Z.0 && n.pos[2] < BOX_Z.1;
        if in_box && n.action & 0x800 == 0 {
            return Ok((false, format!("box_ground@{k} z{:.0} act {} :: {}", n.pos[2], dsl::action_name(n.action), sigs())));
        }
        if n.dialog == 34 || matches!(n.action, 0x0000_1308 | 0x2000_1306) {
            return Ok((false, format!("caught@{k}")));
        }
        if n.pos[1] < 700.0 && n.pos[2] < -177.0 {
            return Ok((false, format!("fell@{k} z{:.0}", n.pos[2])));
        }
        let d = ((n.pos[0] - DOOR[0]).powi(2) + (n.pos[1] - DOOR[1]).powi(2) + (n.pos[2] - DOOR[2]).powi(2)).sqrt();
        if n.level == 6 || (n.level == obrrt_core::layout::LEVEL_CASTLE_GROUNDS as u16 && d < 350.0) {
            return Ok((true, format!("ok@{}", k + 1)));
        }
    }
    Ok((false, "no_arrival".into()))
}

fn main() -> Result<()> {
    unsafe {
        std::env::set_var("OBRRT_LAKITU_ARC", "1");
        std::env::set_var("OBRRT_GROUNDS_FIX", "1");
    }
    obrrt_objectives::moves::install_controller();
    let a: Vec<String> = std::env::args().collect();
    let gg = SM64GameGenerator::from_file(&a[1])?;
    let sbytes = std::fs::read(&a[2])?;
    let (na, nb): (usize, usize) = (a[3].parse()?, a[4].parse()?);
    let seed: u64 = a[5].parse()?;
    let out_tsv = a[6].clone();
    let pols: Vec<String> = a[7].split(',').map(|s| s.to_string()).collect();
    let threads: usize = opt(&a, "--threads").map_or(Ok(4), |s| s.parse())?;
    let dir = opt(&a, "--dir").unwrap_or("policy/runs/inv-lakitu/inv").to_string();

    // the problem, only for its task (the Lakitu hill objective) and roll settings
    let mut pb = load("lakitu_chain", ".", &gg, true)?;
    let tmpl = pb.cases.remove(0);
    pb.cases.clear();

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut g = gg.create_game()?;
    let mut f = 0u32;
    for p in boot_inputs() {
        g.step_game(p)?;
        f += 1;
    }
    let mut st = read_state(&g, f, false);
    let mut still = 0;
    while still < 10 || !st.controllable() {
        let b = st.pos;
        g.step_game(GamePad::new(0, 0, 0))?;
        f += 1;
        st = read_state(&g, f, false);
        still = if st.pos == b { still + 1 } else { 0 };
    }
    let boot_save = g.save_state();
    let boot_f = f;
    let tele = |g: &mut SM64Game, x: f32, z: f32, yaw: i16, f: &mut u32| -> Result<()> {
        let tris = g.with_memory(obrrt_core::geo::read_floors).0;
        let y = obrrt_core::geo::floor_below(&tris, x, z, 3000.0).unwrap_or(811.0);
        for _ in 0..4 {
            write(g, M + 0x3C, &[x.to_le_bytes(), y.to_le_bytes(), z.to_le_bytes()].concat())?;
            write(g, M + 0x48, &[0f32.to_le_bytes(); 3].concat())?;
            write(g, M + 0x54, &0f32.to_le_bytes())?;
            write(g, M + 0x2E, &yaw.to_le_bytes())?;
            write(g, M + 0x0C, &0x0C40_0201u32.to_le_bytes())?;
            g.step_game(GamePad::new(0, 0, 0))?;
            *f += 1;
        }
        for _ in 0..45 {
            g.step_game(GamePad::new(0, 0, 0))?;
            *f += 1;
        }
        Ok(())
    };
    // the lab's object generation point (keeps the rng/state sequence of hill_fuzz: it teleports first)
    tele(&mut g, 0.0, -100.0, 0x8000u16 as i16, &mut f)?;
    let mut cases: Vec<Case> = Vec::new();
    g.load_state(&boot_save)?;
    let st0 = read_state(&g, boot_f, true);
    let (bx, bz) = (st0.pos[0], st0.pos[2]);
    cases.push(mkcase(&g, "A0-spawn".into(), &st0, &tmpl));
    for i in 1..=na {
        g.load_state(&boot_save)?;
        let (dx, dz): (f32, f32) = (rng.random_range(-800.0..800.0), rng.random_range(-500.0..500.0));
        let yaw: i16 = rng.random_range(0..65536u32) as u16 as i16;
        let mut ff = boot_f;
        tele(&mut g, bx + dx, bz + dz, yaw, &mut ff)?;
        let pre = rng.random_range(0..25usize);
        let (sx, sy): (i8, i8) = (rng.random_range(-70..70), rng.random_range(-70..70));
        for _ in 0..pre {
            g.step_game(GamePad::new(0, sx, sy))?;
            ff += 1;
        }
        let s0 = read_state(&g, ff, true);
        cases.push(mkcase(&g, format!("A{i}"), &s0, &tmpl));
    }
    // `--skip-a N`: consume the rng draws of N family-A starts without building them, so family B's starts are the
    // same as hill_fuzz's when it was run with N family-A starts (B row i here = B row i there).
    if let Some(n) = opt(&a, "--skip-a").map(|s| s.parse::<usize>()).transpose()? {
        for _ in na + 1..=n {
            let _: (f32, f32) = (rng.random_range(-800.0..800.0), rng.random_range(-500.0..500.0));
            let _: i16 = rng.random_range(0..65536u32) as u16 as i16;
            let _ = rng.random_range(0..25usize);
            let _: (i8, i8) = (rng.random_range(-70..70), rng.random_range(-70..70));
        }
    }
    for i in 0..=nb {
        g.load_state(&SaveState::from_bytes(sbytes.clone())?)?;
        let s_base = read_state(&g, 1648, true);
        let (dx, dz, fs, dy): (f32, f32, f32, i32) =
            if i == 0 { (0.0, 0.0, 1.0, 0) } else { (rng.random_range(-60.0..60.0), rng.random_range(-60.0..60.0), rng.random_range(0.90..1.04), rng.random_range(-0x500..0x500)) };
        if i > 0 {
            let fv = s_base.forward_vel * fs;
            let yaw = (s_base.yaw as i32 + dy) as i16;
            write(&mut g, M + 0x3C, &[(s_base.pos[0] + dx).to_le_bytes(), s_base.pos[1].to_le_bytes(), (s_base.pos[2] + dz).to_le_bytes()].concat())?;
            write(&mut g, M + 0x54, &fv.to_le_bytes())?;
            write(&mut g, M + 0x2E, &yaw.to_le_bytes())?;
        }
        let s0 = read_state(&g, 1648, true);
        cases.push(mkcase(&g, format!("B{i}"), &s0, &tmpl));
    }
    eprintln!("{} starts built ({} A incl. spawn, {} B incl. base)", cases.len(), na + 1, nb + 1);
    pb.cases = cases;

    let env = Env::full();
    let ev = Eval::new(&pb, &env);
    let inv = Inventory::open(std::path::Path::new(&dir))?;
    let mut gs: Vec<SM64Game> = (0..threads).map(|_| gg.create_game()).collect::<Result<_>>()?;
    let idx: Vec<usize> = (0..pb.cases.len()).collect();
    let mut tsv = String::from("policy\tstart\tobjective\tindependent\tdetail\tframes\tmpc_frames\tfired\n");
    for what in &pols {
        let pol = if what.ends_with(".pol") { dsl::parse(&std::fs::read_to_string(what)?)? } else { inv.bodies.get(what).cloned().ok_or_else(|| anyhow::anyhow!("no policy {what}"))? };
        let t0 = std::time::Instant::now();
        let ps = ev.run(&mut gs, &inv.bodies, &pol.body, &idx, &instance_of(&pol), true)?;
        let (mut oa, mut ob, mut ia, mut ib, mut ba, mut bb) = (0, 0, 0, 0, 0, 0);
        let (mut fr, mut nfr) = (0usize, 0usize);
        let mut cause: std::collections::BTreeMap<String, usize> = Default::default();
        for (i, p) in ps.iter().enumerate() {
            let c = &pb.cases[i];
            if std::env::var("LFUZZ_TRACE").is_ok_and(|t| t == c.name) {
                for r in &p.desc.rows {
                    println!("{r}");
                }
            }
            let (ok, detail) = verify(&mut gs[0], c, &p.out.pads)?;
            let obj = p.out.done.is_some();
            if std::env::var("LFUZZ_DEBUG").is_ok() {
                eprintln!("{} {} end {:?} frames {} done {:?} start fv {:.1} act {} pos {:?}", pol.name, c.name, p.out.end, p.out.frames, p.out.done, c.start.forward_vel, dsl::action_name(c.start.action), c.start.pos);
            }
            let fam_a = c.name.starts_with('A');
            if fam_a {
                oa += obj as usize;
                ia += ok as usize;
                ba += detail.starts_with("box_ground") as usize;
            } else {
                ob += obj as usize;
                ib += ok as usize;
                bb += detail.starts_with("box_ground") as usize;
            }
            if ok {
                fr += p.out.pads.len();
                nfr += 1;
            } else {
                *cause.entry(detail.split('@').next().unwrap_or("").to_string()).or_default() += 1;
            }
            let mut fired: Vec<_> = p.desc.fired.iter().collect();
            fired.sort_by(|x, y| y.1.cmp(x.1));
            tsv.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                pol.name,
                c.name,
                obj,
                ok,
                detail,
                p.out.pads.len(),
                p.desc.mpc_frames,
                fired.iter().take(6).map(|(k, v)| format!("{k}:{v}")).collect::<Vec<_>>().join(",")
            ));
        }
        let (n_a, n_b) = (na + 1, nb + 1);
        println!(
            "{:<28} A {}/{} (objective {}/{}, box-ground {})  B {}/{} (objective {}/{}, box-ground {})  mean frames {:.0}  causes {:?}  [{:.0}s]",
            pol.name,
            ia,
            n_a,
            oa,
            n_a,
            ba,
            ib,
            n_b,
            ob,
            n_b,
            bb,
            fr as f64 / nfr.max(1) as f64,
            cause,
            t0.elapsed().as_secs_f64()
        );
    }
    std::fs::write(&out_tsv, tsv)?;
    Ok(())
}
