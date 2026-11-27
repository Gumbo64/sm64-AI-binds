//! The open-loop GA (with the objective's overlay) on a problem's cases, as a
//! baseline: gabase <rom> <problem> [threads=4] [seeds=1]. Every start needs its
//! own search; it doesn't generalise. Prints frames to Done and emulator frames.

use anyhow::Result;
use obrrt_gp::problems::load;
use obrrt_search::ga::{GaParams, Worker};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use sm64_binds::SM64GameGenerator;

fn main() -> Result<()> {
    unsafe {
        std::env::set_var("OBRRT_GROUNDS_FIX", "1");
    }
    obrrt_objectives::moves::install_controller();
    let args: Vec<String> = std::env::args().collect();
    let gg = SM64GameGenerator::from_file(&args[1])?;
    let pb = load(&args[2], ".", &gg, false)?;
    let threads: usize = args.get(3).map_or(Ok(4), |s| s.parse())?;
    let seeds: u64 = args.get(4).map_or(Ok(1), |s| s.parse())?;
    let params = GaParams { scale: 3.0, pink: 0.0 };
    let n = pb.cases.len() * seeds as usize;
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Option<(Option<usize>, u64)>>> = (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let mut w = Worker::new(&gg).expect("worker");
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let c = &pb.cases[i % pb.cases.len()];
                    let obj = pb.tasks[c.task].objective().expect("objective");
                    w.base.load_state(&c.save).expect("load");
                    let st = obrrt_core::read_state(&w.base, c.frame as u32, true);
                    let mut rng = SmallRng::seed_from_u64(i as u64 + 1);
                    let att = w.attempt(&[obj.as_ref()], &st, &c.save, &mut rng, params).expect("attempt");
                    let seg = att.segments.first();
                    let done = seg.filter(|s| s.success).map(|s| s.played);
                    *out[i].lock().unwrap() = Some((done, att.frames));
                }
            });
        }
    });
    let (mut tr, mut te) = (vec![], vec![]);
    for (i, o) in out.iter().enumerate() {
        let (d, f) = o.lock().unwrap().unwrap();
        let c = &pb.cases[i % pb.cases.len()];
        println!("{:<3} {:<44} {:>5} {:>10}", if c.train { "tr" } else { "TE" }, c.name, d.map_or("-".to_string(), |d| d.to_string()), f);
        if c.train { tr.push((d, f)) } else { te.push((d, f)) }
    }
    for (l, v) in [("train", &tr), ("test", &te)] {
        let s = v.iter().filter(|x| x.0.is_some()).count();
        let fr = v.iter().filter_map(|x| x.0).sum::<usize>() as f64 / s.max(1) as f64;
        println!("GA {l}: success {s}/{} mean frames {fr:.1} emu per attempt {:.2} M", v.len(), v.iter().map(|x| x.1).sum::<u64>() as f64 / v.len().max(1) as f64 / 1e6);
    }
    Ok(())
}
