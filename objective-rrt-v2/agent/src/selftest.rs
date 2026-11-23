//! `ar selftest`: the framework's own checks on a session (read-only for
//! the session; scratch files go to a temp dir).
//!
//! 1. checkpoints: `load_at(f)` gives the same save state bytes as a replay
//!    from power-on, at random frames of HEAD's path;
//! 2. determinism: the controller commits the same pads with 1 and N threads;
//! 3. population journal: every recorded generation reconstructs exactly;
//! 4. rewind: a controller restored at a frame from the journal commits the
//!    same pads as the original run did from there.

use anyhow::{Result, anyhow, bail};
use obrrt_core::{GamePad, Objective};
use obrrt_search::ga::rhstep::{PlanRec, RhController};
use rand::{Rng, SeedableRng};
use serde_json::Value;

use crate::ckpt;
use crate::objective;
use crate::popjournal;
use crate::session::Session;

fn same(a: &PlanRec, b: &PlanRec) -> bool {
    a.pads == b.pads && a.raw == b.raw && a.origin == b.origin && a.generation == b.generation && a.fitness.to_bits() == b.fitness.to_bits() && a.at == b.at && a.first_done == b.first_done && a.end == b.end && a.final_eta.to_bits() == b.final_eta.to_bits()
}

pub fn run(s: &Session, sel: Option<Value>, steps: usize, threads: usize, samples: usize) -> Result<()> {
    let gg = sm64_binds::SM64GameGenerator::from_file(&s.cfg.rom)?;
    let head = s.head()?;
    let pads = s.path_pads(&head, true)?;
    let mut ck = ckpt::Store::open(&s.dir)?;
    let mut fails = 0;

    // 1. checkpoints vs replay from power-on.
    let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
    let mut frames: Vec<u32> = (0..samples).map(|_| rng.random_range(1..=pads.len() as u32)).collect();
    frames.push(pads.len() as u32);
    frames.sort();
    let mut replay = gg.create_game()?;
    let mut at = 0u32;
    let mut game = gg.create_game()?;
    let mut ok = 0;
    for &f in &frames {
        while at < f {
            replay.step_game(pads[at as usize])?;
            at += 1;
        }
        s.load_at(&mut ck, &mut game, &head, Some(f))?;
        if game.save_state().as_bytes() == replay.save_state().as_bytes() {
            ok += 1;
        } else {
            println!("  checkpoint MISMATCH at frame {f}");
            fails += 1;
        }
    }
    println!("1. checkpoints: {ok}/{} frames byte-identical to a replay from power-on ({} checkpoints on disk)", frames.len(), ck.entries.len());

    // The objective for 2-4: the given selector, else the queue's front.
    let st0 = s.load_at(&mut ck, &mut game, &head, None)?;
    let save = game.save_state();
    let sel = match sel {
        Some(v) => v,
        None => {
            let q = s.queue()?;
            let id = q.first().ok_or_else(|| anyhow!("selftest: the queue is empty; pass --sel"))?;
            s.objectives()?.into_iter().find(|o| &o.id == id).unwrap().sel
        }
    };
    let o = objective::resolve(&sel, "selftest", &st0, &[])?.ok_or_else(|| anyhow!("selftest: {sel} not offered at frame {}", st0.frame))?;
    let stages: Vec<&dyn Objective> = vec![o.as_ref()];
    let run_ctl_b = |back: usize, threads: usize, journal: Option<&std::path::Path>, restore: Option<&[PlanRec]>, from: (&sm64_binds::SaveState, &obrrt_core::St), n: usize| -> Result<(Vec<GamePad>, Vec<(u32, Vec<PlanRec>)>)> {
        let mut cfg = crate::serve::with_session(s, s.cfg.rh.clone());
        cfg.back = back;
        let mut ctl = RhController::new(&gg, threads, from.0, from.1.clone(), cfg)?;
        ctl.journal = journal.is_some();
        if let Some(p) = restore {
            ctl.set_population(p);
        }
        let mut w = match journal {
            Some(p) => Some(popjournal::Writer::open(p, s.cfg.keyframe_every.min(8))?),
            None => None,
        };
        let mut out = Vec::new();
        let mut snaps = Vec::new();
        for _ in 0..n {
            let f = ctl.state().frame;
            let r = ctl.step(&stages, None)?;
            if let Some(w) = &mut w {
                for g in &r.gens {
                    w.record(ctl.generation, f, g)?;
                    snaps.push((f, g.clone()));
                }
            }
            if r.stuck {
                break;
            }
            out.extend(r.pads);
            if ctl.met() > 0 {
                break;
            }
        }
        if back > 0 {
            // The window's base rarely moves in a short test: frozen + tentative.
            out = ctl.state_frame_pads(&out);
        }
        if let Some(w) = &mut w {
            w.flush()?;
        }
        Ok((out, snaps))
    };
    let run_ctl = |threads: usize, journal: Option<&std::path::Path>, restore: Option<&[PlanRec]>, from: (&sm64_binds::SaveState, &obrrt_core::St), n: usize| run_ctl_b(0, threads, journal, restore, from, n);
    let t0 = std::time::Instant::now();
    let tmp = std::env::temp_dir().join(format!("ar-selftest-{}", std::process::id()));
    std::fs::create_dir_all(&tmp)?;
    let jp = tmp.join("t.pj");
    let (a, snaps) = run_ctl(threads, Some(&jp), None, (&save, &st0), steps)?;
    let secs = t0.elapsed().as_secs_f64();
    let (b, _) = run_ctl(1, None, None, (&save, &st0), steps)?;
    if a == b {
        println!("2. determinism: {} committed pads identical with {threads} and 1 threads ({:.1} s for {threads} threads)", a.len(), secs);
    } else {
        let d = a.iter().zip(&b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
        println!("2. determinism: MISMATCH ({threads} threads {} pads, 1 thread {} pads, first difference at {d})", a.len(), b.len());
        fails += 1;
    }

    // 2b. determinism with the window's backward part (back_frames, anchored share).
    let back = s.cfg.back_frames as usize;
    if back > 0 {
        let (x, _) = run_ctl_b(back, threads, None, None, (&save, &st0), steps)?;
        let (y, _) = run_ctl_b(back, 1, None, None, (&save, &st0), steps)?;
        if x == y && !x.is_empty() {
            println!("2b. determinism (back {back}): the {} frames of frozen + tentative route identical with {threads} and 1 threads", x.len());
        } else {
            println!("2b. determinism (back {back}): MISMATCH ({} vs {} frames)", x.len(), y.len());
            fails += 1;
        }
    }

    // 3. journal reconstruction.
    let heads = popjournal::index(&jp)?;
    let mut good = 0;
    for (i, (f, snap)) in snaps.iter().enumerate() {
        let r = popjournal::read_at(&jp, &heads, i)?;
        if heads[i].frame == *f && r.len() == snap.len() && r.iter().zip(snap).all(|(x, y)| same(x, y)) {
            good += 1;
        } else {
            println!("  journal MISMATCH at record {i}");
            fails += 1;
        }
    }
    let size = std::fs::metadata(&jp)?.len();
    println!("3. journal: {good}/{} generations reconstruct exactly ({} keyframes, {} bytes, {:.0} bytes/generation)", snaps.len(), heads.iter().filter(|h| h.kind == 0).count(), size, size as f64 / snaps.len().max(1) as f64);

    // 4. rewind: restore at the middle record and continue.
    if a.len() >= 4 {
        let k = a.len() / 2;
        let mut g = gg.create_game()?;
        g.load_state(&save)?;
        for p in &a[..k] {
            g.step_game(*p)?;
        }
        let stk = obrrt_core::read_state(&g, st0.frame + k as u32, true);
        let i = popjournal::find_frame(&heads, stk.frame).ok_or_else(|| anyhow!("no journal record at frame {}", stk.frame))?;
        let pop = popjournal::read_at(&jp, &heads, i)?;
        let (c, _) = run_ctl(threads, None, Some(&pop), (&g.save_state(), &stk), steps - k)?;
        let n = c.len().min(a.len() - k);
        if c[..n] == a[k..k + n] && n > 0 {
            println!("4. rewind: restored at frame {} from the journal, the next {n} committed pads match the original run", stk.frame);
        } else {
            let d = c.iter().zip(&a[k..]).position(|(x, y)| x != y).unwrap_or(n);
            println!("4. rewind: MISMATCH after restoring at frame {} (first difference after {d} of {n})", stk.frame);
            fails += 1;
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    if fails > 0 {
        bail!("{fails} check(s) failed");
    }
    println!("all checks passed");
    Ok(())
}
