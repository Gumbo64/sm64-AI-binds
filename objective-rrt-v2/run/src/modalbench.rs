//! Dialog/menu input bench: replays a run's route to a node and, at every
//! dialog or menu on it, plays that option's objective with the fast
//! policies and with the old macros (`OBRRT_FAST_MODAL=0`), from the same
//! saved state, and reports the frame each closes on.
//!
//!     modalbench <rom> <run dir> [node=earliest star] [trace.csv]
//!
//! Per option: frames to close with the old macro and with the fast policy
//! (`fail` = the option wasn't the one taken), and two checks for dialogs
//! without a choice: `mash` (A and B alternated every frame: nothing can be
//! sooner) and `mario_acts_from` (the first frame whose A press makes Mario
//! jump while the box closes: the earliest the dialog can count as over;
//! -1 = not before gDialogID clears).
//!
//! With `trace.csv`, also writes every frame of the route: the node it
//! belongs to, its objective, the input, and the state after it.

use std::io::Write as _;

use anyhow::{Result, ensure};
use obrrt_core::{Budget, Generator, GamePad, Objective, St, read_state};
use obrrt_objectives::Registry;
use rand::SeedableRng;
use sm64_binds::{SM64Game, SM64GameGenerator, SaveState};

fn records(dir: &str) -> Result<Vec<(i32, Vec<GamePad>)>> {
    let bytes = std::fs::read(format!("{dir}/tree.bin"))?;
    let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
    let count = u32_at(0) as usize;
    let mut out = Vec::with_capacity(count);
    let mut at = 4;
    for _ in 0..count {
        let parent = u32_at(at) as i32;
        let n = u32_at(at + 4) as usize;
        let pads = bytes[at + 8..at + 8 + 4 * n]
            .chunks_exact(4)
            .map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8))
            .collect();
        out.push((parent, pads));
        at += 8 + 4 * n;
    }
    Ok(out)
}

/// Plays `budget` (a macro or a policy) for `obj` from `save`: the index at
/// which it's met (and Mario's action then).
fn play(game: &mut SM64Game, save: &SaveState, start: &St, obj: &dyn Objective, budget: Option<Budget>) -> Result<Option<usize>> {
    Ok(play_act(game, save, start, obj, budget)?.map(|(i, _)| i))
}

/// (v2: an objective without a fixed play, `None`, isn't met.)
fn play_act(game: &mut SM64Game, save: &SaveState, start: &St, obj: &dyn Objective, budget: Option<Budget>) -> Result<Option<(usize, u32)>> {
    let Some(budget) = budget else { return Ok(None) };
    game.load_state(save)?;
    let mut traj = vec![start.clone()];
    // v2: met is the fold of `eta` (from the start, frame by frame).
    let mut mem = obrrt_core::Mem::default();
    if obj.eta(start, start, &mut mem).done() {
        return Ok(Some((0, start.action)));
    }
    let mut played: Vec<GamePad> = Vec::new();
    let limit = budget.frames();
    for f in 0..limit {
        let pad = match &budget {
            Budget::Macro(p) => p[f],
            Budget::Policy { policy, .. } => policy.pad(&traj, &played),
        };
        game.step_game(pad)?;
        played.push(pad);
        traj.push(read_state(game, start.frame + f as u32 + 1, false));
        if obj.eta(traj.last().unwrap(), start, &mut mem).done() {
            let i = traj.len() - 1;
            return Ok(Some((i, traj[i].action)));
        }
        if obj.modal() && !obj.exists_in(traj.last().unwrap()) {
            return Ok(None);
        }
    }
    Ok(None)
}

fn main() -> Result<()> {
    // The steering controller (objectives/src/moves; move library off).
    obrrt_objectives::moves::install_controller();
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() >= 3, "usage: modalbench <rom> <run dir> [node] [trace.csv]");
    let dir = args[2].trim_end_matches('/');
    let recs = records(dir)?;
    let csv = std::fs::read_to_string(format!("{dir}/nodes.csv"))?;
    let rows: Vec<Vec<String>> = csv.lines().skip(1).map(|l| l.split(',').map(String::from).collect()).collect();
    let target: u32 = match args.get(3).filter(|s| *s != "star") {
        Some(n) => n.parse()?,
        None => rows
            .iter()
            .filter(|r| r[8].parse::<i16>().unwrap_or(0) > rows[0][8].parse::<i16>().unwrap_or(0))
            .min_by_key(|r| r[11].parse::<u32>().unwrap_or(u32::MAX))
            .map(|r| r[0].parse().unwrap())
            .ok_or_else(|| anyhow::anyhow!("no star in {dir}"))?,
    };
    let mut ids = vec![target];
    while let Some(&n) = ids.last() {
        let p = recs[n as usize + 1].0;
        if p < 0 {
            break;
        }
        ids.push(p as u32);
    }
    ids.reverse();
    let mut trace = match args.get(4) {
        Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p)?)),
        None => None,
    };
    if let Some(t) = trace.as_mut() {
        writeln!(t, "frame,node,via,off,button,sx,sy,ctrl,modal,dialog,dstate,lastpage,choice,line,menu_mode,menu_timer,act_select,act_cursor,action,play_mode,level,area,transition,stars")?;
    }
    let game_gen = SM64GameGenerator::from_file(&args[1])?;
    let mut game = game_gen.create_game()?;
    let mut side = game_gen.create_game()?;
    for p in &recs[0].1 {
        game.step_game(*p)?;
    }
    let mut frame = recs[0].1.len() as u32;
    let mut before = read_state(&game, frame, true);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
    let (mut fast_total, mut old_total, mut tree_total) = (0i64, 0i64, 0i64);
    println!("node via start_frame option old fast saved tree_inputs");
    // Each route node's segment, then the node itself if it is a dialog or menu.
    let none = Vec::new();
    for k in 1..=ids.len() {
        let id = ids.get(k).copied().unwrap_or(target);
        let end = k == ids.len();
        let via = if end { "end".to_string() } else { rows[id as usize][12].clone() };
        let pads = if end { &none } else { &recs[id as usize + 1].1 };
        if via == "Mario::GoToCastle" {
            let save = game.save_state();
            for o in Registry.generate(&before, &mut rng) {
                if o.kind() != "Mario::GoToCastle" || !o.base_criteria(&before) {
                    continue;
                }
                obrrt_objectives::set_fast_modal(false);
                let old = play(&mut side, &save, &before, o.as_ref(), o.fixed())?;
                obrrt_objectives::set_fast_modal(true);
                let fast = play(&mut side, &save, &before, o.as_ref(), o.fixed())?;
                let show = |r: Option<usize>| r.map_or("fail".to_string(), |i| i.to_string());
                println!("{id} {via} {} {} {} {} - {}", before.frame, o.key().replace(' ', "_"), show(old), show(fast), pads.len());
                if let (Some(a), Some(b)) = (old, fast) {
                    old_total += a as i64;
                    fast_total += b as i64;
                    tree_total += pads.len() as i64;
                }
            }
        }
        if before.modal() && (end || via.starts_with("Dialog::") || via.starts_with("Menu::")) {
            let save = game.save_state();
            for o in Registry.generate(&before, &mut rng) {
                if !o.modal() || !o.base_criteria(&before) {
                    continue;
                }
                obrrt_objectives::set_fast_modal(false);
                let old = play(&mut side, &save, &before, o.as_ref(), o.fixed())?;
                obrrt_objectives::set_fast_modal(true);
                let fast_act = play_act(&mut side, &save, &before, o.as_ref(), o.fixed())?;
                let fast = fast_act.map(|f| f.0);
                obrrt_objectives::set_fast_modal(false);
                let old_act = play_act(&mut side, &save, &before, o.as_ref(), o.fixed())?.map_or(0, |f| f.1);
                obrrt_objectives::set_fast_modal(true);
                let taken = o.kind() == via || end;
                // Check: A and B alternated on every frame (for dialogs
                // without a choice, nothing can be sooner).
                let mash = if o.kind() == "Dialog::Read" {
                    let m = obrrt_core::PadPolicy::new(|_, played| {
                        let a = played.len() % 2 == 0;
                        GamePad::new(if a { obrrt_core::buttons::A_BUTTON } else { obrrt_core::buttons::B_BUTTON }, 0, 0)
                    });
                    play(&mut side, &save, &before, o.as_ref(), Some(Budget::Policy { limit: 900, policy: m }))?.map_or(-1, |i| i as i64)
                } else {
                    -1
                };
                // Check: the first frame whose A press makes Mario jump
                // (A/B mashed until the box closes, then one A at frame k):
                // where Mario can first act. -1 = not before gDialogID clears.
                let mut free = -1i64;
                if o.kind() == "Dialog::Read" {
                    let (a_b, b_b) = (obrrt_core::buttons::A_BUTTON, obrrt_core::buttons::B_BUTTON);
                    'k: for k in 0..400u32 {
                        side.load_state(&save)?;
                        let mut st = before.clone();
                        let mut closing = false;
                        for f in 0..k + 4 {
                            closing |= st.dialog_state == 3;
                            let b = if f == k { a_b } else if closing { 0 } else if f % 2 == 0 { a_b } else { b_b };
                            if f == k && !closing {
                                continue 'k;
                            }
                            side.step_game(GamePad::new(b, 0, 0))?;
                            st = read_state(&side, before.frame + f + 1, false);
                            if f >= k && st.action & 0x800 != 0 {
                                free = k as i64;
                                break 'k;
                            }
                            if f < k && st.dialog < 0 {
                                break 'k;
                            }
                        }
                    }
                }
                let show = |r: Option<usize>| r.map_or("fail".to_string(), |i| i.to_string());
                println!(
                    "{id} {via} {} {}{} {} {} {} {} mash={mash} mario_acts_from={free} action_at_close old={old_act:#x} fast={:#x}",
                    before.frame,
                    o.key().replace(' ', "_"),
                    if taken { "*" } else { "" },
                    show(old),
                    show(fast),
                    match (old, fast) {
                        (Some(a), Some(b)) => (a as i64 - b as i64).to_string(),
                        _ => "-".into(),
                    },
                    pads.len(),
                    fast_act.map_or(0, |f| f.1)
                );
                if taken && let (Some(a), Some(b)) = (old, fast) {
                    old_total += a as i64;
                    fast_total += b as i64;
                    tree_total += pads.len() as i64;
                }
            }
        }
        for (off, p) in pads.iter().enumerate() {
            game.step_game(*p)?;
            frame += 1;
            let s = read_state(&game, frame, false);
            if let Some(t) = trace.as_mut() {
                writeln!(
                    t,
                    "{},{id},{via},{off},{:#x},{},{},{},{},{},{},{},{},{},{},{},{},{},{:#x},{},{},{},{},{}",
                    frame, p.button, p.stick_x, p.stick_y, s.controllable() as u8, s.modal() as u8, s.dialog, s.dialog_state,
                    s.dialog_last_page as u8, s.dialog_choice as u8, s.menu_line, s.menu_mode, s.menu_timer, s.act_select as u8,
                    s.act_cursor, s.action, s.play_mode, s.level, s.area, s.transition, s.stars
                )?;
            }
            let _ = s;
        }
        before = read_state(&game, frame, true);
    }
    println!(
        "route to node {target} (frame {}): taken options close in {old_total} frames (old) vs {fast_total} (fast), saved {}; the tree's segments for them had {tree_total} inputs",
        before.frame,
        old_total - fast_total
    );
    Ok(())
}
