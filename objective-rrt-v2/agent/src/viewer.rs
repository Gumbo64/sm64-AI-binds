//! Viewer data for sm64-viewer's agent-route page (served by oss2's
//! tools/serve.py under /ar/<session>/):
//!
//! - `viewer.json`: the session at a glance: commit tree (every branch, with
//!   whether it is on HEAD's path), the tentative route, the queue and
//!   objectives, status, recent events.
//! - `paths/<commit>.json`: Mario's path over that commit's inputs,
//!   `{"fields": [frame, level, area, x, y, z, action], "every": 3, "points": [...]}`;
//!   `paths/tentative.json` the same for the daemon's tentative route.
//!
//! Commits never change, so their paths are made once (by replay from the
//! nearest checkpoint); the daemon refreshes viewer.json and the tentative
//! path every couple of seconds, `ar viewer` does it once.

use anyhow::Result;
use obrrt_core::{GamePad, St, read_state};
use serde_json::{Value, json};
use sm64_binds::SM64Game;

use crate::ckpt;
use crate::session::Session;
use crate::util::*;

pub const EVERY: u32 = 3;

fn point(st: &St) -> Value {
    json!([st.frame, st.level, st.area, st.pos[0].round(), st.pos[1].round(), st.pos[2].round(), format!("{:#x}", st.action)])
}

/// Steps `pads` on `game` (at `frame0`), sampling every EVERY frames and the last.
fn sample(game: &mut SM64Game, frame0: u32, pads: &[GamePad]) -> Result<Vec<Value>> {
    let mut pts = vec![point(&read_state(game, frame0, false))];
    for (i, p) in pads.iter().enumerate() {
        game.step_game(*p)?;
        let f = frame0 + i as u32 + 1;
        if (i as u32 + 1) % EVERY == 0 || i + 1 == pads.len() {
            pts.push(point(&read_state(game, f, false)));
        }
    }
    Ok(pts)
}

fn path_doc(points: Vec<Value>, extra: Value) -> Value {
    let mut v = json!({"fields": ["frame", "level", "area", "x", "y", "z", "action"], "every": EVERY, "points": points});
    if let (Some(m), Some(e)) = (v.as_object_mut(), extra.as_object()) {
        for (k, x) in e {
            m.insert(k.clone(), x.clone());
        }
    }
    v
}

/// Paths for commits that have none yet (at most `budget` frames replayed per call).
pub fn sync_paths(s: &Session, gg: &sm64_binds::SM64GameGenerator, ck: &mut ckpt::Store, budget: u64) -> Result<usize> {
    let dir = s.dir.join("paths");
    std::fs::create_dir_all(&dir)?;
    let mut done = 0;
    let mut spent = 0u64;
    let mut game: Option<SM64Game> = None;
    for c in s.commits()? {
        let f = dir.join(format!("{}.json", c.id));
        if f.exists() || spent > budget {
            continue;
        }
        let pads = s.pads_of(&c.id)?;
        // The game at the commit's start: its parent's end.
        let g = match &mut game {
            Some(g) => g,
            None => game.insert(gg.create_game()?),
        };
        let start = match &c.parent {
            Some(p) => {
                s.load_at(ck, g, p, Some(c.from)).ok();
                c.from
            }
            None => {
                *g = gg.create_game()?;
                0
            }
        };
        // The root's boot is long and not interesting: sample its last 300 frames only.
        let (skip, pads) = if c.parent.is_none() { (pads.len().saturating_sub(300), &pads[..]) } else { (0, &pads[..]) };
        for p in &pads[..skip] {
            g.step_game(*p)?;
        }
        let pts = sample(g, start + skip as u32, &pads[skip..])?;
        spent += (pads.len() + (c.from - start) as usize) as u64;
        write_json(&f, &path_doc(pts, json!({"commit": c.id})))?;
        done += 1;
    }
    Ok(done)
}

/// `paths/live.json`: from HEAD's end through the frozen work in progress
/// and the daemon's tentative route; `base` is where the tentative part
/// (still open to change) begins.
pub fn write_live(s: &Session, gg: &sm64_binds::SM64GameGenerator, ck: &mut ckpt::Store) -> Result<()> {
    std::fs::create_dir_all(s.dir.join("paths"))?;
    let head = s.head()?;
    let (w, wp) = s.wip_pads()?;
    let base = w.from + wp.len() as u32;
    let mut pads = wp;
    if let Ok(t) = read_json(s.dir.join("tentative.json")) {
        if t["wip"] == w.id.as_str() && t["from"].as_u64() == Some(base as u64) {
            pads.extend(pads_from_bytes(&std::fs::read(s.dir.join("inputs/tentative.pads")).unwrap_or_default()));
        }
    }
    let mut g = gg.create_game()?;
    s.load_at(ck, &mut g, &head, Some(w.from))?;
    let pts = sample(&mut g, w.from, &pads)?;
    write_json(s.dir.join("paths/live.json"), &path_doc(pts, json!({"from": w.from, "base": base, "frontier": w.from + pads.len() as u32, "head": head})))
}

/// Everything at once (`ar viewer`, and the daemon every few seconds).
pub fn sync(s: &Session, gg: &sm64_binds::SM64GameGenerator, ck: &mut ckpt::Store, budget: u64) -> Result<()> {
    sync_paths(s, gg, ck, budget)?;
    write_live(s, gg, ck)?;
    write_viewer(s)
}

fn label(sel: &Value) -> String {
    if let Some(k) = sel["native"].as_str() {
        return format!("{k} {}", sel["key"].as_str().unwrap_or("")).trim().to_string();
    }
    if let Some(g) = sel.get("goto") {
        return format!("goto {g}");
    }
    if let Some(l) = sel.get("lambda") {
        return format!("lambda {}", l.as_str().unwrap_or("(inline)"));
    }
    if let Some(sp) = sel.get("spec") {
        return format!("spec {}", sp["name"].as_str().unwrap_or(""));
    }
    sel.to_string()
}

/// viewer.json.
pub fn write_viewer(s: &Session) -> Result<()> {
    let head = s.head()?;
    let path = s.on_path(&head)?;
    let objs = s.objectives()?;
    let olabel = |id: &str| objs.iter().find(|o| o.id == id).map(|o| label(&o.sel)).unwrap_or_default();
    let commits: Vec<Value> = s
        .commits()?
        .into_iter()
        .map(|c| {
            json!({
                "id": c.id, "parent": c.parent, "kind": c.kind, "objective": c.objective,
                "label": c.objective.as_deref().map(olabel).unwrap_or_default(),
                "from": c.from, "to": c.to, "frames": c.to - c.from, "note": c.note, "summary": c.summary,
                "on_path": path.contains(&c.id), "time": c.time,
                "has_path": s.dir.join(format!("paths/{}.json", c.id)).exists(),
            })
        })
        .collect();
    let (w, wp) = s.wip_pads()?;
    let tentative = read_json(s.dir.join("tentative.json")).ok();
    let status = read_json(s.dir.join("status.json")).ok();
    let live = s.serve_pid().is_some();
    let ev = s.events()?;
    let events: Vec<Value> = ev.iter().filter(|e| e["ev"] != "ack" && e["ev"] != "resolved").rev().take(80).cloned().collect::<Vec<_>>().into_iter().rev().collect();
    let q = s.queue()?;
    let objectives: Vec<Value> = objs
        .iter()
        .map(|o| json!({"id": o.id, "label": label(&o.sel), "sel": o.sel, "status": o.status, "note": o.note, "met_commits": o.met_commits, "queue_pos": q.iter().position(|x| x == &o.id)}))
        .collect();
    write_json(
        s.dir.join("viewer.json"),
        &json!({
            "session": s.cfg.name, "updated": now(), "live": live, "head": head,
            "start": s.cfg.start_desc,
            "wip": {"id": w.id, "parent": w.parent, "from": w.from, "frames": wp.len()},
            "tentative": tentative, "status": status, "queue": q, "objectives": objectives,
            "commits": commits, "events": events,
            "settings": {"window": s.cfg.window, "back_frames": s.cfg.back_frames, "anchor": s.cfg.back_anchor_share, "rh": s.cfg.rh},
        }),
    )
}
