//! Commands that change a session: the queue, objectives, settings, rewinds.
//! Applied by the daemon from its inbox, or by the CLI directly when no
//! daemon runs (the same function, so the history is the same either way).

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};

use crate::session::{Commit, Session};

/// Where a rewind lands.
pub struct Landing {
    pub head: String,
    pub frame: u32,
}

/// Applies one command; returns a JSON result. `cur_end`: the frame the
/// session's path currently ends at (HEAD plus the working segment).
pub fn apply(s: &mut Session, cmd: &Value) -> Result<Value> {
    let c = cmd["cmd"].as_str().unwrap_or("");
    match c {
        "push" => {
            let sel = cmd["sel"].clone();
            crate::objective::validate(&sel)?;
            check_rh(s, &cmd["rh"])?;
            let id = s.next_obj_id()?;
            s.obj_event(json!({"ev": "add", "id": id, "sel": sel, "note": cmd["note"], "rh": cmd["rh"], "stall_frames": cmd["stall_frames"]}))?;
            let mut q = s.queue()?;
            let pos = cmd["pos"].as_u64().map_or(q.len(), |p| (p as usize).min(q.len()));
            q.insert(pos, id.clone());
            s.set_queue(&q)?;
            Ok(json!({"id": id, "pos": pos, "queue": q}))
        }
        "scrap" => {
            let id = obj_id(cmd)?;
            let mut q = s.queue()?;
            q.retain(|x| x != &id);
            s.set_queue(&q)?;
            s.obj_event(json!({"ev": "status", "id": id, "status": "scrapped", "note": cmd["note"]}))?;
            Ok(json!({"id": id, "queue": q}))
        }
        "move" => {
            let id = obj_id(cmd)?;
            let mut q = s.queue()?;
            if !q.contains(&id) {
                // A scrapped (or met elsewhere) objective back into the queue.
                s.obj_event(json!({"ev": "status", "id": id, "status": "queued", "note": "moved back into the queue"}))?;
            }
            q.retain(|x| x != &id);
            let pos = cmd["pos"].as_u64().map_or(q.len(), |p| (p as usize).min(q.len()));
            q.insert(pos, id.clone());
            s.set_queue(&q)?;
            Ok(json!({"id": id, "queue": q}))
        }
        "edit" => {
            let id = obj_id(cmd)?;
            if let Some(sel) = cmd.get("sel").filter(|v| !v.is_null()) {
                crate::objective::validate(sel)?;
            }
            check_rh(s, &cmd["rh"])?;
            s.obj_event(json!({"ev": "edit", "id": id, "sel": cmd["sel"], "rh": cmd["rh"], "note": cmd["note"], "stall_frames": cmd["stall_frames"]}))?;
            Ok(json!({"id": id}))
        }
        "set" => {
            let key = cmd["key"].as_str().ok_or_else(|| anyhow!("set: key"))?;
            let mut v = serde_json::to_value(&s.cfg)?;
            let ptr = format!("/{}", key.replace('.', "/"));
            let slot = v.pointer_mut(&ptr).ok_or_else(|| anyhow!("set: no setting {key} (see `ar status --config`)"))?;
            let old = slot.clone();
            *slot = cmd["value"].clone();
            let cfg: crate::session::Config = serde_json::from_value(v).map_err(|e| anyhow!("set {key}: {e}"))?;
            if let Some(why) = crate::session::below_floor(&cfg.rh) {
                bail!("set {key}: {why}");
            }
            s.cfg = cfg;
            s.save_cfg()?;
            s.event("set", json!({"key": key, "old": old, "value": cmd["value"]}))?;
            Ok(json!({"key": key, "old": old, "value": cmd["value"]}))
        }
        _ => bail!("unknown command {c}"),
    }
}

/// Per-objective overrides may raise the compute, never go below the floor.
fn check_rh(s: &Session, over: &Value) -> Result<()> {
    let Some(o) = over.as_object() else { return Ok(()) };
    let mut v = serde_json::to_value(&s.cfg.rh)?;
    for (k, x) in o {
        v[k] = x.clone();
    }
    let c: obrrt_search::ga::rhstep::RhConfig = serde_json::from_value(v).map_err(|e| anyhow!("--rh: {e}"))?;
    if let Some(why) = crate::session::below_floor(&c) {
        bail!("--rh: {why}");
    }
    Ok(())
}

fn obj_id(cmd: &Value) -> Result<String> {
    cmd["id"].as_str().map(Into::into).ok_or_else(|| anyhow!("{}: an objective id (oNNNN)", cmd["cmd"]))
}

/// Moves HEAD for a rewind (or a branch switch): `target` is `cNNNN` (that
/// commit's end), `cNNNN:F` (frame F inside it), `fF` (frame F on the
/// current path) or `-N` (N frames back). Unfinished work becomes an
/// "abandoned" commit first; landing inside a segment makes a "split"
/// commit. Objectives are requeued (or scrapped with `drop`) to match the
/// new path. Files only; the daemon then reloads its controller.
pub fn rewind(s: &Session, target: &str, drop: bool, note: &str) -> Result<Landing> {
    let head = s.head()?;
    let (w, wpads) = s.wip_pads()?;
    let end = w.from + wpads.len() as u32;
    let commits = s.commit_map()?;
    // Resolve the target to (commit, frame) before touching anything.
    let chain = s.chain(&head)?;
    enum T {
        End(String),
        In(String, u32),
        Wip(u32),
    }
    let in_path = |f: u32| -> Result<T> {
        if f > end {
            bail!("frame {f} is past the path's end ({end})");
        }
        if f >= w.from && w.parent == head {
            return Ok(if f == w.from { T::End(head.clone()) } else { T::Wip(f) });
        }
        let c = chain.iter().find(|c| c.from < f && f <= c.to).ok_or_else(|| anyhow!("frame {f} is before the session's start"))?;
        Ok(if f == c.to { T::End(c.id.clone()) } else { T::In(c.id.clone(), f) })
    };
    let t = if let Some(n) = target.strip_prefix('-') {
        in_path(end.saturating_sub(n.parse::<u32>()?))?
    } else if let Some(f) = target.strip_prefix('f') {
        in_path(f.parse()?)?
    } else if let Some((id, f)) = target.split_once(':') {
        let c = commits.get(id).ok_or_else(|| anyhow!("no commit {id}"))?;
        let f: u32 = f.parse()?;
        if f < c.from || f > c.to {
            bail!("{id} covers frames {}..{}", c.from, c.to);
        }
        if f == c.to { T::End(id.into()) } else if f == c.from { T::End(c.parent.clone().ok_or_else(|| anyhow!("{id} is the root"))?) } else { T::In(id.into(), f) }
    } else {
        let c = commits.get(target).ok_or_else(|| anyhow!("no commit {target} (targets: cNNNN, cNNNN:F, fF, -N)"))?;
        T::End(c.id.clone())
    };
    if matches!(&t, T::Wip(f) if *f == end) {
        return Ok(Landing { head, frame: end });
    }
    // Keep unfinished work as history.
    let abandoned: Option<Commit> = if !wpads.is_empty() { Some(s.cut("abandoned", None, None, note)?) } else { None };
    let (new_head, frame) = match t {
        T::End(id) => {
            let f = commits.get(&id).map_or(0, |c| c.to);
            (id, f)
        }
        T::In(id, f) => {
            let c = s.split(&commits[&id], f, None, note)?;
            (c.id, f)
        }
        T::Wip(f) => {
            let a = abandoned.as_ref().unwrap();
            let c = s.split(a, f, None, note)?;
            (c.id, f)
        }
    };
    s.set_head(&new_head)?;
    s.new_wip(&new_head, frame)?;
    // Objectives to match the new path.
    let path = s.on_path(&new_head)?;
    let mut q = s.queue()?;
    let mut back = Vec::new();
    for o in s.objectives()? {
        let met_here = o.met_commits.iter().any(|c| path.contains(c));
        if o.status == "met" && !met_here {
            if drop {
                s.obj_event(json!({"ev": "status", "id": o.id, "status": "scrapped", "note": format!("rewind to {target}: dropped")}))?;
            } else {
                s.obj_event(json!({"ev": "status", "id": o.id, "status": "queued", "note": format!("rewind to {target}")}))?;
                back.push(o.id.clone());
            }
        } else if o.status == "queued" && met_here {
            let c = o.met_commits.iter().find(|c| path.contains(*c)).cloned();
            s.obj_event(json!({"ev": "status", "id": o.id, "status": "met", "commit": c, "note": format!("met on the branch of {target}")}))?;
            q.retain(|x| x != &o.id);
        }
    }
    // Requeued ones go first, in the order they were met.
    back.sort();
    q.retain(|x| !back.contains(x));
    let q: Vec<String> = back.iter().cloned().chain(q).collect();
    s.set_queue(&q)?;
    s.event("rewind", json!({"target": target, "head": new_head, "frame": frame, "abandoned": abandoned.map(|a| a.id), "requeued": back, "dropped": drop, "note": note}))?;
    Ok(Landing { head: new_head, frame })
}
