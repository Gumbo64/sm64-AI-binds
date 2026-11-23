//! The solver daemon (`ar serve`): works through the objective window with
//! the rolling-horizon controller, one committed frame per step, recording
//! inputs, checkpoints, every generation's population and events; applies
//! the agent's commands from the inbox between steps.

use anyhow::{Context, Result, anyhow};
use obrrt_core::{Budget, GamePad, Mem, Objective};
use obrrt_search::ga::rhstep::{RhConfig, RhController};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::ckpt;
use crate::commands;
use crate::objective;
use crate::popjournal;
use crate::session::{Session, Wip, summary};
use crate::util::*;

/// Frames a press-through may take before the daemon pauses.
const PRESS_LIMIT: u32 = 1500;

struct Front {
    id: String,
    key: String,
    /// Lowest ETA so far and the frame it was reached.
    best: f64,
    best_frame: u32,
    stall_next: u32,
}

pub struct Daemon {
    s: Session,
    gg: sm64_binds::SM64GameGenerator,
    ck: ckpt::Store,
    ctl: RhController,
    head: String,
    wip: Wip,
    journal: Option<popjournal::Writer>,
    /// Objectives resolved, by id; and every one seen per id (fallbacks).
    resolved: HashMap<String, Arc<dyn Objective>>,
    seen: HashMap<String, Vec<Arc<dyn Objective>>>,
    paused: Option<String>,
    stop: bool,
    last_ckpt: u32,
    last_zone: (u16, u8),
    front: Option<Front>,
    reported: std::collections::HashSet<String>,
    last_status: Instant,
    t0: Instant,
    frames0: u32,
    emu0: u64,
    base_rh: RhConfig,
    /// Objectives met on this daemon's watch (re-solved by the backward window).
    done_objs: HashMap<String, Arc<dyn Objective>>,
    /// Objectives the best plan meets behind the frontier (tentative).
    tent: Vec<String>,
    /// Consecutive press-through frames at the frontier.
    pressing: u32,
    last_viewer: Instant,
}

impl Daemon {
    pub fn start(name: &str) -> Result<Daemon> {
        let s = Session::open(name)?;
        if let Some(pid) = s.serve_pid().filter(|&p| p != std::process::id()) {
            anyhow::bail!("a daemon already serves {} (pid {pid})", s.dir.display());
        }
        std::fs::write(s.dir.join("serve.pid"), format!("{}\n", std::process::id()))?;
        let gg = sm64_binds::SM64GameGenerator::from_file(&s.cfg.rom)?;
        let mut ck = ckpt::Store::open(&s.dir)?;
        let head = s.head()?;
        let mut game = gg.create_game()?;
        // The base: the end of the frozen work in progress (the tentative
        // route after it is restored as the best plan below).
        let (w0, wp0) = s.wip_pads()?;
        let st = s.load_at(&mut ck, &mut game, &head, Some(w0.from + wp0.len() as u32))?;
        if st.controllable() {
            obrrt_core::check_layout(&game).map_err(|e| anyhow!("layout check: {e}"))?;
        }
        let save = game.save_state();
        let mut ctl = RhController::new(&gg, s.cfg.threads, &save, st.clone(), with_session(&s, s.cfg.rh.clone()))?;
        ctl.journal = s.cfg.journal;
        let wip = s.wip()?;
        let base_rh = s.cfg.rh.clone();
        let mut d = Daemon {
            gg,
            ck,
            ctl,
            head,
            wip,
            journal: None,
            resolved: HashMap::new(),
            seen: HashMap::new(),
            paused: None,
            stop: false,
            last_ckpt: st.frame,
            last_zone: st.zone(),
            front: None,
            reported: Default::default(),
            last_status: Instant::now() - Duration::from_secs(10),
            t0: Instant::now(),
            frames0: st.frame,
            emu0: 0,
            base_rh,
            done_objs: HashMap::new(),
            tent: Vec::new(),
            pressing: 0,
            last_viewer: Instant::now() - Duration::from_secs(60),
            s,
        };
        d.restore_population(st.frame)?;
        // A tentative route left by the last run (same base): resume it.
        if let Ok(t) = read_json(d.s.dir.join("tentative.json")) {
            let (w, wp) = d.s.wip_pads()?;
            if t["wip"] == w.id.as_str() && t["from"].as_u64() == Some((w.from + wp.len() as u32) as u64) {
                let pads = pads_from_bytes(&std::fs::read(d.s.dir.join("inputs/tentative.pads")).unwrap_or_default());
                if !pads.is_empty() {
                    d.s.event("tentative", json!({"restored_frames": pads.len()}))?;
                    d.ctl.set_tentative(pads);
                }
            }
        }
        d.open_journal()?;
        d.s.event("serve", json!({"pid": std::process::id(), "head": d.head, "frame": st.frame, "threads": d.s.cfg.threads, "rh": d.s.cfg.rh}))?;
        Ok(d)
    }

    fn open_journal(&mut self) -> Result<()> {
        self.journal = if self.s.cfg.journal { Some(popjournal::Writer::open(&self.s.journal_path(&self.wip.id), self.s.cfg.keyframe_every)?) } else { None };
        Ok(())
    }

    /// The population recorded at `frame` on this path, if any: a segment
    /// HEAD is a prefix of, or a child of HEAD (its first records).
    fn restore_population(&mut self, frame: u32) -> Result<bool> {
        let commits = self.s.commits()?;
        let head = commits.iter().find(|c| c.id == self.head).cloned();
        let mut cands: Vec<String> = Vec::new();
        cands.push(self.wip.id.clone());
        if let Some(p) = head.as_ref().and_then(|h| h.prefix_of.clone()) {
            cands.push(p);
        }
        for c in commits.iter().rev() {
            if c.parent.as_deref() == Some(&self.head) {
                cands.push(c.id.clone());
            }
        }
        for seg in cands {
            let p = self.s.journal_path(&seg);
            if !p.exists() {
                continue;
            }
            let heads = popjournal::index(&p)?;
            if let Some(i) = popjournal::find_frame(&heads, frame) {
                let pop = popjournal::read_at(&p, &heads, i)?;
                self.ctl.set_population(&pop);
                self.s.event("population", json!({"restored_from": seg, "record": i, "generation": heads[i].generation, "frame": frame, "plans": pop.len()}))?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Reloads the controller at HEAD's end (after a rewind).
    fn reposition(&mut self, fresh_pop: bool) -> Result<()> {
        if let Some(j) = &mut self.journal {
            j.flush()?;
        }
        self.head = self.s.head()?;
        self.wip = self.s.wip()?;
        let mut game = self.gg.create_game()?;
        let (w0, wp0) = self.s.wip_pads()?;
        let st = self.s.load_at(&mut self.ck, &mut game, &self.head, Some(w0.from + wp0.len() as u32))?;
        self.ctl.load(&game.save_state(), st.clone(), true)?;
        if !fresh_pop {
            self.restore_population(st.frame)?;
        }
        self.open_journal()?;
        self.resolved.clear();
        self.front = None;
        self.last_ckpt = st.frame;
        self.last_zone = st.zone();
        self.reported.clear();
        Ok(())
    }

    fn frame(&self) -> u32 {
        self.ctl.state().frame
    }

    /// The window: the queue's first objectives that resolve at `st` (the
    /// frontier): those the best plan met behind the frontier (`behind`),
    /// then `window` more.
    fn window(&mut self, st: &obrrt_core::St, behind: usize) -> Result<Vec<(String, Arc<dyn Objective>)>> {
        let q = self.s.queue()?;
        let objs: HashMap<String, crate::session::Objective> = self.s.objectives()?.into_iter().map(|o| (o.id.clone(), o)).collect();
        let mut out = Vec::new();
        for id in q.iter().take(behind + self.s.cfg.window.max(1)) {
            let Some(o) = objs.get(id) else { continue };
            if let Some(r) = self.resolved.get(id) {
                out.push((id.clone(), r.clone()));
                continue;
            }
            let seen = self.seen.get(id).cloned().unwrap_or_default();
            match objective::resolve(&o.sel, id, st, &seen) {
                Ok(Some(r)) => {
                    let e = self.seen.entry(id.clone()).or_default();
                    if !e.iter().any(|x| x.key() == r.key()) {
                        e.push(r.clone());
                    }
                    self.resolved.insert(id.clone(), r.clone());
                    self.s.event("resolved", json!({"objective": id, "kind": r.kind(), "key": r.key(), "eta": obrrt_core::start_eta(r.as_ref(), st), "frame": st.frame}))?;
                    out.push((id.clone(), r));
                }
                Ok(None) => {
                    if out.len() <= behind && self.reported.insert(format!("unresolved:{id}:{}", st.frame / 300)) {
                        self.s.event("unresolved", json!({"objective": id, "sel": o.sel, "frame": st.frame, "state": summary(st), "hint": "not offered at the frontier: check `ar objectives`, edit or scrap it, or push a nearer objective before it"}))?;
                    }
                    break;
                }
                Err(e) => {
                    if self.reported.insert(format!("bad:{id}")) {
                        self.s.event("error", json!({"objective": id, "error": format!("{e:#}")}))?;
                    }
                    break;
                }
            }
        }
        Ok(out)
    }

    fn append(&mut self, pads: &[GamePad]) -> Result<()> {
        self.s.wip_append(&self.wip, pads)
    }

    /// An objective's success became permanent (the window's base moved
    /// past it): its commit. Only the last of a step's has the base state
    /// (and a checkpoint); earlier ones are rebuilt by replay when needed.
    fn commit_met(&mut self, id: &str, o: &dyn Objective, last: bool) -> Result<()> {
        if let Some(j) = &mut self.journal {
            j.flush()?;
        }
        if let Some(r) = self.resolved.get(id) {
            self.done_objs.insert(id.to_string(), r.clone());
        }
        let st = self.ctl.state().clone();
        let c = self.s.cut("met", Some(id), last.then_some(&st), "")?;
        if last {
            self.ck.put(st.frame, &c.id, st.zone(), &self.ctl.save_state())?;
            self.last_ckpt = st.frame;
        }
        self.s.obj_event(json!({"ev": "status", "id": id, "status": "met", "commit": c.id}))?;
        let mut q = self.s.queue()?;
        q.retain(|x| x != id);
        self.s.set_queue(&q)?;
        self.head = c.id.clone();
        self.wip = self.s.wip()?;
        self.open_journal()?;
        self.resolved.remove(id);
        self.tent.retain(|x| x != id);
        self.s.event("committed", json!({"objective": id, "kind": o.kind(), "key": o.key(), "commit": c.id, "segment_frames": c.to - c.from, "to": c.to}))?;
        Ok(())
    }

    /// Settings while `id` is current: the session's, its overrides, the
    /// window's backward part; never below the floor.
    fn rh_for(&self, id: &str) -> Result<RhConfig> {
        let mut v = serde_json::to_value(&self.s.cfg.rh)?;
        if let Some(o) = self.s.objectives()?.into_iter().find(|o| o.id == id) {
            if let (Some(m), Some(over)) = (v.as_object_mut(), o.rh.as_object()) {
                for (k, x) in over {
                    m.insert(k.clone(), x.clone());
                }
            }
        }
        Ok(with_session(&self.s, serde_json::from_value(v)?))
    }

    fn stall_for(&self, id: &str) -> u32 {
        self.s.objectives().ok().and_then(|v| v.into_iter().find(|o| o.id == id)).and_then(|o| o.stall_frames).unwrap_or(self.s.cfg.stall_frames)
    }

    fn checkpoint_maybe(&mut self) -> Result<()> {
        let st = self.ctl.state();
        let (f, z) = (st.frame, st.zone());
        if f >= self.last_ckpt + self.s.cfg.ckpt_every || z != self.last_zone {
            let save = self.ctl.save_state();
            self.ck.put(f, &self.wip.id, z, &save)?;
            self.last_ckpt = f;
            if z != self.last_zone {
                self.s.event("zone", json!({"from": [self.last_zone.0, self.last_zone.1], "to": [z.0, z.1], "frame": f}))?;
            }
            self.last_zone = z;
        }
        Ok(())
    }

    /// The tentative route (base to frontier) for the CLI's inspection.
    fn write_tentative(&self) -> Result<()> {
        let (w, wp) = self.s.wip_pads()?;
        write_atomic(self.s.dir.join("inputs/tentative.pads"), &pads_to_bytes(self.ctl.tentative()))?;
        write_json(self.s.dir.join("tentative.json"), &json!({"wip": w.id, "from": w.from + wp.len() as u32, "frames": self.ctl.frontier(), "dones": self.ctl.tentative_dones(), "met": self.tent}))
    }

    fn status(&mut self, window: &[(String, Arc<dyn Objective>)]) -> Result<()> {
        if self.last_status.elapsed() < Duration::from_secs(2) {
            return Ok(());
        }
        self.last_status = Instant::now();
        self.write_tentative()?;
        let st = self.ctl.state();
        let n = self.ctl.frontier() as u32;
        let secs = self.t0.elapsed().as_secs_f64().max(1e-3);
        let w: Vec<Value> = window.iter().map(|(id, o)| json!({"id": id, "kind": o.kind(), "key": o.key()})).collect();
        let v = json!({
            "pid": std::process::id(),
            "head": self.head,
            "wip": {"id": self.wip.id, "from": self.wip.from, "frames": st.frame - self.wip.from},
            "base_frame": st.frame,
            "frontier_frame": st.frame + n,
            "frame": st.frame + n,
            "behind": n,
            "tentatively_met": self.tent,
            "state": summary(st),
            "front": self.front.as_ref().map(|f| json!({"id": f.id, "key": f.key, "eta": self.ctl.frontier_eta(), "best": f.best, "frames_on_it": (st.frame + n).saturating_sub(self.wip.objective_from), "since_best": (st.frame + n).saturating_sub(f.best_frame)})),
            "window": w,
            "paused": self.paused,
            "generation": self.ctl.generation,
            "anchored_children": self.ctl.anchored,
            "rh": self.ctl.cfg,
            "speed": {"frontier_fps": (st.frame + n).saturating_sub(self.frames0) as f64 / secs, "emulated_fps": (self.ctl.frames - self.emu0) as f64 / secs},
            "journal_bytes": self.journal.as_ref().map(|j| j.bytes),
            "checkpoints": self.ck.summary(),
            "updated": now(),
        });
        write_json(self.s.dir.join("status.json"), &v)?;
        // Viewer data (sm64-viewer's agent-route page), every 5 s.
        if self.last_viewer.elapsed() >= Duration::from_secs(5) {
            self.last_viewer = Instant::now();
            if let Err(e) = crate::viewer::sync(&self.s, &self.gg, &mut self.ck, 20_000) {
                eprintln!("viewer: {e:#}");
            }
        }
        Ok(())
    }

    fn inbox(&mut self) -> Result<()> {
        for (id, cmd) in self.s.take_inbox()? {
            let c = cmd["cmd"].as_str().unwrap_or("").to_string();
            let r: Result<Value> = (|| {
                Ok(match c.as_str() {
                    "pause" => {
                        self.paused = Some(cmd["reason"].as_str().unwrap_or("by the agent").into());
                        json!({"paused": true})
                    }
                    "resume" => {
                        self.paused = None;
                        self.reported.clear();
                        json!({"paused": false})
                    }
                    "stop" => {
                        self.stop = true;
                        json!({"stopping": true})
                    }
                    "rewind" => {
                        // The tentative route is part of the path a rewind sees: freeze it first.
                        self.freeze_tentative()?;
                        let l = commands::rewind(&self.s, cmd["target"].as_str().unwrap_or(""), cmd["drop"].as_bool().unwrap_or(false), cmd["note"].as_str().unwrap_or(""))?;
                        self.reposition(cmd["fresh_pop"].as_bool().unwrap_or(false))?;
                        json!({"head": l.head, "frame": l.frame})
                    }
                    "cut" => {
                        self.freeze_tentative()?;
                        let st = self.ctl.state().clone();
                        let c = self.s.cut("manual", None, Some(&st), cmd["note"].as_str().unwrap_or(""))?;
                        self.ck.put(st.frame, &c.id, st.zone(), &self.ctl.save_state())?;
                        self.head = c.id.clone();
                        self.wip = self.s.wip()?;
                        self.open_journal()?;
                        json!({"commit": c.id})
                    }
                    "inject" => {
                        let plans: Vec<Vec<GamePad>> = cmd["files"].as_array().into_iter().flatten().filter_map(|f| f.as_str()).map(read_m64).collect::<Result<_>>()?;
                        let n = plans.len();
                        self.ctl.inject(plans);
                        json!({"injected": n})
                    }
                    "pop_seed" => {
                        let seg = cmd["segment"].as_str().ok_or_else(|| anyhow!("pop_seed: segment"))?;
                        let p = self.s.journal_path(seg);
                        let heads = popjournal::index(&p)?;
                        let i = cmd["record"].as_u64().map_or(heads.len().saturating_sub(1), |x| x as usize);
                        let pop = popjournal::read_at(&p, &heads, i.min(heads.len().saturating_sub(1)))?;
                        let n = pop.len();
                        self.ctl.set_population(&pop);
                        json!({"plans": n, "segment": seg, "record": i})
                    }
                    "fresh_pop" => {
                        self.ctl.set_population(&[]);
                        json!({"population": "fresh at the next step"})
                    }
                    _ => {
                        let r = commands::apply(&mut self.s, &cmd)?;
                        if c == "set" {
                            self.base_rh = self.s.cfg.rh.clone();
                            let cfg = with_session(&self.s, self.s.cfg.rh.clone());
                            self.ctl.set_config(cfg);
                            self.ctl.journal = self.s.cfg.journal;
                            self.front = None;
                        }
                        if c == "edit" || c == "scrap" || c == "move" {
                            if let Some(i) = cmd["id"].as_str() {
                                self.resolved.remove(i);
                            }
                            self.front = None;
                        }
                        self.reported.retain(|k| !k.starts_with("window"));
                        r
                    }
                })
            })();
            match r {
                Ok(v) => self.s.event("ack", json!({"inbox": id, "cmd": c, "result": v}))?,
                Err(e) => self.s.event("ack", json!({"inbox": id, "cmd": c, "error": format!("{e:#}")}))?,
            }
        }
        Ok(())
    }

    /// Makes the tentative route permanent up to the frontier (before a
    /// rewind, a manual cut or a stop), committing the objectives it met.
    fn freeze_tentative(&mut self) -> Result<()> {
        let n = self.ctl.frontier();
        if n == 0 {
            return Ok(());
        }
        let pads = self.ctl.tentative().to_vec();
        let dones = self.ctl.tentative_dones().to_vec();
        let ids = self.tent.clone();
        let objs: Vec<Option<Arc<dyn Objective>>> = ids.iter().map(|i| self.resolved.get(i).cloned()).collect();
        let (save, st) = {
            let (st, save) = self.ctl.frontier_state()?;
            (save, st)
        };
        // Split at the successes; the base moves to the frontier.
        let mut prev = 0;
        let mut game = self.gg.create_game()?;
        game.load_state(&self.ctl.save_state())?;
        let base = self.ctl.state().frame;
        for (j, &d) in dones.iter().enumerate() {
            let (Some(id), Some(Some(o))) = (ids.get(j), objs.get(j)) else { break };
            for p in &pads[prev..d] {
                game.step_game(*p)?;
            }
            self.append(&pads[prev..d])?;
            prev = d;
            let stj = obrrt_core::read_state(&game, base + d as u32, true);
            let c = self.s.cut("met", Some(id), Some(&stj), "frozen at a rewind/cut/stop")?;
            self.ck.put(stj.frame, &c.id, stj.zone(), &game.save_state())?;
            self.s.obj_event(json!({"ev": "status", "id": id, "status": "met", "commit": c.id}))?;
            let mut q = self.s.queue()?;
            q.retain(|x| x != id);
            self.s.set_queue(&q)?;
            self.head = c.id.clone();
            self.wip = self.s.wip()?;
            self.resolved.remove(id);
            self.s.event("committed", json!({"objective": id, "key": o.key(), "commit": c.id, "segment_frames": c.to - c.from, "to": c.to}))?;
        }
        self.append(&pads[prev..])?;
        self.tent.clear();
        self.ctl.load(&save, st, false)?;
        self.open_journal()?;
        Ok(())
    }

    /// One round: commands, then one controller step.
    pub fn tick(&mut self) -> Result<bool> {
        self.inbox()?;
        if self.stop {
            return Ok(false);
        }
        if self.paused.is_some() {
            self.status(&[])?;
            std::thread::sleep(Duration::from_millis(200));
            return Ok(true);
        }
        let n = self.ctl.frontier();
        let (fst, fsave) = if n > 0 { self.ctl.frontier_state()? } else { (self.ctl.state().clone(), self.ctl.save_state()) };
        let behind = self.ctl.tentative_dones().len();
        let win = self.window(&fst, behind)?;
        self.status(&win)?;
        if win.len() <= behind {
            if self.reported.insert("window:empty".into()) {
                let q = self.s.queue()?;
                self.s.event(if q.len() <= behind { "window_empty" } else { "waiting" }, json!({"queued": q.len(), "tentatively_met": behind, "frame": fst.frame, "state": summary(&fst)}))?;
            }
            std::thread::sleep(Duration::from_millis(250));
            return Ok(true);
        }
        let mut win = win;
        let (mut fid, mut fo) = (win[behind].0.clone(), win[behind].1.clone());
        if self.front.as_ref().is_none_or(|f| f.id != fid) {
            // A lambda objective binds at the state where it becomes current.
            if let Some(o) = self.s.objectives()?.into_iter().find(|o| o.id == fid).filter(|o| o.sel.get("lambda").is_some()) {
                if let Some(r) = objective::resolve(&o.sel, &fid, &fst, &[])? {
                    self.resolved.insert(fid.clone(), r.clone());
                    win[behind].1 = r.clone();
                    fo = r;
                    fid = o.id.clone();
                }
            }
            let cfg = self.rh_for(&fid)?;
            self.ctl.set_config(cfg);
            self.wip.objective_from = fst.frame;
            self.s.set_wip(&self.wip)?;
            self.front = Some(Front { id: fid.clone(), key: fo.key(), best: f64::INFINITY, best_frame: fst.frame, stall_next: self.stall_for(&fid) });
            self.reported.retain(|k| !k.starts_with("window"));
            self.s.event("front", json!({"objective": fid, "kind": fo.kind(), "key": fo.key(), "eta": obrrt_core::start_eta(fo.as_ref(), &fst), "window": win.iter().map(|(i, o)| json!([i, o.key()])).collect::<Vec<_>>(), "frame": fst.frame, "behind": n}))?;
        }
        // Inputs every plan takes at the frontier: a dialog/menu
        // objective's program, or the press-through.
        let forced: Option<Vec<GamePad>> = if let Some(prog) = fo.fixed() {
            let mut g = self.gg.create_game()?;
            g.load_state(&fsave)?;
            let (p, done) = play_program_on(&mut g, &fst, fo.as_ref(), &prog)?;
            if !done {
                self.paused = Some(format!("fixed objective {fid} ({}) not met after its {} frames", fo.key(), p.len()));
                self.s.event("failed", json!({"objective": fid, "key": fo.key(), "frames": p.len(), "state": summary(&fst)}))?;
                return Ok(true);
            }
            Some(p)
        } else {
            // Dialogs and cutscenes are pressed through inside every plan (`press`).
            let stuck = !fst.controllable() || fst.modal();
            self.pressing = if stuck { self.pressing + 1 } else { 0 };
            if self.pressing > PRESS_LIMIT {
                self.paused = Some(format!("not controllable at the frontier for {PRESS_LIMIT} steps"));
                self.s.event("stuck", json!({"reason": "press-through", "state": summary(&fst)}))?;
                return Ok(true);
            }
            None
        };
        let stages: Vec<&dyn Objective> = win.iter().map(|(_, o)| o.as_ref()).collect();
        let f0 = self.frame();
        let out = self.ctl.step(&stages, forced.as_deref())?;
        if let Some(j) = &mut self.journal {
            for (gi, g) in out.gens.iter().enumerate() {
                j.record(self.ctl.generation - (out.gens.len() - 1 - gi) as u64, f0, g)?;
            }
        }
        let met = self.ctl.met();
        if out.stuck && met == 0 {
            self.paused = Some("stuck: every plan dies or loses the objective at once".into());
            self.s.event("stuck", json!({"objective": fid, "state": summary(&fst), "eta": out.eta}))?;
            return Ok(true);
        }
        // Frozen frames: commits at the successes in them.
        let mut prev = 0;
        let ats: Vec<usize> = out.met_at.iter().copied().take(met).collect();
        for (j, &d) in ats.iter().enumerate() {
            self.append(&out.pads[prev..d.min(out.pads.len())])?;
            prev = d.min(out.pads.len());
            self.commit_met(&win[j].0.clone(), win[j].1.clone().as_ref(), j + 1 == met)?;
        }
        for j in ats.len()..met {
            self.commit_met(&win[j].0.clone(), win[j].1.clone().as_ref(), j + 1 == met)?;
        }
        self.append(&out.pads[prev..])?;
        if met > 0 {
            self.ctl.pop_front(met);
        }
        self.checkpoint_maybe()?;
        // Tentative successes (the best plan, behind the frontier).
        let base = self.frame();
        let td = self.ctl.tentative_dones().to_vec();
        let win_after: Vec<&(String, Arc<dyn Objective>)> = win.iter().skip(met).collect();
        let now: Vec<String> = win_after.iter().take(td.len()).map(|x| x.0.clone()).collect();
        for (k, id) in now.iter().enumerate() {
            if !self.tent.contains(id) {
                let o = &win_after[k].1;
                self.s.event("met", json!({"objective": id, "kind": o.kind(), "key": o.key(), "tentative": true, "frame": base + td[k] as u32, "frames": (base + td[k] as u32).saturating_sub(self.wip.objective_from), "note": "met by the best plan; the inputs before it may still change until the window's base passes it (`committed`)"}))?;
                self.front = None;
            }
        }
        for id in self.tent.iter().filter(|i| !now.contains(i)) {
            self.s.event("unmet", json!({"objective": id, "note": "the best plan no longer meets it"}))?;
        }
        if now.len() > self.tent.len() {
            let q = self.s.queue()?;
            if q.len() < now.len() + self.s.cfg.window {
                self.s.event("window_low", json!({"queued": q.len(), "tentatively_met": now.len(), "window": self.s.cfg.window}))?;
            }
        }
        self.tent = now;
        // Stall watch on the front objective, at the frontier.
        let (frame, eta) = (base + self.ctl.frontier() as u32, self.ctl.frontier_eta());
        if !eta.is_finite() && self.reported.insert(format!("lost:{fid}:{}", frame / 300)) {
            self.s.event("lost", json!({"objective": fid, "frame": frame, "state": summary(&fst)}))?;
        }
        if let Some(f) = self.front.as_mut() {
            if eta < f.best - 0.5 {
                f.best = eta;
                f.best_frame = frame;
            }
            if frame.saturating_sub(f.best_frame) >= f.stall_next {
                f.stall_next = f.stall_next.saturating_mul(2);
                let v = json!({"objective": fid, "key": f.key, "eta": eta, "best": f.best, "best_frame": f.best_frame, "frames_on_it": frame.saturating_sub(self.wip.objective_from), "state": summary(&fst), "hint": "inspect (`ar brief`), then insert a nearer objective, rewind, raise the budget (`ar edit --rh`), or scrap"});
                self.s.event("stalled", v)?;
                if self.s.cfg.stall_pause {
                    self.paused = Some(format!("stalled on {fid}"));
                }
            }
        }
        Ok(true)
    }

    pub fn finish(&mut self) -> Result<()> {
        if let Some(j) = &mut self.journal {
            j.flush()?;
        }
        self.write_tentative()?;
        let st = self.ctl.state().clone();
        self.ck.put(st.frame, &self.wip.id, st.zone(), &self.ctl.save_state())?;
        self.s.event("stopped", json!({"frame": st.frame, "state": summary(&st)}))?;
        let _ = std::fs::remove_file(self.s.dir.join("serve.pid"));
        Ok(())
    }
}

pub fn serve(name: &str) -> Result<()> {
    let mut d = Daemon::start(name).context("starting the daemon")?;
    let r = (|| -> Result<()> {
        while d.tick()? {}
        Ok(())
    })();
    if let Err(e) = &r {
        let _ = d.s.event("crashed", json!({"error": format!("{e:#}")}));
    }
    d.finish()?;
    r
}

/// Plays a dialog/menu objective's fixed program (macro or policy) on
/// `game` (in state `start`) until the objective is met or the program
/// ends: the inputs played, and whether it was met.
fn play_program_on(game: &mut sm64_binds::SM64Game, start: &obrrt_core::St, o: &dyn Objective, b: &Budget) -> Result<(Vec<GamePad>, bool)> {
    let mut mem = Mem::default();
    let _ = o.eta(start, start, &mut mem);
    let (mut traj, mut played) = (vec![start.clone()], Vec::new());
    let limit = match b {
        Budget::Macro(p) => p.len(),
        Budget::Policy { limit, .. } => *limit,
    };
    for f in 0..limit {
        let pad = match b {
            Budget::Macro(p) => p[f],
            Budget::Policy { policy, .. } => policy.pad(&traj, &played),
        };
        game.step_game(pad)?;
        played.push(pad);
        let st = obrrt_core::read_state(game, start.frame + played.len() as u32, true);
        let e = o.eta(&st, start, &mut mem);
        traj.push(st);
        if e.done() {
            return Ok((played, true));
        }
    }
    Ok((played, false))
}

/// The session's controller settings: the window's backward part and the
/// anchored share from the session, never below the floor.
pub fn with_session(s: &Session, mut c: RhConfig) -> RhConfig {
    c.back = s.cfg.back_frames as usize;
    c.anchor = s.cfg.back_anchor_share;
    c.press = s.cfg.auto_modal;
    crate::session::floor(c)
}
