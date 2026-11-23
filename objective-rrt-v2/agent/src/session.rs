//! A session directory: the commit tree, the working segment, the objective
//! queue and its history, the daemon's events and the CLI's inbox. Every
//! history file is append-only; nothing is ever deleted (a rewind moves HEAD
//! and branches; abandoned work becomes a commit of kind "abandoned").

use anyhow::{Context, Result, anyhow, bail};
use obrrt_core::{GamePad, St, read_state};
use obrrt_search::ga::rhstep::RhConfig;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sm64_binds::SM64Game;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;

use crate::ckpt;
use crate::util::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub name: String,
    pub rom: String,
    pub start: Value,
    pub start_desc: String,
    pub rh: RhConfig,
    pub threads: usize,
    /// Objectives solved together (the current one first).
    pub window: usize,
    /// Checkpoint every this many frames along the path (and at commits and zone entries).
    pub ckpt_every: u32,
    /// Record every generation's population.
    pub journal: bool,
    pub keyframe_every: usize,
    /// Press A through dialogs, cutscenes and menus unless the current
    /// objective is a dialog/menu objective itself.
    pub auto_modal: bool,
    /// Emit `stalled` after this many frames on one objective without a
    /// new lowest ETA (and again at each doubling).
    pub stall_frames: u32,
    /// Pause at a stall (else keep going until the agent acts).
    pub stall_pause: bool,
    /// The window's backward part: when an objective is met at frame F, the
    /// route from B is re-solved, B being at least this many frames before F
    /// and rounded back to the start of the objective it falls in; kept only
    /// if it meets the last objective sooner (route bench's `window`). 0: off.
    pub back_frames: u32,
    /// In a backward re-solve, this share of the children bred each
    /// generation keep the incumbent's inputs up to the most recent
    /// objective success (they only vary after it); the rest vary freely.
    pub back_anchor_share: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            name: String::new(),
            rom: DEFAULT_ROM.into(),
            start: json!({"boot": "own"}),
            start_desc: String::new(),
            rh: RhConfig::default(),
            threads: 8,
            window: 3,
            ckpt_every: 300,
            journal: true,
            keyframe_every: 64,
            auto_modal: true,
            stall_frames: 600,
            stall_pause: false,
            back_frames: 600,
            back_anchor_share: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Commit {
    pub id: String,
    pub parent: Option<String>,
    /// root, met, modal, manual, abandoned, split, import
    pub kind: String,
    pub objective: Option<String>,
    /// Frames from power-on at its start and end (its inputs: to − from).
    pub from: u32,
    pub to: u32,
    /// A split's inputs are a prefix of this commit's.
    #[serde(default)]
    pub prefix_of: Option<String>,
    #[serde(default)]
    pub summary: Value,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub time: f64,
}

/// The working segment: inputs since HEAD, under the id its commit will get.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wip {
    pub id: String,
    pub parent: String,
    pub from: u32,
    /// When the current front objective became current (frame).
    #[serde(default)]
    pub objective_from: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Objective {
    pub id: String,
    pub sel: Value,
    /// queued, met, scrapped
    pub status: String,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub met_commit: Option<String>,
    /// Every commit that met it (on any branch).
    #[serde(default)]
    pub met_commits: Vec<String>,
    /// Per-objective controller settings while it is current ({"g": 4, "pop": 256, ...}).
    #[serde(default)]
    pub rh: Value,
    /// Stall threshold override (frames).
    #[serde(default)]
    pub stall_frames: Option<u32>,
    #[serde(default)]
    pub added: f64,
}

pub struct Session {
    pub dir: PathBuf,
    pub cfg: Config,
}

impl Session {
    pub fn path_of(name: &str) -> PathBuf {
        if name.contains('/') { PathBuf::from(name) } else { PathBuf::from(format!("{ROOT}/agent/sessions/{name}")) }
    }

    pub fn open(name: &str) -> Result<Session> {
        let dir = Self::path_of(name);
        let cfg: Config = serde_json::from_value(read_json(dir.join("session.json")).with_context(|| format!("no session at {}", dir.display()))?)?;
        Ok(Session { dir, cfg })
    }

    /// Makes the session and its root commit (the start's inputs from
    /// power-on, checkpointed at its end).
    pub fn create(name: &str, mut cfg: Config, gg: &sm64_binds::SM64GameGenerator) -> Result<Session> {
        let dir = Self::path_of(name);
        if dir.join("session.json").exists() {
            bail!("{} exists", dir.display());
        }
        for d in ["inputs", "ckpt", "pop", "inbox"] {
            std::fs::create_dir_all(dir.join(d))?;
        }
        let (pads, desc) = source_inputs(&cfg.start)?;
        cfg.start_desc = desc;
        cfg.name = dir.file_name().unwrap().to_string_lossy().into();
        write_json(dir.join("session.json"), &serde_json::to_value(&cfg)?)?;
        let s = Session { dir, cfg };
        let mut game = gg.create_game()?;
        for p in &pads {
            game.step_game(*p)?;
        }
        let st = read_state(&game, pads.len() as u32, true);
        if !st.controllable() {
            eprintln!("warning: the start state (frame {}) is not controllable; the daemon presses through", st.frame);
        }
        std::fs::write(s.dir.join("inputs/c0000.pads"), pads_to_bytes(&pads))?;
        let root = Commit { id: "c0000".into(), parent: None, kind: "root".into(), objective: None, from: 0, to: pads.len() as u32, prefix_of: None, summary: summary(&st), note: s.cfg.start_desc.clone(), time: now() };
        append_jsonl(s.dir.join("commits.jsonl"), &serde_json::to_value(&root)?)?;
        std::fs::write(s.dir.join("HEAD"), "c0000\n")?;
        let mut ck = ckpt::Store::open(&s.dir)?;
        ck.put(st.frame, "c0000", st.zone(), &game.save_state())?;
        s.new_wip("c0000", st.frame)?;
        write_json(s.dir.join("queue.json"), &json!({"queue": []}))?;
        s.event("created", json!({"start": s.cfg.start_desc, "state": summary(&st)}))?;
        Ok(s)
    }

    pub fn save_cfg(&self) -> Result<()> {
        write_json(self.dir.join("session.json"), &serde_json::to_value(&self.cfg)?)
    }

    // ---- commits ----

    pub fn commits(&self) -> Result<Vec<Commit>> {
        read_jsonl(self.dir.join("commits.jsonl"))?.into_iter().map(|v| Ok(serde_json::from_value(v)?)).collect()
    }

    pub fn commit_map(&self) -> Result<HashMap<String, Commit>> {
        Ok(self.commits()?.into_iter().map(|c| (c.id.clone(), c)).collect())
    }

    pub fn head(&self) -> Result<String> {
        Ok(std::fs::read_to_string(self.dir.join("HEAD"))?.trim().to_string())
    }

    pub fn set_head(&self, id: &str) -> Result<()> {
        write_atomic(self.dir.join("HEAD"), format!("{id}\n").as_bytes())
    }

    /// One more than the largest cNNNN id in use (commits and the reserved
    /// work-in-progress id).
    fn next_id(&self) -> Result<String> {
        let num = |id: &str| id.strip_prefix('c').and_then(|x| x.parse::<usize>().ok());
        let mut n = self.commits()?.iter().filter_map(|c| num(&c.id)).max().map_or(0, |x| x + 1);
        if let Ok(w) = self.wip() {
            n = n.max(num(&w.id).map_or(0, |x| x + 1));
        }
        Ok(format!("c{n:04}"))
    }

    pub fn wip(&self) -> Result<Wip> {
        Ok(serde_json::from_value(read_json(self.dir.join("wip.json"))?)?)
    }

    pub fn new_wip(&self, parent: &str, from: u32) -> Result<Wip> {
        let id = self.next_id()?;
        let w = Wip { id: id.clone(), parent: parent.into(), from, objective_from: from };
        std::fs::write(self.dir.join(format!("inputs/{id}.pads")), [])?;
        write_json(self.dir.join("wip.json"), &serde_json::to_value(&w)?)?;
        Ok(w)
    }

    pub fn set_wip(&self, w: &Wip) -> Result<()> {
        write_json(self.dir.join("wip.json"), &serde_json::to_value(w)?)
    }

    pub fn pads_of(&self, id: &str) -> Result<Vec<GamePad>> {
        let p = self.dir.join(format!("inputs/{id}.pads"));
        Ok(pads_from_bytes(&std::fs::read(&p).with_context(|| p.display().to_string())?))
    }

    /// Appends inputs to the working segment.
    pub fn wip_append(&self, w: &Wip, pads: &[GamePad]) -> Result<()> {
        let mut f = std::fs::OpenOptions::new().append(true).open(self.dir.join(format!("inputs/{}.pads", w.id)))?;
        f.write_all(&pads_to_bytes(pads))?;
        Ok(())
    }

    /// The working segment's inputs so far.
    pub fn wip_pads(&self) -> Result<(Wip, Vec<GamePad>)> {
        let w = self.wip()?;
        let p = self.pads_of(&w.id)?;
        Ok((w, p))
    }

    /// Turns the working segment into a commit (kind met / modal / manual /
    /// abandoned) and starts a new one after it (unless abandoned: then the
    /// new one starts from the old parent). Returns the commit.
    pub fn cut(&self, kind: &str, objective: Option<&str>, st: Option<&St>, note: &str) -> Result<Commit> {
        let (w, pads) = self.wip_pads()?;
        let c = Commit {
            id: w.id.clone(),
            parent: Some(w.parent.clone()),
            kind: kind.into(),
            objective: objective.map(Into::into),
            from: w.from,
            to: w.from + pads.len() as u32,
            prefix_of: None,
            summary: st.map(summary).unwrap_or(Value::Null),
            note: note.into(),
            time: now(),
        };
        append_jsonl(self.dir.join("commits.jsonl"), &serde_json::to_value(&c)?)?;
        if kind == "abandoned" {
            self.new_wip(&w.parent, w.from)?;
        } else {
            self.set_head(&c.id)?;
            self.new_wip(&c.id, c.to)?;
        }
        Ok(c)
    }

    /// A new commit holding the first `to − from` inputs of `of` (a rewind
    /// into the middle of a segment).
    pub fn split(&self, of: &Commit, to: u32, st: Option<&St>, note: &str) -> Result<Commit> {
        let id = self.next_id()?;
        let pads = self.pads_of(&of.id)?;
        let n = (to - of.from) as usize;
        std::fs::write(self.dir.join(format!("inputs/{id}.pads")), pads_to_bytes(&pads[..n]))?;
        let c = Commit {
            id,
            parent: of.parent.clone(),
            kind: "split".into(),
            objective: None,
            from: of.from,
            to,
            prefix_of: Some(of.id.clone()),
            summary: st.map(summary).unwrap_or(Value::Null),
            note: note.into(),
            time: now(),
        };
        append_jsonl(self.dir.join("commits.jsonl"), &serde_json::to_value(&c)?)?;
        Ok(c)
    }

    /// The chain root → `id`.
    pub fn chain(&self, id: &str) -> Result<Vec<Commit>> {
        let m = self.commit_map()?;
        let mut out = Vec::new();
        let mut cur = Some(id.to_string());
        while let Some(c) = cur {
            let x = m.get(&c).ok_or_else(|| anyhow!("no commit {c}"))?.clone();
            cur = x.parent.clone();
            out.push(x);
        }
        out.reverse();
        Ok(out)
    }

    /// Inputs from power-on along HEAD (and the working segment).
    pub fn path_pads(&self, head: &str, with_wip: bool) -> Result<Vec<GamePad>> {
        let mut out = Vec::new();
        for c in self.chain(head)? {
            out.extend(self.pads_of(&c.id)?);
        }
        if with_wip {
            let (w, p) = self.wip_pads()?;
            if w.parent == head {
                let end = w.from + p.len() as u32;
                out.extend(p);
                // The daemon's tentative route (base to frontier), if it is on this end.
                if let Ok(t) = read_json(self.dir.join("tentative.json")) {
                    if t["wip"] == w.id.as_str() && t["from"].as_u64() == Some(end as u64) {
                        out.extend(pads_from_bytes(&std::fs::read(self.dir.join("inputs/tentative.pads")).unwrap_or_default()));
                    }
                }
            }
        }
        Ok(out)
    }

    /// Segments whose checkpoints lie on HEAD's path (with the working one),
    /// as owner → last valid frame.
    pub fn valid_owners(&self, head: &str, with_wip: bool) -> Result<HashMap<String, u32>> {
        let mut v = HashMap::new();
        for c in self.chain(head)? {
            v.insert(c.id.clone(), c.to);
            if let Some(p) = &c.prefix_of {
                v.insert(p.clone(), c.to);
            }
        }
        if with_wip {
            let (w, p) = self.wip_pads()?;
            if w.parent == head {
                v.insert(w.id.clone(), w.from + p.len() as u32);
            }
        }
        Ok(v)
    }

    /// The game at `frame` on HEAD's path (nearest checkpoint, then a replay
    /// of at most `ckpt_every` frames); the state read with objects.
    pub fn load_at(&self, ck: &mut ckpt::Store, game: &mut SM64Game, head: &str, frame: Option<u32>) -> Result<St> {
        let pads = self.path_pads(head, true)?;
        let target = frame.unwrap_or(pads.len() as u32);
        anyhow::ensure!(target as usize <= pads.len(), "frame {target} is past the path's end ({})", pads.len());
        let owners = self.valid_owners(head, true)?;
        let from = match ck.best(target, |o, f| owners.get(o).is_some_and(|&to| f <= to)) {
            Some(e) => {
                game.load_state(&ck.get(&e)?)?;
                e.frame
            }
            None => {
                *game = sm64_binds::SM64GameGenerator::from_file(&self.cfg.rom)?.create_game()?;
                0
            }
        };
        for p in &pads[from as usize..target as usize] {
            game.step_game(*p)?;
        }
        Ok(read_state(game, target, true))
    }

    // ---- objectives ----

    pub fn objectives(&self) -> Result<Vec<Objective>> {
        let mut m: Vec<Objective> = Vec::new();
        for v in read_jsonl(self.dir.join("objectives.jsonl"))? {
            let id = v["id"].as_str().unwrap_or("").to_string();
            match v["ev"].as_str().unwrap_or("") {
                "add" => m.push(Objective {
                    id,
                    sel: v["sel"].clone(),
                    status: "queued".into(),
                    note: v["note"].as_str().unwrap_or("").into(),
                    rh: v.get("rh").cloned().unwrap_or(Value::Null),
                    stall_frames: v["stall_frames"].as_u64().map(|x| x as u32),
                    added: v["t"].as_f64().unwrap_or(0.0),
                    met_commit: None,
                    met_commits: Vec::new(),
                }),
                ev => {
                    if let Some(o) = m.iter_mut().find(|o| o.id == id) {
                        match ev {
                            "status" => {
                                o.status = v["status"].as_str().unwrap_or("").into();
                                o.met_commit = v["commit"].as_str().map(Into::into);
                                if let Some(c) = &o.met_commit {
                                    if !o.met_commits.contains(c) {
                                        o.met_commits.push(c.clone());
                                    }
                                }
                            }
                            "edit" => {
                                if let Some(s) = v.get("sel").filter(|s| !s.is_null()) {
                                    o.sel = s.clone();
                                }
                                if let Some(r) = v.get("rh").filter(|s| !s.is_null()) {
                                    o.rh = r.clone();
                                }
                                if let Some(n) = v["note"].as_str() {
                                    o.note = n.into();
                                }
                                if let Some(n) = v["stall_frames"].as_u64() {
                                    o.stall_frames = Some(n as u32);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        Ok(m)
    }

    pub fn obj_event(&self, mut v: Value) -> Result<()> {
        v["t"] = json!(now());
        append_jsonl(self.dir.join("objectives.jsonl"), &v)
    }

    pub fn queue(&self) -> Result<Vec<String>> {
        let v = read_json(self.dir.join("queue.json"))?;
        Ok(v["queue"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(Into::into)).collect()).unwrap_or_default())
    }

    pub fn set_queue(&self, q: &[String]) -> Result<()> {
        write_json(self.dir.join("queue.json"), &json!({"queue": q}))
    }

    pub fn next_obj_id(&self) -> Result<String> {
        Ok(format!("o{:04}", self.objectives()?.len() + 1))
    }

    // ---- events, inbox ----

    pub fn event(&self, ev: &str, mut v: Value) -> Result<()> {
        let p = self.dir.join("events.jsonl");
        let seq = std::fs::read_to_string(&p).map(|s| s.lines().count()).unwrap_or(0);
        if !v.is_object() {
            v = json!({"data": v});
        }
        v["ev"] = json!(ev);
        v["seq"] = json!(seq + 1);
        v["t"] = json!(now());
        append_jsonl(p, &v)
    }

    pub fn events(&self) -> Result<Vec<Value>> {
        read_jsonl(self.dir.join("events.jsonl"))
    }

    pub fn serve_pid(&self) -> Option<u32> {
        let pid: u32 = std::fs::read_to_string(self.dir.join("serve.pid")).ok()?.trim().parse().ok()?;
        pid_alive(pid).then_some(pid)
    }

    /// Queues a command for the daemon.
    pub fn send(&self, cmd: &Value) -> Result<String> {
        let id = format!("{:.6}-{}", now(), std::process::id());
        write_atomic(self.dir.join(format!("inbox/{id}.json")), serde_json::to_string(cmd)?.as_bytes())?;
        Ok(id)
    }

    /// Takes the inbox's commands in order.
    pub fn take_inbox(&self) -> Result<Vec<(String, Value)>> {
        let mut names: Vec<String> = std::fs::read_dir(self.dir.join("inbox"))?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".json"))
            .collect();
        names.sort();
        let mut out = Vec::new();
        for n in names {
            let p = self.dir.join("inbox").join(&n);
            let v = read_json(&p)?;
            std::fs::remove_file(&p)?;
            out.push((n.trim_end_matches(".json").to_string(), v));
        }
        Ok(out)
    }

    pub fn journal_path(&self, seg: &str) -> PathBuf {
        self.dir.join(format!("pop/{seg}.pj"))
    }

    /// Commits on HEAD's path, as a set.
    pub fn on_path(&self, head: &str) -> Result<HashSet<String>> {
        Ok(self.chain(head)?.into_iter().map(|c| c.id).collect())
    }
}

/// The user's gold standard is the least compute a step may have:
/// population ≥ 128, plans ≥ 600 frames, ≥ 1 generation per step, at most
/// 1 frame committed per step, pink noise in fresh populations.
pub fn below_floor(c: &RhConfig) -> Option<String> {
    let mut v = Vec::new();
    if c.pop < 128 {
        v.push(format!("pop {} < 128", c.pop));
    }
    if c.h < 600 {
        v.push(format!("h {} < 600", c.h));
    }
    if c.g < 1 {
        v.push("g < 1".to_string());
    }
    if c.k > 1 {
        v.push(format!("k {} > 1 (commit one frame at a time)", c.k));
    }
    if c.pink <= 0.0 {
        v.push("pink 0 (fresh populations must have pink noise)".to_string());
    }
    (!v.is_empty()).then(|| format!("below the compute floor (pop 128, h 600, g 1, k 1, pink): {}", v.join(", ")))
}

/// Raises a config to the floor.
pub fn floor(mut c: RhConfig) -> RhConfig {
    c.pop = c.pop.max(128);
    c.h = c.h.max(600);
    c.g = c.g.max(1);
    c.k = 1.min(c.k.max(1));
    if c.pink <= 0.0 {
        c.pink = 1.0;
    }
    c
}

/// A state's short summary (commit records, events).
pub fn summary(st: &St) -> Value {
    json!({
        "frame": st.frame,
        "level": st.level, "area": st.area, "act": st.act, "course": st.course,
        "pos": [st.pos[0].round(), st.pos[1].round(), st.pos[2].round()],
        "action": format!("{:#x}", st.action),
        "stars": st.stars, "keys": st.keys, "coins": st.coins, "health": st.health >> 8, "lives": st.lives,
        "save_flags": format!("{:#x}", st.save_flags),
        "controllable": st.controllable(),
    })
}
