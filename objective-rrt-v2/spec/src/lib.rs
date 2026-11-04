//! Declarative ("spec") objectives: one generic [`Objective`] interpreted
//! from JSON, so an objective can be written (by hand or by an LLM) without
//! Rust. tools/spec/FORMAT.md documents the format; tools/spec/README.md the
//! validator, the context packer and the Eureka-style loop around them.
//!
//! A spec is a gate (when it may start), zones, and phases. Each phase has
//! a target, a `done` predicate, a pace or fixed frames, and optional latch
//! and repeat. The ETA (GOALS.md's rules) is the current phase's distance
//! / pace + its frames, plus every later phase's frames, a later phase's
//! distance measured from the previous phase's target. Phase progress is
//! kept in [`Mem`] (f[0] phase, f[1] repeat count, f[2] the phase's first
//! frame; flags bit 0 initialised, bit 1 `done` held last frame).
//!
//! Hooks (`OBRRT_SPEC=<file.json or dir>`): [`wrap`] adds the specs to a
//! generator (obrrt2), [`extra`] lists them for one state (oss2). Unset,
//! neither changes anything.

pub mod actions;
mod eval;
pub mod json;
pub mod model;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result, anyhow, bail};
use obrrt_core::{Arrival, Eta, GaBudget, Generator, Mem, Objective, Precision, St};
use serde_json::Value;

pub use eval::{find, holds, target_dist};
pub use eval::Cx as PredCx;
use eval::Cx;
use json::{J, check_keys, err, idx, key};
use model::*;

/// Mem flag: initialised (phase 0 set up).
const INIT: u32 = 1;
/// Mem flag: the current phase's `done` held on the previous frame (repeat
/// edges).
const PREV_DONE: u32 = 2;
/// Mem flag: f[3] and flags bits 8-25 hold the `thrown_off` object's
/// position last frame (x, z at 1/16 unit packed in f[3]; y at 1/4 unit).
const PREV_OBJ: u32 = 4;

fn pack_prev(mem: &mut Mem, p: Option<[f32; 3]>) {
    let Some(p) = p else {
        mem.flags &= !PREV_OBJ;
        return;
    };
    let q = |x: f32, k: f32, max: u64| (((x + 16384.0) * k).round().max(0.0) as u64).min(max);
    let (x, z) = (q(p[0], 16.0, (1 << 19) - 1), q(p[2], 16.0, (1 << 19) - 1));
    mem.f[3] = (x * (1u64 << 19) + z) as f64;
    let y = q(p[1], 4.0, (1 << 17) - 1) as u32;
    mem.flags = (mem.flags & 0xFF) | PREV_OBJ | (y << 8);
}

fn unpack_prev(mem: &Mem) -> Option<[f32; 3]> {
    if mem.flags & PREV_OBJ == 0 {
        return None;
    }
    let v = mem.f[3] as u64;
    let (x, z) = (v >> 19, v & ((1 << 19) - 1));
    let y = (mem.flags >> 8) & ((1 << 17) - 1);
    Some([x as f32 / 16.0 - 16384.0, y as f32 / 4.0 - 16384.0, z as f32 / 16.0 - 16384.0])
}
/// Most variants a spec may expand into.
pub const MAX_VARIANTS: usize = 8;
/// Frames a further repetition costs when neither the repeat nor the phase
/// says (a re-pound: off the post, up, down; switches.rs REPOUND_FRAMES 35).
const REPEAT_FRAMES: f64 = 30.0;

/// One spec (one variant of it), ready to be an objective.
#[derive(Debug, Clone)]
pub struct Spec {
    pub name: String,
    pub variant: Option<String>,
    kind: &'static str,
    key: String,
    pub zones: Vec<Zone>,
    pub gate: Pred,
    pub phases: Vec<Phase>,
    pub lost: Option<Pred>,
    pub hints: Hints,
    /// For the validator: the milestone this spec is meant to reach.
    pub expect: Option<Pred>,
    needs_objects: bool,
    /// Some predicate uses `thrown_off` (the object's last position is kept).
    tracks: bool,
}

const TOP_KEYS: &[&str] = &["name", "zone", "zones", "gate", "phases", "lost", "hints", "variants", "params", "expect"];

/// Replaces every string `"$name"` in `v` by `params[name]`.
fn substitute(v: &Value, params: &serde_json::Map<String, Value>, path: &str) -> Result<Value> {
    Ok(match v {
        Value::String(s) if s.starts_with('$') => {
            let k = &s[1..];
            params.get(k).cloned().ok_or_else(|| anyhow!("{path}: \"{s}\" names no parameter (params: {})", params.keys().cloned().collect::<Vec<_>>().join(", ")))?
        }
        Value::Array(a) => Value::Array(a.iter().enumerate().map(|(i, x)| substitute(x, params, &idx(path, i))).collect::<Result<_>>()?),
        Value::Object(m) => Value::Object(
            m.iter()
                .filter(|(k, _)| k.as_str() != "variants" && k.as_str() != "params")
                .map(|(k, x)| Ok((k.clone(), substitute(x, params, &key(path, k))?)))
                .collect::<Result<_>>()?,
        ),
        v => v.clone(),
    })
}

impl Spec {
    /// Parses one spec object into its variants (one without `variants`).
    pub fn parse(v: &Value, path: &str, nodes: Option<&NodePos>) -> Result<Vec<Spec>> {
        let m = J::new(v, path).obj()?;
        check_keys(m, path, TOP_KEYS)?;
        let pn = key(path, "name");
        let name = J::new(m.get("name").ok_or_else(|| anyhow!("{path}: a spec needs \"name\""))?, &pn).str()?.to_string();
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.') {
            return err(&pn, format!("\"{name}\": use letters, digits, _ - . only"));
        }
        let defaults = match m.get("params") {
            Some(p) => J::new(p, &key(path, "params")).obj()?.clone(),
            None => Default::default(),
        };
        let mut variants: Vec<(Option<String>, serde_json::Map<String, Value>)> = Vec::new();
        match m.get("variants") {
            None => variants.push((None, defaults.clone())),
            Some(vs) => {
                let pv = key(path, "variants");
                let a = J::new(vs, &pv).arr()?;
                if a.is_empty() || a.len() > MAX_VARIANTS {
                    return err(&pv, format!("1 to {MAX_VARIANTS} variants"));
                }
                for (i, x) in a.iter().enumerate() {
                    let pi = idx(&pv, i);
                    let vm = J::new(x, &pi).obj()?;
                    check_keys(vm, &pi, &["name", "params"])?;
                    let vname = match vm.get("name") {
                        Some(n) => J::new(n, &key(&pi, "name")).str()?.to_string(),
                        None => format!("v{i}"),
                    };
                    if variants.iter().any(|(n, _)| n.as_deref() == Some(vname.as_str())) {
                        return err(&pi, format!("variant name \"{vname}\" used twice"));
                    }
                    let mut params = defaults.clone();
                    if let Some(p) = vm.get("params") {
                        for (k, x) in J::new(p, &key(&pi, "params")).obj()? {
                            params.insert(k.clone(), x.clone());
                        }
                    }
                    variants.push((Some(vname), params));
                }
            }
        }
        variants
            .into_iter()
            .map(|(vname, params)| {
                let doc = substitute(v, &params, path)?;
                let at = match &vname {
                    Some(n) => format!("{path}(variant {n})"),
                    None => path.to_string(),
                };
                Spec::parse_one(&doc, &at, name.clone(), vname, nodes)
            })
            .collect()
    }

    fn parse_one(v: &Value, path: &str, name: String, variant: Option<String>, nodes: Option<&NodePos>) -> Result<Spec> {
        let m = v.as_object().unwrap();
        let zones = match (m.get("zone"), m.get("zones")) {
            (Some(z), None) => parse_zones(J::new(z, &key(path, "zone")))?,
            (None, Some(z)) => parse_zones(J::new(z, &key(path, "zones")))?,
            _ => return err(path, "give exactly one of \"zone\" or \"zones\""),
        };
        let gate = match m.get("gate") {
            Some(g) => parse_pred(J::new(g, &key(path, "gate")), nodes)?,
            None => Pred::All(Vec::new()),
        };
        if gate.uses_target() {
            return err(&key(path, "gate"), "the gate has no phase target (dist/facing need an explicit \"to\")");
        }
        let pp = key(path, "phases");
        let pa = J::new(m.get("phases").ok_or_else(|| anyhow!("{path}: a spec needs \"phases\""))?, &pp).arr()?;
        if pa.is_empty() || pa.len() > 16 {
            return err(&pp, "1 to 16 phases");
        }
        let phases = pa
            .iter()
            .enumerate()
            .map(|(i, x)| parse_phase(J::new(x, &idx(&pp, i)), i, nodes))
            .collect::<Result<Vec<_>>>()?;
        let lost = match m.get("lost") {
            Some(l) => Some(parse_pred(J::new(l, &key(path, "lost")), nodes)?),
            None => None,
        };
        if lost.as_ref().is_some_and(|l| l.uses_target()) {
            return err(&key(path, "lost"), "uses the phase target; give dist/hdist/dy an explicit \"to\"");
        }
        let hints = match m.get("hints") {
            Some(h) => {
                let ph = key(path, "hints");
                let hm = J::new(h, &ph).obj()?;
                check_keys(hm, &ph, HINT_KEYS)?;
                parse_hints(hm, &ph)?
            }
            None => parse_hints(&Default::default(), path)?,
        };
        let expect = match m.get("expect") {
            Some(e) => Some(parse_pred(J::new(e, &key(path, "expect")), nodes)?),
            None => None,
        };
        let needs_objects = gate.needs_objects()
            || lost.as_ref().is_some_and(|l| l.needs_objects())
            || phases.iter().any(|p| {
                p.done.needs_objects()
                    || matches!(p.target.as_ref().map(|t| &t.shape), Some(Shape::Object(_)))
                    || p.repeat.as_ref().is_some_and(|r| r.count.is_some())
            });
        let tracks = lost.as_ref().is_some_and(|l| l.thrown_ref().is_some())
            || phases.iter().any(|p| p.done.thrown_ref().is_some() || p.penalty.iter().any(|(w, _)| w.thrown_ref().is_some()));
        let needs_objects = needs_objects || phases.iter().any(|p| p.penalty.iter().any(|(w, _)| w.needs_objects()));
        let kind: &'static str = Box::leak(format!("Spec::{name}").into_boxed_str());
        let key = match &variant {
            Some(v) => format!("Spec {name} {v}"),
            None => format!("Spec {name}"),
        };
        Ok(Spec { name, variant, kind, key, zones, gate, phases, lost, hints, expect, needs_objects, tracks })
    }

    /// The same spec with its GA budget hint set (legs' per-leg budgets).
    pub fn clone_with_budget(&self, b: GaBudget) -> Spec {
        let mut s = self.clone();
        s.hints.ga_budget = Some(b);
        s
    }

    pub fn label(&self) -> String {
        match &self.variant {
            Some(v) => format!("{}/{v}", self.name),
            None => self.name.clone(),
        }
    }

    fn in_zone(&self, s: &St) -> bool {
        zone_has(&self.zones, s.zone())
    }

    /// Whether the gate holds at `s` (as a start) in a listed zone.
    pub fn gate_holds(&self, s: &St) -> bool {
        self.in_zone(s) && holds(&self.gate, &Cx { s, start: s, target: None, phase_start: s.frame })
    }

    /// Generated from `b`: the gate holds there, outside dialogs and menus.
    pub fn applies(&self, b: &St) -> bool {
        !b.modal() && self.gate_holds(b)
    }

    fn phase_done(&self, i: usize, s: &St, start: &St, phase_start: u32) -> bool {
        let p = &self.phases[i];
        holds(&p.done, &Cx { s, start, target: p.target.as_ref(), phase_start })
    }

    /// Whether phase `i` (the current one) is satisfied this frame, updating
    /// its repeat count.
    fn phase_met(&self, i: usize, s: &St, start: &St, mem: &mut Mem) -> bool {
        let p = &self.phases[i];
        let phase_start = mem.f[2] as u32;
        match &p.repeat {
            None => self.phase_done(i, s, start, phase_start),
            Some(r) => {
                if let Some((obj, f, per)) = &r.count {
                    let cx = Cx { s, start, target: p.target.as_ref(), phase_start };
                    let c = eval::count_of(&cx, obj, *f, *per);
                    if let Some(c) = c {
                        mem.f[1] = c as f64;
                    }
                    // An explicit `done` must hold too.
                    c.is_some_and(|c| c >= r.n) && (!p.done_explicit || self.phase_done(i, s, start, phase_start))
                } else {
                    let now = self.phase_done(i, s, start, phase_start);
                    if now && mem.flags & PREV_DONE == 0 {
                        mem.f[1] += 1.0;
                    }
                    if now {
                        mem.flags |= PREV_DONE;
                    } else {
                        mem.flags &= !PREV_DONE;
                    }
                    mem.f[1] as u32 >= r.n
                }
            }
        }
    }

    fn enter_phase(i: usize, s: &St, mem: &mut Mem) {
        mem.f[0] = i as f64;
        mem.f[1] = 0.0;
        mem.f[2] = s.frame as f64;
        mem.flags &= !PREV_DONE;
    }

    /// Advances the phase pointer in `mem` for state `s`; returns the
    /// current phase (`phases.len()` = all done).
    pub fn step(&self, s: &St, start: &St, mem: &mut Mem) -> usize {
        if mem.flags & INIT == 0 {
            mem.flags |= INIT;
            Self::enter_phase(0, s, mem);
        }
        let mut cur = mem.f[0] as usize;
        // Unlatched earlier phases must keep holding.
        for j in 0..cur.min(self.phases.len()) {
            if !self.phases[j].latch && !self.phase_done(j, s, start, s.frame) {
                cur = j;
                Self::enter_phase(j, s, mem);
                break;
            }
        }
        while cur < self.phases.len() && self.phase_met(cur, s, start, mem) {
            cur += 1;
            Self::enter_phase(cur, s, mem);
        }
        cur
    }

    /// Frames for phase `i` from point `from` (Mario for the current phase,
    /// the previous target for later ones); `reps_done` repetitions done
    /// and `elapsed` frames into it (only for the current phase).
    fn phase_frames(&self, i: usize, s: &St, from: [f32; 3], reps_done: u32, elapsed: f64) -> f64 {
        let p = &self.phases[i];
        let travel = p.target.as_ref().and_then(|t| target_dist(t, s, from)).map_or(0.0, |(d, _)| d as f64 / p.pace);
        // A phase without a target counts its fixed frames down.
        let fixed = if p.target.is_none() { (p.frames - elapsed).max(if p.frames > 0.0 { 1.0 } else { 0.0 }) } else { p.frames };
        let reps = p.repeat.as_ref().map_or(0.0, |r| {
            let per = r.frames.unwrap_or(if p.frames > 0.0 { p.frames } else { REPEAT_FRAMES });
            r.n.saturating_sub(reps_done + 1) as f64 * per
        });
        travel + fixed + reps
    }

    /// Where phase `i` ends, for measuring the next phase from.
    fn phase_end(&self, i: usize, s: &St, prev: [f32; 3]) -> [f32; 3] {
        self.phases[i].target.as_ref().and_then(|t| target_dist(t, s, prev)).map_or(prev, |(_, c)| c)
    }

    /// The ETA and current phase (`phases.len()` when done).
    pub fn eta_phase(&self, s: &St, start: &St, mem: &mut Mem) -> (Eta, usize) {
        if !self.tracks {
            return self.eta_inner(s, start, mem);
        }
        // `thrown_off`: the tracked object's position last frame in, this
        // frame's out (the current phase's reference, else the lost one's).
        eval::PREV_OBJ.with(|c| c.set(unpack_prev(mem)));
        let out = self.eta_inner(s, start, mem);
        let cur = (mem.f[0] as usize).min(self.phases.len() - 1);
        let p = &self.phases[cur];
        let r = p.done.thrown_ref().or_else(|| p.penalty.iter().find_map(|(w, _)| w.thrown_ref())).or_else(|| self.lost.as_ref().and_then(|l| l.thrown_ref()));
        let pos = r.and_then(|r| eval::resolve(r, p.target.as_ref(), s)).map(|o| o.pos);
        pack_prev(mem, pos);
        eval::PREV_OBJ.with(|c| c.set(None));
        out
    }

    fn eta_inner(&self, s: &St, start: &St, mem: &mut Mem) -> (Eta, usize) {
        if let Some(l) = &self.lost {
            if holds(l, &Cx { s, start, target: None, phase_start: mem.f[2] as u32 }) {
                return (Eta::Lost, mem.f[0] as usize);
            }
        }
        let cur = self.step(s, start, mem);
        cache_phase(self, s.frame, cur);
        if cur >= self.phases.len() {
            return (Eta::Done, cur);
        }
        if !self.in_zone(s) {
            return (Eta::Lost, cur);
        }
        let p = &self.phases[cur];
        if let Some(Target { shape: Shape::Object(sel), .. }) = &p.target {
            if sel.required && s.objects.is_some() && find(sel, s).is_none() {
                return (Eta::Lost, cur);
            }
        }
        let elapsed = s.frame as f64 - mem.f[2];
        let mut total = self.phase_frames(cur, s, s.pos, mem.f[1] as u32, elapsed);
        for (when, f) in &p.penalty {
            if holds(when, &Cx { s, start, target: p.target.as_ref(), phase_start: mem.f[2] as u32 }) {
                total += f;
            }
        }
        let mut at = self.phase_end(cur, s, s.pos);
        for j in cur + 1..self.phases.len() {
            total += self.phase_frames(j, s, at, 0, 0.0);
            at = self.phase_end(j, s, at);
        }
        (Eta::Frames(total), cur)
    }

    fn steer_phase(&self, s: &St) -> usize {
        cached_phase(self, s.frame).unwrap_or_else(|| {
            // Stateless guess: the first phase whose done doesn't hold now.
            (0..self.phases.len()).find(|&i| !self.phase_done(i, s, s, s.frame)).unwrap_or(self.phases.len() - 1)
        })
    }
}

thread_local! {
    /// Per thread: (spec, frame) -> its current phase from the last eta
    /// call, for steering (which gets no Mem).
    static PHASE: RefCell<HashMap<usize, (u32, usize)>> = RefCell::new(HashMap::new());
}

fn cache_phase(spec: &Spec, frame: u32, phase: usize) {
    PHASE.with(|m| {
        m.borrow_mut().insert(spec as *const Spec as usize, (frame, phase));
    });
}

fn cached_phase(spec: &Spec, frame: u32) -> Option<usize> {
    PHASE.with(|m| {
        m.borrow().get(&(spec as *const Spec as usize)).filter(|(f, _)| frame.abs_diff(*f) <= 1).map(|&(_, p)| p.min(spec.phases.len() - 1))
    })
}

impl Objective for Spec {
    fn kind(&self) -> &'static str {
        self.kind
    }
    fn key(&self) -> String {
        self.key.clone()
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        self.hints.obj_weight
    }
    fn base_criteria(&self, s: &St) -> bool {
        self.gate_holds(s)
    }
    fn exists_in(&self, s: &St) -> bool {
        self.gate_holds(s)
    }
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        self.eta_phase(s, start, mem).0
    }
    fn ga_budget(&self) -> Option<GaBudget> {
        self.hints.ga_budget
    }
    fn precision(&self) -> Precision {
        self.hints.precision
    }
    fn kinship(&self) -> u32 {
        self.hints.kinship
    }
    fn temperature(&self) -> f64 {
        self.hints.temperature.unwrap_or(obrrt_core::START_TEMPERATURE)
    }
    fn done_frames(&self, _s: &St, _mem: &Mem) -> f64 {
        self.hints.done_frames
    }
    fn needs_objects(&self) -> bool {
        self.needs_objects
    }
    fn success_value(&self) -> f64 {
        self.hints.success_value
    }
    fn diag_tag(&self) -> Option<String> {
        Some(format!("spec/{}", self.label()))
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let p = &self.phases[self.steer_phase(s)];
        let t = match &p.steer {
            Some(None) => return None,
            Some(Some(t)) => t,
            None => p.target.as_ref()?,
        };
        target_dist(t, s, s.pos).map(|(_, c)| c)
    }
    fn steer_button(&self, s: &St) -> u16 {
        let p = &self.phases[self.steer_phase(s)];
        let Some((b, within)) = p.button.or(self.hints.steer_button) else { return 0 };
        let near = p.target.as_ref().and_then(|t| target_dist(t, s, s.pos)).is_none_or(|(d, _)| d <= within);
        if near { b } else { 0 }
    }
    fn arrival(&self, s: &St) -> Arrival {
        self.phases[self.steer_phase(s)].arrival.unwrap_or(self.hints.arrival)
    }
}

/// Node positions from a run's nodes.csv (`{"node": id}` targets).
pub fn node_positions(run_dir: &str) -> Result<HashMap<u32, [f32; 3]>> {
    let csv = std::fs::read_to_string(format!("{run_dir}/nodes.csv")).with_context(|| format!("reading {run_dir}/nodes.csv"))?;
    let mut out = HashMap::new();
    for l in csv.lines().skip(1) {
        let c: Vec<&str> = l.split(',').collect();
        if c.len() > 5 {
            if let (Ok(id), Ok(x), Ok(y), Ok(z)) = (c[0].parse(), c[3].parse(), c[4].parse(), c[5].parse()) {
                out.insert(id, [x, y, z]);
            }
        }
    }
    Ok(out)
}

/// Parses a spec document: one spec object, an array of them, or
/// `{"specs": [...]}`. `origin` names it in errors.
pub fn parse_doc(text: &str, origin: &str, nodes: Option<&NodePos>) -> Result<Vec<Spec>> {
    let v: Value = serde_json::from_str(text).map_err(|e| anyhow!("{origin}: not JSON: {e}"))?;
    let (list, base): (Vec<&Value>, String) = match &v {
        Value::Array(a) => (a.iter().collect(), "$".into()),
        Value::Object(m) if m.contains_key("specs") => {
            check_keys(m, "$", &["specs"])?;
            (J::new(&m["specs"], "$.specs").arr()?.iter().collect(), "$.specs".into())
        }
        Value::Object(_) => (vec![&v], String::new()),
        _ => bail!("{origin}: expected a spec object, an array of specs or {{\"specs\": [...]}}"),
    };
    let mut out = Vec::new();
    for (i, s) in list.iter().enumerate() {
        let path = if base.is_empty() { "$".to_string() } else { idx(&base, i) };
        out.extend(Spec::parse(s, &path, nodes).map_err(|e| anyhow!("{origin}: {e}"))?);
    }
    let mut keys = std::collections::HashSet::new();
    for s in &out {
        if !keys.insert(s.key.clone()) {
            bail!("{origin}: two specs share the name/variant \"{}\"", s.label());
        }
    }
    Ok(out)
}

/// Loads a spec file, or every `*.json` of a directory (sorted).
pub fn load(path: &str, run_dir: Option<&str>) -> Result<Vec<Spec>> {
    let nodes = match run_dir {
        Some(d) => Some(node_positions(d)?),
        None => None,
    };
    let lookup = nodes.map(|n| move |id: u32| n.get(&id).copied());
    let nodes_fn: Option<&NodePos> = lookup.as_ref().map(|f| f as &NodePos);
    let meta = std::fs::metadata(path).with_context(|| format!("OBRRT_SPEC: {path}"))?;
    let files: Vec<String> = if meta.is_dir() {
        let mut v: Vec<String> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    } else {
        vec![path.to_string()]
    };
    let mut out = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f).with_context(|| format!("reading {f}"))?;
        out.extend(parse_doc(&text, &f, nodes_fn)?);
    }
    let mut keys = std::collections::HashSet::new();
    for s in &out {
        if !keys.insert(s.key.clone()) {
            bail!("{path}: two specs share the name/variant \"{}\"", s.label());
        }
    }
    Ok(out)
}

/// The specs named by `OBRRT_SPEC` (node targets from `OBRRT_SPEC_RUN`),
/// loaded once; None when unset.
pub fn from_env() -> Result<Option<&'static [Arc<Spec>]>> {
    static SPECS: OnceLock<Option<Vec<Arc<Spec>>>> = OnceLock::new();
    if SPECS.get().is_none() {
        let v = match std::env::var("OBRRT_SPEC") {
            Ok(p) if !p.is_empty() => {
                let run = std::env::var("OBRRT_SPEC_RUN").ok();
                let specs = load(&p, run.as_deref())?;
                eprintln!("OBRRT_SPEC: {} spec objectives from {p}: {}", specs.len(), specs.iter().map(|s| s.label()).collect::<Vec<_>>().join(", "));
                Some(specs.into_iter().map(Arc::new).collect())
            }
            _ => None,
        };
        let _ = SPECS.set(v);
    }
    Ok(SPECS.get().unwrap().as_deref())
}

/// The specs that apply at `b` (empty with `OBRRT_SPEC` unset). For oss2's
/// GA, whose eligible list is built from Registry directly. Panics on a bad
/// spec file (load it with [`from_env`] at startup to fail early).
pub fn extra(b: &St) -> Vec<Arc<dyn Objective>> {
    match from_env().expect("OBRRT_SPEC") {
        None => Vec::new(),
        Some(specs) => specs.iter().filter(|s| s.applies(b)).map(|s| s.clone() as Arc<dyn Objective>).collect(),
    }
}

/// A generator plus the specs.
struct WithSpecs {
    inner: Arc<dyn Generator>,
    specs: &'static [Arc<Spec>],
}

impl Generator for WithSpecs {
    fn generate(&self, b: &St, rng: &mut rand::rngs::SmallRng) -> Vec<Arc<dyn Objective>> {
        let mut out = self.inner.generate(b, rng);
        out.extend(self.specs.iter().filter(|s| s.applies(b)).map(|s| s.clone() as Arc<dyn Objective>));
        out
    }
    fn wants_records(&self) -> bool {
        self.inner.wants_records()
    }
    fn summary_blocks(&self) -> Vec<(String, String)> {
        let mut v = self.inner.summary_blocks();
        let names: Vec<String> = self.specs.iter().map(|s| format!("\"{}\"", s.label())).collect();
        v.push(("spec".into(), format!("{{\"specs\": [{}]}}", names.join(", "))));
        v
    }
}

/// `inner` plus the `OBRRT_SPEC` specs; `inner` itself when unset.
pub fn wrap(inner: Arc<dyn Generator>) -> Result<Arc<dyn Generator>> {
    Ok(match from_env()? {
        None => inner,
        Some(specs) => Arc::new(WithSpecs { inner, specs }),
    })
}
