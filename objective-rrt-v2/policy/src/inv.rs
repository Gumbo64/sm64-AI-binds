//! The policy ecology: a persistent, append-only inventory of policies
//! (rule-list GA, Koza GP, LLM-written, hand, portfolio, lambda), quality-
//! diversity niches, market-share sampling with adaptive weights, a meta
//! layer of portfolios, an inbox for LLM-authored `.pol` files, and a
//! resumable generation loop. Nothing is ever deleted: `retired` is a status.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use obrrt_gp::problems::Problem;
use obrrt_gp::rollout::{Roll, par};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sm64_binds::SM64Game;

use crate::dsl::{self, C, Pol};
use crate::eval::{Env, Instance, Played, Resolver, play_policy};
use crate::evo::{self, Gen};

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Entry {
    pub id: String,
    /// rule_ga | koza | llm | hand | portfolio | lambda | seed
    pub origin: String,
    pub parents: Vec<String>,
    pub note: String,
    /// active | retired
    pub status: String,
    pub round: u64,
    pub niche: String,
    pub fit: f64,
    pub val_fit: f64,
    pub succ: usize,
    pub frames_done: f64,
    pub per_start: Vec<f64>,
    /// Training starts on which it is the best policy in the inventory.
    pub best_on: Vec<usize>,
    pub usage: u64,
    pub credit: f64,
    pub size: usize,
    pub retired_reason: String,
    /// Operator that produced it.
    pub op: String,
}

/// The bound arguments of a policy: `(meta bind "p0=1.5 p1=-2 seed=7")`.
pub fn instance_of(p: &Pol) -> Instance {
    let mut i = Instance::default();
    for (k, v) in &p.meta {
        if k == "bind" {
            for kv in v.split_whitespace() {
                if let Some((a, b)) = kv.split_once('=') {
                    if a == "seed" {
                        i.seed = b.parse().unwrap_or(0);
                    } else if let (Some(n), Ok(x)) = (a.strip_prefix('p').and_then(|n| n.parse::<usize>().ok()), b.parse::<f32>()) {
                        if n < 8 {
                            i.args[n] = x;
                        }
                    }
                }
            }
        }
    }
    i
}

/// Samplers declared by `(meta sample "p0 uniform lo hi")`, `normal mu sd`, `choice a b c`.
pub fn draw_instance(p: &Pol, r: &mut SmallRng) -> Instance {
    let mut i = instance_of(p);
    for (k, v) in &p.meta {
        if k != "sample" {
            continue;
        }
        let t: Vec<&str> = v.split_whitespace().collect();
        let Some(n) = t.first().and_then(|a| a.strip_prefix('p')).and_then(|n| n.parse::<usize>().ok()).filter(|n| *n < 8) else { continue };
        let f = |j: usize| t.get(j).and_then(|x| x.parse::<f32>().ok()).unwrap_or(0.0);
        i.args[n] = match t.get(1).copied() {
            Some("uniform") => r.random_range(f(2).min(f(3))..=f(2).max(f(3))),
            Some("normal") => f(2) + f(3) * (r.random::<f32>() + r.random::<f32>() + r.random::<f32>() - 1.5) * 2.0,
            Some("choice") if t.len() > 2 => f(2 + r.random_range(0..t.len() - 2)),
            _ => i.args[n],
        };
    }
    i.seed = r.random();
    i
}

pub fn bind_text(i: &Instance) -> String {
    let mut s: Vec<String> = i.args.iter().enumerate().filter(|(_, x)| **x != 0.0).map(|(k, x)| format!("p{k}={x:.4}")).collect();
    s.push(format!("seed={}", i.seed));
    s.join(" ")
}

pub struct Eval<'a> {
    pub pb: &'a Problem,
    pub env: &'a Env,
    pub train: Vec<usize>,
    pub val: Vec<usize>,
    pub test: Vec<usize>,
    pub emu: std::sync::atomic::AtomicU64,
}

struct Bodies<'a>(&'a HashMap<String, Pol>);
impl Resolver for Bodies<'_> {
    fn body(&self, name: &str) -> Option<(&C, Instance)> {
        self.0.get(name).map(|p| (&p.body, instance_of(p)))
    }
}

impl<'a> Eval<'a> {
    pub fn new(pb: &'a Problem, env: &'a Env) -> Eval<'a> {
        let idx = |f: &dyn Fn(&obrrt_gp::rollout::Case) -> bool| (0..pb.cases.len()).filter(|&i| f(&pb.cases[i])).collect::<Vec<_>>();
        Eval {
            pb,
            env,
            train: idx(&|c| c.train),
            val: idx(&|c| c.val),
            test: idx(&|c| !c.train && !c.val),
            emu: Default::default(),
        }
    }

    pub fn run(&self, games: &mut [SM64Game], bodies: &HashMap<String, Pol>, body: &C, idx: &[usize], inst: &Instance, keep: bool) -> Result<Vec<Played>> {
        let res = Bodies(bodies);
        let out = par(games, idx.len(), |g, i| {
            let c = &self.pb.cases[idx[i]];
            play_policy(g, c, self.pb.tasks[c.task].as_ref(), body, &res, self.env, &self.pb.roll, keep, inst)
        })?;
        self.emu.fetch_add(out.iter().map(|p| p.out.frames).sum(), std::sync::atomic::Ordering::Relaxed);
        Ok(out)
    }
}

/// Niche key from a policy's plays: how much of its time is in the air / sliding, and its mean speed.
pub fn niche_of(ps: &[Played]) -> String {
    let n = ps.len().max(1) as f64;
    let tot = |f: &dyn Fn(&Played) -> f64| ps.iter().map(f).sum::<f64>() / n;
    let fr = |p: &Played| p.desc.frames.max(1) as f64;
    let air = tot(&|p| p.desc.air as f64 / fr(p));
    let slide = tot(&|p| (p.desc.slide + p.desc.land) as f64 / fr(p));
    let fv = tot(&|p| p.desc.fv_sum / fr(p));
    let b4 = |x: f64, cuts: [f64; 3]| cuts.iter().filter(|c| x > **c).count();
    format!("a{}s{}v{}", b4(air, [0.15, 0.35, 0.6]), b4(slide, [0.1, 0.3, 0.55]), b4(fv, [30.0, 50.0, 70.0]))
}

pub struct Inventory {
    pub dir: PathBuf,
    pub entries: Vec<Entry>,
    pub bodies: HashMap<String, Pol>,
    /// Best fitness per training start and who holds it.
    pub best_start: Vec<(f64, String)>,
    /// Adaptive operator weights (meta-evolved) and origin priors.
    pub op_w: BTreeMap<String, f64>,
    pub origin_prior: BTreeMap<String, f64>,
    pub round: u64,
    pub counter: u64,
}

const OPS: [&str; 10] = ["random", "mutate", "crossover", "rule-cross", "graft", "chain", "portfolio", "seed-mut", "gate-mut", "tune"];

impl Inventory {
    pub fn open(dir: &Path) -> Result<Inventory> {
        std::fs::create_dir_all(dir.join("incoming/done"))?;
        std::fs::create_dir_all(dir.join("archive"))?;
        let mut inv = Inventory {
            dir: dir.into(),
            entries: vec![],
            bodies: HashMap::new(),
            best_start: vec![],
            op_w: OPS.iter().map(|o| (o.to_string(), 1.0)).collect(),
            origin_prior: [("llm", 2.0), ("hand", 1.5), ("koza", 1.0), ("rule_ga", 1.0), ("portfolio", 1.0), ("lambda", 1.0), ("seed", 1.0)].iter().map(|(a, b)| (a.to_string(), *b)).collect(),
            round: 0,
            counter: 0,
        };
        let ix = dir.join("index.json");
        if ix.exists() {
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&ix)?)?;
            inv.entries = serde_json::from_value(v["entries"].clone())?;
            if let Some(o) = v["op_w"].as_object() {
                for (k, x) in o {
                    inv.op_w.insert(k.clone(), x.as_f64().unwrap_or(1.0));
                }
            }
            inv.round = v["round"].as_u64().unwrap_or(0);
            inv.counter = v["counter"].as_u64().unwrap_or(0);
            inv.best_start = serde_json::from_value(v["best_start"].clone()).unwrap_or_default();
        }
        for e in &inv.entries {
            if let Ok(t) = std::fs::read_to_string(dir.join(format!("{}.pol", e.id))) {
                if let Ok(p) = dsl::parse(&t) {
                    inv.bodies.insert(e.id.clone(), p);
                }
            }
        }
        Ok(inv)
    }

    pub fn save(&self) -> Result<()> {
        let v = serde_json::json!({"entries": self.entries, "op_w": self.op_w, "round": self.round, "counter": self.counter, "best_start": self.best_start});
        let tmp = self.dir.join("index.json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&v)?)?;
        std::fs::rename(tmp, self.dir.join("index.json"))?;
        Ok(())
    }

    /// Append-only log of everything that happens.
    pub fn log(&self, kind: &str, id: &str, note: &str) {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join("changelog.jsonl")) {
            let _ = writeln!(f, "{}", serde_json::json!({"round": self.round, "kind": kind, "id": id, "note": note}));
        }
    }

    pub fn fresh_id(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{prefix}{:05}", self.counter)
    }

    pub fn get(&self, id: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.id == id)
    }

    fn active(&self) -> Vec<usize> {
        (0..self.entries.len()).filter(|&i| self.entries[i].status == "active").collect()
    }

    /// Market share of each active entry: origin prior x (credit + base) x niche rarity, with a floor for new / rarely used ones.
    pub fn shares(&self) -> Vec<(usize, f64)> {
        let act = self.active();
        let mut per_niche: HashMap<&str, usize> = HashMap::new();
        for &i in &act {
            *per_niche.entry(self.entries[i].niche.as_str()).or_default() += 1;
        }
        let mut w: Vec<(usize, f64)> = act
            .iter()
            .map(|&i| {
                let e = &self.entries[i];
                let prior = self.origin_prior.get(&e.origin).copied().unwrap_or(1.0);
                let rarity = 1.0 / (per_niche[e.niche.as_str()] as f64).sqrt();
                let explore = 1.0 / (1.0 + e.usage as f64 / 20.0); // UCB-like bonus for rarely used
                let bonus = if e.best_on.is_empty() { 1.0 } else { 2.0 }; // specialists keep a share
                (i, prior * (0.2 + e.credit) * rarity * bonus + 0.5 * explore)
            })
            .collect();
        let tot: f64 = w.iter().map(|x| x.1).sum::<f64>().max(1e-9);
        let floor = 0.5 / act.len().max(1) as f64; // every active entry keeps at least half a uniform share
        for x in w.iter_mut() {
            x.1 = (x.1 / tot).max(floor);
        }
        let t2: f64 = w.iter().map(|x| x.1).sum();
        let mut w: Vec<(usize, f64)> = w.iter().map(|&(i, x)| (i, x / t2)).collect();
        // LLM-authored policies are first-class: together they keep at least a quarter of the market (when any are active).
        let llm: f64 = w.iter().filter(|(i, _)| self.entries[*i].origin == "llm").map(|x| x.1).sum();
        if llm > 0.0 && llm < 0.25 {
            let (up, down) = (0.25 / llm, 0.75 / (1.0 - llm));
            for x in w.iter_mut() {
                x.1 *= if self.entries[x.0].origin == "llm" { up } else { down };
            }
        }
        w
    }

    pub fn draw(&mut self, r: &mut SmallRng) -> Option<usize> {
        let sh = self.shares();
        if sh.is_empty() {
            return None;
        }
        let mut u: f64 = r.random();
        for (i, p) in &sh {
            u -= p;
            if u <= 0.0 {
                self.entries[*i].usage += 1;
                return Some(*i);
            }
        }
        let i = sh.last().unwrap().0;
        self.entries[i].usage += 1;
        Some(i)
    }

    fn pick_op(&self, r: &mut SmallRng) -> String {
        let tot: f64 = self.op_w.values().sum();
        let mut u = r.random::<f64>() * tot;
        for (k, w) in &self.op_w {
            u -= w;
            if u <= 0.0 {
                return k.clone();
            }
        }
        "mutate".into()
    }

    /// Score a candidate on the training starts (and validation) and admit it if useful.
    /// Returns (admitted?, id, gain) where gain is the credit to share with its parents.
    #[allow(clippy::too_many_arguments)]
    pub fn consider(&mut self, ev: &Eval, games: &mut [SM64Game], pol: Pol, origin: &str, op: &str, parents: &[String], note: &str, force: bool) -> Result<(bool, String, f64)> {
        let inst = instance_of(&pol);
        let train = ev.run(games, &self.bodies, &pol.body, &ev.train, &inst, false)?;
        let per: Vec<f64> = train.iter().map(|p| p.out.fitness).collect();
        let fit = per.iter().sum::<f64>() / per.len().max(1) as f64;
        let succ = train.iter().filter(|p| p.out.done.is_some()).count();
        let fd = if succ > 0 { train.iter().filter_map(|p| p.out.done).sum::<usize>() as f64 / succ as f64 } else { 0.0 };
        let niche = niche_of(&train);
        if self.best_start.len() != per.len() {
            self.best_start = vec![(-1e18, String::new()); per.len()];
        }
        let mut best_on = vec![];
        for (i, &f) in per.iter().enumerate() {
            if f > self.best_start[i].0 + 1e-6 {
                best_on.push(i);
            }
        }
        let elite = self.entries.iter().filter(|e| e.status == "active" && e.niche == niche).map(|e| e.fit).fold(f64::NEG_INFINITY, f64::max);
        let new_niche = elite == f64::NEG_INFINITY;
        let improves_niche = fit > elite + 1e-6;
        let gbest = self.entries.iter().map(|e| e.fit).fold(-1e9, f64::max);
        let admitted = force || new_niche || improves_niche || !best_on.is_empty();
        let gain = if fit > gbest { (fit - gbest).clamp(1.0, 20.0) } else if improves_niche { 0.3 } else if new_niche { 0.5 } else if !best_on.is_empty() { 0.4 } else { 0.0 };
        if !admitted {
            self.log("rejected", &pol.name, &format!("op={op} fit={fit:.1} niche={niche}"));
            return Ok((false, String::new(), 0.0));
        }
        let id = if pol.name.is_empty() || self.bodies.contains_key(&pol.name) { self.fresh_id(match origin { "llm" => "L", "koza" => "K", "rule_ga" => "R", "portfolio" => "P", "hand" => "H", _ => "X" }) } else { pol.name.clone() };
        let val = if ev.val.is_empty() { 0.0 } else {
            let v = ev.run(games, &self.bodies, &pol.body, &ev.val, &inst, false)?;
            v.iter().map(|p| p.out.fitness).sum::<f64>() / v.len() as f64
        };
        for &i in &best_on {
            self.best_start[i] = (per[i], id.clone());
        }
        let mut pol = pol;
        pol.name = id.clone();
        std::fs::write(self.dir.join(format!("{id}.pol")), dsl::print(&pol))?;
        let e = Entry { id: id.clone(), origin: origin.into(), parents: parents.to_vec(), note: note.into(), status: "active".into(), round: self.round, niche: niche.clone(), fit, val_fit: val, succ, frames_done: fd, per_start: per, best_on, usage: 0, credit: gain, size: pol.body.size(), retired_reason: String::new(), op: op.into() };
        self.log("admit", &id, &format!("origin={origin} op={op} parents={parents:?} fit={fit:.1} succ={succ} niche={niche} note={note}"));
        self.entries.push(e);
        self.bodies.insert(id.clone(), pol);
        // parents share the credit
        for p in parents {
            if let Some(pe) = self.entries.iter_mut().find(|e| &e.id == p) {
                pe.credit = (pe.credit + gain / parents.len().max(1) as f64).min(50.0);
            }
        }
        Ok((true, id, gain))
    }

    /// Retire entries that are dominated (not their niche's elite, best on no start, never a source of credit) -- a status, never a deletion.
    pub fn maintain(&mut self) {
        let mut elite: HashMap<String, f64> = HashMap::new();
        for e in self.entries.iter().filter(|e| e.status == "active") {
            let x = elite.entry(e.niche.clone()).or_insert(f64::NEG_INFINITY);
            *x = x.max(e.fit);
        }
        let bs: Vec<String> = self.best_start.iter().map(|b| b.1.clone()).collect();
        let mut msgs = vec![];
        for e in self.entries.iter_mut() {
            if e.status != "active" || e.origin == "llm" || e.origin == "hand" {
                continue; // authored policies are retired only by their authors
            }
            let is_elite = (e.fit - elite[&e.niche]).abs() < 1e-9;
            let best = bs.contains(&e.id);
            if !is_elite && !best && e.credit < 0.3 && e.usage > 30 {
                e.status = "retired".into();
                e.retired_reason = "dominated in its niche, best on no start, no credit".into();
                msgs.push(e.id.clone());
            }
        }
        for m in msgs {
            self.log("retire", &m, "auto: dominated, never contributing (kept on disk)");
        }
        for e in self.entries.iter_mut() {
            e.credit *= 0.995; // credit decays: share must be re-earned
        }
    }

    pub fn set_status(&mut self, id: &str, status: &str, reason: &str) -> Result<()> {
        let Some(e) = self.entries.iter_mut().find(|e| e.id == id) else { bail!("no policy {id}") };
        e.status = status.into();
        e.retired_reason = reason.into();
        self.log(status, id, reason);
        self.save()
    }

    pub fn gen_cfg(&self, ev: &Eval) -> Gen {
        Gen { skills: ev.env.skills.iter().map(|s| (s.name.clone(), s.range.clone())).collect(), acts: evo::acts_of_interest(), max_depth: 7, max_size: 90 }
    }

    /// Produce one candidate by an operator drawn from the adaptive weights.
    pub fn make(&mut self, g: &Gen, r: &mut SmallRng) -> Option<(Pol, String, String, Vec<String>)> {
        let op = self.pick_op(r);
        let mut body_of = |inv: &mut Inventory, r: &mut SmallRng| -> Option<(String, C)> {
            let i = inv.draw(r)?;
            let id = inv.entries[i].id.clone();
            inv.bodies.get(&id).map(|p| (id, p.body.clone()))
        };
        let (body, origin, parents): (C, &str, Vec<String>) = match op.as_str() {
            "random" => (evo::ramped(g, r), "koza", vec![]),
            "mutate" | "seed-mut" => {
                let (id, b) = body_of(self, r)?;
                (evo::mutate(g, &b, r), self.get(&id).map(|e| if e.origin == "portfolio" { "portfolio" } else if op == "seed-mut" { "rule_ga" } else { "koza" }).unwrap_or("koza"), vec![id])
            }
            "tune" => {
                let (id, b) = body_of(self, r)?;
                let k = r.random_range(1..=3);
                (evo::tune(&b, r, k), "rule_ga", vec![id])
            }
            "chain" => {
                let (id, b) = body_of(self, r)?;
                (evo::chain_edit(g, &b, r), "rule_ga", vec![id])
            }
            "crossover" => {
                let (a, ba) = body_of(self, r)?;
                let (b, bb) = body_of(self, r)?;
                (evo::crossover(g, &ba, &bb, r), "koza", vec![a, b])
            }
            "rule-cross" => {
                let (a, ba) = body_of(self, r)?;
                let (b, bb) = body_of(self, r)?;
                (evo::rule_crossover(g, &ba, &bb, r), "rule_ga", vec![a, b])
            }
            "graft" => {
                let (a, ba) = body_of(self, r)?;
                let (b, bb) = body_of(self, r)?;
                (evo::graft_rule(&ba, &bb, r), "rule_ga", vec![a, b])
            }
            "portfolio" => {
                let k = r.random_range(2..=3);
                let mut ids = vec![];
                for _ in 0..k {
                    let i = self.draw(r)?;
                    let id = self.entries[i].id.clone();
                    if self.entries[i].origin != "portfolio" && !ids.contains(&id) {
                        ids.push(id);
                    }
                }
                if ids.len() < 2 {
                    return None;
                }
                let mut body = C::Switch(ids[ids.len() - 1].clone());
                for id in ids[..ids.len() - 1].iter().rev() {
                    body = C::If(Box::new(evo::rand_b(g, r, 2)), Box::new(C::Switch(id.clone())), Box::new(body));
                }
                (body, "portfolio", ids)
            }
            "gate-mut" => {
                let i = self.draw(r)?;
                let e = &self.entries[i];
                if e.origin != "portfolio" {
                    return None;
                }
                let id = e.id.clone();
                let mut b = self.bodies.get(&id)?.body.clone();
                // new condition on a random gate
                let n = evo::count_c(&b);
                let k = r.random_range(0..n);
                if let Some(C::If(..)) = evo::get_c(&b, k) {
                    if let Some(C::If(_, x, y)) = evo::get_c(&b, k).cloned() {
                        evo::put_c(&mut b, k, C::If(Box::new(evo::rand_b(g, r, 2)), x, y));
                    }
                }
                (b, "portfolio", vec![id])
            }
            _ => return None,
        };
        Some((Pol { name: String::new(), meta: vec![("origin".into(), origin.into()), ("op".into(), op.clone())], body }, origin.into(), op, parents))
    }

    /// Load `incoming/*.pol` (LLM-authored policies), evaluate and admit them. A file that
    /// switches to a policy not admitted yet waits until its dependencies are in (repeated passes).
    pub fn inbox(&mut self, ev: &Eval, games: &mut [SM64Game]) -> Result<usize> {
        let dir = self.dir.join("incoming");
        let mut n = 0;
        loop {
            let mut files: Vec<_> = std::fs::read_dir(&dir)?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "pol")).collect();
            files.sort();
            let mut progress = false;
            for f in files {
                let text = std::fs::read_to_string(&f)?;
                let moved = dir.join("done").join(f.file_name().unwrap());
                match dsl::parse(&text) {
                    Ok(p) => {
                        let mut deps = vec![];
                        p.body.switches(&mut deps);
                        if let Some(d) = deps.iter().find(|d| !self.bodies.contains_key(*d) && **d != p.name) {
                            self.log("inbox-wait", &p.name, &format!("waiting on {d}"));
                            continue;
                        }
                        let meta = |k: &str| p.meta.iter().find(|m| m.0 == k).map(|m| m.1.clone()).unwrap_or_default();
                        let origin = if meta("origin").is_empty() { "llm".to_string() } else { meta("origin") };
                        let parent: Vec<String> = meta("parent").split_whitespace().map(String::from).collect();
                        let note = meta("note");
                        let (ok, id, _) = self.consider(ev, games, p, &origin, "inbox", &parent, &note, true)?;
                        self.log("inbox", &id, &format!("{} admitted={ok}", f.display()));
                        n += 1;
                    }
                    Err(e) => self.log("inbox-error", &f.display().to_string(), &e.to_string()),
                }
                std::fs::rename(&f, moved)?;
                progress = true;
            }
            if !progress {
                break;
            }
        }
        Ok(n)
    }

    /// Market-share table by origin and the top policies, one line.
    pub fn share_line(&self) -> String {
        let sh = self.shares();
        let mut by: BTreeMap<String, f64> = BTreeMap::new();
        for (i, p) in &sh {
            *by.entry(self.entries[*i].origin.clone()).or_default() += p;
        }
        let mut top: Vec<_> = sh.clone();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        format!(
            "share by origin {} | top {}",
            by.iter().map(|(k, v)| format!("{k}={:.2}", v)).collect::<Vec<_>>().join(" "),
            top.iter().take(3).map(|(i, p)| format!("{}:{:.2}", self.entries[*i].id, p)).collect::<Vec<_>>().join(" ")
        )
    }

    /// Draw instances of a template from its declared samplers; report how
    /// sampling compares with the default args, and promote the best draw
    /// into a bound lambda (origin=lambda, parent=template).
    pub fn sample_template(&mut self, ev: &Eval, games: &mut [SM64Game], id: &str, draws: usize, r: &mut SmallRng) -> Result<String> {
        let Some(tp) = self.bodies.get(id).cloned() else { bail!("no policy {id}") };
        let score = |ps: &[Played]| (ps.iter().map(|p| p.out.fitness).sum::<f64>() / ps.len().max(1) as f64, ps.iter().filter(|p| p.out.done.is_some()).count());
        let d0 = score(&ev.run(games, &self.bodies, &tp.body, &ev.train, &instance_of(&tp), false)?);
        let mut res: Vec<(f64, usize, Instance)> = vec![];
        for _ in 0..draws {
            let inst = draw_instance(&tp, r);
            let (f, s) = score(&ev.run(games, &self.bodies, &tp.body, &ev.train, &inst, false)?);
            res.push((f, s, inst));
        }
        res.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let mean = res.iter().map(|x| x.0).sum::<f64>() / res.len().max(1) as f64;
        let better = res.iter().filter(|x| x.0 > d0.0).count();
        let mut msg = format!("template {id}: default fit {:.1} succ {} | {draws} draws: mean {mean:.1} best {:.1} (succ {}) | {better}/{draws} beat the default", d0.0, d0.1, res[0].0, res[0].1);
        if res[0].0 > d0.0 {
            let mut p = tp.clone();
            p.meta.retain(|m| m.0 != "bind" && m.0 != "origin");
            p.meta.push(("origin".into(), "lambda".into()));
            p.meta.push(("bind".into(), bind_text(&res[0].2)));
            p.name = String::new();
            let (ok, nid, _) = self.consider(ev, games, p, "lambda", "bind", &[id.to_string()], "best of sampled draws", true)?;
            msg += &format!(" -> promoted {nid} (admitted {ok})");
        }
        self.log("sample", id, &msg);
        Ok(msg)
    }

    /// Seed with hand-written policies (DSL text) if the inventory is empty.
    pub fn seed_hand(&mut self, ev: &Eval, games: &mut [SM64Game]) -> Result<()> {
        for t in HAND {
            if let Ok(p) = dsl::parse(t) {
                if !self.bodies.contains_key(&p.name) {
                    self.consider(ev, games, p, "hand", "seed", &[], "hand-written seed", true)?;
                }
            }
        }
        self.save()
    }

    /// One generation round of `n` candidates.
    pub fn round(&mut self, ev: &Eval, games: &mut [SM64Game], r: &mut SmallRng, n: usize) -> Result<(usize, usize)> {
        self.round += 1;
        let g = self.gen_cfg(ev);
        let (mut tried, mut kept) = (0, 0);
        let mut stat: HashMap<String, (f64, f64)> = HashMap::new();
        for _ in 0..n {
            let Some((pol, origin, op, parents)) = self.make(&g, r) else { continue };
            tried += 1;
            let (ok, _, gain) = self.consider(ev, games, pol, &origin, &op, &parents, "", false)?;
            let s = stat.entry(op).or_default();
            s.0 += 1.0;
            s.1 += if ok { 1.0 + gain } else { 0.0 };
            kept += ok as usize;
        }
        // meta-evolution of the operator weights: multiplicative weights on admissions per try
        for (op, (t, g)) in stat {
            let rate = g / t.max(1.0);
            let w = self.op_w.entry(op).or_insert(1.0);
            *w = (*w * (0.85 + 0.5 * rate.min(1.0))).clamp(0.05, 50.0);
        }
        // keep the weights relative: renormalise to mean 1 (otherwise they all decay to the floor when admissions are rare)
        let mean = self.op_w.values().sum::<f64>() / self.op_w.len().max(1) as f64;
        for w in self.op_w.values_mut() {
            *w = (*w / mean).clamp(0.05, 50.0);
        }
        self.maintain();
        Ok((tried, kept))
    }
}

pub const HAND: &[&str] = &[
    "(policy hand-steer (meta origin \"hand\") (meta note \"always steer at the target\") (skill steer 0))",
    "(policy hand-kick-chain (meta origin \"hand\") (meta note \"jump kick refresh on landings, else steer\")
      (rules (when (act-in freefall_land freefall_land_stop jump_land long_jump_land) (skill kick))
             (when (act-in dive_slide forward_rollout) (skill steer 0))
             (else (skill steer-a 0))))",
    "(policy hand-slide-kick (meta origin \"hand\") (meta note \"slide kick chain on the ground at speed\")
      (rules (when (and (act-in walking idle) (> fv 28)) (skill slide))
             (when (act-in crouch_slide) (skill slide-kick))
             (when (act-in slide_kick_slide) (cell A full aim target))
             (else (skill steer 0))))",
];
