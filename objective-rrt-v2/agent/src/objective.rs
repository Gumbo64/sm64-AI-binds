//! Objective selectors, resolved to objective-rrt-v2 objectives at a state.
//!
//! - native (the default and preferred): `{"native": "Star::Collect", "key": "bob", "near": [x,y,z], "within": 800}`:
//!   of what `Registry::generate` offers at the state, the kind's objective
//!   whose key contains `key` (`a|b` alternatives, in order), that may start
//!   there (base criteria, exists), with a finite start ETA; nearest `near`
//!   by its steer target when given, else the lowest ETA. Once resolved, an
//!   objective is kept for its id (objects keep their identity by behaviour
//!   and home).
//! - `{"spec": {...}}`: a spec objective (tools/spec/FORMAT.md).
//! - `{"goto": [x,y,z], "r": 150, "grounded": true}`: shorthand for a spec
//!   with one phase: inside the sphere (and on the ground, controllable).
//! - lambda objectives, Objective(b): `{"lambda": TEMPLATE | "libname", "args": {...}}`.
//!   TEMPLATE is a spec whose strings `"=EXPR"` are expressions and `"$name"`
//!   names an arg; both are evaluated at the base state b, the state where the
//!   objective becomes current (again after a rewind). EXPR: numbers, + - * /,
//!   parentheses, sin cos (degrees) sqrt abs min max atan2 (degrees), and b's
//!   x y z yaw (degrees, 0 = +z, 90 = +x) fv vy floor_y stars coins red_coins
//!   health frame level area, plus the args. E.g. 800 units ahead:
//!   `{"lambda": {"zone": "bob", "phases": [{"target": {"point": ["=x+800*sin(yaw)", "$y", "=z+800*cos(yaw)"]}}]}, "args": {"y": "=floor_y"}}`.
//!   A library template lives in agent/lib/NAME.json (`ar lib`).

use anyhow::{Result, anyhow, bail};
use obrrt_core::{Generator, Objective, St};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use serde_json::{Value, json};
use std::sync::Arc;

pub fn validate(sel: &Value) -> Result<()> {
    let n = ["native", "spec", "goto", "lambda"].iter().filter(|k| sel.get(**k).is_some()).count();
    if n != 1 {
        bail!("an objective selector has exactly one of native, spec, goto, lambda: {sel}");
    }
    if sel.get("lambda").is_some() {
        // Bound at a neutral state just to check it parses.
        let b = bind_vars(None);
        let v = lambda_spec(sel, "check", &b)?;
        let mut specs = obrrt_spec::parse_doc(&v.to_string(), "lambda check", None)?;
        specs.pop().ok_or_else(|| anyhow!("lambda: no spec"))?;
        return Ok(());
    }
    if sel.get("spec").is_some() || sel.get("goto").is_some() {
        spec_of(sel, "check")?;
    }
    Ok(())
}

fn spec_of(sel: &Value, id: &str) -> Result<obrrt_spec::Spec> {
    let v = if let Some(g) = sel.get("goto") {
        let r = sel["r"].as_f64().unwrap_or(150.0);
        let mut done = json!({"in_sphere": {"center": g, "r": r}});
        if sel["grounded"].as_bool().unwrap_or(true) {
            done["action_class"] = json!(["stationary", "moving"]);
            done["controllable"] = json!(true);
        }
        let zone = sel.get("zone").cloned().unwrap_or(Value::Null);
        let mut s = json!({"name": format!("Goto_{id}"), "phases": [{"target": {"point": g}, "done": done}]});
        if zone.is_null() {
            bail!("goto needs \"zone\" (e.g. \"bob\", \"castle:1\")");
        }
        s["zone"] = zone;
        s
    } else {
        let mut s = sel["spec"].clone();
        if s.get("name").is_none() {
            s["name"] = json!(format!("Obj_{id}"));
        }
        s
    };
    let mut specs = obrrt_spec::parse_doc(&v.to_string(), &format!("objective {id}"), None)?;
    specs.pop().ok_or_else(|| anyhow!("objective {id}: no spec"))
}

/// Everything the Registry offers at `st` (seeded as legs does).
pub fn offered(st: &St) -> Vec<Arc<dyn Objective>> {
    let mut rng = SmallRng::seed_from_u64(0x5eed);
    obrrt_objectives::Registry.generate(st, &mut rng)
}

fn dist(a: [f32; 3], b: &Value) -> f32 {
    let p = |i: usize| b[i].as_f64().unwrap_or(0.0) as f32;
    ((a[0] - p(0)).powi(2) + (a[1] - p(1)).powi(2) + (a[2] - p(2)).powi(2)).sqrt()
}

/// May `o` be worked on from `st`?
pub fn usable(o: &dyn Objective, st: &St) -> bool {
    o.base_criteria(st) && o.exists_in(st) && obrrt_core::start_eta(o, st).is_finite()
}

/// The objective `sel` names at `st`, if any. `seen`: objectives resolved
/// earlier for this selector (reused where the Registry doesn't offer them).
pub fn resolve(sel: &Value, id: &str, st: &St, seen: &[Arc<dyn Objective>]) -> Result<Option<Arc<dyn Objective>>> {
    if sel.get("lambda").is_some() {
        let v = lambda_spec(sel, id, &bind_vars(Some(st)))?;
        let mut specs = obrrt_spec::parse_doc(&v.to_string(), &format!("objective {id} (lambda)"), None)?;
        return Ok(Some(Arc::new(specs.pop().ok_or_else(|| anyhow!("lambda {id}: no spec"))?)));
    }
    if sel.get("spec").is_some() || sel.get("goto").is_some() {
        return Ok(Some(Arc::new(spec_of(sel, id)?)));
    }
    let kind = sel["native"].as_str().ok_or_else(|| anyhow!("native: a kind name, e.g. \"Star::Collect\""))?;
    let keys: Vec<&str> = sel["key"].as_str().unwrap_or("").split('|').collect();
    let near = sel.get("near").filter(|v| v.is_array());
    let within = sel["within"].as_f64().map(|x| x as f32);
    let pick = |cands: Vec<Arc<dyn Objective>>| -> Option<Arc<dyn Objective>> {
        for k in &keys {
            let mut best: Option<(f32, Arc<dyn Objective>)> = None;
            for o in cands.iter().filter(|o| o.kind() == kind && o.key().contains(k) && usable(o.as_ref(), st)) {
                let score = match near {
                    Some(n) => match o.steer_target(st) {
                        Some(t) => {
                            let d = dist(t, n);
                            if within.is_some_and(|w| d > w) {
                                continue;
                            }
                            d
                        }
                        None => continue,
                    },
                    None => obrrt_core::start_eta(o.as_ref(), st) as f32,
                };
                if best.as_ref().is_none_or(|b| score < b.0) {
                    best = Some((score, o.clone()));
                }
            }
            if let Some(b) = best {
                return Some(b.1);
            }
        }
        None
    };
    if let Some(o) = pick(offered(st)) {
        return Ok(Some(o));
    }
    Ok(pick(seen.to_vec()))
}

/// One line about an objective at a state.
pub fn describe(o: &dyn Objective, st: &St) -> Value {
    json!({
        "kind": o.kind(),
        "key": o.key(),
        "eta": obrrt_core::start_eta(o, st),
        "target": o.steer_target(st),
        "modal": o.modal(),
        "fixed": o.fixed().is_some(),
        "needs_objects": o.needs_objects(),
    })
}


// ---- lambda objectives ----

pub fn lib_dir() -> String {
    format!("{}/agent/lib", crate::util::ROOT)
}

/// The base state's variables (zeros without a state: parse checks).
pub fn bind_vars(st: Option<&St>) -> std::collections::HashMap<String, f64> {
    let mut m = std::collections::HashMap::new();
    if let Some(s) = st {
        let yaw = (s.yaw as u16) as f64 * 360.0 / 65536.0;
        for (k, v) in [
            ("x", s.pos[0] as f64),
            ("y", s.pos[1] as f64),
            ("z", s.pos[2] as f64),
            ("yaw", yaw),
            ("fv", s.forward_vel as f64),
            ("vy", s.vel[1] as f64),
            ("floor_y", s.floor_y as f64),
            ("stars", s.stars as f64),
            ("coins", s.coins as f64),
            ("red_coins", s.red_coins as f64),
            ("health", (s.health >> 8) as f64),
            ("frame", s.frame as f64),
            ("level", s.level as f64),
            ("area", s.area as f64),
        ] {
            m.insert(k.to_string(), v);
        }
    } else {
        for k in ["x", "y", "z", "yaw", "fv", "vy", "floor_y", "stars", "coins", "red_coins", "health", "frame", "level", "area"] {
            m.insert(k.to_string(), 0.0);
        }
    }
    m
}

/// The template bound at b: `"$arg"` and `"=expr"` strings replaced by numbers.
pub fn lambda_spec(sel: &Value, id: &str, vars: &std::collections::HashMap<String, f64>) -> Result<Value> {
    let mut t = match &sel["lambda"] {
        Value::String(name) => {
            let p = format!("{}/{name}.json", lib_dir());
            let v = crate::util::read_json(&p).map_err(|e| anyhow!("lambda {name}: {e:#} (see `ar lib list`)"))?;
            v.get("template").cloned().unwrap_or(v)
        }
        v @ Value::Object(_) => v.clone(),
        _ => bail!("lambda: a spec template object or a library name"),
    };
    let mut vars = vars.clone();
    // Defaults in the template's own "args", then the selector's.
    let mut args = serde_json::Map::new();
    if let Some(o) = t.as_object_mut().and_then(|o| o.remove("args")).and_then(|a| a.as_object().cloned()) {
        args.extend(o);
    }
    if let Some(o) = sel["args"].as_object() {
        args.extend(o.clone());
    }
    for (k, v) in &args {
        let x = match v {
            Value::Number(n) => n.as_f64().unwrap_or(0.0),
            Value::String(s) => eval(s.strip_prefix('=').unwrap_or(s), &vars).map_err(|e| anyhow!("lambda arg {k}: {e}"))?,
            _ => bail!("lambda arg {k}: a number or an expression string"),
        };
        vars.insert(k.clone(), x);
    }
    fn walk(v: &mut Value, vars: &std::collections::HashMap<String, f64>) -> Result<()> {
        match v {
            Value::String(s) => {
                if let Some(e) = s.strip_prefix('=') {
                    *v = json!(eval(e, vars).map_err(|x| anyhow!("{s}: {x}"))?);
                } else if let Some(n) = s.strip_prefix('$').filter(|n| vars.contains_key(*n)) {
                    *v = json!(vars[n]);
                }
            }
            Value::Array(a) => a.iter_mut().try_for_each(|x| walk(x, vars))?,
            Value::Object(o) => o.values_mut().try_for_each(|x| walk(x, vars))?,
            _ => {}
        }
        Ok(())
    }
    walk(&mut t, &vars)?;
    t["name"] = json!(format!("L_{id}"));
    if t.get("zone").is_none() && t.get("zones").is_none() {
        t["zone"] = json!([vars["level"] as u16, vars["area"] as u8]);
    }
    Ok(t)
}

/// A small expression evaluator (lambda args).
pub fn eval(src: &str, vars: &std::collections::HashMap<String, f64>) -> Result<f64, String> {
    struct P<'a> {
        t: Vec<char>,
        i: usize,
        vars: &'a std::collections::HashMap<String, f64>,
    }
    impl P<'_> {
        fn ws(&mut self) {
            while self.i < self.t.len() && self.t[self.i].is_whitespace() {
                self.i += 1;
            }
        }
        fn peek(&mut self) -> Option<char> {
            self.ws();
            self.t.get(self.i).copied()
        }
        fn expr(&mut self) -> Result<f64, String> {
            let mut a = self.term()?;
            while let Some(c @ ('+' | '-')) = self.peek() {
                self.i += 1;
                let b = self.term()?;
                a = if c == '+' { a + b } else { a - b };
            }
            Ok(a)
        }
        fn term(&mut self) -> Result<f64, String> {
            let mut a = self.unary()?;
            while let Some(c @ ('*' | '/')) = self.peek() {
                self.i += 1;
                let b = self.unary()?;
                a = if c == '*' { a * b } else { a / b };
            }
            Ok(a)
        }
        fn unary(&mut self) -> Result<f64, String> {
            if self.peek() == Some('-') {
                self.i += 1;
                return Ok(-self.unary()?);
            }
            self.atom()
        }
        fn atom(&mut self) -> Result<f64, String> {
            match self.peek() {
                Some('(') => {
                    self.i += 1;
                    let v = self.expr()?;
                    if self.peek() != Some(')') {
                        return Err("expected )".into());
                    }
                    self.i += 1;
                    Ok(v)
                }
                Some(c) if c.is_ascii_digit() || c == '.' => {
                    let a = self.i;
                    while self.i < self.t.len() && (self.t[self.i].is_ascii_digit() || self.t[self.i] == '.' || self.t[self.i] == 'e') {
                        self.i += 1;
                    }
                    let s: String = self.t[a..self.i].iter().collect();
                    s.parse().map_err(|_| format!("bad number {s}"))
                }
                Some(c) if c.is_alphabetic() || c == '_' => {
                    let a = self.i;
                    while self.i < self.t.len() && (self.t[self.i].is_alphanumeric() || self.t[self.i] == '_') {
                        self.i += 1;
                    }
                    let name: String = self.t[a..self.i].iter().collect();
                    if self.peek() == Some('(') {
                        self.i += 1;
                        let mut args = vec![self.expr()?];
                        while self.peek() == Some(',') {
                            self.i += 1;
                            args.push(self.expr()?);
                        }
                        if self.peek() != Some(')') {
                            return Err(format!("{name}(: expected )"));
                        }
                        self.i += 1;
                        let r = |d: f64| d.to_radians();
                        return Ok(match (name.as_str(), args.as_slice()) {
                            ("sin", [x]) => r(*x).sin(),
                            ("cos", [x]) => r(*x).cos(),
                            ("sqrt", [x]) => x.sqrt(),
                            ("abs", [x]) => x.abs(),
                            ("min", [x, y]) => x.min(*y),
                            ("max", [x, y]) => x.max(*y),
                            ("atan2", [y, x]) => y.atan2(*x).to_degrees(),
                            _ => return Err(format!("unknown function {name}/{}", args.len())),
                        });
                    }
                    self.vars.get(&name).copied().ok_or_else(|| format!("unknown variable {name}"))
                }
                c => Err(format!("unexpected {c:?}")),
            }
        }
    }
    let mut p = P { t: src.chars().collect(), i: 0, vars };
    let v = p.expr()?;
    if p.peek().is_some() {
        return Err(format!("trailing input at {}", p.i));
    }
    Ok(v)
}
