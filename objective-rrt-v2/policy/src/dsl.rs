//! The policy language: typed s-expression trees (Koza-style GP trees; rule
//! lists are sugar for nested `if`). Text form is stable: `parse(print(t)) == t`.
//!
//! Types: F (float), B (bool), C (a [`Choice`]: skill call, menu cell, switch
//! to another policy, or `if` over choices). See policy/DSL.md.

use anyhow::{Result, bail};

use crate::actions::ACTIONS;

/// Float expression.
#[derive(Clone, Debug, PartialEq)]
pub enum F {
    Const(f32),
    /// A builtin observation (see [`VARS`]).
    Var(u8),
    /// Any named game field (generic Reader, `(field "name")`).
    Field(String),
    /// Field of the k-th nearest object of a category/behaviour: `(obj CAT K "field")`.
    Obj(String, u8, String),
    /// What-if: field of the state N frames ahead if `pad` is held: `(ahead N CELL "field")`.
    Ahead(u8, Box<Cell>, String),
    Add(Box<F>, Box<F>),
    Sub(Box<F>, Box<F>),
    Mul(Box<F>, Box<F>),
    Min(Box<F>, Box<F>),
    Max(Box<F>, Box<F>),
    Abs(Box<F>),
    Sqrt(Box<F>),
    /// atan2(y, x) in degrees.
    Atan2(Box<F>, Box<F>),
    Sin(Box<F>),
    Cos(Box<F>),
    /// Float-valued conditional: `(fif B a b)`.
    If(Box<B>, Box<F>, Box<F>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum B {
    Gt(F, F),
    Lt(F, F),
    And(Vec<B>),
    Or(Vec<B>),
    Not(Box<B>),
    /// Mario's action is one of these (ACT values).
    ActIn(Vec<u32>),
    PrevActIn(Vec<u32>),
    /// Last pad had these buttons down (bit mask A=1 B=2 Z=4).
    Held(u8),
}

/// A menu cell as text: buttons, stick magnitude class, angle class, reference.
#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    pub btn: u8,
    pub mag: u8,
    pub ang: u8,
    pub r: u8,
    pub fine_ang: Box<F>,
    pub fine_mag: Box<F>,
    /// World yaw for ref `world`.
    pub yaw: Box<F>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum C {
    If(Box<B>, Box<C>, Box<C>),
    /// `(skill NAME p...)`: library index by name, params.
    Skill(String, Vec<F>),
    Cell(Cell),
    /// Run another inventory policy this frame.
    Switch(String),
    /// Wrap: a label for the "which rule fired" report; evaluates the inner choice.
    Tag(String, Box<C>),
}

/// Builtin float observations.
pub const VARS: &[&str] = &[
    "fv", "vy", "hsp", "ny", "hgt", "inact", "timer", "down", "health", "dist", "dist1", "yawerr", "pace",
    "wall-f", "wall-l", "wall-r", "wall-b", "ceil", "edge", "ledge", "land-frames", "land-ny", "land-class", "land-zone",
    "in-zone", "ahead-ny", "ahead-class", "ahead-hgt", "holding", "stick-x", "stick-y", "frame", "slope-class", "obj-dist",
    "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "rnd", "wallhit", "walltan", "wp0-dx", "wp0-dz", "wp1-dx", "wp1-dz", "wp2-dx", "wp2-dz", "pos-x", "pos-z", "vel-x", "vel-z", "start-x", "start-y", "start-z", "start-fv", "start-dx", "start-dz", "since-start",
    "dj-land-x", "dj-land-z", "dj-land-y", "dj-land-fv", "dj-land-frames", "dj-land-zone", "kick-land-x", "kick-land-z", "kick-land-y", "kick-land-fv", "kick-land-frames", "kick-land-zone",
    "jump-land-x", "jump-land-z", "jump-land-y", "jump-land-fv", "jump-land-frames", "jump-land-zone", "lj-land-x", "lj-land-z", "lj-land-y", "lj-land-fv", "lj-land-frames", "lj-land-zone", "land-x", "land-z", "land-y", "land-fv",
    "dj-range", "kick-range", "jump-range", "lj-range",
];

pub const BTN: [char; 3] = ['A', 'B', 'Z'];
pub const MAGS: [&str; 3] = ["neutral", "short", "full"];
pub const ANGS: [&str; 6] = ["aim", "left", "right", "side-left", "side-right", "back"];
pub const REFS: [&str; 7] = ["target", "face", "vel", "uphill", "downhill", "cam", "world"];

/// Reader path of "the K-th nearest object of a category or behaviour" field.
/// Object quantities computed from the Sense (not struct fields of the Reader): xz distance, 3D distance, height difference, bearing in degrees from the heading.
pub fn is_derived_obj_field(n: &str) -> bool {
    matches!(n, "dist" | "dist3" | "dy" | "relyaw" | "radius")
}

pub fn obj_path(cat: &str, k: u8, field: &str) -> String {
    let key = if cat.starts_with("bhv") { "bhv" } else { "cat" };
    format!("nearest({key}={cat},k={k}).{field}")
}

pub fn action_id(name: &str) -> Option<u32> {
    ACTIONS.iter().find(|a| a.0 == name).map(|a| a.1)
}

pub fn action_name(v: u32) -> String {
    ACTIONS.iter().find(|a| a.1 == v).map_or(format!("0x{v:08X}"), |a| a.0.to_string())
}

// ------------------------------------------------------------------ sizes

impl F {
    pub fn size(&self) -> usize {
        match self {
            F::Add(a, b) | F::Sub(a, b) | F::Mul(a, b) | F::Min(a, b) | F::Max(a, b) => 1 + a.size() + b.size(),
            F::Abs(a) | F::Sqrt(a) | F::Sin(a) | F::Cos(a) => 1 + a.size(),
            F::Atan2(a, b) => 1 + a.size() + b.size(),
            F::If(c, a, b) => 1 + c.size() + a.size() + b.size(),
            F::Ahead(_, c, _) => 1 + c.size(),
            _ => 1,
        }
    }
}
impl Cell {
    pub fn size(&self) -> usize {
        1 + self.fine_ang.size() + self.fine_mag.size() + self.yaw.size()
    }
}
impl B {
    pub fn size(&self) -> usize {
        match self {
            B::Gt(a, b) | B::Lt(a, b) => 1 + a.size() + b.size(),
            B::And(v) | B::Or(v) => 1 + v.iter().map(B::size).sum::<usize>(),
            B::Not(a) => 1 + a.size(),
            _ => 1,
        }
    }
}
impl C {
    pub fn size(&self) -> usize {
        match self {
            C::If(b, x, y) => 1 + b.size() + x.size() + y.size(),
            C::Skill(_, p) => 1 + p.iter().map(F::size).sum::<usize>(),
            C::Cell(c) => c.size(),
            C::Switch(_) => 1,
            C::Tag(_, c) => c.size(),
        }
    }
    pub fn depth(&self) -> usize {
        match self {
            C::If(_, x, y) => 1 + x.depth().max(y.depth()),
            C::Tag(_, c) => c.depth(),
            _ => 1,
        }
    }
    /// Names of the policies this one switches to.
    pub fn switches(&self, out: &mut Vec<String>) {
        match self {
            C::If(_, x, y) => {
                x.switches(out);
                y.switches(out)
            }
            C::Switch(n) => out.push(n.clone()),
            C::Tag(_, c) => c.switches(out),
            _ => {}
        }
    }
    /// Named fields and object fields this policy reads (collected once at load).
    pub fn fields(&self, out: &mut Vec<String>) {
        fn f(e: &F, out: &mut Vec<String>) {
            match e {
                F::Field(n) => out.push(n.clone()),
                F::Obj(c, k, n) => {
                if !is_derived_obj_field(n) {
                    out.push(obj_path(c, *k, n))
                }
            }
                F::Ahead(_, c, n) => {
                    out.push(format!("ahead:{n}"));
                    f(&c.fine_ang, out);
                }
                F::Add(a, b) | F::Sub(a, b) | F::Mul(a, b) | F::Min(a, b) | F::Max(a, b) => {
                    f(a, out);
                    f(b, out)
                }
                F::Abs(a) | F::Sqrt(a) | F::Sin(a) | F::Cos(a) => f(a, out),
                F::Atan2(x, y) => {
                    f(x, out);
                    f(y, out)
                }
                F::If(c, x, y) => {
                    b(c, out);
                    f(x, out);
                    f(y, out)
                }
                _ => {}
            }
        }
        fn b(e: &B, out: &mut Vec<String>) {
            match e {
                B::Gt(x, y) | B::Lt(x, y) => {
                    f(x, out);
                    f(y, out)
                }
                B::And(v) | B::Or(v) => v.iter().for_each(|x| b(x, out)),
                B::Not(x) => b(x, out),
                _ => {}
            }
        }
        match self {
            C::If(c, x, y) => {
                b(c, out);
                x.fields(out);
                y.fields(out)
            }
            C::Skill(_, p) => p.iter().for_each(|e| f(e, out)),
            C::Cell(c) => {
                f(&c.fine_ang, out);
                f(&c.fine_mag, out);
                f(&c.yaw, out)
            }
            C::Tag(_, c) => c.fields(out),
            C::Switch(_) => {}
        }
    }
}

// ------------------------------------------------------------------ print

fn num(x: f32) -> String {
    if x == x.trunc() && x.abs() < 1e7 { format!("{}", x as i64) } else { format!("{:.4}", x).trim_end_matches('0').trim_end_matches('.').to_string() }
}

fn pf(e: &F) -> String {
    match e {
        F::Const(x) => num(*x),
        F::Var(i) => VARS[*i as usize].to_string(),
        F::Field(n) => format!("(field \"{n}\")"),
        F::Obj(c, k, n) => format!("(obj {c} {k} \"{n}\")"),
        F::Ahead(n, c, f) => format!("(ahead {n} {} \"{f}\")", pcell(c)),
        F::Add(a, b) => format!("(+ {} {})", pf(a), pf(b)),
        F::Sub(a, b) => format!("(- {} {})", pf(a), pf(b)),
        F::Mul(a, b) => format!("(* {} {})", pf(a), pf(b)),
        F::Min(a, b) => format!("(min {} {})", pf(a), pf(b)),
        F::Max(a, b) => format!("(max {} {})", pf(a), pf(b)),
        F::Abs(a) => format!("(abs {})", pf(a)),
        F::Sqrt(a) => format!("(sqrt {})", pf(a)),
        F::Sin(a) => format!("(sin {})", pf(a)),
        F::Cos(a) => format!("(cos {})", pf(a)),
        F::Atan2(a, b) => format!("(atan2 {} {})", pf(a), pf(b)),
        F::If(c, a, b) => format!("(fif {} {} {})", pb(c), pf(a), pf(b)),
    }
}

fn pb(e: &B) -> String {
    match e {
        B::Gt(a, b) => format!("(> {} {})", pf(a), pf(b)),
        B::Lt(a, b) => format!("(< {} {})", pf(a), pf(b)),
        B::And(v) => format!("(and {})", v.iter().map(pb).collect::<Vec<_>>().join(" ")),
        B::Or(v) => format!("(or {})", v.iter().map(pb).collect::<Vec<_>>().join(" ")),
        B::Not(a) => format!("(not {})", pb(a)),
        B::ActIn(v) => format!("(act-in {})", v.iter().map(|x| action_name(*x)).collect::<Vec<_>>().join(" ")),
        B::PrevActIn(v) => format!("(prev-act-in {})", v.iter().map(|x| action_name(*x)).collect::<Vec<_>>().join(" ")),
        B::Held(m) => format!("(held {})", btn_text(*m)),
    }
}

pub fn btn_text(m: u8) -> String {
    if m == 0 {
        return "-".into();
    }
    BTN.iter().enumerate().filter(|(i, _)| m >> i & 1 != 0).map(|(_, c)| *c).collect()
}

fn pcell(c: &Cell) -> String {
    let mut s = format!("(cell {} {} {} {}", btn_text(c.btn), MAGS[c.mag as usize], ANGS[c.ang as usize], REFS[c.r as usize]);
    let zero = |e: &F| *e == F::Const(0.0);
    if !zero(&c.fine_ang) || !zero(&c.fine_mag) || !zero(&c.yaw) {
        s += &format!(" {} {} {}", pf(&c.fine_ang), pf(&c.fine_mag), pf(&c.yaw));
    }
    s + ")"
}

pub fn pc(e: &C, ind: usize) -> String {
    let pad = " ".repeat(ind);
    match e {
        C::If(b, x, y) => format!("(if {}\n{pad}  {}\n{pad}  {})", pb(b), pc(x, ind + 2), pc(y, ind + 2)),
        C::Skill(n, p) => {
            if p.is_empty() {
                format!("(skill {n})")
            } else {
                format!("(skill {n} {})", p.iter().map(pf).collect::<Vec<_>>().join(" "))
            }
        }
        C::Cell(c) => pcell(c),
        C::Switch(n) => format!("(switch {n})"),
        C::Tag(t, c) => format!("(tag {t} {})", pc(c, ind + 2)),
    }
}

// ------------------------------------------------------------------ parse

#[derive(Clone, Debug)]
enum S {
    Atom(String),
    Str(String),
    List(Vec<S>),
}

fn lex(t: &str) -> Result<S> {
    let cs: Vec<char> = t.chars().collect();
    let mut i = 0;
    fn go(cs: &[char], i: &mut usize) -> Result<S> {
        while *i < cs.len() && (cs[*i].is_whitespace() || cs[*i] == ';') {
            if cs[*i] == ';' {
                while *i < cs.len() && cs[*i] != '\n' {
                    *i += 1;
                }
            } else {
                *i += 1;
            }
        }
        if *i >= cs.len() {
            bail!("unexpected end");
        }
        if cs[*i] == '(' {
            *i += 1;
            let mut v = vec![];
            loop {
                while *i < cs.len() && (cs[*i].is_whitespace() || cs[*i] == ';') {
                    if cs[*i] == ';' {
                        while *i < cs.len() && cs[*i] != '\n' {
                            *i += 1;
                        }
                    } else {
                        *i += 1;
                    }
                }
                if *i >= cs.len() {
                    bail!("missing )");
                }
                if cs[*i] == ')' {
                    *i += 1;
                    return Ok(S::List(v));
                }
                v.push(go(cs, i)?);
            }
        }
        if cs[*i] == '"' {
            *i += 1;
            let st = *i;
            while *i < cs.len() && cs[*i] != '"' {
                *i += 1;
            }
            let s: String = cs[st..*i].iter().collect();
            *i += 1;
            return Ok(S::Str(s));
        }
        let st = *i;
        while *i < cs.len() && !cs[*i].is_whitespace() && cs[*i] != '(' && cs[*i] != ')' {
            *i += 1;
        }
        Ok(S::Atom(cs[st..*i].iter().collect()))
    }
    go(&cs, &mut i)
}

fn atom(s: &S) -> Result<&str> {
    match s {
        S::Atom(a) => Ok(a),
        _ => bail!("expected a symbol, got {s:?}"),
    }
}

fn pstr(s: &S) -> Result<String> {
    match s {
        S::Str(a) | S::Atom(a) => Ok(a.clone()),
        _ => bail!("expected a string"),
    }
}

fn idx(list: &[&str], name: &str, what: &str) -> Result<u8> {
    list.iter().position(|x| *x == name).map(|i| i as u8).ok_or_else(|| anyhow::anyhow!("unknown {what} {name}"))
}

fn pf_parse(s: &S) -> Result<F> {
    match s {
        S::Atom(a) => {
            if let Ok(x) = a.parse::<f32>() {
                return Ok(F::Const(x));
            }
            Ok(F::Var(idx(VARS, a, "variable")?))
        }
        S::List(v) if !v.is_empty() => {
            let h = atom(&v[0])?;
            let two = |v: &[S]| -> Result<(Box<F>, Box<F>)> {
                if v.len() != 3 {
                    bail!("({h} a b) takes two arguments");
                }
                Ok((Box::new(pf_parse(&v[1])?), Box::new(pf_parse(&v[2])?)))
            };
            Ok(match h {
                "+" => {
                    let (a, b) = two(v)?;
                    F::Add(a, b)
                }
                "-" => {
                    let (a, b) = two(v)?;
                    F::Sub(a, b)
                }
                "*" => {
                    let (a, b) = two(v)?;
                    F::Mul(a, b)
                }
                "min" => {
                    let (a, b) = two(v)?;
                    F::Min(a, b)
                }
                "max" => {
                    let (a, b) = two(v)?;
                    F::Max(a, b)
                }
                "abs" => F::Abs(Box::new(pf_parse(&v[1])?)),
                "sqrt" => F::Sqrt(Box::new(pf_parse(&v[1])?)),
                "sin" => F::Sin(Box::new(pf_parse(&v[1])?)),
                "cos" => F::Cos(Box::new(pf_parse(&v[1])?)),
                "atan2" => {
                    let (a, b) = two(v)?;
                    F::Atan2(a, b)
                }
                "fif" => F::If(Box::new(pb_parse(&v[1])?), Box::new(pf_parse(&v[2])?), Box::new(pf_parse(&v[3])?)),
                "field" => F::Field(pstr(&v[1])?),
                "obj" => F::Obj(pstr(&v[1])?, atom(&v[2])?.parse()?, pstr(&v[3])?),
                "ahead" => F::Ahead(atom(&v[1])?.parse()?, Box::new(pcell_parse(&v[2])?), pstr(&v[3])?),
                _ => bail!("unknown float function {h}"),
            })
        }
        _ => bail!("bad float expression {s:?}"),
    }
}

fn parse_acts(v: &[S]) -> Result<Vec<u32>> {
    v.iter().map(|a| {
        let n = atom(a)?;
        action_id(n).ok_or_else(|| anyhow::anyhow!("unknown action {n}"))
    }).collect()
}

fn pb_parse(s: &S) -> Result<B> {
    let S::List(v) = s else { bail!("bad bool expression {s:?}") };
    let h = atom(&v[0])?;
    Ok(match h {
        ">" => B::Gt(pf_parse(&v[1])?, pf_parse(&v[2])?),
        "<" => B::Lt(pf_parse(&v[1])?, pf_parse(&v[2])?),
        "and" => B::And(v[1..].iter().map(pb_parse).collect::<Result<_>>()?),
        "or" => B::Or(v[1..].iter().map(pb_parse).collect::<Result<_>>()?),
        "not" => B::Not(Box::new(pb_parse(&v[1])?)),
        "act-in" => B::ActIn(parse_acts(&v[1..])?),
        "prev-act-in" => B::PrevActIn(parse_acts(&v[1..])?),
        "held" => B::Held(parse_btn(atom(&v[1])?)),
        _ => bail!("unknown bool function {h}"),
    })
}

pub fn parse_btn(t: &str) -> u8 {
    if t == "-" {
        return 0;
    }
    BTN.iter().enumerate().filter(|(_, c)| t.contains(**c)).fold(0, |m, (i, _)| m | 1 << i)
}

fn fullmag_default() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("POLICY_FULLMAG").is_ok_and(|v| v == "1"))
}

fn pcell_parse(s: &S) -> Result<Cell> {
    let S::List(v) = s else { bail!("bad cell {s:?}") };
    if atom(&v[0])? != "cell" || v.len() < 5 {
        bail!("(cell BTN MAG ANG REF [fine_ang fine_mag yaw])");
    }
    let z = || Box::new(F::Const(0.0));
    Ok(Cell {
        btn: parse_btn(atom(&v[1])?),
        mag: idx(&MAGS, atom(&v[2])?, "mag")?,
        ang: idx(&ANGS, atom(&v[3])?, "ang")?,
        r: idx(&REFS, atom(&v[4])?, "ref")?,
        fine_ang: if v.len() > 5 { Box::new(pf_parse(&v[5])?) } else { z() },
        fine_mag: if v.len() > 6 {
            Box::new(pf_parse(&v[6])?)
        } else if fullmag_default() && atom(&v[2])? == "full" {
            // POLICY_FULLMAG=1: a Full cell without an explicit fine_mag is stick magnitude 64 (intendedMag 32: +0.15 u/f/frame in the air),
            // not 56.5 (-0.18 u/f/frame): the authors' pre-Reader policies assumed the former.
            Box::new(F::Const(1.0))
        } else {
            z()
        },
        yaw: if v.len() > 7 { Box::new(pf_parse(&v[7])?) } else { z() },
    })
}

fn pc_parse(s: &S) -> Result<C> {
    let S::List(v) = s else { bail!("bad choice {s:?}") };
    let h = atom(&v[0])?;
    Ok(match h {
        "if" => C::If(Box::new(pb_parse(&v[1])?), Box::new(pc_parse(&v[2])?), Box::new(pc_parse(&v[3])?)),
        "skill" => C::Skill(atom(&v[1])?.to_string(), v[2..].iter().map(pf_parse).collect::<Result<_>>()?),
        "cell" => C::Cell(pcell_parse(s)?),
        "switch" => C::Switch(atom(&v[1])?.to_string()),
        "tag" => C::Tag(atom(&v[1])?.to_string(), Box::new(pc_parse(&v[2])?)),
        // Rule-list sugar: (rules (when COND CHOICE)... (else CHOICE))
        "rules" => {
            let mut out: Option<C> = None;
            for r in v[1..].iter().rev() {
                let S::List(rv) = r else { bail!("bad rule") };
                match atom(&rv[0])? {
                    "else" => out = Some(pc_parse(&rv[1])?),
                    "when" => {
                        let tail = out.take().ok_or_else(|| anyhow::anyhow!("rules need a final (else ...)"))?;
                        out = Some(C::If(Box::new(pb_parse(&rv[1])?), Box::new(pc_parse(&rv[2])?), Box::new(tail)));
                    }
                    x => bail!("unknown rule head {x}"),
                }
            }
            out.ok_or_else(|| anyhow::anyhow!("empty rules"))?
        }
        _ => bail!("unknown choice form {h}"),
    })
}

/// A parsed policy file: `(policy NAME (meta key "value")... BODY)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Pol {
    pub name: String,
    pub meta: Vec<(String, String)>,
    pub body: C,
}

pub fn parse(text: &str) -> Result<Pol> {
    let s = lex(text)?;
    let S::List(v) = &s else { bail!("expected (policy ...)") };
    if v.len() < 3 || atom(&v[0])? != "policy" {
        bail!("expected (policy NAME ... BODY)");
    }
    let name = atom(&v[1])?.to_string();
    let mut meta = vec![];
    for m in &v[2..v.len() - 1] {
        if let S::List(mv) = m {
            if mv.len() == 3 && atom(&mv[0])? == "meta" {
                let (k, v) = (atom(&mv[1])?.to_string(), pstr(&mv[2])?);
                if k == "bind" {
                    // whitespace-separated pK=value or seed=N; anything else (e.g. ';' separators) is an error, not zeros
                    for kv in v.split_whitespace() {
                        let ok = kv.split_once('=').is_some_and(|(a, b)| (a == "seed" && b.parse::<u64>().is_ok()) || (a.len() == 2 && a.starts_with('p') && a.as_bytes()[1] < b'8' && a.as_bytes()[1] >= b'0' && b.parse::<f32>().is_ok()));
                        if !ok {
                            bail!("bad bind entry '{kv}': use (meta bind \"p0=1.5 p1=-2 seed=7\"), space separated");
                        }
                    }
                }
                meta.push((k, v));
                continue;
            }
        }
        bail!("only (meta key \"value\") may precede the body");
    }
    Ok(Pol { name, meta, body: pc_parse(&v[v.len() - 1])? })
}

pub fn print(p: &Pol) -> String {
    let mut s = format!("(policy {}\n", p.name);
    for (k, v) in &p.meta {
        s += &format!("  (meta {k} \"{}\")\n", v.replace('"', "'"));
    }
    s + &format!("  {})\n", pc(&p.body, 2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let t = "(policy t (meta origin \"llm\") (rules (when (and (act-in dive_slide) (> fv 60)) (cell A full aim target)) (when (< hgt 30) (skill steer 1 (+ fv 2))) (else (cell - neutral aim face))))";
        let p = parse(t).unwrap();
        let q = parse(&print(&p)).unwrap();
        assert_eq!(p, q);
        assert_eq!(print(&p), print(&q));
    }
}
