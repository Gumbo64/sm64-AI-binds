//! Read any game field by name from the emulator's memory.
//!
//! A policy names the fields it uses once ([`FieldSet::parse`]); each frame
//! [`FieldSet::fetch`] reads exactly those, in one pass over the memory slice
//! (a few nanoseconds per field). The names come from `fields.tsv`, generated
//! from the decomp headers and the build's linker map by `tools/gen_fields.py`
//! (`FIELDS.md` lists them all; [`list_fields`] enumerates them for a DSL).
//!
//! Path grammar (`root (step)*`):
//! - `root`: a global (`gGlobalTimer`, `gRandomSeed16`, `gMarioStates`, `gCamera`,
//!   `gLakituState`, `gHudDisplay`, ...), the alias `gMarioState` (= `gMarioStates[0]`),
//!   an object: `obj[SLOT]`, `mario_obj`, `held`, `floor_obj`, or
//!   `nearest(bhv=bhvBobomb,k=0)` / `nearest(cat=enemy,k=1)` (the k-th nearest loaded
//!   object of a behaviour or category, by 3D distance to Mario; `k` defaults to 0).
//! - `.field` or `->field` (a pointer is followed when a field is taken from it, so
//!   `gCamera.mode` works too), `[i]` for array elements (`gMarioStates[0].pos[1]`).
//!
//! A missing value (null pointer, no such object, outside memory) is
//! [`Value::Missing`], `NaN` from [`FieldSet::fetch`].

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{Result, bail};
use obrrt_core::St;

use super::behaviours;

const TSV: &str = include_str!("fields.tsv");
const POOL: usize = obrrt_core::layout::ADDR_gObjectPool as usize;
const OBJ_SIZE: usize = obrrt_core::layout::OBJECT_SIZE;
const POOL_CAP: usize = obrrt_core::layout::OBJECT_POOL_CAPACITY as usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    U8,
    S8,
    U16,
    S16,
    U32,
    S32,
    F32,
    Ptr,
    /// An inline struct (only for globals).
    Struct,
}

impl Ty {
    fn parse(s: &str) -> Ty {
        match s {
            "u8" => Ty::U8,
            "s8" => Ty::S8,
            "u16" => Ty::U16,
            "s16" => Ty::S16,
            "u32" => Ty::U32,
            "s32" => Ty::S32,
            "f32" => Ty::F32,
            "ptr" => Ty::Ptr,
            _ => Ty::Struct,
        }
    }
    pub fn size(self) -> usize {
        match self {
            Ty::U8 | Ty::S8 => 1,
            Ty::U16 | Ty::S16 => 2,
            _ => 4,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Ty::U8 => "u8",
            Ty::S8 => "s8",
            Ty::U16 => "u16",
            Ty::S16 => "s16",
            Ty::U32 => "u32",
            Ty::S32 => "s32",
            Ty::F32 => "f32",
            Ty::Ptr => "ptr",
            Ty::Struct => "struct",
        }
    }
}

#[derive(Clone, Debug)]
pub struct GlobalDef {
    pub name: String,
    pub ty: Ty,
    pub addr: usize,
    pub size: usize,
    pub count: usize,
    /// For pointers and structs: the struct name.
    pub st: String,
}

#[derive(Clone, Debug)]
pub struct FieldDef {
    pub name: String,
    pub ty: Ty,
    pub off: usize,
    pub count: usize,
    pub st: String,
}

pub struct Table {
    pub globals: HashMap<String, GlobalDef>,
    pub structs: HashMap<String, HashMap<String, FieldDef>>,
}

pub fn struct_size(name: &str) -> Option<usize> {
    match name {
        "MarioState" => Some(obrrt_core::layout::MARIO_STATE_SIZE),
        "Object" => Some(OBJ_SIZE),
        "Surface" => Some(obrrt_core::layout::SURFACE_SIZE),
        "SurfaceNode" => Some(obrrt_core::layout::SURFACE_NODE_SIZE),
        "HudDisplay" => Some(14),
        _ => None,
    }
}

pub fn table() -> &'static Table {
    static T: OnceLock<Table> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Table { globals: HashMap::new(), structs: HashMap::new() };
        for line in TSV.lines() {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            match f[0] {
                "G" => {
                    let g = GlobalDef { name: f[1].into(), ty: Ty::parse(f[2]), addr: f[3].parse().unwrap(), size: f[4].parse().unwrap(), count: f[5].parse().unwrap(), st: f.get(6).copied().unwrap_or("").into() };
                    t.globals.insert(g.name.clone(), g);
                }
                "F" => {
                    let d = FieldDef { name: f[2].into(), ty: Ty::parse(f[3]), off: f[4].parse().unwrap(), count: f[5].parse().unwrap(), st: f.get(6).copied().unwrap_or("").into() };
                    t.structs.entry(f[1].into()).or_default().insert(d.name.clone(), d);
                }
                _ => {}
            }
        }
        t
    })
}

/// Every readable name, for a DSL / GP to enumerate: globals, then `Struct.field`
/// (for objects `Object.oAction`; use it as `obj[..].oAction`, `nearest(..).oAction`).
pub fn list_fields(prefix: &str) -> Vec<String> {
    let t = table();
    let mut v: Vec<String> = t.globals.keys().cloned().collect();
    for (s, fs) in &t.structs {
        v.extend(fs.keys().map(|f| format!("{s}.{f}")));
    }
    v.retain(|n| n.starts_with(prefix));
    v.sort();
    v
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    I(i64),
    F(f32),
    Missing,
}

impl Value {
    pub fn f32(self) -> f32 {
        match self {
            Value::I(i) => i as f32,
            Value::F(f) => f,
            Value::Missing => f32::NAN,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelKey {
    Bhv(u32),
    Cat(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Selector {
    key: SelKey,
    k: usize,
}

#[derive(Clone, Copy, Debug)]
enum Root {
    Abs(usize),
    Slot(usize),
    Nearest(usize),
}

#[derive(Clone, Copy, Debug)]
enum Step {
    Add(usize),
    Deref,
}

#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    root: Root,
    steps: Vec<Step>,
    pub ty: Ty,
}

/// What the path currently points at while parsing.
#[derive(Clone)]
enum Cur {
    /// A struct (inline, at the current address).
    Struct(String),
    /// A pointer to a struct (the address holds the pointer).
    PtrTo(String),
    /// An array of struct elements (index step still due).
    StructArr(String, usize),
    /// A scalar field with `count` elements (index optional).
    Scalar(Ty, usize),
}

struct Parser<'a> {
    s: &'a str,
    i: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.s[self.i..].chars().next()
    }
    fn ident(&mut self) -> String {
        let st = self.i;
        while self.peek().is_some_and(|c| c.is_alphanumeric() || c == '_') {
            self.i += 1;
        }
        self.s[st..self.i].to_string()
    }
    fn eat(&mut self, t: &str) -> bool {
        if self.s[self.i..].starts_with(t) {
            self.i += t.len();
            true
        } else {
            false
        }
    }
    fn number(&mut self) -> Option<usize> {
        let st = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
        }
        self.s[st..self.i].parse().ok()
    }
}

fn mario_off(field: &str) -> usize {
    table().structs["MarioState"][field].off
}

/// A set of fields to fetch together.
#[derive(Clone, Debug, Default)]
pub struct FieldSet {
    fields: Vec<Field>,
    sels: Vec<Selector>,
}

impl FieldSet {
    pub fn parse(names: &[&str]) -> Result<FieldSet> {
        let mut fs = FieldSet::default();
        for n in names {
            let f = fs.add(n)?;
            fs.fields.push(f);
        }
        Ok(fs)
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.fields.iter().map(|f| f.name.as_str())
    }
    pub fn field(&self, i: usize) -> &Field {
        &self.fields[i]
    }

    fn selector(&mut self, s: Selector) -> usize {
        if let Some(i) = self.sels.iter().position(|x| *x == s) {
            return i;
        }
        self.sels.push(s);
        self.sels.len() - 1
    }

    fn add(&mut self, path: &str) -> Result<Field> {
        let t = table();
        let orig = path;
        // Aliases.
        let expanded;
        let path = if let Some(rest) = path.strip_prefix("gMarioState") {
            if rest.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
                path
            } else {
                expanded = format!("gMarioStates[0]{rest}");
                &expanded
            }
        } else if let Some(rest) = path.strip_prefix("floor_obj") {
            expanded = format!("gMarioStates[0].floor->object{rest}");
            &expanded
        } else {
            path
        };
        let mut p = Parser { s: path, i: 0 };
        let mut steps: Vec<Step> = Vec::new();
        let (root, mut cur): (Root, Cur);
        if p.eat("obj[") {
            let Some(slot) = p.number() else { bail!("{path}: slot number expected") };
            if !p.eat("]") || slot >= POOL_CAP {
                bail!("{path}: bad slot");
            }
            root = Root::Slot(slot);
            cur = Cur::Struct("Object".into());
        } else if p.eat("nearest(") {
            let (mut key, mut k) = (None, 0usize);
            loop {
                let name = p.ident();
                let val_start;
                if p.eat("=") {
                    val_start = p.i;
                    p.ident();
                    let val = &path[val_start..p.i];
                    match name.as_str() {
                        "bhv" => {
                            let Some(a) = obrrt_core::bhv_addr(val) else { bail!("{path}: unknown behaviour {val}") };
                            key = Some(SelKey::Bhv(a));
                        }
                        "cat" => {
                            let Some(c) = behaviours::category_by_name(val) else { bail!("{path}: unknown category {val}") };
                            key = Some(SelKey::Cat(c));
                        }
                        "k" => k = val.parse()?,
                        _ => bail!("{path}: unknown selector key {name}"),
                    }
                } else if let Ok(n) = name.parse::<usize>() {
                    k = n;
                } else {
                    bail!("{path}: selector syntax");
                }
                if p.eat(",") {
                    continue;
                }
                break;
            }
            if !p.eat(")") {
                bail!("{path}: ')' expected");
            }
            let Some(key) = key else { bail!("{path}: nearest() needs bhv= or cat=") };
            root = Root::Nearest(self.selector(Selector { key, k }));
            cur = Cur::Struct("Object".into());
        } else {
            let name = p.ident();
            match name.as_str() {
                "mario_obj" | "held" => {
                    let f = if name == "held" { "heldObj" } else { "marioObj" };
                    root = Root::Abs(obrrt_core::layout::ADDR_gMarioStates as usize);
                    steps.push(Step::Add(mario_off(f)));
                    steps.push(Step::Deref);
                    cur = Cur::Struct("Object".into());
                }
                _ => {
                    let Some(g) = t.globals.get(&name) else { bail!("{path}: unknown global {name}") };
                    root = Root::Abs(g.addr);
                    cur = match g.ty {
                        Ty::Struct => {
                            if g.count > 1 {
                                Cur::StructArr(g.st.clone(), g.count)
                            } else {
                                Cur::Struct(g.st.clone())
                            }
                        }
                        Ty::Ptr if !g.st.is_empty() && g.st != "void" => Cur::PtrTo(g.st.clone()),
                        ty => Cur::Scalar(ty, g.count),
                    };
                }
            }
        }
        // Steps.
        loop {
            if p.i >= path.len() {
                break;
            }
            if p.eat("[") {
                let Some(i) = p.number() else { bail!("{path}: index expected") };
                if !p.eat("]") {
                    bail!("{path}: ']' expected");
                }
                match &cur {
                    Cur::StructArr(s, n) => {
                        let Some(sz) = struct_size(s) else { bail!("{path}: size of struct {s} unknown") };
                        if i >= *n {
                            bail!("{path}: index {i} out of range {n}");
                        }
                        steps.push(Step::Add(i * sz));
                        cur = Cur::Struct(s.clone());
                    }
                    Cur::Scalar(ty, n) => {
                        if i >= *n {
                            bail!("{path}: index {i} out of range {n}");
                        }
                        steps.push(Step::Add(i * ty.size()));
                        cur = Cur::Scalar(*ty, 1);
                    }
                    Cur::Struct(_) if i == 0 => {}
                    _ => bail!("{path}: [] on a non-array"),
                }
                continue;
            }
            let arrow = p.eat("->");
            if !arrow && !p.eat(".") {
                bail!("{path}: unexpected '{}'", &path[p.i..]);
            }
            let fname = p.ident();
            if fname.is_empty() {
                bail!("{path}: field name expected");
            }
            // `.x/.y/.z` of a 3-vector field (pos.x, normal.y).
            if let Cur::Scalar(ty, n) = &cur
                && *n >= 3
                && matches!(fname.as_str(), "x" | "y" | "z")
            {
                let i = (fname.as_bytes()[0] - b'x') as usize;
                steps.push(Step::Add(i * ty.size()));
                cur = Cur::Scalar(*ty, 1);
                continue;
            }
            // Auto-follow pointers / first array element.
            if let Cur::PtrTo(s) = &cur {
                steps.push(Step::Deref);
                cur = Cur::Struct(s.clone());
            }
            if let Cur::StructArr(s, _) = &cur {
                cur = Cur::Struct(s.clone());
            }
            let Cur::Struct(sname) = &cur else { bail!("{path}: field {fname} of a non-struct") };
            let Some(fd) = t.structs.get(sname).and_then(|m| m.get(&fname)) else { bail!("{path}: struct {sname} has no field {fname}") };
            steps.push(Step::Add(fd.off));
            cur = match fd.ty {
                Ty::Ptr if !fd.st.is_empty() && fd.st != "void" && fd.count == 1 => Cur::PtrTo(fd.st.clone()),
                ty => Cur::Scalar(ty, fd.count),
            };
        }
        let ty = match cur {
            Cur::Scalar(ty, _) => ty,
            Cur::PtrTo(_) => Ty::Ptr,
            _ => bail!("{path}: names a struct, not a value"),
        };
        // Fold consecutive Adds.
        let mut folded: Vec<Step> = Vec::new();
        for s in steps {
            match (folded.last_mut(), s) {
                (Some(Step::Add(a)), Step::Add(b)) => *a += b,
                _ => folded.push(s),
            }
        }
        Ok(Field { name: orig.to_string(), root, steps: folded, ty })
    }

    /// The pool slots of this frame's selectors.
    fn slots(&self, st: &St) -> Vec<Option<usize>> {
        if self.sels.is_empty() {
            return Vec::new();
        }
        let mut best: Vec<Vec<(f32, u16)>> = vec![Vec::new(); self.sels.len()];
        if let Some(objs) = st.objects.as_ref() {
            for o in objs.iter() {
                let d = {
                    let (dx, dy, dz) = (o.pos[0] - st.pos[0], o.pos[1] - st.pos[1], o.pos[2] - st.pos[2]);
                    dx * dx + dy * dy + dz * dz
                };
                let c = behaviours::cat_of(o.bhv);
                for (i, s) in self.sels.iter().enumerate() {
                    let ok = match s.key {
                        SelKey::Bhv(a) => o.bhv == a,
                        SelKey::Cat(k) => c == k,
                    };
                    if !ok {
                        continue;
                    }
                    let v = &mut best[i];
                    if v.len() <= s.k {
                        v.push((d, o.slot));
                        v.sort_by(|a, b| a.0.total_cmp(&b.0));
                    } else if d < v[s.k].0 {
                        v[s.k] = (d, o.slot);
                        v.sort_by(|a, b| a.0.total_cmp(&b.0));
                    }
                }
            }
        }
        best.iter().zip(&self.sels).map(|(v, s)| v.get(s.k).map(|x| x.1 as usize)).collect()
    }

    fn read(f: &Field, m: &[u8], slots: &[Option<usize>]) -> Value {
        let mut a = match f.root {
            Root::Abs(a) => a,
            Root::Slot(s) => POOL + s * OBJ_SIZE,
            Root::Nearest(i) => match slots[i] {
                Some(s) if s < POOL_CAP => POOL + s * OBJ_SIZE,
                _ => return Value::Missing,
            },
        };
        for s in &f.steps {
            match *s {
                Step::Add(o) => a += o,
                Step::Deref => {
                    let Some(b) = m.get(a..a + 4) else { return Value::Missing };
                    a = u32::from_le_bytes(b.try_into().unwrap()) as usize;
                    if a == 0 {
                        return Value::Missing;
                    }
                }
            }
        }
        let n = f.ty.size();
        let Some(b) = m.get(a..a + n) else { return Value::Missing };
        match f.ty {
            Ty::U8 => Value::I(b[0] as i64),
            Ty::S8 => Value::I(b[0] as i8 as i64),
            Ty::U16 => Value::I(u16::from_le_bytes([b[0], b[1]]) as i64),
            Ty::S16 => Value::I(i16::from_le_bytes([b[0], b[1]]) as i64),
            Ty::U32 | Ty::Ptr => Value::I(u32::from_le_bytes(b.try_into().unwrap()) as i64),
            Ty::S32 => Value::I(i32::from_le_bytes(b.try_into().unwrap()) as i64),
            Ty::F32 => Value::F(f32::from_le_bytes(b.try_into().unwrap())),
            Ty::Struct => Value::Missing,
        }
    }

    /// Typed values of every field, in the order parsed.
    pub fn fetch_values(&self, m: &[u8], st: &St, out: &mut Vec<Value>) {
        out.clear();
        let slots = self.slots(st);
        out.extend(self.fields.iter().map(|f| Self::read(f, m, &slots)));
    }

    /// Every field as `f32` (missing = NaN; integers above 2^24 lose low bits).
    pub fn fetch(&self, m: &[u8], st: &St, out: &mut [f32]) {
        let slots = self.slots(st);
        for (o, f) in out.iter_mut().zip(&self.fields) {
            *o = Self::read(f, m, &slots).f32();
        }
    }
}
