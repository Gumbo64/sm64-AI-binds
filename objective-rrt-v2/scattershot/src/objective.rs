//! Objectives and the tree that composes them.
//!
//! A schema file is a JSON tree of objective instances:
//! `{"kind": "Lens", "name": "LobbyBLJ", "doc": "...", "params": {...}, "children": [...]}`.
//! Ids are assigned in pre-order DFS (a node, then its children in order).
//!
//! An objective writes key entries through [`Cx`]; the core stamps each one
//! with the writing objective's id, so the key is `(obj_id, field, value)`.
//! Writing nothing means the objective is inactive. A child is evaluated
//! only when its parent calls [`Cx::child`].

use anyhow::{Result, bail};
use obrrt_core::St;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub type Key = Vec<(u16, u8, i32)>;

pub trait Objective: Send + Sync {
    fn fields(&self) -> &[&'static str];
    fn eval(&self, st: &St, cx: &mut Cx);
    /// Whether `eval` looks at `st.objects` (read once per frame if any does).
    fn needs_objects(&self) -> bool {
        false
    }
    /// The grid of its packed "pos" field, if it has one: written into
    /// objectives.json's params as "grid_log2" for the GUI.
    fn pos_grid(&self) -> Option<u32> {
        None
    }
    /// Whether this objective is a sampling group (a stage of the route): cells
    /// are sampled within their group, and groups compete by their progress.
    fn is_group(&self) -> bool {
        false
    }
    /// For a group: whether `st` is in it.
    fn in_group(&self, _st: &St) -> bool {
        false
    }
    /// Whether `st` only needs time to pass: a door animation, an open dialog
    /// or menu. Rollouts from such a state press through (see `Tree::busy`)
    /// instead of using random inputs.
    fn busy(&self, _st: &St) -> bool {
        false
    }
}

/// One schema node as written in the file.
#[derive(Serialize, Deserialize, Clone)]
pub struct Spec {
    pub kind: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub doc: Option<String>,
    #[serde(default)]
    pub params: Value,
    #[serde(default)]
    pub children: Vec<Spec>,
}

/// One objective instance in the arena; `objectives.json` is a list of these.
#[derive(Serialize)]
pub struct Inst {
    pub id: usize,
    pub kind: String,
    pub name: Option<String>,
    pub doc: Option<String>,
    pub params: Value,
    pub children: Vec<usize>,
    pub fields: Vec<&'static str>,
    /// Ids `id..end` are this instance's subtree.
    #[serde(skip)]
    pub end: usize,
    #[serde(skip)]
    pub obj: Box<dyn Objective>,
}

pub struct Tree {
    pub insts: Vec<Inst>,
    pub objects: bool,
    /// Ids of the group objectives, in route (pre-order) order.
    pub groups: Vec<usize>,
}

impl Tree {
    pub fn load(json: &str) -> Result<Tree> {
        let spec: Spec = serde_json::from_str(json)?;
        let mut insts = Vec::new();
        add(&spec, &mut insts)?;
        let objects = insts.iter().any(|i| i.obj.needs_objects());
        let groups = insts.iter().filter(|i| i.obj.is_group()).map(|i| i.id).collect();
        Ok(Tree { insts, objects, groups })
    }

    /// The group `st` is in: the first group objective that holds it, else 0
    /// (the root).
    pub fn group(&self, st: &St) -> u16 {
        self.groups.iter().find(|&&i| self.insts[i].obj.in_group(st)).map_or(0, |&i| i as u16)
    }

    /// Whether any objective says `st` is busy (a door animation, a dialog, a menu).
    pub fn busy(&self, st: &St) -> bool {
        self.insts.iter().any(|i| i.obj.busy(st))
    }

    /// Route order by objective id: 0 for the root and anything not a group,
    /// 1.. for the groups in order.
    pub fn rank(&self) -> Vec<u16> {
        let mut rank = vec![0; self.insts.len()];
        for (r, &g) in self.groups.iter().enumerate() {
            rank[g] = r as u16 + 1;
        }
        rank
    }

    pub fn key(&self, st: &St) -> Key {
        let mut cx = Cx { insts: &self.insts, st, cur: 0, key: Vec::new(), absorbed: false };
        self.insts[0].obj.eval(st, &mut cx);
        cx.key
    }
}

fn add(spec: &Spec, insts: &mut Vec<Inst>) -> Result<usize> {
    let obj = crate::objectives::build(&spec.kind, &spec.params)?;
    let id = insts.len();
    let mut params = spec.params.clone();
    if let Some(g) = obj.pos_grid() {
        if !params.is_object() {
            params = Value::Object(Default::default());
        }
        params["grid_log2"] = g.into();
    }
    if id > u16::MAX as usize || obj.fields().len() > 256 {
        bail!("too many objectives or fields");
    }
    insts.push(Inst {
        id,
        kind: spec.kind.clone(),
        name: spec.name.clone(),
        doc: spec.doc.clone(),
        params,
        children: Vec::new(),
        fields: obj.fields().to_vec(),
        end: 0,
        obj,
    });
    for c in &spec.children {
        let cid = add(c, insts)?;
        insts[id].children.push(cid);
    }
    insts[id].end = insts.len();
    Ok(id)
}

/// What an objective writes to while it is evaluated.
pub struct Cx<'a> {
    insts: &'a [Inst],
    st: &'a St,
    cur: usize,
    key: Key,
    /// Set by [`Cx::absorb`]: nothing more is written.
    absorbed: bool,
}

impl Cx<'_> {
    /// Writes `field` (an index into `fields()`) of the current objective.
    pub fn put(&mut self, field: usize, v: i32) {
        if !self.absorbed {
            self.key.push((self.cur as u16, field as u8, v));
        }
    }

    /// Makes the whole key a single state: drops every position written so far
    /// (fields named `pos`, the ones that tell states apart by where they are),
    /// writes `field` = `v` of the current objective and ignores everything
    /// after, including the other objectives. What is left is the progress
    /// fields (level, keys, stars), so all such states share one cell.
    pub fn absorb(&mut self, field: usize, v: i32) {
        let insts = self.insts;
        self.key.retain(|&(o, f, _)| !insts[o as usize].fields[f as usize].ends_with("pos"));
        self.put(field, v);
        self.absorbed = true;
    }

    pub fn n_children(&self) -> usize {
        self.insts[self.cur].children.len()
    }

    /// Evaluates the current objective's `i`-th child.
    pub fn child(&mut self, i: usize) {
        if self.absorbed {
            return;
        }
        let insts = self.insts;
        let (parent, st) = (self.cur, self.st);
        self.cur = insts[parent].children[i];
        insts[self.cur].obj.eval(st, self);
        self.cur = parent;
    }

    pub fn all_children(&mut self) {
        for i in 0..self.n_children() {
            self.child(i);
        }
    }
}
