//! Genetic operators on policy trees: Koza-style generation (ramped
//! half-and-half), subtree crossover and mutation by type, and rule-list
//! operators (insert / delete / reorder / swap) on the `if` chain.

use rand::Rng;
use rand::rngs::SmallRng;

use crate::dsl::{self, B, C, Cell, F, VARS};

/// What the generator may use.
#[derive(Clone)]
pub struct Gen {
    /// (name, nparams) of the callable skills.
    pub skills: Vec<(String, Vec<(f32, f32)>)>,
    /// Actions worth testing for (ACT values).
    pub acts: Vec<u32>,
    pub max_depth: usize,
    pub max_size: usize,
}

const USEFUL_VARS: &[&str] = &[
    "fv", "vy", "hgt", "ny", "inact", "timer", "down", "dist", "yawerr", "wall-f", "edge", "land-frames", "land-ny", "land-zone", "ahead-ny", "pace", "rnd", "wallhit", "walltan",
];

fn rvar(r: &mut SmallRng) -> F {
    let n = USEFUL_VARS[r.random_range(0..USEFUL_VARS.len())];
    F::Var(VARS.iter().position(|v| *v == n).unwrap() as u8)
}

fn rconst(r: &mut SmallRng) -> F {
    F::Const(match r.random_range(0..4) {
        0 => r.random_range(-1.0..1.0f32),
        1 => r.random_range(0.0..100.0f32).round(),
        2 => r.random_range(0.0..10.0f32).round(),
        _ => r.random_range(-180.0..180.0f32).round(),
    })
}

pub fn rand_f(r: &mut SmallRng, d: usize) -> F {
    if d == 0 || r.random_bool(0.4) {
        return if r.random_bool(0.55) { rvar(r) } else { rconst(r) };
    }
    let (a, b) = (Box::new(rand_f(r, d - 1)), Box::new(rand_f(r, d - 1)));
    match r.random_range(0..6) {
        0 => F::Add(a, b),
        1 => F::Sub(a, b),
        2 => F::Mul(a, b),
        3 => F::Min(a, b),
        4 => F::Max(a, b),
        _ => F::Abs(a),
    }
}

pub fn rand_b(g: &Gen, r: &mut SmallRng, d: usize) -> B {
    if d == 0 || r.random_bool(0.5) {
        return match r.random_range(0..5) {
            0 | 1 => B::ActIn((0..r.random_range(1..4)).map(|_| g.acts[r.random_range(0..g.acts.len())]).collect()),
            2 => B::Gt(rvar(r), rconst(r)),
            3 => B::Lt(rvar(r), rconst(r)),
            _ => B::Held(r.random_range(1..8)),
        };
    }
    match r.random_range(0..4) {
        0 => B::And((0..2).map(|_| rand_b(g, r, d - 1)).collect()),
        1 => B::Or((0..2).map(|_| rand_b(g, r, d - 1)).collect()),
        2 => B::Not(Box::new(rand_b(g, r, d - 1))),
        _ => B::Gt(rand_f(r, d - 1), rand_f(r, d - 1)),
    }
}

pub fn rand_cell(r: &mut SmallRng) -> Cell {
    let z = || Box::new(F::Const(0.0));
    Cell {
        btn: r.random_range(0..8),
        mag: [0u8, 1, 2, 2, 2][r.random_range(0..5)],
        ang: [0u8, 0, 0, 1, 2, 3, 4, 5][r.random_range(0..8)],
        r: [0u8, 0, 0, 1, 2, 3, 4, 5][r.random_range(0..8)],
        fine_ang: if r.random_bool(0.3) { Box::new(rand_f(r, 1)) } else { z() },
        fine_mag: z(),
        yaw: if r.random_bool(0.1) { Box::new(F::Const(r.random_range(-32768..32768) as f32)) } else { z() },
    }
}

pub fn rand_leaf(g: &Gen, r: &mut SmallRng) -> C {
    if !g.skills.is_empty() && r.random_bool(0.7) {
        let (n, rg) = &g.skills[r.random_range(0..g.skills.len())];
        let k = r.random_range(0..=rg.len());
        C::Skill(n.clone(), rg[..k].iter().map(|(lo, hi)| F::Const(r.random_range(lo.min(*hi)..=lo.max(*hi)))).collect())
    } else {
        C::Cell(rand_cell(r))
    }
}

/// Ramped half-and-half.
pub fn rand_c(g: &Gen, r: &mut SmallRng, d: usize, full: bool) -> C {
    if d == 0 || (!full && r.random_bool(0.3)) {
        return rand_leaf(g, r);
    }
    C::If(Box::new(rand_b(g, r, 2)), Box::new(rand_c(g, r, d - 1, full)), Box::new(rand_c(g, r, d - 1, full)))
}

pub fn ramped(g: &Gen, r: &mut SmallRng) -> C {
    let d = r.random_range(2..=g.max_depth.min(5));
    let full = r.random_bool(0.5);
    rand_c(g, r, d, full)
}

// ------------------------------------------------------- point access

/// Number of C-typed points (nodes) in a tree.
pub fn count_c(c: &C) -> usize {
    1 + match c {
        C::If(_, x, y) => count_c(x) + count_c(y),
        C::Tag(_, c) => count_c(c),
        _ => 0,
    }
}

pub fn get_c(c: &C, mut i: usize) -> Option<&C> {
    if i == 0 {
        return Some(c);
    }
    i -= 1;
    match c {
        C::If(_, x, y) => {
            let nx = count_c(x);
            if i < nx { get_c(x, i) } else { get_c(y, i - nx) }
        }
        C::Tag(_, c) => get_c(c, i),
        _ => None,
    }
}

pub fn put_c(c: &mut C, mut i: usize, new: C) -> bool {
    if i == 0 {
        *c = new;
        return true;
    }
    i -= 1;
    match c {
        C::If(_, x, y) => {
            let nx = count_c(x);
            if i < nx { put_c(x, i, new) } else { put_c(y, i - nx, new) }
        }
        C::Tag(_, c) => put_c(c, i, new),
        _ => false,
    }
}

/// The B conditions of the `if` nodes, in order.
fn conds(c: &mut C, out: &mut Vec<*mut B>) {
    match c {
        C::If(b, x, y) => {
            out.push(&mut **b as *mut B);
            conds(x, out);
            conds(y, out);
        }
        C::Tag(_, c) => conds(c, out),
        _ => {}
    }
}

fn consts(e: &mut F, out: &mut Vec<*mut F>) {
    match e {
        F::Const(_) => out.push(e as *mut F),
        F::Add(a, b) | F::Sub(a, b) | F::Mul(a, b) | F::Min(a, b) | F::Max(a, b) => {
            consts(a, out);
            consts(b, out)
        }
        F::Abs(a) | F::Sqrt(a) | F::Sin(a) | F::Cos(a) => consts(a, out),
        F::Atan2(a, b) => {
            consts(a, out);
            consts(b, out)
        }
        F::If(_, a, b) => {
            consts(a, out);
            consts(b, out)
        }
        _ => {}
    }
}

fn bconsts(b: &mut B, out: &mut Vec<*mut F>) {
    match b {
        B::Gt(x, y) | B::Lt(x, y) => {
            consts(x, out);
            consts(y, out)
        }
        B::And(v) | B::Or(v) => v.iter_mut().for_each(|x| bconsts(x, out)),
        B::Not(a) => bconsts(a, out),
        _ => {}
    }
}

fn cconsts(c: &mut C, out: &mut Vec<*mut F>) {
    match c {
        C::If(b, x, y) => {
            bconsts(b, out);
            cconsts(x, out);
            cconsts(y, out)
        }
        C::Skill(_, p) => p.iter_mut().for_each(|e| consts(e, out)),
        C::Cell(cl) => {
            consts(&mut cl.fine_ang, out);
            consts(&mut cl.yaw, out)
        }
        C::Tag(_, c) => cconsts(c, out),
        C::Switch(_) => {}
    }
}

// ------------------------------------------------------------ operators

/// Subtree crossover on C points (one point in each parent), depth/size limited.
pub fn crossover(g: &Gen, a: &C, b: &C, r: &mut SmallRng) -> C {
    for _ in 0..8 {
        let mut child = a.clone();
        let (i, j) = (r.random_range(0..count_c(a)), r.random_range(0..count_c(b)));
        let sub = get_c(b, j).unwrap().clone();
        put_c(&mut child, i, sub);
        if child.depth() <= g.max_depth && child.size() <= g.max_size {
            return child;
        }
    }
    a.clone()
}

/// A mutation: subtree replace, constant jitter, condition regen, cell tweak, skill swap, rule-chain edit.
pub fn mutate(g: &Gen, c: &C, r: &mut SmallRng) -> C {
    let mut out = c.clone();
    match r.random_range(0..7) {
        0 => {
            let i = r.random_range(0..count_c(&out));
            let (dd, ff) = (r.random_range(0..3), r.random_bool(0.5));
            let new = rand_c(g, r, dd, ff);
            put_c(&mut out, i, new);
        }
        1 | 2 => {
            let mut cs = vec![];
            cconsts(&mut out, &mut cs);
            if !cs.is_empty() {
                let k = cs[r.random_range(0..cs.len())];
                // SAFETY: pointers into `out`, which is not otherwise touched here.
                if let F::Const(x) = unsafe { &mut *k } {
                    *x = if r.random_bool(0.2) { rconst_val(r) } else { *x * r.random_range(0.8..1.25f32) + r.random_range(-0.1..0.1f32) };
                }
            }
        }
        3 => {
            let mut cs = vec![];
            conds(&mut out, &mut cs);
            if !cs.is_empty() {
                let k = cs[r.random_range(0..cs.len())];
                unsafe { *k = rand_b(g, r, 2) };
            }
        }
        4 => {
            // leaf tweak: a different skill / cell
            let n = count_c(&out);
            let i = r.random_range(0..n);
            if matches!(get_c(&out, i), Some(C::Skill(..) | C::Cell(_))) {
                put_c(&mut out, i, rand_leaf(g, r));
            }
        }
        _ => out = chain_edit(g, &out, r),
    }
    if out.depth() > g.max_depth + 2 || out.size() > g.max_size { c.clone() } else { out }
}

fn rconst_val(r: &mut SmallRng) -> f32 {
    match rconst(r) {
        F::Const(x) => x,
        _ => 0.0,
    }
}

/// View the `if` chain as rules: (cond, choice)* + final else.
pub fn to_rules(c: &C) -> (Vec<(B, C)>, C) {
    let mut rules = vec![];
    let mut cur = c;
    while let C::If(b, x, y) = cur {
        rules.push(((**b).clone(), (**x).clone()));
        cur = y;
    }
    (rules, cur.clone())
}

pub fn from_rules(rules: Vec<(B, C)>, last: C) -> C {
    rules.into_iter().rev().fold(last, |acc, (b, x)| C::If(Box::new(b), Box::new(x), Box::new(acc)))
}

pub fn chain_edit(g: &Gen, c: &C, r: &mut SmallRng) -> C {
    let (mut rules, last) = to_rules(c);
    match r.random_range(0..4) {
        0 => {
            let at = r.random_range(0..=rules.len());
            rules.insert(at, (rand_b(g, r, 2), rand_leaf(g, r)));
        }
        1 if !rules.is_empty() => {
            rules.remove(r.random_range(0..rules.len()));
        }
        2 if rules.len() >= 2 => {
            let (i, j) = (r.random_range(0..rules.len()), r.random_range(0..rules.len()));
            rules.swap(i, j);
        }
        _ if !rules.is_empty() => {
            let i = r.random_range(0..rules.len());
            rules[i].1 = rand_leaf(g, r);
        }
        _ => {}
    }
    from_rules(rules, last)
}

/// Rule-list crossover: a prefix of one parent's rules with a suffix of the other's, plus the other's default.
pub fn rule_crossover(g: &Gen, a: &C, b: &C, r: &mut SmallRng) -> C {
    let (ra, la) = to_rules(a);
    let (rb, lb) = to_rules(b);
    let ca = r.random_range(0..=ra.len());
    let cb = r.random_range(0..=rb.len());
    let mut rules: Vec<(B, C)> = ra[..ca].to_vec();
    rules.extend(rb[cb..].iter().cloned());
    let out = from_rules(rules, if r.random_bool(0.5) { la } else { lb });
    if out.depth() > g.max_depth + 40 || out.size() > g.max_size { a.clone() } else { out }
}

/// A fragment (rule) from `b` inserted into `a` at a random position: rule-fragment reuse across policies.
pub fn graft_rule(a: &C, b: &C, r: &mut SmallRng) -> C {
    let (mut ra, la) = to_rules(a);
    let (rb, _) = to_rules(b);
    if rb.is_empty() {
        return a.clone();
    }
    let f = rb[r.random_range(0..rb.len())].clone();
    ra.insert(r.random_range(0..=ra.len()), f);
    from_rules(ra, la)
}

pub fn acts_of_interest() -> Vec<u32> {
    [
        "walking", "braking", "turning_around", "crouch_slide", "slide_kick", "slide_kick_slide", "dive", "dive_slide", "butt_slide", "stomach_slide",
        "jump", "double_jump", "triple_jump", "long_jump", "wall_kick_air", "jump_kick", "freefall", "forward_rollout", "ground_pound", "jump_land",
        "freefall_land", "double_jump_land", "long_jump_land", "freefall_land_stop", "jump_land_stop", "air_hit_wall", "idle", "side_flip", "backflip",
        "steep_jump", "butt_slide_air", "decelerating", "begin_sliding",
    ]
    .iter()
    .filter_map(|n| dsl::action_id(n))
    .collect()
}

/// Local search step: jitter `n` constants of a tree (parameter tuning of an elite).
pub fn tune(c: &C, r: &mut SmallRng, n: usize) -> C {
    let mut out = c.clone();
    for _ in 0..n {
        let mut cs = vec![];
        cconsts(&mut out, &mut cs);
        if cs.is_empty() {
            break;
        }
        let k = cs[r.random_range(0..cs.len())];
        // SAFETY: pointers into `out`, which is not otherwise touched here.
        if let F::Const(x) = unsafe { &mut *k } {
            let sd = 0.05 + 0.25 * r.random::<f32>();
            *x = *x * (1.0 + sd * r.random_range(-1.0..1.0f32)) + r.random_range(-0.2..0.2f32) * sd * 10.0;
        }
    }
    out
}
