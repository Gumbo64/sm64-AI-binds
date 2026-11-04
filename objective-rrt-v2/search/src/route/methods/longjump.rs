//! `longjump`: re-timing with movement tech. Running tops out around 32
//! units/frame; a long jump goes 48 (`forwardVel * 1.5`, capped at 48:
//! `set_mario_action_airborne`, ACT_LONG_JUMP, mario.c), a dive adds 15
//! (capped at 48, ACT_DIVE, same function). So on long flat stretches the
//! route's running can be replaced by long jumps, dives and jump chains.
//!
//! - **Spans**: windows of the best route (from a reached state, 50 to 240
//!   frames, one zone) weighted by how much of their time looks recoverable:
//!   frames minus the horizontal distance at long-jump speed, scaled down by
//!   climbing. Failed spans lose weight; the weights are rebuilt whenever the
//!   route improves. Half the trials keep every waypoint in the span, half
//!   only the essential ones and the end (a long jump may cut a corner past a
//!   non-essential one). One trial in six re-solves the whole route (like
//!   `full`) with the tech mutations.
//! - **Mutations** (fn pointers, so they only see the stream): find a frame
//!   where the stick has been held hard with no A/B/Z for a few frames
//!   ("running"), and write a tech input there, keeping the stick direction
//!   that was there. Every press clears its button on the frame before, as
//!   the game acts on presses (`INPUT_A_PRESSED` etc.), not on holds.
//!   A family is picked per trial; families that improved the route get
//!   picked more.

use obrrt_core::GamePad;
use obrrt_core::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use rand::Rng;
use rand::rngs::SmallRng;

use crate::route::{Ctx, GaSpec, Method, Mutation, Outcome, Trial};

const ACTION_BUTTONS: u16 = A_BUTTON | B_BUTTON | Z_TRIG;
/// Speed the span weights assume tech reaches (long jump/dive cap is 48).
const TECH_SPEED: f32 = 45.0;
const MIN_SPAN: usize = 50;
const MAX_SPAN: usize = 900;

// ---------------------------------------------------------------------------
// Stream helpers.

fn mag(p: &GamePad) -> f32 {
    ((p.stick_x as f32).powi(2) + (p.stick_y as f32).powi(2)).sqrt()
}

/// Whether the stream looks like running just before `i`: the stick held hard
/// in a steady direction, no A/B/Z, for the last 5 frames.
fn running_at(g: &[GamePad], i: usize) -> bool {
    if i < 5 || i >= g.len() {
        return false;
    }
    let w = &g[i - 5..i];
    if !w.iter().all(|p| mag(p) >= 80.0 && p.button & ACTION_BUTTONS == 0) {
        return false;
    }
    let ang = |p: &GamePad| (p.stick_y as f32).atan2(p.stick_x as f32);
    let mut d = (ang(&w[0]) - ang(&w[4])).abs();
    if d > std::f32::consts::PI {
        d = 2.0 * std::f32::consts::PI - d;
    }
    d < 0.6
}

/// A random running frame, if one is found in a few tries.
fn find_running(g: &[GamePad], rng: &mut SmallRng) -> Option<usize> {
    if g.len() < 12 {
        return None;
    }
    for _ in 0..16 {
        let i = rng.random_range(5..g.len() - 4);
        if running_at(g, i) {
            return Some(i);
        }
    }
    None
}

/// The running stick before `i` (the direction already there).
fn forward(g: &[GamePad], i: usize) -> (i8, i8) {
    let p = g[i.saturating_sub(1)];
    (p.stick_x, p.stick_y)
}

/// Frame `i`: only `buttons` of A/B/Z, stick `st`.
fn put(g: &mut [GamePad], i: usize, buttons: u16, st: (i8, i8)) {
    if let Some(p) = g.get_mut(i) {
        p.button = (p.button & !ACTION_BUTTONS) | buttons;
        p.stick_x = st.0;
        p.stick_y = st.1;
    }
}

/// A press of `b` at `i` (released the frame before), held `hold` frames;
/// the stick is left as it is.
fn press(g: &mut [GamePad], i: usize, b: u16, hold: usize) {
    if i > 0 && i - 1 < g.len() {
        g[i - 1].button &= !b;
    }
    for k in i..(i + hold.max(1)).min(g.len()) {
        g[k].button |= b;
    }
}

fn clear(g: &mut [GamePad], from: usize, to: usize, b: u16) {
    let n = g.len();
    for p in &mut g[from.min(n)..to.min(n)] {
        p.button &= !b;
    }
}

// ---------------------------------------------------------------------------
// Tech inserts. Each returns false if it found no place.

/// Long jump at running frame `a`: Z for 1-2 frames starts a crouch slide
/// (`act_walking` -> `INPUT_Z_PRESSED` -> ACT_CROUCH_SLIDE,
/// mario_actions_moving.c), then A within the slide's first 30 frames gives
/// ACT_LONG_JUMP if forwardVel > 10 (`act_crouch_slide`); the slide loses
/// speed, so A comes 0-2 frames after Z. Stick held forward throughout.
/// Returns the A frame.
fn long_jump_at(g: &mut [GamePad], a: usize, rng: &mut SmallRng) -> usize {
    let st = forward(g, a);
    let z = rng.random_range(1..=2);
    let gap = rng.random_range(0..=2);
    let z_on_a = rng.random_bool(0.5);
    for k in 0..z {
        put(g, a + k, Z_TRIG, st);
    }
    for k in 0..gap {
        put(g, a + z + k, if z_on_a { Z_TRIG } else { 0 }, st);
    }
    let at = a + z + gap;
    put(g, at, A_BUTTON | if z_on_a { Z_TRIG } else { 0 }, st);
    // Holding A in the air changes little for a long jump; vary it.
    let hold = rng.random_range(1..12);
    for k in at + 1..(at + hold).min(g.len()) {
        g[k].button = (g[k].button & !(B_BUTTON | Z_TRIG)) | A_BUTTON;
    }
    at
}

fn long_jump(g: &mut Vec<GamePad>, rng: &mut SmallRng) -> bool {
    let Some(a) = find_running(g, rng) else { return false };
    long_jump_at(g, a, rng);
    true
}

/// Chained long jumps: in `act_long_jump_land` A only counts with Z held
/// (`if (!(m->input & INPUT_Z_DOWN)) m->input &= ~INPUT_A_PRESSED`), and then
/// gives another ACT_LONG_JUMP (sLongJumpLandAction's aPressedAction). The
/// long jump itself doesn't ground-pound on Z (`act_long_jump` has no Z
/// check), so Z is held across the expected landing and A pressed every other
/// frame there to hit the landing frames.
fn long_jump_chain(g: &mut Vec<GamePad>, rng: &mut SmallRng) -> bool {
    let Some(a) = find_running(g, rng) else { return false };
    let mut at = long_jump_at(g, a, rng);
    for _ in 0..rng.random_range(1..=3) {
        let land = at + rng.random_range(16..=32);
        if land + 6 >= g.len() {
            break;
        }
        let st = forward(g, land);
        clear(g, at + 1, land, B_BUTTON);
        for k in land.saturating_sub(rng.random_range(1..=3))..land {
            g[k].button = (g[k].button & !(A_BUTTON | B_BUTTON)) | Z_TRIG;
        }
        let presses = rng.random_range(1..=3);
        for k in 0..6 {
            let a_now = k % 2 == 0 && k / 2 < presses;
            put(g, land + k, Z_TRIG | if a_now { A_BUTTON } else { 0 }, st);
        }
        at = land;
    }
    true
}

/// Dive from running: B with forwardVel >= 29 and the stick held gives
/// ACT_DIVE ("speed kick", `check_ground_dive_or_punch`), +15 speed. Then
/// maybe a rollout: A or B on the dive slide (`act_dive_slide`) gives
/// ACT_FORWARD_ROLLOUT, which keeps the speed and gets back to running.
fn dive(g: &mut Vec<GamePad>, rng: &mut SmallRng) -> bool {
    let Some(a) = find_running(g, rng) else { return false };
    let st = forward(g, a);
    put(g, a, B_BUTTON, st);
    clear(g, a + 1, a + 8, B_BUTTON | Z_TRIG);
    if rng.random_bool(0.7) {
        let r = a + rng.random_range(8..=28);
        let b = if rng.random_bool(0.7) { A_BUTTON } else { B_BUTTON };
        press(g, r, b, rng.random_range(1..=3));
    }
    true
}

/// Jump then dive: A from running (ACT_JUMP), then B in the air gives
/// ACT_DIVE when forwardVel > 28, else a jump kick (`check_kick_or_dive_in_air`,
/// mario_actions_airborne.c); maybe a rollout on the slide after.
fn jump_dive(g: &mut Vec<GamePad>, rng: &mut SmallRng) -> bool {
    let Some(a) = find_running(g, rng) else { return false };
    let st = forward(g, a);
    put(g, a, A_BUTTON, st);
    let hold = rng.random_range(2..=10);
    press(g, a, A_BUTTON, hold);
    clear(g, a, a + 14, Z_TRIG);
    let d = a + rng.random_range(2..=12);
    press(g, d, B_BUTTON, 1);
    clear(g, d + 1, d + 8, B_BUTTON);
    if rng.random_bool(0.6) {
        let r = d + rng.random_range(10..=30);
        press(g, r, A_BUTTON, rng.random_range(1..=3));
    }
    true
}

/// Double/triple jumps: A on landing. Landing sets doubleJumpTimer to 5
/// (sJumpLandAction's second field, `common_landing_action`); A pressed
/// while it runs gives ACT_DOUBLE_JUMP after ACT_JUMP_LAND, and after
/// ACT_DOUBLE_JUMP_LAND an ACT_TRIPLE_JUMP if forwardVel > 20
/// (`set_jump_from_landing`, mario.c). A single jump is about 20-30 frames
/// in the air on flat ground, a double a little more.
fn jump_chain(g: &mut Vec<GamePad>, rng: &mut SmallRng) -> bool {
    let Some(a) = find_running(g, rng) else { return false };
    let st = forward(g, a);
    put(g, a, A_BUTTON, st);
    let mut at = a;
    let mut hold = rng.random_range(3..=12);
    press(g, at, A_BUTTON, hold);
    clear(g, a, a + 4, B_BUTTON | Z_TRIG);
    for j in 0..rng.random_range(1..=2) {
        let next = at + rng.random_range(if j == 0 { 14..=30 } else { 18..=36 });
        if next >= g.len() {
            break;
        }
        clear(g, at + hold, next, A_BUTTON | B_BUTTON | Z_TRIG);
        hold = rng.random_range(4..=18);
        press(g, next, A_BUTTON, hold);
        at = next;
    }
    true
}

/// Moves one existing A/B/Z press (with its hold) 1-3 frames earlier or
/// later: tunes the timing of tech already in the stream.
fn shift_press(g: &mut Vec<GamePad>, rng: &mut SmallRng) -> bool {
    let b = [A_BUTTON, A_BUTTON, B_BUTTON, Z_TRIG][rng.random_range(0..4)];
    let starts: Vec<usize> = (0..g.len()).filter(|&i| g[i].button & b != 0 && (i == 0 || g[i - 1].button & b == 0)).collect();
    if starts.is_empty() {
        return false;
    }
    let s = starts[rng.random_range(0..starts.len())];
    let mut e = s;
    while e < g.len() && g[e].button & b != 0 {
        e += 1;
    }
    let d = rng.random_range(1..=3i64) as isize * if rng.random_bool(0.5) { 1 } else { -1 };
    let ns = (s as isize + d).max(0) as usize;
    let ne = (e as isize + d).max(1) as usize;
    clear(g, s, e, b);
    press(g, ns, b, ne.saturating_sub(ns));
    true
}

// ---------------------------------------------------------------------------
// Mutation families (one per trial). A family falls back to shifting a press
// when it finds no running frame.

fn run_family(g: &mut Vec<GamePad>, rng: &mut SmallRng, fs: &[fn(&mut Vec<GamePad>, &mut SmallRng) -> bool]) {
    if rng.random_bool(0.2) && shift_press(g, rng) {
        return;
    }
    let f = fs[rng.random_range(0..fs.len())];
    if !f(g, rng) {
        shift_press(g, rng);
    }
}

fn mut_lj(g: &mut Vec<GamePad>, rng: &mut SmallRng) {
    run_family(g, rng, &[long_jump, long_jump, long_jump_chain]);
}
fn mut_dive(g: &mut Vec<GamePad>, rng: &mut SmallRng) {
    run_family(g, rng, &[dive, jump_dive]);
}
fn mut_jumps(g: &mut Vec<GamePad>, rng: &mut SmallRng) {
    run_family(g, rng, &[jump_chain, jump_dive]);
}
fn mut_mix(g: &mut Vec<GamePad>, rng: &mut SmallRng) {
    run_family(g, rng, &[long_jump, long_jump, long_jump_chain, dive, jump_dive, jump_chain]);
}

const FAMILIES: [(&str, Mutation); 4] = [("lj", mut_lj), ("dive", mut_dive), ("jumps", mut_jumps), ("mix", mut_mix)];

// ---------------------------------------------------------------------------
// The method.

#[derive(Clone)]
struct Span {
    from: usize,
    /// Reached states after `from` up to the span's end, in order.
    states: Vec<usize>,
    /// Reached states after the span up to the next essential one: kept as
    /// stages so the trial must still line up with the route after it (a
    /// span alone, faster but with other momentum, rarely survives the
    /// splice: the old stream after it no longer reaches the next door).
    tail: Vec<usize>,
    weight: f64,
}

#[derive(Default)]
pub struct LongJump {
    spans: Vec<Span>,
    /// Goal frame the spans were built for.
    built_for: u32,
    /// Per family: improvements.
    wins: [u32; 4],
}

fn build_spans(ctx: &Ctx) -> Vec<Span> {
    let reached = ctx.reached();
    let b = ctx.best;
    let mut out = Vec::new();
    for (ki, &from) in reached.iter().enumerate() {
        let (Some((f0, _)), Some(s0)) = (b.arrival[from], b.states[from].as_ref()) else { continue };
        let zone = s0.zone();
        let mut prev = s0.pos;
        let (mut hd, mut dy) = (0.0f32, 0.0f32);
        let mut states = Vec::new();
        for &j in &reached[ki + 1..] {
            let (Some((fj, _)), Some(sj)) = (b.arrival[j], b.states[j].as_ref()) else { break };
            if sj.zone() != zone || ctx.waypoints[j - 1].modal.is_some() {
                break;
            }
            let frames = fj - f0;
            if frames > MAX_SPAN {
                break;
            }
            hd += ((sj.pos[0] - prev[0]).powi(2) + (sj.pos[2] - prev[2]).powi(2)).sqrt();
            dy += (sj.pos[1] - prev[1]).abs();
            prev = sj.pos;
            states.push(j);
            if frames < MIN_SPAN {
                continue;
            }
            // Recoverable fraction of the span's time, scaled down by climbing.
            let gain = (frames as f32 - hd / TECH_SPEED).max(0.0) / frames as f32;
            let flat = hd / (hd + 4.0 * dy + 1.0);
            // Too slow means waiting/fighting, not running: tech helps less.
            let speed = hd / frames as f32;
            let running = if speed < 8.0 { 0.3 } else { 1.0 };
            let size = (frames as f32 / 200.0).sqrt().min(2.0);
            let weight = (gain * flat * running * size) as f64;
            if weight > 0.01 {
                let mut tail = Vec::new();
                if !ctx.essential(j) {
                    for &t in reached.iter().filter(|&&t| t > j) {
                        tail.push(t);
                        if ctx.essential(t) {
                            break;
                        }
                    }
                }
                out.push(Span { from, states: states.clone(), tail, weight });
            }
        }
    }
    out
}

impl LongJump {
    fn pick_family(&self, rng: &mut SmallRng) -> usize {
        let w: Vec<f64> = self.wins.iter().map(|&n| 1.0 + 2.0 * n as f64).collect();
        let mut x = rng.random_range(0.0..w.iter().sum::<f64>());
        for (i, wi) in w.iter().enumerate() {
            if x < *wi {
                return i;
            }
            x -= wi;
        }
        w.len() - 1
    }
}

impl Method for LongJump {
    fn name(&self) -> &'static str {
        "longjump"
    }

    fn propose(&mut self, ctx: &Ctx, rng: &mut SmallRng) -> Trial {
        if self.built_for != ctx.best.goal_frame {
            self.spans = build_spans(ctx);
            self.built_for = ctx.best.goal_frame;
            if std::env::var("LONGJUMP_DEBUG").is_ok() {
                let mut s = self.spans.clone();
                s.sort_by(|a, b| b.weight.total_cmp(&a.weight));
                for &j in &ctx.reached() {
                    if let (Some(a), Some(st)) = (ctx.best.arrival[j], ctx.best.states[j].as_ref()) {
                        eprintln!("  state {j}: frame {} stream {} zone {:?} pos {:?} vel {:.1} action {:#x} ess {}", a.0, a.1, st.zone(), st.pos, st.forward_vel, st.action, ctx.essential(j));
                    }
                }
                eprintln!("longjump: {} spans; top:", s.len());
                for sp in s.iter().take(10) {
                    eprintln!("  {} -> {} ({} states, tail {:?}) w {:.3}", sp.from, sp.states.last().unwrap(), sp.states.len(), sp.tail, sp.weight);
                }
            }
        }
        let fam = self.pick_family(rng);
        let (fname, mutation) = FAMILIES[fam];
        let total: f64 = self.spans.iter().map(|s| s.weight).sum();
        if self.spans.is_empty() || total <= 0.0 || rng.random_range(0..3) == 0 {
            return Trial {
                from: 0,
                stages: ctx.reached().into_iter().filter(|&i| i > 0).collect(),
                spec: GaSpec::default(),
                seed: None,
                mutation: Some(mutation),
                tag: format!("full {fname} f{fam}"), ..Default::default() };
        }
        let mut x = rng.random_range(0.0..total);
        let mut k = self.spans.len() - 1;
        for (i, s) in self.spans.iter().enumerate() {
            if x < s.weight {
                k = i;
                break;
            }
            x -= s.weight;
        }
        let sp = &self.spans[k];
        let all = rng.random_bool(0.5);
        let last = *sp.states.last().unwrap();
        let stages: Vec<usize> =
            sp.states.iter().copied().filter(|&j| all || j == last || ctx.essential(j)).chain(sp.tail.iter().copied()).collect();
        Trial {
            from: sp.from,
            stages,
            spec: GaSpec { population: 16, generations: 8, horizon: None },
            seed: None,
            mutation: Some(mutation),
            tag: format!("span {}->{} {} {fname} f{fam} k{k}", sp.from, last, if all { "all" } else { "ess" }), ..Default::default() }
    }

    fn feedback(&mut self, o: &Outcome, ctx: &Ctx) {
        let tag = &o.trial.tag;
        let fam = tag.split_whitespace().find_map(|w| w.strip_prefix('f')?.parse::<usize>().ok());
        if o.improved {
            if let Some(f) = fam.filter(|&f| f < FAMILIES.len()) {
                self.wins[f] += 1;
            }
            return; // spans get rebuilt for the new route
        }
        if self.built_for != ctx.best.goal_frame {
            return;
        }
        // A span that didn't help loses weight (less if it at least met all).
        if let Some(k) = tag.split_whitespace().find_map(|w| w.strip_prefix('k')?.parse::<usize>().ok()) {
            if let Some(s) = self.spans.get_mut(k) {
                let faster = o.span_frames.zip(o.best_span_frames).is_some_and(|(a, b)| a < b);
                s.weight *= if faster { 0.9 } else if o.met == o.trial.stages.len() { 0.8 } else { 0.6 };
            }
        }
    }
}
