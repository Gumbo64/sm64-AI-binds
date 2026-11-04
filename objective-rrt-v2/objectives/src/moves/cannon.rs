//! Cannon aiming and firing as GA policies (`OBRRT_MOVE_GA`), plausible
//! only in ACT_IN_CANNON.
//!
//! The flight (act_shot_from_cannon, mario_actions_airborne.c; apply_gravity,
//! mario_step.c) is a parabola on the discrete frame sequence, with two
//! small deviations:
//!
//! - firing (act_in_cannon, mario_actions_automatic.c) moves Mario 120 along
//!   the aim and sets `forwardVel = 100 cos p`, `vel.y = 100 sin p`; the shot
//!   starts on the next frame;
//! - each shot frame moves by the velocity (four quarter steps, each checked
//!   for walls / floors / ceilings), then `vel.y -= 1` (at least -75, a
//!   terminal speed reached only after 75+ frames of fall) and
//!   `forwardVel -= 0.05` (at least 10): a drag of 0.05 a frame, so after n
//!   frames the horizontal distance is `120c + 100c n - 0.025 n(n-1)`.
//! - No input steers the shot. A floor ends it in a dive slide (A or B then
//!   rolls out), a wall in a backward air knockback.
//!
//! So for any point the aim is solved in closed form on that sequence
//! ([`solve`]): yaw straight at it (the flight never turns), pitch the low
//! or high root of `y(n(p)) = dy` (scanned for sign changes, then
//! bisected). The objective's [`Arrival`] picks the aim point:
//! [`Arrival::Pass`] passes through the target (a star, a box, a trigger:
//! hit at any speed); a target to stop at is landed on short by the dive
//! slide's run-out on the floor there ([`slide_out`]), so Mario comes to
//! rest at it. [`predict`] replays a solved aim against the zone's static
//! floors and walls (geom.rs) to reject aims that hit a wall or land short.
//!
//! Policies: `cannon-aim` (low root: the fastest flight), `cannon-lob`
//! (high root: steeper, slower landing), `cannon-edge` (low root onto the
//! near edge of the target's floor, along the line from the cannon, at
//! most 600 short of the target: the shortest flight that still lands on
//! it, for islands and ledges). The cannon's yaw range
//! (±0x4000 around the cannon object's) isn't visible in the state: an aim
//! that doesn't settle fires anyway after [`FIRE_BY`] frames.

use obrrt_core::buttons::A_BUTTON;
use obrrt_core::layout::ACT_IN_CANNON;
use obrrt_core::{Arrival, GamePad, St};

use super::geom;
use super::modes::{angle, pad, wrap};

/// Highest pitch the cannon takes (act_in_cannon).
pub const MAX_PITCH: i32 = 0x38E3;
/// Launch speed and offset.
pub const LAUNCH_V: f32 = 100.0;
pub const LAUNCH_OFF: f32 = 120.0;
/// Fire after this many rollout frames even if not aimed.
pub const FIRE_BY: usize = 240;

/// The game's sins/coss: a 4096-entry table indexed by `angle >> 4`.
pub fn sins(a: i32) -> f32 {
    let i = ((a as u16) >> 4) as f32;
    (i * std::f32::consts::TAU / 4096.0).sin()
}

pub fn coss(a: i32) -> f32 {
    let i = ((a as u16) >> 4) as f32;
    (i * std::f32::consts::TAU / 4096.0).cos()
}

/// Horizontal distance and height (relative to Mario in the cannon) after
/// `n` shot frames at pitch `p` (no collisions).
pub fn at_frame(p: i32, n: f32) -> (f32, f32) {
    let (c, s) = (coss(p), sins(p));
    let h = LAUNCH_OFF * c + LAUNCH_V * c * n - 0.025 * n * (n - 1.0);
    let vy0 = LAUNCH_V * s;
    // vel.y = vy0 - k on frame k (k from 0) until it reaches -75.
    let nt = vy0 + 75.0; // frames until terminal speed
    let y = if n <= nt {
        vy0 * n - n * (n - 1.0) / 2.0
    } else {
        vy0 * nt - nt * (nt - 1.0) / 2.0 - 75.0 * (n - nt)
    };
    (h, LAUNCH_OFF * s + y)
}

/// Shot frames (continuous) to cover horizontal distance `d` at pitch `p`,
/// None if the drag stops it short (it never does within 10000).
pub fn frames_to(p: i32, d: f32) -> Option<f32> {
    let c = coss(p);
    let d = d - LAUNCH_OFF * c;
    if d <= 0.0 {
        return Some(0.0);
    }
    // 0.025 n² - (100c + 0.025) n + d = 0, smaller root.
    let (a, b) = (0.025f32, -(LAUNCH_V * c + 0.025));
    let disc = b * b - 4.0 * a * d;
    (disc >= 0.0).then(|| (-b - disc.sqrt()) / (2.0 * a))
}

/// Height at distance `d` (relative), and the frames to get there.
pub fn height_at(p: i32, d: f32) -> Option<(f32, f32)> {
    let n = frames_to(p, d)?;
    Some((at_frame(p, n).1, n))
}

/// Which root of the aim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Root {
    Low,
    High,
}

/// The pitch whose arc passes through (`d` out, `dy` up) from Mario in the
/// cannon, and its flight frames; the `root` asked for if both exist, else
/// the one there is; None out of range. The low root is the one with fewer
/// frames.
pub fn solve(d: f32, dy: f32, root: Root) -> Option<(i32, f32)> {
    let f = |p: i32| height_at(p, d).map(|(y, _)| y - dy);
    // Sign changes of f over the pitch range.
    let step = 0x40;
    let mut roots: Vec<i32> = Vec::new();
    let mut prev: Option<(i32, f32)> = None;
    let mut p = 0;
    while p <= MAX_PITCH {
        if let Some(v) = f(p) {
            if let Some((p0, v0)) = prev {
                if (v0 <= 0.0) != (v <= 0.0) {
                    let (mut lo, mut hi) = (p0, p);
                    while hi - lo > 1 {
                        let mid = (lo + hi) / 2;
                        let vm = f(mid).unwrap_or(f32::MIN);
                        if (vm <= 0.0) == (v0 <= 0.0) {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    roots.push(if f(lo).unwrap_or(f32::MAX).abs() <= f(hi).unwrap_or(f32::MAX).abs() { lo } else { hi });
                }
            }
            prev = Some((p, v));
        }
        p = if p == MAX_PITCH { MAX_PITCH + 1 } else { (p + step).min(MAX_PITCH) };
    }
    let pick = match root {
        Root::Low => roots.first(),
        Root::High => roots.last(),
    }?;
    Some((*pick, height_at(*pick, d)?.1))
}

/// The dive slide's run-out after landing at horizontal speed `v` on a
/// floor of surface type `kind` (update_sliding: `vel *= 0.92` a frame on
/// default floors, 0.96 slippery, 0.98 very slippery, until below 8), with
/// the stick left alone.
pub fn slide_out(v: f32, kind: i16) -> f32 {
    let loss = match geom::slippery_limit(kind) {
        l if l > 0.98 => 0.98,
        l if l > 0.93 => 0.96,
        _ => 0.92,
    };
    let (mut s, mut d) = (v, 0.0);
    while s > 8.0 && d < 4000.0 {
        s *= loss;
        d += s;
    }
    d
}

/// Where a solved shot ends against the zone's static geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Flight {
    pub frames: usize,
    pub end: [f32; 3],
    /// It ends on a floor (else a wall, or still flying at the cap).
    pub landed: bool,
    /// Horizontal speed at the end.
    pub fv: f32,
}

/// Replays the shot from `st` (in the cannon) at (`pitch`, `yaw`) frame by
/// frame against the zone's floors and walls (geom.rs), at most `cap`
/// frames. None without a grid for the zone.
pub fn predict(st: &St, pitch: i32, yaw: i32, cap: usize) -> Option<Flight> {
    geom::with_grid(st.zone(), |g| {
        let (c, s) = (coss(pitch), sins(pitch));
        let (sy, cy) = (sins(yaw), coss(yaw));
        let mut p = [st.pos[0] + LAUNCH_OFF * c * sy, st.pos[1] + LAUNCH_OFF * s, st.pos[2] + LAUNCH_OFF * c * cy];
        let (mut fv, mut vy) = (LAUNCH_V * c, LAUNCH_V * s);
        for k in 0..cap {
            let v = [fv * sy, vy, fv * cy];
            for _ in 0..4 {
                let q = [p[0] + v[0] / 4.0, p[1] + v[1] / 4.0, p[2] + v[2] / 4.0];
                let len = (v[0] * v[0] + v[2] * v[2]).sqrt() / 4.0;
                if len > 0.1 {
                    let dir = [v[0] / 4.0 / len, v[2] / 4.0 / len];
                    if g.wall_hit([p[0], p[1] + 60.0, p[2]], dir, len + 50.0, 0.3).is_some_and(|w| w <= len + 50.0) {
                        return Flight { frames: k + 1, end: p, landed: false, fv };
                    }
                }
                if let Some(f) = g.floor_below(q[0], q[2], p[1].max(q[1]) + 1.0) {
                    if q[1] <= f.y {
                        return Flight { frames: k + 1, end: [q[0], f.y, q[2]], landed: true, fv };
                    }
                }
                p = q;
            }
            vy = (vy - 1.0).max(-75.0);
            fv = (fv - 0.05).max(10.0);
        }
        Flight { frames: cap, end: p, landed: false, fv }
    })
}

/// The cannon policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Aim,
    Lob,
    Edge,
}

/// The aim (pitch, yaw) for policy `k` from `st` (in the cannon) at
/// `target`.
pub fn aim(k: Kind, st: &St, target: [f32; 3], arrival: Arrival) -> Option<(i32, i32)> {
    let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
    let hd = (dx * dx + dz * dz).sqrt();
    if hd < 200.0 {
        return None;
    }
    let dir = [dx / hd, dz / hd];
    let yaw = angle(dx, dz) as i32;
    let root = if k == Kind::Lob { Root::High } else { Root::Low };
    let floor_at = |d: f32, y: f32| geom::with_grid(st.zone(), |g| g.floor_below(st.pos[0] + dir[0] * d, st.pos[2] + dir[1] * d, y + 150.0)).flatten();
    // The point to pass through / land on (horizontal distance, height).
    let (d, y) = match (k, arrival) {
        (Kind::Edge, _) => {
            // Walk back from the target towards the cannon while there is
            // floor near the target's height: the near edge, 40 in.
            let mut d = hd;
            while d > 200.0 && d > hd - 600.0 && floor_at(d - 25.0, target[1]).is_some_and(|f| (f.y - target[1]).abs() < 120.0) {
                d -= 25.0;
            }
            let y = floor_at(d + 40.0, target[1]).map_or(target[1], |f| f.y);
            (d + 40.0, y)
        }
        (_, Arrival::Pass) => (hd, target[1] + 20.0),
        _ => {
            // Land short by the run-out at the landing speed (iterated once
            // on the solved pitch).
            let floor = floor_at(hd, target[1]);
            let y = floor.map_or(target[1], |f| f.y);
            let kind = floor.map_or(0, |f| f.kind);
            let first = solve(hd, y - st.pos[1], root).map(|(p, _)| p)?;
            let out = slide_out(LAUNCH_V * coss(first), kind).min(0.5 * hd).max(0.0);
            (hd - out, y)
        }
    };
    let (pitch, _) = solve(d, y - st.pos[1], root)?;
    Some((pitch, yaw))
}

/// Policy `k`'s pad in the cannon (None elsewhere): stick towards the aim
/// (`pitch -= 10 stickY`, `yaw -= 10 stickX`), A on even frames once aimed
/// within 0x30 (or after [`FIRE_BY`] frames).
pub fn pad_for(k: Kind, st: &St, target: [f32; 3], arrival: Arrival, frame: usize) -> Option<GamePad> {
    if st.action != ACT_IN_CANNON {
        return None;
    }
    let (pitch, yaw) = aim(k, st, target, arrival)?;
    let pe = pitch - st.pitch as i32;
    let ye = wrap(yaw - st.yaw as i32) as i32;
    let aimed = pe.abs() < 0x30 && ye.abs() < 0x30;
    let fire = (aimed || frame >= FIRE_BY) && frame % 2 == 0;
    // Under 20 the stick can't move it (|cy| < 2 reads as 0).
    let cx = if ye.abs() < 20 { 0.0 } else { (-(ye as f32) / 10.0).clamp(-64.0, 64.0) };
    let cy = if pe.abs() < 20 { 0.0 } else { (-(pe as f32) / 10.0).clamp(-64.0, 64.0) };
    Some(pad(if fire { A_BUTTON } else { 0 }, cx, cy))
}

/// Whether policy `k` is worth simulating: in the cannon, an aim exists,
/// and (with the zone's geometry) its flight doesn't end on a wall well
/// short of the aim point. `cannon-edge` needs a floor near the target
/// (else it is `cannon-aim`).
pub fn plausible(k: Kind, st: &St, target: [f32; 3], arrival: Arrival) -> bool {
    if st.action != ACT_IN_CANNON {
        return false;
    }
    let Some((pitch, yaw)) = aim(k, st, target, arrival) else { return false };
    if k == Kind::Lob && aim(Kind::Aim, st, target, arrival).is_some_and(|a| (a.0 - pitch).abs() < 0x200) {
        return false; // both roots (nearly) the same: the lob adds nothing
    }
    if k == Kind::Edge && geom::floor_y(st.zone(), target[0], target[2], target[1] + 50.0).is_none() {
        return false;
    }
    match predict(st, pitch, yaw, 400) {
        Some(f) => f.landed || st.hdist(f.end) > st.hdist(target) - 300.0,
        None => true,
    }
}
