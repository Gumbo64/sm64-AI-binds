//! `Mario::PosTime(p)`: get within a radius of `p` sooner than the search
//! already has: faster routes to places already reached.
//!
//! `OBRRT_POSTIME=near` (default): `p` is sampled near b like PosDist's
//! points (`OBRRT_POSTIME_N` candidates, default 2), and the deadline is the
//! record there: the earliest frame any tree node in b's zone and progress
//! context (keys, stars, collected set with OBRRT_STARSET_KEY) was within
//! [`RADIUS`] of `p` ([`obrrt_core::records`]). Candidates no node has
//! reached are dropped (PosDist covers new places). Starts only from b's
//! context. Key: zone, context, `p`'s 400-unit cell and the deadline's
//! `OBRRT_POSTIME_BUCKET` (10 frames) bucket, so a tighter record makes a
//! new objective.
//! `OBRRT_POSTIME=self` (the old behaviour, and the fallback when the
//! driver keeps no records): `p` = b's position, the deadline b's frame.
//!
//! `OBRRT_PACE=1` (off by default): the deadline also comes from a target
//! pace. From a start `s` at distance `d0` the arrival must be within
//! `d0 / pace` frames of `s` as well as before b's frame. The pace is a rung
//! of a ladder picked per b (by a hash of b, so it's deterministic and
//! stateless): `OBRRT_PACE_SPEED` (35 u/frame, TAS-like; the TAS averages
//! 44 in levels, the tree 16-24), `OBRRT_PACE_SPEED2` (25), or none (the old
//! rule, the fallback). The rung is in the key. A rung that keeps failing
//! decays like any objective, so the pool's weight drifts to the rungs that
//! succeed. With the switch on, PosTime and PosDist also get a dive-rollout
//! steering hint ([`pace_button`], `OBRRT_PACE_HINT=0` turns it off).

use obrrt_core::buttons::{A_BUTTON, B_BUTTON};
use obrrt_core::layout::{ACT_FLAG_AIR, ACT_FLAG_MOVING, ACT_FLAG_SWIMMING};
use obrrt_core::{Eta, Mem, Objective, St, frames_for};

pub const RADIUS: f32 = 150.0;
/// Units per frame no route beats (Mario's top speeds are around 50), so
/// `frame + dist / MAX_SPEED` is an optimistic arrival time.
const MAX_SPEED: f32 = 60.0;
/// How much sooner than `b` counts: any frame, the same "strictly sooner"
/// the duplicate check uses (PosDist has no margin either). It was 30, a
/// whole second, which made every smaller real gain count for nothing; each
/// success makes a new, harder PosTime (the deadline is in the key), so
/// small steps chain.
const MARGIN: u32 = 1;

/// The margin, overridable with OBRRT_POSTIME_MARGIN (frames) for experiments.
fn margin() -> u32 {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *M.get_or_init(|| obrrt_core::sys::env::var("OBRRT_POSTIME_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(MARGIN))
}

fn env_f32(name: &str, default: f32) -> f32 {
    obrrt_core::sys::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// `OBRRT_POSTIME=self`: the old PosTime (b's own position and frame).
pub fn postime_self() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| obrrt_core::sys::env::var("OBRRT_POSTIME").is_ok_and(|v| v == "self"))
}

/// `OBRRT_POSTIME_N`: candidate points per b (default 2).
pub fn postime_n() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| obrrt_core::sys::env::var("OBRRT_POSTIME_N").ok().and_then(|v| v.parse().ok()).unwrap_or(2))
}

/// `OBRRT_POSTIME_BUCKET`: frames per deadline bucket in the key (default 10).
fn bucket() -> u32 {
    static B: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *B.get_or_init(|| obrrt_core::sys::env::var("OBRRT_POSTIME_BUCKET").ok().and_then(|v| v.parse().ok()).unwrap_or(10u32).max(1))
}

/// `OBRRT_PACE=1`: travel objectives push for speed (see the module doc).
pub(super) fn pace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| obrrt_core::sys::env::var("OBRRT_PACE").is_ok_and(|v| v == "1"))
}

/// The pace ladder (u/frame): `OBRRT_PACE_SPEED`, `OBRRT_PACE_SPEED2`.
fn ladder() -> [f32; 2] {
    static L: std::sync::OnceLock<[f32; 2]> = std::sync::OnceLock::new();
    *L.get_or_init(|| [env_f32("OBRRT_PACE_SPEED", 35.0), env_f32("OBRRT_PACE_SPEED2", 25.0)])
}

/// Steering hint for travel under the pace switch: dive at running speed
/// (B on the ground needs `forwardVel >= 29` and the stick past 48, in the
/// air `> 28`), roll out of the dive slide with A. It only replaces the
/// controller's default "A while the target is higher" when it presses
/// something, so jumps up still happen; the GA's genomes are untouched
/// (its own B/Z/A mutations find long jumps and jump kicks).
pub(super) fn pace_button(s: &St, target: [f32; 3]) -> u16 {
    static HINT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !pace_on() || !*HINT.get_or_init(|| obrrt_core::sys::env::var("OBRRT_PACE_HINT").map_or(true, |v| v != "0")) {
        return 0;
    }
    const ACT_DIVE: u32 = 0x0188088A;
    const ACT_DIVE_SLIDE: u32 = 0x00880456;
    const ACT_FORWARD_ROLLOUT: u32 = 0x010008A6;
    if s.action & ACT_FLAG_SWIMMING != 0 || target[1] > s.pos[1] + 50.0 || s.hdist(target) < 300.0 {
        return 0;
    }
    let air = s.action & ACT_FLAG_AIR != 0;
    let moving = s.action & ACT_FLAG_MOVING != 0;
    match s.action {
        ACT_DIVE_SLIDE | ACT_FORWARD_ROLLOUT => A_BUTTON,
        ACT_DIVE => 0,
        _ if air && s.forward_vel > 28.0 => B_BUTTON,
        _ if moving && !air && s.forward_vel >= 29.0 => B_BUTTON,
        _ => 0,
    }
}

pub struct PosTime {
    p: [f32; 3],
    zone: (u16, u8),
    /// Arrive before this frame.
    deadline: u32,
    /// Pace rung (u/frame) under `OBRRT_PACE=1`: also arrive within
    /// `d0 / pace` frames of the start. 0 = none.
    pace: f32,
    /// Near b (`OBRRT_POSTIME=near`): the record's context; starts only
    /// from it. None: the old PosTime at b.
    ctx: Option<obrrt_core::records::Ctx>,
    /// The pool weight (1 for the drawn ones; a position goal's is `OBRRT_POSGOAL_WEIGHT`).
    gw: f64,
    /// Made for a position goal (`PosTime::goal`): its key says so, so the goal's attempts can be told from the rest.
    goal: bool,
}

/// A stateless, deterministic pace rung per (b, p) under `OBRRT_PACE=1`:
/// 35, 25 or none.
fn rung(b: &St, p: [f32; 3]) -> f32 {
    if !pace_on() {
        return 0.0;
    }
    let h = (b.frame as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(p[0].to_bits() as u64 ^ ((p[2].to_bits() as u64) << 32))
        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
    match (h >> 40) % 3 {
        0 => ladder()[0],
        1 => ladder()[1],
        _ => 0.0,
    }
}

impl PosTime {
    /// The old PosTime: b's position, before b's frame.
    pub fn new(b: &St) -> PosTime {
        PosTime { p: b.pos, zone: b.zone(), deadline: b.frame.saturating_sub(margin()), pace: rung(b, b.pos), ctx: None, gw: 1.0, goal: false }
    }

    /// `p` near b, before `record` (the earliest frame a node in b's zone
    /// and context `ctx` was within [`RADIUS`] of `p`).
    pub fn near(b: &St, p: [f32; 3], record: u32, ctx: obrrt_core::records::Ctx) -> PosTime {
        PosTime { p, zone: b.zone(), deadline: record.saturating_sub(margin()), pace: rung(b, p), ctx: Some(ctx), gw: 1.0, goal: false }
    }

    /// Candidate `p` near b: a PosTime if some node of b's zone and context
    /// has been within [`RADIUS`] of it (the records hook).
    pub fn sample(b: &St, p: [f32; 3]) -> Option<PosTime> {
        let c = obrrt_core::records::ctx(b);
        obrrt_core::records::earliest_near(b.zone(), c, p, RADIUS).map(|f| PosTime::near(b, p, f, c))
    }

    /// A position goal's PosTime (`obrrt_core::posgoal`) from `b`'s zone: before the tree's record within
    /// [`RADIUS`] of `p` (in b's zone and progress context), or, where no node of that context has been
    /// there yet, before an ETA bound from b (twice the walking frames plus 90). Pool weight
    /// `OBRRT_POSGOAL_WEIGHT` (default 10), so the goal gets a share of the draws.
    pub fn goal(b: &St, p: [f32; 3]) -> PosTime {
        let c = obrrt_core::records::ctx(b);
        let record = obrrt_core::records::earliest_near(b.zone(), c, p, RADIUS).unwrap_or_else(|| b.frame + (2.0 * frames_for(b.dist(p))) as u32 + 90);
        let mut o = PosTime::near(b, p, record, c);
        o.gw = obrrt_core::posgoal::weight();
        o.goal = true;
        o
    }

    /// Arrive before this frame, for an attempt from `start`: the earlier of
    /// b's deadline and the pace one.
    fn deadline_from(&self, start: &St) -> u32 {
        if self.pace > 0.0 {
            let by = start.frame + (start.dist(self.p) / self.pace).ceil() as u32;
            self.deadline.min(by)
        } else {
            self.deadline
        }
    }
}

impl Objective for PosTime {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): exploration: a state every 30 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 30, air: false }
    }
    fn kind(&self) -> &'static str {
        // The user's position goal is a kind of its own (its own mean weight and learned multiplier in the kind draw).
        if self.goal { "Mario::UserGoal" } else { "Mario::PosTime" }
    }
    fn key(&self) -> String {
        let c = |v: f32| (v / 400.0).round() as i32;
        // The deadline (in frames) is part of it: reaching a place sooner
        // makes a new, harder PosTime for it instead of the stale one.
        let k = match self.ctx {
            None => format!("PosTime {:?} {},{},{} by {}", self.zone, c(self.p[0]), c(self.p[1]), c(self.p[2]), self.deadline),
            Some(x) => format!(
                "PosTime {:?} k{}s{}c{:x} {},{},{} by~{}",
                self.zone, x.0, x.1, x.2, c(self.p[0]), c(self.p[1]), c(self.p[2]), self.deadline / bucket()
            ),
        };
        let k = if self.goal { format!("{k} goal") } else { k };
        if self.pace > 0.0 { format!("{k} pace {}", self.pace) } else { k }
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        self.gw
    }
    fn base_criteria(&self, s: &St) -> bool {
        let d = s.dist(self.p);
        s.zone() == self.zone
            && d > RADIUS
            && (s.frame as f32 + d / MAX_SPEED) < self.deadline as f32
            && self.ctx.is_none_or(|c| obrrt_core::records::ctx(s) == c)
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        let deadline = self.deadline_from(start);
        if s.zone() != self.zone || s.frame >= deadline {
            return Eta::Lost;
        }
        let d = s.dist(self.p);
        if d <= RADIUS {
            return Eta::Done;
        }
        // Even the optimistic arrival (at MAX_SPEED, to the radius) is late.
        if s.frame as f32 + (d - RADIUS) / MAX_SPEED >= deadline as f32 {
            return Eta::Lost;
        }
        Eta::Frames(frames_for(d))
    }
    fn success_value(&self) -> f64 {
        1.0
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 150 })
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.p)
    }
    fn steer_button(&self, s: &St) -> u16 {
        pace_button(s, self.p)
    }
}
