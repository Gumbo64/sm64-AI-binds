//! `Mario::PosTime(p)`: get within a radius of `p` sooner than the search
//! already has: faster routes to places already reached.
//!
//! `p` is sampled near b like PosDist's points ([`POSTIME_N`] candidates), and the deadline is the record
//! there: the earliest frame any tree node in b's zone and progress context (keys, stars, collected set) was
//! within [`RADIUS`] of `p` ([`obrrt_core::records`]). Candidates no node has reached are dropped (PosDist covers
//! new places). Starts only from b's context. Key: zone, context, `p`'s 400-unit cell and the deadline's
//! 10-frame bucket, so a tighter record makes a new objective. Where the driver keeps no records, the fallback is
//! `p` = b's position, the deadline b's frame.

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

/// Candidate points per b.
pub const POSTIME_N: usize = 2;

/// Frames per deadline bucket in the key.
const BUCKET: u32 = 10;

pub struct PosTime {
    p: [f32; 3],
    zone: (u16, u8),
    /// Arrive before this frame.
    deadline: u32,
    /// Near b: the record's context; starts only from it. None: the old PosTime at b.
    ctx: Option<obrrt_core::records::Ctx>,
    /// The pool weight (1 for the drawn ones; a position goal's is `OBRRT_POSGOAL_WEIGHT`).
    gw: f64,
    /// Made for a position goal (`PosTime::goal`): its key says so, so the goal's attempts can be told from the rest.
    goal: bool,
}

impl PosTime {
    /// The old PosTime: b's position, before b's frame.
    pub fn new(b: &St) -> PosTime {
        PosTime { p: b.pos, zone: b.zone(), deadline: b.frame.saturating_sub(MARGIN), ctx: None, gw: 1.0, goal: false }
    }

    /// `p` near b, before `record` (the earliest frame a node in b's zone
    /// and context `ctx` was within [`RADIUS`] of `p`).
    pub fn near(b: &St, p: [f32; 3], record: u32, ctx: obrrt_core::records::Ctx) -> PosTime {
        PosTime { p, zone: b.zone(), deadline: record.saturating_sub(MARGIN), ctx: Some(ctx), gw: 1.0, goal: false }
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

}

impl Objective for PosTime {
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
                self.zone, x.0, x.1, x.2, c(self.p[0]), c(self.p[1]), c(self.p[2]), self.deadline / BUCKET
            ),
        };
        if self.goal { format!("{k} goal") } else { k }
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
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        let deadline = self.deadline;
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
}
