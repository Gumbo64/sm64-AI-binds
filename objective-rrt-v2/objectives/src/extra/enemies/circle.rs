//! `MrI::Circle(id)` / `MrBlizzard::Circle(id)`: run circles around an enemy
//! that turns to face Mario until it gets dizzy and dies. Neither can be
//! attacked directly (both are pure damage to touch).
//!
//! - Mr. I (`INTERACT_DAMAGE`) turns to follow Mario while he's within 800;
//!   once it has turned a full 0x10000 in one direction without a pause it
//!   dies (mr_i_act_2 -> action 3, mr_i_act_3). The big one (2nd param
//!   byte 1, BBH's secret room) leaves the star named by its 1st param byte;
//!   small ones leave a blue coin (mr_i.inc.c).
//! - Mr. Blizzard (`INTERACT_MR_BLIZZARD`) turns to face Mario on the
//!   ground, building dizziness; past a 0x3000 roll it falls over
//!   (`MR_BLIZZARD_ACT_DEATH` = 6), drops 3 coins, and gives back Mario's cap
//!   if it was wearing it (mr_blizzard_act_rotate / _death,
//!   mr_blizzard.inc.c).

use obrrt_core::{Eta, GaBudget, Mem, Obj, ObjId, Objective, PACE, St, course_of_level, frames_for};

use super::common::{bp1, bp2, capless, find, hdist};

const MR_I_ACT_DYING: i32 = 3;
const MR_BLIZZARD_ACT_DEATH: i32 = 6;

/// Angle to sweep round it in one direction: Mr. I dies past a 0x10000 turn
/// (mr_i_act_2); Mr. Blizzard's dizziness (+8/frame² while Mario is off its
/// facing) passes the 0x3000 roll after ~55 frames of steady circling,
/// about one lap at its radius and PACE (mr_blizzard_act_rotate).
const LAP_ANGLE: f64 = std::f64::consts::TAU;
/// `mem` (v1's progress accumulator, folded per frame): bit 0 = in range
/// last frame; f[0] = signed angle swept since entering range, f[1] = last
/// frame's angle.
const IN_RANGE: u32 = 1;
/// Frames for Mr. I to notice Mario and start turning (mr_i_act_1: he turns round at 256 a frame until he faces him,
/// the leg Mario spends in front of him facing him).
const MR_I_NOTICE: f64 = 30.0;

pub struct Circle {
    id: ObjId,
    zone: (u16, u8),
    kind: &'static str,
    weight: f64,
    star: Option<(i16, u8)>,
    /// Mario must stay this close for the turning to count.
    range: f32,
    radius: f32,
    death_action: i32,
    /// The big Mr. I of BBH's secret room: Mario needs the vanish cap to get to it
    /// ([`super::mansion::SECRET_GATE_WEST`]).
    secret: bool,
}

impl Circle {
    pub fn mr_i(b: &St, o: &Obj) -> Circle {
        let star = (bp2(o) != 0).then(|| (course_of_level(b.level as u32), bp1(o)));
        Circle {
            id: o.id,
            zone: b.zone(),
            kind: "MrI::Circle",
            weight: if star.is_some() { 20.0 } else { 2.0 },
            star,
            range: 800.0,
            radius: 300.0,
            death_action: MR_I_ACT_DYING,
            secret: star.is_some() && super::mansion::in_bbh(b),
        }
    }

    pub fn mr_blizzard(b: &St, o: &Obj) -> Circle {
        // Without its cap Mario takes double damage; Mr. Blizzard may be
        // wearing it (SAVE_FLAG_CAP_ON_MR_BLIZZARD, not readable here).
        let capless = capless(b);
        Circle {
            id: o.id,
            zone: b.zone(),
            kind: "MrBlizzard::Circle",
            weight: if capless { 4.0 } else { 1.5 },
            star: None,
            range: 1000.0,
            radius: 250.0,
            death_action: MR_BLIZZARD_ACT_DEATH,
            secret: false,
        }
    }

    fn dead(&self, o: Option<&Obj>) -> bool {
        o.is_none_or(|o| o.action == self.death_action)
    }

    fn is_mr_i(&self) -> bool {
        self.death_action == MR_I_ACT_DYING
    }

    /// The way to the big Mr. I's chamber for a Mario outside it: the vanish cap first (a vanish `!` box, broken,
    /// then its cap), then the west gate through the vanish-cap walls, then the room.
    fn secret_eta(&self, s: &St, o: &Obj) -> Eta {
        use super::mansion::{SECRET_GATE_EAST, SECRET_GATE_WEST};
        // From the gate to the circling: inside, ~300 further, then a lap (Mr. I turns to him and he faces it).
        let inside = frames_for(obrrt_core::dist(SECRET_GATE_WEST, SECRET_GATE_EAST))
            + frames_for((hdist(SECRET_GATE_EAST, o.pos) - self.radius).max(0.0))
            + self.laps_left(0.0)
            + MR_I_NOTICE;
        Eta::Frames(secret_to_gate(s, MARGIN) + inside)
    }

    /// Frames of circling left at this radius for the angle still to sweep.
    fn laps_left(&self, swept: f64) -> f64 {
        (LAP_ANGLE - swept.abs()).max(0.0) * self.radius as f64 / PACE
    }
}

/// The cap time a capped Mario must have beyond the walk to the gate (factor, frames): the real walk is slower
/// than `gate_frames` (agent mri's tests: ~230 frames across the third floor where the graph says 98).
const MARGIN: (f64, f64) = (1.3, 30.0);
/// The same on the third floor itself, where `gate_frames` is the walk alone (98 frames by the graph, ~230 real):
/// t1 step 881 banked 142 MrI::Climb successes at 1.3x and got 0 of 143 MrI::Gate (the cap ran out on the way).
const MARGIN_3F: (f64, f64) = (2.4, 30.0);

fn margin_at(p: [f32; 3]) -> (f64, f64) {
    if p[1] >= 1700.0 { MARGIN_3F } else { MARGIN }
}

/// Frames from Mario (outside the chamber) to its west gate with the vanish cap: the vanish cap first (a vanish `!` box,
/// broken, then its cap), then the wall-kick column to the third floor and across it (`mansion.rs`).
fn secret_to_gate(s: &St, margin: (f64, f64)) -> f64 {
    use super::mansion::{BOX_FRAMES, MARIO_VANISH_CAP, gate_frames, reach_frames};
    if s.mario_flags & MARIO_VANISH_CAP != 0 {
        // Only while the cap lasts to the gate (`cap_timer`, 600 from the box), with a margin: past it, another
        // cap. (agent misc's "~750-frame walks" were the floor graph's 20000-unit UNREACHABLE penalty: it has
        // no way up to the third floor. `gate_frames` goes by the wall-kick column, ~300 frames from the
        // second-floor box.)
        let g = gate_frames(s, s.pos);
        let margin = if s.pos[1] >= 1700.0 { MARGIN_3F } else { margin };
        if s.cap_timer as f64 >= margin.0 * g + margin.1 {
            return g;
        }
    }
    super::mansion::vanish_sources(s)
        .into_iter()
        .map(|p| reach_frames(s, p) + BOX_FRAMES + gate_frames(s, p))
        .fold(f64::INFINITY, f64::min)
}

/// `MrI::Box(id)` / `MrI::Climb(id)` / `MrI::Gate(id)`: the climb stage of BBH's big Mr. I, so the trees bank it. Done: Mario on the third floor
/// with the vanish cap, still lasting to the gate with a margin (`secret_to_gate`'s rule). The stage before is the
/// box and the wall-kick column (mansion.rs), which the whole `MrI::Circle` ETA also guides but whose attempts
/// are few and long (60 M-frame tests: 0 of 76 reached it with the cap alive; wall kicks up the column took ~550
/// frames from the box where the best chain takes ~170).
pub struct Climb {
    circle: Circle,
    /// The gate stage (`MrI::Gate`) instead of the climb: done at the gate walls with the cap alive.
    gate: bool,
    /// The box stage (`MrI::Box`): done once a cap that lasts to the gate is picked up.
    boxed: bool,
}

impl Climb {
    /// Only for the big Mr. I behind the vanish-cap walls.
    pub fn new(b: &St, o: &Obj, gate: bool, boxed: bool) -> Option<Climb> {
        let circle = Circle::mr_i(b, o);
        (circle.secret && circle.base_criteria(b)).then_some(Climb { circle, gate, boxed })
    }

    fn reached(&self, s: &St) -> bool {
        use super::mansion::{MARIO_VANISH_CAP, SECRET_GATE_WEST, gate_frames, in_secret_room};
        if s.mario_flags & MARIO_VANISH_CAP == 0 {
            return false;
        }
        if self.boxed {
            // A cap that lasts to the gate by the column: the second-floor box's (the first floor's is too far).
            return s.cap_timer as f64 >= MARGIN.0 * gate_frames(s, s.pos) + MARGIN.1;
        }
        if s.pos[1] < 1700.0 {
            return false;
        }
        if self.gate {
            // At the west wall (or already inside) with a few frames of cap to cross the two walls.
            return in_secret_room(s.pos) || (hdist(s.pos, SECRET_GATE_WEST) < 150.0 && s.cap_timer >= 20);
        }
        let m = margin_at(s.pos);
        s.cap_timer as f64 >= m.0 * gate_frames(s, s.pos) + m.1
    }
}

impl Objective for Climb {
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn kind(&self) -> &'static str {
        if self.boxed { "MrI::Box" } else if self.gate { "MrI::Gate" } else { "MrI::Climb" }
    }
    fn key(&self) -> String {
        format!("{} {:?} {}", self.kind(), self.circle.zone, self.circle.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        self.circle.weight
    }
    fn base_criteria(&self, s: &St) -> bool {
        self.circle.base_criteria(s)
    }
    fn ga_budget(&self) -> Option<GaBudget> {
        self.circle.ga_budget()
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.circle.zone {
            return Eta::Lost;
        }
        if s.objects.is_some() && self.circle.dead(find(s, self.circle.id, false)) || self.reached(s) {
            return Eta::Done;
        }
        // The gate stage is the walk to the walls; the climb's is the same sum (its end is on the way).
        Eta::Frames(secret_to_gate(s, if self.gate { (1.0, 10.0) } else { MARGIN }))
    }
    fn success_value(&self) -> f64 {
        3.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.circle.steer_target(s)
    }
    fn exists_in(&self, s: &St) -> bool {
        self.circle.base_criteria(s)
    }
}

impl Objective for Circle {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        self.kind
    }
    fn key(&self) -> String {
        format!("{} {:?} {}", self.kind, self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        self.weight
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && !self.dead(find(s, self.id, false))
            && self.star.is_none_or(|(c, i)| !s.has_star(c, i))
    }
    fn ga_budget(&self) -> Option<GaBudget> {
        // The big Mr. I's way is long and technical (the wall-kick column, the cap clock): twice the budget.
        Some(if self.secret { GaBudget { population: 32, generations: 8, horizon: 300 } } else { GaBudget { population: 16, generations: 5, horizon: 240 } })
    }
    fn eta(&self, s: &St, _start: &St, mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let o = find(s, self.id, false);
        // Gone or dying; a state whose objects weren't read can't tell.
        if s.objects.is_some() && self.dead(o) {
            return Eta::Done;
        }
        let Some(o) = o else {
            mem.flags &= !IN_RANGE;
            return Eta::Lost;
        };
        // The secret room's chamber is walled: from outside only the vanish cap gets Mario in.
        if self.secret && !super::mansion::in_secret_room(s.pos) {
            mem.flags &= !IN_RANGE;
            mem.f[0] = 0.0;
            return self.secret_eta(s, o);
        }
        let d = hdist(s.pos, o.pos);
        // BBH's rooms stack: another floor's Mr. I (or Mr. Blizzard) is out
        // of range however close above or below.
        let other_floor = super::mansion::in_bbh(s) && (s.pos[1] - o.pos[1]).abs() > 400.0;
        if d > self.range || other_floor {
            // Out of range: the turning resets (v1's accumulator did too).
            mem.flags &= !IN_RANGE;
            mem.f[0] = 0.0;
            // BBH's secret room: up through the mansion's rooms and doors.
            if super::mansion::in_bbh(s) {
                return Eta::Frames((super::mansion::walk_frames(s, o.pos) - frames_for(self.radius)).max(0.0) + self.laps_left(0.0));
            }
            return Eta::Frames(frames_for(d - self.radius) + self.laps_left(0.0));
        }
        let a = ((s.pos[0] - o.pos[0]) as f64).atan2((s.pos[2] - o.pos[2]) as f64);
        if self.is_mr_i() && s.objects.is_some() {
            // Mr. I counts his own turn (`value` = oMrIUnkFC, read.rs): the number to beat is exact, and it
            // only runs in his action 2 (Mario within 700 and in front of him, his face turned to Mario).
            mem.flags |= IN_RANGE;
            mem.f[1] = a;
            let swept = if o.action == 2 { o.value.max(0) as f64 / 65536.0 * std::f64::consts::TAU } else { 0.0 };
            mem.f[0] = swept;
            let notice = if o.action == 2 { 0.0 } else { MR_I_NOTICE };
            return Eta::Frames(frames_for((d - self.radius).abs()) + self.laps_left(swept) + notice);
        }
        if mem.flags & IN_RANGE != 0 {
            let mut da = a - mem.f[1];
            if da > std::f64::consts::PI {
                da -= std::f64::consts::TAU;
            } else if da < -std::f64::consts::PI {
                da += std::f64::consts::TAU;
            }
            mem.f[0] += da;
        }
        mem.flags |= IN_RANGE;
        mem.f[1] = a;
        // Getting to the circle's radius, then the rest of the lap.
        Eta::Frames(frames_for((d - self.radius).abs()) + self.laps_left(mem.f[0]))
    }
    fn success_value(&self) -> f64 {
        if self.star.is_some() { 10.0 } else { 1.5 }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let o = find(s, self.id, false)?;
        // The secret room: the vanish cap (box, then the cap), then the west gate, then the room.
        if self.secret && !super::mansion::in_secret_room(s.pos) {
            use super::mansion::{
                COLUMN_BASE, MARIO_VANISH_CAP, SECRET_GATE_EAST, SECRET_GATE_WEST, THIRD_FLOOR_LANDING, in_column, reach_frames, reach_waypoint,
                vanish_sources,
            };
            let goal = if s.mario_flags & MARIO_VANISH_CAP != 0 {
                if in_column(s.pos) {
                    // Up the wall-kick column: toward the third floor's landing.
                    return Some([THIRD_FLOOR_LANDING[0], s.pos[1] + 200.0, THIRD_FLOOR_LANDING[2]]);
                } else if s.pos[1] < 1700.0 {
                    COLUMN_BASE
                } else if s.pos[0] > 60.0 || hdist(s.pos, SECRET_GATE_WEST) < 120.0 {
                    // Line up with the gate west of it, then through.
                    SECRET_GATE_EAST
                } else {
                    SECRET_GATE_WEST
                }
            } else {
                vanish_sources(s).into_iter().min_by(|a, b| reach_frames(s, *a).total_cmp(&reach_frames(s, *b)))?
            };
            return Some(reach_waypoint(s, goal, 600.0));
        }
        if super::mansion::in_bbh(s) && hdist(s.pos, o.pos) > self.range * 0.8 {
            Some(super::mansion::waypoint(s, o.pos, 600.0))
        } else if hdist(s.pos, o.pos) > self.range * 0.8 {
            Some([o.pos[0], s.pos[1], o.pos[2]])
        } else {
            Some(super::common::circle_point(s, o.pos, self.radius))
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::testkit::{obj, st};
    use obrrt_core::fold;

    /// Circling Mr. I: the swept angle (in `mem`) lowers the ETA each step,
    /// leaving its range resets it, and its dying act is `Done`.
    #[test]
    fn mr_i_circle_accumulates() {
        // His own count (`value`, oMrIUnkFC) rises 4000 a frame in action 2 as Mario runs round him.
        let mr_i = |action, value| {
            let mut o = obj("bhvMrI", 3, [0.0; 3], action, 0);
            o.value = value;
            o
        };
        let round = |a: f32, action| st(4, [300.0 * a.sin(), 0.0, 300.0 * a.cos()], vec![mr_i(action, (a * 6000.0) as i32)]);
        let mut traj: Vec<St> = (0..10).map(|i| round(i as f32 * 0.6, 2)).collect();
        let c = Circle::mr_i(&traj[0], &traj[0].objects()[0]);
        let f: Vec<f64> = fold(&c, &traj).1.iter().map(|e| e.frames()).collect();
        assert!(f.windows(2).all(|w| w[1] < w[0]), "ETA must fall: {f:?}");
        // Out of range: back to a whole lap (and the way in).
        traj.push(st(4, [0.0, 0.0, 1200.0], vec![mr_i(0, 0)]));
        let e = fold(&c, &traj).1;
        assert!(e[10].frames() > f[0]);
        traj.push(round(0.0, MR_I_ACT_DYING));
        assert_eq!(fold(&c, &traj).0, Some(11));
    }

    /// BBH's big Mr. I (second param byte 1): the ETA to its chamber goes by the vanish cap, the wall-kick column to the
    /// third floor and the gate: with a cap that lasts it falls up the column and across the third floor; with a
    /// cap that runs out first it is the way via a box again.
    #[test]
    fn big_mr_i_staged_eta() {
        let mut big = obj("bhvMrI", 3, [923.0, 1741.0, -332.0], 0, 0);
        big.beh_params = 0x0501_0000;
        let at = |p: [f32; 3], flags: u32, cap: u16| {
            let mut s = st(4, p, vec![big.clone()]);
            s.mario_flags = flags;
            s.cap_timer = cap;
            s
        };
        let c = Circle::mr_i(&at([0.0; 3], 0, 0), &big);
        assert!(c.secret);
        let eta = |s: &St| c.eta(s, s, &mut Mem::default()).frames();
        // Capped, up the column: each step up shortens the way; the third floor is nearer the gate still.
        let (base, mid, top, floor3) = (
            eta(&at([3330.0, 1024.0, 520.0], 2, 500)),
            eta(&at([3100.0, 1400.0, 450.0], 2, 450)),
            eta(&at([2600.0, 1700.0, 300.0], 2, 400)),
            eta(&at([1800.0, 1741.0, 478.0], 2, 350)),
        );
        assert!(base > mid && mid > top && top > floor3, "{base} {mid} {top} {floor3}");
        // A cap with 20 frames left at the column's foot can't last to the gate: the box again, farther than going on.
        assert!(eta(&at([3330.0, 1024.0, 520.0], 2, 20)) > base + 100.0);
    }
}
