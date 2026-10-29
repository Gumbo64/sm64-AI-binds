//! Kick technique objectives: `Tech::SlideKick` (`slidekick`) and
//! `Tech::JumpKick` (`jumpkick`).
//!
//! As with travel.rs, the duplicate cell ignores speed and action, so the
//! success is about time to a place: get within [`RADIUS`] of a point `p`
//! within `dist(start, p) / speed + SLACK` frames, with the kick in the
//! segment. The two differ from `diverollout`/`ljchain` in where `p` is and
//! which move counts:
//!
//! - **Slide kick** (900–2400 away; half the time `p` is 250–800 *below* b,
//!   down a slope or off a drop, at `speed` 40, else level within ±150 at
//!   `speed` 27: on flat ground the kick lands at once and ACT_SLIDE_KICK_SLIDE
//!   loses 32 → 7 in 25 frames, so only an immediate rollout keeps ~27–30;
//!   winning segments average 19–21). Z while
//!   walking/running enters ACT_CROUCH_SLIDE (act_walking); B there with
//!   `forwardVel >= 10` slide kicks (act_crouch_slide, mario_actions_moving.c): 12 up, forward speed
//!   raised to 32 (set_mario_action_airborne, mario.c). It lands in
//!   ACT_SLIDE_KICK_SLIDE, which slides with update_sliding (speeds up
//!   downhill, no cap but the slope's), and A there rolls out
//!   (ACT_FORWARD_ROLLOUT, speed kept). The TAS climbs to BoB's Chomp
//!   plateau with these chains at 86–96 u/frame; the bot averages 16–24.
//!   The downhill speed (40) is above dive rollouts (30) and long-jump
//!   chains (36); the level one counts slide kick–rollout chains on flat or
//!   gently sloped ground (the TAS's BoB start to the Chomp).
//! - **Jump kick** (`speed` 28, `p` level within ±150, 700–1600 away). In
//!   the air B with `forwardVel <= 28` jump kicks (check_kick_or_dive_in_air,
//!   mario_actions_airborne.c): 20 up, forward speed kept. On the landing
//!   frame of a dive rollout or long jump (ACT_*_LAND_STOP, stick neutral)
//!   B punches (check_common_landing_cancels) and A held turns the punch into
//!   a jump kick on the same frame (act_punching), keeping the landing speed
//!   instead of stopping; while walking, B below 29 is a move punch and A
//!   held kicks (act_move_punching). The speed is near the top run speed
//!   (32), so it takes a running or rolling start and a kick that doesn't
//!   lose it. The TAS jump kicks 4.4 times per 1,000 frames, the bot 0–0.6.
//!
//! Steering hints only make the buttons available on alternate frames
//! (`steer_button`); the GA does the timing.
//!
//! Fixes from the TAS catalogue (ref/tas/techniques/CATALOGUE.md), each on
//! by default and off with `OBRRT_KICKS_OFF` (a comma list of `jk`, `hop`,
//! `high`, `boost`, or `all`; see [`fix`]):
//!
//! - `jk` (G15): the landing-frame kick needs A *down but not newly pressed*
//!   and B pressed on the LAND_STOP frame, stick neutral (act_punching exits
//!   on any stick; its A_DOWN check runs on its first frame only, the same
//!   frame). The old hint gave B alone there, and A only on alternate frames
//!   coming down, so the punch never became a kick. Now a rule overlay holds
//!   A (B released) on every falling frame of a rollout or long jump and
//!   plays A | B with a short stick on a LAND_STOP frame after one
//!   ([`landing_kick_overlay`]).
//! - `boost` (G07): a slope-boost variant of `slidekick` (and of
//!   `diverollout`, travel.rs), [`Boost`]: on a downhill floor, slide (dive
//!   slide, slide kick, butt slide) to 75+ and leave it in the air at that
//!   speed (rollout, slide kick, jump kick, jump), 1500+ from the start, in
//!   60 frames.
//! - `hop`, `high`: dive-rollout variants, travel.rs.

use std::sync::OnceLock;

use obrrt_core::layout::{ACT_FLAG_AIR, ACT_FLAG_MOVING};
use obrrt_core::{Budget, GamePad, Objective, PadOverlay, Progress, St, buttons};

use super::travel::{ACT_CROUCH_SLIDE, ACT_DIVE, ACT_DIVE_SLIDE, ACT_FORWARD_ROLLOUT, ACT_FREEFALL, ACT_LONG_JUMP, aim_arrival, aim_key, aim_progress, aim_weight, can_start, unit};

/// Whether the catalogue fix `name` (`jk`, `hop`, `high`, `boost`) is on:
/// all are unless `OBRRT_KICKS_OFF` lists it (or `all`). With all off the
/// techniques behave exactly as before the fixes.
pub(super) fn fix(name: &str) -> bool {
    static OFF: OnceLock<Vec<String>> = OnceLock::new();
    let off = OFF.get_or_init(|| {
        let v: Vec<String> = std::env::var("OBRRT_KICKS_OFF")
            .unwrap_or_default()
            .split(',')
            .map(|w| w.trim().to_string())
            .filter(|w| !w.is_empty())
            .collect();
        for w in &v {
            assert!(["all", "jk", "hop", "high", "boost"].contains(&w.as_str()), "OBRRT_KICKS_OFF: unknown fix {w:?}");
        }
        v
    });
    !off.iter().any(|w| w == name || w == "all")
}

/// The objective weight of a movement technique (diverollout, ljchain and
/// their variants, jumpkick, slidekick, flipheight, the wall-kick modes)
/// whose built-in weight is `w`. `OBRRT_MOVE_WEIGHT=<x>` scales them all so
/// the 0.5 ones weigh `x` (at 1 or more, their progress-only attempts are
/// kept as nodes); unset, `w` is unchanged.
pub(super) fn move_weight(w: f64) -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    w * SCALE.get_or_init(|| {
        std::env::var("OBRRT_MOVE_WEIGHT").map_or(1.0, |v| v.parse::<f64>().expect("OBRRT_MOVE_WEIGHT: a number") / 0.5)
    })
}

/// Actions (include/sm64.h).
const ACT_SLIDE_KICK: u32 = 0x018008AA;
const ACT_SLIDE_KICK_SLIDE: u32 = 0x0080045A;
const ACT_JUMP_KICK: u32 = 0x018008AC;
const ACT_PUNCHING: u32 = 0x00800380;
const ACT_MOVE_PUNCHING: u32 = 0x00800457;
const ACT_JUMP_LAND_STOP: u32 = 0x0C000230;
const ACT_DOUBLE_JUMP_LAND_STOP: u32 = 0x0C000231;
const ACT_FREEFALL_LAND_STOP: u32 = 0x0C000232;
const ACT_SIDE_FLIP_LAND_STOP: u32 = 0x0C000233;
const ACT_LONG_JUMP_LAND_STOP: u32 = 0x0800023B;
const ACT_BACKWARD_ROLLOUT: u32 = 0x010008AD;
const ACT_BUTT_SLIDE: u32 = 0x00840452;

/// The landing-stop actions whose B punches (check_common_landing_cancels).
fn land_stop(action: u32) -> bool {
    matches!(
        action,
        ACT_FREEFALL_LAND_STOP | ACT_JUMP_LAND_STOP | ACT_DOUBLE_JUMP_LAND_STOP | ACT_SIDE_FLIP_LAND_STOP | ACT_LONG_JUMP_LAND_STOP
    )
}

/// The G15 landing-kick rule as a pad overlay: A held (B released) on every
/// falling frame of a rollout or long jump; on a landing-stop frame with A
/// held from the frame before and B not, A | B with the genome's stick cut
/// to raw length 50 (stickMag <= 44 after adjust_analog_stick). A neutral
/// stick punches from the landing stop (check_common_landing_cancels) and a
/// short one walks first, where B under stickMag 48 move-punches
/// (check_ground_dive_or_punch); either way the same frame's A_DOWN kicks
/// (act_punching / act_move_punching) and the speed is kept. The TAS does
/// the latter (BoB f21918: A held, then A | B with raw stick (0, 54) at 90).
/// Those frames are key (the GA mutates their stick). Elsewhere the
/// genome's pad.
fn landing_kick_overlay() -> PadOverlay {
    use buttons::{A_BUTTON, B_BUTTON};
    PadOverlay::keyed(|traj, played, genome| {
        let s = traj.last().unwrap();
        let prev = played.last();
        let held = |b: u16| prev.is_some_and(|p| p.button & b != 0);
        match s.action {
            ACT_FORWARD_ROLLOUT | ACT_BACKWARD_ROLLOUT | ACT_LONG_JUMP if s.vel[1] < 0.0 => {
                (GamePad::new((genome.button | A_BUTTON) & !B_BUTTON, genome.stick_x, genome.stick_y), false)
            }
            a if land_stop(a) && held(A_BUTTON) && !held(B_BUTTON) => {
                let (x, y) = (genome.stick_x as f32, genome.stick_y as f32);
                let k = (50.0 / (x * x + y * y).sqrt().max(1.0)).min(1.0);
                (GamePad::new(A_BUTTON | B_BUTTON, (x * k) as i8, (y * k) as i8), true)
            }
            _ => (genome, false),
        }
    })
}

/// Horizontal speed.
fn hs(s: &St) -> f32 {
    (s.vel[0] * s.vel[0] + s.vel[2] * s.vel[2]).sqrt()
}

/// The unit downhill direction (x, z) of Mario's floor, if it slopes.
fn downhill(s: &St) -> Option<[f32; 2]> {
    let [nx, _, nz] = s.floor_normal;
    let h = (nx * nx + nz * nz).sqrt();
    (h > 0.05).then(|| [nx / h, nz / h])
}

/// Whether Mario faces within 0x3000 (67.5°) of `dir`.
fn facing(s: &St, dir: [f32; 2]) -> bool {
    let a = s.yaw as f32 * std::f32::consts::TAU / 65536.0;
    a.sin() * dir[0] + a.cos() * dir[1] >= 0.383
}

/// Slope-boost speed, air speed a success needs, and its frames.
const BOOST_HS: f32 = 75.0;
const BOOST_DISP: f32 = 1500.0;
const BOOST_FRAMES: usize = 60;
/// Boost progress counts displacement only after an air speed above this
/// (a dive's cap, 48: only a slope slide gets faster).
const BOOST_GATE_HS: f32 = 48.0;

/// The slope-boost objective (G07), a variant of `slidekick` and
/// `diverollout`. The TAS dive-slides 46 → 94 in 18 frames down BoB's first
/// slope and butt-slides 35 → 105 down the WF fortress ramp: update_sliding /
/// apply_slope_accel (mario_actions_moving.c) add slope acceleration every
/// frame up to about 100; a rollout (A or B in DIVE_SLIDE / SLIDE_KICK_SLIDE)
/// keeps the speed, a butt-slide jump is ×0.8. A dive caps at 48, so no dive
/// above 44.
///
/// - Base: grounded or diving at 25+, on a floor with 0.3 < ny < 0.95 whose
///   downhill is within 60° of the objective's and that Mario faces within
///   67.5° of, 800–3500 from the target.
/// - Target: 1500–3000 down the fall line from b (dropping by the slope).
/// - Heuristic: distance / 80.
/// - Success: in an air action at 75+ horizontal speed, then 1500+ from the
///   start, within 60 frames.
/// - Progress: 0.6 · min(best air speed, 100) / 100 + 0.4 · min(disp, 1500) / 1500.
#[derive(Clone, Debug)]
pub struct Boost {
    pub p: [f32; 3],
    dir: [f32; 2],
}

impl Boost {
    fn slope(s: &St) -> bool {
        let ny = s.floor_normal[1];
        ny > 0.3 && ny < 0.95
    }
    fn grounded(s: &St) -> bool {
        s.action & ACT_FLAG_AIR == 0 || s.action == ACT_DIVE
    }

    pub fn eligible(b: &St) -> bool {
        can_start(b)
            && b.forward_vel >= 25.0
            && Self::slope(b)
            && Self::grounded(b)
            && downhill(b).is_some_and(|d| facing(b, d))
    }

    pub fn new(b: &St, salt: u64) -> Boost {
        let dir = downhill(b).unwrap_or([0.0, 1.0]);
        let r = 1500.0 + 1500.0 * unit(b, salt, 6);
        let ny = b.floor_normal[1].max(0.3);
        let drop = r * ((1.0 - ny * ny).sqrt() / ny).min(1.5);
        Boost { p: [b.pos[0] + r * dir[0], b.pos[1] - drop, b.pos[2] + r * dir[1]], dir }
    }

    pub fn key(&self, kind: &str, zone: (u16, u8)) -> String {
        let c = |v: f32| (v / 400.0).round() as i32;
        format!("{kind} boost {zone:?} {},{},{}", c(self.p[0]), c(self.p[1]), c(self.p[2]))
    }

    pub fn base(&self, s: &St) -> bool {
        let d = s.dist(self.p);
        can_start(s)
            && s.forward_vel >= 25.0
            && Self::slope(s)
            && Self::grounded(s)
            && downhill(s).is_some_and(|d| d[0] * self.dir[0] + d[1] * self.dir[1] >= 0.5)
            && facing(s, self.dir)
            && (800.0..=3500.0).contains(&d)
    }

    pub fn heuristic(&self, s: &St) -> f64 {
        s.dist(self.p) as f64 / 80.0
    }

    pub fn success(&self, traj: &[St]) -> Option<usize> {
        let start = traj.first()?;
        let n = traj.len().min(BOOST_FRAMES + 1);
        let fast = traj[..n].iter().position(|s| s.action & ACT_FLAG_AIR != 0 && hs(s) >= BOOST_HS)?;
        (fast..n).find(|&j| traj[j].dist(start.pos) >= BOOST_DISP)
    }

    pub fn progress(&self, traj: &[St]) -> Progress {
        let Some(start) = traj.first() else { return Progress { score: f64::MIN, at: 0 } };
        let mut best = (f64::MIN, 0);
        let mut top = 0.0f32;
        for (i, s) in traj.iter().enumerate().take(BOOST_FRAMES + 1) {
            if s.action & ACT_FLAG_AIR != 0 {
                top = top.max(hs(s));
            }
            // Displacement only once airborne faster than flat ground gives
            // (audit: a walk off the slope scored its distance).
            let disp = if top > BOOST_GATE_HS { s.dist(start.pos).min(BOOST_DISP) } else { 0.0 };
            let score = 0.6 * top.min(100.0) as f64 / 100.0 + 0.4 * (disp / BOOST_DISP) as f64;
            if score > best.0 {
                best = (score, i);
            }
        }
        Progress { score: best.0, at: best.1 }
    }

    pub fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: BOOST_FRAMES + 10 }
    }

    /// Hints: slide on, leave the slide (A) at the top speed or once the
    /// floor flattens (ny > 0.98, the lip); dive (B) only below 44, else it
    /// caps at 48; `slide`: into it by crouch slide → slide kick, else by a
    /// dive. Landing frames: A | B (the G15 kick, with the overlay).
    pub fn button(s: &St, slide: bool) -> u16 {
        use buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
        let air = s.action & ACT_FLAG_AIR != 0;
        let moving = s.action & ACT_FLAG_MOVING != 0;
        let flat = s.floor_normal[1] > 0.98;
        match s.action {
            ACT_DIVE_SLIDE | ACT_SLIDE_KICK_SLIDE if s.forward_vel >= BOOST_HS || flat => A_BUTTON,
            ACT_BUTT_SLIDE if s.forward_vel >= BOOST_HS / 0.8 || flat => A_BUTTON,
            ACT_DIVE_SLIDE | ACT_SLIDE_KICK_SLIDE | ACT_BUTT_SLIDE => 0,
            ACT_CROUCH_SLIDE if s.forward_vel >= 10.0 => B_BUTTON,
            ACT_DIVE | ACT_SLIDE_KICK | ACT_FORWARD_ROLLOUT => 0,
            a if land_stop(a) => A_BUTTON | B_BUTTON,
            ACT_FREEFALL if s.forward_vel < 44.0 => B_BUTTON,
            _ if air && s.forward_vel > 28.0 && s.forward_vel < 44.0 => B_BUTTON,
            _ if moving && !air && slide && s.forward_vel >= 16.0 => Z_TRIG,
            _ if moving && !air && s.forward_vel >= 29.0 && s.forward_vel < 44.0 => B_BUTTON,
            _ => 0,
        }
    }

    pub fn overlay() -> PadOverlay {
        landing_kick_overlay()
    }
}

/// Arrival radius, as PosDist's.
const RADIUS: f32 = 150.0;
/// Frames of slack on top of `dist / speed`.
const SLACK: f32 = 8.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kick {
    Slide,
    Jump,
}

impl Kick {
    fn kind(self) -> &'static str {
        match self {
            Kick::Slide => "Tech::SlideKick",
            Kick::Jump => "Tech::JumpKick",
        }
    }
    /// Start distances to the target an attempt may begin at.
    pub(super) fn starts(self) -> (f32, f32) {
        match self {
            Kick::Slide => (600.0, 3000.0),
            Kick::Jump => (500.0, 2000.0),
        }
    }
    fn key_action(self) -> u32 {
        match self {
            Kick::Slide => ACT_SLIDE_KICK,
            Kick::Jump => ACT_JUMP_KICK,
        }
    }
    /// Entries into the kick in `traj`.
    fn count(self, traj: &[St]) -> usize {
        let key = self.key_action();
        traj.windows(2).filter(|w| w[1].action == key && w[0].action != key).count()
    }
    fn salt(self) -> u64 {
        match self {
            Kick::Slide => 0x5C,
            Kick::Jump => 0x7C,
        }
    }
}

pub struct KickRun {
    kick: Kick,
    p: [f32; 3],
    /// Mean straight-line speed (u/frame) a success needs.
    speed: f32,
    /// Whether the start must be above `p` (a downhill slide kick).
    downhill: bool,
    zone: (u16, u8),
    /// The slope-boost variant (`boost`), half the time from a b on a slope.
    boost: Option<Boost>,
    /// Aimed at a route goal `p` (`OBRRT_TRAVEL_AIM`, travel.rs `generate_aim`).
    aim: bool,
}

impl KickRun {
    /// From `b`: a target in front of Mario half the time (chains go
    /// straight), else in any direction; for a slide kick 900–2400 away and
    /// half the time 250–800 below (else level within ±150), for a jump kick
    /// 700–1600 away and level within ±150.
    pub fn new(b: &St, kick: Kick) -> KickRun {
        let s = kick.salt();
        let downhill = kick == Kick::Slide && unit(b, s, 4) < 0.5;
        let (r, dy, speed) = match kick {
            Kick::Slide if downhill => (900.0 + 1500.0 * unit(b, s, 0), -250.0 - 550.0 * unit(b, s, 3), 40.0),
            Kick::Slide => (900.0 + 1500.0 * unit(b, s, 0), (unit(b, s, 3) - 0.5) * 300.0, 27.0),
            Kick::Jump => (700.0 + 900.0 * unit(b, s, 0), (unit(b, s, 3) - 0.5) * 300.0, 28.0),
        };
        let face = b.yaw as f32 * std::f32::consts::TAU / 65536.0;
        let a = if unit(b, s, 1) < 0.5 {
            face + (unit(b, s, 2) - 0.5) * std::f32::consts::FRAC_PI_2
        } else {
            unit(b, s, 2) * std::f32::consts::TAU
        };
        let p = [b.pos[0] + r * a.sin(), b.pos[1] + dy, b.pos[2] + r * a.cos()];
        let boost = (kick == Kick::Slide && fix("boost") && Boost::eligible(b) && unit(b, s, 5) < 0.5).then(|| Boost::new(b, s));
        KickRun { kick, p, speed, downhill, zone: b.zone(), boost, aim: false }
    }

    pub fn applies(b: &St) -> bool {
        can_start(b)
    }

    /// Aimed at the route goal `p`: a slide kick at 40 when `p` is 250+
    /// below b (and the start must stay above it), else 27; a jump kick 28.
    pub fn aimed(b: &St, kick: Kick, p: [f32; 3]) -> KickRun {
        let downhill = kick == Kick::Slide && p[1] < b.pos[1] - 250.0;
        let speed = match kick {
            Kick::Slide if downhill => 40.0,
            Kick::Slide => 27.0,
            Kick::Jump => 28.0,
        };
        KickRun { kick, p, speed, downhill, zone: b.zone(), boost: None, aim: true }
    }

    /// Frames a start at distance `d` has to arrive in.
    fn limit(&self, d: f32) -> usize {
        (d / self.speed + SLACK).ceil() as usize
    }
}

impl Objective for KickRun {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): technique setups are frame-precise: every state, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn kind(&self) -> &'static str {
        self.kick.kind()
    }
    fn key(&self) -> String {
        if let Some(bo) = &self.boost {
            return bo.key(self.kick.kind(), self.zone);
        }
        if self.aim {
            return aim_key(if self.downhill { "Tech::SlideKick down" } else { self.kick.kind() }, self.zone, self.p);
        }
        let c = |v: f32| (v / 400.0).round() as i32;
        format!("{} {:?} {},{},{}", self.kick.kind(), self.zone, c(self.p[0]), c(self.p[1]), c(self.p[2]))
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        if self.aim { aim_weight() } else { move_weight(0.5) }
    }
    fn base_criteria(&self, s: &St) -> bool {
        if let Some(bo) = &self.boost {
            return s.zone() == self.zone && bo.base(s);
        }
        let d = s.dist(self.p);
        let (lo, hi) = self.kick.starts();
        // A downhill slide kick needs the target below.
        let below = !self.downhill || s.pos[1] > self.p[1] + 150.0;
        s.zone() == self.zone && can_start(s) && (lo..=hi).contains(&d) && below
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        if let Some(bo) = &self.boost {
            return bo.heuristic(s);
        }
        s.dist(self.p) as f64 / 500.0 - s.forward_vel.max(0.0) as f64 / 32.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        if let Some(bo) = &self.boost {
            return bo.success(traj);
        }
        let start = traj.first()?;
        let limit = self.limit(start.dist(self.p));
        let i = if self.aim {
            aim_arrival(traj, self.zone, self.p, limit)?
        } else {
            traj.iter().take(limit + 1).position(|s| s.zone() == self.zone && s.dist(self.p) <= RADIUS)?
        };
        (self.kick.count(&traj[..=i]) >= 1).then_some(i)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        if let Some(bo) = &self.boost {
            return bo.progress(traj);
        }
        let Some(start) = traj.first() else { return Progress { score: f64::MIN, at: 0 } };
        let d0 = start.dist(self.p);
        let limit = self.limit(d0);
        if self.aim {
            return aim_progress(traj, self.zone, self.p, limit, self.speed, 1, |w| self.kick.count(w));
        }
        // Approach in the time allowed, credited only from the first kick
        // on (audit: walking at p scored); before it, the start's 0.
        let (mut n, mut best) = (0, Progress { score: 0.0, at: 0 });
        for i in 1..traj.len().min(limit + 1) {
            n += self.kick.count(&traj[i - 1..=i]);
            let s = &traj[i];
            if n == 0 || s.zone() != self.zone {
                continue;
            }
            let score = 200.0 * n.min(1) as f64 + (d0 - s.dist(self.p)) as f64;
            if score > best.score {
                best = Progress { score, at: i };
            }
        }
        best
    }
    fn budget(&self) -> Budget {
        if let Some(bo) = &self.boost {
            return bo.budget();
        }
        Budget::Ga { population: 16, generations: 5, horizon: self.limit(self.kick.starts().1) + 10 }
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.boost.as_ref().map_or(self.p, |bo| bo.p))
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        if self.boost.is_some() {
            return Some(Boost::overlay());
        }
        (self.kick == Kick::Jump && fix("jk")).then(landing_kick_overlay)
    }
    fn steer_button(&self, s: &St) -> u16 {
        use buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
        if self.boost.is_some() {
            return Boost::button(s, true);
        }
        let air = s.action & ACT_FLAG_AIR != 0;
        let moving = s.action & ACT_FLAG_MOVING != 0;
        match self.kick {
            Kick::Slide => match s.action {
                ACT_CROUCH_SLIDE if s.forward_vel >= 10.0 => B_BUTTON,
                ACT_SLIDE_KICK_SLIDE => A_BUTTON,
                ACT_SLIDE_KICK | ACT_FORWARD_ROLLOUT => 0,
                _ if moving && !air && s.forward_vel >= 16.0 => Z_TRIG,
                _ => 0,
            },
            Kick::Jump => match s.action {
                ACT_FREEFALL_LAND_STOP | ACT_JUMP_LAND_STOP | ACT_DOUBLE_JUMP_LAND_STOP
                | ACT_SIDE_FLIP_LAND_STOP | ACT_LONG_JUMP_LAND_STOP => {
                    // A held + B (G15; the overlay keeps A held and the
                    // stick neutral); before the fix B alone.
                    if fix("jk") { A_BUTTON | B_BUTTON } else { B_BUTTON }
                }
                ACT_PUNCHING | ACT_MOVE_PUNCHING => A_BUTTON,
                ACT_DIVE_SLIDE => A_BUTTON,
                ACT_JUMP_KICK | ACT_DIVE => 0,
                // Hold A coming down from a rollout or long jump, for the
                // landing frame's punch-to-kick.
                ACT_FORWARD_ROLLOUT | ACT_LONG_JUMP if s.vel[1] < 0.0 => A_BUTTON,
                _ if air && s.forward_vel <= 28.0 => B_BUTTON,
                _ if moving && !air && s.forward_vel >= 29.0 => B_BUTTON,
                _ => 0,
            },
        }
    }
}
