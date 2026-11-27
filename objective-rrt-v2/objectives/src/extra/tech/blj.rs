//! `Tech::Blj` (`blj`): a backwards long jump at a known spot, to a place
//! the BLJ reaches far faster than walking. The shared BLJ hints live here
//! too (`lblj` uses them).
//!
//! The mechanism (decomp): a long jump multiplies forward speed by 1.5 and
//! caps it only above +48 (set_mario_action_airborne, mario.c "(BLJ's)");
//! air drag below -16 is only +2 a frame (update_air_without_turn,
//! mario_actions_airborne.c). On ACT_LONG_JUMP_LAND, A with Z held
//! long-jumps again (act_long_jump_land; no Shindou reset in the US build).
//! When a rising floor ends each backwards long jump within a frame or two
//! the speed grows ×1.5 a jump. Long jumps need the floor not steep for the
//! facing (mario_floor_is_steep: facing downhill is fine), so Mario faces
//! down the slope and moves up it backwards.
//!
//! The duplicate cell ignores speed, so a BLJ by itself is no success, and
//! a generic one (any slope, any place reached fast) would fill the tree
//! with states in arbitrary places. Success is a specific new place, at the
//! spots where TASes use a BLJ (ref/tas/ANALYSIS.md, dumps/*.csv):
//! - **BoB elevator** (TAS 2026 f22551–22602; il-bob-king-3110): the
//!   checkerboard elevator by the mountain path (not a slope, see
//!   ref/blj-conditions.md 3.1), to -420, then on to the summit. Success:
//!   the summit, y ≥ 4000. With `poleredirect` on, also flying at 300+
//!   with a tree ahead on the velocity ray, or touching it at 300+ (TAS:
//!   -420 into the tree at (4208, 927, 3772)): Tech::PoleRedirect takes
//!   the speed on from there (the TAS reaches the summit only through two
//!   tree redirects). Then the summit no longer counts (speed only).
//! - **WF stairs** (il-wf-wildblue-757, frames ~8789–8816): BLJ on the
//!   stairs at (4460, 256, 4097), facing +x, to -336 u/frame, then
//!   backwards across the level to the Wild Blue ledge at (-2616, 1331,
//!   -742). Success: within 500 of the ledge (or the star).
//! - **Lobby left-hand steps** (TAS 2026 f3245–3316): a riser BLJ to about
//!   -300, then backwards through the lobby's back wall into CCM area 2.
//!   Success: past the wall or in CCM.
//!
//! - **`bljspots`** (separate OBRRT_TECH name): the bot-relevant spots of
//!   ref/tas/techniques/CATALOGUE.md 3.1 with their TAS exit goals (lobby
//!   centre steps -> WF, WF-room steps -> BoB, lobby left steps -> PSS, the
//!   castle 1F seam -> PSS, BoB steep slope -> red-coin posts, CCM east ->
//!   chimney warp, CCM plateau -> bottom warp pad), all as `Tech::BljRun`;
//!   after the chain the genome steers to the goal, and with `hyperwalk`
//!   on a chain at -250 hands over to Tech::HyperWalk.
//!
//! Spots with a known rule (BoB elevator, lobby steps) are attempted as
//! `Tech::BljRun`: the GA with [`overlay`], which does the setup and every
//! button and leaves only the press frames' stick to the genome. The WF
//! stairs keep the hint GA (`Tech::Blj`):
//!
//! **Speed only (2026-10-25, bljtree):** every BLJ objective here (and
//! `lblj`, `bljany`) only builds speed. v2: the ETA is [`build_eta`] (walk
//! to the setup point + [`SETUP_FRAMES_ETA`] + [`FIRST_FLIGHT_FRAMES`], then
//! [`JUMP_FRAMES`] per x1.5 re-jump left to the target speed), the chain
//! state folded per frame by [`chain_step`]; success is a chain built in the trajectory reaching
//! [`CHAIN_SPEED`] (the spot's hand-off speed with HyperWalk /
//! PoleRedirect on), or a spot's place / star / tree only after a chain
//! reached [`CHAIN_BUILT`]. In v42-tech-s1 the BLJ kinds' added nodes were
//! mostly walks (Lblj: 13 of 13 without a growing chain, peak landing
//! <= 16, the end nearest the trap). Rule spots
//! run in two stages ([`staged_overlay`]): setup to the pose (idle at the
//! run-up point, facing the yaw), then the rule, so starts anywhere within
//! 1500 of a `blj` spot work.
//!
//! Hints only make the BLJ available; the GA does the setup and timing: in
//! a long jump or its landing the stick is held back (away from Mario's
//! facing) and A+Z come on alternate frames (a press needs a release; Z
//! must be down for A to count on the landing); in a crouch slide A+Z too.
//! Elsewhere the steer target is the goal and no button is suggested (Z in
//! most air actions would ground-pound). The graded BLJ [`metric`]
//! (ref/blj-conditions.md section 5) is kept for labs; objectives use the
//! ETA.

use obrrt_core::buttons::{A_BUTTON, Z_TRIG};
use obrrt_core::layout::{ACT_FLAG_MOVING, LEVEL_BOB, LEVEL_CASTLE, LEVEL_CCM, LEVEL_PSS, LEVEL_WF};
use obrrt_core::{Eta, GamePad, Mem, Objective, PadOverlay, PadPolicy, Precision, St, frames_for};

use super::enabled;
use super::poleredirect;

pub(super) const ACT_LONG_JUMP: u32 = 0x0300_0888;
pub(super) const ACT_LONG_JUMP_LAND: u32 = 0x0000_0479;
pub(super) const ACT_CROUCH_SLIDE: u32 = 0x0480_8459;

pub(super) fn long_jumping(s: &St) -> bool {
    matches!(s.action, ACT_LONG_JUMP | ACT_LONG_JUMP_LAND)
}

/// The BLJ steer hint: in a long jump, a point behind Mario's facing (the
/// stick held back), else `goal`.
pub(super) fn steer(s: &St, goal: [f32; 3]) -> [f32; 3] {
    if long_jumping(s) {
        let yaw = s.yaw as f32 * std::f32::consts::TAU / 65536.0;
        [s.pos[0] - 1000.0 * yaw.sin(), s.pos[1], s.pos[2] - 1000.0 * yaw.cos()]
    } else {
        goal
    }
}

/// The BLJ button hint (pressed on alternate frames by the controller).
pub(super) fn button(s: &St) -> u16 {
    if long_jumping(s) || s.action == ACT_CROUCH_SLIDE { A_BUTTON | Z_TRIG } else { 0 }
}

/// The graded BLJ metric of ref/blj-conditions.md section 5 (validated there
/// on 31 TAS chains; port of run/examples/blj_lab.rs `metric`). Per chain of
/// BLJ jumps (a ×1.5 re-jump from ACT_LONG_JUMP_LAND, every frame after it in
/// a long-jump action): `log1.5(max(16, peak landing speed) / 16) + 0.25 G
/// + 0.1 min(J, 10)` (G jumps whose landing speed grew, J jumps); the best
/// chain, +0.5 if the trajectory ends still chaining below -16. Knockback,
/// slides and plain long jumps score 0. Also returns the best chain's peak.
pub fn metric(traj: &[St]) -> (f64, f32) {
    let (m, peak, _, _) = metric_at(traj);
    (m, peak)
}

/// A BLJ re-jump from `p` to the next frame `s`: a x1.5 jump from
/// ACT_LONG_JUMP_LAND moving backwards (the metric's jump detector).
pub fn rejump(p: &St, s: &St) -> bool {
    rejump_from(p.action, p.forward_vel, s)
}

/// The backwards speed a BLJ chain must reach to count as built
/// (`Tech::BljRun` / `Tech::Lblj` success; the HyperWalk base is 100).
pub const CHAIN_SPEED: f32 = 150.0;
/// A chain this fast counts as built for the place successes (an exit
/// reached through the chain, not by walking there).
pub const CHAIN_BUILT: f32 = 100.0;

/// Per trajectory index: the fastest backwards speed in a long-jump action
/// since the first BLJ re-jump after the start (0 before one). A start
/// already mid-chain doesn't count as a re-jump; its speed counts only once
/// a re-jump follows.
pub fn chain_speed(traj: &[St]) -> Vec<f32> {
    let mut mem = Mem::default();
    traj.iter().map(|s| chain_step(s, &mut mem)).collect()
}

// ---- The per-frame chain tracker (v2: folded through `Objective::eta`) ----
//
// [`chain_step`] keeps [`chain_speed`]'s state in a [`Mem`]: `f[0]` the
// previous state's action, `f[1]` its forward speed, `f[2]` the chain's best
// backwards speed, and flag bits [`M_SEEN`] (a state was seen: this one is
// not the start) and [`M_ON`] (a re-jump happened). `f[3]` and the bits from
// [`M_FREE`] up are the objective's own.

/// Mem flag: a state was folded (the current one is not the start).
pub const M_SEEN: u32 = 1 << 0;
/// Mem flag: a BLJ re-jump happened since the start.
pub const M_ON: u32 = 1 << 1;
/// The first Mem flag bit free for the objective.
pub const M_FREE: u32 = 1 << 2;

/// [`rejump`] from the previous state's action and forward speed.
fn rejump_from(pa: u32, pfv: f32, s: &St) -> bool {
    pa == ACT_LONG_JUMP_LAND
        && long_jumping(s)
        && pfv < -1.0
        && (s.forward_vel - 1.5 * pfv).abs() <= 4.5
        && (s.action == ACT_LONG_JUMP || s.vel[1] > 20.0)
}

/// One frame of [`chain_speed`]: folds `s` into `mem` and returns its
/// value at `s` (the fastest backwards long-jump speed since the first BLJ
/// re-jump after the start, 0 before one). Call once per state, in order.
pub fn chain_step(s: &St, mem: &mut Mem) -> f32 {
    if mem.flags & M_SEEN != 0 {
        if rejump_from(mem.f[0] as u32, mem.f[1] as f32, s) {
            mem.flags |= M_ON;
        }
        if mem.flags & M_ON != 0 && long_jumping(s) {
            mem.f[2] = mem.f[2].max(-s.forward_vel as f64);
        }
    }
    mem.flags |= M_SEEN;
    mem.f[0] = s.action as f64;
    mem.f[1] = s.forward_vel as f64;
    mem.f[2] as f32
}

/// Whether `s` is the start (the first state folded): call before
/// [`chain_step`].
pub fn is_start(mem: &Mem) -> bool {
    mem.flags & M_SEEN == 0
}

/// Frames of one BLJ re-jump cycle (a press on the landing, a jump that
/// lands again at once): 2 (pausebuf.rs: "the 2-frame cycle"; the moat
/// TAS's stall, x1.5 per 2 frames).
pub const JUMP_FRAMES: f64 = 2.0;
/// A plain backwards long jump lands at about this speed (air drag +2 a
/// frame below -16): the chain's first speed.
const CHAIN_BASE: f32 = 16.0;
/// The first (setup) long jump's flight, from the crouch slide's press to
/// its backwards landing (a long jump has about 25-31 air frames at -16,
/// ref/blj-conditions.md 1.4).
pub const FIRST_FLIGHT_FRAMES: f64 = 25.0;
/// From standing at the setup point to the first long jump: turn, run-up
/// to 10.5, crouch slide, press (the rule's 40-unit run-up: ~10 frames).
pub const SETUP_FRAMES_ETA: f64 = 10.0;

/// Frames of x1.5 re-jumps from backwards speed `v` to `target`.
pub fn chain_frames(v: f32, target: f32) -> f64 {
    let v = v.max(CHAIN_BASE);
    if v >= target { 0.0 } else { (target as f64 / v as f64).ln() / 1.5f64.ln() * JUMP_FRAMES }
}

/// The ETA of building a BLJ chain to backwards speed `target` from `s`,
/// with the setup at `setup` (None: wherever Mario is): in the chain
/// (a long jump or its landing moving backwards) the re-jumps left; in the
/// first long jump its flight and the whole chain; in the crouch slide one
/// press more; else the walk to the setup point and the setup.
pub fn build_eta(s: &St, setup: Option<[f32; 3]>, target: f32) -> f64 {
    let chain = chain_frames(CHAIN_BASE, target);
    if long_jumping(s) && s.forward_vel < -1.0 {
        chain_frames(-s.forward_vel, target)
    } else if long_jumping(s) {
        FIRST_FLIGHT_FRAMES + chain
    } else if s.action == ACT_CROUCH_SLIDE {
        1.0 + FIRST_FLIGHT_FRAMES + chain
    } else {
        setup.map_or(0.0, |p| frames_for(s.hdist(p))) + SETUP_FRAMES_ETA + FIRST_FLIGHT_FRAMES + chain
    }
}

/// [`metric`], the trajectory index where the best chain's score was
/// reached (0 if there is no chain) and that chain's growth landings (G).
pub fn metric_at(traj: &[St]) -> (f64, f32, usize, u32) {
    let in_lj = long_jumping;
    let jump = rejump;
    let score = |peak: f32, grown: u32, jumps: u32| {
        (peak.max(16.0) / 16.0).ln() as f64 / 1.5f64.ln() + 0.25 * grown.min(20) as f64 + 0.1 * jumps.min(10) as f64
    };
    let (mut best, mut best_peak, mut best_at, mut best_grown) = (0.0f64, 0.0f32, 0usize, 0u32);
    let (mut on, mut pending, mut peak, mut jumps, mut grown, mut last_land) = (false, false, 0f32, 0u32, 0u32, 0f32);
    for (i, w) in traj.windows(2).enumerate() {
        let (p, s) = (&w[0], &w[1]);
        if !in_lj(s) {
            on = false;
            continue;
        }
        if jump(p, s) {
            if !on {
                (on, peak, jumps, grown, last_land) = (true, 0.0, 0, 0, p.forward_vel);
            }
            jumps += 1;
            pending = true;
        }
        if on && pending && s.action == ACT_LONG_JUMP_LAND {
            pending = false;
            let v = s.forward_vel;
            if v < last_land - 0.5 && v < -16.0 {
                grown += 1;
            }
            last_land = v;
            peak = peak.max(-v);
        }
        if on {
            let c = score(peak, grown, jumps);
            if c > best {
                (best, best_peak, best_at, best_grown) = (c, peak, i + 1, grown);
            }
        }
    }
    let alive = traj.last().is_some_and(|l| on && in_lj(l) && l.forward_vel < -16.0);
    (best + if alive { 0.5 } else { 0.0 }, best_peak, best_at, best_grown)
}

/// The cosine between `s`'s horizontal velocity and the straight-line
/// bearing to `goal` (0 when not moving).
pub fn alignment(s: &St, goal: [f32; 3]) -> f32 {
    let (vx, vz) = (s.vel[0], s.vel[2]);
    let (gx, gz) = (goal[0] - s.pos[0], goal[2] - s.pos[2]);
    let n = vx.hypot(vz) * gx.hypot(gz);
    if n < 1e-3 { 0.0 } else { (vx * gx + vz * gz) / n }
}



/// The full-tilt raw stick that points `intendedYaw` along `yaw` (s16
/// units), camera-relative as in update_mario_joystick_inputs.
fn stick_along(s: &St, yaw: i32) -> (i8, i8) {
    stick_tilt(s, yaw, 127.0)
}

/// adjust_analog_stick (game_init.c) moves each raw axis 6 towards 0 (and
/// zeroes |raw| < 8), and intendedYaw comes from the adjusted values: aim
/// those, then add the 6 back.
fn stick_tilt(s: &St, yaw: i32, tilt: f32) -> (i8, i8) {
    let rel = (yaw - s.cam_yaw as i32) as f32 * std::f32::consts::TAU / 65536.0;
    let t = tilt - 6.0;
    let raw = |v: f32| if v.abs() < 0.5 { 0 } else { (v + 6.0 * v.signum()).round().clamp(-127.0, 127.0) as i8 };
    (raw(t * rel.sin()), raw(-t * rel.cos()))
}

/// SM64 yaw of the direction from `a` to `b`.
fn yaw_to(a: [f32; 3], b: [f32; 3]) -> i32 {
    ((b[0] - a[0]).atan2(b[2] - a[2]) * 32768.0 / std::f32::consts::PI) as i32
}

/// The BLJ as a closed-loop policy (`Budget::Policy`), from the decomp:
/// - **Setup:** walk to `run_up` before `start`, then run along `yaw` (Mario's facing for
///   the BLJ: down the slope or stairs, mario_floor_is_steep lets him long
///   jump facing downhill). At `forwardVel >= 10.5` Z starts a crouch slide
///   (act_walking), and a new A press in it long-jumps (act_crouch_slide
///   needs `forwardVel > 10`).
/// - **Chain:** in ACT_LONG_JUMP the stick is held back (away from the
///   facing: `forwardVel += 1.5 * cos(dYaw)` a frame, update_air_without_turn)
///   with Z held and A released; in ACT_LONG_JUMP_LAND with `forwardVel < 0`,
///   standing on the chain's lowest floor, a new A press (Z down,
///   act_long_jump_land) jumps again before any deceleration, multiplying the
///   speed by 1.5 with no negative cap (set_mario_action_airborne). Against
///   a stair riser that jump lands on the step in the same frame, so there's
///   no air drag; the wall pushes him back down and it repeats (the TAS
///   lobby BLJ, ref/tas/dumps/full-120star-2026-U.csv frames 3279–3302).
/// - Before the first landing the stick aims at `spot` instead of straight
///   back, to land the first jump on the BLJ place (a stair riser).
/// - Past `stop` backwards speed it presses nothing more and lets him fly.
///
/// The speed only grows when the floor or a ceiling ends each jump within
/// a few frames (net air drag at < -16 is about +0.85 a frame against the
/// x1.5): that's the start point's job.
///
/// Status (2026-10-24, scratch lab booted from the 2026 TAS at frame 3244,
/// lobby left steps): it reproduces the TAS's first two stair jumps
/// (-15.9 -> -23.1 -> -33.0; TAS -15.9 -> -22.7 -> -31.3) but the next
/// push-back lands on the x = -2230 side wall instead of the riser and the
/// chain flies off; best -33.9. Not offered by any objective yet.
/// Backwards speed above which a BLJ grows even on flat ground.
const FLAT_GROWS: f32 = 53.0;

pub fn policy(start: [f32; 3], yaw: i32, run_up: f32, spot: Option<[f32; 3]>, stop: f32) -> PadPolicy {
    let r = Rule { start, yaw, run_up, spot, stop, free_exit: false };
    PadPolicy::new(move |traj, played| r.pad(traj, played, None))
}

/// The same rule as a [`PadOverlay`] on the GA's inputs: everything from
/// the rule except the stick on the chain's press frames, which is the
/// genome's. That stick sets the jump's sideways speed, which persists
/// through the landing frames and decides where the riser pushes Mario back
/// to, so it decides whether the next press lands on the step (the TAS
/// steers each press; in the lab, searching just those sticks took the
/// chain from -33.9 to -137.7). Those press frames are the overlay's key
/// frames, where the GA focuses part of its mutation.
pub fn overlay(start: [f32; 3], yaw: i32, run_up: f32, spot: Option<[f32; 3]>, stop: f32) -> PadOverlay {
    overlay_exit(start, yaw, run_up, spot, stop, false)
}

/// [`overlay`]; with `free_exit`, once the chain has ended the genome's pad
/// is played (it steers the exit to the spot's goal) instead of nothing.
pub fn overlay_exit(start: [f32; 3], yaw: i32, run_up: f32, spot: Option<[f32; 3]>, stop: f32, free_exit: bool) -> PadOverlay {
    let r = Rule { start, yaw, run_up, spot, stop, free_exit };
    PadOverlay::keyed(move |traj, played, genome| {
        let pad = r.pad(traj, played, Some(genome));
        // Key frames: the chain's presses (the genome's stick is used there).
        let key = pad.button & A_BUTTON != 0 && traj.last().is_some_and(|s| s.action == ACT_LONG_JUMP_LAND)
            || (r.free_exit && traj.last().is_some_and(|s| s.action == ACT_LONG_JUMP) && !traj.iter().any(|t| t.action == ACT_LONG_JUMP_LAND));
        (pad, key)
    })
}

#[derive(Clone, Copy)]
struct Rule {
    start: [f32; 3],
    yaw: i32,
    run_up: f32,
    spot: Option<[f32; 3]>,
    stop: f32,
    free_exit: bool,
}

impl Rule {
    /// This frame's pad; with `genome`, its stick where the GA steers.
    fn pad(&self, traj: &[St], played: &[GamePad], genome: Option<GamePad>) -> GamePad {
        let Rule { start, yaw, run_up, spot, stop, free_exit } = *self;
        let ga_stick = |own: (i8, i8)| genome.map_or(own, |g| (g.stick_x, g.stick_y));
        let s = traj.last().unwrap();
        let held_a = played.last().is_some_and(|p| p.button & A_BUTTON != 0);
        // The chain's frames: from the start if it is already mid-chain
        // (moving backwards in a long jump), else after it, so a start in a
        // forward long jump or its landing (a tree state) isn't taken for a
        // chain that broke: the setup runs instead.
        let mid_chain = long_jumping(&traj[0]) && traj[0].forward_vel < 0.0;
        let chain = if mid_chain { traj } else { &traj[1..] };
        let begun = mid_chain || chain.iter().any(|t| t.action == ACT_LONG_JUMP);
        // Until the first landing the stick aims at `spot` (if any), so the
        // first long jump comes down at the BLJ place; then straight back.
        let landed = chain.iter().any(|t| t.action == ACT_LONG_JUMP_LAND);
        let back = match spot {
            Some(p) if !landed && s.hdist(p) > 30.0 => stick_along(s, yaw_to(s.pos, p)),
            _ => stick_along(s, s.yaw as i32 + 0x8000),
        };
        if s.action == ACT_LONG_JUMP && begun {
            // Free exit (the `bljspots` spots, whose first landing isn't
            // tuned): the first jump's air stick is the genome's, so the GA
            // can find the landing on the riser.
            let (x, y) = if free_exit && !landed { ga_stick(back) } else { back };
            return GamePad::new(Z_TRIG, x, y);
        }
        if s.action == ACT_LONG_JUMP_LAND && begun {
            // Jump again only from the chain's lowest floor (the first
            // landing's), so each jump comes down on the step behind at once
            // (a press from the step top flies over the next one). Between
            // presses the stick is neutral: common_landing_action then keeps
            // a negative forwardVel (no friction below +16).
            let base_y = chain.iter().find(|t| t.action == ACT_LONG_JUMP_LAND).map_or(s.pos[1], |t| t.pos[1]);
            // And only once a frame's travel reaches where the last jump came
            // down on the step (the riser is just before it), or before the
            // first such landing, `spot`.
            let step = chain.iter().rev().find(|t| t.action == ACT_LONG_JUMP_LAND && t.pos[1] > base_y + 10.0).map(|t| t.pos);
            // ACT_LONG_JUMP_LAND lasts 6 frames (sLongJumpLandAction) and a
            // press on its last one only jumps (ACT_JUMP), so the 5th is the
            // last chance. A press restarts the count (the jump lands again
            // in the same frame).
            let since_press = played.iter().rev().take_while(|p| p.button & A_BUTTON == 0).count();
            let landed_for = traj.iter().rev().take(since_press).take_while(|t| t.action == ACT_LONG_JUMP_LAND).count();
            let near = landed_for >= 4 || step.or(spot).is_none_or(|p| s.hdist(p) <= -s.forward_vel);
            // Without a riser point (a rising floor: an elevator) every
            // landing frame is a chance: the jump lands at once anyway.
            // Past about 53 even flat ground grows (n < |v| / 1.7 with ~31 air
            // frames, ref/blj-conditions.md 1.4): press at every landing on
            // the lowest floor, near the riser or not (the TAS's last press
            // is on the flat floor past the steps).
            let gated = spot.is_none() || (s.pos[1] <= base_y + 10.0 && (near || -s.forward_vel > FLAT_GROWS));
            let press = s.forward_vel < 0.0 && -s.forward_vel < stop && !held_a && gated;
            // On the press the jump's first air step runs this frame: aim the
            // stick at the riser point, so its sideways speed (up to 10,
            // update_air_without_turn) lines the jump up with the step.
            let aim = match step.or(spot) {
                Some(p) if s.hdist(p) > 5.0 => stick_along(s, yaw_to(s.pos, p)),
                _ => back,
            };
            let aim = ga_stick(aim);
            // Between presses the stick stays neutral: then neither
            // forwardVel nor vel[0]/vel[2] is recomputed (common_landing_action
            // skips apply_landing_accel), so the sideways speed the press
            // frame's stick gave the jump carries Mario along the riser.
            return if press { GamePad::new(Z_TRIG | A_BUTTON, aim.0, aim.1) } else { GamePad::new(Z_TRIG, 0, 0) };
        }
        if begun {
            // The chain broke (bonk, wall, steep floor) or ended: stop, or
            // (free exit) the genome steers on.
            return match genome {
                Some(g) if free_exit => g,
                _ => GamePad::new(0, 0, 0),
            };
        }
        if s.action == ACT_CROUCH_SLIDE {
            // The long jump's first air step runs this same frame, so the
            // stick is already the air one.
            return GamePad::new(Z_TRIG | if held_a { 0 } else { A_BUTTON }, back.0, back.1);
        }
        // Run-up: from a point `run_up` before `start` (against `yaw`), along `yaw`.
        let r = yaw as f32 * std::f32::consts::TAU / 65536.0;
        let pre = [start[0] - run_up * r.sin(), start[1], start[2] - run_up * r.cos()];
        if !traj.iter().any(|t| t.hdist(pre) < 40.0) {
            // Walk in slowly at the end so he doesn't overshoot (and stops
            // there: a still start turns at once).
            let d = s.hdist(pre);
            let tilt = if d > 700.0 { 127.0 } else { (127.0 * d / 700.0).max(30.0) };
            let (x, y) = stick_tilt(s, yaw_to(s.pos, pre), tilt);
            return GamePad::new(0, x, y);
        }
        let lined_up = ((s.yaw as i32 - yaw) as i16).unsigned_abs() < 0x180;
        if !lined_up && s.action & ACT_FLAG_MOVING != 0 {
            // Stop before turning: from ACT_IDLE the stick sets the facing
            // at once (act_idle -> ACT_WALKING), while walking turns slowly.
            return GamePad::new(0, 0, 0);
        }
        let (x, y) = stick_along(s, yaw);
        // Z must be a new press (act_walking reads INPUT_Z_PRESSED): pulse it.
        let held_z = played.last().is_some_and(|p| p.button & Z_TRIG != 0);
        let z = lined_up && s.forward_vel >= 10.5 && s.hdist(pre) > run_up * 0.5 && !held_z;
        GamePad::new(if z { Z_TRIG } else { 0 }, x, y)
    }
}

pub struct Spot {
    name: &'static str,
    level: u32,
    area: u8,
    /// Where the BLJ starts, and how close a base state must be.
    start: [f32; 3],
    start_radius: f32,
    /// The place it reaches: within `goal_radius` of `goal`, or (if set)
    /// at least `goal_y` high anywhere in the area.
    goal: [f32; 3],
    goal_radius: f32,
    goal_y: Option<f32>,
    /// Entering this level also counts as arriving.
    dest: Option<u32>,
    /// The star the place is for (course, index): not offered once had.
    star: Option<(i16, u8)>,
    /// A known BLJ here, attempted with the [`overlay`] (`Tech::BljRun`)
    /// instead of the hint GA: the rule's facing `yaw`, `run_up` and the
    /// riser point (`spot` of [`policy`]; none on a rising floor), starting
    /// from `start`.
    rule: Option<(i32, f32, Option<[f32; 3]>)>,
    /// The OBRRT_TECH name that turns the spot on.
    tech: &'static str,
    /// Entering this area of `level` also counts as arriving.
    dest_area: Option<u8>,
    /// The rollout horizon (frames).
    horizon: usize,
    /// After the chain the genome steers to the goal (an exit that needs
    /// hyperspeed walking, jumps or kicks), and with `hyperwalk` on a chain
    /// at [`HANDOFF`] is a success too (Tech::HyperWalk takes it on).
    free_exit: bool,
    /// With `poleredirect` on, flying at [`TREE_SPEED`]+ with a pole ahead
    /// on the velocity ray (Tech::PoleRedirect's base), or touching one at
    /// that speed, is a success: the hand-off to Tech::PoleRedirect.
    tree_handoff: bool,
}

/// The lobby-steps chain target (TAS -317 at the back wall).
const LOBBY_WALL_SPEED: f32 = 300.0;
/// The `tree_handoff` speed (TAS: -420 into the BoB tree).
const TREE_SPEED: f32 = 300.0;
/// The `free_exit` hyperwalk hand-off speed (the TAS chains end at -280 to
/// -560).
const HANDOFF: f32 = 250.0;

pub const SPOTS: &[Spot] = &[
    // Not a slope (ref/blj-conditions.md 3.1): the checkerboard elevator
    // (bhvCheckerboardPlatformSub, rising 10 a frame) by the mountain path.
    // TAS 2026 f22551-22602: idle on it at (1628, 221, 4561), facing -13393,
    // walk ~40, long jump, land on it, then every press lands at once
    // against the wall at x = 1742: -15 -> -420, then over to the summit.
    Spot {
        name: "bob-elevator",
        level: LEVEL_BOB,
        area: 1,
        start: [1591.0, 300.0, 4552.0],
        start_radius: 400.0,
        // King Bob-omb (levels/bob/script.c:48).
        goal: [1636.0, 4242.0, -5567.0],
        goal_radius: 0.0,
        goal_y: Some(4000.0),
        dest: None,
        star: Some((1, 0)),
        rule: Some((-13393, 40.0, None)),
        tech: "blj",
        dest_area: None,
        horizon: 150,
        free_exit: false,
        tree_handoff: true,
    },
    Spot {
        name: "wf-stairs",
        level: LEVEL_WF,
        area: 1,
        start: [4460.0, 256.0, 4097.0],
        start_radius: 700.0,
        goal: [-2616.0, 1331.0, -742.0],
        goal_radius: 500.0,
        goal_y: None,
        dest: None,
        star: Some((2, 2)),
        rule: None,
        tech: "blj",
        dest_area: None,
        horizon: 240,
        free_exit: false,
        tree_handoff: false,
    },
    // The TASes' lobby BLJ (full-120star-2026-U frames 3245-3316): from the
    // floor at the first of the left-hand small steps, facing about +z, a
    // riser BLJ (each jump lands on the 26-high step in the same frame) to
    // about -300, then backwards -z through the lobby's back wall (z -973)
    // to a warp at (-2224, -36, -4477) into CCM area 2, with 1 star. In the
    // lab the overlay's GA got -97 / -158 in 2 of 3 seeds (48 x 20).
    Spot {
        name: "lobby-steps",
        level: LEVEL_CASTLE,
        area: 1,
        start: [-2105.0, 0.0, 501.0],
        start_radius: 400.0,
        goal: [-2240.0, 0.0, -2600.0],
        goal_radius: 1300.0,
        goal_y: None,
        dest: Some(LEVEL_CCM),
        star: None,
        rule: Some((-153, 40.0, Some([-2268.0, 0.0, 320.0]))),
        tech: "blj",
        dest_area: None,
        horizon: 150,
        free_exit: false,
        tree_handoff: false,
    },
    // `bljspots` (TAS catalogue section 3.1, bot-relevant rows; from
    // ref/tas/dumps/full-120star-2026-U.csv: `start` is where the TAS's
    // crouch slide long-jumps, the facing is that jump's, and the riser
    // point is its first landing). Each keeps
    // the TAS's exit goal; the exits need the speed steered (hyperspeed
    // walking, kicks), so after the chain the genome steers (`free_exit`).
    // Lobby centre steps, riser 0/26 (f15680-15717, -317) -> through the
    // WF-room door to the WF painting.
    Spot {
        name: "lobby-centre",
        level: LEVEL_CASTLE,
        area: 1,
        start: [-25.0, 0.0, 431.0],
        start_radius: 400.0,
        goal: [87.0, -60.0, -4770.0],
        goal_radius: 600.0,
        goal_y: None,
        dest: Some(LEVEL_WF),
        star: None,
        rule: Some((407, 40.0, Some([183.0, 0.0, 302.0]))),
        tech: "bljspots",
        dest_area: None,
        horizon: 240,
        free_exit: true,
        tree_handoff: false,
    },
    // Steps between the WF room and the lobby, risers -25/0/77, ceiling 512
    // (f21488-21550, -403), facing +x -> west to the BoB painting.
    Spot {
        name: "wf-room-steps",
        level: LEVEL_CASTLE,
        area: 1,
        start: [501.0, -101.0, -1970.0],
        start_radius: 400.0,
        goal: [-5259.0, 386.0, -482.0],
        goal_radius: 600.0,
        goal_y: None,
        dest: Some(LEVEL_BOB),
        star: None,
        rule: Some((16353, 10.0, Some([538.0, 77.0, -1842.0]))),
        tech: "bljspots",
        dest_area: None,
        horizon: 240,
        free_exit: true,
        tree_handoff: false,
    },
    // Lobby left steps, riser 51/77 under the ceiling 1083 (f14666-14688,
    // -400) -> PSS entrance (crouch-slide wall stop + backflip in the TAS).
    Spot {
        name: "lobby-left-pss",
        level: LEVEL_CASTLE,
        area: 1,
        start: [-2212.0, 0.0, 329.0],
        start_radius: 400.0,
        goal: [1982.0, 768.0, -2095.0],
        goal_radius: 400.0,
        goal_y: None,
        dest: Some(LEVEL_PSS),
        star: None,
        rule: Some((24622, 40.0, Some([-2379.0, 51.0, 285.0]))),
        tech: "bljspots",
        dest_area: None,
        horizon: 240,
        free_exit: true,
        tree_handoff: false,
    },
    // Castle 1F west seam: flat next to a 15° ramp (f11319-11383, -361),
    // facing -z -> PSS via the crouch stop at (1907, 614, -2194).
    Spot {
        name: "castle-seam",
        level: LEVEL_CASTLE,
        area: 1,
        start: [-2555.0, -409.0, -3805.0],
        start_radius: 400.0,
        goal: [1986.0, 768.0, -2091.0],
        goal_radius: 400.0,
        goal_y: None,
        dest: Some(LEVEL_PSS),
        star: None,
        rule: Some((-32432, 40.0, Some([-2687.0, -409.0, -3600.0]))),
        tech: "bljspots",
        dest_area: None,
        horizon: 240,
        free_exit: true,
        tree_handoff: false,
    },
    // BoB steep not-slippery slope 0x15, ny 0.51-0.69 (f25020-25045, -544),
    // facing down it -> the red-coin posts.
    Spot {
        name: "bob-slope",
        level: LEVEL_BOB,
        area: 1,
        start: [-5591.0, 895.0, -5880.0],
        start_radius: 400.0,
        goal: [-5566.0, 768.0, 2928.0],
        goal_radius: 600.0,
        goal_y: None,
        dest: None,
        star: None,
        rule: Some((-31783, 40.0, Some([-5463.0, 848.0, -6015.0]))),
        tech: "bljspots",
        dest_area: None,
        horizon: 240,
        free_exit: true,
        tree_handoff: false,
    },
    // CCM east: blocked quarter step on 0x37 (f4817-4836, -282) -> the
    // cabin chimney warp into the slide (area 2).
    Spot {
        name: "ccm-east",
        level: LEVEL_CCM,
        area: 1,
        start: [3796.0, -818.0, 424.0],
        start_radius: 400.0,
        goal: [-181.0, 2918.0, -1486.0],
        goal_radius: 400.0,
        goal_y: None,
        dest: None,
        star: None,
        rule: Some((-6015, 40.0, None)),
        tech: "bljspots",
        dest_area: Some(2),
        horizon: 240,
        free_exit: true,
        tree_handoff: false,
    },
    // CCM Mr Blizzard plateau: flat 0x14 by a 0x37 rock, position frozen
    // (f8335-8359, -380; f10043-10069, -559) -> the bottom warp pad.
    Spot {
        name: "ccm-plateau",
        level: LEVEL_CCM,
        area: 1,
        start: [-3437.0, -1740.0, 6120.0],
        start_radius: 400.0,
        goal: [3349.0, -4694.0, -183.0],
        goal_radius: 300.0,
        goal_y: None,
        dest: None,
        star: None,
        rule: Some((9870, 40.0, None)),
        tech: "bljspots",
        dest_area: None,
        horizon: 240,
        free_exit: true,
        tree_handoff: false,
    },
];

pub struct Blj {
    spot: &'static Spot,
    /// `Tech::BljRun`: the spot's rule as an overlay on the GA.
    run: bool,
    /// The pole objects of the base (`tree_handoff` with `poleredirect` on).
    poles: Vec<[f32; 3]>,
}

impl Blj {
    /// The hint GA at spots without a rule, the staged overlay GA
    /// ([`staged_overlay`]: setup to the pose, then the rule) at spots with
    /// one.
    pub fn generate(b: &St, out: &mut Vec<std::sync::Arc<dyn Objective>>) {
        for spot in SPOTS.iter().filter(|s| enabled(s.tech)) {
            let poles = if spot.tree_handoff && enabled("poleredirect") { poleredirect::poles(b) } else { Vec::new() };
            let o = Blj { spot, run: spot.rule.is_some(), poles };
            if o.start(b) {
                out.push(std::sync::Arc::new(o));
            }
        }
    }

    fn in_area(&self, s: &St) -> bool {
        s.level as u32 == self.spot.level && s.area == self.spot.area
    }

    fn start(&self, s: &St) -> bool {
        if self.run {
            // Anywhere near the spot (the setup stage walks to the pose),
            // not already moving backwards fast (that's HyperWalk's). The
            // wide radius only at the `blj` spots, whose rules are tuned
            // (lab); the `bljspots` rules don't build speed yet (peaks <= 24,
            // RUNS.md pole row), so they keep their start radius (in 40 M
            // trees with the wide radius they took 2/3 of the BljRun budget).
            let radius = if self.spot.tech == "blj" { SETUP_RADIUS.max(self.spot.start_radius) } else { self.spot.start_radius };
            return spot_open(self.spot, s) && s.controllable() && s.forward_vel > -CHAIN_BUILT && s.dist(self.spot.start) < radius;
        }
        self.in_area(s) && self.spot.star.is_none_or(|(c, i)| !s.has_star(c, i)) && s.dist(self.spot.start) < self.spot.start_radius
    }

    /// The backwards speed whose chain is this spot's success: the tree
    /// hand-off's with PoleRedirect on, the HyperWalk hand-off's at a free
    /// exit with HyperWalk on, else [`CHAIN_SPEED`].
    fn chain_target(&self) -> f32 {
        if self.spot.name == "lobby-steps" {
            // Through the lobby's back wall (z -973) to the CCM painting:
            // the TAS passes at -317; HyperWalk from -210 is stopped by it.
            LOBBY_WALL_SPEED
        } else if !self.poles.is_empty() {
            TREE_SPEED
        } else if self.spot.free_exit && enabled("hyperwalk") {
            HANDOFF
        } else {
            CHAIN_SPEED
        }
    }

    /// The `tree_handoff` success (empty `poles` unless it's on).
    fn at_tree(&self, s: &St) -> bool {
        self.in_area(s)
            && s.alive()
            && self.poles.iter().any(|&p| {
                poleredirect::approaching(s, p, TREE_SPEED) || poleredirect::touched(s, p, TREE_SPEED) || poleredirect::on_pole(s, p, TREE_SPEED)
            })
    }

    fn arrived(&self, s: &St) -> bool {
        if self.spot.dest.is_some_and(|l| s.level as u32 == l) {
            return true;
        }
        if self.spot.dest_area.is_some_and(|a| s.level as u32 == self.spot.level && s.area == a) {
            return true;
        }
        self.in_area(s)
            && s.alive()
            && match self.spot.goal_y {
                // With the tree hand-off on, speed only: no summit credit
                // (it could be walked to; the TAS gets there by redirects).
                Some(y) => self.poles.is_empty() && s.pos[1] >= y,
                None => s.dist(self.spot.goal) < self.spot.goal_radius,
            }
    }
}

impl Objective for Blj {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): technique setups are frame-precise: every state, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn kind(&self) -> &'static str {
        if self.run { "Tech::BljRun" } else { "Tech::Blj" }
    }
    fn key(&self) -> String {
        format!("{} {}", self.kind(), self.spot.name)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        if self.start(b) { 8.0 } else { 0.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        self.start(s)
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        if self.run {
            // The lab's 48 x 20 is about 7 generations at the default 3x
            // scale; the horizon has room for the setup stage too.
            Some(obrrt_core::GaBudget { population: 48, generations: 7, horizon: self.spot.horizon + SETUP_FRAMES })
        } else {
            Some(obrrt_core::GaBudget { population: 24, generations: 6, horizon: self.spot.horizon })
        }
    }
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        // Only through a BLJ chain built in this rollout: the chain at the
        // spot's speed (the hand-off to HyperWalk / PoleRedirect, which
        // spend it), or the spot's place, star or tree once a chain reached
        // CHAIN_BUILT. Walking there is no success.
        let first = is_start(mem);
        let sp = chain_step(s, mem);
        // Not from a base already at the hand-off (that's PoleRedirect's).
        let tree = !self.poles.is_empty() && !self.at_tree(start);
        let target = self.chain_target();
        let slow = start.forward_vel > -target;
        let built = sp >= CHAIN_BUILT;
        let speed_ok = slow && !first && sp > 0.0 && self.in_area(s) && s.alive() && long_jumping(s) && s.forward_vel <= -target;
        let place_ok = built
            && (self.arrived(s)
                || self.spot.star.is_some_and(|(c, k)| s.stars > start.stars && s.has_star(c, k))
                || (!first && tree && self.at_tree(s)));
        if speed_ok || place_ok {
            return Eta::Done;
        }
        if !self.in_area(s) || !s.alive() {
            return Eta::Lost;
        }
        // The chain to the target speed, set up at the pose (run spots),
        // or the start point.
        let setup = pose_point(self.spot).filter(|_| self.run).map_or(self.spot.start, |(p, _)| p);
        let speed = build_eta(s, Some(setup), target);
        // Once built, the place at the chain's speed is the other way in.
        let place = if built && self.spot.goal_y.is_none() {
            ((s.dist(self.spot.goal) - self.spot.goal_radius).max(0.0) / (-s.forward_vel).max(obrrt_core::PACE as f32)) as f64
        } else {
            f64::INFINITY
        };
        Eta::Frames(speed.min(place))
    }
    fn precision(&self) -> Precision {
        Precision::Precise
    }
    fn success_value(&self) -> f64 {
        3.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // With the overlay only the press frames' stick is the genome's;
        // seed it at the riser before the chain, then straight back.
        if let Some((pre, _)) = pose_point(self.spot).filter(|_| self.run && !long_jumping(s)) {
            // The setup stage's genome part (far from the pose).
            if s.hdist(pre) > SETUP_GENOME && s.forward_vel > -CHAIN_BUILT {
                return Some(pre);
            }
        }
        match self.spot.rule {
            // A free exit: the goal once the chain has the speed.
            Some(_) if self.run && self.spot.free_exit && s.forward_vel < -100.0 && !long_jumping(s) => Some(self.spot.goal),
            Some((_, _, Some(riser))) if self.run => Some(steer(s, riser)),
            _ => Some(steer(s, self.spot.goal)),
        }
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        self.spot.rule.filter(|_| self.run)?;
        Some(staged_overlay(self.spot))
    }
    fn steer_button(&self, s: &St) -> u16 {
        button(s)
    }
}

/// The point `run_up` before `start` against `yaw`: where the rule's run-up
/// starts (Mario idle there, facing `yaw`).
fn pose_point(spot: &Spot) -> Option<([f32; 3], i32)> {
    let (yaw, run_up, _) = spot.rule?;
    let r = yaw as f32 * std::f32::consts::TAU / 65536.0;
    Some(([spot.start[0] - run_up * r.sin(), spot.start[1], spot.start[2] - run_up * r.cos()], yaw))
}

/// How close (horizontally) a pose must be to the run-up point, and its
/// facing error.
const POSE_DIST: f32 = 30.0;
const POSE_YAW: u16 = 0x400;
/// Setup bases: within this of the spot's start (and in its area).
const SETUP_RADIUS: f32 = 1500.0;
const ACT_IDLE: u32 = 0x0C40_0201;

fn yaw_err(a: i32, b: i32) -> u16 {
    ((a - b) as i16).unsigned_abs()
}

fn spot_open(spot: &Spot, s: &St) -> bool {
    s.level as u32 == spot.level && s.area == spot.area && spot.star.is_none_or(|(c, i)| !s.has_star(c, i))
}

/// Whether `s` is the start pose of a rule spot (the `Tech::BljRun` base):
/// idle at the run-up point (within [`POSE_DIST`]), facing the rule's yaw
/// (within [`POSE_YAW`]), as the TAS starts it and as the lab's successes
/// started (a moving start gives the rule a different run-up every time).
pub fn at_pose(spot: &Spot, s: &St) -> bool {
    let Some((pre, yaw)) = pose_point(spot) else { return false };
    spot_open(spot, s) && s.controllable() && s.action == ACT_IDLE && s.hdist(pre) < POSE_DIST && yaw_err(s.yaw as i32, yaw) < POSE_YAW
}

/// Setup stage: the genome's pad farther than this from the pose point.
const SETUP_GENOME: f32 = 100.0;

/// The setup stage's pad: the genome's while more than [`SETUP_GENOME`]
/// away (the GA finds the way; the controller seeds steer to the point),
/// then walk in slowly (tilt from 127 at 700 down to 30), stop once within
/// 0.6 [`POSE_DIST`], and from idle a one-frame light tilt along the yaw if
/// the facing is off (from ACT_IDLE the stick sets the facing at once),
/// then stop again; idle too far off, walk in again.
fn setup_pad(pre: [f32; 3], yaw: i32, traj: &[St], genome: GamePad) -> GamePad {
    let s = traj.last().unwrap();
    let d = s.hdist(pre);
    let arrived = traj.iter().any(|t| t.hdist(pre) < POSE_DIST * 0.6);
    if !arrived && d > SETUP_GENOME {
        return genome;
    }
    if s.action == ACT_IDLE && d < POSE_DIST {
        if yaw_err(s.yaw as i32, yaw) < POSE_YAW / 2 {
            return GamePad::new(0, 0, 0);
        }
        let (x, y) = stick_tilt(s, yaw, 24.0);
        return GamePad::new(0, x, y);
    }
    if arrived && s.action & ACT_FLAG_MOVING != 0 {
        return GamePad::new(0, 0, 0);
    }
    if arrived && s.action != ACT_IDLE {
        // Landing, stopping: wait.
        return GamePad::new(0, 0, 0);
    }
    let tilt = (127.0 * d / 700.0).clamp(30.0, 127.0);
    let (x, y) = stick_tilt(s, yaw_to(s.pos, pre), tilt);
    GamePad::new(0, x, y)
}

/// A run spot's overlay in two stages (`Tech::BljRun`): [`setup_pad`]
/// until a state is [`at_pose`] (idle at the run-up point facing the yaw:
/// the TAS's and the lab's start; from a moving start the rule's run-up
/// lands differently every time), then the rule ([`overlay_exit`]) on the
/// trajectory from that state, so a tree state anywhere near the spot is a
/// start (lab, 24 real lobby tree states within 550 of the spot x 2
/// seeds: 15 / 48 chains at -150; the old base and rule: offered at 26,
/// 9 reached -150, 0 successes).
pub fn staged_overlay(spot: &'static Spot) -> PadOverlay {
    let (yaw, run_up, riser) = spot.rule.expect("rule spot");
    let (pre, _) = pose_point(spot).unwrap();
    let rule = overlay_exit(spot.start, yaw, run_up, riser, 1000.0, spot.free_exit);
    PadOverlay::keyed(move |traj, played, genome| match traj.iter().position(|s| at_pose(spot, s)) {
        None => (setup_pad(pre, yaw, traj, genome), false),
        Some(k) => rule.pad_keyed(&traj[k..], &played[k.min(played.len())..], genome),
    })
}


/// The rough start region's radius of a spot (the riser probe's search area).
const GENERIC_RADIUS: f32 = 600.0;

/// The facing whose backwards travel points from the spot's start at its
/// destination (straight-line bearing).
fn away_yaw(spot: &Spot) -> i32 {
    yaw_to(spot.goal, spot.start)
}

/// A step riser for a BLJ in `spot`'s start region, from the zone's static
/// floors ([`obrrt_core::geo::static_floors`]; ref/blj-conditions.md 1.4:
/// a rise of 7.5..32 within a long jump's first quarter steps behind Mario,
/// along the backwards travel from the start to the destination): the low
/// side's point nearest the start. Cached per spot once the floors are
/// known; None before that or if there is none.
pub fn probe_riser(spot: &'static Spot) -> Option<[f32; 3]> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<&'static str, Option<[f32; 3]>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(r) = cache.lock().unwrap().get(spot.name) {
        return *r;
    }
    let tris = obrrt_core::geo::static_floors((spot.level as u16, spot.area))?;
    let b = away_yaw(spot) as f32 * std::f32::consts::TAU / 65536.0;
    // Backwards travel: against the facing.
    let dir = [-b.sin(), -b.cos()];
    let radius = GENERIC_RADIUS.max(spot.start_radius);
    let mut best: Option<(f32, [f32; 3])> = None;
    let n = (radius / 20.0) as i32;
    for i in -n..=n {
        for k in -n..=n {
            let p = [spot.start[0] + 20.0 * i as f32, spot.start[2] + 20.0 * k as f32];
            let d = (p[0] - spot.start[0]).hypot(p[1] - spot.start[2]);
            if d > radius || best.is_some_and(|(bd, _)| d >= bd) {
                continue;
            }
            let Some(h0) = obrrt_core::geo::floor_below(&tris, p[0], p[1], spot.start[1] + 400.0) else { continue };
            let riser = [16.0f32, 24.0, 32.0, 48.0].iter().any(|&r| {
                obrrt_core::geo::floor_below(&tris, p[0] + r * dir[0], p[1] + r * dir[1], h0 + 100.0)
                    .is_some_and(|h1| (7.5..=32.0).contains(&(h1 - h0)))
            });
            if riser {
                best = Some((d, [p[0], h0, p[1]]));
            }
        }
    }
    let r = best.map(|b| b.1);
    if let Some(p) = r {
        // The rise direction: of 16 directions, the one in which the floor
        // steps up soonest (perpendicular to the riser face).
        let mut rise: Option<(f32, i32)> = None;
        for k in 0..16 {
            let yaw = k * 0x1000;
            let a = yaw as f32 * std::f32::consts::TAU / 65536.0;
            for r in (4..=48).step_by(4) {
                let r = r as f32;
                let up = obrrt_core::geo::floor_below(&tris, p[0] + r * a.sin(), p[2] + r * a.cos(), p[1] + 100.0)
                    .is_some_and(|h1| (7.5..=32.0).contains(&(h1 - p[1])));
                if up {
                    if rise.is_none_or(|(br, _)| r < br) {
                        rise = Some((r, yaw));
                    }
                    break;
                }
            }
        }
        if let Some((_, yaw)) = rise {
            RISE.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap().insert(spot.name, yaw);
        }
    }
    cache.lock().unwrap().insert(spot.name, r);
    r
}

/// The probed riser's rise direction (s16 yaw) per spot.
static RISE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<&'static str, i32>>> = std::sync::OnceLock::new();

/// The generic setup's facing: away from the probed riser's rise direction
/// (Mario faces down the steps and BLJs up them), else [`away_yaw`]. (A
/// slope's would be downhill; only risers are probed so far.)
fn setup_facing(spot: &'static Spot) -> i32 {
    probe_riser(spot);
    RISE.get().and_then(|m| m.lock().unwrap().get(spot.name).copied()).map_or_else(|| away_yaw(spot), |rise| rise + 0x8000)
}

/// [`setup_facing`] of the spot named `name` (labs, diagnostics).
pub fn facing_named(name: &str) -> Option<i32> {
    SPOTS.iter().find(|s| s.name == name).map(setup_facing)
}

/// [`probe_riser`] of the spot named `name` (labs, diagnostics).
pub fn probe_named(name: &str) -> Option<[f32; 3]> {
    SPOTS.iter().find(|s| s.name == name).and_then(probe_riser)
}

/// Frames added to a run spot's horizon for the setup stage (GA budget).
const SETUP_FRAMES: usize = 150;
/// Frames added to the spot's horizon for the GA's own setup (GA budget).

/// The preset spots (the BLJ spot registry's `preset` source): name, zone,
/// start, the rule's facing (None: a hint spot), and whether the rule has a
/// riser point (else a rising floor or slope).
pub fn preset_spots() -> Vec<(&'static str, (u16, u8), [f32; 3], Option<i32>, bool)> {
    SPOTS.iter().map(|s| (s.name, (s.level as u16, s.area), s.start, s.rule.map(|r| r.0), s.rule.is_some_and(|r| r.2.is_some()))).collect()
}
