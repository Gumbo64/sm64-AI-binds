//! Fast BLJ setups (`OBRRT_BLJ_FAST=1`, default off: nothing changes), for
//! `Tech::BljAny` (every clause) and the preset `Tech::BljRun` spots.
//!
//! Before (measured with run/examples/bljsetup_lab.rs on the successful
//! BLJs of runs/q6-p-s1, q8-g995-s1, q8-d200-s2): the overlays scripted
//! the approach (walk in slowly to the setup point, stop idle, turn, walk,
//! Z, Z+A), so the genome couldn't make it faster, and the ETA's setup part
//! was a constant (blj::SETUP_FRAMES_ETA = 10, after the walk at PACE), so
//! a faster setup earned nothing under gamma. On the BoB elevator the
//! rollouts rode the platforms (43-122 frames on object floors before the
//! first chain jump) because the ETA had no platform phase.
//!
//! With the switch:
//! - **Approach**: the genome's pad (the GA, seeded by the controller
//!   steering to the spot) until Mario is within [`ENGAGE`] of the spot
//!   (for a platform: standing on it while it rises). Any approach mode
//!   (run, long-jump approach, turnaround, side flip, arriving backwards) is
//!   the GA's.
//! - **Engage frame**: one more genome frame, played as is and marked key;
//!   its pad sets the setup's parameters (explicit, GA-mutated, ranges per
//!   clause in [`Params`]; [`params`]): the facing (the stick's direction,
//!   within +-`yaw_range` of the clause's), the Z point along the facing
//!   (the tilt, in the variant's half of `z_lo..z_hi`) and the highest
//!   launch speed (the same tilt, [`ZMAX`]'s range); the frame plays them
//!   back as its stick ([`engage_pad`]), so from idle it sets the facing.
//! - **Run-up** (the frame-critical part only): stick along the facing line
//!   through the Z point at full tilt (neutral when faster than the launch
//!   speed near it, so he slows), Z (a new press) once facing within
//!   [`FACE_TOL`] at 10.5..zmax inside the Z window; past it, back round.
//! - **Chain**: from the crouch slide, the clause's own chain overlay on the
//!   trajectory since (blj::overlay's Rule, the riser or wall overlay).
//!
//! ETA ([`setup_eta`], replacing the constant): the approach at PACE, the
//! turn from the arrival facing to the BLJ facing at the game's turn rates
//! (freesetup::turn_frames: 0x800 a frame walking, one frame from idle,
//! the skid round), the runway left (to 10.5 along the facing at
//! update_walking_speed's +1.1 - v/43, freesetup::speed_frames, plus Z and
//! A), and for checkerboard platforms the wait
//! until a rising phase that leaves room for the chain ([`platform_wait`]).
//!
//! Variants: the key carries `v<side><bucket>` (approach side relative to
//! the facing: behind / ahead of the spot; the Z-point bucket: the near or
//! far half of `z_lo..z_hi`): at most 4 pool entries per spot.

use obrrt_core::buttons::Z_TRIG;
use obrrt_core::layout::{ACT_FLAG_MOVING, ACT_FLAG_STATIONARY};
use obrrt_core::{GamePad, PadOverlay, St, frames_for};

use super::blj::{self, ACT_CROUCH_SLIDE, long_jumping};

/// OBRRT_BLJ_FAST=1.
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OBRRT_BLJ_FAST").is_ok_and(|v| v == "1"))
}

const ACT_FLAG_AIR: u32 = 0x800;

/// The rule takes over within this of the spot (horizontally; 150 in y).
pub const ENGAGE: f32 = 250.0;
/// The Z window ends this far past the Z point (then back round).
pub const ZPAST: f32 = 150.0;
/// Z only facing within this of the setup facing.
pub const FACE_TOL: u16 = 0x400;
/// The launch speed cap's range (from the engage pad): act_crouch_slide
/// long-jumps above 10; faster crouch-slide long jumps fly further before
/// the stick-back drag turns them (blj::generic_overlay's LAUNCH_FV 15).
pub const ZMAX: [f32; 4] = [11.0, 12.5, 14.5, 17.0];

/// The setup's GA-tunable ranges for a clause.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// Facing offset range (s16 units, +-).
    pub yaw_range: i32,
    /// The Z point's range along the facing from the spot (units).
    pub z_lo: f32,
    pub z_hi: f32,
}

/// Stair and wall risers: ref/blj-conditions.md 6.1 (stair +-0x400), 6.6
/// (wall pose 100 out, the GA's facing nudge); the Z point from 60 before
/// to 20 past the setup point.
pub const RISER: Params = Params { yaw_range: 0x400, z_lo: -60.0, z_hi: 20.0 };
/// Side risers: parallel +-0x100 (6.1), a d0 band of -1..8, so along only.
pub const SIDE: Params = Params { yaw_range: 0x100, z_lo: -60.0, z_hi: 20.0 };
/// Slopes, ceilings, platforms: any facing near the clause's.
pub const OPEN: Params = Params { yaw_range: 0x800, z_lo: -80.0, z_hi: 40.0 };
/// Preset spots (their rule yaw and a 40 run-up: blj.rs SPOTS).
pub const PRESET: Params = Params { yaw_range: 0x400, z_lo: -60.0, z_hi: 20.0 };

/// What the rule waits for before taking over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Board {
    /// Within [`ENGAGE`] of the spot.
    Near,
    /// Standing on this object while it rises.
    Slot(u16),
    /// Standing on any rising object floor within 400 of the spot.
    Rising,
}

fn yaw_err(a: i32, b: i32) -> u16 {
    ((a - b) as i16).unsigned_abs()
}

fn dir(yaw: i32) -> [f32; 2] {
    let r = yaw as f32 * std::f32::consts::TAU / 65536.0;
    [r.sin(), r.cos()]
}

fn yaw_to(a: [f32; 3], b: [f32; 3]) -> i32 {
    ((b[0] - a[0]).atan2(b[2] - a[2]) * 32768.0 / std::f32::consts::PI) as i32
}

/// The approach side of `s` relative to the spot and facing: 0 behind the
/// spot (a run-up along the facing), 1 ahead of it (a turn-round).
pub fn side(s: &St, start: [f32; 3], yaw: i32) -> u8 {
    let d = dir(yaw);
    let along = (s.pos[0] - start[0]) * d[0] + (s.pos[2] - start[2]) * d[1];
    (along > 0.0) as u8
}

/// The variant of a base: side x Z-point bucket (`bucket` 0 / 1).
pub fn variant(b: &St, start: [f32; 3], yaw: i32, bucket: u8) -> u8 {
    side(b, start, yaw) * 2 + (bucket & 1)
}

/// The engage pad's stick magnitudes (raw, after the 6 dead zone) that map
/// to the parameter's 0..1.
const T_LO: f32 = 24.0;
const T_HI: f32 = 80.0;

/// The setup parameters from the engage pad `g` (camera yaw `cam`):
/// (facing, Z point along the facing, launch speed cap). The stick's
/// direction is the facing (clamped to `yaw` +- `yaw_range`), its tilt
/// (T_LO..T_HI) the fraction f: the Z point in the variant's half of
/// z_lo..z_hi and the launch cap 11 + 6 f (a longer run-up, a faster
/// launch: ZMAX's range). [`engage_pad`] writes them back as the stick
/// (a fixed point: the GA mutates the played pad).
pub fn params(p: Params, bucket: u8, yaw: i32, cam: i16, g: GamePad) -> (i32, f32, f32) {
    // adjust_analog_stick: |raw| < 8 is 0, else 6 toward 0.
    let adj = |v: i8| if (v as i32).abs() < 8 { 0.0 } else { v as f32 - 6.0 * (v as f32).signum() };
    let (x, y) = (adj(g.stick_x), adj(g.stick_y));
    let t = x.hypot(y);
    let gy = if t < 1.0 { yaw } else { (x.atan2(-y) * 32768.0 / std::f32::consts::PI) as i32 + cam as i32 };
    let off = ((gy - yaw) as i16 as i32).clamp(-p.yaw_range, p.yaw_range);
    let f = ((t - T_LO) / (T_HI - T_LO)).clamp(0.0, 1.0);
    let mid = 0.5 * (p.z_lo + p.z_hi);
    let (lo, hi) = if bucket & 1 == 0 { (p.z_lo, mid) } else { (mid, p.z_hi) };
    (yaw + off, lo + (hi - lo) * f, ZMAX[0] + (ZMAX[3] - ZMAX[0]) * f)
}

/// The engage frame's pad: the stick along the decoded facing at the
/// decoded tilt (from idle it sets Mario's facing at once), no buttons.
pub fn engage_pad(p: Params, bucket: u8, yaw: i32, cam: i16, g: GamePad) -> GamePad {
    let (face, zs, _) = params(p, bucket, yaw, cam, g);
    let mid = 0.5 * (p.z_lo + p.z_hi);
    let (lo, hi) = if bucket & 1 == 0 { (p.z_lo, mid) } else { (mid, p.z_hi) };
    let f = if hi > lo { (zs - lo) / (hi - lo) } else { 0.0 };
    let t = T_LO + f * (T_HI - T_LO);
    let rel = (face - cam as i32) as f32 * std::f32::consts::TAU / 65536.0;
    let raw = |v: f32| if v.abs() < 0.5 { 0 } else { (v + 6.0 * v.signum()).round().clamp(-127.0, 127.0) as i8 };
    GamePad::new(0, raw(t * rel.sin()), raw(-t * rel.cos()))
}

/// The full-tilt (or `mag` 0..32) raw stick along `yaw` (camera-relative).
fn stick(s: &St, yaw: i32, mag: f32) -> (i8, i8) {
    let rel = (yaw - s.cam_yaw as i32) as f32 * std::f32::consts::TAU / 65536.0;
    let t = if mag >= 32.0 { 80.0 } else { (128.0 * mag.max(0.0)).sqrt() };
    let raw = |v: f32| if v.abs() < 0.5 { 0 } else { (v + 6.0 * v.signum()).round().clamp(-127.0, 127.0) as i8 };
    (raw(t * rel.sin()), raw(-t * rel.cos()))
}

fn engaged(t: &St, start: [f32; 3], board: Board) -> bool {
    let ground = t.action & ACT_FLAG_AIR == 0;
    match board {
        Board::Near => t.hdist(start) < ENGAGE && (t.pos[1] - start[1]).abs() < 150.0,
        Board::Slot(sl) => ground && t.floor_obj == Some(sl) && t.floor_obj_vy.is_some_and(|v| v > 2.0),
        Board::Rising => ground && t.floor_obj.is_some() && t.floor_obj_vy.is_some_and(|v| v > 2.0) && t.hdist(start) < 400.0,
    }
}

/// The fast setup overlay (module doc) around `chain`, the clause's chain
/// overlay, which gets the trajectory from the crouch slide on (or the
/// whole one when the start is already mid-chain).
pub fn overlay(start: [f32; 3], yaw: i32, p: Params, bucket: u8, board: Board, chain: PadOverlay) -> PadOverlay {
    PadOverlay::keyed(move |traj, played, g| {
        if long_jumping(&traj[0]) && traj[0].forward_vel < 0.0 {
            return chain.pad_keyed(traj, played, g);
        }
        let s = traj.last().unwrap();
        let Some(ke) = traj.iter().position(|t| engaged(t, start, board)) else { return (g, false) };
        let n = traj.len() - 1;
        if n == ke {
            // The engage frame (key): the genome's pad sets the parameters,
            // played back as their stick.
            return (engage_pad(p, bucket, yaw, s.cam_yaw, g), true);
        }
        if let Some(kc) = traj[ke + 1..].iter().position(|t| t.action == ACT_CROUCH_SLIDE || long_jumping(t)).map(|k| k + ke + 1) {
            return chain.pad_keyed(&traj[kc..], &played[kc..], g);
        }
        let anchor = if board == Board::Near { start } else { traj[ke].pos };
        let (face, zs, zmax) = params(p, bucket, yaw, traj[ke].cam_yaw, played[ke]);
        let d = dir(face);
        let pz = [anchor[0] + d[0] * zs, anchor[1], anchor[2] + d[1] * zs];
        let along = (s.pos[0] - pz[0]) * d[0] + (s.pos[2] - pz[2]) * d[1];
        if s.action & ACT_FLAG_AIR != 0 {
            // In the air (an approach jump still landing): no buttons
            // (Z would ground-pound), stick toward the line.
            let aim = [pz[0] + d[0] * (along + 80.0), pz[1], pz[2] + d[1] * (along + 80.0)];
            let (x, y) = stick(s, yaw_to(s.pos, aim), 32.0);
            return (GamePad::new(0, x, y), false);
        }
        if s.action & (ACT_FLAG_MOVING | ACT_FLAG_STATIONARY) == 0 {
            return (GamePad::new(0, 0, 0), false);
        }
        let held_z = played.last().is_some_and(|q| q.button & Z_TRIG != 0);
        let lined = yaw_err(s.yaw as i32, face) < FACE_TOL;
        let fv = s.forward_vel;
        // The Z window: from the Z point to ZPAST past it (a start already
        // past the point, e.g. b on the spot, presses there); too fast but
        // half way through, Z anyway (no endless turn-rounds downhill).
        let speed_ok = (10.5..=zmax).contains(&fv) || (fv > zmax && along > ZPAST / 2.0);
        if lined && speed_ok && (-15.0..=ZPAST).contains(&along) && !held_z && s.action & ACT_FLAG_MOVING != 0 {
            let (x, y) = stick(s, face, 32.0);
            return (GamePad::new(Z_TRIG, x, y), true);
        }
        if along > ZPAST {
            // Past the window: back round to before it.
            let back = [pz[0] - d[0] * 120.0, pz[1], pz[2] - d[1] * 120.0];
            let (x, y) = stick(s, yaw_to(s.pos, back), 32.0);
            return (GamePad::new(0, x, y), false);
        }
        // Along the line through the Z point; neutral (slow down) when
        // too fast close to it.
        let aim = [pz[0] + d[0] * (along + 80.0), pz[1], pz[2] + d[1] * (along + 80.0)];
        let mag = if fv > zmax && along > -4.0 * fv { 0.0 } else { 32.0 };
        let (x, y) = if mag == 0.0 { (0, 0) } else { stick(s, yaw_to(s.pos, aim), mag) };
        (GamePad::new(0, x, y), false)
    })
}

/// The setup's frames before the first long jump (module doc): approach
/// at PACE, turn, runway; at least the platform wait.
pub fn setup_eta(s: &St, start: [f32; 3], yaw: i32, wait: Option<(f64, f64)>) -> f64 {
    use super::freesetup::{WALK_TURN, speed_frames, turn_frames, yaw_diff};
    let d = s.hdist(start);
    let approach = frames_for(d);
    let near = d < ENGAGE;
    // The turn onto the BLJ facing at the game's rates (freesetup): from
    // the current state when near, else from the arrival facing.
    let turn = if near { turn_frames(s, yaw) } else { yaw_diff(yaw_to(s.pos, start), yaw) as f64 / WALK_TURN };
    let along_v = if near { (s.forward_vel * (yaw_err(s.yaw as i32, yaw) as f32 * std::f32::consts::TAU / 65536.0).cos()).max(0.0) } else { 0.0 };
    // The runway: to 10.5 at update_walking_speed's rate, then Z and A.
    let runway = speed_frames(along_v, 10.5) + 2.0;
    let a = approach + turn + runway;
    match wait {
        Some((w, window)) => chain_start(a, w, window),
        None => a,
    }
}

/// The checkerboard cycle (checkerboard_platform.inc.c, default params:
/// rise 65 frames at 10, rotate 0x8000 / 512 = 64, fall 65, rotate 64).
pub const RISE: f64 = 65.0;
const ROTATE: f64 = 64.0;
pub const CYCLE: f64 = 2.0 * (RISE + ROTATE);
/// Rising frames a chain needs after boarding (the setup to the first long
/// jump, its flight, and the re-jumps to the hand-off).
pub const NEED: f64 = 12.0 + blj::FIRST_FLIGHT_FRAMES + 12.0;

/// When the setup can start the chain, arriving after `a` frames, with
/// the platform's next rise in `w` frames leaving `window` usable frames:
/// at once if it is still in the window, else the next cycle's.
pub fn chain_start(a: f64, w: f64, window: f64) -> f64 {
    if a <= w + window.max(0.0) { a.max(w) } else { a.max(w + CYCLE) }
}

/// A checkerboard platform's phase: (frames until its next usable rise,
/// usable rising frames from then), from its action, position and home
/// (bhvCheckerboardPlatformSub; behParams 2nd byte 0 starts at the bottom,
/// 1 at the top, 650 above). `slot`: that object, else the platform
/// nearest `near`. None: no such object in `s` (objects not read).
pub fn platform_wait(s: &St, slot: Option<u16>, near: [f32; 3]) -> Option<(f64, f64)> {
    let objs = s.objects();
    let o = match slot {
        Some(sl) => objs.iter().find(|o| o.slot == sl),
        None => objs
            .iter()
            .filter(|o| o.bhv_name() == "bhvCheckerboardPlatformSub")
            .min_by(|a, b| obrrt_core::dist(a.pos, near).total_cmp(&obrrt_core::dist(b.pos, near))),
    }?;
    if o.bhv_name() != "bhvCheckerboardPlatformSub" {
        return None;
    }
    let i = ((o.beh_params >> 16) & 0xFF) as f32;
    let bottom = o.home[1] - 650.0 * i;
    let top = bottom + 650.0;
    let y = o.pos[1] as f64;
    let window = RISE - NEED;
    Some(match o.action {
        // Rising: usable while enough of the rise is left.
        1 => {
            let left = ((top as f64 - y) / 10.0).clamp(0.0, RISE);
            if left >= NEED { (0.0, left - NEED) } else { (left + 2.0 * ROTATE + RISE, window) }
        }
        // Rotating at the top (progress unknown: half), then the fall.
        2 => (ROTATE / 2.0 + RISE + ROTATE, window),
        3 => (((y - bottom as f64) / 10.0).clamp(0.0, RISE) + ROTATE, window),
        4 => (ROTATE / 2.0, window),
        _ => (0.0, window),
    })
}

/// [`blj::build_eta`] with [`setup_eta`] for the setup (the chain terms
/// unchanged).
pub fn build_eta(s: &St, setup: [f32; 3], yaw: i32, target: f32, wait: Option<(f64, f64)>) -> f64 {
    if long_jumping(s) || s.action == ACT_CROUCH_SLIDE {
        return blj::build_eta(s, None, target);
    }
    setup_eta(s, setup, yaw, wait) + blj::FIRST_FLIGHT_FRAMES + blj::chain_frames(0.0, target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::testkit::st;
    use obrrt_core::layout::LEVEL_CASTLE;

    #[test]
    fn setup_eta_falls_as_the_setup_progresses() {
        let spot = [0.0, 0.0, 0.0];
        let yaw = 0; // facing +z
        let at = |p: [f32; 3], yaw_now: i16, fv: f32| {
            let mut s = st(LEVEL_CASTLE, p, vec![]);
            s.yaw = yaw_now;
            s.forward_vel = fv;
            s.action = 0x0400_0440;
            s
        };
        // Far, coming from behind (travel along the facing): no turn.
        let far = setup_eta(&at([0.0, 0.0, -1500.0], 0, 20.0), spot, yaw, None);
        let mid = setup_eta(&at([0.0, 0.0, -600.0], 0, 20.0), spot, yaw, None);
        assert!(mid < far);
        // Far, coming from ahead (a turn-round on arrival) costs more.
        let ahead = setup_eta(&at([0.0, 0.0, 1500.0], -0x8000, 20.0), spot, yaw, None);
        assert!(ahead > far);
        // Near: facing the yaw at speed beats facing away, beats standing.
        let lined = setup_eta(&at([0.0, 0.0, -100.0], 0, 12.0), spot, yaw, None);
        let away = setup_eta(&at([0.0, 0.0, -100.0], 0x6000, 12.0), spot, yaw, None);
        let idle = setup_eta(&at([0.0, 0.0, -100.0], 0x6000, 0.0), spot, yaw, None);
        assert!(lined < away && lined < idle, "{lined} {away} {idle}");
        assert!(idle < away, "from idle the turn is one frame: {idle} {away}");
    }

    #[test]
    fn platform_phase_waits_for_a_usable_rise() {
        // Arriving in 30 frames: a rise starting in 10 with 20 usable frames
        // is caught (start at 30); arriving in 40 misses it (next cycle).
        assert_eq!(chain_start(30.0, 10.0, 20.0), 30.0);
        assert_eq!(chain_start(5.0, 10.0, 20.0), 10.0);
        assert_eq!(chain_start(40.0, 10.0, 20.0), 10.0 + CYCLE);
        let bob = |action: i32, y: f32, i: u32| {
            let mut o = crate::extra::tech::testkit::obj("bhvCheckerboardPlatformSub", 5, [0.0, y, 0.0], action, 0);
            o.home = [0.0, 650.0 * i as f32, 0.0];
            o.beh_params = i << 16;
            st(LEVEL_CASTLE, [0.0; 3], vec![o])
        };
        // Rising near the bottom: usable now.
        let (w, win) = platform_wait(&bob(1, 50.0, 0), Some(5), [0.0; 3]).unwrap();
        assert_eq!(w, 0.0);
        assert!(win > 0.0);
        // Rising near the top: the next rise (after rotate, fall, rotate).
        let (w2, _) = platform_wait(&bob(1, 600.0, 0), Some(5), [0.0; 3]).unwrap();
        assert!(w2 > 2.0 * ROTATE + RISE);
        // Falling half way (the 2nd platform, home at the top): half the
        // fall plus a rotation.
        let (w3, _) = platform_wait(&bob(3, 325.0, 1), Some(5), [0.0; 3]).unwrap();
        assert!((w3 - (32.5 + ROTATE)).abs() < 1e-6, "{w3}");
    }

    #[test]
    fn params_cover_the_ranges_and_round_trip() {
        let g = |x: i8, y: i8| GamePad::new(0, x, y);
        // Camera yaw 0: stick (x, -y) points at yaw atan2(x, -y)... a
        // stick straight "up" (y > 0) is yaw 0x8000 relative.
        let (f, z, m) = params(RISER, 0, 0x1000, 0, g(0, 0));
        assert_eq!((f, z, m), (0x1000, RISER.z_lo, ZMAX[0]));
        // Far off the facing: clamped to the range; full tilt: the top.
        let (f, z, m) = params(RISER, 1, 0, 0, g(0, 127));
        assert_eq!(f.abs(), RISER.yaw_range);
        assert_eq!(z, RISER.z_hi);
        assert_eq!(m, ZMAX[3]);
        // The engage pad decodes to the same parameters.
        for (x, y, cam) in [(40i8, -50i8, 0x1234i16), (-90, 20, -0x3000), (10, -100, 0x7000)] {
            for bucket in 0..2 {
                let a = params(OPEN, bucket, 0x2000, cam, g(x, y));
                let b = params(OPEN, bucket, 0x2000, cam, engage_pad(OPEN, bucket, 0x2000, cam, g(x, y)));
                assert!((a.0 - b.0).abs() < 0x80 && (a.1 - b.1).abs() < 2.0 && (a.2 - b.2).abs() < 0.2, "{a:?} {b:?}");
            }
        }
    }
}
