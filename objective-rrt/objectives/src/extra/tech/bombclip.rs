//! `Tech::BombClip` (`bombclip`): a bloated bob-omb shoves Mario through a
//! thin wall. `Bobomb::GateClip` (extra/enemies/clip.rs, BoB star 6) is an
//! instance of it: the same rule overlay and progress, with the Chomp gate
//! as the wall.
//!
//! **Mechanism** (sm64-port decomp, checked against the 2026 TAS, f21955 to
//! f22060; trace in the bombclip lab):
//! - A bob-omb is INTERACT_GRABBABLE. When Mario touches one he can't grab,
//!   interact_grabbable calls push_mario_out_of_object(m, o, -5)
//!   (interaction.c): Mario is put at `o.hitboxRadius + 37 - 5` from its
//!   centre, away from it; walls are checked once at the new spot
//!   (f32_find_wall_collision, offset 60, radius 50), and the move is kept
//!   if find_floor finds a floor there (or no floor: Mario is then out of
//!   bounds, mario.c queues WARP_OP_DEATH). A step that ends more than 50
//!   past a wall's plane isn't pushed back by it.
//! - A thrown bob-omb lands in BOBOMB_ACT_EXPLODE (bobomb_act_launched) and
//!   grows for 5 frames (cur_obj_scale(1 + oTimer / 5), bobomb_act_explode);
//!   obj_set_hitbox scales its radius 65 -> 117. TAS: a3 t0 r65 at f22002,
//!   r117 at t5 (f22007). Regrabbed in that window (a punch, f22003), it is
//!   held in its explode action, still at 117: bobomb_act_explode only runs
//!   while it's free, and the held loop drops it only once the fuse reaches
//!   151 (bobomb_held_loop). Held 100 in front of Mario, it pushes him
//!   149 - 100 = 49 back every frame (TAS f22008-22058: ~49 a frame at
//!   fv 0 in ACT_PICKING_UP, then through hold jumps).
//! - Released at the hand (mario_throw_held_object: HOLP + 32 along the
//!   facing), it can be much nearer than 100: TAS f22059, an air-throw
//!   landing puts it 40 from Mario, and the next frame's push is 108, into
//!   the cage (no floor there, so out of bounds: the death-fade star).
//!
//! **Objective.**
//! - Base: Mario at a wall (`wall_normal`, near-vertical) holding a bob-omb,
//!   or with one within 600 and his hands free, and a floor known on the
//!   wall's far side 150-300 behind its plane within 150 of his height.
//!   Objectives can't query geometry (core::geo::floor_at takes game memory;
//!   St has none), so "known" means seen: a per-zone map of 100-unit cells
//!   of floors that states passed to `generate` stood on (filled only while
//!   `bombclip` is on, from the engine's single generating thread, so
//!   deterministic). A wall whose far side was never stood on isn't offered
//!   (the gate is the exception: GateClip names its far side). One per state,
//!   keyed by 400-unit cell of the wall point and wall-facing octant.
//! - Target: 200 behind the wall plane (straight line).
//! - Inputs: [`overlay`], a rule on the GA's pads by phase (read from the
//!   state and the trajectory's objects); "the wall point" is the wall's
//!   point while Mario is 300+ from it, its normal nearer:
//!   1. fetch: walk to the nearest bob-omb, B punch-grab facing it (short
//!      stick: B at 29+ with the stick past 48 dives instead);
//!   2. holding a plain one: face the wall point, hold-jump towards it (A
//!      held, stick forward) and air-throw, B forced on the 2nd air frame,
//!      as the TAS does (f21972-21978): the throw's push sends Mario on
//!      ahead and he lands with it behind him (release window);
//!   3. regrab: stick forward while airborne, then to it, punch-grab while
//!      it bloats, B forced once it's in its explode action within 150
//!      (regrab window); regrabbed, Mario faces away from the wall;
//!   4. carry it bloated: face away from the wall point so the push takes
//!      him back to it, in back-to-back hold jumps with the stick towards
//!      the wall (TAS f22019-22055; standing, slopes slide him); the
//!      genome's stick x past 40 turns that line up to 90 degrees;
//!      within 200 of the plane, land, stand with the back to the wall
//!      (the push holds him against it) and throw from the hand, B from the
//!      genome or forced after 6 standing frames (release window);
//!   5. after that final release, the genome for 40 frames, then nothing.
//!   The windows' frames are the key frames: there the genome's B picks
//!   the press frame and its stick x turns the rule's stick (up to 45
//!   degrees), so the GA tunes the release and regrab frames and sticks.
//!   Lab (bombclip lab, GateClip, 2 GA attempts each): from TAS f22008
//!   (regrabbed, 2300 from the gate) 2/2 into the cage with the overlay
//!   (carry 49 a frame to the bars, ground throw at the hand, push 130),
//!   0/2 without; from f21955 (before the grab) 0/2 (the carry slides on
//!   the slope before the gate).
//! - Progress (signature-gated, [`progress`]): 0 until a push frame: Mario
//!   moved 20+ more than his speed explains with a bloated bob-omb (explode
//!   action, or launched within 8 frames of it) within 160 of him. Then
//!   1000 + the ground gained on push frames (towards the wall point while
//!   300+ away, along -wall_normal nearer) summed over push frames only (clamped to -900..4000; walking between or after pushes earns
//!   nothing), - 0.5 x the push frame's distance from the wall point's
//!   normal line, +500 once a push frame ends past the plane near that line.
//! - Success: Mario past the plane by 60+ (lateral 150, height within 300
//!   of the far floor), not holding, a bloated bob-omb or its explosion
//!   within 500 (the push's trace: the GA checks success on the first and
//!   last states only, so the push frame itself can't be required), and
//!   standing on a floor alive, or a star gained (death fade).
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use obrrt_core::buttons::{A_BUTTON, B_BUTTON};
use obrrt_core::{Budget, GamePad, Obj, Objective, PadOverlay, Precision, Progress, St};

const BOBOMB: &str = "bhvBobomb";
const EXPLOSION: &str = "bhvExplosion";
/// BOBOMB_ACT_LAUNCHED, BOBOMB_ACT_EXPLODE (include/object_constants.h).
const ACT_LAUNCHED: i32 = 1;
const ACT_EXPLODE: i32 = 3;
/// ACT_FLAG_AIR, ACT_FLAG_MOVING (include/sm64.h).
const ACT_FLAG_AIR: u32 = 0x0000_0800;
const ACT_FLAG_MOVING: u32 = 0x0000_0400;
const ACT_FLAG_BUTT_OR_STOMACH_SLIDE: u32 = 0x0004_0000;
/// ACT_HOLD_IDLE, ACT_HOLD_WALKING (include/sm64.h).
const ACT_HOLD_IDLE: u32 = 0x0800_0207;
const ACT_HOLD_WALKING: u32 = 0x0000_0442;

/// Mario's hitbox 37 + a bloated bob-omb's 117 - 5 = 149, and some slack.
const PUSH_REACH: f32 = 160.0;
/// A push frame moves Mario this much more than his speed does.
const PUSH_EXCESS: f32 = 20.0;
/// Frames a thrown bob-omb counts as bloated after its explode action.
const BLOAT_MEMORY: usize = 8;
/// Mario's wall radius: his centre is this far from a wall he touches.
const WALL_RADIUS: f32 = 50.0;
/// Base states are within this of the wall point (depth and lateral).
const NEAR_WALL: f32 = 300.0;
/// A bob-omb this near Mario (hands free) is within reach.
const FETCH: f32 = 600.0;
/// Holding a bloated bob-omb nearer than this to the plane, the release
/// window is open.
const RELEASE_DIST: f32 = 200.0;
/// Hold-jump air frames before the rule air-throws itself.
const AIR_THROW_FORCE: usize = 1;
/// Window frames before the rule releases a bloated one itself.
const RELEASE_FORCE: usize = 6;
/// Frames of genome after the final release, of them key.
const POST_FREE: usize = 40;
const POST_KEY: usize = 10;
/// Progress after the first push, and past the plane.
const PUSHED: f64 = 1000.0;
const PAST: f64 = 500.0;
/// The far side's floor is searched this far behind the plane.
const FAR: [f32; 4] = [150.0, 200.0, 250.0, 300.0];
/// Height tolerance of the far floor.
const FAR_DY: f32 = 150.0;
/// Success is this near the normal line through the wall point (the push
/// goes straight through; farther out is round the wall's end).
const SUCCESS_LATERAL: f32 = 150.0;
/// The target's depth behind the plane.
const TARGET_DEPTH: f32 = 200.0;
/// Learned floor map: cell size, heights per cell, cells in all.
const FCELL: f32 = 100.0;
const FHEIGHTS: usize = 4;
const FMAX: usize = 1 << 20;

/// A wall plane: `p` on it, `n` the horizontal unit normal pointing out of
/// it, to the side Mario starts on.
#[derive(Debug, Clone, Copy)]
pub struct Wall {
    pub p: [f32; 3],
    pub n: [f32; 2],
}

impl Wall {
    /// How far `q` is behind the plane (negative in front).
    pub fn depth(&self, q: [f32; 3]) -> f32 {
        (self.p[0] - q[0]) * self.n[0] + (self.p[2] - q[2]) * self.n[1]
    }

    /// Distance of `q` from the plane's normal line through `p`.
    pub fn lateral(&self, q: [f32; 3]) -> f32 {
        ((q[0] - self.p[0]) * self.n[1] - (q[2] - self.p[2]) * self.n[0]).abs()
    }

    /// The point `d` in front of `p` (behind for negative `d`).
    pub fn front(&self, d: f32) -> [f32; 3] {
        [self.p[0] + d * self.n[0], self.p[1], self.p[2] + d * self.n[1]]
    }

    /// The yaw that faces away from the wall (along `n`).
    fn away_yaw(&self) -> i32 {
        (self.n[0].atan2(self.n[1]) * 32768.0 / std::f32::consts::PI) as i32
    }
}

fn hdist(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).hypot(a[2] - b[2])
}

pub fn is_bobomb(o: &Obj) -> bool {
    o.bhv_name() == BOBOMB
}

/// A bob-omb setup with most of its fuse left: held but not yet bloated,
/// or a fetchable one within 600 (patrolling, fuse unlit, or chasing, just
/// lit; the fuse burns 151 frames from then, bobomb.inc.c). Starts like this
/// can do the whole clip, so they get more weight.
pub fn fresh_setup(s: &St) -> bool {
    match held_bobomb(s) {
        Some(o) => o.action != ACT_EXPLODE,
        None => !s.holding
            && nearest_bobomb(s).is_some_and(|o| s.hdist(o.pos) < FETCH && (o.pos[1] - s.pos[1]).abs() < 150.0 && fetchable(o) && o.action != ACT_LAUNCHED),
    }
}

/// A free bob-omb a start can use: not already exploding (a free one in
/// its explode action is gone within 5 frames; a start next to one is
/// knocked back, not set up).
pub fn fetchable(o: &Obj) -> bool {
    is_bobomb(o) && o.action != ACT_EXPLODE
}

fn bobomb_at(s: &St, slot: u16) -> Option<&Obj> {
    s.objects().iter().find(|o| o.slot == slot && is_bobomb(o))
}

fn held_bobomb(s: &St) -> Option<&Obj> {
    s.held().filter(|o| is_bobomb(o))
}

fn nearest_bobomb(s: &St) -> Option<&Obj> {
    s.objects().iter().filter(|o| is_bobomb(o)).min_by(|a, b| s.hdist(a.pos).total_cmp(&s.hdist(b.pos)))
}

/// Whether bob-omb `slot` is bloated at `traj[i]`: in its explode action
/// (held or free), or launched within [`BLOAT_MEMORY`] frames of it (a
/// throw keeps its scale).
fn bloated(traj: &[St], i: usize, slot: u16) -> bool {
    match bobomb_at(&traj[i], slot) {
        Some(o) if o.action == ACT_EXPLODE => true,
        Some(o) if o.action == ACT_LAUNCHED => {
            traj[..i].iter().rev().take(BLOAT_MEMORY).any(|t| bobomb_at(t, slot).is_some_and(|o| o.action == ACT_EXPLODE))
        }
        _ => false,
    }
}

/// Whether frame `i` of `traj` is a push frame: Mario moved [`PUSH_EXCESS`]
/// more than his speed explains, with a bloated bob-omb within
/// [`PUSH_REACH`] before or after.
pub fn push_at(traj: &[St], i: usize) -> bool {
    let (a, s) = (&traj[i - 1], &traj[i]);
    if a.zone() != s.zone() {
        return false;
    }
    let moved = hdist(a.pos, s.pos);
    if moved < s.vel[0].hypot(s.vel[2]) + PUSH_EXCESS || moved > 400.0 {
        return false;
    }
    [i - 1, i].into_iter().any(|k| {
        let t = &traj[k];
        t.objects().iter().any(|o| is_bobomb(o) && hdist(o.pos, t.pos) < PUSH_REACH && bloated(traj, k, o.slot))
    })
}

/// The number of push frames in `traj` (for labs and audits).
pub fn push_count(traj: &[St]) -> usize {
    (1..traj.len()).filter(|&i| push_at(traj, i)).count()
}

/// Signature-gated progress through `wall` (module doc): 0 before the
/// first push frame; then [`PUSHED`] plus the ground gained on push frames
/// only (walking after a push earns nothing): towards the wall point while
/// 300+ from it, through the plane nearer; minus half the distance from
/// the wall point's normal line at the push frame, and [`PAST`] once a push
/// frame ends past the plane within `past_lateral` of that line.
pub fn progress(wall: &Wall, zone: (u16, u8), past_lateral: f32, traj: &[St]) -> Progress {
    let mut top = Progress { score: 0.0, at: 0 };
    let (mut gain, mut past) = (0.0f32, false);
    for i in 1..traj.len() {
        let s = &traj[i];
        if s.zone() != zone {
            break;
        }
        if !push_at(traj, i) {
            continue;
        }
        let d = wall.depth(s.pos);
        let a = &traj[i - 1];
        // Towards the wall point while far from it (straight line), then
        // through the plane.
        gain += if a.hdist(wall.p) > NEAR_WALL { a.hdist(wall.p) - s.hdist(wall.p) } else { d - wall.depth(a.pos) };
        let lateral = wall.lateral(s.pos);
        past |= d > 0.0 && lateral < past_lateral;
        let score = PUSHED + gain.clamp(-900.0, 4000.0) as f64 - 0.5 * lateral.min(2000.0) as f64 + if past { PAST } else { 0.0 };
        if score > top.score {
            top = Progress { score, at: i };
        }
    }
    top
}

/// A bloated bob-omb (explode or launched) or an explosion within 500 of
/// Mario, Mario not holding it: the push's trace in a single state.
pub fn push_trace(s: &St) -> bool {
    !s.holding
        && s.objects().iter().any(|o| {
            (o.bhv_name() == EXPLOSION || (is_bobomb(o) && matches!(o.action, ACT_LAUNCHED | ACT_EXPLODE))) && s.hdist(o.pos) < 500.0
        })
}

/// SM64 yaw of the direction from `a` to `b`.
fn yaw_to(a: [f32; 3], b: [f32; 3]) -> i32 {
    ((b[0] - a[0]).atan2(b[2] - a[2]) * 32768.0 / std::f32::consts::PI) as i32
}

/// The raw stick that points intendedYaw along `yaw` at `tilt`
/// (camera-relative, with adjust_analog_stick's 6 added back; as blj.rs).
fn stick_tilt(s: &St, yaw: i32, tilt: f32) -> (i8, i8) {
    let rel = (yaw - s.cam_yaw as i32) as f32 * std::f32::consts::TAU / 65536.0;
    let t = tilt - 6.0;
    let raw = |v: f32| if v.abs() < 0.5 { 0 } else { (v + 6.0 * v.signum()).round().clamp(-127.0, 127.0) as i8 };
    (raw(t * rel.sin()), raw(-t * rel.cos()))
}

/// The genome's stick x past a dead zone of 40 as a turn of up to 90
/// degrees (the carry line).
fn line_turn(g: GamePad) -> i32 {
    let x = g.stick_x as i32;
    if x.abs() < 40 { 0 } else { (x - 40 * x.signum()) * 0x4000 / 87 }
}

/// The genome's stick x as a turn of up to 45 degrees (key-frame tuning).
fn turn(g: GamePad) -> i32 {
    g.stick_x as i32 * 0x2000 / 127
}

fn yaw_err(s: &St, yaw: i32) -> u16 {
    ((s.yaw as i32 - yaw) as i16).unsigned_abs()
}

/// Where the GA's seeded genomes steer, by phase.
pub fn steer_target(wall: &Wall, s: &St) -> Option<[f32; 3]> {
    match held_bobomb(s) {
        Some(o) if o.action == ACT_EXPLODE => Some(wall.front(-TARGET_DEPTH)),
        Some(_) => Some(wall.p),
        None => nearest_bobomb(s).filter(|o| s.hdist(o.pos) < 2.0 * FETCH).map(|o| o.pos).or(Some(wall.p)),
    }
}

/// The bomb-clip rule as a keyed overlay on the GA's pads (module doc).
pub fn overlay(wall: Wall) -> PadOverlay {
    PadOverlay::keyed(move |traj, played, genome| rule(&wall, traj, played, genome))
}

fn rule(w: &Wall, traj: &[St], played: &[GamePad], g: GamePad) -> (GamePad, bool) {
    let s = traj.last().unwrap();
    if !s.controllable() {
        return (g, false);
    }
    let b_last = played.last().is_some_and(|p| p.button & B_BUTTON != 0);
    let b_recent = played.iter().rev().take(4).any(|p| p.button & B_BUTTON != 0);
    let g_b = g.button & B_BUTTON != 0 && !b_last;
    let neutral = GamePad::new(0, 0, 0);
    // Away from the wall: from its point while far, along its normal near it.
    let away = if s.hdist(w.p) > NEAR_WALL { yaw_to(w.p, s.pos) } else { w.away_yaw() };
    if let Some(h) = held_bobomb(s) {
        if h.action == ACT_EXPLODE {
            // Carrying it bloated: it pushes Mario ~49 a frame backwards.
            // Far from the wall: face away from its point, so the push takes
            // Mario to it; the genome's A may hold-jump on the way (the TAS
            // hold-jumps up the slope while pushed, f22020-22050).
            if w.depth(s.pos) < -RELEASE_DIST {
                let air = s.action & ACT_FLAG_AIR != 0;
                // The carry line: the genome's stick x turns the facing (so
                // the push) up to 90 degrees off the wall point, on the
                // ground between hold jumps (key frames): the GA picks the
                // way round a steep slope (the TAS's line runs ~150 west of
                // the straight one, f22019-22055).
                let line = away + line_turn(g);
                if !air && s.action & ACT_FLAG_BUTT_OR_STOMACH_SLIDE == 0 && yaw_err(s, line) > 0x1000 {
                    let (x, y) = stick_tilt(s, line, 40.0);
                    return (GamePad::new(0, x, y), true);
                }
                // Hold jumps all the way (the TAS, f22019-22055): a new A on
                // the ground (it comes before the stick in hold idle, so the
                // facing stays), A held while rising, the stick back towards
                // the wall in the air (no turning there: it adds backward
                // speed to the push). It keeps Mario off slopes that would
                // slide him. The genome's stick x turns the air stick.
                let a_last = played.last().is_some_and(|p| p.button & A_BUTTON != 0);
                if !air && s.action & ACT_FLAG_BUTT_OR_STOMACH_SLIDE != 0 {
                    // Slid by a steep floor (ACT_HOLD_BUTT_SLIDE): the stick
                    // towards the wall (sliding there, as the TAS does, not
                    // back down the slope), and A jumps once the slide allows it
                    // (actionTimer 5, common_slide_action_with_jump); TAS
                    // f22031-22036.
                    let (x, y) = stick_tilt(s, line.wrapping_add(0x8000), 127.0);
                    let a = s.action_timer >= 5 && !a_last;
                    return (GamePad::new(if a { A_BUTTON } else { 0 }, x, y), true);
                }
                if !air {
                    return (GamePad::new(if a_last { 0 } else { A_BUTTON }, 0, 0), true);
                }
                let (x, y) = stick_tilt(s, line.wrapping_add(0x8000), 127.0);
                return (GamePad::new(if s.vel[1] > 0.0 { A_BUTTON } else { 0 }, x, y), true);
            }
            // Near the wall: land (a hold jump here bounces off the wall
            // sideways), stand with the back to it while the push holds
            // Mario against it, and throw from the hand (the push from the
            // HOLP, ~130 in the lab at the gate). The genome picks the
            // throw frame (B), forced after a few standing frames.
            if s.action & ACT_FLAG_AIR != 0 {
                let (x, y) = stick_tilt(s, away.wrapping_add(0x8000), 127.0);
                return (GamePad::new(0, x, y), false);
            }
            // Square to the wall (the throw's push is straight back from
            // the facing; oblique, it slides along the wall instead).
            if yaw_err(s, away) > 0x800 && s.action & ACT_FLAG_MOVING == 0 {
                let (x, y) = stick_tilt(s, away, 40.0);
                return (GamePad::new(0, x, y), false);
            }
            let open = traj
                .iter()
                .rev()
                .take_while(|t| held_bobomb(t).is_some_and(|o| o.action == ACT_EXPLODE) && w.depth(t.pos) >= -RELEASE_DIST && t.action & ACT_FLAG_AIR == 0)
                .count();
            let throw = g_b || (open > RELEASE_FORCE && !b_last);
            return (GamePad::new(if throw { B_BUTTON } else { 0 }, 0, 0), true);
        }
        // Holding a plain one: face the wall point and throw it that way
        // from a hold jump (the TAS's f21972-21978, yaw -27810 towards the
        // gate): the throw's push sends Mario on ahead of it, he lands with
        // it behind him, turns and regrabs it, now facing away from the
        // wall, so the carry's push takes him back to it.
        let toward = away.wrapping_add(0x8000);
        if yaw_err(s, toward) > 0x1800 {
            if s.action & ACT_FLAG_MOVING != 0 {
                return (neutral, false);
            }
            // From hold idle the stick sets the facing at once.
            let (x, y) = stick_tilt(s, toward, 127.0);
            return (GamePad::new(0, x, y), false);
        }
        // A hold jump with the stick forward, A held (full height, as the
        // TAS holds A+B), and an air throw. On the window's frames the
        // genome picks the throw frame (B) and turns the stick (x).
        let stick = stick_tilt(s, toward + turn(g), 127.0);
        if s.action & ACT_FLAG_AIR == 0 {
            // Jump only from hold idle or a hold walk (A in other landing
            // or moving actions can drop it); else wait.
            let a_last = played.last().is_some_and(|p| p.button & A_BUTTON != 0);
            if s.action & ACT_FLAG_BUTT_OR_STOMACH_SLIDE != 0 {
                // Out of a hold slide by its jump (actionTimer 5).
                let a = s.action_timer >= 5 && !a_last;
                return (GamePad::new(if a { A_BUTTON } else { 0 }, stick.0, stick.1), false);
            }
            if !matches!(s.action, ACT_HOLD_IDLE | ACT_HOLD_WALKING) {
                return (neutral, false);
            }
            return (GamePad::new(if a_last { 0 } else { A_BUTTON }, stick.0, stick.1), true);
        }
        let air = traj.iter().rev().take_while(|t| t.action & ACT_FLAG_AIR != 0 && held_bobomb(t).is_some()).count();
        // A held keeps the jump full height (as the TAS holds A+B).
        let throw = g_b || (air > AIR_THROW_FORCE && !b_last);
        return (GamePad::new(A_BUTTON | if throw { B_BUTTON } else { 0 }, stick.0, stick.1), true);
    }
    if s.holding {
        return (g, false);
    }
    // Hands free. After the final release (a throw of a bloated one), the
    // genome for a while, then nothing.
    if let Some(j) = traj.iter().rposition(|t| held_bobomb(t).is_some()) {
        let thrown = played[j.saturating_sub(12)..played.len().min(j + 1)].iter().any(|p| p.button & B_BUTTON != 0);
        if thrown && held_bobomb(&traj[j]).is_some_and(|o| o.action == ACT_EXPLODE) {
            let since = traj.len() - 1 - j;
            return if since <= POST_FREE { (g, since <= POST_KEY) } else { (neutral, false) };
        }
    }
    let Some(o) = nearest_bobomb(s).filter(|o| s.hdist(o.pos) < 2.0 * FETCH) else { return (g, false) };
    let d = s.hdist(o.pos);
    let to = yaw_to(s.pos, o.pos);
    let i = traj.len() - 1;
    if o.action == ACT_LAUNCHED || bloated(traj, i, o.slot) {
        // Regrab: chase it, punch-grab while it bloats (5 frames).
        // On the window's frames the genome turns the stick (x) and may
        // punch early (B). The punch's stick is short: B with the stick
        // past 48 at 29+ speed dives instead (act_walking).
        let key = d < 300.0;
        if s.action & ACT_FLAG_AIR != 0 {
            // Still flying on after the throw: keep the stick forward.
            let (x, y) = stick_tilt(s, s.yaw as i32 + turn(g), 127.0);
            return (GamePad::new(0, x, y), key);
        }
        let press = (key && g_b) || (o.action == ACT_EXPLODE && d < 150.0 && !b_recent);
        let stick = stick_tilt(s, to + if key { turn(g) } else { 0 }, if press { 30.0 } else { 127.0 });
        return (GamePad::new(if press { B_BUTTON } else { 0 }, stick.0, stick.1), key);
    }
    // Fetch: walk to it, punch-grab facing it.
    // A move punch grabs a few frames after the press: at speed, press
    // early (TAS f21956: B at 258 away at fv 46, grabbed at 53).
    // No B in the air (a dive).
    let press = s.action & ACT_FLAG_AIR == 0 && d < 130.0f32.max(5.0 * s.forward_vel) && yaw_err(s, to) < 0x2000 && !b_recent;
    let (x, y) = stick_tilt(s, to, if press { 30.0 } else { (127.0 * d / 200.0).clamp(40.0, 127.0) });
    (GamePad::new(if press { B_BUTTON } else { 0 }, x, y), false)
}

fn on_floor(s: &St) -> bool {
    s.action & ACT_FLAG_AIR == 0 && s.floor_y > -10000.0 && s.pos[1] - s.floor_y < 5.0
}

type Cell = (u16, u8, i32, i32);

fn floors() -> &'static Mutex<HashMap<Cell, Vec<i16>>> {
    static F: OnceLock<Mutex<HashMap<Cell, Vec<i16>>>> = OnceLock::new();
    F.get_or_init(|| Mutex::new(HashMap::new()))
}

fn fcell(zone: (u16, u8), x: f32, z: f32) -> Cell {
    (zone.0, zone.1, (x / FCELL).floor() as i32, (z / FCELL).floor() as i32)
}

/// Records the floor `s` stands on in the learned floor map.
fn observe(s: &St) {
    if !s.controllable() || !on_floor(s) {
        return;
    }
    let h = s.floor_y.round().clamp(-32768.0, 32767.0) as i16;
    let mut m = floors().lock().unwrap();
    if m.len() >= FMAX {
        return;
    }
    let v = m.entry(fcell(s.zone(), s.pos[0], s.pos[2])).or_default();
    if v.len() < FHEIGHTS && !v.iter().any(|&y| (y as i32 - h as i32).abs() < 40) {
        v.push(h);
    }
}

/// A seen floor in the cell of `q` within `tol` of height `y`.
fn seen_floor(zone: (u16, u8), q: [f32; 3], y: f32, tol: f32) -> Option<f32> {
    let m = floors().lock().unwrap();
    m.get(&fcell(zone, q[0], q[2]))?.iter().map(|&h| h as f32).filter(|h| (h - y).abs() < tol).min_by(|a, b| (a - y).abs().total_cmp(&(b - y).abs()))
}

pub struct BombClip {
    zone: (u16, u8),
    wall: Wall,
    /// The seen floor point behind the wall.
    far: [f32; 3],
    cell: [i32; 2],
    octant: i32,
}

impl BombClip {
    /// At most one per `b` (the wall it touches), if a bob-omb is at hand
    /// and a floor was seen behind the wall. Also feeds the floor map.
    pub fn generate(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
        observe(b);
        if !b.controllable() || !b.alive() || b.action & ACT_FLAG_AIR != 0 {
            return;
        }
        let Some(nw) = b.wall_normal else { return };
        let l = nw[0].hypot(nw[2]);
        if l < 0.9 {
            return;
        }
        let bomb = held_bobomb(b).is_some() || (!b.holding && nearest_bobomb(b).is_some_and(|o| b.hdist(o.pos) < FETCH));
        if !bomb {
            return;
        }
        let n = [nw[0] / l, nw[2] / l];
        let p = [b.pos[0] - WALL_RADIUS * n[0], b.pos[1], b.pos[2] - WALL_RADIUS * n[1]];
        let wall = Wall { p, n };
        let Some(far) = FAR.iter().find_map(|&k| {
            let q = wall.front(-k);
            seen_floor(b.zone(), q, b.floor_y, FAR_DY).map(|h| [q[0], h, q[2]])
        }) else {
            return;
        };
        out.push(Arc::new(BombClip::new(b.zone(), wall, far)));
    }

    /// The objective at `wall` with the far floor point `far` (labs).
    pub fn new(zone: (u16, u8), wall: Wall, far: [f32; 3]) -> BombClip {
        let octant = ((wall.away_yaw() + 0x1000).rem_euclid(0x10000)) / 0x2000;
        let cell = [(wall.p[0] / 400.0).round() as i32, (wall.p[2] / 400.0).round() as i32];
        BombClip { zone, wall, far, cell, octant }
    }

    fn crossed(&self, first: &St, s: &St) -> bool {
        let w = &self.wall;
        s.zone() == self.zone
            && w.depth(first.pos) < 0.0
            && w.depth(s.pos) > WALL_RADIUS + 10.0
            && w.lateral(s.pos) < SUCCESS_LATERAL
            && (s.pos[1] - self.far[1]).abs() < 300.0
            && push_trace(s)
            && ((s.alive() && on_floor(s)) || s.stars > first.stars)
    }
}

impl Objective for BombClip {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): technique setups are frame-precise: every state, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn kind(&self) -> &'static str {
        "Tech::BombClip"
    }
    fn key(&self) -> String {
        format!("Tech::BombClip {:?} {:?} o{}", self.zone, self.cell, self.octant)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        if !self.base_criteria(b) {
            0.0
        } else if fresh_setup(b) {
            8.0
        } else {
            3.0
        }
    }
    fn base_criteria(&self, s: &St) -> bool {
        let w = &self.wall;
        if s.zone() != self.zone || !s.alive() || !s.controllable() {
            return false;
        }
        let d = w.depth(s.pos);
        if !(-NEAR_WALL..0.0).contains(&d) || w.lateral(s.pos) > NEAR_WALL {
            return false;
        }
        held_bobomb(s).is_some() || (!s.holding && nearest_bobomb(s).is_some_and(|o| s.hdist(o.pos) < FETCH && fetchable(o)))
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        let to_wall = s.hdist(self.wall.p) as f64 / 400.0;
        if held_bobomb(s).is_some() {
            return to_wall;
        }
        nearest_bobomb(s).map_or(f64::MAX, |o| to_wall + 1.0 + s.hdist(o.pos) as f64 / 400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let first = &traj[0];
        traj.iter().skip(1).position(|s| self.crossed(first, s)).map(|i| i + 1)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        progress(&self.wall, self.zone, 2.0 * SUCCESS_LATERAL, traj)
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 24, generations: 8, horizon: 240 }
    }
    fn precision(&self) -> Precision {
        Precision::Precise
    }
    fn success_value(&self) -> f64 {
        3.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        steer_target(&self.wall, s)
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        Some(overlay(self.wall))
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && s.objects().iter().any(is_bobomb)
    }
}
