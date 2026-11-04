//! Measured move arcs and an approach planner built on them.
//!
//! Every number here comes from `run/examples/arc_lab.rs` (`arc_lab <rom>
//! measure`), which plays each move in the real game on the castle grounds'
//! flat plateau and writes `ref/arcs/arcs.json` (readable) and
//! `ref/arcs/arcs_table.rs` (the same data as Rust, included below). Rerun
//! the lab to regenerate both. Pure functions only: no RNG, no state, no
//! game access; cheap enough to call every frame (the curves the planner
//! uses are sampled once).
//!
//! # What is measured (see the table types)
//! - [`Rec`]: jump arcs (single, double, triple, long jump from a run and
//!   chained, backflip, side flip, dive, dive + rollout, jump kick, slide
//!   kick with and without the hop out, head-on wall kick) per entry speed,
//!   A-hold time and air stick; per frame `along`/`height` from Mario's
//!   position before the move's first input, landing speed and action, and
//!   the lag until ACT_WALKING with the stick held.
//! - [`WallRec`]: wall kicks by incoming speed, angle off the wall's normal
//!   and A timing after the hit.
//! - [`SlideRec`]: butt / crouch / stomach slides on slopes by steepness.
//! - [`GpRec`], [`FallRec`]: ground pounds (spin, fall, get-up) and fall
//!   damage by height.
//! - [`ChainRec`]: distance covered in a fixed time by repeated moves
//!   (long-jump chains, slide-kick chains) against plain running.
//!
//! # Door entry (interaction.c `interact_door`, `interact_warp_door`)
//! A door opens only when Mario's hitbox (radius 37, height 160) overlaps
//! the door's (radius 80, height 100: Mario's centre within ~117 units of
//! the door object's position, horizontally) **while his action is
//! ACT_WALKING or ACT_DECELERATING**. Speed and facing don't matter (which
//! side he is on picks push or pull, should_push_or_pull_door). Any other
//! action (airborne, a landing action such as LONG_JUMP_LAND, crouch
//! slide, dive slide) just meets the door's collision as a wall: in the air
//! at forward speed > 16 that is a bonk (AIR_HIT_WALL / soft bonk, ~20+
//! frames lost); on the ground Mario stops. Star doors also need the
//! stars (`oBehParams >> 24`), key doors the key. The door object's
//! position is the door's centre on its hinge line; the castle's front
//! doors are two objects 153 apart (x = -76 and +77), each opens on its
//! own. The door action then snaps Mario onto the door's axis (measured:
//! walking at x = 0 into the front doors entered PUSHING_DOOR at x = -76,
//! 90 units in front of the door plane). So the approach needs: be
//! walking (not in a landing action) when Mario gets within ~110 of the
//! door's centre. Landings slide on through their lag at speed (measured,
//! stick held): JUMP_LAND / DOUBLE / TRIPLE / SIDE_FLIP / FREEFALL_LAND 4
//! frames (~100-140 units), LONG_JUMP_LAND **6 frames (~275-300 units at
//! 48)**, the dive's belly slide ~70 (roll out with A: 1 frame). A long
//! jump that lands 170 in front of a door slides into it still in
//! LONG_JUMP_LAND, stops against it and doesn't enter (seen in the lab);
//! the planner therefore ends every landing's lag slide by
//! `goal + stop_short - DOOR_REACH` ([`DOOR_STANDOFF`] = 160 in front of
//! the door, [`DOOR_REACH`] = 110). Also: the castle grounds' bridge is
//! the intro Lakitu's dialog trigger (x -544..545, y > 800, z -2000..-177)
//! until that dialog has played.
//!
//! # Fall damage (mario_actions_airborne.c `check_fall_damage`; measured)
//! On landing with vel.y < -55, from `peakHeight - y` (peakHeight is the
//! height of the last frame he was rising, so a wall kick or any upward
//! frame resets it): more than 3000 units takes 4 wedges (0x400 health,
//! cap on; 6 without) and a hard knockback (ACT_HARD_BACKWARD_GROUND_KB,
//! **82 frames** until he walks again); more than 1150 takes 2 wedges
//! (0x200; 3 without) on a non-slippery floor with no extra lag
//! (FREEFALL_LAND: 4 frames, as from any height). Measured: 1140 no
//! damage, 1160 two wedges, 2990 two, 3010 four + knockback. **A ground
//! pound does not avoid it**: it checks fall damage on landing too, and its
//! 15-frame spin first raises Mario 110 units (20, 18, ... a frame) and sets
//! peakHeight there, so a pound from 1040+ above the floor already costs
//! 2 wedges and from 2890+ the knockback. Twirling, water and lava floors
//! skip the check; slippery floors skip the 1150 tier. See [`fall_damage`].
//!
//! # Ground pound (measured, from rest)
//! Z in the air: the spin takes 15 frames (no horizontal motion at all:
//! a pound out of a full-speed jump lands exactly under the Z press, so it
//! is the way to stop dead over a spot), then he falls at 75 (terminal);
//! Z to landing is 20 frames from 150 up, 26 from 600, 32 from 1000, 38
//! from 1500. GROUND_POUND_LAND then takes 8 frames until he walks with
//! the stick held (A: airborne again after 9; untouched: idle after 27).
//!
//! # Wall kicks (measured on the castle door platform's side walls)
//! A jump into a wall at air speed > 16 gives AIR_HIT_WALL only within
//! **44 degrees** of head-on (45 and more: he slides along it, no kick;
//! this build). A must be pressed on one of the **5 frames** after the
//! hit (press 0-4; at 5+ he has dropped into the soft bonk). The kick
//! leaves at the mirror angle (in 30 off the normal -> out 30 the other
//! side), vy 62 (58 after the frame's gravity), and **keeps the incoming
//! speed only when A comes on the very first frame** (a long jump's 48-49
//! carries out); a later press leaves at ~25. It rises 512 above the hit
//! and lands ~975 out from the wall on flat ground after 34 frames, lag 4.
//! Chains between two parallel facing walls (predicted from that arc,
//! [`wall_kick_chain`]): +200 a kick at a 200 gap, +490-510 at 500-700,
//! +240 at 1000; the door platform's walls (1108 apart) are too far (he
//! lands first), and no narrower pair exists on the grounds to test on.
//!
//! # Slide kicks and slides (measured)
//! Z (crouch slide) then B while moving at >= 10: SLIDE_KICK sets the
//! speed to ~32 regardless of entry speed, with vy 12 and gravity 2; its
//! first landing **bounces** (vy = -vy/2: 7 on the flat after ~14 frames,
//! 14-25 when kicked downhill, where the bounce carries ~38-44 speed for
//! 40-80 air frames), then SLIDE_KICK_SLIDE. Riding the slide out costs
//! ~80 frames for ~1250 units (slower than running); hopping out with A
//! on the slide's first frame (FORWARD_ROLLOUT) gives ~1300 units in 38
//! frames. Repeated (Z, B, A-hop) a chain covers 2107 units in 60 frames
//! against running's 1900; long-jump chains 2958 and dive + rollout chains
//! 2968. Butt slides on the very slippery steep grass (ny 0.62-0.72) gain
//! ~7 a frame up to ~108; a crouch slide on ordinary slopes gains 0.1-1.
//! See [`SLIDES`], [`CHAINS`], [`slide_accel`].

#![allow(clippy::excessive_precision)]

use std::sync::OnceLock;

use obrrt_core::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use obrrt_core::{GamePad, St};

/// The moves measured as arcs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArcKind {
    Single,
    Double,
    Triple,
    LongJump,
    Backflip,
    Sideflip,
    Dive,
    DiveRollout,
    JumpKick,
    WallKick,
    /// Z (crouch slide) held `chain` frames, then B: SLIDE_KICK (one bounce),
    /// SLIDE_KICK_SLIDE until it stops.
    SlideKick,
    /// As `SlideKick`, then A on slide frame `held`: FORWARD_ROLLOUT.
    SlideKickHop,
}

/// The stick in the air, relative to the travel direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AirStick {
    Fwd,
    Neutral,
    Back,
    /// Full tilt 90 degrees to the right of the travel direction.
    Side,
}

/// One measured move (a row of `ref/arcs/arcs_table.rs`).
///
/// `along`/`height`: per frame from the move's first input (index 0 is the
/// position after that frame) to the landing frame, relative to the
/// position before the first input; `along` follows the travel direction
/// (backflip, side flip, wall kick: opposite the facing before the move).
/// `prep` frames come before the takeoff (long jump's crouch slide, the
/// side flip's turn, a backflip's crouch). `held`: frames A is held from
/// the takeoff (255: until falling); for the jump kick, the air frame of B;
/// for slide kicks, the slide frame of the hop. `chain`: long jumps, the
/// jumps before this one (0: from a run); slide kicks, Z frames before B.
#[derive(Debug, Clone, Copy)]
pub struct Rec {
    pub kind: ArcKind,
    pub stick: AirStick,
    pub chain: u8,
    pub held: u8,
    pub v_in: f32,
    pub v_take: f32,
    pub vy_take: f32,
    pub takeoff_action: u32,
    pub prep: u8,
    pub frames: u8,
    pub land_dist: f32,
    pub land_lat: f32,
    pub land_speed: f32,
    pub land_action: u32,
    /// Frames from landing to ACT_WALKING with the stick held (255: never
    /// within 90 frames), and the distance and speed meanwhile.
    pub lag: u8,
    pub lag_dist: f32,
    pub walk_speed: f32,
    pub peak: f32,
    pub peak_frame: u8,
    pub yaw_change: i16,
    pub along: &'static [f32],
    pub height: &'static [f32],
}

/// A wall kick: an incoming jump at `in_speed` (air forward speed at the
/// hit) and `in_angle` degrees off the wall's normal, A pressed `press`
/// frames after the first AIR_HIT_WALL frame (0: the very next input).
#[derive(Debug, Clone, Copy)]
pub struct WallRec {
    pub in_kind: ArcKind,
    pub in_speed: f32,
    pub in_angle: f32,
    pub press: u8,
    /// Whether the hit gave AIR_HIT_WALL at all (else it glanced off or
    /// slid along the wall).
    pub hit: bool,
    pub kicked: bool,
    pub out_speed: f32,
    /// Outgoing travel, degrees off the wall's normal (signed).
    pub out_angle: f32,
    pub vy: f32,
    pub frames: u8,
    pub land_dist: f32,
    pub peak: f32,
    pub lag: u8,
    /// Height above the hit, by distance from the wall: for chains.
    pub along: &'static [f32],
    pub height: &'static [f32],
}

/// How a slide was entered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlideEntry {
    /// Standing on the slope (steep enough: ACT_BEGIN_SLIDING -> BUTT_SLIDE).
    Stand,
    /// Z while running downhill at 16 (CROUCH_SLIDE).
    Crouch,
    /// Dive downhill at 30 (DIVE -> DIVE_SLIDE / STOMACH_SLIDE).
    Dive,
}

/// A slide downhill on a floor of normal y `ny`: speed per frame, and a
/// jump out with A at frame 20.
#[derive(Debug, Clone, Copy)]
pub struct SlideRec {
    pub entry: SlideEntry,
    pub ny: f32,
    pub floor_kind: i16,
    /// Mean speed gain per frame over the slide's frames 3..20.
    pub accel: f32,
    pub top_speed: f32,
    pub slide_action: u32,
    /// Yaw change per frame with the stick held 90 degrees to the side.
    pub steer_rate: i16,
    pub fv: &'static [f32],
    pub dist: &'static [f32],
    /// A at slide frame 20: the action it gives, its air frames, distance.
    pub jump_action: u32,
    pub jump_frames: u8,
    pub jump_dist: f32,
}

/// A ground pound from `height` above the floor (Z on the second frame of
/// a fall from rest).
#[derive(Debug, Clone, Copy)]
pub struct GpRec {
    pub height: f32,
    /// Frames from Z to the first falling frame (the spin).
    pub spin: u8,
    /// Frames from Z to landing.
    pub to_land: u8,
    pub land_action: u32,
    /// Landing to ACT_WALKING with the stick held; landing to airborne
    /// again with A pressed every other frame (the earliest jump out).
    pub lag_stick: u8,
    pub lag_a: u8,
    pub health_loss: i16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FallMode {
    Freefall,
    GroundPound,
}

/// A fall from rest from `height` above a flat non-slippery floor.
#[derive(Debug, Clone, Copy)]
pub struct FallRec {
    pub height: f32,
    pub mode: FallMode,
    /// Health units lost (0x100 a wedge), cap on.
    pub health_loss: i16,
    pub land_action: u32,
    /// Landing to ACT_WALKING with the stick held.
    pub lag: u8,
}

/// Distance covered in `frames` by repeating a move, against running.
#[derive(Debug, Clone, Copy)]
pub struct ChainRec {
    pub name: &'static str,
    pub frames: u16,
    pub dist: f32,
    pub run_dist: f32,
}

/// Ground motion at full speed (32): letting go of the stick, holding it
/// back, a 90-degree turn; the ground turn rate per frame.
#[derive(Debug, Clone, Copy)]
pub struct Ground {
    pub stop_frames: u8,
    pub stop_dist: f32,
    pub skid_frames: u8,
    pub skid_dist: f32,
    pub turn90_frames: u8,
    pub turn90_fv: f32,
    pub turn_rate: i32,
}

/// Air steering: full side stick against full forward stick at the same
/// speed.
#[derive(Debug, Clone, Copy)]
pub struct Steer {
    pub kind: ArcKind,
    pub v: f32,
    pub chain: u8,
    pub lat: f32,
    pub along_loss: f32,
    pub yaw_change: i32,
}

include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../ref/arcs/arcs_table.rs"));

// ---------------------------------------------------------------- actions

pub const ACT_FLAG_AIR: u32 = 0x0000_0800;
const ACT_FLAG_CONTROL_JUMP_HEIGHT: u32 = 0x0200_0000;
pub const ACT_IDLE: u32 = 0x0C40_0201;
pub const ACT_WALKING: u32 = 0x0400_0440;
pub const ACT_DECELERATING: u32 = 0x0400_044A;
pub const ACT_BRAKING: u32 = 0x0400_0445;
pub const ACT_CROUCH_SLIDE: u32 = 0x0480_8459;
pub const ACT_DIVE_SLIDE: u32 = 0x0088_0456;
pub const ACT_LONG_JUMP: u32 = 0x0300_0888;
pub const ACT_LONG_JUMP_LAND: u32 = 0x0000_0479;
pub const ACT_SLIDE_KICK: u32 = 0x0180_08AA;
pub const ACT_SLIDE_KICK_SLIDE: u32 = 0x0080_045A;
pub const ACT_GROUND_POUND_LAND: u32 = 0x0080_023C;
pub const ACT_GROUND_POUND: u32 = 0x0080_08A9;
pub const ACT_TWIRLING: u32 = 0x1080_08A4;

/// Standoff in front of a door for the last landing (see the module docs).
pub const DOOR_STANDOFF: f32 = 160.0;
/// Wall collision radius for Mario in the air (perform_air_quarter_step).
pub const WALL_RADIUS: f32 = 50.0;

// -------------------------------------------------------------- predictor

/// A predicted move: frames from its first input to landing, distance
/// along the travel direction, and what follows the landing.
#[derive(Debug, Clone, PartialEq)]
pub struct Arc {
    pub kind: ArcKind,
    pub prep: u32,
    pub frames: u32,
    pub land_dist: f32,
    pub land_lat: f32,
    pub peak: f32,
    pub peak_frame: u32,
    pub land_speed: f32,
    pub lag: u32,
    pub lag_dist: f32,
    pub walk_speed: f32,
    /// (along, height) per frame, as in [`Rec`].
    pub profile: Vec<[f32; 2]>,
}

fn held_key(h: u8) -> f32 {
    if h == 255 { 40.0 } else { h as f32 }
}

fn recs_for(kind: ArcKind, held: u8, stick: AirStick, chained: bool) -> Vec<&'static Rec> {
    let pick = |stick: AirStick| -> Vec<&'static Rec> {
        let rows: Vec<&Rec> = RECS
            .iter()
            .filter(|r| r.kind == kind && r.stick == stick)
            .filter(|r| match kind {
                ArcKind::LongJump => (r.chain > 0) == chained,
                _ => true,
            })
            .collect();
        // Nearest measured A-hold (or kick / hop frame); slide kicks also by
        // the nearest Z time (1 frame preferred: the fastest).
        let best = rows.iter().map(|r| (held_key(r.held) - held_key(held)).abs()).fold(f32::MAX, f32::min);
        let mut rows: Vec<&Rec> = rows.into_iter().filter(|r| (held_key(r.held) - held_key(held)).abs() == best).collect();
        if matches!(kind, ArcKind::SlideKick | ArcKind::SlideKickHop) {
            let z = rows.iter().map(|r| r.chain).min().unwrap_or(0);
            rows.retain(|r| r.chain == z);
        }
        if kind == ArcKind::LongJump && chained {
            // Chained jumps: one row per entry speed (the latest in the chain).
            rows.sort_by(|a, b| a.v_in.total_cmp(&b.v_in));
        }
        rows
    };
    let mut rows = pick(stick);
    if rows.is_empty() {
        rows = pick(AirStick::Fwd);
    }
    rows.sort_by(|a, b| a.v_in.total_cmp(&b.v_in));
    rows
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn arc_of(r: &Rec) -> Arc {
    Arc {
        kind: r.kind,
        prep: r.prep as u32,
        frames: r.frames as u32,
        land_dist: r.land_dist,
        land_lat: r.land_lat,
        peak: r.peak,
        peak_frame: r.peak_frame as u32,
        land_speed: r.land_speed,
        lag: r.lag as u32,
        lag_dist: r.lag_dist,
        walk_speed: r.walk_speed,
        profile: r.along.iter().zip(r.height).map(|(&a, &h)| [a, h]).collect(),
    }
}

/// The arc of `kind` started now at forward speed `forward_vel`, A held
/// `held_frames` from the takeoff (255: until falling), stick forward,
/// landing on flat ground; long jumps from a run. See [`predict_with`].
pub fn predict(kind: ArcKind, forward_vel: f32, held_frames: u8) -> Arc {
    predict_with(kind, forward_vel, held_frames, AirStick::Fwd, false)
}

/// [`predict`] with the air stick and, for long jumps, whether this one is
/// pressed from a LONG_JUMP_LAND (chained). Interpolates linearly between
/// the measured entry speeds (clamped to their range); frame counts are
/// rounded. Kinds whose arc doesn't depend on the entry speed (backflip,
/// side flip, wall kick) return their measured arc.
pub fn predict_with(kind: ArcKind, forward_vel: f32, held_frames: u8, stick: AirStick, chained: bool) -> Arc {
    // A-hold (kick / hop frame) between two measured ones: blend them.
    let helds: Vec<u8> = {
        let mut h: Vec<u8> = RECS.iter().filter(|r| r.kind == kind && (r.stick == stick || r.stick == AirStick::Fwd)).map(|r| r.held).collect();
        h.sort_unstable();
        h.dedup();
        h
    };
    if !helds.is_empty() && !helds.contains(&held_frames) && kind != ArcKind::LongJump {
        let key = held_key(held_frames);
        let lo = helds.iter().copied().filter(|&h| held_key(h) < key).max_by(|a, b| held_key(*a).total_cmp(&held_key(*b)));
        let hi = helds.iter().copied().filter(|&h| held_key(h) > key).min_by(|a, b| held_key(*a).total_cmp(&held_key(*b)));
        if let (Some(lo), Some(hi)) = (lo, hi) {
            let t = (key - held_key(lo)) / (held_key(hi) - held_key(lo));
            let pa = predict_with(kind, forward_vel, lo, stick, chained);
            let pb = predict_with(kind, forward_vel, hi, stick, chained);
            return blend(kind, &pa, &pb, t);
        }
    }
    let rows = recs_for(kind, held_frames, stick, chained);
    let Some(first) = rows.first() else {
        return Arc {
            kind,
            prep: 0,
            frames: 0,
            land_dist: 0.0,
            land_lat: 0.0,
            peak: 0.0,
            peak_frame: 0,
            land_speed: 0.0,
            lag: 0,
            lag_dist: 0.0,
            walk_speed: 0.0,
            profile: vec![],
        };
    };
    let last = rows[rows.len() - 1];
    if forward_vel <= first.v_in || rows.len() == 1 {
        return arc_of(first);
    }
    if forward_vel >= last.v_in {
        return arc_of(last);
    }
    let i = rows.iter().position(|r| r.v_in >= forward_vel).unwrap_or(rows.len() - 1).max(1);
    let (a, b) = (rows[i - 1], rows[i]);
    let t = if b.v_in > a.v_in { (forward_vel - a.v_in) / (b.v_in - a.v_in) } else { 0.0 };
    blend(kind, &arc_of(a), &arc_of(b), t)
}

/// Blends two measured arcs (`t` of the way from `pa` to `pb`) frame by
/// frame. For one-flight moves the shorter profile continues past its
/// landing with its last step, and the landing is where the blend first
/// comes back down to the landing height; moves with ground phases
/// (slide kicks, dive + rollout, jump kicks) take the blended length.
fn blend(kind: ArcKind, pa: &Arc, pb: &Arc, t: f32) -> Arc {
    let one_flight = !matches!(kind, ArcKind::SlideKick | ArcKind::SlideKickHop | ArcKind::DiveRollout | ArcKind::Dive | ArcKind::JumpKick);
    let n = pa.profile.len().max(pb.profile.len());
    let at = |p: &Arc, k: usize| -> [f32; 2] {
        if k < p.profile.len() {
            return p.profile[k];
        }
        let m = p.profile.len();
        if m < 2 {
            return p.profile.last().copied().unwrap_or([0.0; 2]);
        }
        let (u, v) = (p.profile[m - 2], p.profile[m - 1]);
        let e = (k - (m - 1)) as f32;
        if one_flight { [v[0] + (v[0] - u[0]) * e, v[1] + (v[1] - u[1]) * e * 2.0] } else { [v[0] + (v[0] - u[0]) * e, v[1]] }
    };
    let mut profile: Vec<[f32; 2]> = (0..n)
        .map(|k| {
            let (x, y) = (at(pa, k), at(pb, k));
            [lerp(x[0], y[0], t), lerp(x[1], y[1], t)]
        })
        .collect();
    let dy = lerp(pa.profile.last().map_or(0.0, |p| p[1]), pb.profile.last().map_or(0.0, |p| p[1]), t);
    if one_flight {
        let peak_k = profile.iter().enumerate().fold(0, |b, (k, p)| if p[1] > profile[b][1] { k } else { b });
        if let Some(k) = (peak_k + 1..profile.len()).find(|&k| profile[k][1] <= dy + 0.5) {
            profile.truncate(k + 1);
        }
    } else {
        let len = lerp(pa.profile.len() as f32, pb.profile.len() as f32, t).round().max(1.0) as usize;
        profile.truncate(len);
    }
    let peak_k = profile.iter().enumerate().fold(0, |b, (k, p)| if p[1] > profile[b][1] { k } else { b });
    let land = profile.last().copied().unwrap_or([0.0; 2]);
    Arc {
        kind,
        prep: lerp(pa.prep as f32, pb.prep as f32, t).round() as u32,
        frames: profile.len() as u32,
        land_dist: land[0],
        land_lat: lerp(pa.land_lat, pb.land_lat, t),
        peak: profile.get(peak_k).map_or(0.0, |p| p[1]),
        peak_frame: peak_k as u32 + 1,
        land_speed: lerp(pa.land_speed, pb.land_speed, t),
        lag: lerp(pa.lag as f32, pb.lag as f32, t).round() as u32,
        lag_dist: lerp(pa.lag_dist, pb.lag_dist, t),
        walk_speed: lerp(pa.walk_speed, pb.walk_speed, t),
        profile,
    }
}

/// Where an arc comes down onto a floor `dy` above (negative: below) its
/// takeoff: (frame index, distance), or None if it never gets that high
/// on the way down. Below the takeoff, extends the descent (falling at
/// the last frame's speed plus gravity 4, capped at 75).
pub fn land_on(arc: &Arc, dy: f32) -> Option<(u32, f32)> {
    let p = &arc.profile;
    if p.is_empty() {
        return None;
    }
    let peak = arc.peak_frame.saturating_sub(1) as usize;
    if dy > p[peak.min(p.len() - 1)][1] {
        return None;
    }
    for k in peak.max(1)..p.len() {
        if p[k][1] <= dy {
            let (u, v) = (p[k - 1], p[k]);
            let t = if (u[1] - v[1]).abs() > 1e-3 { ((u[1] - dy) / (u[1] - v[1])).clamp(0.0, 1.0) } else { 1.0 };
            return Some((k as u32 + 1, lerp(u[0], v[0], t)));
        }
    }
    // Below the takeoff: keep falling.
    let m = p.len();
    let (mut x, mut y) = (p[m - 1][0], p[m - 1][1]);
    let (dx, mut vy) = if m >= 2 { (p[m - 1][0] - p[m - 2][0], p[m - 1][1] - p[m - 2][1]) } else { (0.0, -4.0) };
    let mut k = m as u32;
    while y > dy && k < 300 {
        vy = (vy - 4.0).max(-75.0);
        x += dx;
        y += vy;
        k += 1;
    }
    Some((k, x))
}

// ---------------------------------------------------------------- air sim

/// Whether the action is airborne.
pub fn airborne(st: &St) -> bool {
    st.action & ACT_FLAG_AIR != 0
}

/// Simulates the rest of an airborne state's flight over a floor at
/// `floor_y` with a constant stick (`fwd` in -1..1 along the facing,
/// `side` -1..1 to the right), A released: frames to landing, landing
/// position, forward speed at landing. The air physics of
/// update_air_without_turn and perform_air_step (approach 0 by 0.35,
/// +1.5 per full forward tilt, side speed 10 per full side tilt, the
/// long jump's 48 drag threshold, gravity 4 (2 for long jumps and slide
/// kicks), 75 terminal, a released A quartering the rise above 20 for
/// jumps that control their height). Checked against the measured arcs
/// by `arc_lab validate`.
pub fn fly(st: &St, floor_y: f32, fwd: f32, side: f32) -> (u32, [f32; 3], f32) {
    let a = st.action;
    let long = a == ACT_LONG_JUMP;
    let g = if long || a == ACT_SLIDE_KICK { 2.0 } else { 4.0 };
    let drag: f32 = if long { 48.0 } else { 32.0 };
    let yaw = st.yaw as f32 * std::f32::consts::TAU / 65536.0;
    let (s, c) = yaw.sin_cos();
    let mut fv = st.forward_vel;
    let mut vy = st.vel[1];
    let mut p = st.pos;
    // should_strengthen_gravity_for_jump_ascent: MARIO_UNKNOWN_08 (0x100,
    // set by the jumps that control their height) and the action's flag.
    // (apply_gravity checks the long jump's and slide kick's own gravity first.)
    let control = !long && a != ACT_SLIDE_KICK && a & ACT_FLAG_CONTROL_JUMP_HEIGHT != 0 && st.mario_flags & 0x100 != 0;
    let mut k = 0;
    while k < 300 {
        k += 1;
        fv = if fv > 0.0 { (fv - 0.35).max(0.0) } else { (fv + 0.35).min(0.0) };
        fv += 1.5 * fwd;
        // Above the drag threshold only 1 a frame comes off ("uncapped air
        // speed": +0.15 a frame net with the stick forward).
        if fv > drag {
            fv -= 1.0;
        }
        fv = fv.max(-16.0);
        let sv = 10.0 * side;
        p[0] += fv * s + sv * c;
        p[2] += fv * c - sv * s;
        p[1] += vy;
        if p[1] <= floor_y && vy <= 0.0 {
            p[1] = floor_y;
            break;
        }
        if control && vy > 20.0 {
            vy /= 4.0;
        } else {
            vy = (vy - g).max(-75.0);
        }
    }
    (k, p, fv)
}

// ------------------------------------------------------------- wall kicks

/// A predicted wall kick.
#[derive(Debug, Clone, PartialEq)]
pub struct WallKickArc {
    pub out_speed: f32,
    pub out_angle: f32,
    pub vy: f32,
    pub peak: f32,
    pub frames: u32,
    pub land_dist: f32,
    pub lag: u32,
    /// (distance from the wall, height above the hit) per frame.
    pub profile: Vec<[f32; 2]>,
}

/// The wall kick off a jump hitting a wall at air forward speed
/// `in_speed`, `in_angle` degrees off the wall's normal, A pressed
/// `press_frame` frames after the hit (0: the first frame after it).
/// None where the measurements (and act_air_hit_wall) give no kick: air
/// speed 16 or less (no AIR_HIT_WALL), [`WALL_KICK_MAX_ANGLE`] or more off
/// the normal (he slides along the wall), or A outside the 5-frame window.
pub fn predict_wall_kick(in_speed: f32, in_angle: f32, press_frame: u8) -> Option<WallKickArc> {
    let ang = in_angle.abs();
    if in_speed <= 16.0 || ang >= WALL_KICK_MAX_ANGLE {
        return None;
    }
    // Nearest measured row by angle, then speed, among those at this press.
    let rows: Vec<&WallRec> = WALL_KICKS.iter().filter(|r| r.press == press_frame && r.hit).collect();
    if rows.is_empty() {
        return None;
    }
    let r = rows
        .iter()
        .min_by(|a, b| {
            let d = |r: &WallRec| (r.in_angle.abs() - ang).abs() / 15.0 + (r.in_speed - in_speed).abs() / 8.0;
            d(a).total_cmp(&d(b))
        })
        .unwrap();
    if !r.kicked {
        return None;
    }
    Some(WallKickArc {
        out_speed: r.out_speed,
        out_angle: r.out_angle.signum() * r.out_angle.abs() * if in_angle < 0.0 { -1.0 } else { 1.0 },
        vy: r.vy,
        peak: r.peak,
        frames: r.frames as u32,
        land_dist: r.land_dist,
        lag: r.lag as u32,
        profile: r.along.iter().zip(r.height).map(|(&a, &h)| [a, h]).collect(),
    })
}

/// Incoming angle off a wall's normal from which a jump no longer stops
/// against it (measured: 44 kicks, 45 slides along).
pub const WALL_KICK_MAX_ANGLE: f32 = 45.0;

/// The latest A frame after a wall hit that still kicks (from the lab).
pub fn wall_kick_window() -> u8 {
    WALL_KICKS.iter().filter(|r| r.kicked).map(|r| r.press).max().unwrap_or(0)
}

/// Height gained per kick bouncing between two parallel facing walls
/// `gap` apart (surface to surface), kicking head-on on each hit: the
/// head-on kick's height when Mario's centre gets within the wall radius
/// of the far wall. None if he lands (comes back to the kick's height
/// minus 1000) first. Negative: each kick ends lower than it started
/// (the chain loses height and soon can't continue).
pub fn wall_kick_chain(gap: f32) -> Option<f32> {
    let r = WALL_KICKS.iter().find(|r| r.kicked && r.in_angle.abs() < 1.0 && r.press == 0)?;
    let reach = gap - 2.0 * WALL_RADIUS;
    let n = r.along.len();
    for k in 1..n {
        if r.along[k] >= reach {
            let t = (reach - r.along[k - 1]) / (r.along[k] - r.along[k - 1]).max(1e-3);
            return Some(lerp(r.height[k - 1], r.height[k], t));
        }
    }
    // Past the measured landing (flat floor): continue the fall.
    let (mut x, mut y) = (r.along[n - 1], r.height[n - 1]);
    let dx = if n >= 2 { r.along[n - 1] - r.along[n - 2] } else { r.out_speed };
    let mut vy = if n >= 2 { r.height[n - 1] - r.height[n - 2] } else { -4.0 };
    while x < reach && y > -1000.0 {
        vy = (vy - 4.0).max(-75.0);
        x += dx;
        y += vy;
    }
    (x >= reach).then_some(y)
}

/// Kicks needed to climb `dy` between walls `gap` apart (first kick from a
/// jump hitting the wall at the start height), if the chain climbs at all.
pub fn wall_kicks_to_climb(gap: f32, dy: f32) -> Option<u32> {
    let first = WALL_KICKS.iter().find(|r| r.kicked && r.in_angle.abs() < 1.0 && r.press == 0)?;
    if dy <= first.peak {
        return Some(1);
    }
    let per = wall_kick_chain(gap)?;
    if per <= 0.0 {
        return None;
    }
    Some(1 + ((dy - first.peak) / per).ceil() as u32)
}

// ------------------------------------------------------- slides and kicks

/// Speed gain per frame sliding straight down a floor of normal y `ny`
/// (interpolated over the measured slopes; 0 on flat floors: crouch and
/// butt slides lose speed there).
pub fn slide_accel(ny: f32) -> f32 {
    let mut rows: Vec<&SlideRec> = SLIDES.iter().filter(|r| r.entry != SlideEntry::Dive).collect();
    rows.sort_by(|a, b| a.ny.total_cmp(&b.ny));
    if rows.is_empty() || ny >= 0.999 {
        return 0.0;
    }
    if ny <= rows[0].ny {
        return rows[0].accel;
    }
    for w in rows.windows(2) {
        if ny <= w[1].ny {
            let t = (ny - w[0].ny) / (w[1].ny - w[0].ny).max(1e-4);
            return lerp(w[0].accel, w[1].accel, t);
        }
    }
    // Between the gentlest measured slope and flat.
    let l = rows[rows.len() - 1];
    lerp(l.accel, 0.0, ((ny - l.ny) / (1.0 - l.ny)).clamp(0.0, 1.0))
}

/// Frames to slide `dist` straight downhill on a slope of normal y `ny`
/// from speed `v0`, capped by the top speed the measurements reached.
pub fn slide_frames(v0: f32, ny: f32, dist: f32) -> f32 {
    let a = slide_accel(ny);
    let top = SLIDES.iter().map(|r| r.top_speed).fold(32.0f32, f32::max);
    let (mut v, mut d, mut k) = (v0.max(1.0), 0.0, 0.0);
    while d < dist && k < 600.0 {
        v = (v + a).clamp(1.0, top);
        d += v;
        k += 1.0;
    }
    k
}

/// A slide kick from running at `v_in`: Z for `z_frames`, B, then A on slide
/// frame `hop_at` (None: ride the slide out). See [`ArcKind::SlideKick`].
pub fn predict_slide_kick(v_in: f32, z_frames: u8, hop_at: Option<u8>) -> Arc {
    let _ = z_frames;
    match hop_at {
        Some(h) => predict_with(ArcKind::SlideKickHop, v_in, h, AirStick::Fwd, false),
        None => predict_with(ArcKind::SlideKick, v_in, 0, AirStick::Fwd, false),
    }
}

// ------------------------------------------------ ground pounds and falls

/// A predicted ground pound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundPound {
    pub spin: u32,
    pub to_land: u32,
    /// Landing to walking (stick held) and to a jump out (A).
    pub lag_stick: u32,
    pub lag_a: u32,
    /// Z to the next useful action (walking).
    pub total: u32,
    pub health_loss: i16,
}

/// A ground pound started at rest `height` above the floor (interpolated
/// over the measured heights).
pub fn ground_pound(height: f32) -> GroundPound {
    let mut rows: Vec<&GpRec> = GP.iter().collect();
    rows.sort_by(|a, b| a.height.total_cmp(&b.height));
    let pick = |r: &GpRec| GroundPound {
        spin: r.spin as u32,
        to_land: r.to_land as u32,
        lag_stick: r.lag_stick as u32,
        lag_a: r.lag_a as u32,
        total: r.to_land as u32 + r.lag_stick as u32,
        health_loss: r.health_loss,
    };
    let Some(first) = rows.first() else {
        return GroundPound { spin: 0, to_land: 0, lag_stick: 0, lag_a: 0, total: 0, health_loss: 0 };
    };
    if height <= first.height {
        return pick(first);
    }
    for w in rows.windows(2) {
        if height <= w[1].height {
            let t = (height - w[0].height) / (w[1].height - w[0].height).max(1e-3);
            let (a, b) = (pick(w[0]), pick(w[1]));
            let l = |x: u32, y: u32| lerp(x as f32, y as f32, t).round() as u32;
            return GroundPound {
                spin: a.spin,
                to_land: l(a.to_land, b.to_land),
                lag_stick: a.lag_stick,
                lag_a: a.lag_a,
                total: l(a.to_land, b.to_land) + a.lag_stick,
                health_loss: a.health_loss,
            };
        }
    }
    pick(rows[rows.len() - 1])
}

/// Units a ground pound's spin raises Mario (and his peakHeight) before
/// he falls (measured).
pub const GP_RISE: f32 = 110.0;

/// Health lost (units: 0x100 a wedge, cap on) and lag frames (landing to
/// walking with the stick held) of a landing from `peak_y` onto `land_y`
/// in `landing_action`: the game's rule (module docs) with the measured
/// lags (4 for an ordinary landing, 8 after a ground pound, 82 for the
/// hard knockback). For a ground pound started at rest at `peak_y`, pass
/// `peak_y` as is: [`GP_RISE`] is added here. Twirling never hurts.
/// Assumes a non-slippery floor (slippery ones skip the 1150 tier).
pub fn fall_damage(peak_y: f32, land_y: f32, landing_action: u32) -> (i16, u32) {
    let pound = landing_action == ACT_GROUND_POUND_LAND || landing_action == ACT_GROUND_POUND;
    if landing_action == ACT_TWIRLING {
        return (0, 4);
    }
    let h = peak_y - land_y + if pound { GP_RISE } else { 0.0 };
    let measured = |mode: FallMode, f: &dyn Fn(&FallRec) -> u32| -> Option<u32> {
        FALLS.iter().filter(|r| r.mode == mode && r.height <= h + 0.5).max_by(|a, b| a.height.total_cmp(&b.height)).map(f)
    };
    if h > 3000.0 {
        let lag = measured(FallMode::Freefall, &|r| r.lag as u32).filter(|&l| l > 20).unwrap_or(82);
        return (0x400, lag);
    }
    let loss = if h > 1150.0 { 0x200 } else { 0 };
    let lag = if pound { GROUND_POUND_LAND_LAG } else { 4 };
    (loss, lag)
}

/// GROUND_POUND_LAND to walking with the stick held (measured).
pub const GROUND_POUND_LAND_LAG: u32 = 8;

// ------------------------------------------------------------------ doors

/// The point to reach in front of a door and the direction to walk in:
/// `standoff` units out from the door object's position along its normal
/// (oFaceAngleYaw), on Mario's side (`from`: doors open from both sides).
/// See the module docs for the entry condition.
pub fn door_approach(obj_pos: [f32; 3], obj_yaw: i32, from: [f32; 3], standoff: f32) -> ([f32; 3], [f32; 2]) {
    let a = obj_yaw as f32 * std::f32::consts::TAU / 65536.0;
    let n = [a.sin(), a.cos()];
    let side = (from[0] - obj_pos[0]) * n[0] + (from[2] - obj_pos[2]) * n[1];
    let n = if side >= 0.0 { n } else { [-n[0], -n[1]] };
    ([obj_pos[0] + n[0] * standoff, obj_pos[1], obj_pos[2] + n[1] * standoff], [-n[0], -n[1]])
}

// ---------------------------------------------------------------- planner

/// What to do this frame (pure: [`Intent::pad`] turns it into inputs from
/// the action alone).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Intent {
    /// Full stick towards `yaw` (world yaw, 0 = +z), no buttons.
    Run { yaw: i16 },
    /// Stick towards `yaw` at `mag` (0..1).
    Walk { yaw: i16, mag: f32 },
    /// Start (Z in a ground move, Z+A in the crouch slide) or chain (Z+A on
    /// LONG_JUMP_LAND) a long jump towards `yaw`.
    LongJump { yaw: i16 },
    /// A jump (A on the ground), A held while rising.
    Jump { yaw: i16 },
    /// Z (crouch slide), B (slide kick), A on SLIDE_KICK_SLIDE (hop).
    SlideKick { yaw: i16, hop: bool },
    /// In the air: stick `fwd`/`side` (-1..1) relative to `yaw`; A while
    /// rising if `hold_a`.
    Air { yaw: i16, fwd: f32, side: f32, hold_a: bool },
    /// B while running at >= 29 (DIVE), A on DIVE_SLIDE (FORWARD_ROLLOUT).
    DiveRollout { yaw: i16 },
    /// No stick, no buttons (decelerate).
    Stop,
}

/// Raw stick for intendedYaw `yaw` at `tilt` (0..1), relative to the camera
/// yaw (adjust_analog_stick takes 6 off each axis).
pub fn stick(cam_yaw: i16, yaw: i16, tilt: f32) -> (i8, i8) {
    let rel = (yaw as i32 - cam_yaw as i32) as f32 * std::f32::consts::TAU / 65536.0;
    let t = tilt.clamp(0.0, 1.0) * 121.0;
    let raw = |v: f32| if v.abs() < 0.5 { 0 } else { (v + 6.0 * v.signum()).round().clamp(-127.0, 127.0) as i8 };
    (raw(t * rel.sin()), raw(-t * rel.cos()))
}

fn yaw_of(d: [f32; 2]) -> i16 {
    ((d[0].atan2(d[1]) * 65536.0 / std::f32::consts::TAU).round() as i32) as i16
}

impl Intent {
    /// The frame's pad. Buttons come from the action only (no memory of the
    /// last pad): presses land on frames whose previous frame couldn't have
    /// held them (A released in the air once falling, or for long jumps
    /// always; Z/B pressed on the action that takes them).
    pub fn pad(&self, st: &St) -> GamePad {
        let a = st.action;
        let ground_move = matches!(a, ACT_WALKING | ACT_DECELERATING | ACT_BRAKING | ACT_IDLE) || (a & ACT_FLAG_AIR == 0 && a & 0x0400_0000 != 0 && a != ACT_CROUCH_SLIDE);
        let mk = |yaw: i16, mag: f32, b: u16| {
            let (x, y) = stick(st.cam_yaw, yaw, mag);
            GamePad::new(b, x, y)
        };
        match *self {
            Intent::Run { yaw } => mk(yaw, 1.0, 0),
            Intent::Walk { yaw, mag } => mk(yaw, mag, 0),
            Intent::Stop => GamePad::new(0, 0, 0),
            Intent::LongJump { yaw } => {
                let b = match a {
                    ACT_CROUCH_SLIDE | ACT_LONG_JUMP_LAND => Z_TRIG | A_BUTTON,
                    ACT_LONG_JUMP => Z_TRIG,
                    _ if ground_move => Z_TRIG,
                    _ => 0,
                };
                mk(yaw, 1.0, b)
            }
            Intent::Jump { yaw } => {
                let b = if a & ACT_FLAG_AIR != 0 { if st.vel[1] > 0.0 { A_BUTTON } else { 0 } } else { A_BUTTON };
                mk(yaw, 1.0, b)
            }
            Intent::SlideKick { yaw, hop } => {
                let b = match a {
                    ACT_CROUCH_SLIDE => Z_TRIG | B_BUTTON,
                    ACT_SLIDE_KICK_SLIDE if hop => A_BUTTON,
                    ACT_SLIDE_KICK | ACT_SLIDE_KICK_SLIDE => 0,
                    _ if ground_move => Z_TRIG,
                    _ => 0,
                };
                mk(yaw, 1.0, b)
            }
            Intent::DiveRollout { yaw } => {
                let b = match a {
                    ACT_DIVE_SLIDE => A_BUTTON,
                    ACT_WALKING if st.forward_vel >= 29.0 => B_BUTTON,
                    _ => 0,
                };
                mk(yaw, 1.0, b)
            }
            Intent::Air { yaw, fwd, side, hold_a } => {
                let mag = (fwd * fwd + side * side).sqrt().min(1.0);
                let off = side.atan2(fwd) * 65536.0 / std::f32::consts::TAU;
                let b = if hold_a && st.vel[1] > 0.0 { A_BUTTON } else { 0 };
                if mag < 0.05 {
                    return GamePad::new(b, 0, 0);
                }
                mk((yaw as i32 + off.round() as i32) as i16, mag, b)
            }
        }
    }
}

/// One step of a plan.
#[derive(Debug, Clone, PartialEq)]
pub enum Planned {
    Run { dist: f32, frames: f32 },
    Move { kind: ArcKind, chained: bool, dist: f32, frames: u32, land_speed: f32, lag: u32 },
    Slide { dist: f32, frames: f32 },
}

/// A plan from Mario's state to a point (see [`plan_approach`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ApproachPlan {
    /// The standoff point the plan ends at, and the direction of the final
    /// approach.
    pub goal: [f32; 3],
    pub dir: [f32; 2],
    /// A point on the approach axis to go through first (when Mario is too
    /// far off it to come in straight), else the goal.
    pub via: [f32; 3],
    pub steps: Vec<Planned>,
    /// Predicted frames until standing/walking at the goal, and for running
    /// straight there.
    pub frames: f32,
    pub run_frames: f32,
    /// Mario's distance from the approach axis now.
    pub off_axis: f32,
    pub now: Intent,
}

/// Largest angle off the approach axis Mario may come in at (straight line
/// to the goal); further off, the plan goes through a point on the axis.
pub const APPROACH_CONE_DEG: f32 = 30.0;

/// Frames to run `dist` from forward speed `v` at full tilt (the measured
/// run-up from idle, [`RUN_FV`]; above 32 the walk loses 1 a frame).
pub fn run_frames(v: f32, dist: f32) -> f32 {
    if dist <= 0.0 {
        return 0.0;
    }
    let (mut v, mut d, mut k) = (v.max(0.0), 0.0, 0.0);
    while d < dist && k < 2000.0 {
        v = run_next(v);
        d += v;
        k += 1.0;
    }
    // The fraction of the last frame.
    k - ((d - dist) / v.max(1.0)).min(1.0)
}

/// Forward speed after one more frame of running at full tilt.
pub fn run_next(v: f32) -> f32 {
    if v > 32.0 {
        return (v - 1.0).max(32.0);
    }
    // From the measured run-up: the next sample after the first one at
    // or above v (the walk adds 1.1 - v/43 a frame on flat ground).
    if !RUN_FV.is_empty() {
        if let Some(i) = RUN_FV.iter().position(|&x| x > v + 1e-3) {
            let prev = if i == 0 { 0.0 } else { RUN_FV[i - 1] };
            let step = RUN_FV[i] - prev;
            return (v + step).min(32.0);
        }
        return 32.0f32.max(v);
    }
    (v + 1.1 - v / 43.0).min(32.0)
}

struct Curve {
    /// Per entry speed 0..=48 (step 1): distance, frames, landing speed, lag;
    /// and the shortest distance (stick back).
    pts: Vec<(f32, f32, f32, f32, f32)>,
    /// Extra distance per unit the landing floor is below the takeoff
    /// (the last frame's forward over downward motion).
    slope: Vec<f32>,
    /// Distance slid in the landing lag (landing to walking).
    lag_dist: Vec<f32>,
}

impl Curve {
    fn new(kind: ArcKind, held: u8, chained: bool) -> Curve {
        let arcs: Vec<(Arc, Arc)> = (0..=48)
            .map(|v| (predict_with(kind, v as f32, held, AirStick::Fwd, chained), predict_with(kind, v as f32, held, AirStick::Back, chained)))
            .collect();
        let pts = arcs.iter().map(|(f, b)| (f.land_dist, f.frames as f32, f.land_speed, f.lag as f32, b.land_dist.min(f.land_dist))).collect();
        let slope = arcs
            .iter()
            .map(|(f, _)| {
                let p = &f.profile;
                let n = p.len();
                // The last two whole air frames (the landing frame is cut short).
                if n < 3 {
                    return 0.0;
                }
                let (a, b) = (p[n - 3], p[n - 2]);
                let dy = a[1] - b[1];
                if dy > 1.0 { (b[0] - a[0]) / dy } else { 0.0 }
            })
            .collect();
        let lag_dist = arcs.iter().map(|(f, _)| f.lag_dist.max(0.0)).collect();
        Curve { pts, slope, lag_dist }
    }
    fn lag_dist_at(&self, v: f32) -> f32 {
        self.lag_dist[(v.clamp(0.0, 48.0).round() as usize).min(48)]
    }
    fn slope_at(&self, v: f32) -> f32 {
        self.slope[(v.clamp(0.0, 48.0).round() as usize).min(48)]
    }
    fn at(&self, v: f32) -> (f32, f32, f32, f32, f32) {
        let v = v.clamp(0.0, 48.0);
        let i = (v.floor() as usize).min(47);
        let t = v - i as f32;
        let (a, b) = (self.pts[i], self.pts[i + 1]);
        (lerp(a.0, b.0, t), lerp(a.1, b.1, t), lerp(a.2, b.2, t), lerp(a.3, b.3, t), lerp(a.4, b.4, t))
    }
}

struct Curves {
    lj: Curve,
    lj_chain: Curve,
    single: Curve,
    rollout: Curve,
    slide_hop: Curve,
}

fn curves() -> &'static Curves {
    static C: OnceLock<Curves> = OnceLock::new();
    C.get_or_init(|| Curves {
        lj: Curve::new(ArcKind::LongJump, 1, false),
        lj_chain: Curve::new(ArcKind::LongJump, 1, true),
        single: Curve::new(ArcKind::Single, 255, false),
        rollout: Curve::new(ArcKind::DiveRollout, 1, false),
        slide_hop: Curve::new(ArcKind::SlideKickHop, 1, false),
    })
}

/// A floor check for landings: the height of the static floor under (x, z)
/// near `y`, if the caller has one (e.g. from obrrt_core::geo's static
/// floors). Landings over no floor, or one more than `MAX_STEP` off, are
/// not planned.
pub type FloorFn<'a> = &'a dyn Fn(f32, f32, f32) -> Option<f32>;

/// Landing floors planned on: at most this much above Mario's height (a
/// long jump peaks ~240 up, but lands on a rise sooner than the flat arc
/// says) or below it (lands a little later and further: still short of a
/// goal planned for the flat).
const MAX_RISE: f32 = 100.0;
const MAX_DROP: f32 = 300.0;

/// Best way along a straight leg of length `len` from ground speed `v`:
/// (frames, steps, first intent kind). `start_chained`: Mario is in
/// LONG_JUMP_LAND (can chain a long jump now).
/// Where a jump of flat-ground distance `d` from `at` (floor `take_dy`)
/// really lands given the floor under the leg (`land`: floor height
/// relative to Mario's, None where there's none), and that floor's height.
fn adjust(land: &dyn Fn(f32) -> Option<f32>, at: f32, d: f32, take_dy: f32, slope: f32) -> Option<(f32, f32)> {
    let mut x = d;
    let mut dy = land(at + x)?;
    for _ in 0..2 {
        x = (d - (dy - take_dy) * slope).max(d * 0.5);
        dy = land(at + x)?;
    }
    Some((x, dy))
}

/// How far past the goal Mario's centre may get before he must be walking:
/// a door opens from ~110 units ([`DOOR_REACH`]) and only while walking,
/// so the slide of a landing's lag must end by then.
pub const DOOR_REACH: f32 = 110.0;

fn plan_leg(v: f32, len: f32, start_chained: bool, land: &dyn Fn(f32) -> Option<f32>, slack: f32) -> (f32, Vec<Planned>, u8) {
    let c = curves();
    let run_all = run_frames(v, len);
    let mut best = (run_all, vec![Planned::Run { dist: len, frames: run_all }], 0u8);
    // Run-ups: (distance, frames, speed) after each whole frame of running.
    let mut runup = vec![(0.0f32, 0.0f32, v)];
    {
        let (mut d, mut vr, mut k) = (0.0f32, v, 0.0f32);
        while d <= len && k < 400.0 {
            vr = run_next(vr);
            d += vr;
            k += 1.0;
            runup.push((d, k, vr));
        }
    }
    // n long jumps after a run of r, then the rest walked.
    for n in 1..=8usize {
        for &(r, rf, vr) in &runup {
            if r > len {
                break;
            }
            let mut at = r;
            let mut frames = rf;
            let mut steps = vec![];
            if r > 0.0 {
                steps.push(Planned::Run { dist: r, frames: rf });
            }
            let mut speed = vr;
            let mut ok = true;
            let mut lag = 0.0;
            let mut take_dy = land(at).unwrap_or(0.0);
            for k in 0..n {
                let chained = k > 0 || (start_chained && r == 0.0);
                let curve = if chained { &c.lj_chain } else { &c.lj };
                let (d, f, ls, lg, dmin) = curve.at(speed);
                if speed < 10.0 && !chained {
                    ok = false;
                    break;
                }
                let last = k + 1 == n;
                let slope = curve.slope_at(speed);
                let Some((mut dd, mut dy)) = adjust(land, at, d, take_dy, slope) else {
                    ok = false;
                    break;
                };
                // The landing plus its lag slide must end by the goal (+ slack).
                let end = len + slack - curve.lag_dist_at(speed);
                if at + dd > end {
                    // Shorten the jump with the stick back to land in time (only
                    // the last one: an earlier one would be wasted).
                    let Some(gdy) = land(end) else {
                        ok = false;
                        break;
                    };
                    if !last || end <= at || (end - at) - (gdy - take_dy) * slope < dmin {
                        ok = false;
                        break;
                    }
                    dd = end - at;
                    dy = gdy;
                }
                take_dy = dy;
                at += dd;
                frames += f;
                speed = ls;
                lag = lg;
                steps.push(Planned::Move { kind: ArcKind::LongJump, chained, dist: dd, frames: f.round() as u32, land_speed: ls, lag: lg.round() as u32 });
            }
            if !ok || at > len + 1.0 {
                continue;
            }
            let rest = (len - at).max(0.0);
            let wf = run_frames(speed.min(48.0), rest).max(lag.min(rest / speed.max(1.0) + lag));
            let total = frames + wf;
            if rest > 0.5 {
                steps.push(Planned::Run { dist: rest, frames: wf });
            }
            if total < best.0 - 0.5 {
                let first = if r > 0.0 { 0 } else { 1 };
                best = (total, steps, first);
            }
        }
    }
    // One final single jump / dive-rollout / slide-kick hop after a run.
    for (code, curve, min_v) in [(2u8, &c.single, 0.0f32), (3, &c.rollout, 29.0), (4, &c.slide_hop, 10.0)] {
        for &(r, rf, vr) in &runup {
            if r > len {
                break;
            }
            if vr >= min_v {
                let (d0, f, ls, lg, _) = curve.at(vr);
                let adj = adjust(land, r, d0, land(r).unwrap_or(0.0), curve.slope_at(vr));
                if let Some((dist, _)) = adj
                    && r + dist + curve.lag_dist_at(vr) <= len + slack
                {
                    let rest = len - r - dist;
                    let wf = run_frames(ls, rest).max(lg);
                    let total = rf + f + wf;
                    if total < best.0 - 0.5 {
                        let kind = match code {
                            2 => ArcKind::Single,
                            3 => ArcKind::DiveRollout,
                            _ => ArcKind::SlideKickHop,
                        };
                        let mut steps = vec![];
                        if r > 0.0 {
                            steps.push(Planned::Run { dist: r, frames: rf });
                        }
                        steps.push(Planned::Move { kind, chained: false, dist, frames: f.round() as u32, land_speed: ls, lag: lg.round() as u32 });
                        if rest > 0.5 {
                            steps.push(Planned::Run { dist: rest, frames: wf });
                        }
                        best = (total, steps, if r > 0.0 { 0 } else { code });
                    }
                }
            }
        }
    }
    best
}

/// Plans Mario's way to `target`, to stop `stop_short` units before it,
/// coming in along `approach_dir` (unit XZ direction of travel at the end,
/// e.g. into a door: from [`door_approach`]) if given. Straight-line
/// distances only; `floor` (optional) checks each planned landing for a
/// floor from 300 below to 100 above Mario (landings over a pit or onto a
/// step are not planned). Chooses among running, chains of 1-8 long jumps after
/// a run-up of any length (the last one shortened with the stick held
/// back, so that the landing and its lag slide end by the goal plus
/// `stop_short - DOOR_REACH`: Mario is walking again before he reaches a
/// door's opening range, never past it), and a final
/// single jump / dive + rollout / slide-kick hop, by predicted frames.
/// Off the approach axis by more than [`APPROACH_CONE_DEG`] it first goes
/// to a point on the axis (`via`). In the air it steers the landing: stick
/// forward unless that lands past the goal, back as needed, sideways
/// towards the axis (10 units a frame at full side tilt).
///
/// Returns None outside normal play or for a target under 1 unit away.
/// Meant to be called every frame (closed loop): play `plan.now.pad(st)`.
pub fn plan_approach(st: &St, target: [f32; 3], approach_dir: Option<[f32; 2]>, stop_short: f32) -> Option<ApproachPlan> {
    plan_approach_with(st, target, approach_dir, stop_short, None)
}

/// [`plan_approach`] with a floor check (see [`FloorFn`]).
pub fn plan_approach_with(st: &St, target: [f32; 3], approach_dir: Option<[f32; 2]>, stop_short: f32, floor: Option<FloorFn>) -> Option<ApproachPlan> {
    let p = st.pos;
    let to = [target[0] - p[0], target[2] - p[2]];
    let dist = (to[0] * to[0] + to[1] * to[1]).sqrt();
    if dist < 1.0 {
        return None;
    }
    let dir = approach_dir.map_or([to[0] / dist, to[1] / dist], |d| {
        let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-6);
        [d[0] / l, d[1] / l]
    });
    let goal = [target[0] - dir[0] * stop_short, target[1], target[2] - dir[1] * stop_short];
    // Mario relative to the axis through the goal: `ahead` before the goal
    // along dir, `lat` to the side.
    let g = [goal[0] - p[0], goal[2] - p[2]];
    let ahead = g[0] * dir[0] + g[1] * dir[1];
    let lat = -g[0] * dir[1] + g[1] * dir[0];
    let cone = APPROACH_CONE_DEG.to_radians().tan();
    let direct = ahead > 0.0 && lat.abs() <= ahead * cone || lat.abs() < 30.0 && ahead > -stop_short;
    let via = if direct {
        goal
    } else {
        let back = (lat.abs() / cone).max(150.0);
        [goal[0] - dir[0] * back, goal[1], goal[2] - dir[1] * back]
    };
    let leg = [via[0] - p[0], via[2] - p[2]];
    let len = (leg[0] * leg[0] + leg[1] * leg[1]).sqrt();
    let u = if len > 1e-3 { [leg[0] / len, leg[1] / len] } else { dir };
    let yaw = yaw_of(u);
    let tail = if direct { 0.0 } else { (via[0] - goal[0]).hypot(via[2] - goal[2]) };
    // The floor along the leg, relative to Mario's height (looked up every
    // 8 units along it, once per call).
    let memo: std::cell::RefCell<Vec<Option<Option<f32>>>> = std::cell::RefCell::new(vec![None; (len / 8.0) as usize + 64]);
    let land_dy = |d: f32| -> Option<f32> {
        let Some(f) = floor else { return Some(0.0) };
        let i = (d.max(0.0) / 8.0).round() as usize;
        if let Some(Some(v)) = memo.borrow().get(i) {
            return *v;
        }
        let d = i as f32 * 8.0;
        let x = p[0] + u[0] * d;
        let z = p[2] + u[1] * d;
        let v = f(x, z, p[1] + MAX_RISE + 50.0).map(|y| y - p[1]).filter(|&dy| dy <= MAX_RISE && -dy <= MAX_DROP);
        if let Some(slot) = memo.borrow_mut().get_mut(i) {
            *slot = Some(v);
        }
        v
    };
    let fwd_speed = st.forward_vel * ((st.yaw as f32 - yaw as f32) * std::f32::consts::TAU / 65536.0).cos();
    let facing_err = ((st.yaw as i32 - yaw as i32) as i16 as i32).abs();
    let off_axis = lat.abs();
    let run_frames_all = run_frames(fwd_speed.max(0.0), dist - stop_short);

    // Airborne: steer the landing.
    if airborne(st) {
        let (mut k, mut land, _) = fly(st, st.floor_y, 1.0, 0.0);
        if let Some(f) = floor {
            // The floor where he comes down, not the one under him now.
            for _ in 0..2 {
                if let Some(y) = f(land[0], land[2], st.pos[1].max(land[1]) + 100.0) {
                    (k, land, _) = fly(st, y, 1.0, 0.0);
                }
            }
        }
        let (s, c) = (st.yaw as f32 * std::f32::consts::TAU / 65536.0).sin_cos();
        // Past the goal by the landing, plus the landing lag's slide, less the
        // slack before the door stops taking him.
        let slide = if st.action == ACT_LONG_JUMP { curves().lj_chain.lag_dist_at(st.forward_vel) } else { curves().single.lag_dist_at(st.forward_vel.min(32.0)) };
        let slack = if direct { (stop_short - DOOR_REACH).max(0.0) } else { 0.0 };
        let over = (land[0] - via[0]) * u[0] + (land[2] - via[2]) * u[1] + slide - slack;
        let fwd = if over <= 0.0 {
            1.0
        } else {
            // Each frame of back stick takes ~3 units (1.5 lost, and not gained)
            // per remaining frame.
            let per = 3.0 * k as f32 * (k as f32 + 1.0) / 2.0 / k.max(1) as f32;
            (1.0 - over / per.max(1.0)).clamp(-1.0, 1.0)
        };
        // Lateral error of the landing from the leg's line, relative to the facing.
        let e = (land[0] - p[0]) * -u[1] + (land[2] - p[2]) * u[0];
        let lat_line = e + ((p[0] - via[0]) * -u[1] + (p[2] - via[2]) * u[0]);
        let right = [c, -s];
        let side_needed = -lat_line / (10.0 * k.max(1) as f32);
        let sgn = right[0] * -u[1] + right[1] * u[0];
        let side = (side_needed * sgn.signum()).clamp(-1.0, 1.0);
        let hold = st.action & ACT_FLAG_CONTROL_JUMP_HEIGHT != 0 && over < 0.0;
        let now = Intent::Air { yaw: st.yaw, fwd, side, hold_a: hold };
        return Some(ApproachPlan {
            goal,
            dir,
            via,
            steps: vec![],
            frames: k as f32 + run_frames(st.forward_vel.max(0.0), (len - over.max(0.0) - k as f32 * st.forward_vel).max(0.0)) + tail / 32.0,
            run_frames: run_frames_all,
            off_axis,
            now,
        });
    }

    // On the ground (or in a landing): plan the leg.
    let chained = st.action == ACT_LONG_JUMP_LAND;
    let speed = if facing_err < 0x1000 { fwd_speed.max(0.0) } else { 0.0 };
    let slack = if direct { (stop_short - DOOR_REACH).max(0.0) } else { 0.0 };
    let (frames, steps, first) = plan_leg(speed, len, chained && facing_err < 0x800, &land_dy, slack);
    let turn = facing_err as f32 / GROUND.turn_rate.max(1) as f32;
    let total = frames + turn + if direct { 0.0 } else { run_frames(24.0, tail) + GROUND.turn90_frames as f32 * 0.5 };
    // At or past the goal on the axis: walk into the target.
    let to_target = yaw_of([to[0] / dist, to[1] / dist]);
    if direct && ahead <= 30.0 {
        let f = run_frames(fwd_speed.max(0.0), dist);
        return Some(ApproachPlan { goal, dir, via, steps: vec![Planned::Run { dist, frames: f }], frames: f, run_frames: run_frames_all, off_axis, now: Intent::Run { yaw: to_target } });
    }
    let now = if st.action == ACT_DIVE_SLIDE {
        // Roll out of a dive (else ~60 frames of belly slide).
        Intent::DiveRollout { yaw }
    } else if st.action == ACT_SLIDE_KICK_SLIDE {
        Intent::SlideKick { yaw, hop: true }
    } else if len < 2.0 {
        Intent::Run { yaw: yaw_of(dir) }
    } else if facing_err > 0x800 && !chained {
        Intent::Run { yaw }
    } else {
        match first {
            _ if st.action == ACT_LONG_JUMP_LAND && first != 1 => Intent::Run { yaw: if direct { to_target } else { yaw } },
            1 => Intent::LongJump { yaw },
            2 => Intent::Jump { yaw },
            3 => Intent::DiveRollout { yaw },
            4 => Intent::SlideKick { yaw, hop: true },
            _ if st.action == ACT_CROUCH_SLIDE => Intent::LongJump { yaw },
            _ => Intent::Run { yaw: if direct { to_target } else { yaw } },
        }
    };
    Some(ApproachPlan { goal, dir, via, steps, frames: total, run_frames: run_frames_all, off_axis, now })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st_at(pos: [f32; 3], yaw: i16, fv: f32, action: u32) -> St {
        St {
            frame: 0,
            pos,
            vel: [0.0; 3],
            forward_vel: fv,
            yaw,
            pitch: 0,
            angle_vel: [0; 3],
            water_level: -11000,
            cam_yaw: 0,
            action,
            mario_flags: 0,
            holding: false,
            held_slot: None,
            red_coins: 0,
            level: 16,
            area: 1,
            act: 0,
            course: 0,
            stars: 0,
            keys: 0,
            coins: 0,
            lives: 4,
            health: 0x880,
            play_mode: 0,
            transition: 0,
            dialog: -1,
            dialog_choice: false,
            menu_line: 0,
            menu_mode: -1,
            act_select: false,
            act_init: 0,
            act_cursor: 0,
            dialog_state: 0,
            dialog_last_page: false,
            menu_timer: 0,
            floor_y: pos[1],
            ceil_y: 20000.0,
            floor_type: 0,
            floor_normal: [0.0, 1.0, 0.0],
            floor_obj: None,
            floor_obj_vy: None,
            floor_obj_y: None,
            action_timer: 0,
            wall_normal: None,
            course_stars: [0; 25],
            save_flags: 0,
            objects: None,
        }
    }

    #[test]
    fn predict_interpolates_and_is_monotone() {
        if RECS.is_empty() {
            return;
        }
        let a = predict(ArcKind::LongJump, 24.0, 1);
        let b = predict(ArcKind::LongJump, 32.0, 1);
        let m = predict(ArcKind::LongJump, 28.0, 1);
        assert!(a.land_dist < b.land_dist, "{} {}", a.land_dist, b.land_dist);
        assert!(m.land_dist >= a.land_dist - 1.0 && m.land_dist <= b.land_dist + 1.0);
        assert!(a.frames > 10 && a.frames < 60);
        // A held longer jumps higher.
        let lo = predict(ArcKind::Single, 16.0, 1);
        let hi = predict(ArcKind::Single, 16.0, 255);
        assert!(hi.peak > lo.peak + 20.0, "{} {}", hi.peak, lo.peak);
        // Stick back lands shorter.
        let back = predict_with(ArcKind::LongJump, 32.0, 1, AirStick::Back, false);
        assert!(back.land_dist < b.land_dist);
    }

    #[test]
    fn land_on_lower_and_higher_floors() {
        if RECS.is_empty() {
            return;
        }
        let a = predict(ArcKind::Single, 24.0, 255);
        let flat = land_on(&a, 0.0).unwrap();
        let up = land_on(&a, 100.0).unwrap();
        let down = land_on(&a, -300.0).unwrap();
        assert!(up.1 < flat.1 && flat.1 < down.1);
        assert!(land_on(&a, a.peak + 10.0).is_none());
    }

    #[test]
    fn door_approach_is_on_marios_side() {
        let (p, d) = door_approach([0.0, 0.0, 0.0], 0, [10.0, 0.0, 500.0], 100.0);
        assert!((p[2] - 100.0).abs() < 1e-3 && d[1] < -0.99);
        let (p, d) = door_approach([0.0, 0.0, 0.0], 0x8000, [10.0, 0.0, 500.0], 100.0);
        assert!((p[2] - 100.0).abs() < 1e-2 && d[1] < -0.99);
        let (p, _) = door_approach([0.0, 0.0, 0.0], 0, [10.0, 0.0, -500.0], 100.0);
        assert!((p[2] + 100.0).abs() < 1e-3);
    }

    #[test]
    fn plan_never_lands_past_the_goal() {
        if RECS.is_empty() {
            return;
        }
        for d in [300.0f32, 800.0, 1500.0, 2500.0] {
            let st = st_at([0.0, 0.0, 0.0], 0, 32.0, ACT_WALKING);
            let plan = plan_approach(&st, [0.0, 0.0, d], Some([0.0, 1.0]), 160.0).unwrap();
            let mut at = 0.0;
            for s in &plan.steps {
                match s {
                    Planned::Run { dist, .. } | Planned::Slide { dist, .. } => at += dist,
                    Planned::Move { dist, .. } => {
                        at += dist;
                        assert!(at <= d - 160.0 + 51.0, "landing {at} past goal {}", d - 160.0);
                    }
                }
            }
            assert!((at - (d - 160.0)).abs() < 2.0, "{at} vs {}", d - 160.0);
            assert!(plan.frames <= plan.run_frames + 1.0, "{} vs {}", plan.frames, plan.run_frames);
            if d >= 1500.0 {
                assert!(plan.steps.iter().any(|s| matches!(s, Planned::Move { kind: ArcKind::LongJump | ArcKind::DiveRollout, .. })), "{:?}", plan.steps);
            }
        }
    }

    #[test]
    fn plan_goes_via_the_axis_when_far_off() {
        if RECS.is_empty() {
            return;
        }
        let st = st_at([1000.0, 0.0, 500.0], 0, 0.0, ACT_IDLE);
        let plan = plan_approach(&st, [0.0, 0.0, 1000.0], Some([0.0, 1.0]), 100.0).unwrap();
        assert!(plan.via != plan.goal && plan.via[0].abs() < 1e-3 && plan.via[2] < 900.0, "{:?}", plan.via);
        let st = st_at([50.0, 0.0, 0.0], 0, 0.0, ACT_IDLE);
        let plan = plan_approach(&st, [0.0, 0.0, 1000.0], Some([0.0, 1.0]), 100.0).unwrap();
        assert_eq!(plan.via, plan.goal);
    }

    #[test]
    fn planner_skips_landings_without_floor() {
        if RECS.is_empty() {
            return;
        }
        // Floor only for z < 400 and z > 1400: no long jump may land in between.
        let floor = |_x: f32, z: f32, _y: f32| if !(400.0..=1400.0).contains(&z) { Some(0.0) } else { None };
        let st = st_at([0.0, 0.0, 0.0], 0, 32.0, ACT_WALKING);
        let plan = plan_approach_with(&st, [0.0, 0.0, 2500.0], Some([0.0, 1.0]), 100.0, Some(&floor)).unwrap();
        let mut at = 0.0;
        for s in &plan.steps {
            match s {
                Planned::Move { dist, .. } => {
                    at += dist;
                    assert!(!(400.0..=1400.0).contains(&at), "landed at {at}");
                }
                Planned::Run { dist, .. } | Planned::Slide { dist, .. } => at += dist,
            }
        }
    }

    #[test]
    fn fall_damage_tiers() {
        assert_eq!(fall_damage(1000.0, 0.0, 0x0400_0471).0, 0);
        assert!(fall_damage(2000.0, 0.0, 0x0400_0471).0 >= 0x200);
        assert!(fall_damage(3500.0, 0.0, 0x0400_0471).0 >= 0x400);
        // A pound's spin adds 110 to the fall.
        assert_eq!(fall_damage(1000.0, 0.0, ACT_GROUND_POUND_LAND).0, 0);
        assert_eq!(fall_damage(1100.0, 0.0, ACT_GROUND_POUND_LAND).0, 0x200);
        assert_eq!(fall_damage(3500.0, 0.0, 0x0400_0471).1, 82);
    }

    #[test]
    fn wall_kick_limits() {
        assert!(predict_wall_kick(10.0, 0.0, 0).is_none());
        assert!(predict_wall_kick(30.0, 50.0, 0).is_none());
        if !WALL_KICKS.is_empty() {
            assert!(predict_wall_kick(25.0, 0.0, 0).is_some());
            assert!(predict_wall_kick(25.0, 40.0, 4).is_some());
            assert!(predict_wall_kick(25.0, 0.0, 6).is_none());
            assert_eq!(wall_kick_window(), 4);
            // The first-frame press keeps a long jump's speed.
            assert!(predict_wall_kick(48.0, 0.0, 0).unwrap().out_speed > 40.0);
            assert!(predict_wall_kick(48.0, 0.0, 2).unwrap().out_speed < 30.0);
            // Facing walls 1000 apart: the head-on kick can't reach.
            let far = wall_kick_chain(3000.0);
            assert!(far.is_none() || far.unwrap() < 0.0);
        }
    }

    #[test]
    fn run_model_matches_walk_formula() {
        let f = run_frames(32.0, 320.0);
        assert!((f - 10.0).abs() < 0.6, "{f}");
        assert!(run_frames(0.0, 500.0) > run_frames(32.0, 500.0));
    }
}
