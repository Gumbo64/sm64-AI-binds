//! The reduced input space.
//!
//! A raw pad is 8 button sets x ~65k stick values. What the game does with it
//! falls into a handful of outcomes per state (typically 1 to 12: another action,
//! kept / lost speed, up or down). The menu keeps, per action, only the (button
//! set, stick region) cells that reach those outcomes, found by brute force from
//! sampled states of the 2026 / 2012 / rikku 120-star TASs and the 70-star TAS
//! (`bin/menu_check.rs collect`, scratchpad `menu_derive.py` -> `menu_table.rs`),
//! then re-checked against the emulator (`menu_check eval`).
//!
//! Stick regions are measured from a reference direction ([`Ref`]): `Face` for the
//! table; `Target` / `Vel` aim cells are added on top for steering. Angle bounds, from
//! the guards and the data (angle = offset of the stick's world yaw from the reference,
//! positive = increasing yaw, called Left):
//!
//! | Ang        | offset from ref | why the bound                                      |
//! |------------|-----------------|----------------------------------------------------|
//! | Aim        | within 22.5     | on course: speed kept / gained                     |
//! | Left/Right | 22.5 - 67.5     | turns while most of the forward push remains      |
//! | SideLeft/R | 67.5 - 100      | cos(dyaw) near 0: speed neutral; walking turn rate |
//! | Back       | 100 - 260       | `analog_stick_held_back` (|dyaw| > 0x471C = 100 deg): turn-around, braking, bleed |
//!
//! | Mag     | stickMag     | why the bound                                           |
//! |---------|--------------|---------------------------------------------------------|
//! | Neutral | 0            | below raw deadzone (|raw| < 8): no INPUT_NONZERO_ANALOG   |
//! | Short   | 8 - 46       | intendedMag = mag^2/128 < 18; ground B never a speed kick |
//! | Full    | 49 - 64      | stickMag > 48: speed-kick dive; full air steering         |
//!
//! Buttons are levels (A, B, Z down); fresh / held / released follows from the last pad.

use obrrt_core::{GamePad, St};
use obrrt_objectives::moves::arcs;

use crate::types::*;

include!("menu_table.rs");

const A_BTN: u16 = 0x8000;
const B_BTN: u16 = 0x4000;
const Z_BTN: u16 = 0x2000;

pub fn buttons_of(b: Btn) -> u16 {
    (if b.a() { A_BTN } else { 0 }) | (if b.b() { B_BTN } else { 0 }) | (if b.z() { Z_BTN } else { 0 })
}

pub fn btn_of(pad: u16) -> Btn {
    Btn((if pad & A_BTN != 0 { Btn::A } else { 0 }) | (if pad & B_BTN != 0 { Btn::B } else { 0 }) | (if pad & Z_BTN != 0 { Btn::Z } else { 0 }))
}

const MAG_BY: [Mag; 3] = [Mag::Neutral, Mag::Short, Mag::Full];
const ANG_BY: [Ang; 6] = [Ang::Aim, Ang::Left, Ang::Right, Ang::SideLeft, Ang::SideRight, Ang::Back];

type Row = (u32, u8, &'static [(u8, u8, u8)]);

fn entry(act: u32) -> Option<&'static Row> {
    TABLE.binary_search_by_key(&act, |e| e.0).ok().map(|i| &TABLE[i])
}

/// The table row for `act`. Actions with no samples borrow a sampled action of the same group; groups the
/// player does not steer (cutscenes, objects) have none: a single no-op cell.
fn row(act: u32) -> Option<&'static Row> {
    if let Some(e) = entry(act) {
        return Some(e);
    }
    match act & 0x1C0 {
        0x000 => entry(0x0C40_0201), // stationary: idle
        0x040 => entry(0x0400_0440), // moving: walking
        0x080 => entry(0x0100_088C), // airborne: freefall
        0x0C0 => TABLE.iter().find(|e| e.0 & 0x1C0 == 0x0C0),
        _ => None,
    }
}

/// Buttons whose level this action's outcomes depend on (A = 1, B = 2, Z = 4).
pub fn relevant_buttons(act: u32) -> u8 {
    row(act).map_or(0, |e| e.1)
}

fn cell(b: u8, mag: Mag, ang: Ang, r: Ref) -> Cell {
    Cell { btn: Btn(b), stick: StickCls { mag, ang, r } }
}

pub const NOOP: Cell = Cell { btn: Btn(0), stick: StickCls { mag: Mag::Neutral, ang: Ang::Aim, r: Ref::Face } };

/// The reduced menu for this state.
pub fn menu(o: &Obs) -> Vec<Cell> {
    let s = o.ctx.s;
    let Some(e) = row(s.action) else { return vec![NOOP] };
    let mut v: Vec<Cell> = e.2.iter().map(|&(b, m, a)| cell(b, MAG_BY[m as usize], ANG_BY[a as usize], Ref::Face)).collect();
    // Steering cells: aim at the task's target, and along / against the velocity (slides, air).
    let mut btns: Vec<u8> = v.iter().filter(|c| c.stick.ang == Ang::Aim && c.stick.mag != Mag::Neutral).map(|c| c.btn.0).collect();
    btns.sort();
    btns.dedup();
    if let Some(t) = target_yaw(o) {
        if angle_diff(t, s.yaw).abs() > 0x1000 {
            for &b in &btns {
                v.push(cell(b, Mag::Full, Ang::Aim, Ref::Target));
            }
        }
    }
    let hv = s.vel[0].hypot(s.vel[2]);
    if hv > 5.0 && angle_diff(vel_yaw(s), s.yaw).abs() > 0x1000 {
        for &b in &btns {
            v.push(cell(b, Mag::Full, Ang::Aim, Ref::Vel));
        }
        if let Some(&b) = btns.first() {
            v.push(cell(b, Mag::Full, Ang::Back, Ref::Vel));
        }
    }
    // Landing prep: the button levels held on the frames before touchdown decide which cancels exist on
    // the ground (A held + B fresh = jump kick refresh, A fresh = double jump, Z, ...). One frame of
    // outcomes cannot see it (`menu_check carry`: holding a button set through the air changes the touchdown
    // outcomes in 75-98% of airborne states), so near the ground (or when the landing is unknown) every
    // button set is offered, stick on course.
    if s.action & 0x800 != 0 && o.sense.land.is_none_or(|l| l.frames <= AIR_PREP_FRAMES) {
        // Actions whose air phase no button changes (rollout, dive, kicks, long jump, slide kick): only
        // whether A is held matters (touchdown outcomes covered 1.00 by {none, A}; the rest need all eight
        // because B / Z there start dives, kicks and ground pounds).
        let fixed = FIXED_AIR.contains(&s.action);
        for b in 0..8u8 {
            if !fixed || b <= Btn::A {
                v.push(cell(b, Mag::Full, Ang::Aim, Ref::Face));
            }
        }
    }
    v.sort_by_key(|c| (c.btn.0, c.stick.mag as u8, c.stick.ang as u8, c.stick.r as u8));
    v.dedup();
    v
}

/// Airborne actions in which no button changes the air phase: forward / backward rollout, dive, jump kick,
/// long jump, slide kick, butt slide air, ground pound, lava boost, riding shell jump / fall.
const FIXED_AIR: [u32; 11] = [
    0x0100_08A6, 0x0100_08AD, 0x0188_088A, 0x0180_08AC, 0x0300_0888, 0x0180_08AA, 0x0300_088E, 0x0080_08A9, 0x0102_08B7, 0x0281_089A, 0x0081_089B,
];

/// Airborne states this close to touchdown (frames) get every button set.
pub const AIR_PREP_FRAMES: u32 = 3;

// ---------------------------------------------------------------- realize
fn angle_diff(a: i16, b: i16) -> i32 {
    (a as i32 - b as i32) as i16 as i32
}

fn yaw_of(dx: f32, dz: f32) -> i16 {
    ((dx.atan2(dz) * 65536.0 / std::f32::consts::TAU).round() as i64 as i32) as i16
}

fn vel_yaw(s: &St) -> i16 {
    if s.vel[0].hypot(s.vel[2]) < 0.5 { s.yaw } else { yaw_of(s.vel[0], s.vel[2]) }
}

fn target_yaw(o: &Obs) -> Option<i16> {
    let s = o.ctx.s;
    let t = o.ctx.pts[0];
    let (dx, dz) = (t[0] - s.pos[0], t[2] - s.pos[2]);
    if t == [0.0; 3] || dx.hypot(dz) < 30.0 { None } else { Some(yaw_of(dx, dz)) }
}

/// The world yaw a reference points along.
pub fn ref_yaw(r: Ref, o: &Obs, world: i16) -> i16 {
    let s = o.ctx.s;
    match r {
        Ref::Face => s.yaw,
        Ref::Target => target_yaw(o).unwrap_or(s.yaw),
        Ref::Vel => vel_yaw(s),
        Ref::Downhill | Ref::Uphill => {
            let (nx, nz) = (s.floor_normal[0], s.floor_normal[2]);
            let d = if nx.hypot(nz) < 0.02 { s.yaw } else { yaw_of(nx, nz) };
            if r == Ref::Downhill { d } else { d.wrapping_add(0x8000u16 as i16) }
        }
        // The stick pushed straight up on the pad moves along the camera's forward = cam_yaw + 180 deg
        // (moves::arcs::stick: y = -t cos(yaw - cam)).
        Ref::Cam => s.cam_yaw.wrapping_add(0x8000u16 as i16),
        Ref::World => world,
    }
}

const DEG: f32 = 65536.0 / 360.0;

/// (centre offset, half width) in yaw units of an angle class, positive = Left.
pub fn ang_range(a: Ang) -> (f32, f32) {
    match a {
        Ang::Aim => (0.0, 21.0 * DEG),
        Ang::Left => (45.0 * DEG, 22.0 * DEG),
        Ang::Right => (-45.0 * DEG, 22.0 * DEG),
        Ang::SideLeft => (84.0 * DEG, 15.0 * DEG),
        Ang::SideRight => (-84.0 * DEG, 15.0 * DEG),
        Ang::Back => (180.0 * DEG, 78.0 * DEG),
    }
}

/// stickMag (after the 6-unit deadzone shift) of a class at `fine_mag` in -1..1 (1 = the class's largest).
pub fn mag_range(m: Mag, fine: f32) -> f32 {
    let f = fine.clamp(-1.0, 1.0);
    match m {
        Mag::Neutral => 0.0,
        Mag::Short => 27.0 + 19.0 * f,
        Mag::Full => 56.5 + 7.5 * f,
    }
}

/// Raw stick for a world yaw and stickMag, as the game will read it: raw = stick +/- 6 outside the
/// deadzone (|raw| < 8 reads 0), stickMag capped at 64 by the game.
pub fn raw_stick(cam_yaw: i16, yaw: i16, mag: f32) -> (i8, i8) {
    if mag <= 0.0 {
        return (0, 0);
    }
    arcs::stick(cam_yaw, yaw, mag / 121.0)
}

/// Turn a choice into this frame's pad.
pub fn realize(c: &Choice, o: &Obs) -> GamePad {
    let s = o.ctx.s;
    let btn = buttons_of(c.cell.btn);
    let st = c.cell.stick;
    if st.mag == Mag::Neutral {
        return GamePad::new(btn, 0, 0);
    }
    let base = ref_yaw(st.r, o, c.yaw);
    let (centre, hw) = ang_range(st.ang);
    let yaw = (base as i32 + (centre + hw * c.fine_ang.clamp(-1.0, 1.0)).round() as i32) as i16;
    let mag = mag_range(st.mag, c.fine_mag);
    let (x, y) = raw_stick(s.cam_yaw, yaw, mag);
    GamePad::new(btn, x, y)
}

/// A choice at the cell's default point.
pub fn default_choice(cell: Cell) -> Choice {
    Choice { cell, yaw: 0, fine_ang: 0.0, fine_mag: if cell.stick.mag == Mag::Full { 1.0 } else { 0.0 } }
}

/// How a cell's buttons act this frame given the last pad, per button A, B, Z:
/// 0 up, 1 fresh press, 2 held, 3 released this frame.
pub fn edges(last: GamePad, b: Btn) -> [u8; 3] {
    let l = btn_of(last.button);
    let f = |bit: u8| match (l.0 & bit != 0, b.0 & bit != 0) {
        (false, false) => 0,
        (false, true) => 1,
        (true, true) => 2,
        (true, false) => 3,
    };
    [f(Btn::A), f(Btn::B), f(Btn::Z)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The game's reading of a raw stick: (stickMag, intended yaw offset from the camera), as in
    /// adjust_analog_stick and update_mario_joystick_inputs.
    fn read_stick(cam: i16, x: i8, y: i8) -> (f32, i32) {
        let f = |r: i8| if r >= 8 { r as f32 - 6.0 } else if r <= -8 { r as f32 + 6.0 } else { 0.0 };
        let (sx, sy) = (f(x), f(y));
        let mut mag = sx.hypot(sy);
        let k = if mag > 64.0 { 64.0 / mag } else { 1.0 };
        mag = mag.min(64.0);
        let yaw = ((sx * k).atan2(-(sy * k)) * 65536.0 / std::f32::consts::TAU).round() as i32 + cam as i32;
        (mag, yaw as i16 as i32)
    }

    #[test]
    fn realize_matches_class_bounds() {
        for cam in [0i16, 0x1234, -0x6000] {
            for yaw in [0i16, 0x4000, -0x2000, 0x7000] {
                for (m, lo, hi) in [(Mag::Short, 8.0f32, 46.5f32), (Mag::Full, 49.0, 64.0)] {
                    for f in [-1.0f32, 0.0, 1.0] {
                        let mag = mag_range(m, f);
                        let (x, yv) = raw_stick(cam, yaw, mag);
                        let (rm, ryaw) = read_stick(cam, x, yv);
                        assert!(rm >= lo - 1.5 && rm <= hi + 1.5, "{m:?} {f}: mag {rm} outside {lo}..{hi}");
                        let d = ((ryaw - yaw as i32) as i16) as i32;
                        // quantisation to integer raw units: about 0.9 deg at mag 64, ~7 deg at mag 8
                        assert!(d.abs() < (65536.0 / 360.0 * 8.0) as i32, "yaw off by {d} at mag {rm}");
                    }
                }
            }
        }
        // never below the deadzone for a non-neutral class
        let (x, y) = raw_stick(0, 0, mag_range(Mag::Short, -1.0));
        assert!(x != 0 || y != 0);
        // the Short class never makes a speed-kick dive (stickMag > 48), Full always does
        let (rm, _) = read_stick(0, raw_stick(0, 0, mag_range(Mag::Short, 1.0)).0, raw_stick(0, 0, mag_range(Mag::Short, 1.0)).1);
        assert!(rm <= 48.0);
        let (rm, _) = read_stick(0, raw_stick(0, 0, mag_range(Mag::Full, -1.0)).0, raw_stick(0, 0, mag_range(Mag::Full, -1.0)).1);
        assert!(rm > 48.0);
    }

    #[test]
    fn table_sorted_and_stand_ins_exist() {
        assert!(TABLE.windows(2).all(|w| w[0].0 < w[1].0));
        for a in [0x0C40_0201u32, 0x0400_0440, 0x0100_088C] {
            assert!(entry(a).is_some(), "stand-in {a:#x}");
        }
    }
}
