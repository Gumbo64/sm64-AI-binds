//! Unit tests of the mode policies (cannon.rs, water_air.rs, slide.rs,
//! pole.rs) and their registration in policies.rs.

use obrrt_core::{Arrival, St};

use super::policies::{self, POLICIES};
use super::{cannon, pole, slide, water_air};

pub(super) fn mk(zone: u16, action: u32, pos: [f32; 3], fv: f32, yaw: i16, floor_n: [f32; 3]) -> St {
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
        level: zone,
        area: 1,
        act: 1,
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
        floor_normal: floor_n,
        floor_obj: None,
        floor_obj_vy: None,
        floor_obj_y: None,
        action_timer: 0, cap_timer: 0,
        wall_normal: None,
        course_stars: [0; 25],
        save_flags: 0,
        objects: None,
    }
}


const ACT_WALKING: u32 = 0x0400_0440;
const ACT_BREASTSTROKE: u32 = 0x3000_24D0;

#[test]
fn cannon_closed_form_matches_frame_sequence() {
    for p in [0x200, 0x1000, 0x2000, 0x3000] {
        let (c, s) = (cannon::coss(p), cannon::sins(p));
        let (mut h, mut y) = (120.0 * c, 120.0 * s);
        let (mut fv, mut vy) = (100.0 * c, 100.0 * s);
        for n in 1..=150 {
            h += fv;
            y += vy;
            vy = (vy - 1.0f32).max(-75.0);
            fv = (fv - 0.05f32).max(10.0);
            let (mh, my) = cannon::at_frame(p, n as f32);
            assert!((mh - h).abs() < 0.5 && (my - y).abs() < 0.5, "p {p:#x} n {n}: {mh} {my} vs {h} {y}");
        }
    }
}

#[test]
fn cannon_solve_recovers_pitch() {
    for p0 in [0x0800, 0x1800, 0x3000] {
        let (d, dy) = cannon::at_frame(p0, 60.0);
        let lo = cannon::solve(d, dy, cannon::Root::Low).unwrap();
        let hi = cannon::solve(d, dy, cannon::Root::High).unwrap();
        assert!(lo.0 <= hi.0 && lo.1 <= hi.1 + 0.01);
        let got = if (lo.0 - p0).abs() < (hi.0 - p0).abs() { lo.0 } else { hi.0 };
        assert!((got - p0).abs() <= 0x20, "p0 {p0:#x}: lo {:#x} hi {:#x}", lo.0, hi.0);
    }
    // Out of range.
    assert!(cannon::solve(30000.0, 0.0, cannon::Root::Low).is_none());
}

#[test]
fn cannon_policies_only_in_cannon() {
    let s = mk(950, ACT_WALKING, [0.0; 3], 10.0, 0, [0.0, 1.0, 0.0]);
    for k in [cannon::Kind::Aim, cannon::Kind::Lob, cannon::Kind::Edge] {
        assert!(!cannon::plausible(k, &s, [0.0, 0.0, 3000.0], Arrival::Pass));
        assert!(cannon::pad_for(k, &s, [0.0, 0.0, 3000.0], Arrival::Pass, 0).is_none());
    }
    let mut c = mk(951, 0x1371, [0.0, 500.0, 0.0], 0.0, 0, [0.0; 3]);
    c.pitch = 0;
    let target = [0.0, 0.0, 4000.0];
    assert!(cannon::plausible(cannon::Kind::Aim, &c, target, Arrival::Pass));
    let (p, y) = cannon::aim(cannon::Kind::Aim, &c, target, Arrival::Pass).unwrap();
    assert_eq!(y, 0);
    // The solved arc passes within 30 of the aim point (+20 up).
    let (yy, _) = cannon::height_at(p, 4000.0).unwrap();
    assert!((c.pos[1] + yy - 20.0).abs() < 30.0, "{yy}");
    // Not aimed yet: no A, stick moving the pitch up (stick down).
    let pad = cannon::pad_for(cannon::Kind::Aim, &c, target, Arrival::Pass, 0).unwrap();
    assert_eq!(pad.button, 0);
    assert!(pad.stick_y < 0);
    // Aimed: A on an even frame.
    c.pitch = p as i16;
    let pad = cannon::pad_for(cannon::Kind::Aim, &c, target, Arrival::Pass, 2).unwrap();
    assert_ne!(pad.button & obrrt_core::buttons::A_BUTTON, 0);
    // A stop target lands shorter (the dive slide's run-out): from a
    // cannon level with the target (from 500 above it only the steep root
    // lands short of where pitch 0 does).
    c.pos[1] = 0.0;
    let (pp, _) = cannon::aim(cannon::Kind::Aim, &c, target, Arrival::Pass).unwrap();
    let (ps, _) = cannon::aim(cannon::Kind::Aim, &c, target, Arrival::Stop).unwrap();
    assert!(ps < pp, "{ps:#x} {pp:#x}");
    let (pl, _) = cannon::aim(cannon::Kind::Lob, &c, target, Arrival::Pass).unwrap();
    assert!(pl > 0x2000, "{pl:#x}");
}

#[test]
fn slide_out_by_floor() {
    let d0 = cannon::slide_out(60.0, 0);
    let d1 = cannon::slide_out(60.0, 0x13);
    assert!(d0 > 300.0 && d1 > d0);
}

#[test]
fn water_jump_arc() {
    let t = water_air::water_jump_down(100.0).unwrap();
    assert!((water_air::water_jump_h(t) - 100.0).abs() < 1.0);
    assert!(t > 11.0);
    assert!(water_air::water_jump_down(400.0).is_none());
}

#[test]
fn swim_policies_only_in_water() {
    let s = mk(952, ACT_WALKING, [0.0; 3], 10.0, 0, [0.0, 1.0, 0.0]);
    for k in [water_air::Swim::Tas, water_air::Swim::Surface, water_air::Swim::Exit] {
        assert!(!water_air::swim_plausible(k, &s, [0.0, 0.0, 1000.0]));
        assert!(water_air::swim_pad(k, &s, [0.0, 0.0, 1000.0], 0).is_none());
    }
    let mut w = mk(953, ACT_BREASTSTROKE, [0.0, -500.0, 0.0], 20.0, 0, [0.0; 3]);
    w.water_level = 0;
    // (swim-tas only without OBRRT_SWIM_HOLD=1, the test env)
    assert!(water_air::swim_plausible(water_air::Swim::Tas, &w, [0.0, -600.0, 1000.0]) != obrrt_core::config::current().objectives.swim_hold);
    assert!(water_air::swim_plausible(water_air::Swim::Exit, &w, [0.0, 50.0, 1000.0]));
    assert!(!water_air::swim_plausible(water_air::Swim::Exit, &w, [0.0, -50.0, 1000.0]));
    // Metal cap: not swimming.
    w.mario_flags = 0x4;
    assert!(!water_air::swim_plausible(water_air::Swim::Tas, &w, [0.0, -600.0, 1000.0]));
}

#[test]
fn fly_model() {
    let mut f = water_air::Flyer { pos: [0.0; 3], fv: 50.0, pitch: -0x2000, yaw: 0, pitch_vel: 0, yaw_vel: 0 };
    let v0 = f.fv;
    for _ in 0..10 {
        water_air::fly_step(&mut f, 0.0, 64.0); // stick up: dive
    }
    assert!(f.fv > v0, "diving gains speed: {}", f.fv);
    assert!(f.pitch >= -0x2AAA && f.pos[1] < 0.0);
    let mut g = water_air::Flyer { pos: [0.0; 3], fv: 40.0, pitch: 0x1000, yaw: 0, pitch_vel: 0, yaw_vel: 0 };
    for _ in 0..10 {
        water_air::fly_step(&mut g, 0.0, -64.0);
    }
    assert!(g.fv < 40.0, "climbing costs speed");
    // Turning right: stick x positive turns yaw down.
    let mut h = water_air::Flyer { pos: [0.0; 3], fv: 40.0, pitch: 0, yaw: 0, pitch_vel: 0, yaw_vel: 0 };
    for _ in 0..10 {
        water_air::fly_step(&mut h, 64.0, 0.0);
    }
    assert!(h.yaw < 0);
}

#[test]
fn fly_policies_only_flying() {
    let s = mk(954, ACT_WALKING, [0.0; 3], 10.0, 0, [0.0, 1.0, 0.0]);
    for k in [water_air::Fly::Plan, water_air::Fly::High, water_air::Fly::Land] {
        assert!(!water_air::fly_plausible(k, &s, [0.0, -500.0, 3000.0]));
        assert!(water_air::fly_pad(k, &s, [0.0, -500.0, 3000.0], Arrival::Pass, 0).is_none());
    }
    let mut f = mk(955, obrrt_core::layout::ACT_FLYING, [0.0, 1000.0, 0.0], 40.0, 0, [0.0; 3]);
    f.floor_y = -11000.0;
    // Target ahead and below: the plan dives (stick up).
    let p = water_air::fly_pad(water_air::Fly::Plan, &f, [0.0, 0.0, 3000.0], Arrival::Pass, 0).unwrap();
    assert!(p.stick_y > 0, "{p:?}");
    // Over the target, above: fly-land pounds.
    let p = water_air::fly_pad(water_air::Fly::Land, &f, [10.0, 500.0, 10.0], Arrival::Stop, 0).unwrap();
    assert_ne!(p.button & obrrt_core::buttons::Z_TRIG, 0);
}

#[test]
fn slide_policies_only_sliding() {
    let s = mk(27, ACT_WALKING, [0.0; 3], 10.0, 0, [0.0, 1.0, 0.0]);
    for k in [slide::Slide::Hold, slide::Slide::Course, slide::Slide::Sharp, slide::Slide::Jump] {
        assert!(!slide::slide_plausible(k, &s, [0.0, 0.0, 3000.0]));
        assert!(slide::slide_pad(k, &s, [0.0, 0.0, 3000.0], 0).is_none());
    }
    // A butt slide off a slide course on a default floor: not either.
    let b = mk(956, slide::ACT_BUTT_SLIDE, [0.0; 3], 40.0, 0, [0.0, 0.9, 0.43]);
    assert!(!slide::slide_plausible(slide::Slide::Hold, &b, [0.0, 0.0, 3000.0]));
    // On PSS (27, 1): the hold stick points along the velocity.
    let mut p = mk(27, slide::ACT_BUTT_SLIDE, [0.0; 3], 40.0, 0, [0.0, 0.9, 0.43]);
    p.vel = [40.0, 0.0, 0.0]; // +x
    assert!(slide::slide_plausible(slide::Slide::Course, &p, [0.0, 0.0, 3000.0]));
    // slide-hold needs the floor probe (no grid here): not offered.
    assert!(!slide::slide_plausible(slide::Slide::Hold, &p, [0.0, 0.0, 3000.0]));
    let pad = slide::slide_pad(slide::Slide::Hold, &p, [0.0, 0.0, 3000.0], 0).unwrap();
    // cam_yaw 0: stick x = 127 sin(rel), +x direction is rel = 90°.
    assert!(pad.stick_x > 100 && pad.stick_y.abs() < 10, "{pad:?}");
    // The course stick is at most 35° off the velocity.
    let pad = slide::slide_pad(slide::Slide::Course, &p, [-3000.0, 0.0, 0.0], 0).unwrap();
    let rel = (pad.stick_x as f32).atan2(-(pad.stick_y as f32)).to_degrees();
    assert!((rel - 90.0).abs() <= 36.0, "{rel}");
}

#[test]
fn brake_slide_states() {
    // Walking downhill fast on PSS: offered; uphill: not.
    let mut w = mk(27, ACT_WALKING, [0.0; 3], 30.0, 0, [0.0, 0.9, 0.43]);
    assert!(slide::slide_plausible(slide::Slide::Brake, &w, [0.0, -2000.0, 3000.0]));
    let pad = slide::slide_pad(slide::Slide::Brake, &w, [0.0, -2000.0, 3000.0], 0).unwrap();
    assert_ne!(pad.button & obrrt_core::buttons::U_CBUTTONS, 0);
    w.yaw = i16::MIN;
    assert!(!slide::slide_plausible(slide::Slide::Brake, &w, [0.0, -2000.0, 3000.0]));
}

#[test]
fn pole_policies_only_on_pole() {
    let s = mk(957, ACT_WALKING, [0.0; 3], 10.0, 0, [0.0, 1.0, 0.0]);
    for k in [pole::Pole::Top, pole::Pole::Kick, pole::Pole::Drop] {
        assert!(!pole::pole_plausible(k, &s, [0.0, 500.0, 300.0]));
        assert!(pole::pole_pad(k, &s, [0.0, 500.0, 300.0], 0).is_none());
    }
    let h = mk(958, obrrt_core::layout::ACT_HOLDING_POLE, [0.0, 300.0, 0.0], 0.0, 0, [0.0; 3]);
    // Target above, close: climb (stick up), no A.
    let p = pole::pole_pad(pole::Pole::Kick, &h, [0.0, 700.0, 200.0], 0).unwrap();
    assert!(p.stick_y > 0 && p.button == 0);
    // Higher targets need higher kick-offs.
    assert!(pole::kick_height([0.0, 500.0, 0.0], 200.0) > pole::kick_height([0.0, 300.0, 0.0], 200.0));
}

/// The identity rule: in walking and jumping states no control-mode
/// policy is plausible or plays a pad of its own (the speed techniques,
/// tech.rs, are ground/air policies and may be); and without
/// `OBRRT_MOVES_MODES=1` none of them is plausible anywhere.
#[test]
fn mode_policies_inert_elsewhere() {
    let n = POLICIES.len();
    let names: std::collections::HashSet<&str> = POLICIES.iter().map(|p| p.name()).collect();
    assert_eq!(names.len(), n);
    for (zone, action) in [(959u16, ACT_WALKING), (960, 0x0300_0880 /* ACT_JUMP */), (27, ACT_WALKING)] {
        let s = mk(zone, action, [0.0, 100.0, 0.0], 20.0, 0, [0.0, 1.0, 0.0]);
        for (i, p) in POLICIES.iter().enumerate() {
            let Some(m) = p.mode() else { continue };
            assert!(!policies::plausible(i, &s, [0.0, 600.0, 2000.0], Arrival::Pass), "switch off: {}", p.name());
            if matches!(m, policies::Mode::Tech(_)) {
                continue;
            }
            assert!(!policies::mode_plausible(*p, &s, [0.0, 600.0, 2000.0], Arrival::Pass), "{}", p.name());
            assert!(policies::mode_policy_pad(i, &s, [0.0, 600.0, 2000.0], Arrival::Pass, 0, 0).is_none(), "{}", p.name());
        }
    }
}

#[test]
fn tech_gates() {
    use super::tech::{Tech, tech_plausible};
    let mut w = mk(961, ACT_WALKING, [0.0; 3], 31.0, 0, [0.0, 1.0, 0.0]);
    assert!(tech_plausible(Tech::SpeedKick, &w, [0.0, 0.0, 3000.0], Arrival::Pass));
    assert!(tech_plausible(Tech::HighDive, &w, [0.0, 0.0, 3000.0], Arrival::Pass));
    // Near target / slow: not.
    assert!(!tech_plausible(Tech::SpeedKick, &w, [0.0, 0.0, 200.0], Arrival::Stop));
    w.forward_vel = 10.0;
    assert!(!tech_plausible(Tech::HighDive, &w, [0.0, 0.0, 3000.0], Arrival::Pass));
    // Swimming: none.
    let s = mk(962, ACT_BREASTSTROKE, [0.0; 3], 31.0, 0, [0.0, 1.0, 0.0]);
    for k in [Tech::SpeedKick, Tech::EdgeDive, Tech::HighDive] {
        assert!(!tech_plausible(k, &s, [0.0, 0.0, 3000.0], Arrival::Pass));
    }
    // The speed kick's B comes with a stick of at most ~48.
    let mut f = mk(963, ACT_WALKING, [0.0; 3], 31.0, 0, [0.0, 1.0, 0.0]);
    f.forward_vel = 31.0;
    let p = super::tech::tech_pad(Tech::SpeedKick, &f, [0.0, 0.0, 3000.0], 0).unwrap();
    assert_ne!(p.button & obrrt_core::buttons::B_BUTTON, 0);
    assert!((p.stick_x as i32).abs().max((p.stick_y as i32).abs()) <= 56, "{p:?}");
}

/// extra.rs's policies are appended, not offered without their switch, and
/// inert without it: obrrt2's plans draw as before.
#[test]
fn extra_policies_off_by_default() {
    assert_eq!(policies::BASE_POLICIES, 39);
    assert_eq!(POLICIES[38], policies::Policy::HighDive);
    assert!(POLICIES[..policies::BASE_POLICIES].iter().all(|p| p.extra().is_none()));
    assert!(POLICIES[policies::BASE_POLICIES..].iter().all(|p| p.extra().is_some()));
    if obrrt_core::sys::env::var_os("OBRRT_MOVES_EXTRA").is_some() {
        return;
    }
    assert_eq!(policies::offered(), policies::BASE_POLICIES);
    let s = mk(970, ACT_WALKING, [0.0, 100.0, 0.0], 20.0, 0, [0.0, 1.0, 0.0]);
    for i in policies::BASE_POLICIES..POLICIES.len() {
        assert!(!policies::plausible(i, &s, [0.0, 600.0, 2000.0], Arrival::Pass), "{}", POLICIES[i].name());
        assert!(policies::policy_pad(i, &s, [0.0, 600.0, 2000.0], Arrival::Pass, 0, 0).is_none(), "{}", POLICIES[i].name());
    }
}

/// The A+B kick (G15): A down all the way, B only on a landing's last frame
/// (landing actions ignore it) and in the stop action, with a short stick
/// where a long one would dive.
#[test]
fn jump_kick_buttons() {
    use super::extra::{Extra, extra_pad};
    use obrrt_core::buttons::{A_BUTTON, B_BUTTON};
    let t = [0.0, 500.0, 3000.0];
    let pad = |s: &St, f: usize| extra_pad(Extra::JumpKick, s, t, Arrival::Pass, f).unwrap();
    const FREEFALL_LAND: u32 = 0x0400_0471;
    const FREEFALL_LAND_STOP: u32 = 0x0C00_0232;
    const JUMP_KICK: u32 = 0x0180_08AC;
    let air = mk(971, JUMP_KICK, [0.0; 3], 25.0, 0, [0.0, 1.0, 0.0]);
    assert_eq!(pad(&air, 1).button, A_BUTTON);
    let mut land = mk(972, FREEFALL_LAND, [0.0; 3], 25.0, 0, [0.0, 0.8, -0.6]);
    assert_eq!(pad(&land, 1).button, A_BUTTON);
    land.action_timer = 3;
    assert_eq!(pad(&land, 1).button, A_BUTTON | B_BUTTON);
    // At 29+ a stick over 48 dives instead.
    let stop = mk(973, FREEFALL_LAND_STOP, [0.0; 3], 40.0, 0, [0.0, 1.0, 0.0]);
    let p = pad(&stop, 1);
    assert_eq!(p.button, A_BUTTON | B_BUTTON);
    assert!((p.stick_x as i32).pow(2) + (p.stick_y as i32).pow(2) <= 54 * 54, "{p:?}");
    // Facing well off the target: walks a frame to turn (the chain can't turn).
    let walk = mk(974, ACT_WALKING, [0.0; 3], 25.0, 0x4000, [0.0, 1.0, 0.0]);
    assert_eq!(pad(&walk, 0).button, A_BUTTON);
    let walk = mk(975, ACT_WALKING, [0.0; 3], 25.0, 0, [0.0, 1.0, 0.0]);
    assert_eq!(pad(&walk, 0).button, A_BUTTON | B_BUTTON);
}

/// A jump dives only above fv 28, a steep jump at any speed.
#[test]
fn dives_respect_the_games_gates() {
    use super::extra::{Extra, extra_pad};
    use obrrt_core::buttons::{A_BUTTON, B_BUTTON};
    let t = [0.0, 0.0, 3000.0];
    const ACT_JUMP: u32 = 0x0300_0880;
    const ACT_STEEP_JUMP: u32 = 0x0300_0885;
    const ACT_FREEFALL: u32 = 0x0100_088C;
    let mut j = mk(976, ACT_JUMP, [0.0; 3], 26.0, 0, [0.0, 1.0, 0.0]);
    j.vel = [0.0, 30.0, 26.0];
    assert_eq!(extra_pad(Extra::JumpDive, &j, t, Arrival::Pass, 0).unwrap().button & B_BUTTON, 0);
    j.forward_vel = 29.0;
    assert_ne!(extra_pad(Extra::JumpDive, &j, t, Arrival::Pass, 0).unwrap().button & B_BUTTON, 0);
    let s = mk(977, ACT_STEEP_JUMP, [0.0; 3], 5.0, 0, [0.0, 1.0, 0.0]);
    assert_eq!(extra_pad(Extra::DiveLadder, &s, t, Arrival::Pass, 0).unwrap().button, B_BUTTON);
    let f = mk(978, ACT_FREEFALL, [0.0; 3], 11.0, 0, [0.0, 1.0, 0.0]);
    assert_eq!(extra_pad(Extra::EdgeHop, &f, t, Arrival::Pass, 0).unwrap().button, B_BUTTON);
    let _ = A_BUTTON;
}
