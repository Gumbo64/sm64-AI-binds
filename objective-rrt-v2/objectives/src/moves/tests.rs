//! classify on synthetic states over synthetic floors.

use super::*;
use obrrt_core::geo::Tri;

/// A test zone per test, so the global grid cache doesn't mix them.
fn st(zone: u16, action: u32, pos: [f32; 3], fv: f32, yaw: i16, floor_n: [f32; 3]) -> St {
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

/// A quad over x0..x1, z0..z1 with height y0 at z0 and y1 at z1.
fn quad(x0: f32, x1: f32, z0: f32, z1: f32, y0: f32, y1: f32) -> [Tri; 2] {
    let (dz, dy) = (z1 - z0, y1 - y0);
    let l = (dz * dz + dy * dy).sqrt();
    let n = [0.0, dz / l, -dy / l];
    [
        Tri { v: [[x0, y0, z0], [x1, y0, z0], [x1, y1, z1]], n, kind: 0, room: 0 },
        Tri { v: [[x0, y0, z0], [x1, y1, z1], [x0, y1, z1]], n, kind: 0, room: 0 },
    ]
}

const UP: [f32; 3] = [0.0, 1.0, 0.0];
/// Yaw facing +z.
const NORTH: i16 = 0;

#[test]
fn no_grid_flat_is_run() {
    let s = st(900, ACT_WALKING, [0.0, 0.0, 0.0], 20.0, NORTH, UP);
    assert_eq!(classify(&s, [0.0, 0.0, 600.0]), MoveKind::Run);
    assert_eq!(key(&s, [0.0, 0.0, 600.0]).geom, GeomClass::Unknown);
}

#[test]
fn steep_floor_without_grid_is_uphill_kick() {
    // A 40° slope rising towards +z: normal (0, cos, -sin).
    let a = 40f32.to_radians();
    let n = [0.0, a.cos(), -a.sin()];
    let s = st(901, ACT_WALKING, [0.0, 0.0, 0.0], 5.0, NORTH, n);
    assert_eq!(key(&s, [0.0, 500.0, 600.0]).geom, GeomClass::SteepUp);
    assert_eq!(classify(&s, [0.0, 500.0, 600.0]), MoveKind::UphillKick);
    // Facing down it (target downhill): not a climb.
    assert_ne!(classify(&s, [0.0, -500.0, -600.0]), MoveKind::UphillKick);
}

#[test]
fn gentle_slope_is_run() {
    let a = 10f32.to_radians();
    let n = [0.0, a.cos(), -a.sin()];
    let s = st(902, ACT_WALKING, [0.0, 0.0, 0.0], 20.0, NORTH, n);
    assert_eq!(classify(&s, [0.0, 100.0, 600.0]), MoveKind::Run);
}

#[test]
fn grid_slope_ahead_is_uphill_kick() {
    // Flat to z 100, then rising 1 per 1 (45°) to z 1000.
    let mut t = Vec::new();
    t.extend(quad(-500.0, 500.0, -500.0, 100.0, 0.0, 0.0));
    t.extend(quad(-500.0, 500.0, 100.0, 1000.0, 0.0, 900.0));
    geom::insert((903, 1), &t);
    let s = st(903, ACT_WALKING, [0.0, 0.0, 0.0], 15.0, NORTH, UP);
    assert_eq!(classify(&s, [0.0, 800.0, 900.0]), MoveKind::UphillKick);
}

#[test]
fn level_far_run_up_is_long_jump() {
    let t = quad(-500.0, 500.0, -500.0, 3000.0, 0.0, 0.0);
    geom::insert((904, 1), &t);
    let s = st(904, ACT_WALKING, [0.0, 0.0, 0.0], 28.0, NORTH, UP);
    assert_eq!(key(&s, [0.0, 0.0, 2500.0]).geom, GeomClass::Level);
    assert_eq!(classify(&s, [0.0, 0.0, 2500.0]), MoveKind::LongJumpChain);
    // Slow, or facing away: just run.
    let slow = st(904, ACT_WALKING, [0.0, 0.0, 0.0], 8.0, NORTH, UP);
    assert_eq!(classify(&slow, [0.0, 0.0, 2500.0]), MoveKind::Run);
    let away = st(904, ACT_WALKING, [0.0, 0.0, 0.0], 28.0, i16::MIN, UP);
    assert_eq!(classify(&away, [0.0, 0.0, 2500.0]), MoveKind::Run);
    // A target the jump would overshoot: run in.
    assert!(!lands_short(Jump::Long, &Ctx::new(&s, [0.0, 0.0, 700.0]), false, 150.0));
    assert_eq!(classify(&s, [0.0, 0.0, 700.0]), MoveKind::Run);
    // On the landing of one: chain.
    let land = st(904, ACT_LONG_JUMP_LAND, [0.0, 0.0, 0.0], 30.0, NORTH, UP);
    assert_eq!(classify(&land, [0.0, 0.0, 2500.0]), MoveKind::LongJumpChain);
}

#[test]
fn step_ahead_is_ledge_jump() {
    let mut t = Vec::new();
    t.extend(quad(-500.0, 500.0, -500.0, 120.0, 0.0, 0.0));
    t.extend(quad(-500.0, 500.0, 120.0, 1000.0, 120.0, 120.0));
    geom::insert((905, 1), &t);
    let s = st(905, ACT_WALKING, [0.0, 0.0, 0.0], 20.0, NORTH, UP);
    assert!(matches!(key(&s, [0.0, 120.0, 600.0]).geom, GeomClass::StepUp(2)));
    assert_eq!(classify(&s, [0.0, 120.0, 600.0]), MoveKind::LedgeJump);
    // Far from it still: run.
    let far = st(905, ACT_WALKING, [0.0, 0.0, -400.0], 10.0, NORTH, UP);
    assert_eq!(classify(&far, [0.0, 120.0, 600.0]), MoveKind::Run);
}

#[test]
fn gap_with_far_side_is_jump_and_pit_is_stop() {
    let mut t = Vec::new();
    t.extend(quad(-500.0, 500.0, -500.0, 60.0, 0.0, 0.0));
    t.extend(quad(-500.0, 500.0, 260.0, 1500.0, 0.0, 0.0));
    geom::insert((906, 1), &t);
    let s = st(906, ACT_WALKING, [0.0, 0.0, 0.0], 15.0, NORTH, UP);
    assert_eq!(key(&s, [0.0, 0.0, 800.0]).geom, GeomClass::Gap);
    assert_eq!(classify(&s, [0.0, 0.0, 800.0]), MoveKind::GapJump);
    let fast = st(906, ACT_WALKING, [0.0, 0.0, 0.0], 30.0, NORTH, UP);
    assert!(matches!(classify(&fast, [0.0, 0.0, 1200.0]), MoveKind::GapJump | MoveKind::GapLongJump));
    // Over the void (no floor at all): never off the edge.
    geom::insert((907, 1), &quad(-500.0, 500.0, -500.0, 60.0, 0.0, 0.0));
    let p = st(907, ACT_WALKING, [0.0, 0.0, 0.0], 15.0, NORTH, UP);
    assert_eq!(key(&p, [0.0, 0.0, 800.0]).geom, GeomClass::Pit);
    assert_eq!(classify(&p, [0.0, 0.0, 800.0]), MoveKind::FallSafe);
    // A harmless drop (400) with no far side: stop at the edge, unless
    // the target is down there.
    let mut t = Vec::new();
    t.extend(quad(-500.0, 500.0, -500.0, 60.0, 0.0, 0.0));
    t.extend(quad(-500.0, 500.0, 60.0, 2000.0, -400.0, -400.0));
    geom::insert((917, 1), &t);
    let q = st(917, ACT_WALKING, [0.0, 0.0, 0.0], 15.0, NORTH, UP);
    assert_eq!(classify(&q, [0.0, 0.0, 800.0]), MoveKind::Stop);
    assert_eq!(classify(&q, [0.0, -400.0, 800.0]), MoveKind::Run);
}

#[test]
fn cliff_and_wall_is_wall_slide() {
    let mut t = Vec::new();
    t.extend(quad(-500.0, 500.0, -500.0, 100.0, 0.0, 0.0));
    t.extend(quad(-500.0, 500.0, 100.0, 1000.0, 1000.0, 1000.0));
    geom::insert((908, 1), &t);
    let mut s = st(908, ACT_WALKING, [0.0, 0.0, 50.0], 2.0, NORTH, UP);
    s.wall_normal = Some([0.0, 0.0, -1.0]);
    assert_eq!(key(&s, [300.0, 1000.0, 600.0]).geom, GeomClass::Cliff);
    assert_eq!(classify(&s, [300.0, 1000.0, 600.0]), MoveKind::WallSlide);
    // The pad's stick runs along the wall (x), towards the target's side.
    let p = pad_for(MoveKind::WallSlide, &s, [300.0, 1000.0, 600.0], 0);
    assert!(p.stick_x > 100 && p.stick_y.abs() < 10, "{:?}", (p.stick_x, p.stick_y));
}

#[test]
fn state_moves() {
    let flat = [0.0, 1.0, 0.0];
    let mut g = st(909, ACT_LEDGE_GRAB, [0.0, 0.0, 0.0], 0.0, NORTH, flat);
    assert_eq!(classify(&g, [0.0, 100.0, 500.0]), MoveKind::LedgeClimb);
    g.action = ACT_AIR_HIT_WALL;
    assert_eq!(classify(&g, [0.0, 400.0, 500.0]), MoveKind::WallKick);
    g.action = 0x0084_0452; // ACT_BUTT_SLIDE
    g.vel = [0.0, 0.0, 20.0];
    assert_eq!(classify(&g, [0.0, 400.0, 500.0]), MoveKind::SlideEscape);
    assert_eq!(classify(&g, [0.0, -400.0, 500.0]), MoveKind::SlideRide);
    g.action = ACT_DIVE_SLIDE;
    assert_eq!(classify(&g, [0.0, 0.0, 500.0]), MoveKind::DiveRollout);
    g.action = ACT_JUMP;
    assert_eq!(classify(&g, [0.0, 0.0, 500.0]), MoveKind::Air);
}

#[test]
fn pads_press_on_edges() {
    let flat = [0.0, 1.0, 0.0];
    // Landing: A at once.
    let land = st(910, ACT_FREEFALL_LAND, [0.0, 0.0, 0.0], 5.0, NORTH, flat);
    assert_eq!(pad_for(MoveKind::UphillKick, &land, [0.0, 300.0, 500.0], 1).button, A_BUTTON);
    let walk = st(910, ACT_WALKING, [0.0, 0.0, 0.0], 5.0, NORTH, flat);
    // Walking: A held, B on even frames (the kick).
    assert_eq!(pad_for(MoveKind::UphillKick, &walk, [0.0, 300.0, 500.0], 1).button, A_BUTTON);
    assert_eq!(pad_for(MoveKind::UphillKick, &walk, [0.0, 300.0, 500.0], 2).button, A_BUTTON | B_BUTTON);
    // In the jump: A while rising, let go falling.
    let mut jump = st(910, ACT_JUMP, [0.0, 0.0, 0.0], 10.0, NORTH, flat);
    jump.vel[1] = 30.0;
    assert_eq!(pad_for(MoveKind::UphillKick, &jump, [0.0, 300.0, 500.0], 1).button, A_BUTTON);
    jump.vel[1] = 15.0;
    assert_eq!(pad_for(MoveKind::UphillKick, &jump, [0.0, 300.0, 500.0], 1).button, 0);
    // Long jump landing: Z+A.
    let lj = st(910, ACT_LONG_JUMP_LAND, [0.0, 0.0, 0.0], 30.0, NORTH, flat);
    assert_eq!(pad_for(MoveKind::LongJumpChain, &lj, [0.0, 0.0, 2500.0], 1).button, Z_TRIG | A_BUTTON);
    // Hit wall: A at once (the window is the next frame).
    let w = st(910, ACT_AIR_HIT_WALL, [0.0, 0.0, 0.0], 0.0, NORTH, flat);
    assert_eq!(pad_for(MoveKind::WallKick, &w, [0.0, 300.0, 500.0], 0).button, A_BUTTON);
}

#[test]
fn every_move_has_a_name() {
    let mut names: Vec<_> = MoveKind::ALL.iter().map(|k| k.name()).collect();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), MoveKind::ALL.len());
}

#[test]
fn ground_pound_and_fall_safe() {
    let flat = [0.0, 1.0, 0.0];
    // Right over the target with speed: pound.
    let mut p = st(911, ACT_JUMP, [0.0, 800.0, 0.0], 30.0, NORTH, flat);
    p.floor_y = 0.0;
    p.vel[1] = 5.0;
    assert_eq!(classify(&p, [0.0, 0.0, 30.0]), MoveKind::GroundPound);
    // Slow: just fall.
    p.forward_vel = 2.0;
    assert_ne!(classify(&p, [0.0, 0.0, 30.0]), MoveKind::GroundPound);
    // A ledge at y 0 ending at z 60 over floors `depth` below; a target
    // level and far beyond (no far side): stop at the edge of a drop that
    // hurts, by health.
    let ledge = |zone: u16, depth: f32| {
        let mut t = Vec::new();
        t.extend(quad(-500.0, 500.0, -500.0, 60.0, 0.0, 0.0));
        t.extend(quad(-500.0, 500.0, 60.0, 3000.0, -depth, -depth));
        geom::insert((zone, 1), &t);
    };
    ledge(915, 3500.0);
    let s = st(915, ACT_WALKING, [0.0, 0.0, 0.0], 20.0, NORTH, flat);
    assert_eq!(classify(&s, [0.0, 0.0, 1500.0]), MoveKind::FallSafe);
    // Even with the target down there (an 82-frame knockback and 4 wedges).
    assert_eq!(classify(&s, [0.0, -3500.0, 1500.0]), MoveKind::FallSafe);
    ledge(916, 1500.0);
    let mut s = st(916, ACT_WALKING, [0.0, 0.0, 0.0], 20.0, NORTH, flat);
    assert_ne!(classify(&s, [0.0, -1500.0, 1500.0]), MoveKind::FallSafe);
    s.health = 0x280;
    assert_eq!(classify(&s, [0.0, -1500.0, 1500.0]), MoveKind::FallSafe);
    assert_eq!(pad_for(MoveKind::FallSafe, &s, [0.0, -1500.0, 1500.0], 0).stick_x, 0);
}

#[test]
fn slides_ride_or_escape() {
    // A slope falling towards +z (normal leans +z).
    let a = 30f32.to_radians();
    let n = [0.0, a.cos(), a.sin()];
    let mut s = st(912, ACT_DIVE_SLIDE, [0.0, 0.0, 0.0], 40.0, NORTH, n);
    s.vel = [0.0, -10.0, 40.0];
    assert_eq!(classify(&s, [0.0, -800.0, 1500.0]), MoveKind::BellySlide);
    // Target uphill: roll out.
    assert_eq!(classify(&s, [0.0, 800.0, -1500.0]), MoveKind::DiveRollout);
    s.action = 0x0084_0452; // ACT_BUTT_SLIDE
    assert_eq!(classify(&s, [0.0, -800.0, 1500.0]), MoveKind::SlideRide);
    assert_eq!(classify(&s, [0.0, 800.0, -1500.0]), MoveKind::SlideEscape);
    // Slide kick rhythm: Z from walking, Z+B in the crouch slide, A in its slide.
    let w = st(912, ACT_WALKING, [0.0, 0.0, 0.0], 25.0, NORTH, n);
    assert_eq!(pad_for(MoveKind::SlideKick, &w, [0.0, -800.0, 1500.0], 0).button, Z_TRIG);
    let c = st(912, ACT_CROUCH_SLIDE, [0.0, 0.0, 0.0], 25.0, NORTH, n);
    assert_eq!(pad_for(MoveKind::SlideKick, &c, [0.0, -800.0, 1500.0], 1).button, Z_TRIG | B_BUTTON);
}

/// A wall in the plane z = `z`, facing `-sign` z (so `sign` = 1 blocks
/// movement towards +z), x -500..500, y -100..2000.
fn wall(z: f32, sign: f32) -> [Tri; 2] {
    let n = [0.0, 0.0, -sign];
    [
        Tri { v: [[-500.0, -100.0, z], [500.0, -100.0, z], [500.0, 2000.0, z]], n, kind: 0, room: 0 },
        Tri { v: [[-500.0, -100.0, z], [500.0, 2000.0, z], [-500.0, 2000.0, z]], n, kind: 0, room: 0 },
    ]
}

#[test]
fn walls_reverse_and_chain() {
    let floor = quad(-500.0, 500.0, -2000.0, 2000.0, 0.0, 0.0);
    let mut w = Vec::new();
    w.extend(wall(150.0, 1.0));
    geom::insert_with_walls((913, 1), &floor, &w);
    // Running at the wall (+z) with the target behind: kick off it.
    let s = st(913, ACT_WALKING, [0.0, 0.0, 0.0], 30.0, NORTH, UP);
    let ah = geom::ahead(&s, [0.0, 1.0], 1000.0, 0.0).unwrap();
    assert!(ah.wall.is_some_and(|d| (d - 150.0).abs() < 1.0), "{:?}", ah.wall);
    assert_eq!(classify(&s, [0.0, 0.0, -1500.0]), MoveKind::WallKick);
    // The stick goes into the wall (along the facing), with A.
    let p = pad_for(MoveKind::WallKick, &s, [0.0, 0.0, -1500.0], 0);
    assert!(p.stick_y < -100, "{}", p.stick_y);
    // Walls are one-sided: from behind it, nothing.
    let behind = st(913, ACT_WALKING, [0.0, 0.0, 300.0], 30.0, i16::MIN, UP);
    assert!(geom::ahead(&behind, [0.0, -1.0], 1000.0, 0.0).unwrap().wall.is_none());
    // Two facing walls 300 apart and the target above: chain kicks.
    let mut w2 = Vec::new();
    w2.extend(wall(150.0, 1.0));
    w2.extend(wall(-150.0, -1.0));
    geom::insert_with_walls((914, 1), &floor, &w2);
    let c = st(914, ACT_WALKING, [0.0, 0.0, 0.0], 20.0, NORTH, UP);
    assert_eq!(key(&c, [0.0, 800.0, 400.0]).geom, GeomClass::Cliff);
    assert_eq!(classify(&c, [0.0, 800.0, 400.0]), MoveKind::WallKick);
}
