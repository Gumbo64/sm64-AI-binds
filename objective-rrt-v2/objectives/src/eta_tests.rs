//! Unit tests of the v2 `eta` of the castle/travel/menu objectives (door,
//! star, painting, mario/*, menu) on synthetic states.

use obrrt_core::layout::*;
use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St, fold, start_eta, start_rank};

use crate::{door, mario, menu, painting, painting_table, star};

/// Walking (not stationary).
const ACT_WALKING: u32 = 0x0400_0440;

fn st(level: u32, area: u8, pos: [f32; 3]) -> St {
    St {
        frame: 100,
        pos,
        vel: [0.0; 3],
        forward_vel: 0.0,
        yaw: 0,
        pitch: 0,
        angle_vel: [0; 3],
        water_level: -11000,
        cam_yaw: 0,
        action: ACT_IDLE,
        mario_flags: 0,
        holding: false,
        held_slot: None,
        red_coins: 0,
        level: level as u16,
        area,
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
        floor_normal: [0.0, 1.0, 0.0],
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

fn obj(pos: [f32; 3], beh_params: u32, interact_type: u32) -> Obj {
    Obj {
        id: ObjId { bhv: 0, home: [pos[0] as i32, pos[1] as i32, pos[2] as i32] },
        slot: 0,
        bhv: 0,
        pos,
        home: pos,
        interact_type,
        interact_status: 0,
        action: 0,
        health: 0,
        value: 0,
        loot_coins: 0,
        beh_params,
        held_state: 0,
        intangible_timer: 0,
        yaw: 0,
        raw_1ac: [0; 4],
    }
}

/// `s` moved to `x` (along +x), `df` frames later.
fn at(s: &St, x: f32, df: u32) -> St {
    let mut t = s.clone();
    t.pos[0] = x;
    t.frame += df;
    t
}

fn frames(e: Eta) -> f64 {
    match e {
        Eta::Frames(f) => f,
        other => panic!("expected Frames, got {other:?}"),
    }
}

/// ETAs of `obj` along `traj`, folded in order.
fn etas(obj: &dyn Objective, traj: &[St]) -> Vec<Eta> {
    fold(obj, traj).1
}

#[test]
fn door_eta_falls_and_is_done_in_the_door_or_out_of_zone() {
    let b = st(LEVEL_CASTLE, 1, [0.0, 0.0, 0.0]);
    let d = door::Enter::new(&b, &obj([3000.0, 0.0, 0.0], 0, INTERACT_DOOR));
    let e = etas(&d, &[b.clone(), at(&b, 1000.0, 10), at(&b, 2000.0, 20)]);
    assert!(frames(e[0]) > frames(e[1]) && frames(e[1]) > frames(e[2]));
    assert!((frames(e[0]) - 100.0).abs() < 1e-6, "3000 units at 30 u/f");
    let mut inside = at(&b, 2900.0, 30);
    inside.action = ACT_PUSHING_DOOR;
    assert_eq!(d.eta(&inside, &b, &mut Mem::default()), Eta::Done);
    let out = st(LEVEL_CASTLE, 2, [0.0; 3]);
    assert_eq!(d.eta(&out, &b, &mut Mem::default()), Eta::Done);
}

#[test]
fn star_done_on_its_flag_lost_out_of_zone() {
    let mut b = st(LEVEL_BOB, 1, [0.0; 3]);
    b.course = 1;
    let s = star::Collect::new(&b, &obj([1500.0, 0.0, 0.0], 2 << 24, INTERACT_STAR_OR_KEY));
    assert!(start_eta(&s, &b) > start_eta(&s, &at(&b, 1000.0, 0)));
    let mut got = at(&b, 1500.0, 30);
    got.course_stars[0] = 1 << 2;
    assert_eq!(s.eta(&got, &b, &mut Mem::default()), Eta::Done);
    // Another star of the course is not this one.
    let mut other = at(&b, 1500.0, 30);
    other.course_stars[0] = 1 << 3;
    assert!(matches!(s.eta(&other, &b, &mut Mem::default()), Eta::Frames(_)));
    assert_eq!(s.eta(&st(LEVEL_CASTLE, 1, [0.0; 3]), &b, &mut Mem::default()), Eta::Lost);
}

#[test]
fn key_done_on_any_progress_gained() {
    let mut b = st(LEVEL_BOB, 1, [0.0; 3]);
    b.course = 0; // outside a course: a key
    let k = star::Collect::new(&b, &obj([500.0, 0.0, 0.0], 0, INTERACT_STAR_OR_KEY));
    assert!(matches!(start_eta(&k, &b), f if f.is_finite() && f > 0.0));
    let mut got = b.clone();
    got.keys = 1;
    assert_eq!(k.eta(&got, &b, &mut Mem::default()), Eta::Done);
}

#[test]
fn painting_done_on_its_level_lost_elsewhere() {
    let bob = &painting_table::PAINTINGS[0];
    let b = st(LEVEL_CASTLE, 1, [-3000.0, 0.0, 0.0]);
    let p = painting::Enter::new(&b, bob);
    let near = at(&b, -4000.0, 30);
    assert!(start_eta(&p, &b) > start_eta(&p, &near));
    assert_eq!(p.eta(&st(LEVEL_BOB, 1, [0.0; 3]), &b, &mut Mem::default()), Eta::Done);
    assert_eq!(p.eta(&st(LEVEL_CASTLE, 2, [0.0; 3]), &b, &mut Mem::default()), Eta::Lost);
}

#[test]
fn posdist_done_in_radius_lost_out_of_zone() {
    let b = st(LEVEL_BOB, 1, [0.0; 3]);
    let o = mario::PosDist::new(&b, [1500.0, 0.0, 0.0]);
    let e = etas(&o, &[b.clone(), at(&b, 500.0, 15), at(&b, 1400.0, 40)]);
    assert!(frames(e[0]) > frames(e[1]));
    assert_eq!(e[2], Eta::Done);
    assert_eq!(o.eta(&st(LEVEL_BOB, 2, [0.0; 3]), &b, &mut Mem::default()), Eta::Lost);
}

#[test]
fn postime_deadline_is_lost() {
    // b reached (2000, 0, 0) at frame 400.
    let mut b = st(LEVEL_BOB, 1, [2000.0, 0.0, 0.0]);
    b.frame = 400;
    let o = mario::PosTime::new(&b);
    let s0 = st(LEVEL_BOB, 1, [0.0; 3]); // frame 100
    let e = etas(&o, &[s0.clone(), at(&s0, 1000.0, 30), at(&s0, 1900.0, 60)]);
    assert!(frames(e[0]) > frames(e[1]));
    assert_eq!(e[2], Eta::Done);
    // Arriving at b's frame (deadline 399) is too late.
    assert_eq!(o.eta(&at(&s0, 1900.0, 300), &s0, &mut Mem::default()), Eta::Lost);
    // Too far to make it even at top speed.
    assert_eq!(o.eta(&at(&s0, -20000.0, 10), &s0, &mut Mem::default()), Eta::Lost);
}

#[test]
fn postime_near_done_before_the_record_lost_after() {
    use std::sync::{Arc, RwLock};
    // A node of b's zone and context reached (2000, 0, 0) at frame 400;
    // one of another context got there at frame 50 (it doesn't count).
    let b = st(LEVEL_BOB, 1, [0.0; 3]); // frame 100
    let c = obrrt_core::records::ctx(&b);
    let mut r = obrrt_core::records::Records::default();
    r.insert(b.zone(), c, [2000.0, 0.0, 0.0], 400);
    r.insert(b.zone(), (c.0, c.1 + 1, c.2), [2000.0, 0.0, 0.0], 50);
    obrrt_core::records::install(Some(Arc::new(RwLock::new(r))));
    // Never reached: no objective.
    assert!(mario::PosTime::sample(&b, [-2000.0, 0.0, 0.0]).is_none());
    let o = mario::PosTime::sample(&b, [2100.0, 0.0, 0.0]).expect("a record near (2100, 0, 0)");
    obrrt_core::records::install(None);
    assert!(o.base_criteria(&b));
    // Another context can't start it.
    assert!(!o.base_criteria(&St { stars: 1, ..b.clone() }));
    let e = etas(&o, &[b.clone(), at(&b, 1000.0, 30), at(&b, 2000.0, 60)]);
    assert!(frames(e[0]) > frames(e[1]));
    assert_eq!(frames(e[1]), obrrt_core::frames_for(1100.0));
    assert_eq!(e[2], Eta::Done);
    // The deadline is the record (400) minus the margin (1): Done at 398,
    // Lost at 399 and at the record itself.
    assert_eq!(o.eta(&at(&b, 2000.0, 298), &b, &mut Mem::default()), Eta::Done);
    assert_eq!(o.eta(&at(&b, 2000.0, 299), &b, &mut Mem::default()), Eta::Lost);
    assert_eq!(o.eta(&at(&b, 2000.0, 300), &b, &mut Mem::default()), Eta::Lost);
    // Short of the radius before the deadline: Frames; too far for even
    // MAX_SPEED: Lost.
    assert!(matches!(o.eta(&at(&b, 1000.0, 250), &b, &mut Mem::default()), Eta::Frames(_)));
    assert_eq!(o.eta(&at(&b, -20000.0, 10), &b, &mut Mem::default()), Eta::Lost);
    // Out of the zone: Lost.
    assert_eq!(o.eta(&st(LEVEL_BOB, 2, [2000.0, 0.0, 0.0]), &b, &mut Mem::default()), Eta::Lost);
    // The deadline bucket and context are in the key.
    assert!(o.key().contains("by~39"), "{}", o.key());
    assert_ne!(o.key(), mario::PosTime::new(&b).key());
}

#[test]
fn settle_done_when_still_after_the_start() {
    let mut b = st(LEVEL_BOB, 1, [0.0; 3]);
    b.action = ACT_WALKING;
    b.forward_vel = 20.0;
    let o = mario::Settle::new(&b);
    let mut slower = at(&b, 0.0, 5);
    slower.forward_vel = 10.0;
    let still = at(&b, 0.0, 15);
    let still = St { action: ACT_IDLE, forward_vel: 0.0, ..still };
    let e = etas(&o, &[b.clone(), slower, still.clone()]);
    assert!(frames(e[0]) > frames(e[1]));
    assert_eq!(e[2], Eta::Done);
    // Already still at the start: not met at the start itself.
    assert!(matches!(o.eta(&still, &still, &mut Mem::default()), Eta::Frames(_)));
}

#[test]
fn wait_counts_down_n_frames() {
    let b = st(LEVEL_BOB, 1, [0.0; 3]);
    let o = mario::Wait::new(&b, 30);
    assert_eq!(start_eta(&o, &b), 30.0);
    assert_eq!(o.eta(&at(&b, 0.0, 29), &b, &mut Mem::default()), Eta::Frames(1.0));
    assert_eq!(o.eta(&at(&b, 0.0, 30), &b, &mut Mem::default()), Eta::Done);
}

#[test]
fn gotocastle_done_in_the_castle() {
    let b = st(LEVEL_BOB, 1, [0.0; 3]);
    let o = mario::GoToCastle::new(&b);
    assert!(o.fixed().is_some());
    assert!(matches!(o.eta(&b, &b, &mut Mem::default()), Eta::Frames(_)));
    assert_eq!(o.eta(&st(LEVEL_CASTLE_GROUNDS, 1, [0.0; 3]), &b, &mut Mem::default()), Eta::Done);
}

fn dialog_state(id: i16, line: i8, df: u32) -> St {
    let mut s = st(LEVEL_BOB, 1, [0.0; 3]);
    s.dialog = id;
    s.dialog_choice = true;
    s.menu_line = line;
    s.frame += df;
    s
}

#[test]
fn dialog_done_on_close_with_its_line_only() {
    let b = dialog_state(5, 1, 0);
    let yes = menu::Dialog::options(&b).into_iter().find(|o| o.kind() == "Dialog::Yes").unwrap();
    let no = menu::Dialog::options(&b).into_iter().find(|o| o.kind() == "Dialog::No").unwrap();
    let closed = St { dialog: -1, ..dialog_state(5, 1, 3) };
    let traj = [b.clone(), dialog_state(5, 1, 1), dialog_state(5, 1, 2), closed.clone()];
    let ey = etas(&yes, &traj);
    assert!(matches!(ey[0], Eta::Frames(_)) && matches!(ey[2], Eta::Frames(_)));
    assert_eq!(ey[3], Eta::Done);
    assert_eq!(etas(&no, &traj)[3], Eta::Lost);
    // Never met at the start, even if the dialog is closed there.
    assert!(matches!(start_eta(&yes, &closed), f if f.is_finite()));
}

#[test]
fn save_menu_done_on_close_with_its_option() {
    let mut b = st(LEVEL_BOB, 1, [0.0; 3]);
    b.menu_mode = MENU_MODE_RENDER_COURSE_COMPLETE_SCREEN as i16;
    b.menu_line = MENU_OPT_SAVE_AND_CONTINUE as i8;
    let opts = menu::SaveMenu::options();
    let closed = St { menu_mode: -1, frame: b.frame + 1, ..b.clone() };
    let traj = [b.clone(), closed];
    assert_eq!(etas(&opts[0], &traj)[1], Eta::Done);
    assert_eq!(etas(&opts[1], &traj)[1], Eta::Lost);
}

/// Start bias (v1's non-distance rankings): GoToCastle prefers more stars,
/// then more coins; a dialog its earliest state; the ETA stays pure.
#[test]
fn start_bias_ranks_like_v1() {
    let b = st(LEVEL_BOB, 1, [0.0; 3]);
    let o = mario::GoToCastle::new(&b);
    let more = |stars: i16, coins: i16| St { stars, coins, ..b.clone() };
    assert_eq!(start_eta(&o, &more(5, 0)), start_eta(&o, &more(0, 0)));
    assert!(start_rank(&o, &more(1, 0)) < start_rank(&o, &more(0, 50)));
    assert!(start_rank(&o, &more(1, 10)) < start_rank(&o, &more(1, 0)));
    let d = dialog_state(5, 1, 0);
    let yes = menu::Dialog::options(&d).into_iter().find(|o| o.kind() == "Dialog::Yes").unwrap();
    let later = dialog_state(5, 1, 40);
    assert_eq!(start_eta(&yes, &d), start_eta(&yes, &later));
    assert_eq!(start_rank(&yes, &later) - start_rank(&yes, &d), 40.0);
    // No bias: the rank is the ETA.
    let p = obrrt_core::Objective::start_bias(&star_obj(), &d);
    assert_eq!(p, 0.0);
}

fn star_obj() -> star::Collect {
    let b = st(LEVEL_BOB, 1, [0.0; 3]);
    let mut b2 = b.clone();
    b2.objects = Some(std::sync::Arc::new(vec![obj([100.0, 0.0, 0.0], 0, INTERACT_STAR_OR_KEY)]));
    star::Collect::new(&b2, &b2.objects()[0])
}
