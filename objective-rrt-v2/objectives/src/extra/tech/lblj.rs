//! `Tech::Lblj` (`lblj`): the lobby backwards long jump past the 8-star
//! door, the only way to Bowser in the Dark World (and key 1) with fewer
//! than 8 stars.
//!
//! The mechanism (decomp):
//! - The 8-star door is two `bhvStarDoor`s at (-2706, 512, -1409) and
//!   (-2598, 512, -1517), yaw 45° (levels/castle_inside/script.c:23–24), on
//!   the 512 landing left of the main stairs. interact_door opens it only
//!   with `numStars >= 8` (interaction.c). The corridor behind it runs along
//!   the door's normal to `bhvCastleFloorTrap` at (-5513, 717, -4324)
//!   (script.c:253), which opens when Mario stands on it, whatever his stars
//!   (castle_floor_trap.inc.c), and drops him into the BitDW warp.
//! - A long jump multiplies forward speed by 1.5 and caps it only above +48
//!   (set_mario_action_airborne, mario.c "(BLJ's)"); air drag below -16 is
//!   only +2 a frame (update_air_without_turn, mario_actions_airborne.c). On
//!   ACT_LONG_JUMP_LAND, A with Z held long-jumps again
//!   (act_long_jump_land, sLongJumpLandAction; the US build has no Shindou
//!   reset). If a rising floor (stairs, a slope) or a ceiling ends each
//!   backwards long jump within a frame or two, the speed grows ×1.5 a jump.
//!   Mario moves in quarter steps of vel/4 (mario_step.c), so at a few
//!   hundred units a frame he passes through the door's collision.
//!
//! The objective: from lobby states in front of the door with fewer than 8
//! stars, end up in the corridor behind it (a new place) or in BitDW. The
//! hints only make the BLJ available, the GA does the rest: in a long jump
//! or its landing the stick is held back (away from Mario's facing) and A+Z
//! come on alternate frames (a press needs a release; Z must be down for A
//! to count on the landing). Elsewhere the steer target is the trap floor
//! and no button is suggested (Z in most air actions would ground-pound).
//! The ETA is blj::build_eta (the setup, then the x1.5 re-jumps left to
//! blj::CHAIN_SPEED), so a walk earns nothing. Success
//! needs a chain built in the trajectory: a chain at blj::CHAIN_SPEED (the
//! hand-off to Tech::HyperWalk), or the corridor / BitDW once a chain
//! reached blj::CHAIN_BUILT.

use obrrt_core::layout::LEVEL_CASTLE;
use obrrt_core::{Eta, Mem, Objective, Precision, St};

use super::blj;

const LEVEL_BITDW: u16 = 17;

/// Midpoint of the 8-star door pair.
const DOOR: [f32; 3] = [-2652.0, 512.0, -1463.0];
/// The door's front normal (yaw 45°): towards the lobby; behind is -N.
const N: [f32; 2] = [std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2];
/// The BitDW entry: the warp under the trap floor (warps/table.rs "bitdw").
const TRAP: [f32; 3] = [-5513.0, 512.0, -4324.0];
/// Base states: within this (horizontal) of the door, in front of it.
const BASE_RADIUS: f32 = 2600.0;

/// Signed distance of `p` past the door plane, into the corridor
/// (negative in the lobby), and its offset across the corridor.
fn behind(p: [f32; 3]) -> (f32, f32) {
    let (dx, dz) = (p[0] - DOOR[0], p[2] - DOOR[2]);
    (-(dx * N[0] + dz * N[1]), dx * N[1] - dz * N[0])
}

fn in_lobby(s: &St) -> bool {
    s.level as u32 == LEVEL_CASTLE && s.area == 1
}

/// Whether `s` is in the corridor behind the 8-star door: past the door
/// plane by more than its thickness, within the corridor's width (about
/// ±360, castle_inside area 1 collision) and on its floor level, up to the
/// trap floor. The BoB room and the rest of the lobby's left wing are also
/// behind the plane but far off the corridor's axis.
pub fn behind_door(s: &St) -> bool {
    let (along, across) = behind(s.pos);
    in_lobby(s) && (150.0..4600.0).contains(&along) && across.abs() < 600.0 && s.pos[1] > 400.0
}

pub struct Lblj;

impl Lblj {
    /// Lobby states in front of the door with fewer than 8 stars (with 8
    /// the door just opens) and no key yet.
    pub fn start(s: &St) -> bool {
        in_lobby(s)
            && s.stars < 8
            && s.keys == 0
            && s.pos[1] > -60.0
            && behind(s.pos).0 < 0.0
            && s.hdist(DOOR) < BASE_RADIUS
    }
}

impl Objective for Lblj {
    fn kind(&self) -> &'static str {
        "Tech::Lblj"
    }
    fn key(&self) -> String {
        "Tech::Lblj".to_string()
    }
    fn obj_weight(&self, b: &St) -> f64 {
        if Lblj::start(b) { 12.0 } else { 0.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        Lblj::start(s)
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 24, generations: 6, horizon: 240 })
    }
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        // Only through a chain built in this rollout: the chain at
        // blj::CHAIN_SPEED in the lobby (HyperWalk spends it), or BitDW /
        // the corridor once a chain reached blj::CHAIN_BUILT (not walked).
        let sp = blj::chain_step(s, mem);
        let slow = start.forward_vel > -blj::CHAIN_SPEED;
        if (slow && sp > 0.0 && in_lobby(s) && s.alive() && blj::long_jumping(s) && s.forward_vel <= -blj::CHAIN_SPEED)
            || (sp >= blj::CHAIN_BUILT && (s.level == LEVEL_BITDW || (behind_door(s) && s.alive())))
        {
            return Eta::Done;
        }
        if !in_lobby(s) || !s.alive() {
            return Eta::Lost;
        }
        // The chain to CHAIN_SPEED, set up wherever Mario is (the GA picks
        // the place).
        Eta::Frames(blj::build_eta(s, None, blj::CHAIN_SPEED))
    }
    fn precision(&self) -> Precision {
        Precision::Precise
    }
    fn kinship(&self) -> u32 {
        10
    }
    fn success_value(&self) -> f64 {
        6.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        Some(blj::steer(s, TRAP))
    }
    fn steer_button(&self, s: &St) -> u16 {
        blj::button(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(action: u32, pos: [f32; 3], fv: f32, vy: f32) -> St {
        St {
            frame: 0,
            pos,
            vel: [0.0, vy, 0.0],
            forward_vel: fv,
            yaw: 0x6000,
            pitch: 0,
            angle_vel: [0; 3],
            water_level: -11000,
            cam_yaw: 0,
            action,
            mario_flags: 0,
            holding: false,
            held_slot: None,
            red_coins: 0,
            level: LEVEL_CASTLE as u16,
            area: 1,
            act: 1,
            course: 0,
            stars: 1,
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

    const IDLE: u32 = 0x0C40_0201;

    /// A riser chain in front of the door: ETA falls at every landing, `Done`
    /// on the first long-jump state at -150 after a re-jump.
    #[test]
    fn chain_eta_falls_then_done() {
        let p = [-2000.0, 0.0, -900.0];
        let mut traj = vec![st(IDLE, p, 0.0, 0.0), st(blj::ACT_CROUCH_SLIDE, p, 12.0, 0.0), st(blj::ACT_LONG_JUMP, p, 18.0, 30.0)];
        let mut v = -16.0f32;
        while v > -200.0 {
            traj.push(st(blj::ACT_LONG_JUMP_LAND, p, v, 0.0));
            v *= 1.5;
            traj.push(st(blj::ACT_LONG_JUMP, p, v, 30.0));
        }
        assert!(Lblj::start(&traj[0]));
        let mut mem = Mem::default();
        let etas: Vec<Eta> = traj.iter().map(|s| Lblj.eta(s, &traj[0], &mut mem)).collect();
        let done = etas.iter().position(|e| e.done()).expect("met");
        assert!(traj[done].forward_vel <= -blj::CHAIN_SPEED && blj::long_jumping(&traj[done]));
        assert!(traj[done - 1].forward_vel > -blj::CHAIN_SPEED);
        // Down at every phase change before it (setup -> slide -> flight -> landings).
        for w in etas[..done].windows(2) {
            assert!(w[1].frames() <= w[0].frames(), "{etas:?}");
        }
        // The same speed without a re-jump (a start already that fast) is no BLJ.
        let fast = st(blj::ACT_LONG_JUMP, p, -160.0, 30.0);
        assert!(!Lblj.eta(&fast, &fast, &mut Mem::default()).done());
    }
}
