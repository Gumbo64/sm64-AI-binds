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
//! The GA's progress is the graded BLJ metric (blj::metric) alone, at the
//! best chain's state (none without a chain, so walks add nothing). Success
//! needs a chain built in the trajectory: a chain at blj::CHAIN_SPEED (the
//! hand-off to Tech::HyperWalk), or the corridor / BitDW once a chain
//! reached blj::CHAIN_BUILT.

use obrrt_core::layout::LEVEL_CASTLE;
use obrrt_core::{Budget, Objective, Precision, Progress, St};

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
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(TRAP) as f64 / 400.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        // Only through a chain built in this trajectory: the chain at
        // blj::CHAIN_SPEED in the lobby (HyperWalk spends it), or BitDW /
        // the corridor once a chain reached blj::CHAIN_BUILT (not walked).
        let sp = blj::chain_speed(traj);
        let slow = traj[0].forward_vel > -blj::CHAIN_SPEED;
        traj.iter().enumerate().position(|(i, s)| {
            (slow && sp[i] > 0.0 && in_lobby(s) && s.alive() && blj::long_jumping(s) && s.forward_vel <= -blj::CHAIN_SPEED && blj::aligned_ok(s, TRAP))
                || (sp[i] >= blj::CHAIN_BUILT && (s.level == LEVEL_BITDW || (behind_door(s) && s.alive())))
        })
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // The chain metric only (no closeness to the trap: walking there
        // isn't a BLJ).
        blj::progress_toward(traj, TRAP)
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 24, generations: 6, horizon: 240 }
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
