//! Hazy Maze Cave star 3, "Metal-Head Mario Can Move!" (course 6, star index 2; `bhvStar` at (6200, -4400, 2300)).
//!
//! The star hangs over a ledge at the far end of a dry tunnel that leaves the underground lake's east shore. The
//! tunnel's entrance is shut by a grill (`bhvOpenableGrill` at (614, -4690, 2330), grill_door.inc.c) that opens
//! once the purple switch on the lake bed (`bhvFloorSwitchGrills` at (-307, -4997, 2483)) is pressed. That switch
//! is under 206 units of water (the lake's surface is y = -4791, levels/hmc/areas/1/collision.inc.c) and
//! `bhv_purple_switch_loop` (purple_switch.inc.c) takes a press only from a Mario standing on it
//! (`gMarioObject->platform == o`, within 4 of the floor, platform_displacement.c) whose action is not
//! `ACT_FLAG_SWIMMING` (0x2000, the `MARIO_UNKNOWN_13` it tests): a swimmer never presses it, a metal Mario
//! (ACT_METAL_WATER_WALKING) does. Metal caps (2nd behaviour byte 1) come from `!` boxes once the metal cap switch
//! is pressed and last 600 frames. Past the grill and the door at (922, -4689, 2330) the tunnel runs straight east
//! along z 2300: a corridor (x 1075 to 3021), a pit (death floor at -7761) crossed by two 1024-wide gaps with a
//! 512-wide island between (x 4045 to 4557), and the ledge (x 5581 to 6605). That part is on the straight line to
//! the star, so `Star::Collect` takes it; what the search lacked was the cap, the lake bed and the switch.
//!
//! Tests (route-0792 = the entry at 105 stars, OBRRT_GEO=1, 80 M frames each, `Hmc::MetalSwitch` attempts/successes):
//! hmc2-full1 (the original objective's box choice, only the loaded boxes) 49/0; hmc2-full2 (only the maze boxes
//! counted) 53/0; hmc2-full3 88/0; hmc2-full4 (the lake-side box only, see `usable_sources`) **56/28**: the first
//! success's route (the tree's own, not pasted) walks to the west pit, drops to the tunnel, breaks the box at
//! (-3000, -2250, -6400), and walks the lake bed in the metal cap to the switch in 2160 frames from the entry
//! (the cap flag 0x4 first seen at frame 537109, the switch reached at 537591, about 100 frames of the 600 left), replayed
//! with tasplay.
//!
//! `Hmc::MetalSwitch`: its ETA is the way to the switch, through a metal cap first (the cap if one is out, else
//! the nearest `!` box of metal caps and the box's break), and it is `Done` when the switch is pressed (its action
//! leaves 0). Generated in HMC with the star missing and the switch not yet pressed.

use std::sync::Arc;

use obrrt_core::layout::LEVEL_HMC;
use obrrt_core::{Eta, GaBudget, Mem, Objective, St, frames_for};

/// The purple switch on the lake bed (levels/hmc/script.c, `bhvFloorSwitchGrills`).
const SWITCH: [f32; 3] = [-307.0, -4997.0, 2483.0];
/// `MARIO_METAL_CAP` (include/sm64.h).
const MARIO_METAL_CAP: u32 = 0x04;
/// The metal cap `!` box on the way down to the lake (levels/hmc/areas/1/macro.inc.c, `macro_box_metal_cap`; the other
/// four are at (-6924, 2440, 7364), (5860, -550, -739), (1939, -600, -2920) and (5100, -600, -4500)).
const LAKE_BOX: [f32; 3] = [-3000.0, -2250.0, -6400.0];
/// Frames to break a box and pick its cap up.
const BOX_FRAMES: f64 = 40.0;
/// A walker's pace over a metal Mario's underwater walk: `update_metal_water_walking_speed` (mario_actions_submerged.c)
/// tops out at intendedMag / 1.5 = 21 a frame against the 30 of `frames_for`; the cap's 600 frames are spent at it.
const METAL_SLOW: f64 = 1.45;
/// `bhvFloorSwitchGrills` actions: 0 idle, 1 pressed, 2 ticking (purple_switch.inc.c).
const SWITCH_IDLE: i32 = 0;

pub struct MetalSwitch;

fn missing(s: &St) -> bool {
    s.zone() == (LEVEL_HMC as u16, 1) && !s.has_star(6, 2)
}

fn switch_pos(s: &St) -> [f32; 3] {
    s.objects().iter().find(|o| o.bhv_name() == "bhvFloorSwitchGrills").map_or(SWITCH, |o| o.pos)
}

fn pressed(s: &St) -> bool {
    s.objects().iter().any(|o| o.bhv_name() == "bhvFloorSwitchGrills" && o.action != SWITCH_IDLE)
}

/// The cap sources planned on: the box on the way down to the lake, at (-3000, -2250, -6400), 300 over the floor of
/// the tunnel that the lake's slope starts from (the main run's own way down to the lake, t1 steps 792-: the west
/// pit at (-3585, 1493, -6955), the slope, then the lake at (-4152, -4153, -3177); its mesh walk to the switch is
/// 323 frames, 470 at the metal pace, inside the cap's 600). The other four are not: the box at the spawn has a
/// mesh walk of 370 frames to the switch through falls the real level doesn't allow (the first test, hmc2-full1,
/// 80 M frames from the route-0792 entry, took its cap there and ran it out; the main run took 2300 frames from
/// the entry to the lake) and the three maze boxes (y -600, for the gas) have mesh walks of 244 to 329 that the
/// second and third tests (hmc2-full2/3: 30 cap-holding states in the maze, none nearer the switch than 4100
/// units) could not make real. With the cap already lying out, that one counts.
fn usable_sources(s: &St, _sw: [f32; 3]) -> Vec<[f32; 3]> {
    let caps: Vec<[f32; 3]> = s.objects().iter().filter(|o| o.bhv_name() == "bhvMetalCap").map(|o| o.pos).collect();
    if caps.is_empty() { vec![LAKE_BOX] } else { caps }
}

fn walk(s: &St, a: [f32; 3], b: [f32; 3]) -> f64 {
    frames_for(obrrt_core::geo::path_dist(s.zone(), a, b).unwrap_or_else(|| obrrt_core::dist(a, b)))
}

impl Objective for MetalSwitch {
    fn kind(&self) -> &'static str {
        "Hmc::MetalSwitch"
    }
    fn key(&self) -> String {
        "Hmc::MetalSwitch".into()
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        15.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        missing(s) && !pressed(s)
    }
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 16, generations: 5, horizon: 400 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != (LEVEL_HMC as u16, 1) {
            return Eta::Lost;
        }
        if s.objects.is_some() && pressed(s) {
            return Eta::Done;
        }
        let sw = switch_pos(s);
        if s.mario_flags & MARIO_METAL_CAP != 0 {
            // Only while the cap lasts to the switch (`cap_timer`, 600 from the box; the lake bed is walked at 21 a
            // frame, `METAL_SLOW`); past it, another cap.
            let w = walk(s, s.pos, sw);
            if s.cap_timer as f64 >= w * METAL_SLOW {
                return Eta::Frames(w);
            }
        }
        let best = usable_sources(s, sw).into_iter().map(|p| walk(s, s.pos, p) + BOX_FRAMES + walk(s, p, sw)).fold(f64::INFINITY, f64::min);
        Eta::Frames(best)
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        if s.mario_flags & MARIO_METAL_CAP != 0 {
            return Some(switch_pos(s));
        }
        usable_sources(s, switch_pos(s)).into_iter().min_by(|a, b| s.dist(*a).total_cmp(&s.dist(*b)))
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}

pub fn generate_level(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    if MetalSwitch.base_criteria(b) {
        out.push(Arc::new(MetalSwitch));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::testkit::{obj, st};
    use obrrt_core::fold;

    /// Without the cap the ETA goes through the box; with it straight to the switch; a pressed switch is `Done`.
    #[test]
    fn metal_cap_then_switch() {
        let sw = |action| obj("bhvFloorSwitchGrills", 1, SWITCH, action, 0);
        let mut capped = st(7, [-307.0, -4900.0, 2000.0], vec![sw(0)]);
        capped.mario_flags |= MARIO_METAL_CAP;
        capped.cap_timer = 600;
        let bare = st(7, [-307.0, -4900.0, 2000.0], vec![sw(0)]);
        let e = |s: &St| fold(&MetalSwitch, std::slice::from_ref(s)).1[0];
        assert!(e(&bare).frames() > e(&capped).frames() + 100.0);
        let done = st(7, [-307.0, -4997.0, 2483.0], vec![sw(2)]);
        assert_eq!(fold(&MetalSwitch, &[bare.clone(), done]).0, Some(1));
    }
}
