//! `WdwWater::Lower`: drain Wet-Dry World's water with its diamonds.
//!
//! Mechanism (levels/wdw/script.c, src/game/behaviors/wdw_water_level.inc.c): every `bhvWaterLevelDiamond` holds a
//! target level, its own height (`oWaterLevelTriggerTargetWaterLevel = oPosY`). Touching one in its idle action
//! (oAction 1) while `gWDWWaterLevelChanging` is 0 starts it (action 2): the level of the area's water box
//! (`*gEnvironmentLevels`; St::water_level is the level under Mario) moves to the target at 10 a frame, whatever
//! Mario then does; the diamond spins (action 3) until Mario is off it again. Area 1 starts at 31
//! (areas/1/collision.inc.c) with diamonds at y 2560, 1792, 1024, 256 and 40; the downtown (area 2) starts at -127
//! with diamonds at -2508 (drain) and -127. The level stays where it was left when WDW is entered again (route-0649
//! enters at 1039, cp-0365 at 50).
//!
//! Why it is a goal of its own: nothing but this rewarded the change (the generic `WaterDiamond::Touch` of
//! switches.rs is a 2-point switch press; its ETA ignores the level, so a diamond at the level the water already has
//! "succeeds" and changes nothing). `Lower` is done when the level at Mario is at most [`low`]:
//! - area 1, while star 4 (the express elevator) is missing: 150, under the floor (205) of the elevator's shaft:
//!   its only way in is a door at the shaft's foot, 200 wide and 199 high (platform.rs `WdwRide`), which Mario can
//!   walk through but not swim (a swimming Mario's wall radius is 110, perform_water_step); the nearest diamond
//!   (1810, 40, -3118) is 1000 from the tower. (A first version raised the water instead and got the trees to swim
//!   up to the elevator's east end, 600 below it, and stuck at the door: see the RUNS notes in the diff.)
//! - area 2 (the downtown), while stars 5 and 6 are missing: -2450 (its floors are at -2559 and up; the whole
//!   town is under the -127 water otherwise; see [`low`]).

use std::sync::Arc;

use obrrt_core::layout::*;
use obrrt_core::{Eta, Mem, Obj, Objective, St, dist, frames_for};

/// WDW area 1's secret triggers (`bhvHiddenStarTrigger`, 100 wide and 100 high, areas/1/macro.inc.c) that hang in
/// the air: (-3760, 650, 4120) has only the pool floor (y 0) under it, 650 below; a Mario touching it must be
/// swimming at that height (his hitbox reaches 160 up from his feet, so y >= 490), which takes a water level of at
/// least `y + 100` (a Mario at the surface floats about 80 under it; the diamonds' levels are 40, 256, 1024, 1792,
/// 2560). The other four stand on a floor, or within a jump of one: (943, 3830, -1779) 246 over the tower roof
/// (3584), (-2075, 3000, -524) 184 over the signpost's ledge (2816), (-2200, 384, 1145) and (-3689, 0, 3205) on
/// the ground. Returns the level the highest "wet" trigger left in `s` needs, None if all of them can be walked to.
/// (The `Secrets::Tour` of the main run, t1 step 705-708, started at level ~150 after the elevator's drain: the
/// first two triggers it took were the dry ones, then nothing raised the water for the fourth.)
pub fn wet_need(s: &St) -> Option<i32> {
    if s.level as u32 != LEVEL_WDW || s.area != 1 || s.objects.is_none() {
        return None;
    }
    // (From the level's collision: a floor-height table would depend on when the static floors get stored.)
    s.objects()
        .iter()
        .filter(|o| o.bhv_name() == "bhvHiddenStarTrigger" && o.pos[1] > 600.0 && o.pos[1] < 700.0 && o.pos[0] < -3500.0)
        .map(|o| o.pos[1] as i32 + 100)
        .max()
}

/// The level the Bob-omb Buddy's slab needs (None: not needed). The buddy that opens WDW's cannon (the way into the
/// downtown, warps.rs `wdw_town_missing`) stands at (-2, 3200, 3583) on a free-standing slab (static collision: x -255..256,
/// z 3328..3840, y 3072..3200, nothing under it but the pool floor; the nearest floors are the arrow lifts at 2177
/// and the 2304 plateau 600 away). The way up is out of the water: a Mario at the surface of the top diamond's level
/// (2560) jumps out of it (check_water_jump, mario_actions_submerged.c: vel 62, about 500 up) and grabs the slab's
/// edge. The 4 diamonds under 2560 give 1792 at best (a 1400 climb). Needed while the cannon's lid is shut.
pub fn buddy_need(s: &St) -> Option<i32> {
    if s.level as u32 != LEVEL_WDW || s.area != 1 || s.objects.is_none() || !super::super::warps::wdw_town_missing(s) {
        return None;
    }
    s.objects().iter().any(|o| o.bhv_name() == "bhvCannonClosed").then_some(2560)
}

/// The level a tree in WDW area 1 should bring the water to: the highest of the secrets' ([`wet_need`], while star 2
/// is missing) and the buddy's slab ([`buddy_need`]).
fn raise_need(s: &St) -> Option<i32> {
    let secrets = if s.has_star(11, 2) { None } else { wet_need(s) };
    secrets.max(buddy_need(s))
}

/// The diamond a Mario at `from` should touch to take the level to at least `need` in WDW (idle, target >= `need`,
/// not a change that is running to a lower one), the walk to it (units) and the frames the level then needs from
/// `level`; no diamond and no walk if a change to a good target is already running; None if no diamond can help.
pub fn raise_plan<'a>(s: &'a St, from: [f32; 3], need: i32, level: f32) -> Option<(Option<&'a Obj>, f32, f64)> {
    let diamonds = || s.objects().iter().filter(|o| o.bhv_name() == "bhvWaterLevelDiamond");
    if diamonds().any(|o| o.action == 2 && o.pos[1] >= need as f32) {
        return Some((None, 0.0, ((need as f32 - level).max(0.0) / RATE) as f64));
    }
    diamonds()
        .filter(|o| o.action == 1 && o.pos[1] >= need as f32)
        .map(|o| {
            let walk = obrrt_core::geo::path_dist(s.zone(), from, o.pos).unwrap_or_else(|| dist(from, o.pos));
            (Some(o), walk, ((o.pos[1] - level).max(0.0) / RATE) as f64)
        })
        .min_by(|a, b| (frames_for(a.1) + a.2).total_cmp(&(frames_for(b.1) + b.2)))
}

/// `WdwWater::Raise`: the other way: take WDW's water up to the level the area's remaining airborne secret needs
/// ([`wet_need`]), by a diamond (see the top of this file for how they work), while star 2 ("Secrets in the
/// Shallows & Sky") is missing. Done at the level, so the tree keeps the state that opened the water; without it
/// the only reward for the diamond walk was the tour's ETA (a goal of 30 whose success is a trigger count).
pub struct Raise {
    zone: (u16, u8),
}

impl Raise {
    pub fn generate(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
        if b.level as u32 == LEVEL_WDW && b.area == 1 && raise_need(b).is_some_and(|n| (b.water_level as i32) < n) {
            out.push(Arc::new(Raise { zone: b.zone() }));
        }
    }
}

impl Objective for Raise {
    fn kind(&self) -> &'static str {
        "WdwWater::Raise"
    }
    fn key(&self) -> String {
        format!("{} {:?}", self.kind(), self.zone)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        if buddy_need(b).is_some() { 20.0 } else { 8.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && raise_need(s).is_some_and(|n| (s.water_level as i32) < n)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 20, generations: 5, horizon: 300 })
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone || !s.alive() {
            return Eta::Lost;
        }
        // The need is the start's: touching the trigger that needed it removes it from `s`.
        let Some(need) = raise_need(start) else { return Eta::Lost };
        let level = s.water_level as i32;
        if level >= need {
            return Eta::Done;
        }
        if s.objects.is_none() {
            return Eta::Frames(300.0);
        }
        match raise_plan(s, s.pos, need, level as f32) {
            Some((_, walk, f)) => Eta::Frames(frames_for(walk) + f),
            None => Eta::Lost,
        }
    }
    fn success_value(&self) -> f64 {
        4.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let need = raise_need(s)?;
        raise_plan(s, s.pos, need, s.water_level as f32)?.0.map(|o| o.pos)
    }
}

/// `Lower` is done at this level in `area`. The downtown's: -2450, the diamond's own level is -2508, over the floors
/// at -2559: a swimming Mario can't break a box (ACT_WATER_PUNCH has no ACT_FLAG_ATTACKING, sm64.h; only the
/// walking/air attacks and ground pounds do, interaction.c determine_interaction), and the red coins' giant boxes
/// stand at y -2175..-1000 and the vanish cap box floats at -2240 (-2000, as first written here, left the lowest
/// ones under water: wdw2b-g/h, 0 boxes broken in 3 runs of 40-60 M, all swimming).
fn low(area: u8) -> i32 {
    if area == 1 { 150 } else { -2450 }
}
/// The frames the downtown's trees still pay before everything is dry (see [`low`]): the walk to the draining diamond
/// and its drain, 0 once the level is at the target; added to the ETAs of what needs dry ground there (the red coins'
/// boxes, the vanish box: a swimming Mario can't break a box), so that a state that drained the water ranks over
/// one that swam to a box (wdw2b-i: 15 successes of `Lower`, none of the 28 `Break` attempts started from them).
/// Zero outside WDW's downtown and while the objects aren't loaded.
pub fn town_drain_frames(s: &St) -> f64 {
    if s.level as u32 != LEVEL_WDW || s.area != 2 || s.objects.is_none() || (s.water_level as i32) <= low(2) {
        return 0.0;
    }
    let level = s.water_level as f32;
    let diamonds = s.objects().iter().filter(|o| o.bhv_name() == "bhvWaterLevelDiamond" && o.pos[1] <= low(2) as f32);
    if let Some(running) = diamonds.clone().find(|o| o.action == 2) {
        return ((level - running.pos[1]).max(0.0) / RATE) as f64;
    }
    diamonds
        .filter(|o| o.action == 1)
        .map(|o| frames_for(obrrt_core::geo::path_dist(s.zone(), s.pos, o.pos).unwrap_or_else(|| dist(s.pos, o.pos))) + ((level - o.pos[1]).max(0.0) / RATE) as f64)
        .fold(f64::INFINITY, f64::min)
        .min(600.0)
}

/// A diamond's level change per frame (approach_f32_symmetric(.., 10)).
const RATE: f32 = 10.0;

pub struct Lower {
    zone: (u16, u8),
    area: u8,
}

impl Lower {
    pub fn generate(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
        if b.level as u32 != LEVEL_WDW {
            return;
        }
        if (b.area == 1 && !b.has_star(11, 3)) || (b.area == 2 && (!b.has_star(11, 4) || !b.has_star(11, 5))) {
            out.push(Arc::new(Lower { zone: b.zone(), area: b.area }));
        }
    }

    fn level(s: &St) -> i32 {
        s.water_level as i32
    }

    fn satisfied(&self, s: &St) -> bool {
        Self::level(s) <= low(self.area) && Self::level(s) > -10000
    }

    fn good_target(&self, o: &Obj) -> bool {
        (o.pos[1] as i32) <= low(self.area)
    }

    fn diamonds<'a>(&self, s: &'a St) -> impl Iterator<Item = &'a Obj> + 'a {
        s.objects().iter().filter(|o| o.bhv_name() == "bhvWaterLevelDiamond")
    }

    /// The diamond to go to (idle, with a good target, not at the level already) and the frames until the level
    /// is there; None when no diamond can help. A running change to a good target: just the frames it needs.
    fn plan<'a>(&self, s: &'a St) -> Option<(Option<&'a Obj>, f64)> {
        let l = Self::level(s) as f32;
        let need = ((l - low(self.area) as f32).max(0.0) / RATE) as f64;
        if self.diamonds(s).any(|o| o.action == 2 && self.good_target(o)) {
            return Some((None, need));
        }
        // A change running to a bad target blocks the touch (gWDWWaterLevelChanging): then none.
        self.diamonds(s)
            .filter(|o| o.action == 1 && self.good_target(o) && (o.pos[1] - l).abs() > 30.0)
            .map(|o| {
                let walk = obrrt_core::geo::path_dist(s.zone(), s.pos, o.pos).unwrap_or_else(|| dist(s.pos, o.pos));
                (Some(o), frames_for(walk) + need)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }
}

impl Objective for Lower {
    fn kind(&self) -> &'static str {
        "WdwWater::Lower"
    }
    fn key(&self) -> String {
        format!("{} {:?} {}", self.kind(), self.zone, self.area)
    }
    /// Area 1's is a goal (over OBRRT_GOAL_MIN 20): the elevator star needs it. The downtown's is one step.
    fn obj_weight(&self, _b: &St) -> f64 {
        if self.area == 1 { 25.0 } else { 8.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && !self.satisfied(s) && self.plan(s).is_some()
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 20, generations: 5, horizon: 300 })
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone || !s.alive() {
            return Eta::Lost;
        }
        if self.satisfied(s) {
            return Eta::Done;
        }
        if s.objects.is_none() {
            return Eta::Frames(300.0);
        }
        match self.plan(s) {
            Some((_, f)) => Eta::Frames(f),
            None => Eta::Lost,
        }
    }
    fn success_value(&self) -> f64 {
        6.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.plan(s)?.0.map(|o| o.pos)
    }
}

/// `WdwWater::Slab`: stand on the Bob-omb Buddy's slab (see [`buddy_need`]) with the water at its top level. The way
/// up is the square floating platform at (-767, water level, 3584) (bhvWdwSquareFloatingPlatform: it rides the
/// water, its top 64 above its object; collision 512 square, floating_platform.inc.c), whose east edge (x -511) is
/// 256 west of the slab's (x -255) and whose top (2624 at the highest water) is 576 under the slab's top: a triple
/// jump's rise (vel 69 + a quarter of the speed, gravity 4: ~600-740) puts Mario's feet at the ledge-grab height
/// (a ledge is grabbed with the feet 100-160 under it, falling). A water jump out of the surface does not do it: it
/// peaks at ~2740 and bonks the slab from under (wdw2b-d: 460 under the buddy, 0 of 72 talks). Done: on the slab, on
/// the ground. The ETA runs by the platform's top, then to the slab's edge; a Mario under the slab pays the way out.
pub struct Slab {
    zone: (u16, u8),
}

/// The slab's x, z extent and top (static collision of area 1).
const SLAB_X: (f32, f32) = (-255.0, 256.0);
const SLAB_Z: (f32, f32) = (3328.0, 3840.0);
const SLAB_TOP: f32 = 3200.0;

impl Slab {
    pub fn generate(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
        if buddy_need(b).is_some() && b.water_level as i32 >= 2400 {
            out.push(Arc::new(Slab { zone: b.zone() }));
        }
    }

    fn on_slab(s: &St) -> bool {
        s.pos[0] > SLAB_X.0 && s.pos[0] < SLAB_X.1 && s.pos[2] > SLAB_Z.0 && s.pos[2] < SLAB_Z.1 && s.pos[1] > SLAB_TOP - 30.0 && s.pos[1] < SLAB_TOP + 120.0 && !s.action_has(ACT_FLAG_AIR)
    }
}

impl Objective for Slab {
    fn kind(&self) -> &'static str {
        "WdwWater::Slab"
    }
    fn key(&self) -> String {
        format!("{} {:?}", self.kind(), self.zone)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        25.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && s.alive() && s.water_level as i32 >= 2400 && !Self::on_slab(s)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 20, generations: 5, horizon: 300 })
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone || !s.alive() {
            return Eta::Lost;
        }
        if Self::on_slab(s) {
            return Eta::Done;
        }
        if s.objects.is_none() {
            return Eta::Frames(300.0);
        }
        let edge = [-200.0, SLAB_TOP, 3584.0];
        let Some(plat) = s.objects().iter().find(|o| o.bhv_name() == "bhvWdwSquareFloatingPlatform" && (o.home[0] + 767.0).abs() < 50.0 && (o.home[2] - 3584.0).abs() < 50.0) else {
            return Eta::Lost;
        };
        let top = [plat.pos[0], plat.pos[1] + 64.0, plat.pos[2]];
        // Over the gap or the slab at a height from which the edge is within a jump: straight on (airborne, east
        // of the platform's west half, at least a little above its top).
        let crossing = s.action_has(ACT_FLAG_AIR) && s.pos[0] > -700.0 && s.pos[0] < SLAB_X.1 && s.pos[1] > top[1] + 40.0;
        let on_top = s.pos[0] > -1023.0 && s.pos[0] < -511.0 && s.pos[2] > 3328.0 && s.pos[2] < 3840.0 && s.pos[1] > top[1] - 20.0;
        let via = if crossing { s.dist(edge) } else if on_top { s.dist(edge) + 40.0 } else { s.dist(top) + dist(top, edge) + 40.0 };
        Eta::Frames(frames_for(via))
    }
    fn success_value(&self) -> f64 {
        6.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let plat = s.objects().iter().find(|o| o.bhv_name() == "bhvWdwSquareFloatingPlatform" && (o.home[0] + 767.0).abs() < 50.0)?;
        Some([plat.pos[0], plat.pos[1] + 64.0, plat.pos[2]])
    }
}

/// `WdwWater::Quick`: "Quick Race Through Downtown!" (star index 5): a plain `bhvStar` at (2180, -840, 3720)
/// over a pillar (top -1125) in the town's south-east corner (levels/wdw/script.c, area 2), and between it and the
/// town a diagonal wall of `SURFACE_VANISH_CAP_WALLS` (areas/2/collision.inc.c: (1710, 3073) - (1275, 3508) -
/// (1120, 3840), y -2559..-511, which find_wall_collisions skips for a Mario with the vanish cap, 600 frames of it).
/// The town's vanish-cap `!` boxes (macro_box_vanish_cap, 2nd byte 2) are at (-1779, -2240, 3644), outside, and
/// (1300, -2260, 3740), inside the wall: so Mario breaks the outer box, takes the cap, and swims through the wall to the
/// star within the cap's time ("quick race"). wdw2b-g (60 M, `Star::Collect` of this star): 0 of 214, the best state 897
/// away on the wall's west side. This objective's ETA runs box, cap, star (Done: star had); a Mario with the cap goes
/// straight to the star. Generated in the downtown while the star is missing.
pub struct Quick {
    zone: (u16, u8),
}

/// The star (script.c `bhvStar` 0x05, area 2).
const QUICK_STAR: [f32; 3] = [2180.0, -840.0, 3720.0];
const MARIO_VANISH: u32 = 0x02;

impl Quick {
    pub fn generate(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
        if b.level as u32 == LEVEL_WDW && b.area == 2 && !b.has_star(11, 5) {
            out.push(Arc::new(Quick { zone: b.zone() }));
        }
    }

    /// Where the cap comes from: a vanish cap lying out, else the nearest solid vanish `!` box outside the wall
    /// (x < 1000), with the frames to break it.
    fn source(s: &St) -> Option<([f32; 3], f64)> {
        let o = s.objects();
        if let Some(c) = o.iter().filter(|c| c.bhv_name() == "bhvVanishCap").min_by(|a, b| s.dist(a.pos).total_cmp(&s.dist(b.pos))) {
            return Some((c.pos, 0.0));
        }
        o.iter()
            .filter(|b| b.bhv_name() == "bhvExclamationBox" && (b.beh_params >> 16) & 0xFF == 2 && b.action == 2 && b.pos[0] < 1000.0)
            .min_by(|a, b| s.dist(a.pos).total_cmp(&s.dist(b.pos)))
            .map(|b| (b.pos, 40.0))
    }
}

impl Objective for Quick {
    fn kind(&self) -> &'static str {
        "WdwWater::Quick"
    }
    fn key(&self) -> String {
        format!("{} {:?}", self.kind(), self.zone)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        30.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && s.alive() && !s.has_star(11, 5)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 20, generations: 5, horizon: 300 })
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone || !s.alive() {
            return Eta::Lost;
        }
        if s.has_star(11, 5) {
            return Eta::Done;
        }
        if s.objects.is_none() {
            return Eta::Frames(300.0);
        }
        if s.mario_flags & MARIO_VANISH != 0 {
            return Eta::Frames(frames_for(s.dist(QUICK_STAR)));
        }
        // Not yet capped, or Mario is already on the star's side: straight to it.
        if s.pos[0] > 1300.0 && s.pos[2] > 3000.0 {
            return Eta::Frames(frames_for(s.dist(QUICK_STAR)));
        }
        let Some((src, hit)) = Self::source(s) else { return Eta::Lost };
        Eta::Frames(town_drain_frames(s) + frames_for(s.dist(src)) + hit + frames_for(dist(src, QUICK_STAR)))
    }
    fn success_value(&self) -> f64 {
        8.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        if s.mario_flags & MARIO_VANISH != 0 || s.pos[0] > 1300.0 {
            return Some(QUICK_STAR);
        }
        Self::source(s).map(|x| x.0)
    }
}
