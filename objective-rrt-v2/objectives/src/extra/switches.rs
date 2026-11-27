//! Switches and buttons: purple switches, cap switches, floor switches, ! switches, ground-pound posts and pillars, pushable blocks, poles and trees to climb, anything that changes the level when pressed or pounded.
//!
//! Generated objectives for this category; see `extra/mod.rs`. One struct,
//! [`Switch`], covers them all; [`K`] says which thing it is and holds the
//! per-kind domain knowledge (when it's ready, when it's done, how to steer).
//!
//! Many of these behaviours have no `SET_HOME` in their script
//! (data/behavior_data.c), so their `oHome` is 0 and their `ObjId`s collide
//! (every tree is `bhvTree@0,0,0`). Those are identified by behaviour plus
//! their horizontal position instead (see [`Tgt`]); all of them are static
//! horizontally.
//!
//! NEEDS: the water level (`gEnvironmentLevels[0]`) to know which WDW water diamond would change it (one at the current level does nothing).
//! NEEDS: `gTHIWaterDrained` (THI tiny island top: once drained, a reload leaves its oAction at 0, indistinguishable from not pounded).

use std::sync::Arc;

use obrrt_core::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use obrrt_core::layout::{ACT_FLAG_AIR, ACT_FLAG_ON_POLE, INTERACT_POLE, LEVEL_BBH, LEVEL_BOB, LEVEL_CASTLE, LEVEL_CASTLE_GROUNDS, LEVEL_SSL, LEVEL_THI};
use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St, dist, frames_for};

// Mario actions, from include/sm64.h.
const ACT_GROUND_POUND: u32 = 0x008008A9;
const ACT_GROUND_POUND_LAND: u32 = 0x0080023C;
const ACT_TOP_OF_POLE_TRANSITION: u32 = 0x00100344;
const ACT_TOP_OF_POLE: u32 = 0x00100345;

/// Castle basement water pillars stand at y = -1178 (levels/castle_inside/
/// script.c); pounded, or with the moat already drained, they sit 80 lower
/// (water_pillar.inc.c), while oAction of a drained one stays 0.
const PILLAR_Y: f32 = -1178.0;

/// What the switch-like thing is.
#[derive(Debug, Clone, Copy, PartialEq)]
enum K {
    /// Purple floor switches (bhvFloorSwitch*, purple_switch.inc.c): stand on
    /// it within 127.5 of its middle; oAction 0 idle, 1 pressed, 2 ticking,
    /// 3 unpressing, 4 waiting for Mario to get off.
    Purple,
    /// Red/green/blue cap switches (bhvCapSwitch, capswitch.inc.c): land on
    /// it; oAction 1 unpressed, 2 being pressed, 3 pressed (also at load if
    /// its save flag is set).
    Cap,
    /// Blue coin switch (bhvBlueCoinSwitch, blue_coin.inc.c): ground pound
    /// it; oAction 0 idle, 1 receding, 2 ticking, then deleted.
    BlueCoin,
    /// Castle basement water pillars (bhvWaterLevelPillar): ground pound both
    /// to drain the moat (SAVE_FLAG_MOAT_DRAINED).
    Pillar,
    /// The Chain Chomp's post (bhvWoodenPost spawned by bhvChainChomp,
    /// chain_chomp.inc.c): pounded down to 190 below its home, it frees the
    /// Chomp, who breaks the gate to BoB's star 6.
    ChompPost,
    /// A wooden post that gives 5 coins when Mario runs around it 3 times
    /// within 200 frames, staying within 400 (bhv_wooden_post_update).
    PostCircle,
    /// THI's tiny island top (bhvThiTinyIslandTop, thi_top.inc.c): a ground
    /// pound landing within 500 drains the water.
    IslandTop,
    /// WDW water level diamond (bhvWaterLevelDiamond, wdw_water_level.inc.c):
    /// touch it; oAction 1 idle, 2 changing the level, 3 spinning.
    Diamond,
    /// SSL pyramid pillar touch detector (bhvPyramidPillarTouchDetector,
    /// pyramid_top.inc.c): touch all 4 to open the pyramid top; each one is
    /// deactivated when touched.
    PyramidPillar,
    /// BBH library book switch (bhvBookSwitch, flying_bookend_switch.inc.c),
    /// with its index (2nd behaviour byte, 0-2): punched in index order they
    /// move the haunted bookshelf.
    Book(u8),
    /// Pushable metal box (bhvPushableMetalBox, metal_box.inc.c).
    Box,
    /// Anything with INTERACT_POLE: climb to its top.
    PoleTop,
}

// ---- ETA phase estimates (frames; rough, calibrated later by
// tools/analysis/eta_calib.py) ----

/// A ground pound from standing next to the target: jump (~12 frames up),
/// the pound's spin (10, act_ground_pound) and the fall onto it.
const POUND_FRAMES: f64 = 25.0;
/// Ground-pound fall speed (act_ground_pound sets vel.y = -50 after the spin).
const POUND_FALL: f32 = 50.0;
/// The pound's landing frame that sinks the post / presses the pillar.
const POUND_LAND_FRAMES: f64 = 2.0;
/// Each further Chomp-post pound once one is done: off the post (it sinks
/// ~65), back up and down again (~a jump and a pound).
const REPOUND_FRAMES: f64 = 35.0;
/// How far one pound sinks the Chomp post (chain_chomp.inc.c, ~65; 190 total
/// frees the Chomp).
const POST_SINK: f32 = 65.0;
/// Depth below home that frees the Chomp.
const POST_FREE: f32 = 190.0;
/// Radius Post::Circle runs at (the steering's circle).
const CIRCLE_R: f32 = 220.0;
/// Angle the wooden post needs swept (3 laps; bhv_wooden_post_update: 0x30000).
const CIRCLE_ANGLE: f64 = 3.0 * std::f64::consts::TAU;
/// The wooden post's window (oTimer < 200 from entering its 400 range).
const CIRCLE_WINDOW: u32 = 200;
/// Leaving the post's range (400) and coming back to restart its window.
const CIRCLE_RESET_FRAMES: f64 = 20.0;
/// Pole climbing speed: oMarioPolePos += stickY / 8 (act_climbing_pole), 8 at full stick.
const CLIMB_PACE: f32 = 8.0;
/// Climb height assumed for a pole (bhvTree's hitbox is 500 high; poles vary).
const POLE_HEIGHT: f32 = 500.0;
/// Grabbing the pole (a jump into it) and the top-of-pole transition.
const POLE_GRAB_FRAMES: f64 = 10.0;
const POLE_TOP_FRAMES: f64 = 5.0;
/// Landing on a cap switch from next to it: a jump onto its top (~a
/// single jump's airtime, 20 frames at 42 up / 4 gravity).
const CAP_LAND_FRAMES: f64 = 20.0;
/// thi_top.inc.c: a pound landing within this of the island top drains it.
const ISLAND_RADIUS: f32 = 500.0;
/// The steering pounds within this of the island top (steer_button), well
/// inside [`ISLAND_RADIUS`].
const ISLAND_NEAR: f32 = 300.0;
/// A punch: act_punching's first hit frame (~3) and a frame to line up.
const PUNCH_FRAMES: f64 = 5.0;
/// How far the metal box must move from its start (v1's success: 200).
const BOX_PUSH: f32 = 200.0;
/// metal_box.inc.c bhv_pushable_loop: pushed, it moves at oForwardVel 4.
const BOX_PACE: f32 = 4.0;

/// How an objective finds its object in a state.
#[derive(Debug, Clone, Copy)]
enum Tgt {
    /// Objects with a home: their [`ObjId`].
    Id(ObjId),
    /// Objects without one: behaviour and horizontal position.
    Fixed { bhv: u32, x: f32, z: f32 },
}

impl Tgt {
    fn of(o: &Obj) -> Tgt {
        if o.id.home == [0, 0, 0] { Tgt::Fixed { bhv: o.bhv, x: o.pos[0], z: o.pos[2] } } else { Tgt::Id(o.id) }
    }

    fn find<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        match *self {
            Tgt::Id(id) => s.find(id),
            Tgt::Fixed { bhv, x, z } => s
                .objects()
                .iter()
                .filter(|o| o.bhv == bhv)
                .map(|o| (o, (o.pos[0] - x).powi(2) + (o.pos[2] - z).powi(2)))
                .filter(|(_, d2)| *d2 < 80.0 * 80.0)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(o, _)| o),
        }
    }

    fn key(&self) -> String {
        match *self {
            Tgt::Id(id) => id.to_string(),
            Tgt::Fixed { bhv, x, z } => format!("{}@{},{}", obrrt_core::bhv_name(bhv).unwrap_or("?"), x.round() as i32, z.round() as i32),
        }
    }
}

pub struct Switch {
    k: K,
    tgt: Tgt,
    zone: (u16, u8),
    weight: f64,
    /// The star this leads to: no point once it's collected.
    star: Option<(i16, u8)>,
}

fn hdist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn add(p: [f32; 3], dy: f32) -> [f32; 3] {
    [p[0], p[1] + dy, p[2]]
}

fn airborne(s: &St) -> bool {
    s.action_has(ACT_FLAG_AIR)
}

/// Books of the BBH bookshelf puzzle in `s`: (index, out?, action).
fn books(s: &St) -> impl Iterator<Item = (u8, bool, i32)> + '_ {
    s.objects().iter().filter(|o| o.bhv_name() == "bhvBookSwitch").map(|o| {
        // pos z = home z - offset, the offset approaching 50 when sticking out.
        (((o.beh_params >> 16) & 0xFF) as u8, o.home[2] - o.pos[2] > 25.0, o.action)
    })
}

/// Whether `o` is a wooden post that the Chain Chomp is chained to: the
/// Chomp spawns it at its own home, which stays there until it's released.
fn is_chomp_post(b: &St, o: &Obj) -> bool {
    b.objects().iter().any(|c| c.bhv_name() == "bhvChainChomp" && hdist(c.home, o.home) < 50.0)
}

/// `mem` of Post::Circle: bit 0 = within range (an angle seen), f[0] = angle
/// swept since entering, f[1] = the previous angle, f[2] = frame entered.
const IN_RANGE: u32 = 1;

impl Switch {
    fn new(b: &St, o: &Obj, k: K, weight: f64, star: Option<(i16, u8)>) -> Switch {
        Switch { k, tgt: Tgt::of(o), zone: b.zone(), weight, star }
    }

    /// Whether Mario is on this pole.
    fn on_pole(&self, s: &St, o: &Obj) -> bool {
        s.action_has(ACT_FLAG_ON_POLE) && hdist(s.pos, o.pos) < 100.0
    }

    /// Whether the object, as it is in `s`, can still be activated.
    fn ready(&self, s: &St, o: &Obj) -> bool {
        match self.k {
            K::Purple => o.action == 0,
            K::Cap => o.action == 1,
            K::BlueCoin => o.action == 0,
            K::Pillar => o.action == 0 && o.pos[1] > PILLAR_Y - 40.0,
            K::ChompPost => o.pos[1] > o.home[1] - 189.0,
            // Coins not given yet, and not pounded down (offset 0).
            K::PostCircle => o.loot_coins > 0 && (o.pos[1] - o.home[1]).abs() < 1.0,
            K::IslandTop => o.action == 0,
            K::Diamond => o.action == 1,
            K::PyramidPillar => true,
            // Sticking out, every lower-indexed book already pushed in (a
            // wrong press pops them all out again), every higher one out.
            K::Book(i) => o.action == 1 && books(s).all(|(j, out, a)| j == i || if j < i { !out && a == 0 } else { out }),
            K::Box => true,
            K::PoleTop => !(self.on_pole(s, o) && matches!(s.action, ACT_TOP_OF_POLE | ACT_TOP_OF_POLE_TRANSITION)),
        }
    }

    /// Whether `s` (relative to the attempt's start `s0`) has it activated
    /// (v1's success predicate).
    fn done(&self, s0: &St, s: &St) -> bool {
        if s.zone() != self.zone || s.objects.is_none() {
            return false;
        }
        let o = self.tgt.find(s);
        match self.k {
            K::Purple => o.is_some_and(|o| matches!(o.action, 1 | 2 | 4)),
            K::Cap => o.is_some_and(|o| o.action >= 2),
            // Deleted once its coins are gone or time is up.
            K::BlueCoin => o.is_none_or(|o| o.action >= 1),
            K::Pillar => o.is_some_and(|o| o.action >= 1),
            // Each pound sinks it ~65 (3 free the Chomp): one pound is a
            // success, so the tree can grow a pound at a time.
            K::ChompPost => o.is_some_and(|o| {
                o.pos[1] <= o.home[1] - 189.0 || self.tgt.find(s0).is_some_and(|o0| o.pos[1] <= o0.pos[1] - 50.0)
            }),
            K::PostCircle => o.is_some_and(|o| o.loot_coins == 0),
            K::IslandTop => o.is_some_and(|o| o.action >= 1),
            K::Diamond => o.is_some_and(|o| o.action >= 2),
            // Deactivated (gone from the pool) when touched.
            K::PyramidPillar => o.is_none(),
            K::Book(_) => o.is_some_and(|o| o.action == 2),
            K::Box => match (self.tgt.find(s0), o) {
                (Some(a), Some(b)) => hdist(a.pos, b.pos) >= BOX_PUSH,
                _ => false,
            },
            K::PoleTop => o.is_some_and(|o| self.on_pole(s, o) && matches!(s.action, ACT_TOP_OF_POLE | ACT_TOP_OF_POLE_TRANSITION)),
        }
    }

    /// The point to reach (the top of the thing, roughly).
    fn point(&self, o: &Obj) -> [f32; 3] {
        match self.k {
            // The cap switch's pos is already 71 up its base; its top is higher.
            K::Cap => add(o.pos, 60.0),
            // "the switch is 100 units tall" (blue_coin.inc.c).
            K::BlueCoin => add(o.pos, 100.0),
            K::ChompPost => add(o.pos, 100.0),
            K::PoleTop => add(o.pos, 100.0),
            _ => o.pos,
        }
    }

    /// Ground-pound kinds.
    fn pound(&self) -> bool {
        matches!(self.k, K::BlueCoin | K::Pillar | K::ChompPost | K::IslandTop)
    }

    /// Frames to land a ground pound on `top` from `s`: falling onto it in
    /// a ground pound, else the approach plus a whole pound.
    fn pound_eta(s: &St, top: [f32; 3]) -> f64 {
        if s.action == ACT_GROUND_POUND && hdist(s.pos, top) < 100.0 && s.pos[1] >= top[1] - 50.0 {
            ((s.pos[1] - top[1]).max(0.0) / POUND_FALL) as f64 + POUND_LAND_FRAMES
        } else {
            frames_for(dist(s.pos, top)) + POUND_FRAMES
        }
    }

    /// Frames to a pound landing within [`ISLAND_RADIUS`] of the island top
    /// `o`: falling to the floor in a ground pound there, else the way to
    /// within [`ISLAND_NEAR`] of it plus a whole pound.
    fn island_eta(s: &St, o: &Obj) -> f64 {
        let hd = hdist(s.pos, o.pos);
        if s.action == ACT_GROUND_POUND && hd < ISLAND_RADIUS {
            ((s.pos[1] - s.floor_y).max(0.0) / POUND_FALL) as f64 + POUND_LAND_FRAMES
        } else {
            frames_for((dist(s.pos, o.pos) - ISLAND_NEAR).max(0.0)) + POUND_FRAMES
        }
    }

    /// Post::Circle: folds this frame's angle round the post into `mem`.
    fn circle_eta(s: &St, o: &Obj, mem: &mut Mem) -> f64 {
        let hd = hdist(s.pos, o.pos);
        let laps = CIRCLE_ANGLE * (CIRCLE_R as f64) / obrrt_core::PACE;
        // v1 reset the sweep beyond 380 (the game's 400, with a margin).
        if hd > 380.0 {
            mem.flags &= !IN_RANGE;
            mem.f[0] = 0.0;
            return frames_for((hd - CIRCLE_R).max(0.0)) + laps;
        }
        let a = ((s.pos[0] - o.pos[0]) as f64).atan2((s.pos[2] - o.pos[2]) as f64);
        if mem.flags & IN_RANGE != 0 {
            let mut da = a - mem.f[1];
            while da > std::f64::consts::PI {
                da -= std::f64::consts::TAU;
            }
            while da < -std::f64::consts::PI {
                da += std::f64::consts::TAU;
            }
            mem.f[0] += da;
        } else {
            mem.flags |= IN_RANGE;
            mem.f[0] = 0.0;
            mem.f[2] = s.frame as f64;
        }
        mem.f[1] = a;
        let left = (CIRCLE_ANGLE - mem.f[0].abs()).max(0.0) * CIRCLE_R as f64 / obrrt_core::PACE;
        let used = s.frame as f64 - mem.f[2];
        if used + left > CIRCLE_WINDOW as f64 {
            // The window can't fit the laps left: out of range and back in
            // (the game resets oTimer and the angle beyond 400), then all 3.
            frames_for(400.0 - hd) + CIRCLE_RESET_FRAMES + laps
        } else {
            left
        }
    }
}

impl Objective for Switch {
    fn precision(&self) -> obrrt_core::Precision {
        if matches!(self.kind(), "Post::Pound" | "Pillar::Pound" | "IslandTop::Pound" | "BlueCoinSwitch::Pound") { obrrt_core::Precision::Precise } else { obrrt_core::Precision::Explore }
    }
    fn kind(&self) -> &'static str {
        match self.k {
            K::Purple => "PurpleSwitch::Press",
            K::Cap => "CapSwitch::Press",
            K::BlueCoin => "BlueCoinSwitch::Pound",
            K::Pillar => "Pillar::Pound",
            K::ChompPost => "Post::Pound",
            K::PostCircle => "Post::Circle",
            K::IslandTop => "IslandTop::Pound",
            K::Diamond => "WaterDiamond::Touch",
            K::PyramidPillar => "PyramidPillar::Touch",
            K::Book(_) => "Book::Punch",
            K::Box => "Block::Push",
            K::PoleTop => "Pole::Climb",
        }
    }
    fn key(&self) -> String {
        format!("{} {:?} {}", self.kind(), self.zone, self.tgt.key())
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        self.weight
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && self.star.is_none_or(|(c, i)| !s.has_star(c, i))
            && self.tgt.find(s).is_some_and(|o| self.ready(s, o))
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        let (population, generations, horizon) = match self.k {
            K::Purple | K::Book(_) => (16, 5, 150),
            K::Cap | K::Diamond | K::Box => (20, 5, 200),
            K::BlueCoin | K::Pillar | K::IslandTop => (20, 6, 200),
            K::ChompPost => (24, 6, 300),
            K::PostCircle | K::PoleTop | K::PyramidPillar => (20, 5, 240),
        };
        Some(obrrt_core::GaBudget { population, generations, horizon })
    }
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        if self.done(start, s) {
            return Eta::Done;
        }
        // Objects not read this frame: no judgement possible; hold on to the
        // distance to where it was at the start.
        let Some(o) = self.tgt.find(s) else {
            return match self.tgt.find(start) {
                Some(o0) if s.objects.is_none() => Eta::Frames(frames_for(dist(s.pos, self.point(o0)))),
                _ => Eta::Lost,
            };
        };
        Eta::Frames(match self.k {
            K::Purple | K::Diamond | K::PyramidPillar => frames_for(dist(s.pos, o.pos)),
            K::Cap => frames_for(dist(s.pos, self.point(o))) + CAP_LAND_FRAMES,
            K::BlueCoin => Self::pound_eta(s, self.point(o)),
            K::IslandTop => Self::island_eta(s, o),
            K::Book(_) => frames_for(dist(s.pos, o.pos)) + PUNCH_FRAMES,
            K::Box => {
                // To the box, then the rest of the 200 at its push speed.
                let moved = self.tgt.find(start).map_or(0.0, |o0| hdist(o0.pos, o.pos));
                frames_for(dist(s.pos, o.pos)) + ((BOX_PUSH - moved).max(0.0) / BOX_PACE) as f64
            }
            K::Pillar => Self::pound_eta(s, o.pos),
            K::ChompPost => {
                // One pound is Done (v1). The pounds the post still needs
                // after it count too, so deeper posts rank first as starts
                // (v1's base heuristic: + depth left / 40); within a rollout
                // the term is constant until the pound.
                let left = ((o.pos[1] - (o.home[1] - POST_FREE)) / POST_SINK).ceil().max(1.0);
                Self::pound_eta(s, self.point(o)) + (left as f64 - 1.0) * REPOUND_FRAMES
            }
            K::PostCircle => Self::circle_eta(s, o, mem),
            K::PoleTop if self.on_pole(s, o) => {
                ((POLE_HEIGHT - (s.pos[1] - o.pos[1])).max(0.0) / CLIMB_PACE) as f64 + POLE_TOP_FRAMES
            }
            K::PoleTop => {
                frames_for(dist(s.pos, self.point(o))) + POLE_GRAB_FRAMES + (POLE_HEIGHT / CLIMB_PACE) as f64 + POLE_TOP_FRAMES
            }
        })
    }
    fn success_value(&self) -> f64 {
        match self.k {
            K::ChompPost => 15.0,
            K::Cap => 10.0,
            K::Pillar | K::IslandTop => 8.0,
            K::PyramidPillar => 6.0,
            K::Book(_) => 4.0,
            K::Purple | K::BlueCoin => 3.0,
            K::Diamond | K::PostCircle => 2.0,
            K::Box | K::PoleTop => 0.5,
        }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        obrrt_core::Arrival::Stop
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let o = self.tgt.find(s)?;
        match self.k {
            K::PoleTop if self.on_pole(s, o) => {
                // Above Mario on the pole: the steering climbs (stick up,
                // mario_actions_automatic.c) and at the top stands on it.
                Some([o.pos[0], s.pos[1] + 300.0, o.pos[2]])
            }
            // Poles are grabbed by jumping into them: keep the target above.
            K::PoleTop => Some([o.pos[0], o.pos[1].max(s.pos[1]) + 100.0, o.pos[2]]),
            K::PostCircle if hdist(s.pos, o.pos) < 600.0 => {
                // Run around it: aim a bit ahead along a 220-unit circle.
                let a = (s.pos[0] - o.pos[0]).atan2(s.pos[2] - o.pos[2]) + 0.8;
                Some([o.pos[0] + 220.0 * a.sin(), s.pos[1], o.pos[2] + 220.0 * a.cos()])
            }
            _ => Some(self.point(o)),
        }
    }
    fn steer_button(&self, s: &St) -> u16 {
        let Some(o) = self.tgt.find(s) else { return 0 };
        let hd = hdist(s.pos, o.pos);
        if self.pound() {
            // The island top only needs a pound landing within 500 of it.
            let near = if self.k == K::IslandTop { ISLAND_NEAR } else { 70.0 };
            return match (airborne(s), s.action == ACT_GROUND_POUND_LAND) {
                (_, true) => 0,
                (true, _) if hd < near => Z_TRIG,
                (false, _) if hd < near * 1.5 => A_BUTTON,
                _ => 0,
            };
        }
        match self.k {
            K::Book(_) if dist(s.pos, o.pos) < 160.0 => B_BUTTON,
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
    fn kinship(&self) -> u32 {
        match self.k {
            K::ChompPost | K::Cap | K::Pillar | K::PyramidPillar | K::Book(_) | K::IslandTop => 10,
            _ => 6,
        }
    }
}

/// Objectives `o` offers, seen from base state `b` (called for every
/// object within range of `b`; `b` is never a dialog or menu state).
pub fn generate(b: &St, o: &Obj, out: &mut Vec<Arc<dyn Objective>>) {
    let level = b.level as u32;
    let made = match o.bhv_name() {
        // Hidden-box switches reveal stairs of boxes (WDW, DDD, BitDW, TTM,
        // JRB, THI); grills open the cages in BoB and HMC; animating ones
        // raise BitDW / BITS / RR platforms.
        "bhvFloorSwitchHiddenObjects" | "bhvFloorSwitchAnimatesObject" => Some((K::Purple, 4.0, None)),
        "bhvFloorSwitchGrills" => Some((K::Purple, 3.0, None)),
        "bhvFloorSwitchHardcodedModel" => Some((K::Purple, 2.0, None)),
        // Wing (TOTWC), metal (COTMC), vanish (VCUTM): they turn the course
        // ! boxes solid everywhere.
        // OBRRT_GOAL_WIDE=1: a goal, as the star of its course is (the
        // switch is what the course is for; campaign.py `CAP`).
        "bhvCapSwitch" => Some((K::Cap, if super::warps::goal_wide() { 25.0 } else { 10.0 }, None)),
        "bhvBlueCoinSwitch" => Some((K::BlueCoin, 3.0, None)),
        // A goal (over OBRRT_GOAL_MIN) while the campaign's moat step asks for the drain (OBRRT_STOP_FLAG
        // with 0x200): no castle tree pounded them otherwise, and the vanish cap course is behind the moat.
        "bhvWaterLevelPillar" if level == LEVEL_CASTLE => Some((K::Pillar, if moat_step() { 40.0 } else { 7.0 }, None)),
        // BoB's star 6, "Behind Chain Chomp's Gate".
        "bhvWoodenPost" if level == LEVEL_BOB && is_chomp_post(b, o) => Some((K::ChompPost, 20.0, Some((1, 5)))),
        // Coin posts: BoB's four and THI's two (macro_wooden_post).
        "bhvWoodenPost" if o.beh_params & 0xFF00 == 0 && !is_chomp_post(b, o) => Some((K::PostCircle, 2.0, None)),
        // Draining THI opens the way to Wiggler (star 6, "Make Wiggler Squirm").
        "bhvThiTinyIslandTop" if level == LEVEL_THI => Some((K::IslandTop, 8.0, Some((13, 5)))),
        "bhvWaterLevelDiamond" => Some((K::Diamond, 4.0, None)),
        // SSL star 4, "Stand Tall on the Four Pillars".
        "bhvPyramidPillarTouchDetector" if level == LEVEL_SSL => Some((K::PyramidPillar, 8.0, Some((8, 3)))),
        // BBH star 3, "Secret of the Haunted Books".
        "bhvBookSwitch" if level == LEVEL_BBH => Some((K::Book(((o.beh_params >> 16) & 0xFF) as u8), 5.0, Some((5, 2)))),
        "bhvPushableMetalBox" => Some((K::Box, 1.5, None)),
        _ if o.interact_type == INTERACT_POLE => {
            let w = match o.bhv_name() {
                // WF's giant pole leads to the top of the fortress area.
                "bhvGiantPole" => 3.0,
                // OBRRT_GROUNDS_FIX=1: the grounds' trees lead nowhere (no
                // star or warp at their tops); a quarter of the weight.
                "bhvTree" if level == LEVEL_CASTLE_GROUNDS && crate::extra::warps::grounds_fix() => 0.1,
                "bhvTree" => 0.4,
                "bhvKoopaFlag" => 0.2,
                _ => 0.6,
            };
            Some((K::PoleTop, w, None))
        }
        _ => None,
    };
    let Some((k, w, star)) = made else { return };
    // SSL's pillars also open the pyramid's top hole (`pyramid::TopHole`, the way down to "Pyramid Puzzle", star
    // 5), so they count while that star is missing even once "Stand Tall on the Four Pillars" is had.
    let star = if k == K::PyramidPillar && !b.has_star(8, 5) { None } else { star };
    if star.is_some_and(|(c, i)| b.has_star(c, i)) {
        return;
    }
    let sw = Switch::new(b, o, k, w, star);
    if sw.ready(b, o) {
        out.push(Arc::new(sw));
    }
}

/// Objectives that come from the level rather than one object (called once
/// per `b`).
pub fn generate_level(_b: &St, _out: &mut Vec<Arc<dyn Objective>>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use obrrt_core::{fold, start_eta};

    fn st(pos: [f32; 3], action: u32, post_y: f32) -> St {
        let post = Obj {
            id: ObjId { bhv: obrrt_core::bhv_addr("bhvWoodenPost").unwrap_or(1), home: [100, 0, 200] },
            slot: 7,
            bhv: obrrt_core::bhv_addr("bhvWoodenPost").unwrap_or(1),
            pos: [100.0, post_y, 200.0],
            home: [100.0, 0.0, 200.0],
            interact_type: 0,
            interact_status: 0,
            action: 0,
            health: 0,
            value: 0,
            loot_coins: 0,
            beh_params: 0,
            held_state: 0,
            intangible_timer: 0,
            yaw: 0,
            raw_1ac: [0; 4],
        };
        St {
            frame: 0,
            pos,
            vel: [0.0; 3],
            forward_vel: 0.0,
            yaw: 0,
            pitch: 0,
            angle_vel: [0; 3],
            water_level: -11000,
            cam_yaw: 0,
            action,
            mario_flags: 0,
            holding: false,
            held_slot: None,
            red_coins: 0,
            level: LEVEL_BOB as u16,
            area: 1,
            act: 1,
            course: 1,
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
            objects: Some(Arc::new(vec![post])),
        }
    }

    fn post_obj(s: &St) -> Switch {
        Switch::new(s, &s.objects()[0], K::ChompPost, 20.0, Some((1, 5)))
    }

    #[test]
    fn chomp_post_pound_done_and_eta_drops() {
        let walk = 0x0400_0440;
        let start = st([1000.0, 0.0, 200.0], walk, 0.0);
        let obj = post_obj(&start);
        let traj = vec![
            start.clone(),
            st([500.0, 0.0, 200.0], walk, 0.0),
            // In a ground pound above the post.
            st([100.0, 300.0, 200.0], ACT_GROUND_POUND, 0.0),
            st([100.0, 150.0, 200.0], ACT_GROUND_POUND, 0.0),
            // Landed: the post sank 65.
            st([100.0, 35.0, 200.0], ACT_GROUND_POUND_LAND, -65.0),
        ];
        let (met, etas) = fold(&obj, &traj);
        assert_eq!(met, Some(4), "{etas:?}");
        let f: Vec<f64> = etas.iter().map(|e| e.frames()).collect();
        assert!(f.windows(2).all(|w| w[1] < w[0]), "ETA not decreasing: {f:?}");
        // A sinking of less than 50 is not a pound.
        assert!(!obj.eta(&st([100.0, 35.0, 200.0], walk, -30.0), &start, &mut Mem::default()).done());
    }

    #[test]
    fn deeper_post_ranks_first() {
        let at = |y| st([600.0, 0.0, 200.0], 0x0400_0440, y);
        let obj = post_obj(&at(0.0));
        let (e0, e1, e2) = (start_eta(&obj, &at(0.0)), start_eta(&obj, &at(-65.0)), start_eta(&obj, &at(-130.0)));
        assert!(e0 > e1 && e1 > e2, "{e0} {e1} {e2}");
        // Fully down (190) is Done from any start.
        let s = at(-190.0);
        assert!(obj.eta(&s, &at(0.0), &mut Mem::default()).done());
    }

    #[test]
    fn post_circle_counts_laps_in_window() {
        let mk = |a: f32, frame: u32| {
            let mut s = st([100.0 + 220.0 * a.sin(), 0.0, 200.0 + 220.0 * a.cos()], 0x0400_0440, 0.0);
            s.frame = frame;
            Arc::make_mut(s.objects.as_mut().unwrap())[0].loot_coins = 5;
            s
        };
        let s0 = mk(0.0, 0);
        let obj = Switch::new(&s0, &s0.objects()[0], K::PostCircle, 2.0, None);
        let mut mem = Mem::default();
        let mut last = f64::INFINITY;
        for i in 0..40 {
            let e = obj.eta(&mk(i as f32 * 0.3, i), &s0, &mut mem).frames();
            assert!(e <= last + 1e-9, "frame {i}: {e} > {last}");
            last = e;
        }
        // Past the 200-frame window the laps restart (leave and re-enter).
        let late = obj.eta(&mk(12.3, 260), &s0, &mut mem).frames();
        assert!(late > last);
    }
}

#[cfg(test)]
mod restored_tests {
    use super::*;
    use crate::extra::tech::testkit::{ACT_WALKING, obj, st, with};
    use obrrt_core::fold;

    #[test]
    fn box_push_counts_down_to_200() {
        let mut b = obj("bhvPushableMetalBox", 4, [500.0, 0.0, 500.0], 0, 0);
        let at = |mario: [f32; 3], bx: f32, b: &mut Obj| {
            b.pos = [bx, 0.0, 500.0];
            with(st(LEVEL_BBH, 1, mario, ACT_WALKING), vec![b.clone()])
        };
        let s0 = at([0.0, 0.0, 500.0], 500.0, &mut b);
        let sw = Switch::new(&s0, &s0.objects()[0], K::Box, 1.5, None);
        let traj = vec![
            s0.clone(),
            at([350.0, 0.0, 500.0], 500.0, &mut b),
            at([450.0, 0.0, 500.0], 600.0, &mut b),
            at([550.0, 0.0, 500.0], 700.0, &mut b),
        ];
        let (met, etas) = fold(&sw, &traj);
        assert_eq!(met, Some(3), "{etas:?}");
        let f: Vec<f64> = etas.iter().map(|e| e.frames()).collect();
        assert!(f.windows(2).all(|w| w[1] < w[0]), "ETA not decreasing: {f:?}");
    }

    #[test]
    fn island_top_pound_phases() {
        let top = |action: i32| obj("bhvThiTinyIslandTop", 4, [0.0, 500.0, 0.0], action, 0);
        let at = |p: [f32; 3], a: u32, floor: f32, act: i32| {
            let mut s = with(st(LEVEL_THI, 2, p, a), vec![top(act)]);
            s.floor_y = floor;
            s
        };
        let s0 = at([2000.0, 300.0, 0.0], ACT_WALKING, 300.0, 0);
        let sw = Switch::new(&s0, &s0.objects()[0], K::IslandTop, 8.0, Some((13, 5)));
        let traj = vec![
            s0.clone(),
            at([600.0, 400.0, 0.0], ACT_WALKING, 400.0, 0),
            at([300.0, 700.0, 0.0], ACT_GROUND_POUND, 500.0, 0),
            at([300.0, 520.0, 0.0], ACT_GROUND_POUND, 500.0, 0),
            at([300.0, 500.0, 0.0], ACT_GROUND_POUND_LAND, 500.0, 1),
        ];
        let (met, etas) = fold(&sw, &traj);
        assert_eq!(met, Some(4), "{etas:?}");
        let f: Vec<f64> = etas.iter().map(|e| e.frames()).collect();
        assert!(f.windows(2).all(|w| w[1] < w[0]), "ETA not decreasing: {f:?}");
    }
}

/// The campaign's moat step is on: OBRRT_STOP_FLAG includes SAVE_FLAG_MOAT_DRAINED (0x200).
fn moat_step() -> bool {
    obrrt_core::config::current().stop.flag & 0x200 != 0
}
