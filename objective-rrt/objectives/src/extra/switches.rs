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
use obrrt_core::layout::{ACT_FLAG_AIR, ACT_FLAG_ON_POLE, INTERACT_POLE, LEVEL_BBH, LEVEL_BOB, LEVEL_CASTLE, LEVEL_SSL, LEVEL_THI};
use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

// Mario actions, from include/sm64.h.
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

    /// Whether `s` (relative to the attempt's start `s0`) has it activated.
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
                (Some(a), Some(b)) => hdist(a.pos, b.pos) >= 200.0,
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
    fn base_heuristic(&self, s: &St) -> f64 {
        let Some(o) = self.tgt.find(s) else { return f64::MAX };
        match self.k {
            // Already on the pole: by how far is left to climb.
            K::PoleTop if self.on_pole(s, o) => -2.0,
            // A post half pounded is better.
            K::ChompPost => dist(s.pos, self.point(o)) as f64 / 400.0 + (o.pos[1] - (o.home[1] - 190.0)) as f64 / 40.0,
            _ => dist(s.pos, self.point(o)) as f64 / 400.0,
        }
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let s0 = &traj[0];
        traj.iter().position(|s| self.done(s0, s))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let s0 = &traj[0];
        let start = self.tgt.find(s0).map(|o| o.pos);
        let mut best = (0usize, f64::MIN);
        // Post::Circle: the angle swept around the post while within 400.
        let mut swept = 0.0f64;
        let mut prev_angle: Option<f64> = None;
        for (i, s) in traj.iter().enumerate() {
            if s.zone() != self.zone {
                continue;
            }
            let Some(o) = self.tgt.find(s) else { continue };
            let d = dist(s.pos, self.point(o)) as f64;
            let score = match self.k {
                K::ChompPost => -(o.pos[1] - (o.home[1] - 190.0)) as f64 * 10.0 - hdist(s.pos, o.pos) as f64,
                K::Box => start.map_or(0.0, |p| hdist(p, o.pos) as f64 * 10.0) - d,
                K::PoleTop if self.on_pole(s, o) => 10000.0 + (s.pos[1] - o.pos[1]) as f64,
                K::PostCircle => {
                    if hdist(s.pos, o.pos) > 380.0 {
                        swept = 0.0;
                        prev_angle = None;
                        -d
                    } else {
                        let a = ((s.pos[0] - o.pos[0]) as f64).atan2((s.pos[2] - o.pos[2]) as f64);
                        if let Some(p) = prev_angle {
                            let mut da = a - p;
                            while da > std::f64::consts::PI {
                                da -= std::f64::consts::TAU;
                            }
                            while da < -std::f64::consts::PI {
                                da += std::f64::consts::TAU;
                            }
                            swept += da;
                        }
                        prev_angle = Some(a);
                        swept.abs() * 1000.0
                    }
                }
                _ => -d,
            };
            if score > best.1 {
                best = (i, score);
            }
        }
        Progress { score: best.1, at: best.0 }
    }
    fn budget(&self) -> Budget {
        let (population, generations, horizon) = match self.k {
            K::Purple | K::Book(_) => (16, 5, 150),
            K::Cap | K::Diamond | K::Box => (20, 5, 200),
            K::BlueCoin | K::Pillar | K::IslandTop => (20, 6, 200),
            K::ChompPost => (24, 6, 300),
            K::PostCircle | K::PoleTop | K::PyramidPillar => (20, 5, 240),
        };
        Budget::Ga { population, generations, horizon }
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
            let near = if self.k == K::IslandTop { 300.0 } else { 70.0 };
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
        "bhvCapSwitch" => Some((K::Cap, 10.0, None)),
        "bhvBlueCoinSwitch" => Some((K::BlueCoin, 3.0, None)),
        "bhvWaterLevelPillar" if level == LEVEL_CASTLE => Some((K::Pillar, 7.0, None)),
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
                "bhvTree" => 0.4,
                "bhvKoopaFlag" => 0.2,
                _ => 0.6,
            };
            Some((K::PoleTop, w, None))
        }
        _ => None,
    };
    let Some((k, w, star)) = made else { return };
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
