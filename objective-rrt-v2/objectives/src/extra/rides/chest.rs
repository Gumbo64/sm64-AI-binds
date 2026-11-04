//! `Chest::Open(set, n)`: open treasure chest `n` (1-4) of a 4-chest puzzle
//! (treasure_chest.inc.c). A chest bottom (`bhvTreasureChestBottom`, oAction
//! 0 closed) opens (oAction 1) when Mario is within 150 of it facing it
//! (his yaw + 0x8000 within 0x3000 of its yaw, i.e. standing in front of it
//! looking at it) and it is the next in order; a wrong one buzzes (oAction 2)
//! and closes them all. The 4th one completes the puzzle:
//! - JRB's ocean cave (bhvTreasureChestsJrb, behParam 0x02): star 3.
//! - DDD (bhvTreasureChests, behParam 0x01): star 2.
//! - JRB's sunken ship (bhvTreasureChestsShip): drains the ship's water,
//!   which star 1 ("Plunder in the Sunken Ship") is behind.
//!
//! Facing mode (technique `chestfacing`, CATALOGUE G29, c5 §7): the open
//! test (treasure_chest.inc.c:70-73) is `is_point_within_radius_of_mario
//! (150)` (3D) **and** `obj_check_if_facing_toward_angle(oMoveAngleYaw,
//! gfx yaw + 0x8000, 0x3000)`: Mario's facing must be within 0x3000 of
//! the chest's facing reversed, wherever he stands, and a wrong chest
//! buzzes (resets the order) only under the same test. So a wrong chest
//! can be passed close by while facing away (the TAS passes chest 4 at
//! 189 units, f81776). With the switch: progress adds the facing error
//! (beyond 0x3000, up to 300 units) near the chest; steering detours
//! sideways past a closed wrong chest within 320 when the way to the
//! target would face it (a point 400 to its side, so the walk faces
//! across it).
//!
//! The order is the chest's `oBehParams2ndByte`, set directly by
//! spawn_treasure_chest (St doesn't have it, `oBehParams` stays 0) and the
//! bottoms have no SET_HOME, so a chest is known by its spawn x, z.

use std::sync::Arc;

use obrrt_core::layout::*;
use obrrt_core::{Eta, GaBudget, Mem, Obj, Objective, St, dist, frames_for};

use super::common::{Ident, hdist};

/// `mem.flags`: a wrong chest has buzzed this attempt (v1 cut its progress
/// there): the order is reset, so a chest after the first is lost.
const BUZZED: u32 = 1;
/// Chest 1 after a buzz: the buzzing chest's reset before all close again
/// and the order restarts (treasure_chest.inc.c, ~1 s).
const BUZZ_RESET_FRAMES: f64 = 30.0;

struct Set {
    name: &'static str,
    level: u32,
    /// Spawn x, z of chests 1 to 4 (spawn_treasure_chest calls).
    chests: [[f32; 2]; 4],
    star: (i16, u8),
    weight: f64,
}

const SETS: &[Set] = &[
    Set {
        name: "JrbCave",
        level: LEVEL_JRB,
        chests: [[-1700.0, -1150.0], [-1150.0, -1550.0], [-2400.0, -1800.0], [-1800.0, -2100.0]],
        star: (3, 2),
        weight: 12.0,
    },
    Set {
        name: "JrbShip",
        level: LEVEL_JRB,
        chests: [[400.0, -2700.0], [650.0, -940.0], [-550.0, -770.0], [100.0, -1700.0]],
        star: (3, 0),
        weight: 6.0,
    },
    Set {
        name: "Ddd",
        level: LEVEL_DDD,
        chests: [[-4500.0, 1300.0], [-1800.0, 1050.0], [-4500.0, -1100.0], [-2400.0, 125.0]],
        star: (9, 1),
        weight: 12.0,
    },
];

const BOTTOM: &str = "bhvTreasureChestBottom";

fn chest_at<'a>(s: &'a St, xz: [f32; 2]) -> Option<&'a Obj> {
    s.objects().iter().find(|o| o.bhv_name() == BOTTOM && hdist(o.pos, [xz[0], 0.0, xz[1]]) < 40.0)
}

pub fn generate(b: &St, o: &Obj, out: &mut Vec<Arc<dyn Objective>>) {
    if o.bhv_name() != BOTTOM {
        return;
    }
    for (si, set) in SETS.iter().enumerate() {
        if set.level != b.level as u32 {
            continue;
        }
        if let Some(n) = set.chests.iter().position(|c| hdist(o.pos, [c[0], 0.0, c[1]]) < 40.0) {
            let facing = crate::extra::tech::enabled("chestfacing");
            out.push(Arc::new(Open { set: si, n, ident: Ident::of(o, false), zone: b.zone(), facing }));
        }
    }
}

pub struct Open {
    set: usize,
    /// 0-based order.
    n: usize,
    ident: Ident,
    zone: (u16, u8),
    /// `chestfacing`.
    facing: bool,
}

/// How far Mario's facing is from opening chest `o` (0 = exactly; the
/// chest opens below 0x3000): |(s16)(yaw + 0x8000 - chest yaw)|.
fn facing_err(yaw: i32, o: &Obj) -> u16 {
    ((yaw + 0x8000 - o.yaw) as i16).unsigned_abs()
}

/// SM64 yaw from `a` to `b`.
fn yaw_to(a: [f32; 3], b: [f32; 3]) -> i32 {
    ((b[0] - a[0]).atan2(b[2] - a[2]) * 32768.0 / std::f32::consts::PI) as i32
}

impl Open {
    fn set(&self) -> &'static Set {
        &SETS[self.set]
    }

    fn chest<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        if s.zone() != self.zone {
            return None;
        }
        self.ident.find(s)
    }

    /// The point in front of the chest where Mario faces it.
    fn front(o: &Obj, d: f32) -> [f32; 3] {
        let a = o.yaw as f32 * std::f32::consts::TAU / 65536.0;
        [o.pos[0] + d * a.sin(), o.pos[1], o.pos[2] + d * a.cos()]
    }

    /// The chests' oActions in order (None if not loaded).
    fn actions(&self, s: &St) -> [Option<i32>; 4] {
        self.set().chests.map(|c| chest_at(s, c).map(|o| o.action))
    }

    /// Exactly the chests before this one are open, this one is closed.
    fn ready(&self, s: &St) -> bool {
        let a = self.actions(s);
        a.iter().enumerate().all(|(i, a)| match a {
            Some(a) => {
                if i < self.n {
                    *a == 1
                } else {
                    *a == 0
                }
            }
            None => false,
        })
    }

    /// Facing mode: a point beside a closed wrong chest within 320 of Mario
    /// when heading for `t` would face it (within 0x3800), 400 to the side
    /// he's on, so he walks across its facing instead of into it.
    fn detour(&self, s: &St, t: [f32; 3]) -> Option<[f32; 3]> {
        let heading = yaw_to(s.pos, t);
        self.set().chests.iter().enumerate().filter(|(i, _)| *i != self.n).find_map(|(_, c)| {
            let o = chest_at(s, *c)?;
            if o.action != 0 || dist(s.pos, o.pos) > 320.0 || facing_err(heading, o) >= 0x3800 {
                return None;
            }
            let a = o.yaw as f32 * std::f32::consts::TAU / 65536.0;
            // The chest's sideways axis, signed to Mario's side.
            let (px, pz) = (a.cos(), -a.sin());
            let side = if (s.pos[0] - o.pos[0]) * px + (s.pos[2] - o.pos[2]) * pz >= 0.0 { 1.0 } else { -1.0 };
            Some([o.pos[0] + 400.0 * side * px, o.pos[1], o.pos[2] + 400.0 * side * pz])
        })
    }

    /// A chest went from not-buzzing to buzzing (the order was broken).
    fn buzzed(&self, s0: &St, s: &St) -> bool {
        let (a0, a) = (self.actions(s0), self.actions(s));
        a0.iter().zip(a.iter()).any(|(x, y)| *x != Some(2) && *y == Some(2))
    }
}

impl Objective for Open {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "Chest::Open"
    }
    fn key(&self) -> String {
        format!("Chest {:?} {} {} {}", self.zone, self.set().name, self.n + 1, self.ident.key())
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        self.set().weight
    }
    fn base_criteria(&self, s: &St) -> bool {
        let (c, i) = self.set().star;
        s.zone() == self.zone && !s.has_star(c, i) && self.ready(s)
    }
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let Some(o) = self.chest(s) else { return Eta::Lost };
        if o.action == 1 {
            return Eta::Done;
        }
        if self.buzzed(start, s) {
            mem.flags |= BUZZED;
        }
        let buzzing = self.actions(s).contains(&Some(2));
        let reset = if mem.flags & BUZZED != 0 {
            // The chests before this one closed with the buzz.
            if self.n > 0 {
                return Eta::Lost;
            }
            if buzzing { BUZZ_RESET_FRAMES } else { 0.0 }
        } else {
            0.0
        };
        let d = dist(s.pos, Self::front(o, 120.0)).min(dist(s.pos, o.pos));
        // Facing mode: near the chest, also the facing it needs (v1: up to
        // 300 units).
        let turn = if self.facing && dist(s.pos, o.pos) < 300.0 {
            300.0 * (facing_err(s.yaw as i32, o).saturating_sub(0x3000)) as f32 / 0x5000 as f32
        } else {
            0.0
        };
        Eta::Frames(reset + frames_for(d + turn))
    }
    fn success_value(&self) -> f64 {
        3.0
    }
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 16, generations: 5, horizon: 200 })
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        obrrt_core::Arrival::Stop
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // To the front first, then walk into it (so Mario faces it).
        let o = self.chest(s)?;
        let f = Self::front(o, 120.0);
        let t = if dist(s.pos, f) > 60.0 && dist(s.pos, o.pos) > 140.0 { f } else { o.pos };
        if self.facing {
            return Some(self.detour(s, t).unwrap_or(t));
        }
        Some(t)
    }
    fn exists_in(&self, s: &St) -> bool {
        self.chest(s).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::testkit::{obj, st};
    use obrrt_core::fold;

    fn chests(actions: [i32; 4]) -> Vec<Obj> {
        SETS[0]
            .chests
            .iter()
            .zip(actions)
            .enumerate()
            .map(|(i, (c, a))| obj(BOTTOM, 20 + i as u16, [c[0], 0.0, c[1]], a, 0))
            .collect()
    }

    /// Chest 2 of JRB's cave: closer is less; a wrong chest buzzing loses
    /// it for good (the order reset), even after the buzz ends; chest 1
    /// survives a buzz.
    #[test]
    fn chest_buzz_latch() {
        let at = |x: f32, actions| st(LEVEL_JRB, [x, 0.0, -1550.0 + 120.0], chests(actions));
        let s0 = at(-1500.0, [1, 0, 0, 0]);
        let o = &s0.objects()[1];
        let open = |n| Open { set: 0, n, ident: Ident::of(o, false), zone: s0.zone(), facing: false };
        let second = open(1);
        let e = fold(&second, &[s0.clone(), at(-1300.0, [1, 0, 0, 0]), at(-1300.0, [1, 0, 2, 0]), at(-1300.0, [0, 0, 0, 0])]).1;
        assert!(e[1].frames() < e[0].frames());
        assert_eq!(e[2], Eta::Lost);
        assert_eq!(e[3], Eta::Lost);
        assert_eq!(fold(&second, &[s0.clone(), at(-1150.0, [1, 1, 0, 0])]).0, Some(1));

        let s1 = at(-1500.0, [0, 0, 0, 0]);
        let first = Open { set: 0, n: 0, ident: Ident::of(&s1.objects()[0], false), zone: s1.zone(), facing: false };
        let e = fold(&first, &[s1.clone(), at(-1500.0, [0, 0, 2, 0]), at(-1500.0, [0, 0, 0, 0])]).1;
        assert!(matches!(e[1], Eta::Frames(_)) && e[2].frames() < e[1].frames());
    }
}
