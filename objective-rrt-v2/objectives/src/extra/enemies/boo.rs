//! `Boo::HitFromBehind(id)`: punch or kick a Boo from behind.
//!
//! A Boo is only solid (`INTERACT_BOUNCE_TOP`) while it is opaque, and it
//! turns see-through when it faces Mario while Mario faces it
//! (boo_vanish_or_appear, boo_chase_mario), so it has to be approached from
//! behind, facing its back. Stomping only bounces off it; any other attack
//! kills it (boo_get_attack_status: action 3, then 4 for the ghost hunt
//! ones; boo.inc.c). Every small Boo carries a coin, blue in BBH
//! (coin_inside_boo_act_0, coin.inc.c).
//!
//! - bhvGhostHuntBoo: BBH act 1's five Boos; with all dead the Big Boo of
//!   star 1 appears (the courtyard triplets use it too, with yellow coins).
//! - bhvMerryGoRoundBoo: the merry-go-round's five, for star 2's Big Boo.
//! - bhvBoo: BBH's other Boos (blue coins).
//! - bhvBooWithCage: the courtyard's caged Boo (12+ stars); killing it drops
//!   the cage that is the way into BBH (bhv_boo_with_cage_*).
//!
//! Big Boos are bosses (the `bosses` category).

use obrrt_core::buttons::B_BUTTON;
use obrrt_core::layout::LEVEL_BBH;
use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St, dist, frames_for};

use super::common::find;

/// The point behind it Mario attacks from (v1's steering and progress).
const BACK: f32 = 130.0;
/// At its back to its death: the punch / kick wind-up and hit (~1/4 s).
const HIT_FRAMES: f64 = 8.0;

pub struct HitFromBehind {
    id: ObjId,
    zone: (u16, u8),
    /// Base weight, and the star its death counts towards (weight 6 then).
    weight: f64,
    towards: Option<(i16, u8)>,
}

impl HitFromBehind {
    pub fn new(b: &St, o: &Obj) -> Option<HitFromBehind> {
        let bbh = b.level as u32 == LEVEL_BBH;
        let (weight, towards) = match o.bhv_name() {
            "bhvBooWithCage" => (15.0, None),
            "bhvGhostHuntBoo" if bbh => (2.0, Some((5, 0))),
            "bhvGhostHuntBoo" => (1.0, None),
            "bhvMerryGoRoundBoo" => (2.0, Some((5, 1))),
            "bhvBoo" => (if bbh { 2.0 } else { 1.0 }, None),
            _ => return None,
        };
        Some(HitFromBehind { id: o.id, zone: b.zone(), weight, towards })
    }

    fn dead(o: Option<&Obj>) -> bool {
        o.is_none_or(|o| o.action == 3 || o.action == 4)
    }
}

impl Objective for HitFromBehind {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "Boo::HitFromBehind"
    }
    fn key(&self) -> String {
        format!("BooHit {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        match self.towards {
            Some((c, i)) if !b.has_star(c, i) => 6.0,
            _ => self.weight,
        }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && !Self::dead(find(s, self.id, false))
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 180 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let o = find(s, self.id, false);
        if Self::dead(o) {
            return Eta::Done;
        }
        let o = o.expect("not dead, so present");
        // To its back (it vanishes when faced), then the punch.
        Eta::Frames(frames_for(dist(s.pos, o.behind(BACK))) + HIT_FRAMES)
    }
    fn success_value(&self) -> f64 {
        self.weight.min(5.0)
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let o = find(s, self.id, false)?;
        let back = o.behind(BACK);
        if dist(s.pos, back) > 100.0 && dist(s.pos, o.pos) > 160.0 { Some(back) } else { Some(o.pos) }
    }
    fn steer_button(&self, s: &St) -> u16 {
        match find(s, self.id, false) {
            Some(o) if dist(s.pos, o.pos) < 220.0 => B_BUTTON,
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}
