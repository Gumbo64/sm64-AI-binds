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
//! - `Boo::GhostHunt`: all five of BBH's ghost hunt at once (see below).
//!
//! Big Boos are bosses (the `bosses` category).

use obrrt_core::buttons::B_BUTTON;
use obrrt_core::layout::LEVEL_BBH;
use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St, dist, frames_for};

use super::common::find;
use super::mansion;

/// The point behind it Mario attacks from (v1's steering and progress).
const BACK: f32 = 130.0;
/// At its back to its death: the punch / kick wind-up and hit (~1/4 s).
const HIT_FRAMES: f64 = 8.0;

/// A Boo hit to its dialog opening: its flight (30 frames), then the dialog
/// box's 12.
const DYING_FRAMES: f64 = 30.0;
const DIALOG_FRAMES: f64 = 12.0;

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

    /// Dying: hit (action 3, 30 frames of flying off) or waiting for its
    /// dialog (action 4; boo_act_3 / boo_act_4).
    fn dying(o: Option<&Obj>) -> bool {
        o.is_some_and(|o| o.action == 3 || o.action == 4)
    }

    /// Done: gone, or its death dialog is open. A Boo with its 2nd behaviour
    /// byte 0 (every ghost hunt, merry-go-round and BBH one) does not just
    /// vanish; it disables itself and shows a dialog (107; 108 with the last
    /// of the ghost hunt) that takes Mario's control until it is read.
    /// Counting the kill at the hit left every kill's node mid-flight, 30
    /// frames short of a dialog no segment of the next goal could get out of
    /// (t2: 59 kills, none chained, 51 nodes with no child); with the dialog
    /// open the node is a dialog state, which `Dialog::Read` finishes.
    fn finished(s: &St, o: Option<&Obj>) -> bool {
        match o {
            None => true,
            Some(o) if o.action == 3 => o.beh_params >> 16 & 0xFF != 0,
            Some(o) => o.action == 4 && s.dialog >= 0,
        }
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
        s.zone() == self.zone && !Self::dying(find(s, self.id, false)) && find(s, self.id, false).is_some()
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 240 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let o = find(s, self.id, false);
        if Self::finished(s, o) {
            return Eta::Done;
        }
        if o.is_some_and(|o| o.action == 3) {
            // Flying off: 30 frames (boo_update_during_death), then the dialog.
            return Eta::Frames(DYING_FRAMES);
        }
        if Self::dying(o) {
            return Eta::Frames(DIALOG_FRAMES);
        }
        let o = o.expect("not finished and not dying, so present");
        // To its back (it vanishes when faced), then the punch. In BBH the
        // way is the mansion's (rooms, doors, the elevator down), up to
        // the Boo, less the BACK that the walk ends short of.
        if mansion::in_bbh(s) {
            return Eta::Frames((mansion::walk_frames(s, o.pos) - frames_for(BACK)).max(0.0) + HIT_FRAMES);
        }
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
        // BBH: round the walls, to the Boo's room, before its back.
        if mansion::in_bbh(s) && s.hdist(o.pos) > 700.0 {
            return Some(mansion::waypoint(s, o.pos, 600.0));
        }
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

// ---------------------------------------------------------------------------
// The whole ghost hunt.

/// A walk from one Boo's home to the next, in frames: the five lie 2000 to
/// 5000 apart through the mansion (levels/bbh/script.c); a typical hop.
const HOP_FRAMES: f64 = 150.0;
/// One Boo's hit and death, when already at it (see [`HIT_FRAMES`]).
const KILL_FRAMES: f64 = HIT_FRAMES + DYING_FRAMES + DIALOG_FRAMES;

/// `Boo::GhostHunt`: kill all five Boos of BBH's act 1 (star 1, "Go on a
/// Ghost Hunt"). Each Boo alone (`Boo::HitFromBehind`) was a goal of its own
/// whose success left the others as far as ever, so no state with two dead
/// was ever a better start than one with none (t2: 59 kills, in a lineage at
/// most one distinct Boo). This one counts the Boos left: the walk to the
/// nearest and a hop and a kill for each other, so every kill is progress
/// and the states that killed more are the ones the next attempts start from.
/// Done when none lives and the last one's dialog is open (the Big Boo then
/// wakes, `BigBoo::Hit`; boo.inc.c boo_act_4: dialog 108 with the last).
pub struct GhostHunt {
    zone: (u16, u8),
}

impl GhostHunt {
    pub fn new(b: &St) -> Option<GhostHunt> {
        let star1 = b.has_star(5, 0);
        (mansion::in_bbh(b) && b.act == 1 && !star1 && Self::alive(b).next().is_some()).then(|| GhostHunt { zone: b.zone() })
    }

    /// The five's living (action 0 to 2: waiting, chasing, bounced).
    fn alive(s: &St) -> impl Iterator<Item = &Obj> {
        s.objects().iter().filter(|o| o.bhv_name() == "bhvGhostHuntBoo" && o.action < 3)
    }
}

impl Objective for GhostHunt {
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "Boo::GhostHunt"
    }
    fn key(&self) -> String {
        format!("BooGhostHunt {:?}", self.zone)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        6.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && s.act == 1 && !s.has_star(5, 0) && Self::alive(s).next().is_some()
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 24, generations: 6, horizon: 300 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let left = Self::alive(s).count();
        if left == 0 {
            let dying = s.objects().iter().any(|o| o.bhv_name() == "bhvGhostHuntBoo" && (o.action == 3 || o.action == 4));
            return if !dying || s.dialog >= 0 { Eta::Done } else { Eta::Frames(DYING_FRAMES + DIALOG_FRAMES) };
        }
        let nearest = Self::alive(s).map(|o| (mansion::walk_frames(s, o.pos) - frames_for(BACK)).max(0.0)).fold(f64::MAX, f64::min);
        Eta::Frames(nearest + left as f64 * KILL_FRAMES + (left - 1) as f64 * HOP_FRAMES)
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let o = Self::alive(s).min_by(|a, b| mansion::walk_frames(s, a.pos).total_cmp(&mansion::walk_frames(s, b.pos)))?;
        if s.hdist(o.pos) > 700.0 {
            return Some(mansion::waypoint(s, o.pos, 600.0));
        }
        let back = o.behind(BACK);
        if dist(s.pos, back) > 100.0 && dist(s.pos, o.pos) > 160.0 { Some(back) } else { Some(o.pos) }
    }
    fn steer_button(&self, s: &St) -> u16 {
        match Self::alive(s).find(|o| dist(s.pos, o.pos) < 220.0) {
            Some(_) => B_BUTTON,
            None => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
    fn kinship(&self) -> u32 {
        10
    }
}
