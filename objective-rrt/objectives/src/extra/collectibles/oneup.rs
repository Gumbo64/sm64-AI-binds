//! `OneUp::Collect(id)`: touch a 1-up mushroom (mushroom_1up.inc.c). 1-ups
//! have no interact type: they check collision with Mario themselves and add
//! a life, so success is `lives` going up. Most move: walking ones drift,
//! running-away ones flee Mario (action 1), sliding ones roll downhill once he
//! is within 1000, pole ones home in on him; all are steered to as they move.
//!
//! The hidden ones (bhvHidden1up, bhvHidden1upInPole) stay invisible in
//! action 0 until their triggers are touched (`OneUpTrigger::Touch`), so they
//! are only attempted once revealed.

use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St};

use super::common::{closest, find};

pub fn is_one_up(bhv: &str) -> bool {
    matches!(
        bhv,
        "bhv1Up" | "bhv1upWalking" | "bhv1upRunningAway" | "bhv1upSliding" | "bhv1upJumpOnApproach" | "bhvHidden1up" | "bhvHidden1upInPole"
    )
}

pub struct Collect {
    id: ObjId,
    zone: (u16, u8),
    hidden: bool,
}

impl Collect {
    pub fn new(b: &St, o: &Obj) -> Collect {
        Collect { id: o.id, zone: b.zone(), hidden: matches!(o.bhv_name(), "bhvHidden1up" | "bhvHidden1upInPole") }
    }

    fn obj<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        // A hidden 1-up in action 0 is not out yet.
        find(s, self.zone, self.id).filter(|o| !self.hidden || o.action != 0)
    }
}

impl Objective for Collect {
    fn kind(&self) -> &'static str {
        "OneUp::Collect"
    }
    fn key(&self) -> String {
        format!("OneUp {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        1.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        self.obj(s).is_some()
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        self.obj(s).map_or(f64::MAX, |o| s.dist(o.pos) as f64 / 400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let l0 = traj[0].lives;
        traj.iter().position(|s| s.zone() == self.zone && s.lives > l0)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        closest(traj, |s| self.obj(s).map(|o| o.pos))
    }
    fn success_value(&self) -> f64 {
        0.5
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 150 }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.obj(s).map(|o| o.pos)
    }
    fn exists_in(&self, s: &St) -> bool {
        // Kept while it exists at all: a hidden one may be revealed later.
        find(s, self.zone, self.id).is_some()
    }
}
