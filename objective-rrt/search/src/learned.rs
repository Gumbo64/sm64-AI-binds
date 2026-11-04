//! Learned objectives (`OBRRT_MODEL_LEARNED=<n>`, see model.rs): objectives
//! the backbone writes itself, from its v6 edge mode (credit-weighted
//! objective-rrt edges toward a star). Each is a style plus parameters:
//!
//! - `Reach`: be within 150 units of a position (in the base zone).
//! - `Exit`: leave the zone (a door, warp, painting...), heading for a position.
//! - `Event`: fire one event bit (features::events: coin, star, key, red
//!   coin, grab, dialog, cannon, ledge grab, enter water) near a position.
//! - `Speed`: reach a position with at least some forward speed.
//! - `Object`: do a verb to one object of the scene, the one the model's
//!   pointer picked (pool slot + behaviour, followed as it moves): touch it,
//!   stand on it, ground pound onto it, grab it, or make it disappear
//!   (defeat / collect).
//!
//! Progress is how close Mario got to the target (the object's position as
//! it moves for `Object`); the budget is the model's predicted frames. Kinds
//! are per style and verb (`Model::Pound`, ...), so the pool's rate sampler
//! learns during the run which of the model's ideas pay.

use obrrt_core::{Budget, Objective, Progress, St, dist};

pub const STYLES: [&str; 5] = ["reach", "exit", "event", "speed", "object"];
pub const VERBS: [&str; 6] = ["none", "touch", "stand", "pound", "grab", "defeat"];

const RADIUS: f32 = 150.0;
const ACT_FLAG_AIR: u32 = 0x0000_0800;
const ACT_GROUND_POUND_LAND: u32 = 0x0080_023C;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Style {
    Reach,
    Exit,
    Event(u16),
    Speed(f32),
    Object { verb: u8, slot: u16, bhv: u32 },
}

#[derive(Debug, Clone)]
pub struct Learned {
    pub style: Style,
    pub zone: (u16, u8),
    pub target: [f32; 3],
    pub budget: f32,
    pub goal: i32,
}

impl Learned {
    /// The followed object's position in `s` (None: gone, or no objects read).
    fn obj_pos(&self, s: &St) -> Option<[f32; 3]> {
        let Style::Object { slot, bhv, .. } = self.style else { return None };
        s.objects.as_ref()?.iter().find(|o| o.slot == slot && o.bhv == bhv).map(|o| o.pos)
    }

    fn target_at(&self, s: &St) -> [f32; 3] {
        self.obj_pos(s).unwrap_or(self.target)
    }

    fn met(&self, prev: &St, s: &St) -> bool {
        let same = s.zone() == self.zone;
        match self.style {
            Style::Reach => same && s.dist(self.target) <= RADIUS,
            Style::Exit => !same && s.alive(),
            Style::Event(mask) => obrrt_core::features::events(prev, s) & mask as u32 != 0,
            Style::Speed(v) => same && s.dist(self.target) <= 2.0 * RADIUS && s.forward_vel.abs() >= v,
            Style::Object { verb, slot, .. } => {
                if !same {
                    return false;
                }
                let p = self.obj_pos(s);
                let xz = |p: [f32; 3]| ((s.pos[0] - p[0]).powi(2) + (s.pos[2] - p[2]).powi(2)).sqrt();
                match VERBS[verb as usize] {
                    "touch" => p.is_some_and(|p| dist(s.pos, p) < 120.0),
                    "stand" => s.floor_obj == Some(slot) && s.action & ACT_FLAG_AIR == 0,
                    "pound" => {
                        s.action == ACT_GROUND_POUND_LAND
                            && prev.action != ACT_GROUND_POUND_LAND
                            && (s.floor_obj == Some(slot) || p.is_some_and(|p| xz(p) < 200.0))
                    }
                    "grab" => s.holding && s.held_slot == Some(slot),
                    // Gone, where the previous frame still had it.
                    "defeat" => s.objects.is_some() && p.is_none() && self.obj_pos(prev).is_some(),
                    _ => false,
                }
            }
        }
    }

    /// States up to (not including) the first life lost.
    fn alive_len(traj: &[St]) -> usize {
        let l0 = traj.first().map_or(0, |s| s.lives);
        traj.iter().position(|s| s.lives < l0 || !s.alive()).unwrap_or(traj.len())
    }
}

impl Objective for Learned {
    fn kind(&self) -> &'static str {
        match self.style {
            Style::Reach => "Model::Reach",
            Style::Exit => "Model::Exit",
            Style::Event(_) => "Model::Event",
            Style::Speed(_) => "Model::Speed",
            Style::Object { verb, .. } => match VERBS[verb as usize] {
                "touch" => "Model::Touch",
                "stand" => "Model::Stand",
                "pound" => "Model::Pound",
                "grab" => "Model::Grab",
                _ => "Model::Defeat",
            },
        }
    }
    fn key(&self) -> String {
        let c = |v: f32| (v / 400.0).round() as i32;
        let t = self.target;
        match self.style {
            Style::Object { verb, slot, bhv } => format!("Model {:?} {} {slot} {bhv:x}", self.zone, VERBS[verb as usize]),
            Style::Event(m) => format!("Model {:?} event {m:x} {},{},{}", self.zone, c(t[0]), c(t[1]), c(t[2])),
            _ => format!("Model {:?} {} {},{},{}", self.zone, self.kind(), c(t[0]), c(t[1]), c(t[2])),
        }
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        1.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        if s.zone() != self.zone {
            return false;
        }
        match self.style {
            Style::Reach => s.dist(self.target) > RADIUS,
            Style::Object { verb, slot, .. } => {
                // Not already done, and (when objects were read) still there.
                let there = s.objects.is_none() || self.obj_pos(s).is_some();
                there && !(VERBS[verb as usize] == "grab" && s.holding) && !(VERBS[verb as usize] == "stand" && s.floor_obj == Some(slot))
            }
            _ => true,
        }
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        // The model's frames, as PosDist's distance / 500 is roughly seconds.
        (self.budget as f64 / 30.0).max(s.dist(self.target_at(s)) as f64 / 500.0)
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(500.0)
    }
    fn needs_objects(&self) -> bool {
        matches!(self.style, Style::Object { .. })
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let n = Self::alive_len(traj);
        (1..n).find(|&i| self.met(&traj[i - 1], &traj[i]))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let n = Self::alive_len(traj).max(1);
        let (at, d) = traj[..n]
            .iter()
            .enumerate()
            .filter(|(_, s)| s.zone() == self.zone)
            .map(|(i, s)| {
                let mut d = s.dist(self.target_at(s)) as f64;
                if let Style::Speed(v) = self.style {
                    d += 20.0 * (v - s.forward_vel.abs()).max(0.0) as f64;
                }
                (i, d)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, f64::MAX));
        Progress { score: -d, at }
    }
    fn budget(&self) -> Budget {
        let h = (self.budget * 1.5).clamp(30.0, 300.0) as usize;
        Budget::Ga { population: 12, generations: 4, horizon: h }
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        Some(self.target_at(s))
    }
}

/// One server proposal (serve_model.py op 5, 36 bytes) as an objective for a
/// state `st` (read with objects: the pointer's object gets its behaviour
/// from there). None for a pointer to an object `st` doesn't have.
pub fn from_bytes(b: &[u8], st: &St) -> Option<Learned> {
    let f = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let (style, verb) = (b[0] as usize, b[1]);
    let slot = i16::from_le_bytes([b[2], b[3]]);
    let target = [f(4), f(8), f(12)];
    let ev = u16::from_le_bytes([b[20], b[21]]);
    let speed = f(24);
    let budget = f(28);
    let goal = i32::from_le_bytes(b[32..36].try_into().unwrap());
    let style = match STYLES.get(style)? {
        &"reach" => Style::Reach,
        &"exit" => Style::Exit,
        &"event" => Style::Event(ev),
        &"speed" => Style::Speed((speed.abs() * 0.8).clamp(10.0, 40.0)),
        _ => {
            let slot = u16::try_from(slot).ok()?;
            let o = st.objects.as_ref()?.iter().find(|o| o.slot == slot)?;
            Style::Object { verb: verb.min(5), slot, bhv: o.bhv }
        }
    };
    Some(Learned { style, zone: st.zone(), target, budget, goal })
}
