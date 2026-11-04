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
//! The ETA is the model's predicted frames scaled by the fraction of the
//! start's distance to the target still left (the object's position as it
//! moves for `Object`); Done is v1's success predicate. Kinds
//! are per style and verb (`Model::Pound`, ...), so the pool's rate sampler
//! learns during the run which of the model's ideas pay.

use obrrt_core::{Eta, Mem, Objective, St, dist, frames_for};

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

    fn met(&self, prev: &St, prev_obj: bool, s: &St) -> bool {
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
                    "defeat" => s.objects.is_some() && p.is_none() && prev_obj,
                    _ => false,
                }
            }
        }
    }

    /// The previous frame as far as [`Learned::met`] reads it, rebuilt from
    /// what [`Learned::remember`] kept in `mem` (None on the first frame).
    fn prev_of(s: &St, mem: &Mem) -> Option<St> {
        if mem.flags & PREV_VALID == 0 {
            return None;
        }
        let (a, b, c) = (mem.f[0] as u64, mem.f[1] as u64, mem.f[2] as u64);
        let mut p = s.clone();
        p.objects = None;
        p.action = a as u32;
        p.health = b as u16 as i16;
        p.lives = (b >> 16) as u8 as i8;
        p.keys = (b >> 24) as u8;
        p.level = (b >> 32) as u16;
        p.coins = c as u16 as i16;
        p.stars = (c >> 16) as u16 as i16;
        p.red_coins = (c >> 32) as u16 as i16;
        p.area = mem.f[3] as u8;
        p.holding = mem.flags & PREV_HOLDING != 0;
        p.dialog = if mem.flags & PREV_DIALOG != 0 { 0 } else { -1 };
        Some(p)
    }

    /// Keeps the fields of `s` that `met` reads of the previous frame (all
    /// exact in an f64: at most 48 bits each).
    fn remember(&self, s: &St, mem: &mut Mem) {
        mem.f[0] = s.action as f64;
        mem.f[1] = (s.health as u16 as u64 | (s.lives as u8 as u64) << 16 | (s.keys as u64) << 24 | (s.level as u64) << 32) as f64;
        mem.f[2] = (s.coins as u16 as u64 | (s.stars as u16 as u64) << 16 | (s.red_coins as u16 as u64) << 32) as f64;
        mem.f[3] = s.area as f64;
        let mut fl = PREV_VALID;
        if s.holding {
            fl |= PREV_HOLDING;
        }
        if s.dialog >= 0 {
            fl |= PREV_DIALOG;
        }
        if self.obj_pos(s).is_some() {
            fl |= PREV_OBJ;
        }
        mem.flags = fl;
    }
}

const PREV_VALID: u32 = 1;
const PREV_HOLDING: u32 = 2;
const PREV_DIALOG: u32 = 4;
/// The followed object was there on the previous frame (`defeat`).
const PREV_OBJ: u32 = 8;

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
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        // v1's success held only before the first life lost.
        if !s.alive() || s.lives < start.lives {
            return Eta::Lost;
        }
        let prev = Self::prev_of(s, mem);
        let prev_obj = mem.flags & PREV_OBJ != 0;
        self.remember(s, mem);
        if let Some(p) = prev {
            if self.met(&p, prev_obj, s) {
                return Eta::Done;
            }
        }
        if s.zone() != self.zone {
            return if matches!(self.style, Style::Exit) { Eta::Frames(0.0) } else { Eta::Lost };
        }
        let d = s.dist(self.target_at(s));
        let d0 = start.dist(self.target_at(start));
        let mut f = if self.budget > 0.0 && d0 > 1.0 { self.budget as f64 * (d / d0) as f64 } else { frames_for(d) };
        if let Style::Speed(v) = self.style {
            // v1 weighed missing speed at 20 units of distance per u/frame.
            f += frames_for(20.0 * (v - s.forward_vel.abs()).max(0.0));
        }
        Eta::Frames(f)
    }
    fn needs_objects(&self) -> bool {
        matches!(self.style, Style::Object { .. })
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        // v1's: the model's predicted frames, with room.
        let h = (self.budget * 1.5).clamp(30.0, 300.0) as usize;
        Some(obrrt_core::GaBudget { population: 12, generations: 4, horizon: h })
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
