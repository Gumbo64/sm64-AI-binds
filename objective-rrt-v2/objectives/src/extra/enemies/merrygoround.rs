//! `MerryGoRound::Enter`: get into BBH's merry-go-round enclosure, for star 2
//! ("Ride Big Boo's Merry-Go-Round").
//!
//! The mechanism (bbh_merry_go_round.inc.c, boo.inc.c):
//!
//! - `gMarioOnMerryGoRound` is true while Mario stands on a floor of surface
//!   type 0x1A (`SURFACE_MGR_MUSIC`): the enclosure's pit (-2764 high,
//!   x -1433..1024, z -1023..1434) and its two ramps (-2457 high, in from the
//!   doors at x -1996 and x 1587), or the merry-go-round itself.
//! - The Boos of the merry-go-round (`bhvMerryGoRoundBoo`, spawned by
//!   `bhvMerryGoRoundBooManager` at (-1100, -2372, 1100) once Mario is within
//!   1000 of it, two at a time, five in all) only move while that is true
//!   (`boo_should_be_active`, `boo_should_be_stopped`); off it they stand
//!   still, intangible. With the fifth dead the Big Boo of star 2 appears
//!   (`bhvMerryGoRoundBigBoo`, `play_puzzle_jingle`) and the star spawns at
//!   (-1600, -2100, 205) when it is dead (`big_boo_spawn_merry_go_round_star`).
//!   Those fights are `Boo::HitFromBehind` and `BigBoo::Hit`.
//! - The manager, the Boos and the staircase to the basement exist in acts 2
//!   to 6 only (levels/bbh/script.c); act 1 is the ghost hunt, so this is
//!   offered in acts 2 and up, when star 2 is missing (the game loads at most
//!   the first missing act, so after star 1).
//!
//! The way in from the courtyard is the mesh elevator, then the basement's
//! doors: `mansion::walk_frames` (the elevator's ride, the walk over the
//! floors between the rooms) is the ETA to the pit's middle.

use obrrt_core::{Eta, GaBudget, Mem, Objective, St};

use super::mansion;

/// `SURFACE_MGR_MUSIC` (surface_terrains.h).
const SURFACE_MGR_MUSIC: i16 = 0x1A;
/// BBH is course 5; star 2 ("Ride Big Boo's Merry-Go-Round") is index 1.
const COURSE: i16 = 5;
const STAR: u8 = 1;
/// The pit's middle (the merry-go-round turns above it).
const PIT: [f32; 3] = [-205.0, -2764.0, 205.0];

pub struct Enter {
    zone: (u16, u8),
}

impl Enter {
    pub fn new(b: &St) -> Option<Enter> {
        (mansion::in_bbh(b) && b.act >= 2 && !b.has_star(COURSE, STAR) && !Self::on(b)).then(|| Enter { zone: b.zone() })
    }

    fn on(s: &St) -> bool {
        s.floor_type == SURFACE_MGR_MUSIC
    }
}

impl Objective for Enter {
    fn kind(&self) -> &'static str {
        "MerryGoRound::Enter"
    }
    fn key(&self) -> String {
        format!("MerryGoRoundEnter {:?}", self.zone)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        8.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && s.act >= 2 && !s.has_star(COURSE, STAR) && !Self::on(s)
    }
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 16, generations: 5, horizon: 240 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        if Self::on(s) {
            return Eta::Done;
        }
        Eta::Frames(mansion::walk_frames(s, PIT))
    }
    fn success_value(&self) -> f64 {
        5.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        Some(mansion::waypoint(s, PIT, 600.0))
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
    fn kinship(&self) -> u32 {
        10
    }
}
