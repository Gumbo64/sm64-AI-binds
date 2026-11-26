//! Rides and level mechanics: moving and rotating platforms, elevators, seesaws, the Rainbow Ride carpets, Koopa shells (ride), Hoot, water level diamonds (WDW), the moat and DDD/JRB water levels, chests in order (JRB treasure chests), manta and water rings, the Chain Chomp post, falling bridges, rising platforms that need to be stood on.
//!
//! Generated objectives for this category; see `extra/mod.rs`.
//!
//! Here:
//! - `Platform::Board` / `Platform::Ride` / `Seesaw::Board` ([`platform`]):
//!   get onto a moving platform, and stay on it while it carries Mario some
//!   distance (RR carpets, CCM ski lift, elevators, WDW arrow lifts and the
//!   express elevator, WF tower platforms, Dorrie, ...).
//! - `WdwWater::Lower` ([`water`]): drain Wet-Dry World's water with its diamonds (the
//!   express elevator's shaft is entered on foot, the downtown is flooded unless it is low).
//! - `Shell::Ride` ([`shell`]): jump on or walk into a Koopa shell.
//! - `Chest::Open` ([`chest`]): the 4-chest puzzles of JRB and DDD, in order.
//! - `WaterRing::SwimThrough` ([`ring`]): the manta ray's and DDD's jet
//!   stream rings.
//!
//! Left to `switches.rs`, which already has them (so not duplicated here):
//! the Chain Chomp's post (`bhvWoodenPost` next to `bhvChainChomp`), the
//! castle's moat pillars (`bhvWaterLevelPillar`), WDW's water level diamonds
//! (`bhvWaterLevelDiamond`), SSL's pyramid pillars
//! (`bhvPyramidPillarTouchDetector`) and THI's tiny island top.
//!
//! Not done: Hoot (NPC), THI size pipes (warps), the JRB ship rising and the
//! DDD sub leaving (decided by the act / save flags when the level loads, not
//! by anything Mario does there), the BBH merry-go-round (it only spins in
//! place; its Boos are enemies), the beta trampoline (not placed in US).
//!
//! NEEDS: the castle clock's minute hand roll (`oFaceAngleRoll` of
//! `bhvClockMinuteHand`; clock_arm.inc.c picks gTTCSpeedSetting from it when
//! Mario touches the TTC painting) or `gTTCSpeedSetting` itself, for a
//! "TtcClock::EnterAt(speed)" objective.
//! NEEDS: `gMarioState->marioObj->platform` (or the floor's object): "Mario is
//! on this platform" is inferred from his position inside its collision box
//! and a grounded action, which can be fooled by level floor at the same
//! height right next to it.
//! NEEDS: the ring spawner's `oWaterRingSpawnerRingsCollected` (and the
//! ring's `oWaterRingIndex`): a ring swum through out of order resets the
//! count, which `WaterRing::SwimThrough` can't see.

#![allow(unused_imports)]

mod chest;
mod common;
mod platform;
mod ring;
pub(crate) mod rr;
mod shell;
pub mod water;

use std::sync::Arc;

use obrrt_core::{Obj, Objective, St};

/// Objectives `o` offers, seen from base state `b` (called for every
/// object within range of `b`; `b` is never a dialog or menu state).
pub fn generate(b: &St, o: &Obj, out: &mut Vec<Arc<dyn Objective>>) {
    platform::generate(b, o, out);
    shell::generate(b, o, out);
    chest::generate(b, o, out);
    ring::generate(b, o, out);
}

/// Objectives that come from the level rather than one object (called once
/// per `b`).
pub fn generate_level(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    water::Lower::generate(b, out);
    water::Raise::generate(b, out);
    water::Slab::generate(b, out);
    water::Quick::generate(b, out);
    water::RedBreak::generate(b, out);
    water::RedTake::generate(b, out);
    rr::generate_level(b, out);
    rr::generate_route(b, out);
}
