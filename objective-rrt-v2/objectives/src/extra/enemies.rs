//! Enemies other than the simple stomp ones already covered: bullies (push off edges), bob-ombs (throw), Chuckya, piranha plants, Boos (hit from behind), Mr. I, Klepto, Ukiki, Monty Moles, Fly Guys, Snufits, Scuttlebugs, Spindrifts, Mr. Blizzard, Bubba, Skeeters, Swoopers, Lakitu, Amps and the like: how each is beaten, and what hazards to avoid.
//!
//! Generated objectives for this category; see `extra/mod.rs`.
//!
//! `Enemy::Defeat` (parent module) already handles everything with
//! `INTERACT_BOUNCE_TOP` / `BOUNCE_TOP2` / `HIT_FROM_BELOW` / `KOOPA` that
//! dies by disappearing: Goombas, Scuttlebugs, Spindrifts, Fly Guys,
//! Snufits, Swoopers, Skeeters, Lakitu, Pokeys, small fire piranhas. These
//! add the enemies it can't beat or can't see beaten:
//!
//! - `Bully::PushOff`: small and chill bullies, knocked into lava / water.
//! - `Chuckya::Throw`, `Bobomb::Throw`: grab (Chuckya from behind), throw.
//! - `Bobomb::GateClip`: a bob-omb shoves Mario through the Chain Chomp's
//!   gate to BoB's star 6 (the speedrun clip; see `clip.rs`).
//! - `Boo::HitFromBehind`: small Boos (blue coins, ghost hunt and
//!   merry-go-round Big Boo progress) and the courtyard's caged Boo.
//! - `Boo::GhostHunt`: all five of BBH's ghost hunt Boos (star 1).
//! - `MerryGoRound::Enter`: into BBH's merry-go-round enclosure (star 2).
//! - `MrI::Circle`, `MrBlizzard::Circle`: run circles round them.
//! - `Klepto::Hit`: make the condor drop SSL's star or Mario's cap.
//! - `Ukiki::ToCage`, `Ukiki::RetrieveCap`: TTM's monkeys.
//! - `Whomp::GroundPound`: small Whomps, on their backs.
//! - `PiranhaPlant::Sneak`, `FirePiranha::Hit`: sleeping piranha plants
//!   (they respawn instead of disappearing) and THI's five giant ones.
//! - `Moneybag::Reveal`, `Moneybag::Stomp`.
//!
//! Not given objectives, as they can't be beaten (only avoided): Amps
//! (`INTERACT_SHOCK`), Bubba (`INTERACT_CLAM_OR_BUBBA`, eats Mario),
//! Thwomps, Grindels, Heave-Hos (throw Mario), Spinies (punch only knocks
//! them back), Bullet Bills, Mad Pianos, Sushi sharks, Snufit bullets and
//! Mr. Blizzard's snowballs. Monty Moles (`INTERACT_BOUNCE_TOP`, health -1:
//! hidden, not deleted, when hit; monty_mole.inc.c) can't be seen beaten
//! from a state alone and are worth little, so they're left out too.

use std::sync::Arc;

use obrrt_core::layout::INTERACT_BULLY;
use obrrt_core::{Obj, Objective, St};

mod boo;
mod bully;
mod circle;
pub mod clip;
mod common;
mod eel;
mod klepto;
mod merrygoround;
pub mod mansion;
mod moneybag;
mod piranha;
mod throw;
mod ukiki;
mod whomp;

/// Bullies that are bosses (handled by the `bosses` category).
const BIG_BULLIES: &[&str] = &["bhvBigBully", "bhvBigBullyWithMinions", "bhvBigChillBully"];

/// Objectives `o` offers, seen from base state `b` (called for every
/// object within range of `b`; `b` is never a dialog or menu state).
pub fn generate(b: &St, o: &Obj, out: &mut Vec<Arc<dyn Objective>>) {
    let name = o.bhv_name();
    // bhvSmallChillBully has no name in the layout table, so bullies are
    // found by interact type.
    if o.interact_type == INTERACT_BULLY && !BIG_BULLIES.contains(&name) {
        out.push(Arc::new(bully::PushOff::new(b, o)));
        return;
    }
    match name {
        "bhvChuckya" => out.push(Arc::new(throw::Throw::chuckya(b, o))),
        "bhvBobomb" => out.push(Arc::new(throw::Throw::bobomb(b, o))),
        "bhvBoo" | "bhvGhostHuntBoo" | "bhvMerryGoRoundBoo" | "bhvBooWithCage" => {
            if let Some(x) = boo::HitFromBehind::new(b, o) {
                out.push(Arc::new(x));
            }
        }
        "bhvMrI" => {
            let x = circle::Circle::mr_i(b, o);
            if x.base_criteria(b) {
                out.push(Arc::new(x));
            }
            // BBH's big one: its climb under the vanish cap as a stage of its own.
            for (gate, boxed) in [(false, true), (false, false), (true, false)] {
                if let Some(c) = circle::Climb::new(b, o, gate, boxed) {
                    out.push(Arc::new(c));
                }
            }
        }
        "bhvMrBlizzard" => out.push(Arc::new(circle::Circle::mr_blizzard(b, o))),
        "bhvKlepto" => {
            if let Some(x) = klepto::Hit::new(b, o) {
                out.push(Arc::new(x));
            }
        }
        "bhvUnagi" => {
            if let Some(x) = eel::TouchTail::new(b, o) {
                out.push(Arc::new(x));
            }
        }
        "bhvUkiki" | "bhvMacroUkiki" => {
            if let Some(x) = ukiki::Ukiki::new(b, o) {
                out.push(Arc::new(x));
            }
        }
        "bhvSmallWhomp" => out.push(Arc::new(whomp::GroundPound::new(b, o))),
        "bhvPiranhaPlant" | "bhvFirePiranhaPlant" => {
            if let Some(x) = piranha::Hit::new(b, o) {
                out.push(Arc::new(x));
            }
        }
        "bhvMoneybag" | "bhvMoneybagHidden" => {
            if let Some(x) = moneybag::Moneybag::new(b, o) {
                out.push(Arc::new(x));
            }
        }
        _ => {}
    }
}

/// Objectives that come from the level rather than one object (called once
/// per `b`).
pub fn generate_level(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    if let Some(x) = boo::GhostHunt::new(b) {
        out.push(Arc::new(x));
    }
    if let Some(x) = merrygoround::Enter::new(b) {
        out.push(Arc::new(x));
    }
    if let Some(x) = clip::GateClip::new(b).filter(|_| super::tech::enabled("gateclip")) {
        out.push(Arc::new(x));
    }
}
