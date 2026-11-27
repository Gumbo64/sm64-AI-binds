//! Objective-based RRT's domain knowledge: one file per objective type, and
//! [`Registry`], which generates them from a base state `b`.
//!
//! Depends only on `obrrt-core`. Adding an objective type = a new file here
//! plus a line in [`Registry::generate`].

mod boss;
mod breakable;
mod coin;
mod door;
mod enemy;
#[cfg(test)]
mod eta_tests;
pub mod extra;
mod grab;
mod menu;
mod npc;
pub mod mario;
pub mod moves;
mod painting;
mod painting_table;
pub mod setup;
mod star;
pub mod walk;

use std::sync::Arc;

use obrrt_core::layout::*;
use obrrt_core::{Generator, Objective, St, dist};
use rand::Rng;
use rand::rngs::SmallRng;

pub use painting::Painting;

/// Every castle painting (painting_table.rs), for labs.
pub fn paintings() -> &'static [Painting] {
    painting_table::PAINTINGS
}

/// Objects further than this from Mario in `b` don't generate objectives.
const OBJECT_RADIUS: f32 = 5000.0;

/// NPCs worth talking to, and how much: Bob-omb Buddies open cannons
/// (bhvBobombBuddyOpensCannon), Toads give stars.
fn npc_weight(b: &St, o: &obrrt_core::Obj) -> Option<f64> {
    match o.bhv_name() {
        // WDW's buddy opens the cannon into the downtown tunnel (extra/warps.rs wdw_town_missing): a goal while a
        // downtown star is missing and the cannon's lid is still shut.
        "bhvBobombBuddyOpensCannon" if extra::warps::wdw_town_missing(b) && b.objects().iter().any(|x| x.bhv_name() == "bhvCannonClosed") => {
            Some(30.0)
        }
        "bhvBobombBuddyOpensCannon" => Some(8.0),
        "bhvBobombBuddy" => Some(0.5),
        // Toads whose dialog (1st behaviour byte) is a star dialog give a
        // star (mario_misc.c: TOAD_STAR_1/2/3_DIALOG = 82, 76, 83).
        "bhvToadMessage" if matches!(o.beh_params >> 24, 82 | 76 | 83) => Some(15.0),
        "bhvToadMessage" => Some(1.0),
        // Koopa the Quick: bhvKoopa with 2nd behaviour byte 2 (BoB) or 3 (THI).
        "bhvKoopa" if matches!((o.beh_params >> 16) & 0xFF, 2 | 3) => Some(10.0),
        _ => None,
    }
}

/// Bosses beaten by grabbing and throwing (Boss::Hit, Object::Grab from
/// behind). The others have their own objectives in extra/bosses.rs: the Whomp
/// King is pounded, the big bullies pushed off (their oHealth is always 0).
const BOSSES: &[&str] = &["bhvKingBobomb"];

const NOT_DEFEATED_BY_VANISHING: &[&str] = &[
    "bhvBoo", "bhvGhostHuntBoo", "bhvMerryGoRoundBoo", "bhvBooWithCage", "bhvKlepto", "bhvMoneybag",
    "bhvMoneybagHidden", "bhvPiranhaPlant", "bhvMontyMole",
];

/// A point near b for PosDist and PosTime: 300-2500 away horizontally at a
/// random angle, -300..+900 in height.
fn near_point(b: &St, rng: &mut SmallRng) -> [f32; 3] {
    let r = rng.random_range(300.0..2500.0f32);
    let a = rng.random_range(0.0..std::f32::consts::TAU);
    let dy = rng.random_range(-300.0..900.0f32);
    [b.pos[0] + r * a.sin(), b.pos[1] + dy, b.pos[2] + r * a.cos()]
}

pub struct Registry;

impl Generator for Registry {
    /// PosTime near b reads the records.
    fn wants_records(&self) -> bool {
        true
    }
    /// OBRRT_BLJ_SCAN=1: the `blj_scan` block (bljany::scan's diagnostics).
    fn summary_blocks(&self) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = extra::tech::bljany::scan::summary_block().into_iter().collect();
        v.extend(extra::tech::bljany::spots::summary_block());
        v
    }
    /// OBRRT_BLJ_SPOTS: blj_spots.jsonl, blj_misses.jsonl, blj_zones.json.
    fn report(&self, out_dir: &str) {
        extra::tech::bljany::spots::write(out_dir);
    }
    fn generate(&self, b: &St, rng: &mut SmallRng) -> Vec<Arc<dyn Objective>> {
        let mut out: Vec<Arc<dyn Objective>> = Vec::new();
        // OBRRT_TWOSTAR: the coin count as work under way (St::wip).
        extra::missions::twostar::install();
        // In a dialog or menu, its options are the only objectives.
        if b.modal() {
            if b.in_dialog() {
                out.extend(menu::Dialog::options(b).into_iter().map(|o| Arc::new(o) as Arc<dyn Objective>));
            } else if b.in_save_menu() {
                out.extend(menu::SaveMenu::options().into_iter().map(|o| Arc::new(o) as Arc<dyn Objective>));
            } else if b.act_select {
                out.extend(menu::SelectAct::options(b).into_iter().map(|o| Arc::new(o) as Arc<dyn Objective>));
            }
            return out;
        }
        // Mario's self objectives, from every b.
        for _ in 0..2 {
            let p = near_point(b, rng);
            out.push(Arc::new(mario::PosDist::new(b, p)));
        }
        // OBRRT_SWIM_TARGETS=1: and one in the water column from water.
        if mario::swim_targets() {
            if let Some(p) = mario::underwater_point(b, rng) {
                out.push(Arc::new(mario::PosDist::new_swim(b, p)));
            }
        }
        // Faster routes to places already reached: near b, before the
        // tree's record there (candidates drawn like PosDist's, kept only
        // where a node of b's context has been), or to b itself (no records kept).
        if !obrrt_core::records::installed() {
            out.push(Arc::new(mario::PosTime::new(b)));
        } else {
            for _ in 0..mario::POSTIME_N {
                let p = near_point(b, rng);
                if let Some(o) = mario::PosTime::sample(b, p) {
                    out.push(Arc::new(o));
                }
            }
        }
        out.push(Arc::new(mario::Settle::new(b)));
        if rng.random_bool(0.3) {
            out.push(Arc::new(mario::Wait::new(b, [15, 30, 60, 90][rng.random_range(0..4)])));
        }
        if mario::GoToCastle::applies(b) {
            // OBRRT_ZONE_LOCK: a per-zone tree doesn't leave by the pause menu.
            if !obrrt_core::config::current().zone.lock {
                out.push(Arc::new(mario::GoToCastle::new(b)));
            }
        }
        // Objects near Mario, by what they let him do.
        for o in b.objects() {
            // OBRRT_GOAL_WIDE=1: an object further than OBJECT_RADIUS still
            // gives its goals (objectives worth at least OBRRT_GOAL_MIN, default
            // 20: bosses, the big bullies, Klepto with the star, warps to an
            // area with a star). In the zone-locked BoB tree gf2 (2026-11-09)
            // no state came within 5000 units of King Bob-omb in 60 M frames,
            // so Boss::Hit never entered the pool and nothing drew the tree up
            // the mountain; the audit (STAR120-AUDIT.md) lists eight more such
            // goals far from their course's entry.
            let wide = obrrt_core::config::current().goal_focus.wide;
            let goal_min = 20.0;
            let far = dist(o.pos, b.pos).min(dist(star::rest_pos(o), b.pos)) > OBJECT_RADIUS;
            if far && !wide {
                continue;
            }
            let mark = out.len();
            let near_len = |out: &mut Vec<Arc<dyn Objective>>| {
                if far {
                    let mut i = mark;
                    while i < out.len() {
                        if out[i].obj_weight(b) >= goal_min { i += 1 } else { out.swap_remove(i); }
                    }
                }
            };
            match o.interact_type {
                // Hidden blue coins are coins even while invisible and
                // intangible; they're collectable only once out (action 2),
                // which HiddenBlueCoin::Collect handles.
                INTERACT_COIN if o.bhv_name() == "bhvHiddenBlueCoin" => {}
                INTERACT_COIN => out.push(Arc::new(coin::Collect::new(b, o))),
                INTERACT_STAR_OR_KEY => out.push(Arc::new(star::Collect::new(b, o))),
                // A spawned star in its spawn cutscene has no star hitbox yet
                // (bhvSpawnedStar sets it in action 3); target it from the
                // spawn on (star.rs).
                _ if star::spawned_star(o) => out.push(Arc::new(star::Collect::new(b, o))),
                INTERACT_WARP_DOOR | INTERACT_DOOR => out.push(Arc::new(door::Enter::new(b, o))),
                INTERACT_GRABBABLE => {
                    let boss = BOSSES.contains(&o.bhv_name());
                    // A boss whose star is already collected needn't be beaten.
                    if boss && boss::boss_star(o.bhv_name()).is_some_and(|(c, i)| b.has_star(c, i)) {
                        continue;
                    }
                    out.push(Arc::new(grab::Grab::new(b, o, if boss { 20.0 } else { 1.0 }, boss)));
                }
                _ if npc_weight(b, o).is_some() => out.push(Arc::new(npc::Talk::new(b, o, npc_weight(b, o).unwrap()))),
                // Enemies that never disappear when beaten (they respawn,
                // hide, bounce Mario off or flee) can't be judged by
                // Enemy::Defeat; extra/enemies.rs has their own objectives.
                INTERACT_BOUNCE_TOP | INTERACT_BOUNCE_TOP2 | INTERACT_HIT_FROM_BELOW | INTERACT_KOOPA
                    if NOT_DEFEATED_BY_VANISHING.contains(&o.bhv_name()) => {}
                INTERACT_BOUNCE_TOP | INTERACT_BOUNCE_TOP2 | INTERACT_HIT_FROM_BELOW | INTERACT_KOOPA => {
                    out.push(Arc::new(enemy::Defeat::new(b, o)))
                }
                INTERACT_BREAKABLE => out.push(Arc::new(breakable::Break::new(b, o, 1.0))),
                _ => {}
            }
            if o.bhv_name() == "bhvExclamationBox" {
                out.push(Arc::new(breakable::Break::new(b, o, breakable::exclamation_weight(o))));
            }
            extra::generate(b, o, &mut out);
            if BOSSES.contains(&o.bhv_name()) && !boss::boss_star(o.bhv_name()).is_some_and(|(c, i)| b.has_star(c, i)) {
                out.push(Arc::new(boss::Hit::new(b, o)));
            }
            near_len(&mut out);
        }
        extra::generate_level(b, &mut out);
        if b.level as u32 == LEVEL_CASTLE {
            for p in painting_table::PAINTINGS.iter().filter(|p| p.area == b.area) {
                out.push(Arc::new(painting::Enter::new(b, p)));
            }
        }
        out
    }
}

/// `OBRRT_EXIT_SKIP=<levels>` (search/src/tree.rs): courses whose entry is not
/// a good exit of the tree. Their paintings are not attempted and their other
/// ways in weigh next to nothing (extra/warps.rs `dest_worth`). Empty when
/// unset.
pub fn exit_skip() -> &'static [u32] {
    &obrrt_core::config::current().stop.exit_skip
}
