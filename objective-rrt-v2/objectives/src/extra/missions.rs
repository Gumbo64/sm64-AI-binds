//! Per-level missions that aren't one object: for each course's 6 stars plus
//! the 100-coin star, what the star needs that the other categories miss,
//! plus the castle's secret stages.
//!
//! - `Mission::ReachStar` / `Mission::ReachBox`: get to a star (or the ! box
//!   holding it) whose spot is fixed in the level data ([`table::SPOTS`]),
//!   in the right area and act, from anywhere in the level. `Star::Collect`
//!   only exists within 5000 units of `b`; this one gives far stars a route.
//! - `Mission::ReachRedCoinStar`: to the red-coin marker once all 8 red coins
//!   are in (the star appears there); `Mission::ReachHiddenStar`: to the
//!   hidden star once no secret (`bhvHiddenStarTrigger`) is left.
//! - `RedCoins::CollectNext` / `Secrets::FindNext`: one more red coin /
//!   secret in the marker's area (off by default, see [`OFFER_COUNTERS`]).
//! - `Coins::Milestone` / `Coins::Reach100`: the course's coin count up to the
//!   next multiple of 10; at 100 the 100-coin star spawns next to Mario
//!   (src/game/interaction.c, `interact_coin`: main courses only, star 6).
//! - `Mission::PssUnder21`: the Princess's Secret Slide from above the timer
//!   start to the timer end in under 630 frames (`pss_end_slide`,
//!   src/game/interaction.c: star 1 spawns if the timer < 630).
//! - `Mission::ReachLoadedStar`: any star object loaded in `b` but too far for
//!   `Star::Collect` (event stars after a boss, race, NPC, ...), unless the
//!   table already has that star.
//!
//! NEEDS: the castle's secret stars (Toads, MIPS) aren't in `St::course_stars`
//! (they live in the save file's flags), so they can't be told apart as
//! collected; the NPC agents' objectives and `Star::Collect` cover them.

use std::sync::Arc;

use obrrt_core::layout::*;
use obrrt_core::{Eta, Mem, Obj, Objective, St, course_of_level, dist, frames_for};

mod table;
pub mod twostar;

/// What has to happen before a [`Spot`]'s star is there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpotKind {
    /// A placed `bhvStar`: always there.
    Star,
    /// An ! box holding the star: always there; hitting it is another
    /// objective's job once Mario is close.
    Box,
    /// Appears at the red-coin marker once `red_coins` reaches 8.
    RedCoinStar,
    /// Appears once every `bhvHiddenStarTrigger` in the area is touched.
    HiddenStar,
}

/// A star whose spot is fixed in the level data.
#[derive(Debug, Clone, Copy)]
pub struct Spot {
    pub course: i16,
    /// 0-5 (acts), as in the star's behaviour param byte 0.
    pub index: u8,
    pub level: u32,
    pub area: u8,
    /// Bit `n - 1` = act `n`.
    pub acts: u8,
    pub pos: [f32; 3],
    pub kind: SpotKind,
    pub name: &'static str,
}

impl Spot {
    fn zone(&self) -> (u16, u8) {
        (self.level as u16, self.area)
    }

    fn act_ok(&self, s: &St) -> bool {
        self.acts == 0x3f || (1..=6).contains(&s.act) && self.acts & (1 << (s.act - 1)) != 0
    }
}

fn count_bhv(s: &St, name: &str) -> usize {
    s.objects().iter().filter(|o| o.bhv_name() == name).count()
}

/// The nearest loaded object matching `f`.
fn nearest<'a>(s: &'a St, f: impl Fn(&Obj) -> bool) -> Option<&'a Obj> {
    s.objects().iter().filter(|o| f(o)).min_by(|a, b| s.dist(a.pos).total_cmp(&s.dist(b.pos)))
}

// ---------------------------------------------------------------------------
// Reaching a star's spot.

/// ETA to reach `pos` (within `r`) in `zone`, or to have `star`: v1's
/// success (the spot, or this star only, not any star collected on the
/// way) is `Done`; out of the zone without the star, `Lost`.
fn reach_eta(s: &St, zone: (u16, u8), pos: [f32; 3], r: f32, star: (i16, u8)) -> Eta {
    if s.has_star(star.0, star.1) {
        return Eta::Done;
    }
    if s.zone() != zone {
        return Eta::Lost;
    }
    let d = s.dist(pos);
    if d <= r { Eta::Done } else { Eta::Frames(frames_for(d)) }
}

/// How close counts as reaching a spot: a star's hitbox is 80 wide
/// (sCollectStarHitbox, spawn_star.inc.c); a box just needs Mario next to it.
fn reach_radius(kind: SpotKind) -> f32 {
    match kind {
        SpotKind::Box => 250.0,
        _ => 160.0,
    }
}

pub struct ReachSpot {
    spot: &'static Spot,
}

impl ReachSpot {
    /// Whether the spot's star is (or, for boxes, can be made) there in `s`.
    fn ready(&self, s: &St) -> bool {
        match self.spot.kind {
            SpotKind::Star | SpotKind::Box => true,
            SpotKind::RedCoinStar => s.red_coins >= 8,
            // Decidable only with objects: no secrets left in the area.
            SpotKind::HiddenStar => s.objects.is_some() && count_bhv(s, "bhvHiddenStarTrigger") == 0,
        }
    }
}

impl Objective for ReachSpot {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        match self.spot.kind {
            SpotKind::Star => "Mission::ReachStar",
            SpotKind::Box => "Mission::ReachBox",
            SpotKind::RedCoinStar => "Mission::ReachRedCoinStar",
            SpotKind::HiddenStar => "Mission::ReachHiddenStar",
        }
    }
    fn key(&self) -> String {
        format!("Mission {}-{} ({})", self.spot.course, self.spot.index, self.spot.name)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        match self.spot.kind {
            SpotKind::Star => 30.0,
            SpotKind::Box => 25.0,
            SpotKind::RedCoinStar | SpotKind::HiddenStar => 35.0,
        }
    }
    fn base_criteria(&self, s: &St) -> bool {
        let p = self.spot;
        s.zone() == p.zone()
            && p.act_ok(s)
            && !s.has_star(p.course, p.index)
            && s.dist(p.pos) > reach_radius(p.kind)
            && self.ready(s)
            && !crate::star::star_blocked(s, p.course, p.index)
    }
    /// Frames to the spot; `Done` at the spot or with this star had (v1's
    /// success); out of the spot's zone without it: `Lost`.
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 20, generations: 5, horizon: 240 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        let p = self.spot;
        reach_eta(s, p.zone(), p.pos, reach_radius(p.kind), (p.course, p.index))
    }
    fn success_value(&self) -> f64 {
        5.0
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.spot.pos)
    }
    fn kinship(&self) -> u32 {
        10
    }
    fn exists_in(&self, s: &St) -> bool {
        s.level as u32 == self.spot.level && !s.has_star(self.spot.course, self.spot.index)
    }
}

/// A star object loaded in `b` (an event star: boss, race, NPC, ...) too far
/// for `Star::Collect`. A spawned star flies to its home
/// (bhvStarSpawnCoordinates, `spawn_star` in spawn_star.inc.c), so aim there.
pub struct ReachLoadedStar {
    id: obrrt_core::ObjId,
    course: i16,
    index: u8,
    zone: (u16, u8),
    pos: [f32; 3],
}

/// `Star::Collect` is offered within this distance of `b`; beyond it, this.
const FAR: f32 = 4500.0;

impl ReachLoadedStar {
    fn here(&self, s: &St) -> bool {
        s.find(self.id).is_some()
    }
}

impl Objective for ReachLoadedStar {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Mission::ReachLoadedStar"
    }
    fn key(&self) -> String {
        format!("LoadedStar {}-{} {:?}", self.course, self.index, self.zone)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        30.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && !s.has_star(self.course, self.index) && self.here(s) && s.dist(self.pos) > 160.0
    }
    /// As `ReachSpot`: frames to the star's spot, `Done` within 160 or
    /// with this star had, `Lost` out of the zone.
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 20, generations: 5, horizon: 240 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        reach_eta(s, self.zone, self.pos, 160.0, (self.course, self.index))
    }
    fn success_value(&self) -> f64 {
        5.0
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.pos)
    }
    fn kinship(&self) -> u32 {
        10
    }
    fn exists_in(&self, s: &St) -> bool {
        // Without objects, it can't be told; keep it.
        s.zone() == self.zone && !s.has_star(self.course, self.index) && (s.objects.is_none() || self.here(s))
    }
}

// ---------------------------------------------------------------------------
// Counters: red coins, secrets, coins.

/// What a [`Counter`] counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Count {
    /// `gRedCoinsCollected`, kept by the area's red-coin star marker
    /// (bhv_hidden_red_coin_star_loop / bhv_bowser_course_red_coin_star_loop):
    /// meaningful only in the marker's area.
    RedCoins,
    /// `bhvHiddenStarTrigger` objects left in the area (fewer = found).
    Secrets,
}

/// A `Counter` frame without objects (its target unseen): about a coin
/// [`FAR`] away at [`obrrt_core::PACE`].
const UNSEEN_TARGET_FRAMES: f64 = FAR as f64 / obrrt_core::PACE;

/// One more red coin / secret in the zone of `spot`.
pub struct Counter {
    spot: &'static Spot,
    count: Count,
}

impl Counter {
    fn target_bhv(&self) -> &'static str {
        match self.count {
            Count::RedCoins => "bhvRedCoin",
            Count::Secrets => "bhvHiddenStarTrigger",
        }
    }

    fn target(&self, s: &St) -> Option<[f32; 3]> {
        let bhv = self.target_bhv();
        nearest(s, |o| o.bhv_name() == bhv).map(|o| o.pos)
    }

    fn left(&self, s: &St) -> usize {
        count_bhv(s, self.target_bhv())
    }
}

impl Objective for Counter {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        match self.count {
            Count::RedCoins => "RedCoins::CollectNext",
            Count::Secrets => "Secrets::FindNext",
        }
    }
    fn key(&self) -> String {
        format!("{} {}-{} {:?}", self.kind(), self.spot.course, self.spot.index, self.spot.zone())
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        6.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        let p = self.spot;
        if s.zone() != p.zone() || s.has_star(p.course, p.index) {
            return false;
        }
        match self.count {
            Count::RedCoins => s.red_coins < 8 && (s.objects.is_none() || self.left(s) > 0),
            Count::Secrets => s.objects.is_some() && self.left(s) > 0,
        }
    }
    /// Frames to the nearest red coin / secret; `Done` once the count moved
    /// past the start's (v1's success); out of the zone, or (secrets) a
    /// start without objects, where v1 could never succeed: `Lost`.
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 180 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.spot.zone() {
            return Eta::Lost;
        }
        match self.count {
            Count::RedCoins if s.red_coins > start.red_coins => return Eta::Done,
            Count::RedCoins => {}
            Count::Secrets => {
                if start.objects.is_none() {
                    return Eta::Lost;
                }
                if s.objects.is_some() && self.left(s) < self.left(start) {
                    return Eta::Done;
                }
            }
        }
        Eta::Frames(self.target(s).map_or(UNSEEN_TARGET_FRAMES, |t| frames_for(s.dist(t))))
    }
    fn success_value(&self) -> f64 {
        3.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.target(s)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.level as u32 == self.spot.level && !s.has_star(self.spot.course, self.spot.index)
    }
}

/// `Coins::Milestone` / `Coins::Reach100`: the coin count of a main course up
/// to `target` (a multiple of 10, at most 100) while its 100-coin star is
/// missing.
pub struct Coins {
    level: u16,
    course: i16,
    target: i16,
}

/// A start may be at most this many coins short of the target.
const COIN_SPAN: i16 = 15;

/// Frames per coin still to get after the nearest one: a course's coins
/// mostly come in lines, rings and enemy drops a few hundred units apart,
/// with longer walks between groups; ~1200 units (40 frames at PACE) on
/// average. It must exceed the walk to the next coin for the ETA to drop at
/// each coin (the nearest coin's distance jumps up when one is taken).
pub(crate) const FRAMES_PER_COIN: f64 = 40.0;

/// ETA for `coins` to reach `target`: the walk to the nearest coin (one
/// coin's [`FRAMES_PER_COIN`] when none is seen), then `FRAMES_PER_COIN`
/// for each coin after it; `Done` at the target.
pub(crate) fn coins_eta(coins: i16, target: i16, nearest: Option<f32>) -> Eta {
    let left = target - coins;
    if left <= 0 {
        return Eta::Done;
    }
    let first = nearest.map_or(FRAMES_PER_COIN, frames_for);
    Eta::Frames(first + (left - 1) as f64 * FRAMES_PER_COIN)
}

fn nearest_coin(s: &St) -> Option<[f32; 3]> {
    if twostar::on() {
        return twostar::coin_target(s, None);
    }
    nearest(s, |o| o.interact_type == INTERACT_COIN).map(|o| o.pos)
}

impl Objective for Coins {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        if self.target >= 100 { "Coins::Reach100" } else { "Coins::Milestone" }
    }
    fn key(&self) -> String {
        format!("Coins {} -> {}", self.course, self.target)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        if let Some(w) = twostar::milestone_weight(self.target) {
            return w;
        }
        if self.target >= 100 { 10.0 } else { 2.0 + self.target as f64 / 20.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.level == self.level && s.coins < self.target && s.coins >= self.target - COIN_SPAN && !s.has_star(self.course, 6)
    }
    /// [`coins_eta`] to `target`; `Done` at it in the level (v1's
    /// success); the level left (the count resets): `Lost`.
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 240 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.level != self.level {
            return Eta::Lost;
        }
        coins_eta(s.coins, self.target, nearest_coin(s).map(|c| s.dist(c)))
    }
    fn success_value(&self) -> f64 {
        if self.target >= 100 { 10.0 } else { 1.0 }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        nearest_coin(s)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.level == self.level && !s.has_star(self.course, 6)
    }
}

// ---------------------------------------------------------------------------
// The Princess's Secret Slide under 21 seconds.

/// SURFACE_TIMER_START / SURFACE_TIMER_END quads, from
/// levels/pss/areas/1/collision.inc.c: the start spans x 2673..2872,
/// z -6143..-5119 at y ~6075 (Mario enters at x 5632, y 6751, levels/pss/
/// script.c); the end x -6869..-5845, z 3047..3251 at y ~-4500.
const PSS_START_X: f32 = 2872.0;
const PSS_END: ([f32; 2], [f32; 2], f32) = ([-6869.0, -5845.0], [3047.0, 3251.0], -4300.0);
const PSS_END_CENTER: [f32; 3] = [-6357.0, -4500.0, 3149.0];
/// `pss_end_slide`: the star needs a timer under 630 frames; frames since a
/// start before the timer starts bound it from above.
const PSS_FRAMES: u32 = 620;

pub struct PssUnder21;

impl PssUnder21 {
    fn at_end(s: &St) -> bool {
        let (x, z, y) = PSS_END;
        s.level as u32 == LEVEL_PSS
            && (x[0] - 50.0..=x[1] + 50.0).contains(&s.pos[0])
            && (z[0] - 50.0..=z[1] + 50.0).contains(&s.pos[2])
            && s.pos[1] < y
    }
}

impl Objective for PssUnder21 {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Mission::PssUnder21"
    }
    fn key(&self) -> String {
        "PSS under 21 s".to_string()
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        25.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        // Before the timer start: on the entry ledge, east of the start quad.
        s.zone() == (LEVEL_PSS as u16, 1) && !s.has_star(19, 1) && s.pos[0] > PSS_START_X + 50.0 && s.pos[1] > 6000.0
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        // The slide itself takes most of the 21 s.
        Some(obrrt_core::GaBudget { population: 16, generations: 6, horizon: PSS_FRAMES as usize + 10 })
    }
    /// `Done` at the end quad within [`PSS_FRAMES`] of the start (v1's
    /// success; the deadline is from `start.frame`, so no `mem`); past the
    /// deadline or out of PSS: `Lost`; else the straight line to the end
    /// quad's centre in frames (the slide winds, so a lower bound).
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        let elapsed = s.frame.wrapping_sub(start.frame);
        if elapsed >= PSS_FRAMES || s.level as u32 != LEVEL_PSS {
            return Eta::Lost;
        }
        if Self::at_end(s) {
            return Eta::Done;
        }
        Eta::Frames(frames_for(dist(s.pos, PSS_END_CENTER)))
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // Down the slide: its course isn't a straight line, so only steer
        // until the start quad, then let the genome drive.
        (s.pos[0] > PSS_START_X).then_some([PSS_START_X - 200.0, s.pos[1], -5631.0])
    }
    fn exists_in(&self, s: &St) -> bool {
        s.level as u32 == LEVEL_PSS && !s.has_star(19, 1)
    }
}

// ---------------------------------------------------------------------------

/// Whether to offer `RedCoins::CollectNext` / `Secrets::FindNext`. Off:
/// `extra/collectibles.rs` offers `RedCoin::Next` and `SecretTrigger::Touch`
/// for the same counters, and two copies would double their weight. Turn on
/// if those go away.
const OFFER_COUNTERS: bool = false;

/// Objectives `o` offers, seen from base state `b` (called for every
/// object within range of `b`; `b` is never a dialog or menu state).
pub fn generate(_b: &St, _o: &Obj, _out: &mut Vec<Arc<dyn Objective>>) {}

/// Objectives that come from the level rather than one object (called once
/// per `b`).
pub fn generate_level(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    let level = b.level as u32;
    let course = course_of_level(level);
    if course == 0 {
        return;
    }
    for spot in table::SPOTS.iter().filter(|p| p.level == level && !b.has_star(p.course, p.index)) {
        if spot.act_ok(b) {
            out.push(Arc::new(ReachSpot { spot }));
        }
        let count = match spot.kind {
            SpotKind::RedCoinStar => Some(Count::RedCoins),
            SpotKind::HiddenStar => Some(Count::Secrets),
            _ => None,
        };
        if let Some(count) = count.filter(|_| OFFER_COUNTERS && b.zone() == spot.zone()) {
            out.push(Arc::new(Counter { spot, count }));
        }
    }
    // The 100-coin star (main courses only, interact_coin).
    if (1..=15).contains(&course) && !b.has_star(course, 6) && b.coins < 100 {
        let target = ((b.coins / 10 + 1) * 10).min(100);
        out.push(Arc::new(Coins { level: b.level, course, target }));
        if target < 100 && b.coins >= 100 - COIN_SPAN {
            out.push(Arc::new(Coins { level: b.level, course, target: 100 }));
        }
        // OBRRT_TWOSTAR: coins along the way, for two stars a visit.
        if let Some(r) = twostar::CoinRun::new(b) {
            out.push(Arc::new(r));
        }
    }
    if level == LEVEL_PSS && !b.has_star(19, 1) {
        out.push(Arc::new(PssUnder21));
    }
    // Loaded stars too far for Star::Collect and not in the table.
    for o in b.objects() {
        if o.interact_type != INTERACT_STAR_OR_KEY || o.bhv_name().contains("Key") {
            continue;
        }
        let index = (o.beh_params >> 24) as u8;
        if index > 6 || b.has_star(course, index) {
            continue;
        }
        let pos = if o.bhv_name() == "bhvStarSpawnCoordinates" { o.home } else { o.pos };
        if dist(b.pos, pos) <= FAR || table::SPOTS.iter().any(|p| p.course == course && p.index == index) {
            continue;
        }
        out.push(Arc::new(ReachLoadedStar { id: o.id, course, index, zone: b.zone(), pos }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::tech::testkit::{ACT_WALKING, st};
    use obrrt_core::fold;

    #[test]
    fn pss_under_21_deadline() {
        let at = |frame: u32, p: [f32; 3]| {
            let mut s = st(LEVEL_PSS, 1, p, ACT_WALKING);
            s.frame = frame;
            s
        };
        let end = [-6357.0, -4500.0, 3149.0];
        let start = at(1000, [5632.0, 6751.0, -5600.0]);
        let traj = vec![start.clone(), at(1300, [0.0, 1000.0, 0.0]), at(1610, end)];
        let (met, etas) = fold(&PssUnder21, &traj);
        assert_eq!(met, Some(2), "{etas:?}");
        assert!(etas[1].frames() < etas[0].frames());
        // At the end 620 frames after the start: too late.
        let (met, etas) = fold(&PssUnder21, &[start, at(1620, end)]);
        assert_eq!(met, None);
        assert!(etas[1] == Eta::Lost);
    }
}
