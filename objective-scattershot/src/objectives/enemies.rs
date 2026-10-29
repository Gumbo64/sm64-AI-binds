//! Enemies, ported from objective-rrt/objectives (see PORTING.md).
//!
//! Sources: src/extra/enemies.rs, src/extra/enemies/*.rs (except clip.rs).
//!
//! # Model
//! Every enemy goal in rrt is "get near one enemy and do something that
//! changes its action / health / whether it is held or gone". In scattershot
//! that is one generic kind, [`EnemyHit`], that keys on the nearest live
//! enemy of the given behaviour(s) (or the one Mario holds): how far it is,
//! its action (as a class), health, whether Mario holds it, and optionally
//! where Mario is relative to it (approach sector, height, orbit angle) and
//! where it is. The rrt kinds are presets of it: the same kind names (rrt
//! name without `::`) fill in the enemy's behaviour, its "defeated" actions
//! and the quantities it needs; any param can still be overridden.
//!
//! Gate (rrt `base_criteria`): silent when no enemy matching `bhv`/`interact`
//! (and `param2`, `live_actions`) is loaded, when it is in a `dead_actions`
//! action, when `level` is wrong, when `star` is already collected, when
//! `capless` is not as required, or when it is farther than `max_dist`.
//! Defeated and missing enemies therefore are one (inactive) state, apart
//! from the alive ones.
//!
//! # rrt kind -> scattershot kind
//! - `Bully::PushOff` -> `BullyPushOff` (EnemyHit: `interact` = INTERACT_BULLY,
//!   not the big bullies, dead actions 100/101 = lava / death plane; `home_step`
//!   keys how far it was knocked from home and how far it fell).
//! - `Chuckya::Throw` -> `ChuckyaThrow` (bhvChuckya; held; `approach_sectors`
//!   so "from behind" is told from "from the front"; dead = gone).
//! - `Bobomb::Throw` -> `BobombThrow` (bhvBobomb; held; dead action 3 = explode).
//! - `Boo::HitFromBehind` -> `BooHitFromBehind` (bhvBoo, bhvGhostHuntBoo,
//!   bhvMerryGoRoundBoo, bhvBooWithCage; approach sector of Mario relative to
//!   the Boo's back; dead actions 3/4).
//! - `MrI::Circle` -> `MrICircle`, `MrBlizzard::Circle` -> `MrBlizzardCircle`
//!   (`orbit_sectors`: Mario's angle round it, so running circles makes new
//!   cells; `eyaw_sectors`: its facing, which follows Mario; the dizziness /
//!   cumulative turn itself is not in the state, so it is approximated by
//!   those two; dead actions 3 and 6). Mr. Blizzard's "cap worn" weight: `key_cap`.
//! - `Klepto::Hit` -> `KleptoHit` (bhvKlepto; struck action 6 is dead; the
//!   star Klepto = 2nd param byte nonzero, the cap Klepto = byte 0 and actions
//!   0/1; `KleptoHit` covers both: it needs the star not collected (SSL 8/0)
//!   only for the star one, via `star` + `param2_nonzero`).
//!   Preset: level SSL gate is not forced (Klepto only exists there).
//! - `Ukiki::ToCage` -> `UkikiToCage`, `Ukiki::RetrieveCap` -> `UkikiRetrieveCap`
//!   (TTM; held; the cap one needs Mario capless and keys `cap`, so getting the
//!   cap back is a new cell). The dialog in between is `Busy`'s business.
//! - `Whomp::GroundPound` -> `WhompGroundPound` (bhvSmallWhomp; action class
//!   3/4/5/6 = falling .. lying; `dy_step` keys standing on its back; dead 8).
//! - `PiranhaPlant::Sneak` -> `PiranhaPlantSneak` (bhvPiranhaPlant; dead 5-7;
//!   `speed_step` keys Mario's speed, since it wakes at speed > 10 within 400),
//!   `FirePiranha::Hit` -> `FirePiranhaHit` (bhvFirePiranhaPlant with 2nd
//!   param byte nonzero; health keyed, dead when health <= 0).
//! - `Moneybag::Reveal` -> `MoneybagReveal` (bhvMoneybagHidden, dead action 1),
//!   `Moneybag::Stomp` -> `MoneybagStomp` (bhvMoneybag, dead 4/5; `dy_step`).
//! - The mapping of `enemies.rs` itself (`generate`, `generate_level`) is the
//!   object dispatch: a schema lists the kinds wanted instead.
//! - `Bobomb::GateClip` and the other `clip.rs` kinds: not ported here (another agent's).
//! - Not ported (no objective in rrt either): Amps, Bubba, Thwomps, Grindels,
//!   Heave-Hos, Spinies, Bullet Bills, Monty Moles ... (see rrt's module doc).
//!   `EnemyHit` can still be pointed at any of them by `bhv`.
//! - Per-instance weights (`obj_weight`, `success_value`), budgets, steering,
//!   kinship and star-count bonuses: search machinery, not ported.
//!
//! # Example schema fragment
//! ```json
//! {"kind": "BooHitFromBehind", "name": "BBHBoos", "params": {"max_dist": 1500}}
//! {"kind": "EnemyHit", "params": {"bhv": ["bhvKoopa"], "dead_actions": [3], "approach_sectors": 8}}
//! ```

use anyhow::Result;
use obrrt_core::{Obj, St, bhv_addr};
use serde::Deserialize;
use serde_json::Value;

use super::util::{ActionGroup, action_class, bucket, dist, pack_pos, yaw_sector};
use crate::objective::{Cx, Objective};

/// One behaviour name or a list of them.
#[derive(Deserialize, Clone, Default)]
#[serde(untagged)]
pub enum Names {
    #[default]
    None,
    One(String),
    Many(Vec<String>),
}

impl Names {
    fn addrs(&self) -> Vec<u32> {
        match self {
            Names::None => vec![],
            Names::One(s) => bhv_addr(s).into_iter().collect(),
            Names::Many(v) => v.iter().filter_map(|s| bhv_addr(s)).collect(),
        }
    }
    fn is_empty(&self) -> bool {
        matches!(self, Names::None) || matches!(self, Names::Many(v) if v.is_empty())
    }
}

/// Key the state of the nearest live enemy of a behaviour (see the module doc).
///
/// Fields: `dist` (straight-line, buckets of `step`, capped at `max_dist`),
/// `held` (Mario holds it), `eaction` (its action: 1 + index in `actions`,
/// else 0; or the raw action when `actions` is empty and `raw_action`),
/// `health` (only when `key_health`), `approach` (Mario's sector round the
/// enemy relative to its facing, 0 = in front, `approach_sectors`, only
/// within `approach_range`, else -1), `dy` (Mario's height above it per
/// `dy_step`, clamped to -3..3), `orbit` (Mario's absolute azimuth round it,
/// `orbit_sectors`), `eyaw` (its facing, `eyaw_sectors`), `away`/`drop`
/// (distance from home / fall below home, `home_step`), `speed` (Mario's
/// forward speed per `speed_step`), `cap` (Mario has no cap, when
/// `key_cap`), `maction` (Mario's action class, `mario_actions`), `epos`
/// (enemy position on the `grid_log2` grid, when `enemy_pos`). Disabled
/// fields are written as 0.
///
/// ```json
/// {"kind": "EnemyHit", "params": {"bhv": "bhvSmallWhomp", "dead_actions": [8], "actions": [3,4,5,6], "dy_step": 60}}
/// ```
#[derive(Deserialize, Clone)]
pub struct EnemyHit {
    #[serde(default)]
    pub bhv: Names,
    /// Match enemies by `interact_type & interact` != 0 (as bullies have no name).
    pub interact: Option<u32>,
    /// Behaviours that never count (the big bullies).
    #[serde(default)]
    pub exclude_bhv: Names,
    /// The 2nd behaviour parameter byte must be this ...
    pub param2: Option<u32>,
    /// ... or, with this, nonzero (true) / zero (false).
    pub param2_nonzero: Option<bool>,
    /// Only live in these actions (empty: any).
    #[serde(default)]
    pub live_actions: Vec<i32>,
    /// Defeated in these actions: the objective is silent.
    #[serde(default)]
    pub dead_actions: Vec<i32>,
    /// Defeated at or below this health (the fire piranhas).
    pub dead_health: Option<i32>,
    /// Only in this level.
    pub level: Option<u16>,
    /// Silent once this star (course, index) is collected.
    pub star: Option<(i16, u8)>,
    /// Some(true): only while Mario has no cap; Some(false): only with it.
    pub capless: Option<bool>,
    /// Silent when Mario is farther than this from the enemy.
    pub max_dist: Option<f32>,
    #[serde(default = "step")]
    pub step: f32,
    #[serde(default)]
    pub actions: Vec<i32>,
    #[serde(default)]
    pub raw_action: bool,
    #[serde(default)]
    pub key_health: bool,
    #[serde(default)]
    pub approach_sectors: u32,
    #[serde(default = "approach_range")]
    pub approach_range: f32,
    #[serde(default)]
    pub dy_step: f32,
    #[serde(default)]
    pub orbit_sectors: u32,
    #[serde(default)]
    pub eyaw_sectors: u32,
    #[serde(default)]
    pub home_step: f32,
    #[serde(default)]
    pub speed_step: f32,
    #[serde(default)]
    pub key_cap: bool,
    #[serde(default)]
    pub mario_actions: Vec<ActionGroup>,
    #[serde(default)]
    pub enemy_pos: bool,
    #[serde(default = "eight")]
    pub grid_log2: u32,
}

fn step() -> f32 {
    200.0
}
fn approach_range() -> f32 {
    600.0
}
fn eight() -> u32 {
    8
}

const MARIO_CAP_ON_HEAD: u32 = 0x10;
const MARIO_CAP_IN_HAND: u32 = 0x20;

impl Default for EnemyHit {
    fn default() -> Self {
        serde_json::from_value(Value::Object(Default::default())).unwrap()
    }
}

impl EnemyHit {
    fn matches(&self, o: &Obj, bhvs: &[u32], excl: &[u32]) -> bool {
        let by_bhv = !self.bhv.is_empty();
        let ok = (by_bhv && bhvs.contains(&o.bhv)) || self.interact.is_some_and(|m| o.interact_type & m != 0);
        let b2 = (o.beh_params >> 16) & 0xFF;
        ok && !excl.contains(&o.bhv)
            && self.param2.is_none_or(|p| p == b2)
            && self.param2_nonzero.is_none_or(|nz| (b2 != 0) == nz)
            && (self.live_actions.is_empty() || self.live_actions.contains(&o.action))
            && !self.dead_actions.contains(&o.action)
            && self.dead_health.is_none_or(|h| o.health > h)
    }
}

impl Objective for EnemyHit {
    fn fields(&self) -> &[&'static str] {
        &[
            "dist", "held", "eaction", "health", "approach", "dy", "orbit", "eyaw", "away", "drop", "speed", "cap", "maction",
            "epos",
        ]
    }

    fn pos_grid(&self) -> Option<u32> {
        self.enemy_pos.then_some(self.grid_log2)
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if self.level.is_some_and(|l| l != st.level) {
            return;
        }
        if self.star.is_some_and(|(c, i)| st.has_star(c, i)) {
            return;
        }
        let capless = st.mario_flags & (MARIO_CAP_ON_HEAD | MARIO_CAP_IN_HAND) == 0;
        if self.capless.is_some_and(|c| c != capless) {
            return;
        }
        let (bhvs, excl) = (self.bhv.addrs(), self.exclude_bhv.addrs());
        let objs = st.objects();
        let live = || objs.iter().filter(|o| self.matches(o, &bhvs, &excl));
        let held = live().find(|o| Some(o.slot) == st.held_slot);
        let Some(e) = held.or_else(|| live().min_by(|a, b| dist(a.pos, st.pos).total_cmp(&dist(b.pos, st.pos)))) else {
            return;
        };
        let d = dist(e.pos, st.pos);
        if self.max_dist.is_some_and(|m| d > m) {
            return;
        }
        let hd = ((e.pos[0] - st.pos[0]).powi(2) + (e.pos[2] - st.pos[2]).powi(2)).sqrt();
        // Mario's azimuth from the enemy, an s16-style angle (0 = +z).
        let azimuth = (st.pos[0] - e.pos[0]).atan2(st.pos[2] - e.pos[2]) / std::f32::consts::TAU * 65536.0;
        let azimuth = azimuth.round() as i32;
        cx.put(0, bucket(d, self.step));
        cx.put(1, held.is_some() as i32);
        cx.put(
            2,
            if !self.actions.is_empty() {
                self.actions.iter().position(|&a| a == e.action).map_or(0, |i| i as i32 + 1)
            } else if self.raw_action {
                e.action
            } else {
                0
            },
        );
        cx.put(3, if self.key_health { e.health } else { 0 });
        let approach = if self.approach_sectors > 0 {
            if d <= self.approach_range {
                let n = self.approach_sectors;
                // Relative to the facing, offset half a sector so "in front" is a sector's middle.
                let rel = (azimuth - e.yaw).rem_euclid(65536) as u32 + 65536 / (2 * n);
                yaw_sector((rel & 0xFFFF) as u16 as i16, n)
            } else {
                -1
            }
        } else {
            0
        };
        cx.put(4, approach);
        cx.put(5, if self.dy_step > 0.0 { bucket(st.pos[1] - e.pos[1], self.dy_step).clamp(-3, 3) } else { 0 });
        cx.put(6, if self.orbit_sectors > 0 && hd <= self.approach_range.max(1000.0) { yaw_sector(azimuth as i16, self.orbit_sectors) } else { 0 });
        cx.put(7, if self.eyaw_sectors > 0 { yaw_sector(e.yaw as i16, self.eyaw_sectors) } else { 0 });
        let (away, drop) = if self.home_step > 0.0 {
            let h = ((e.pos[0] - e.home[0]).powi(2) + (e.pos[2] - e.home[2]).powi(2)).sqrt();
            (bucket(h, self.home_step), bucket((e.home[1] - e.pos[1]).max(0.0), self.home_step))
        } else {
            (0, 0)
        };
        cx.put(8, away);
        cx.put(9, drop);
        cx.put(10, if self.speed_step > 0.0 { bucket(st.forward_vel.abs(), self.speed_step) } else { 0 });
        cx.put(11, (self.key_cap && capless) as i32);
        cx.put(12, if self.mario_actions.is_empty() { 0 } else { action_class(st.action, &self.mario_actions) });
        cx.put(13, if self.enemy_pos { pack_pos(e.pos, self.grid_log2) } else { 0 });
        cx.all_children();
    }
}

/// Merges `defaults` (the kind's preset) under the schema's `params`.
fn preset(defaults: Value, params: &Value) -> Result<Box<dyn Objective>> {
    let mut m = defaults.as_object().cloned().unwrap_or_default();
    if let Some(p) = params.as_object() {
        m.extend(p.clone());
    }
    Ok(Box::new(serde_json::from_value::<EnemyHit>(Value::Object(m))?))
}

/// The kinds of this family: `Some(objective)` for a kind name it knows, else `None`.
pub fn build(kind: &str, params: &Value) -> Option<Result<Box<dyn Objective>>> {
    use serde_json::json;
    let d = match kind {
        "EnemyHit" => json!({}),
        // bully.rs: INTERACT_BULLY = 131072 (layout.rs); big bullies are bosses.
        "BullyPushOff" => json!({"interact": 131072, "exclude_bhv": ["bhvBigBully", "bhvBigBullyWithMinions", "bhvBigChillBully"],
            "dead_actions": [100, 101], "home_step": 250.0, "max_dist": 2500.0}),
        "ChuckyaThrow" => json!({"bhv": "bhvChuckya", "approach_sectors": 8, "approach_range": 500.0}),
        "BobombThrow" => json!({"bhv": "bhvBobomb", "dead_actions": [3]}),
        "BooHitFromBehind" => json!({"bhv": ["bhvBoo", "bhvGhostHuntBoo", "bhvMerryGoRoundBoo", "bhvBooWithCage"],
            "dead_actions": [3, 4], "approach_sectors": 8, "approach_range": 500.0}),
        "MrICircle" => json!({"bhv": "bhvMrI", "dead_actions": [3], "orbit_sectors": 8, "eyaw_sectors": 8, "max_dist": 800.0}),
        "MrBlizzardCircle" => json!({"bhv": "bhvMrBlizzard", "dead_actions": [6], "orbit_sectors": 8, "eyaw_sectors": 8,
            "max_dist": 1000.0, "key_cap": true}),
        // klepto.rs: struck = action 6; star one has 2nd param byte != 0 (SSL star 8/0), cap one waits for 0/1.
        "KleptoHit" => json!({"bhv": "bhvKlepto", "dead_actions": [6], "star": [8, 0], "max_dist": 2500.0, "raw_action": true}),
        // ukiki.rs: cage Ukiki param2 0 (TTM star 12/1, dead at go-to-cage 4), cap Ukiki param2 1.
        "UkikiToCage" => json!({"bhv": ["bhvUkiki", "bhvMacroUkiki"], "param2": 0, "dead_actions": [4], "star": [12, 1]}),
        "UkikiRetrieveCap" => json!({"bhv": ["bhvUkiki", "bhvMacroUkiki"], "param2": 1, "capless": true, "key_cap": true}),
        "WhompGroundPound" => json!({"bhv": "bhvSmallWhomp", "dead_actions": [8], "actions": [3, 4, 5, 6], "dy_step": 60.0,
            "approach_sectors": 8, "max_dist": 1500.0}),
        "PiranhaPlantSneak" => json!({"bhv": "bhvPiranhaPlant", "dead_actions": [5, 6, 7], "speed_step": 5.0, "max_dist": 1500.0}),
        "FirePiranhaHit" => json!({"bhv": "bhvFirePiranhaPlant", "param2_nonzero": true, "dead_health": 0, "key_health": true}),
        "MoneybagReveal" => json!({"bhv": "bhvMoneybagHidden", "dead_actions": [1], "max_dist": 1500.0}),
        "MoneybagStomp" => json!({"bhv": "bhvMoneybag", "dead_actions": [4, 5], "dy_step": 80.0}),
        _ => return None,
    };
    Some(preset(d, params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{key_of, obj, st, with_objects};

    fn schema(kind: &str, params: &str) -> String {
        format!(r#"{{"kind":"Root","children":[{{"kind":"{kind}","params":{params}}}]}}"#)
    }
    fn key(kind: &str, params: &str, s: &St) -> Vec<(u8, i32)> {
        key_of(&schema(kind, params), 1, s)
    }
    fn at(x: f32) -> St {
        st([x, 0.0, 0.0], 0.0, 9, 1, 0)
    }

    #[test]
    fn silent_without_enemy_or_when_dead() {
        assert!(key("WhompGroundPound", "{}", &at(0.0)).is_empty());
        let w = |a| with_objects(at(0.0), vec![{ let mut o = obj("bhvSmallWhomp", 1, [300.0, 0.0, 0.0]); o.action = a; o }]);
        assert!(!key("WhompGroundPound", "{}", &w(0)).is_empty());
        assert!(key("WhompGroundPound", "{}", &w(8)).is_empty());
        // Wrong behaviour.
        let s = with_objects(at(0.0), vec![obj("bhvBobomb", 1, [300.0, 0.0, 0.0])]);
        assert!(key("WhompGroundPound", "{}", &s).is_empty());
        assert!(!key("BobombThrow", "{}", &s).is_empty());
    }

    #[test]
    fn distance_buckets_and_nearest() {
        let one = |x: f32| with_objects(at(0.0), vec![obj("bhvBobomb", 1, [x, 0.0, 0.0])]);
        let k = |x| key("BobombThrow", r#"{"step": 100}"#, &one(x));
        assert_eq!(k(99.0), k(1.0));
        assert_ne!(k(99.0), k(101.0));
        // The nearer of two bob-ombs decides.
        let s = with_objects(at(0.0), vec![obj("bhvBobomb", 1, [900.0, 0.0, 0.0]), obj("bhvBobomb", 2, [150.0, 0.0, 0.0])]);
        assert_eq!(key("BobombThrow", r#"{"step": 100}"#, &s), k(150.0));
    }

    #[test]
    fn held_is_a_different_cell() {
        let mut s = with_objects(at(0.0), vec![obj("bhvChuckya", 5, [100.0, 0.0, 0.0])]);
        let free = key("ChuckyaThrow", "{}", &s);
        s.held_slot = Some(5);
        s.holding = true;
        assert_ne!(free, key("ChuckyaThrow", "{}", &s));
    }

    #[test]
    fn behind_vs_front() {
        // The Boo faces +z (yaw 0). Mario behind it (z < 0) or in front (z > 0).
        let boo = obj("bhvBoo", 1, [0.0, 0.0, 0.0]);
        let m = |z: f32| with_objects(st([0.0, 0.0, z], 0.0, 9, 1, 0), vec![boo.clone()]);
        assert_ne!(key("BooHitFromBehind", "{}", &m(-200.0)), key("BooHitFromBehind", "{}", &m(200.0)));
        assert_eq!(key("BooHitFromBehind", "{}", &m(-200.0)), key("BooHitFromBehind", "{}", &m(-210.0)));
        // Dead (action 3) is silent.
        let mut dead = boo.clone();
        dead.action = 3;
        assert!(key("BooHitFromBehind", "{}", &with_objects(at(0.0), vec![dead])).is_empty());
    }

    #[test]
    fn bully_by_interact_and_home_displacement() {
        let mut b = obj("bhvSmallBully", 1, [0.0, 0.0, 0.0]);
        b.interact_type = 131072;
        let s = |x: f32, y: f32| {
            let mut o = b.clone();
            o.home = [0.0, 0.0, 0.0];
            o.pos = [x, y, 0.0];
            with_objects(at(-100.0), vec![o])
        };
        assert!(!key("BullyPushOff", "{}", &s(0.0, 0.0)).is_empty());
        assert_ne!(key("BullyPushOff", "{}", &s(0.0, 0.0)), key("BullyPushOff", "{}", &s(600.0, 0.0)));
        assert_ne!(key("BullyPushOff", "{}", &s(0.0, 0.0)), key("BullyPushOff", "{}", &s(0.0, -600.0)));
        let mut lava = s(0.0, 0.0);
        let mut o = lava.objects()[0].clone();
        o.action = 100;
        lava = with_objects(lava, vec![o]);
        assert!(key("BullyPushOff", "{}", &lava).is_empty());
        // A non-bully is ignored.
        let other = with_objects(at(0.0), vec![obj("bhvBobomb", 1, [0.0; 3])]);
        assert!(key("BullyPushOff", "{}", &other).is_empty());
    }

    #[test]
    fn gates_star_level_cap_distance() {
        let o = with_objects(at(0.0), vec![obj("bhvKlepto", 1, [500.0, 0.0, 0.0])]);
        assert!(!key("KleptoHit", "{}", &o).is_empty());
        let mut got = o.clone();
        got.course_stars[7] = 1; // course 8, index 0
        assert!(key("KleptoHit", "{}", &got).is_empty());
        assert!(!key("KleptoHit", r#"{"star": null}"#, &got).is_empty());
        assert!(key("KleptoHit", r#"{"max_dist": 100}"#, &o).is_empty());
        assert!(key("KleptoHit", r#"{"level": 4}"#, &o).is_empty());
        assert!(!key("KleptoHit", r#"{"level": 9}"#, &o).is_empty());
        // Ukiki cap: only while capless, and the cap is keyed.
        let u = with_objects(at(0.0), vec![{ let mut u = obj("bhvUkiki", 1, [300.0, 0.0, 0.0]); u.beh_params = 1 << 16; u }]);
        assert!(!key("UkikiRetrieveCap", "{}", &u).is_empty()); // st() has no cap flags: capless
        let mut capped = u.clone();
        capped.mario_flags = MARIO_CAP_ON_HEAD;
        assert!(key("UkikiRetrieveCap", "{}", &capped).is_empty()); // cap back: done
    }

    #[test]
    fn param2_selects_ukiki_and_alias_override() {
        let mk = |p2: u32| with_objects(at(0.0), vec![{ let mut u = obj("bhvUkiki", 1, [300.0, 0.0, 0.0]); u.beh_params = p2 << 16; u }]);
        assert!(!key("UkikiToCage", "{}", &mk(0)).is_empty());
        assert!(key("UkikiToCage", "{}", &mk(1)).is_empty());
        // Overriding a preset param.
        assert!(!key("UkikiToCage", r#"{"param2": 1}"#, &mk(1)).is_empty());
    }

    #[test]
    fn cap_state_changes_the_cell_for_mr_blizzard() {
        let o = obj("bhvMrBlizzard", 1, [300.0, 0.0, 0.0]);
        let mut a = with_objects(at(0.0), vec![o]);
        let capless = key("MrBlizzardCircle", "{}", &a);
        a.mario_flags = MARIO_CAP_ON_HEAD;
        assert_ne!(capless, key("MrBlizzardCircle", "{}", &a));
    }

    #[test]
    fn circling_makes_new_cells() {
        let o = obj("bhvMrI", 1, [0.0, 0.0, 0.0]);
        let m = |x: f32, z: f32| with_objects(st([x, 0.0, z], 0.0, 9, 1, 0), vec![o.clone()]);
        let k = |x, z| key("MrICircle", "{}", &m(x, z));
        assert_ne!(k(0.0, 300.0), k(300.0, 0.0));
        assert_eq!(k(0.0, 300.0), k(10.0, 300.0));
    }

    #[test]
    fn health_and_actions() {
        let p = |h: i32| with_objects(at(0.0), vec![{ let mut o = obj("bhvFirePiranhaPlant", 1, [300.0, 0.0, 0.0]); o.beh_params = 1 << 16; o.health = h; o }]);
        assert_ne!(key("FirePiranhaHit", "{}", &p(2)), key("FirePiranhaHit", "{}", &p(1)));
        assert!(key("FirePiranhaHit", "{}", &p(0)).is_empty());
        let w = |a| with_objects(at(0.0), vec![{ let mut o = obj("bhvSmallWhomp", 1, [300.0, 0.0, 0.0]); o.action = a; o }]);
        assert_ne!(key("WhompGroundPound", "{}", &w(0)), key("WhompGroundPound", "{}", &w(6)));
        assert_eq!(key("WhompGroundPound", "{}", &w(0)), key("WhompGroundPound", "{}", &w(1)));
    }

    #[test]
    fn unknown_kind_is_none() {
        assert!(build("Nope", &Value::Null).is_none());
    }
}
