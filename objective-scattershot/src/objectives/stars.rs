//! Stars and collectibles, ported from objective-rrt/objectives (see PORTING.md).
//!
//! Sources: `src/star.rs`, `src/extra/collectibles/*.rs`.
//!
//! All kinds here are "go and touch the nearest X" objectives. Each one is
//! gated like the rrt `base_criteria` (silent when no matching, still useful
//! object is loaded), and writes Mario's coarse position (`pos`, `grid_log2`),
//! the straight-line distance to the nearest candidate in buckets of `step`
//! (`dist`), and the few quantities its success reads (see each kind). Which
//! object is "nearest" is by straight-line distance, per behaviour list, so a
//! schema needs no per-instance objectives. All kinds `needs_objects`.
//!
//! Common params: `bhv` (a behaviour name or list; each kind has a default),
//! `step` (distance bucket, default 400 = rrt's `arrival_scale`),
//! `grid_log2` (default 8).
//!
//! | rrt kind | scattershot kind | extra fields / params |
//! |---|---|---|
//! | `Star::Collect` | `StarCollect` | `spawning` (spawn cutscene, the star's action < 3); `block_chomp_gate` |
//! | `Cap::Collect` | `CapCollect` | `cap` = wing/metal/vanish/normal (required) |
//! | `HiddenBlueCoin::Collect` | `HiddenBlueCoinCollect` | `left` (coins out, up to `max_left`) |
//! | `BlueCoinSwitch::Press` (and `BlueCoinSwitch::Pound`, switches.rs, same object) | `BlueCoinSwitchPress` | `pound`: 0 none, 1 ground-pounding, 2 pound landing; times 2 plus 1 when within 90 horizontally |
//! | `RedCoin::Next` | `RedCoinNext` | `red` (red coins, 0..8) |
//! | `CoinFormation::Sweep` | `CoinFormationSweep` | `state` (formation action), `left` (its coins out) |
//! | `Heart::Spin` | `HeartSpin` | `wedges`, `speed` (forward speed bucket of `speed_step`, only within 600 of it, else -1) |
//! | `OneUp::Collect` | `OneUpCollect` | `lives` |
//! | `OneUpTrigger::Touch` | `OneUpTriggerTouch` | `left` (triggers left) |
//! | `SecretTrigger::Touch` | `SecretTriggerTouch` | `left`; gated on the secret star missing, and BoB's needs the wing cap |
//! | `CrazyBox::Bounce` | `CrazyBoxBounce` | `held` (Mario holds it), `air` |
//!
//! Not ported: `Coin::Collect` (coin.rs, not in this family's sources; use
//! `HiddenBlueCoinCollect`/`CoinFormationSweep` style kinds or `Lens`).
//! rrt's `common.rs` is only helpers. Not ported at all: the GP-catch
//! (`OBRRT_GP_CATCH`) Z hint and landing-frame ranking (steering and success
//! ranking, not a state distinction; use `Lens` with action classes near a star
//! if the airborne approach matters), `star_blocked`'s `gateclip` softening
//! (`StarCollect` blocks the BoB-6 star only while `bhvChainChompGate` is
//! loaded; set `block_chomp_gate` false to keep it open), budgets, weights,
//! kinship, steering, `success_*` (search machinery).
//!
//! ```json
//! {"kind": "StarCollect", "params": {"step": 400}}
//! {"kind": "CapCollect", "params": {"cap": "wing"}}
//! ```

use anyhow::{Result, bail};
use obrrt_core::{Obj, St, bhv_addr};
use serde::Deserialize;
use serde_json::Value;

use super::util::{bucket, dist, pack_pos};
use crate::objective::{Cx, Objective};

const ACT_GROUND_POUND: u32 = 0x0080_08A9;
const ACT_GROUND_POUND_LAND: u32 = 0x0080_023C;
const ACT_FLAG_AIR: u32 = 2048;
const MARIO_NORMAL_CAP: u32 = 0x01;
const MARIO_VANISH_CAP: u32 = 0x02;
const MARIO_METAL_CAP: u32 = 0x04;
const MARIO_WING_CAP: u32 = 0x08;
const LEVEL_BOB: u16 = 9;
const LEVEL_SSL: u16 = 8;
const LEVEL_WDW: u16 = 11;
const LEVEL_THI: u16 = 13;
/// star.rs / coins.rs.
const FORMATION_RADIUS: f32 = 450.0;
const FORMATION_HEIGHT: f32 = 800.0;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Star,
    Cap,
    HiddenBlue,
    BlueSwitch,
    RedNext,
    Sweep,
    Heart,
    OneUp,
    OneUpTrigger,
    SecretTrigger,
    CrazyBox,
}

#[derive(Deserialize, Default)]
#[serde(untagged)]
enum OneOrMany {
    #[default]
    None,
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
pub struct Params {
    #[serde(default)]
    bhv: OneOrMany,
    #[serde(default = "step")]
    step: f32,
    #[serde(default = "eight")]
    grid_log2: u32,
    /// CapCollect: wing, metal, vanish or normal.
    #[serde(default)]
    cap: Option<String>,
    /// StarCollect: BoB star 6 is closed while the Chain Chomp gate stands.
    #[serde(default = "yes")]
    block_chomp_gate: bool,
    /// HeartSpin: forward speed per bucket.
    #[serde(default = "speed_step")]
    speed_step: f32,
    /// Counts (`left`) are capped here.
    #[serde(default = "eight_i")]
    max_left: i32,
}

fn step() -> f32 {
    400.0
}
fn eight() -> u32 {
    8
}
fn eight_i() -> i32 {
    8
}
fn yes() -> bool {
    true
}
fn speed_step() -> f32 {
    16.0
}

pub struct Collect {
    kind: Kind,
    bhvs: Vec<u32>,
    p: Params,
    cap_flag: u32,
    fields: &'static [&'static str],
}

fn objs(st: &St) -> &[Obj] {
    st.objects.as_deref().map(|v| v.as_slice()).unwrap_or(&[])
}

fn spawned_star(o: &Obj) -> bool {
    matches!(o.bhv_name(), "bhvSpawnedStar" | "bhvSpawnedStarNoLevelExit" | "bhvStarSpawnCoordinates")
}

/// star.rs `rest_pos`: where a star is, or comes to rest.
fn rest_pos(o: &Obj) -> [f32; 3] {
    if spawned_star(o) { o.home } else { o.pos }
}

/// star.rs `star_of`: (course, index) from the first behaviour byte.
fn star_of(st: &St, o: &Obj) -> Option<(i16, u8)> {
    let index = (o.beh_params >> 24) as u8;
    ((1..=24).contains(&st.course) && index <= 6 && !o.bhv_name().contains("Key")).then_some((st.course, index))
}

/// The star a course's bhvHiddenStar gives (triggers.rs `secret_star`).
fn secret_star(level: u16) -> Option<u8> {
    match level {
        LEVEL_BOB => Some(4),
        LEVEL_WDW => Some(2),
        LEVEL_SSL => Some(5),
        LEVEL_THI => Some(3),
        _ => None,
    }
}

impl Collect {
    fn new(kind: Kind, defaults: &[&str], fields: &'static [&'static str], params: &Value) -> Result<Collect> {
        let p: Params = serde_json::from_value(params.clone())?;
        let names: Vec<String> = match &p.bhv {
            OneOrMany::None => defaults.iter().map(|s| s.to_string()).collect(),
            OneOrMany::One(s) => vec![s.clone()],
            OneOrMany::Many(v) => v.clone(),
        };
        let mut bhvs = Vec::new();
        for n in &names {
            match bhv_addr(n) {
                Some(a) => bhvs.push(a),
                None => bail!("unknown behaviour {n}"),
            }
        }
        let cap_flag = match (kind, p.cap.as_deref()) {
            (Kind::Cap, Some("wing")) => MARIO_WING_CAP,
            (Kind::Cap, Some("metal")) => MARIO_METAL_CAP,
            (Kind::Cap, Some("vanish")) => MARIO_VANISH_CAP,
            (Kind::Cap, Some("normal")) => MARIO_NORMAL_CAP,
            (Kind::Cap, c) => bail!("CapCollect needs cap = wing/metal/vanish/normal, got {c:?}"),
            _ => 0,
        };
        Ok(Collect { kind, bhvs, p, cap_flag, fields })
    }

    /// Whether this loaded object is one the kind would still go for.
    fn wanted(&self, st: &St, o: &Obj) -> bool {
        match self.kind {
            Kind::Star => star_of(st, o).is_none_or(|(c, i)| {
                let gate = self.p.block_chomp_gate && c == 1 && i == 5 && objs(st).iter().any(|g| g.bhv_name() == "bhvChainChompGate");
                !st.has_star(c, i) && !gate
            }),
            Kind::HiddenBlue => o.action == 2,
            Kind::BlueSwitch => o.action == 0,
            Kind::OneUp => !matches!(o.bhv_name(), "bhvHidden1up" | "bhvHidden1upInPole") || o.action != 0,
            Kind::CrazyBox => (o.held_state == 0 || st.held_slot == Some(o.slot)) && (!st.holding || st.held_slot == Some(o.slot)),
            _ => true,
        }
    }

    /// Kind-wide gate on Mario's state and the level (not per object).
    fn open(&self, st: &St) -> bool {
        match self.kind {
            Kind::Cap => st.mario_flags & self.cap_flag == 0,
            Kind::Heart => (1..8).contains(&(st.health as i32 >> 8)),
            Kind::RedNext => {
                let star = objs(st).iter().find_map(|o| match o.bhv_name() {
                    "bhvHiddenRedCoinStar" => Some((o.beh_params >> 24) as u8),
                    "bhvBowserCourseRedCoinStar" => Some(0),
                    _ => None,
                });
                let star = star.filter(|&i| i <= 6 && (1..=24).contains(&st.course)).map(|i| (st.course, i));
                star.is_none_or(|(c, i)| !st.has_star(c, i))
            }
            Kind::SecretTrigger => {
                let index = objs(st)
                    .iter()
                    .find(|o| o.bhv_name() == "bhvHiddenStar")
                    .map(|o| (o.beh_params >> 24) as u8)
                    .or_else(|| secret_star(st.level));
                let owned = index.is_some_and(|i| (1..=24).contains(&st.course) && st.has_star(st.course, i));
                let needs_wings = st.level == LEVEL_BOB && st.mario_flags & MARIO_WING_CAP == 0;
                !owned && !needs_wings
            }
            _ => true,
        }
    }

    fn candidates<'a>(&self, st: &'a St) -> Vec<&'a Obj> {
        objs(st).iter().filter(|o| self.bhvs.contains(&o.bhv) && self.wanted(st, o)).collect()
    }
}

fn nearest<'a>(st: &St, c: &[&'a Obj], at: impl Fn(&Obj) -> [f32; 3]) -> Option<&'a Obj> {
    c.iter().copied().min_by(|a, b| dist(at(a), st.pos).total_cmp(&dist(at(b), st.pos)))
}

impl Objective for Collect {
    fn fields(&self) -> &[&'static str] {
        self.fields
    }

    fn pos_grid(&self) -> Option<u32> {
        Some(self.p.grid_log2)
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if !self.open(st) {
            return;
        }
        let cands = self.candidates(st);
        let at = |o: &Obj| if self.kind == Kind::Star { rest_pos(o) } else { o.pos };
        let Some(t) = nearest(st, &cands, at) else { return };
        let left = |n: usize| (n as i32).min(self.p.max_left);
        let mut d = dist(at(t), st.pos);
        // Extra field values, in order after pos and dist.
        let extra: Vec<i32> = match self.kind {
            Kind::Star => vec![(spawned_star(t) && t.action < 3) as i32],
            Kind::OneUp => vec![st.lives as i32],
            Kind::HiddenBlue | Kind::OneUpTrigger | Kind::SecretTrigger => vec![left(cands.len())],
            Kind::BlueSwitch => {
                let pound = match st.action {
                    ACT_GROUND_POUND => 1,
                    ACT_GROUND_POUND_LAND => 2,
                    _ => 0,
                };
                let hd = ((t.pos[0] - st.pos[0]).powi(2) + (t.pos[2] - st.pos[2]).powi(2)).sqrt();
                vec![pound * 2 + (hd < 90.0) as i32]
            }
            Kind::RedNext => vec![(st.red_coins as i32).clamp(0, 8)],
            Kind::Sweep => {
                let c = t.pos;
                let coins: Vec<&Obj> = objs(st)
                    .iter()
                    .filter(|o| {
                        o.bhv_name() == "bhvYellowCoin"
                            && ((o.pos[0] - c[0]).powi(2) + (o.pos[2] - c[2]).powi(2)).sqrt() < FORMATION_RADIUS
                            && (o.pos[1] - c[1]).abs() < FORMATION_HEIGHT
                    })
                    .collect();
                // Spent formations (spawned, nothing left) are done.
                if t.action != 0 && coins.is_empty() {
                    return;
                }
                if let Some(n) = nearest(st, &coins, |o| o.pos) {
                    d = dist(n.pos, st.pos);
                }
                vec![t.action, left(coins.len())]
            }
            Kind::Heart => {
                let near = dist(t.pos, st.pos) < 600.0;
                vec![st.health as i32 >> 8, if near { bucket(st.forward_vel, self.p.speed_step) } else { -1 }]
            }
            Kind::CrazyBox => vec![(st.held_slot == Some(t.slot)) as i32, (st.action & ACT_FLAG_AIR != 0) as i32],
            Kind::Cap => vec![],
        };
        cx.put(0, pack_pos(st.pos, self.p.grid_log2));
        cx.put(1, bucket(d, self.p.step));
        for (i, v) in extra.into_iter().enumerate() {
            cx.put(2 + i, v);
        }
        cx.all_children();
    }
}

/// The kinds of this family: `Some(objective)` for a kind name it knows, else `None`.
pub fn build(kind: &str, params: &Value) -> Option<Result<Box<dyn Objective>>> {
    const PD: &[&str] = &["pos", "dist"];
    let (k, bhv, f): (Kind, &[&str], &'static [&'static str]) = match kind {
        "StarCollect" => (
            Kind::Star,
            &["bhvStar", "bhvSpawnedStar", "bhvSpawnedStarNoLevelExit", "bhvStarSpawnCoordinates", "bhvBowserKey", "bhvGrandStar"],
            &["pos", "dist", "spawning"],
        ),
        "CapCollect" => (Kind::Cap, &["bhvWingCap", "bhvMetalCap", "bhvVanishCap", "bhvNormalCap"], PD),
        "HiddenBlueCoinCollect" => (Kind::HiddenBlue, &["bhvHiddenBlueCoin"], &["pos", "dist", "left"]),
        "BlueCoinSwitchPress" => (Kind::BlueSwitch, &["bhvBlueCoinSwitch"], &["pos", "dist", "pound"]),
        "RedCoinNext" => (Kind::RedNext, &["bhvRedCoin"], &["pos", "dist", "red"]),
        "CoinFormationSweep" => (Kind::Sweep, &["bhvCoinFormation"], &["pos", "dist", "state", "left"]),
        "HeartSpin" => (Kind::Heart, &["bhvRecoveryHeart"], &["pos", "dist", "wedges", "speed"]),
        "OneUpCollect" => (
            Kind::OneUp,
            &["bhv1Up", "bhv1upWalking", "bhv1upRunningAway", "bhv1upSliding", "bhv1upJumpOnApproach", "bhvHidden1up", "bhvHidden1upInPole"],
            &["pos", "dist", "lives"],
        ),
        "OneUpTriggerTouch" => (Kind::OneUpTrigger, &["bhvHidden1upTrigger", "bhvHidden1upInPoleTrigger"], &["pos", "dist", "left"]),
        "SecretTriggerTouch" => (Kind::SecretTrigger, &["bhvHiddenStarTrigger"], &["pos", "dist", "left"]),
        "CrazyBoxBounce" => (Kind::CrazyBox, &["bhvJumpingBox"], &["pos", "dist", "held", "air"]),
        _ => return None,
    };
    Some(Collect::new(k, bhv, f, params).map(|c| Box::new(c) as Box<dyn Objective>))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{key_of, obj, st, with_objects};

    fn schema(kind: &str, params: &str) -> String {
        format!(r#"{{"kind":"Root","children":[{{"kind":"{kind}","params":{params}}}]}}"#)
    }
    fn at(x: f32) -> St {
        st([x, 0.0, 0.0], 0.0, 9, 1, 0)
    }
    fn key(kind: &str, params: &str, s: &St) -> Vec<(u8, i32)> {
        key_of(&schema(kind, params), 1, s)
    }

    #[test]
    fn star_gate_and_buckets() {
        let star = obj("bhvStar", 3, [1000.0, 0.0, 0.0]);
        let none = at(0.0);
        assert!(key("StarCollect", "{}", &none).is_empty());
        let a = with_objects(at(0.0), vec![star.clone()]);
        let b = with_objects(at(-100.0), vec![star.clone()]);
        // Same 400-wide distance bucket (1000 and 1100 away) but different positions on the 256 grid.
        let (ka, kb) = (key("StarCollect", "{}", &a), key("StarCollect", "{}", &b));
        assert_eq!(ka[1], (1, 2));
        assert_eq!(kb[1], (1, 2));
        assert_ne!(ka[0], kb[0]);
        let c = with_objects(at(700.0), vec![star.clone()]);
        assert_eq!(key("StarCollect", "{}", &c)[1], (1, 0));
        // Owned: silent. (course 9 = BoB, index 0 in beh_params.)
        let mut d = with_objects(at(0.0), vec![star]);
        d.course = 1;
        d.course_stars[0] = 1;
        assert!(key("StarCollect", "{}", &d).is_empty());
    }

    #[test]
    fn spawned_star_uses_home_and_spawning() {
        let mut s = obj("bhvSpawnedStar", 1, [0.0, 5000.0, 0.0]);
        s.home = [800.0, 0.0, 0.0];
        s.action = 1;
        let k = key("StarCollect", "{}", &with_objects(at(0.0), vec![s.clone()]));
        assert_eq!(k[1], (1, 2)); // 800 away, not 5000
        assert_eq!(k[2], (2, 1));
        s.action = 3;
        let k = key("StarCollect", "{}", &with_objects(at(0.0), vec![s]));
        assert_eq!(k[2], (2, 0));
    }

    #[test]
    fn nearest_object_is_chosen() {
        let far = obj("bhvStar", 1, [3000.0, 0.0, 0.0]);
        let near = obj("bhvStar", 2, [500.0, 0.0, 0.0]);
        let k = key("StarCollect", "{}", &with_objects(at(0.0), vec![far, near]));
        assert_eq!(k[1], (1, 1));
    }

    #[test]
    fn chomp_gate_blocks_bob6() {
        let mut star = obj("bhvStar", 1, [500.0, 0.0, 0.0]);
        star.beh_params = 5 << 24;
        let gate = obj("bhvChainChompGate", 2, [0.0, 0.0, 0.0]);
        let mut s = with_objects(at(0.0), vec![star, gate]);
        s.course = 1;
        assert!(key("StarCollect", "{}", &s).is_empty());
        assert!(!key("StarCollect", r#"{"block_chomp_gate":false}"#, &s).is_empty());
    }

    #[test]
    fn caps_gated_by_flag() {
        let cap = obj("bhvWingCap", 1, [300.0, 0.0, 0.0]);
        let mut s = with_objects(at(0.0), vec![cap]);
        let p = r#"{"cap":"wing"}"#;
        assert!(!key("CapCollect", p, &s).is_empty());
        s.mario_flags = MARIO_WING_CAP;
        assert!(key("CapCollect", p, &s).is_empty());
        assert!(Tree_err(&schema("CapCollect", "{}")));
    }

    #[allow(non_snake_case)]
    fn Tree_err(json: &str) -> bool {
        crate::objective::Tree::load(json).is_err()
    }

    #[test]
    fn blue_coins_and_switch() {
        let mut coin = obj("bhvHiddenBlueCoin", 1, [300.0, 0.0, 0.0]);
        let s = with_objects(at(0.0), vec![coin.clone()]);
        assert!(key("HiddenBlueCoinCollect", "{}", &s).is_empty()); // still hidden
        coin.action = 2;
        let two = obj("bhvHiddenBlueCoin", 2, [900.0, 0.0, 0.0]);
        let mut two = two;
        two.action = 2;
        let k1 = key("HiddenBlueCoinCollect", "{}", &with_objects(at(0.0), vec![coin.clone()]));
        let k2 = key("HiddenBlueCoinCollect", "{}", &with_objects(at(0.0), vec![coin, two]));
        assert_eq!(k1[2], (2, 1));
        assert_eq!(k2[2], (2, 2));

        let sw = obj("bhvBlueCoinSwitch", 1, [50.0, 0.0, 0.0]);
        let mut s = with_objects(at(0.0), vec![sw.clone()]);
        assert_eq!(key("BlueCoinSwitchPress", "{}", &s)[2], (2, 1)); // over it, not pounding
        s.action = ACT_GROUND_POUND;
        assert_eq!(key("BlueCoinSwitchPress", "{}", &s)[2], (2, 3));
        let mut pressed = sw;
        pressed.action = 1;
        assert!(key("BlueCoinSwitchPress", "{}", &with_objects(at(0.0), vec![pressed])).is_empty());
    }

    #[test]
    fn red_coins_count_and_star_owned() {
        let coin = obj("bhvRedCoin", 1, [300.0, 0.0, 0.0]);
        let mut s = with_objects(at(0.0), vec![coin.clone()]);
        s.red_coins = 3;
        let a = key("RedCoinNext", "{}", &s);
        s.red_coins = 4;
        assert_ne!(a, key("RedCoinNext", "{}", &s));
        let mut hs = obj("bhvHiddenRedCoinStar", 2, [0.0, 0.0, 0.0]);
        hs.beh_params = 2 << 24;
        let mut s = with_objects(at(0.0), vec![coin, hs]);
        s.course = 4;
        assert!(!key("RedCoinNext", "{}", &s).is_empty());
        s.course_stars[3] = 1 << 2;
        assert!(key("RedCoinNext", "{}", &s).is_empty());
    }

    #[test]
    fn formation_sweep() {
        let f = obj("bhvCoinFormation", 1, [500.0, 0.0, 0.0]);
        let mut spent = f.clone();
        spent.action = 1;
        let c = obj("bhvYellowCoin", 2, [600.0, 0.0, 0.0]);
        // Not spawned yet: active, no coins out.
        let k = key("CoinFormationSweep", "{}", &with_objects(at(0.0), vec![f.clone()]));
        assert_eq!((k[2], k[3]), ((2, 0), (3, 0)));
        // Spawned with a coin out: aims at the coin.
        let k = key("CoinFormationSweep", "{}", &with_objects(at(0.0), vec![spent.clone(), c]));
        assert_eq!((k[1], k[2], k[3]), ((1, 1), (2, 1), (3, 1)));
        // Spawned and all taken: inactive.
        assert!(key("CoinFormationSweep", "{}", &with_objects(at(0.0), vec![spent])).is_empty());
    }

    #[test]
    fn heart_needs_hurt_mario() {
        let h = obj("bhvRecoveryHeart", 1, [300.0, 0.0, 0.0]);
        let mut s = with_objects(at(0.0), vec![h]);
        s.health = 0x880;
        assert!(key("HeartSpin", "{}", &s).is_empty());
        s.health = 0x400;
        s.forward_vel = 40.0;
        let k = key("HeartSpin", "{}", &s);
        assert_eq!((k[2], k[3]), ((2, 4), (3, 2)));
        s.pos[0] = -2000.0;
        assert_eq!(key("HeartSpin", "{}", &s)[3], (3, -1));
    }

    #[test]
    fn oneups_and_triggers() {
        let mut hid = obj("bhvHidden1up", 1, [300.0, 0.0, 0.0]);
        assert!(key("OneUpCollect", "{}", &with_objects(at(0.0), vec![hid.clone()])).is_empty());
        hid.action = 1;
        let mut s = with_objects(at(0.0), vec![hid]);
        s.lives = 4;
        let a = key("OneUpCollect", "{}", &s);
        s.lives = 5;
        assert_ne!(a, key("OneUpCollect", "{}", &s));

        let t = obj("bhvHidden1upTrigger", 1, [300.0, 0.0, 0.0]);
        let t2 = obj("bhvHidden1upTrigger", 2, [400.0, 0.0, 0.0]);
        let a = key("OneUpTriggerTouch", "{}", &with_objects(at(0.0), vec![t.clone(), t2]));
        let b = key("OneUpTriggerTouch", "{}", &with_objects(at(0.0), vec![t]));
        assert_ne!(a[2], b[2]);
    }

    #[test]
    fn secret_triggers_in_bob_need_wings() {
        let t = obj("bhvHiddenStarTrigger", 1, [300.0, 0.0, 0.0]);
        let mut s = with_objects(at(0.0), vec![t]);
        s.course = 1;
        assert!(key("SecretTriggerTouch", "{}", &s).is_empty()); // BoB, no wings
        s.mario_flags = MARIO_WING_CAP;
        assert!(!key("SecretTriggerTouch", "{}", &s).is_empty());
        s.course_stars[0] = 1 << 4; // the secret star owned
        assert!(key("SecretTriggerTouch", "{}", &s).is_empty());
    }

    #[test]
    fn crazy_box() {
        let b = obj("bhvJumpingBox", 7, [300.0, 0.0, 0.0]);
        let mut s = with_objects(at(0.0), vec![b.clone()]);
        assert_eq!(key("CrazyBoxBounce", "{}", &s)[2], (2, 0));
        s.holding = true; // holding something else
        assert!(key("CrazyBoxBounce", "{}", &s).is_empty());
        s.held_slot = Some(7);
        let mut held = b;
        held.held_state = 1;
        let s = with_objects(s, vec![held]);
        assert_eq!(key("CrazyBoxBounce", "{}", &s)[2], (2, 1));
    }
}
