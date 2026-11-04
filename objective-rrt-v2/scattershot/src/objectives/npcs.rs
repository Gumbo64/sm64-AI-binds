//! NPCs and missions, ported from objective-rrt/objectives (see PORTING.md).
//!
//! Sources: src/extra/npcs.rs, src/extra/missions.rs, src/extra/missions/table.rs
//!
//! rrt makes one goal per NPC instance with a "done" condition (dialog started,
//! action left, held, Mario action reached ...). Here the states those
//! conditions read are key fields of the nearest (or held) object with the
//! given behaviour, so the search can tell "NPC in its next phase" or "Mario
//! holding it" or "Mario in its dialog" from "not yet". Gating (`base_criteria`)
//! is: the object is loaded, its star (if any) is missing, Mario has enough stars.
//! NPC phase (`oAction`) is not gated, it is a field, so the state after the
//! trigger stays a cell of its own.
//!
//! # Mapping of rrt kinds
//! | rrt kind | scattershot kind |
//! |---|---|
//! | `Sign::Read` | `SignRead` (bhvMessagePanel, bhvSignOnWall; Mario action reading-sign) |
//! | `Yoshi::Talk` | `YoshiTalk` |
//! | `Lakitu::Intro` | `LakituIntro` (intro Lakitu, byte1 = 1; distance to the bridge point) |
//! | `Mips::Catch` | `MipsCatch` (needs 15 stars) |
//! | `Ukiki::Catch`, `Ukiki::FollowToCage` | `UkikiCatch`, `UkikiFollowToCage` (cage Ukiki, star of bhvUkikiCage) |
//! | `Hoot::Wake`, `Hoot::Ride` | `HootWake`, `HootRide` |
//! | `KoopaRace::Claim` | `KoopaRaceClaim` |
//! | `PenguinRace::Talk`, `PenguinRace::Claim` | `PenguinRaceTalk`, `PenguinRaceClaim` |
//! | `Snowman::WakeBody`, `Snowman::Finish` | `SnowmanWakeBody`, `SnowmanFinish` |
//! | (generic `Approach` spec) | `NpcTalk` (any `bhv`, all params) |
//! | `KoopaRace::ReachFlag`, `PenguinRace::Win` | `KoopaRaceReachFlag`, `PenguinRaceWin` (struct `Race`) |
//! | `Tuxie::ReturnBaby` | `TuxieReturnBaby` |
//! | `Snowman::LeadBody` | `SnowmanLeadBody` |
//! | `Mission::ReachStar/ReachBox/ReachRedCoinStar/ReachHiddenStar` | `MissionReachStar`, `MissionReachBox`, `MissionReachRedCoinStar`, `MissionReachHiddenStar`, or `MissionReach` (all kinds, optional `course`/`index`) |
//! | `RedCoins::CollectNext`, `Secrets::FindNext` | `MissionCounter` (`MissionReach` plus red coin / secret counts and distance to the next one) |
//! | `Mission::ReachLoadedStar` | `MissionLoadedStar` |
//! | `Coins::Reach100`, `Coins::Milestone` | `MissionCoins` (exact coin count; milestones are just count values) |
//! | `Mission::PssUnder21` | `MissionPssUnder21` (the 21 s limit is not a field: the archive keeps the faster state of a cell) |
//!
//! Not ported: `star_blocked` of `Mission::Reach*` (needs rrt's star module; the
//! star simply stays a cell nobody reaches); steering (`steer_target`, buttons
//! B, `Aim`'s steering role) and budgets: only `Aim` is kept as the point the
//! distance is measured to. `Dialog::*` are other families.
//! Mission tables are data: [`SPOTS`] (copied from table.rs).

use anyhow::Result;
use obrrt_core::layout::*;
use obrrt_core::{Obj, St, bhv_addr, course_of_level};
use serde::Deserialize;
use serde_json::Value;

use super::util::{bucket, dist, pack_pos};
use crate::objective::{Cx, Objective};

/// include/sm64.h `ACT_RIDING_HOOT` (not in the layout).
const ACT_RIDING_HOOT: u32 = 0x0000_04A8;
const SNOWMAN_SITE: [f32; 3] = [-4230.0, -1344.0, 1813.0];
const LAKITU_BRIDGE: [f32; 3] = [0.0, 850.0, -1100.0];
const HOOT_REST: [f32; 3] = [-800.0, 150.0, -300.0];

fn byte0(o: &Obj) -> u8 {
    (o.beh_params >> 24) as u8
}
fn byte1(o: &Obj) -> u32 {
    (o.beh_params >> 16) & 0xFF
}
fn bk(d: f32, step: f32, max: i32) -> i32 {
    bucket(d, step).min(max)
}
fn talking(s: &St) -> bool {
    s.in_dialog() || matches!(s.action, ACT_WAITING_FOR_DIALOG | ACT_READING_NPC_DIALOG | ACT_READING_AUTOMATIC_DIALOG)
}
fn with_bhv<'a>(s: &'a St, name: &str) -> impl Iterator<Item = &'a Obj> {
    let b = bhv_addr(name);
    s.objects().iter().filter(move |o| Some(o.bhv) == b)
}
fn nearest<'a>(it: impl Iterator<Item = &'a Obj>, p: [f32; 3]) -> Option<&'a Obj> {
    it.min_by(|a, b| dist(a.pos, p).total_cmp(&dist(b.pos, p)))
}
fn star_missing(s: &St, course: i16, index: u8) -> bool {
    !s.has_star(course, index)
}

#[derive(Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum OneOrMany {
    One(String),
    Many(Vec<String>),
}
impl OneOrMany {
    fn list(&self) -> Vec<String> {
        match self {
            OneOrMany::One(s) => vec![s.clone()],
            OneOrMany::Many(v) => v.clone(),
        }
    }
}

/// Where the distance to the NPC is measured to (rrt `Aim`).
#[derive(Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Aim {
    /// The object itself.
    At,
    /// `aim_dist` in front of it, unless Mario already stands in its front cone.
    Front,
    /// The point `aim_dist` from it towards Mario.
    Near,
    /// `aim_point`.
    Point,
}

/// Parameters of an NPC kind; everything optional, a preset fills in the rest.
///
/// Example: `{"kind":"NpcTalk","params":{"bhv":"bhvToadMessage","step":300}}`
#[derive(Deserialize, Default, Clone)]
#[serde(default)]
pub struct NpcParams {
    /// Behaviour name(s) of the NPC.
    pub bhv: Option<OneOrMany>,
    /// Allowed second behaviour-parameter bytes (`oBehParams2ndByte`); empty = any.
    pub param2: Option<Vec<u32>>,
    /// A fixed star (course, index) that gates the objective once collected.
    pub star: Option<[i16; 2]>,
    /// Behaviour whose first parameter byte is the star index (this course);
    /// "self" = the NPC itself. Gate off while that object is not loaded.
    pub star_bhv: Option<String>,
    pub min_stars: Option<i16>,
    /// Distance bucket size (default 400) and the highest bucket.
    pub step: Option<f32>,
    pub max_bucket: Option<i32>,
    pub aim: Option<Aim>,
    pub aim_dist: Option<f32>,
    pub aim_point: Option<[f32; 3]>,
    /// Mario actions (full action values) that mark success (reading a sign, riding Hoot).
    pub mario_actions: Option<Vec<u32>>,
    /// Write `flags`: bit0 intangible, bit1/2 racing penguin won/cheated, bit3 at its rest spot.
    pub flags: Option<bool>,
    pub rest_offset: Option<[f32; 3]>,
    /// Write the NPC's own position on a grid (moving NPCs).
    pub track_pos: Option<bool>,
    pub grid_log2: Option<u32>,
}

impl NpcParams {
    fn or(mut self, d: NpcParams) -> NpcParams {
        macro_rules! f { ($($x:ident),*) => { $( if self.$x.is_none() { self.$x = d.$x; } )* } }
        f!(bhv, param2, star, star_bhv, min_stars, step, max_bucket, aim, aim_dist, aim_point, mario_actions, flags, rest_offset, track_pos, grid_log2);
        self
    }
}

/// Writes, for the nearest (or held) object of `bhv`: `dist` (bucket of the
/// straight-line distance to the aim point), `npc_action` (its oAction),
/// `held` (Mario holds it), `mario` (1 in a dialog, 2 in one of
/// `mario_actions`), `holding` (Mario holds anything), `flags`, `npc_pos`.
pub struct Npc {
    bhvs: Vec<String>,
    param2: Vec<u32>,
    star: Option<[i16; 2]>,
    star_bhv: Option<String>,
    min_stars: i16,
    step: f32,
    max_bucket: i32,
    aim: Aim,
    aim_dist: f32,
    aim_point: [f32; 3],
    mario_actions: Vec<u32>,
    flags: bool,
    rest_offset: Option<[f32; 3]>,
    track_pos: bool,
    grid_log2: u32,
}

impl Npc {
    fn new(p: NpcParams) -> Result<Npc> {
        let bhvs = p.bhv.ok_or_else(|| anyhow::anyhow!("NPC kind needs `bhv`"))?.list();
        Ok(Npc {
            bhvs,
            param2: p.param2.unwrap_or_default(),
            star: p.star,
            star_bhv: p.star_bhv,
            min_stars: p.min_stars.unwrap_or(0),
            step: p.step.unwrap_or(400.0),
            max_bucket: p.max_bucket.unwrap_or(12),
            aim: p.aim.unwrap_or(Aim::At),
            aim_dist: p.aim_dist.unwrap_or(220.0),
            aim_point: p.aim_point.unwrap_or([0.0; 3]),
            mario_actions: p.mario_actions.unwrap_or_default(),
            flags: p.flags.unwrap_or(false),
            rest_offset: p.rest_offset,
            track_pos: p.track_pos.unwrap_or(false),
            grid_log2: p.grid_log2.unwrap_or(8),
        })
    }

    fn target<'a>(&self, st: &'a St) -> Option<&'a Obj> {
        let cands = || {
            self.bhvs.iter().flat_map(|b| with_bhv(st, b)).filter(|o| self.param2.is_empty() || self.param2.contains(&byte1(o)))
        };
        cands().find(|o| Some(o.slot) == st.held_slot).or_else(|| nearest(cands(), st.pos))
    }

    /// `None`: gate off. `Some(None)`: no star gate. `Some(Some(..))`: the star.
    fn star_of(&self, st: &St, t: &Obj) -> Option<Option<(i16, u8)>> {
        if let Some([c, i]) = self.star {
            return Some(Some((c, i as u8)));
        }
        match self.star_bhv.as_deref() {
            None => Some(None),
            Some("self") => Some(Some((st.course, byte0(t)))),
            Some(name) => with_bhv(st, name).next().map(|o| Some((st.course, byte0(o)))),
        }
    }

    fn aim_at(&self, st: &St, o: &Obj) -> [f32; 3] {
        match self.aim {
            Aim::At => o.pos,
            Aim::Point => self.aim_point,
            Aim::Near => {
                let (dx, dz) = (st.pos[0] - o.pos[0], st.pos[2] - o.pos[2]);
                let l = (dx * dx + dz * dz).sqrt().max(1.0);
                [o.pos[0] + dx / l * self.aim_dist, o.pos[1], o.pos[2] + dz / l * self.aim_dist]
            }
            Aim::Front => {
                let a = o.yaw as f32 * std::f32::consts::TAU / 65536.0;
                let (fx, fz) = (a.sin(), a.cos());
                let (rx, rz) = (st.pos[0] - o.pos[0], st.pos[2] - o.pos[2]);
                let along = rx * fx + rz * fz;
                let side = (rx * fz - rz * fx).abs();
                if along > 60.0 && side < along * 0.4 {
                    o.pos
                } else {
                    [o.pos[0] + self.aim_dist * fx, o.pos[1], o.pos[2] + self.aim_dist * fz]
                }
            }
        }
    }
}

impl Objective for Npc {
    fn fields(&self) -> &[&'static str] {
        &["dist", "npc_action", "held", "mario", "holding", "flags", "npc_pos"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if st.stars < self.min_stars {
            return;
        }
        let Some(t) = self.target(st) else { return };
        let Some(star) = self.star_of(st, t) else { return };
        if star.is_some_and(|(c, i)| !star_missing(st, c, i)) {
            return;
        }
        cx.put(0, bk(dist(st.pos, self.aim_at(st, t)), self.step, self.max_bucket));
        cx.put(1, t.action);
        cx.put(2, (Some(t.slot) == st.held_slot) as i32);
        let mario = if talking(st) {
            1
        } else if self.mario_actions.contains(&st.action) {
            2
        } else {
            0
        };
        cx.put(3, mario);
        cx.put(4, st.holding as i32);
        if self.flags {
            let mut f = (t.intangible_timer != 0) as i32 | (t.raw_1ac[1] != 0) as i32 * 2 | (t.raw_1ac[3] != 0) as i32 * 4;
            if self.rest_offset.is_some_and(|r| dist(t.pos, [t.home[0] + r[0], t.home[1] + r[1], t.home[2] + r[2]]) < 2.0) {
                f |= 8;
            }
            cx.put(5, f);
        }
        if self.track_pos {
            cx.put(6, pack_pos(t.pos, self.grid_log2));
        }
        cx.all_children();
    }
}

fn preset(kind: &str) -> Option<NpcParams> {
    let s = |x: &str| Some(OneOrMany::One(x.to_string()));
    let d = NpcParams::default();
    Some(match kind {
        "SignRead" => NpcParams {
            bhv: Some(OneOrMany::Many(vec!["bhvMessagePanel".into(), "bhvSignOnWall".into()])),
            aim: Some(Aim::Front), aim_dist: Some(220.0), step: Some(200.0), mario_actions: Some(vec![ACT_READING_SIGN]), ..d
        },
        "YoshiTalk" => NpcParams { bhv: s("bhvYoshi"), ..d },
        "LakituIntro" => NpcParams {
            bhv: s("bhvCameraLakitu"), param2: Some(vec![1]), aim: Some(Aim::Point), aim_point: Some(LAKITU_BRIDGE), ..d
        },
        "MipsCatch" => NpcParams { bhv: s("bhvMips"), min_stars: Some(15), track_pos: Some(true), ..d },
        "UkikiCatch" | "UkikiFollowToCage" => NpcParams {
            bhv: s("bhvUkiki"), param2: Some(vec![0]), star_bhv: Some("bhvUkikiCage".into()), track_pos: Some(true), ..d
        },
        "HootWake" => NpcParams { bhv: s("bhvHoot"), flags: Some(true), rest_offset: Some(HOOT_REST), ..d },
        "HootRide" => NpcParams { bhv: s("bhvHoot"), flags: Some(true), mario_actions: Some(vec![ACT_RIDING_HOOT]), ..d },
        "KoopaRaceClaim" => NpcParams { bhv: s("bhvKoopa"), param2: Some(vec![2, 3]), star_bhv: Some("self".into()), ..d },
        "PenguinRaceTalk" | "PenguinRaceClaim" => NpcParams {
            bhv: s("bhvRacingPenguin"), star_bhv: Some("self".into()), flags: Some(true), ..d
        },
        "SnowmanWakeBody" => NpcParams {
            bhv: s("bhvSnowmansBottom"), star_bhv: Some("bhvSnowmansHead".into()), aim: Some(Aim::Near), aim_dist: Some(300.0),
            track_pos: Some(true), ..d
        },
        "SnowmanFinish" => NpcParams {
            bhv: s("bhvSnowmansHead"), star_bhv: Some("self".into()), aim: Some(Aim::Front), aim_dist: Some(500.0), ..d
        },
        "NpcTalk" => d,
        _ => return None,
    })
}

// ---------------------------------------------------------------- races

#[derive(Clone, Copy, PartialEq)]
enum RaceKind {
    Koopa,
    Penguin,
}

/// A race in progress (Koopa the Quick or the big penguin): writes
/// `mario_goal` (bucket of Mario's distance to the finish), `lead` (racer's
/// distance minus Mario's, in buckets, -4..4; positive = Mario ahead) and
/// `won` (the finish rule of rrt holds). Active only while the racer races
/// and its star is missing.
///
/// Example: `{"kind":"KoopaRaceReachFlag","params":{"step":300}}`
#[derive(Deserialize, Default)]
#[serde(default)]
pub struct RaceParams {
    pub step: Option<f32>,
}

pub struct Race {
    kind: RaceKind,
    step: f32,
}

impl Race {
    fn racing(&self, r: &Obj) -> bool {
        match self.kind {
            RaceKind::Koopa => r.action == 3 && matches!(byte1(r), 2 | 3),
            RaceKind::Penguin => matches!(r.action, 2 | 3) && r.raw_1ac[3] == 0,
        }
    }
}

impl Objective for Race {
    fn fields(&self) -> &[&'static str] {
        &["mario_goal", "lead", "won"]
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        let (rb, gb) = match self.kind {
            RaceKind::Koopa => ("bhvKoopa", "bhvKoopaRaceEndpoint"),
            RaceKind::Penguin => ("bhvRacingPenguin", "bhvPenguinRaceFinishLine"),
        };
        let Some(r) = nearest(with_bhv(st, rb).filter(|o| self.racing(o)), st.pos) else { return };
        if !star_missing(st, st.course, byte0(r)) {
            return;
        }
        let Some(g) = nearest(with_bhv(st, gb), r.pos) else { return };
        let (dm, dr) = (dist(st.pos, g.pos), dist(r.pos, g.pos));
        let won = match self.kind {
            RaceKind::Koopa => dm < 390.0 && dr >= 400.0,
            RaceKind::Penguin => dm < 990.0 && st.pos[2] < g.pos[2],
        };
        cx.put(0, bk(dm, self.step, 24));
        cx.put(1, bucket(dr - dm, self.step).clamp(-4, 4));
        cx.put(2, won as i32);
        cx.all_children();
    }
}

// ---------------------------------------------------------------- Tuxie

/// Tuxie's mother and the baby penguin: `phase` (the mother's oAction, 0
/// waiting, 1 baby returned), `carrying` (Mario holds a baby), `remaining`
/// (bucket of Mario-baby-mother straight-line distance, or Mario-mother
/// while carrying; -1 when no baby is loaded). Active while the mother's
/// star (byte 0) is missing.
///
/// Example: `{"kind":"TuxieReturnBaby","params":{"step":400}}`
#[derive(Deserialize)]
#[serde(default)]
pub struct Tuxie {
    pub step: f32,
}
impl Default for Tuxie {
    fn default() -> Self {
        Tuxie { step: 400.0 }
    }
}

impl Objective for Tuxie {
    fn fields(&self) -> &[&'static str] {
        &["phase", "carrying", "remaining"]
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        let Some(m) = nearest(with_bhv(st, "bhvTuxiesMother").filter(|m| matches!(m.action, 0 | 1)), st.pos) else { return };
        if !star_missing(st, st.course, byte0(m)) {
            return;
        }
        let babies = || with_bhv(st, "bhvSmallPenguin").chain(with_bhv(st, "bhvPenguinBaby")).filter(|b| byte1(b) == byte1(m));
        let held = babies().find(|b| Some(b.slot) == st.held_slot);
        let rem = match (held, nearest(babies(), st.pos)) {
            (Some(_), _) => Some(dist(st.pos, m.pos)),
            (None, Some(b)) => Some(dist(st.pos, b.pos) + dist(b.pos, m.pos)),
            _ => None,
        };
        cx.put(0, m.action);
        cx.put(1, held.is_some() as i32);
        cx.put(2, rem.map_or(-1, |r| bk(r, self.step, 30)));
        cx.all_children();
    }
}

// ---------------------------------------------------------------- Snowman

/// Leading the snowman's body to its head: the body's oAction (1-4; 3 and 4
/// mean it arrived), `body_site` (bucket of the body's distance to `site`),
/// `mario_site` (Mario's, coarser) and `checkpoint` (a body checkpoint is
/// loaded). Active while the body is in action 1..=4 and the head's star
/// is missing.
///
/// Example: `{"kind":"SnowmanLeadBody","params":{"site":[-4230,-1344,1813]}}`
#[derive(Deserialize)]
#[serde(default)]
pub struct SnowmanLead {
    pub site: [f32; 3],
    pub body_step: f32,
    pub mario_step: f32,
}
impl Default for SnowmanLead {
    fn default() -> Self {
        SnowmanLead { site: SNOWMAN_SITE, body_step: 1000.0, mario_step: 2000.0 }
    }
}

impl Objective for SnowmanLead {
    fn fields(&self) -> &[&'static str] {
        &["body_action", "body_site", "mario_site", "checkpoint"]
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        let Some(b) = nearest(with_bhv(st, "bhvSnowmansBottom").filter(|o| (1..=4).contains(&o.action)), self.site) else { return };
        let Some(head) = with_bhv(st, "bhvSnowmansHead").next() else { return };
        if !star_missing(st, st.course, byte0(head)) {
            return;
        }
        cx.put(0, b.action);
        cx.put(1, bk(dist(b.pos, self.site), self.body_step, 12));
        cx.put(2, bk(dist(st.pos, self.site), self.mario_step, 8));
        cx.put(3, with_bhv(st, "bhvSnowmansBodyCheckpoint").next().is_some() as i32);
        cx.all_children();
    }
}

// ---------------------------------------------------------------- missions

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum K {
    Star,
    Box,
    RedCoinStar,
    HiddenStar,
}

/// A fixed star spot of the level data (rrt `Spot`).
pub struct Spot {
    pub course: i16,
    pub index: u8,
    pub level: u32,
    pub area: u8,
    /// Bit n-1 = act n; 0x3f = all.
    pub acts: u8,
    pub pos: [f32; 3],
    pub kind: K,
    pub name: &'static str,
}

impl Spot {
    fn act_ok(&self, s: &St) -> bool {
        self.acts == 0x3f || (1..=6).contains(&s.act) && self.acts & (1 << (s.act - 1)) != 0
    }
    fn radius(&self) -> f32 {
        if self.kind == K::Box { 250.0 } else { 160.0 }
    }
}

/// Every star whose spot is fixed in the level data (copied from
/// objective-rrt/objectives/src/extra/missions/table.rs; US).
pub static SPOTS: &[Spot] = &[
    Spot { course: 1, index: 3, level: LEVEL_BOB, area: 1, acts: 0x3f, pos: [-6000.0, 1000.0, 2400.0], kind: K::RedCoinStar, name: "FIND THE 8 RED COINS" },
    Spot { course: 1, index: 4, level: LEVEL_BOB, area: 1, acts: 0x3f, pos: [-6600.0, 1000.0, 1250.0], kind: K::HiddenStar, name: "MARIO WINGS TO THE SKY" },
    Spot { course: 1, index: 5, level: LEVEL_BOB, area: 1, acts: 0x3f, pos: [1550.0, 1200.0, 300.0], kind: K::Star, name: "BEHIND CHAIN CHOMP'S GATE" },
    Spot { course: 1, index: 2, level: LEVEL_BOB, area: 1, acts: 0x3f, pos: [5540.0, 3350.0, 1200.0], kind: K::Box, name: "SHOOT TO THE ISLAND IN THE SKY" },
    Spot { course: 2, index: 1, level: LEVEL_WF, area: 1, acts: 0x3e, pos: [300.0, 5550.0, 0.0], kind: K::Star, name: "TO THE TOP OF THE FORTRESS" },
    Spot { course: 2, index: 2, level: LEVEL_WF, area: 1, acts: 0x3f, pos: [-2500.0, 1500.0, -750.0], kind: K::Star, name: "SHOOT INTO THE WILD BLUE" },
    Spot { course: 2, index: 3, level: LEVEL_WF, area: 1, acts: 0x3f, pos: [4600.0, 550.0, 2500.0], kind: K::RedCoinStar, name: "RED COINS ON THE FLOATING ISLE" },
    Spot { course: 2, index: 4, level: LEVEL_WF, area: 1, acts: 0x3f, pos: [2880.0, 4300.0, 190.0], kind: K::Star, name: "FALL ONTO THE CAGED ISLAND" },
    Spot { course: 2, index: 5, level: LEVEL_WF, area: 1, acts: 0x3f, pos: [590.0, 2450.0, 2650.0], kind: K::Star, name: "BLAST AWAY THE WALL" },
    Spot { course: 3, index: 3, level: LEVEL_JRB, area: 1, acts: 0x3f, pos: [4900.0, 2400.0, 800.0], kind: K::RedCoinStar, name: "RED COINS ON THE SHIP AFLOAT" },
    Spot { course: 3, index: 4, level: LEVEL_JRB, area: 1, acts: 0x3f, pos: [1540.0, 2160.0, 2130.0], kind: K::Box, name: "BLAST TO THE STONE PILLAR" },
    Spot { course: 3, index: 5, level: LEVEL_JRB, area: 1, acts: 0x3e, pos: [5000.0, -4800.0, 2500.0], kind: K::Star, name: "THROUGH THE JET STREAM" },
    Spot { course: 3, index: 0, level: LEVEL_JRB, area: 2, acts: 0x3f, pos: [0.0, 1600.0, 3000.0], kind: K::Box, name: "PLUNDER IN THE SUNKEN SHIP" },
    Spot { course: 4, index: 3, level: LEVEL_CCM, area: 1, acts: 0x3f, pos: [4200.0, -927.0, 400.0], kind: K::RedCoinStar, name: "FROSTY SLIDE FOR 8 RED COINS" },
    Spot { course: 4, index: 5, level: LEVEL_CCM, area: 1, acts: 0x3f, pos: [-2000.0, -2200.0, -3000.0], kind: K::Star, name: "WALL KICKS WILL WORK" },
    Spot { course: 5, index: 2, level: LEVEL_BBH, area: 1, acts: 0x3f, pos: [-2030.0, 1350.0, 1940.0], kind: K::Star, name: "SECRET OF THE HAUNTED BOOKS" },
    Spot { course: 5, index: 3, level: LEVEL_BBH, area: 1, acts: 0x3f, pos: [-204.0, 1100.0, 1576.0], kind: K::RedCoinStar, name: "SEEK THE 8 RED COINS" },
    Spot { course: 6, index: 0, level: LEVEL_HMC, area: 1, acts: 0x3f, pos: [-3600.0, -4000.0, 3600.0], kind: K::Star, name: "SWIMMING BEAST IN THE CAVERN" },
    Spot { course: 6, index: 1, level: LEVEL_HMC, area: 1, acts: 0x3f, pos: [4000.0, 300.0, 5000.0], kind: K::RedCoinStar, name: "ELEVATE FOR 8 RED COINS" },
    Spot { course: 6, index: 2, level: LEVEL_HMC, area: 1, acts: 0x3f, pos: [6200.0, -4400.0, 2300.0], kind: K::Star, name: "METAL-HEAD MARIO CAN MOVE!" },
    Spot { course: 6, index: 3, level: LEVEL_HMC, area: 1, acts: 0x3f, pos: [-2100.0, 2100.0, -7550.0], kind: K::Star, name: "NAVIGATING THE TOXIC MAZE" },
    Spot { course: 6, index: 4, level: LEVEL_HMC, area: 1, acts: 0x3f, pos: [-6500.0, 2700.0, -1600.0], kind: K::Star, name: "A-MAZE-ING EMERGENCY EXIT" },
    Spot { course: 6, index: 5, level: LEVEL_HMC, area: 1, acts: 0x3f, pos: [-5000.0, 3050.0, -6700.0], kind: K::Star, name: "WATCH FOR ROLLING ROCKS" },
    Spot { course: 7, index: 2, level: LEVEL_LLL, area: 1, acts: 0x3f, pos: [-4400.0, 350.0, 250.0], kind: K::RedCoinStar, name: "8-COIN PUZZLE WITH 15 PIECES" },
    Spot { course: 7, index: 3, level: LEVEL_LLL, area: 1, acts: 0x3f, pos: [3100.0, 400.0, 7900.0], kind: K::Star, name: "RED-HOT LOG ROLLING" },
    Spot { course: 7, index: 4, level: LEVEL_LLL, area: 2, acts: 0x3f, pos: [2523.0, 3850.0, -901.0], kind: K::Star, name: "HOT-FOOT-IT INTO THE VOLCANO" },
    Spot { course: 7, index: 5, level: LEVEL_LLL, area: 2, acts: 0x3f, pos: [1800.0, 3400.0, 1450.0], kind: K::Star, name: "ELEVATOR TOUR IN THE VOLCANO" },
    Spot { course: 8, index: 1, level: LEVEL_SSL, area: 1, acts: 0x3f, pos: [-2050.0, 1200.0, -580.0], kind: K::Star, name: "SHINING ATOP THE PYRAMID" },
    Spot { course: 8, index: 4, level: LEVEL_SSL, area: 1, acts: 0x3f, pos: [6000.0, 800.0, 3500.0], kind: K::RedCoinStar, name: "FREE FLYING FOR 8 RED COINS" },
    Spot { course: 8, index: 2, level: LEVEL_SSL, area: 2, acts: 0x3f, pos: [500.0, 5050.0, -500.0], kind: K::Star, name: "INSIDE THE ANCIENT PYRAMID" },
    Spot { course: 8, index: 5, level: LEVEL_SSL, area: 2, acts: 0x3f, pos: [900.0, 1400.0, 2350.0], kind: K::HiddenStar, name: "PYRAMID PUZZLE" },
    Spot { course: 9, index: 0, level: LEVEL_DDD, area: 2, acts: 0x3f, pos: [3900.0, 850.0, -600.0], kind: K::Star, name: "BOARD BOWSER'S SUB" },
    Spot { course: 9, index: 2, level: LEVEL_DDD, area: 2, acts: 0x3f, pos: [5513.0, 1200.0, 900.0], kind: K::RedCoinStar, name: "POLE-JUMPING FOR RED COINS" },
    Spot { course: 9, index: 5, level: LEVEL_DDD, area: 2, acts: 0x3f, pos: [2030.0, -3700.0, -2780.0], kind: K::Star, name: "COLLECT THE CAPS..." },
    Spot { course: 10, index: 0, level: LEVEL_SL, area: 1, acts: 0x3f, pos: [700.0, 4500.0, 690.0], kind: K::Star, name: "SNOWMAN'S BIG HEAD" },
    Spot { course: 10, index: 2, level: LEVEL_SL, area: 1, acts: 0x3f, pos: [4350.0, 1350.0, 4350.0], kind: K::Star, name: "IN THE DEEP FREEZE" },
    Spot { course: 10, index: 4, level: LEVEL_SL, area: 1, acts: 0x3f, pos: [5000.0, 1200.0, 0.0], kind: K::RedCoinStar, name: "SHELL SHREDDIN' FOR RED COINS" },
    Spot { course: 10, index: 3, level: LEVEL_SL, area: 1, acts: 0x3f, pos: [-4700.0, 1300.0, 5850.0], kind: K::Box, name: "WHIRL FROM THE FREEZING POND" },
    Spot { course: 10, index: 5, level: LEVEL_SL, area: 2, acts: 0x3f, pos: [0.0, 500.0, 1000.0], kind: K::Star, name: "INTO THE IGLOO" },
    Spot { course: 11, index: 2, level: LEVEL_WDW, area: 1, acts: 0x3f, pos: [3360.0, 1580.0, 2660.0], kind: K::HiddenStar, name: "SECRETS IN THE SHALLOWS & SKY" },
    Spot { course: 11, index: 3, level: LEVEL_WDW, area: 1, acts: 0x3f, pos: [890.0, 3400.0, -2040.0], kind: K::Star, name: "EXPRESS ELEVATOR--HURRY UP!" },
    Spot { course: 11, index: 0, level: LEVEL_WDW, area: 1, acts: 0x3f, pos: [-2200.0, 2600.0, 3500.0], kind: K::Box, name: "SHOCKING ARROW LIFTS!" },
    Spot { course: 11, index: 1, level: LEVEL_WDW, area: 1, acts: 0x3f, pos: [1550.0, 4350.0, 100.0], kind: K::Box, name: "TOP O' THE TOWN" },
    Spot { course: 11, index: 4, level: LEVEL_WDW, area: 2, acts: 0x3f, pos: [-770.0, -1600.0, 3600.0], kind: K::RedCoinStar, name: "GO TO TOWN FOR RED COINS" },
    Spot { course: 11, index: 5, level: LEVEL_WDW, area: 2, acts: 0x3f, pos: [2180.0, -840.0, 3720.0], kind: K::Star, name: "QUICK RACE THROUGH DOWNTOWN!" },
    Spot { course: 12, index: 0, level: LEVEL_TTM, area: 1, acts: 0x1, pos: [1200.0, 2600.0, 150.0], kind: K::Star, name: "SCALE THE MOUNTAIN" },
    Spot { course: 12, index: 2, level: LEVEL_TTM, area: 1, acts: 0x3f, pos: [-3250.0, -2500.0, -3700.0], kind: K::RedCoinStar, name: "SCARY 'SHROOMS, RED COINS" },
    Spot { course: 12, index: 3, level: LEVEL_TTM, area: 1, acts: 0x3f, pos: [-2900.0, -2700.0, 3650.0], kind: K::Star, name: "MYSTERIOUS MOUNTAINSIDE" },
    Spot { course: 12, index: 4, level: LEVEL_TTM, area: 1, acts: 0x3f, pos: [1800.0, 1200.0, 1050.0], kind: K::Star, name: "BREATHTAKING VIEW FROM BRIDGE" },
    Spot { course: 12, index: 5, level: LEVEL_TTM, area: 1, acts: 0x3f, pos: [7300.0, -3100.0, 1300.0], kind: K::Star, name: "BLAST TO THE LONELY MUSHROOM" },
    Spot { course: 13, index: 1, level: LEVEL_THI, area: 1, acts: 0x3f, pos: [2600.0, 3500.0, -2400.0], kind: K::Box, name: "THE TIP TOP OF THE HUGE ISLAND" },
    Spot { course: 13, index: 3, level: LEVEL_THI, area: 2, acts: 0x3f, pos: [0.0, -700.0, -4500.0], kind: K::HiddenStar, name: "FIVE ITTY BITTY SECRETS" },
    Spot { course: 13, index: 4, level: LEVEL_THI, area: 3, acts: 0x3f, pos: [-1800.0, 800.0, -1500.0], kind: K::RedCoinStar, name: "WIGGLER'S RED COINS" },
    Spot { course: 14, index: 0, level: LEVEL_TTC, area: 1, acts: 0x3f, pos: [-1450.0, -1130.0, -1050.0], kind: K::Star, name: "ROLL INTO THE CAGE" },
    Spot { course: 14, index: 1, level: LEVEL_TTC, area: 1, acts: 0x3f, pos: [-1850.0, 300.0, -950.0], kind: K::Star, name: "THE PIT AND THE PENDULUMS" },
    Spot { course: 14, index: 2, level: LEVEL_TTC, area: 1, acts: 0x3f, pos: [-1300.0, -2250.0, -1300.0], kind: K::Star, name: "GET A HAND" },
    Spot { course: 14, index: 3, level: LEVEL_TTC, area: 1, acts: 0x3f, pos: [2200.0, 7300.0, 2210.0], kind: K::Star, name: "STOMP ON THE THWOMP" },
    Spot { course: 14, index: 4, level: LEVEL_TTC, area: 1, acts: 0x3f, pos: [-1050.0, 2400.0, -790.0], kind: K::Star, name: "TIMED JUMPS ON MOVING BARS" },
    Spot { course: 14, index: 5, level: LEVEL_TTC, area: 1, acts: 0x3f, pos: [1815.0, -3200.0, 800.0], kind: K::RedCoinStar, name: "STOP TIME FOR RED COINS" },
    Spot { course: 15, index: 0, level: LEVEL_RR, area: 1, acts: 0x3f, pos: [1450.0, 3400.0, -2352.0], kind: K::Star, name: "CRUISER CROSSING THE RAINBOW" },
    Spot { course: 15, index: 1, level: LEVEL_RR, area: 1, acts: 0x3f, pos: [-4200.0, 6700.0, -4450.0], kind: K::Star, name: "THE BIG HOUSE IN THE SKY" },
    Spot { course: 15, index: 2, level: LEVEL_RR, area: 1, acts: 0x3f, pos: [-5150.0, -1400.0, 0.0], kind: K::RedCoinStar, name: "COINS AMASSED IN A MAZE" },
    Spot { course: 15, index: 3, level: LEVEL_RR, area: 1, acts: 0x3f, pos: [-5850.0, -700.0, 4950.0], kind: K::Star, name: "SWINGIN' IN THE BREEZE" },
    Spot { course: 15, index: 4, level: LEVEL_RR, area: 1, acts: 0x3f, pos: [3700.0, -400.0, 6600.0], kind: K::Star, name: "TRICKY TRIANGLES!" },
    Spot { course: 15, index: 5, level: LEVEL_RR, area: 1, acts: 0x3f, pos: [5000.0, 4100.0, 4440.0], kind: K::Box, name: "SOMEWHERE OVER THE RAINBOW" },
    Spot { course: 16, index: 0, level: LEVEL_BITDW, area: 1, acts: 0x3f, pos: [7180.0, 3000.0, 0.0], kind: K::RedCoinStar, name: "BOWSER IN THE DARK WORLD" },
    Spot { course: 17, index: 0, level: LEVEL_BITFS, area: 1, acts: 0x3f, pos: [1200.0, 5700.0, 160.0], kind: K::RedCoinStar, name: "BOWSER IN THE FIRE SEA" },
    Spot { course: 18, index: 0, level: LEVEL_BITS, area: 1, acts: 0x3f, pos: [350.0, 6800.0, -6800.0], kind: K::RedCoinStar, name: "BOWSER IN THE SKY" },
    Spot { course: 19, index: 0, level: LEVEL_PSS, area: 1, acts: 0x3f, pos: [-6385.0, -4200.0, 5770.0], kind: K::Box, name: "THE PRINCESS'S SECRET SLIDE" },
    Spot { course: 20, index: 0, level: LEVEL_COTMC, area: 1, acts: 0x3f, pos: [0.0, -200.0, -7000.0], kind: K::RedCoinStar, name: "CAVERN OF THE METAL CAP" },
    Spot { course: 21, index: 0, level: LEVEL_TOTWC, area: 1, acts: 0x3f, pos: [800.0, -1700.0, 0.0], kind: K::RedCoinStar, name: "TOWER OF THE WING CAP" },
    Spot { course: 22, index: 0, level: LEVEL_VCUTM, area: 1, acts: 0x3f, pos: [4600.0, 250.0, -4500.0], kind: K::RedCoinStar, name: "VANISH CAP UNDER THE MOAT" },
    Spot { course: 23, index: 0, level: LEVEL_WMOTR, area: 1, acts: 0x3f, pos: [-160.0, 1950.0, -470.0], kind: K::RedCoinStar, name: "WING MARIO OVER THE RAINBOW" },
    Spot { course: 24, index: 0, level: LEVEL_SA, area: 1, acts: 0x3f, pos: [0.0, -4250.0, 0.0], kind: K::RedCoinStar, name: "THE SECRET AQUARIUM" },
];

fn count_bhv(s: &St, name: &str) -> usize {
    with_bhv(s, name).count()
}

/// Mission spots of the level: the nearest uncollected spot (of `kinds`,
/// optionally one `course`/`index`) in Mario's area. Writes `spot`
/// (course * 8 + index, -1 when the level's spots are in other areas),
/// `dist` (bucket of the straight-line distance), `ready` (8 red coins
/// collected / no hidden-star trigger left / always for plain stars),
/// `reached` (within the spot's reach radius). With `counters`: also
/// `progress` (red coins 0..8, or secrets left 0..5) and `next` (bucket of
/// the distance to the nearest red coin / secret trigger, -1 none).
///
/// Example: `{"kind":"MissionReachRedCoinStar","params":{"step":400}}`
#[derive(Deserialize)]
#[serde(default)]
pub struct Mission {
    pub kinds: Vec<K>,
    pub course: Option<i16>,
    pub index: Option<u8>,
    pub step: f32,
    pub counters: bool,
}
impl Default for Mission {
    fn default() -> Self {
        Mission { kinds: vec![K::Star, K::Box, K::RedCoinStar, K::HiddenStar], course: None, index: None, step: 400.0, counters: false }
    }
}

impl Mission {
    fn spots<'a>(&'a self, st: &'a St) -> impl Iterator<Item = &'static Spot> + 'a {
        SPOTS.iter().filter(move |p| {
            p.level == st.level as u32
                && self.kinds.contains(&p.kind)
                && self.course.is_none_or(|c| c == p.course)
                && self.index.is_none_or(|i| i == p.index)
                && p.act_ok(st)
                && !st.has_star(p.course, p.index)
        })
    }
}

fn ready(st: &St, k: K) -> bool {
    match k {
        K::Star | K::Box => true,
        K::RedCoinStar => st.red_coins >= 8,
        K::HiddenStar => st.objects.is_some() && count_bhv(st, "bhvHiddenStarTrigger") == 0,
    }
}

impl Objective for Mission {
    fn fields(&self) -> &[&'static str] {
        &["spot", "dist", "ready", "reached", "progress", "next"]
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        let all: Vec<&Spot> = self.spots(st).collect();
        if all.is_empty() {
            return;
        }
        let here = all.iter().filter(|p| p.area == st.area).min_by(|a, b| dist(st.pos, a.pos).total_cmp(&dist(st.pos, b.pos)));
        let Some(p) = here else {
            cx.put(0, -1);
            return;
        };
        let d = dist(st.pos, p.pos);
        cx.put(0, p.course as i32 * 8 + p.index as i32);
        cx.put(1, bk(d, self.step, 30));
        cx.put(2, ready(st, p.kind) as i32);
        cx.put(3, (d <= p.radius()) as i32);
        if self.counters {
            let name = match p.kind {
                K::RedCoinStar => Some("bhvRedCoin"),
                K::HiddenStar => Some("bhvHiddenStarTrigger"),
                _ => None,
            };
            let loaded = st.objects.is_some();
            cx.put(4, match p.kind {
                K::RedCoinStar => st.red_coins.min(8) as i32,
                K::HiddenStar if loaded => count_bhv(st, "bhvHiddenStarTrigger").min(5) as i32,
                _ => -1,
            });
            let next = name.and_then(|n| nearest(with_bhv(st, n), st.pos));
            cx.put(5, next.map_or(-1, |o| bk(dist(st.pos, o.pos), self.step, 12)));
        }
        cx.all_children();
    }
}

/// A star object that is loaded but not in [`SPOTS`] and farther than 4500
/// (e.g. spawned by a boss or a quest): writes `star` (index) and `dist`.
///
/// Example: `{"kind":"MissionLoadedStar"}`
#[derive(Deserialize)]
#[serde(default)]
pub struct LoadedStar {
    pub step: f32,
    pub far: f32,
}
impl Default for LoadedStar {
    fn default() -> Self {
        LoadedStar { step: 400.0, far: 4500.0 }
    }
}

impl Objective for LoadedStar {
    fn fields(&self) -> &[&'static str] {
        &["star", "dist"]
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        let course = course_of_level(st.level as u32);
        if course == 0 {
            return;
        }
        let pos = |o: &Obj| if o.bhv_name() == "bhvStarSpawnCoordinates" { o.home } else { o.pos };
        let star = st
            .objects()
            .iter()
            .filter(|o| o.interact_type == INTERACT_STAR_OR_KEY && !o.bhv_name().contains("Key"))
            .filter(|o| byte0(o) <= 6 && !st.has_star(course, byte0(o)))
            .filter(|o| dist(st.pos, pos(o)) > self.far && !SPOTS.iter().any(|p| p.course == course && p.index == byte0(o)))
            .min_by(|a, b| dist(st.pos, pos(a)).total_cmp(&dist(st.pos, pos(b))));
        let Some(o) = star else { return };
        cx.put(0, byte0(o) as i32);
        cx.put(1, bk(dist(st.pos, pos(o)), self.step, 30));
        cx.all_children();
    }
}

/// Coins in a main course (1-15) whose 100-coin star is missing, below 100:
/// `coins` (exact; every milestone of rrt is a value of it) and `near`
/// (bucket of the distance to the nearest coin, -1 none loaded).
///
/// Example: `{"kind":"MissionCoins","params":{"step":300}}`
#[derive(Deserialize)]
#[serde(default)]
pub struct Coins {
    pub step: f32,
}
impl Default for Coins {
    fn default() -> Self {
        Coins { step: 400.0 }
    }
}

impl Objective for Coins {
    fn fields(&self) -> &[&'static str] {
        &["coins", "near"]
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        let course = course_of_level(st.level as u32);
        if !(1..=15).contains(&course) || st.has_star(course, 6) || st.coins >= 100 {
            return;
        }
        cx.put(0, st.coins as i32);
        let near = nearest(st.objects().iter().filter(|o| o.interact_type == INTERACT_COIN), st.pos);
        cx.put(1, near.map_or(-1, |o| bk(dist(st.pos, o.pos), self.step, 6)));
        cx.all_children();
    }
}

const PSS_END: ([f32; 2], [f32; 2], f32) = ([-6869.0, -5845.0], [3047.0, 3251.0], -4300.0);
const PSS_END_CENTER: [f32; 3] = [-6357.0, -4500.0, 3149.0];

/// The Princess's Secret Slide under 21 s: `dist` (bucket of the distance to
/// the slide's end) and `at_end`. The time limit is not a field: of two states
/// in one cell the archive keeps the faster.
///
/// Example: `{"kind":"MissionPssUnder21","params":{"step":300}}`
#[derive(Deserialize)]
#[serde(default)]
pub struct PssUnder21 {
    pub step: f32,
}
impl Default for PssUnder21 {
    fn default() -> Self {
        PssUnder21 { step: 400.0 }
    }
}

impl Objective for PssUnder21 {
    fn fields(&self) -> &[&'static str] {
        &["dist", "at_end"]
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if st.level as u32 != LEVEL_PSS || st.has_star(19, 1) {
            return;
        }
        let (x, z, y) = PSS_END;
        let at_end = (x[0] - 50.0..=x[1] + 50.0).contains(&st.pos[0]) && (z[0] - 50.0..=z[1] + 50.0).contains(&st.pos[2]) && st.pos[1] < y;
        cx.put(0, bk(dist(st.pos, PSS_END_CENTER), self.step, 40));
        cx.put(1, at_end as i32);
        cx.all_children();
    }
}

fn parse<T: for<'de> Deserialize<'de> + 'static + Objective>(p: &Value) -> Option<Result<Box<dyn Objective>>> {
    Some(serde_json::from_value::<T>(p.clone()).map(|t| Box::new(t) as Box<dyn Objective>).map_err(Into::into))
}

/// The kinds of this family: `Some(objective)` for a kind name it knows, else `None`.
pub fn build(kind: &str, params: &Value) -> Option<Result<Box<dyn Objective>>> {
    if let Some(d) = preset(kind) {
        return Some(serde_json::from_value::<NpcParams>(params.clone()).map_err(Into::into).and_then(|p| Npc::new(p.or(d))).map(|n| Box::new(n) as Box<dyn Objective>));
    }
    let race = |k| {
        serde_json::from_value::<RaceParams>(params.clone())
            .map(|p| Box::new(Race { kind: k, step: p.step.unwrap_or(400.0) }) as Box<dyn Objective>)
            .map_err(Into::into)
    };
    let mission = |kinds: &[K], counters: bool| {
        serde_json::from_value::<Mission>(params.clone())
            .map(|mut m| {
                if !kinds.is_empty() {
                    m.kinds = kinds.to_vec();
                }
                m.counters = m.counters || counters;
                Box::new(m) as Box<dyn Objective>
            })
            .map_err(Into::into)
    };
    match kind {
        "KoopaRaceReachFlag" => Some(race(RaceKind::Koopa)),
        "PenguinRaceWin" => Some(race(RaceKind::Penguin)),
        "TuxieReturnBaby" => parse::<Tuxie>(params),
        "SnowmanLeadBody" => parse::<SnowmanLead>(params),
        "MissionReach" => Some(mission(&[], false)),
        "MissionReachStar" => Some(mission(&[K::Star], false)),
        "MissionReachBox" => Some(mission(&[K::Box], false)),
        "MissionReachRedCoinStar" => Some(mission(&[K::RedCoinStar], false)),
        "MissionReachHiddenStar" => Some(mission(&[K::HiddenStar], false)),
        "MissionCounter" => Some(mission(&[K::RedCoinStar, K::HiddenStar], true)),
        "MissionLoadedStar" => parse::<LoadedStar>(params),
        "MissionCoins" => parse::<Coins>(params),
        "MissionPssUnder21" => parse::<PssUnder21>(params),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{key_of, obj, st, with_objects};

    fn at(x: f32) -> St {
        st([x, 0.0, 0.0], 0.0, 9, 1, 0)
    }
    fn k(kind: &str, params: &str, s: &St) -> Vec<(u8, i32)> {
        key_of(&format!(r#"{{"kind":"{kind}","params":{params}}}"#), 0, s)
    }

    #[test]
    fn spots_table_is_complete() {
        assert_eq!(SPOTS.len(), 74);
        assert!(SPOTS.iter().all(|p| p.index <= 6 && !p.name.is_empty()));
    }

    #[test]
    fn silent_without_object_or_when_star_collected_or_few_stars() {
        assert!(k("YoshiTalk", "{}", &at(0.0)).is_empty());
        let y = obj("bhvYoshi", 1, [500.0, 0.0, 0.0]);
        assert!(!k("YoshiTalk", "{}", &with_objects(at(0.0), vec![y.clone()])).is_empty());
        // Mips needs 15 stars.
        let m = obj("bhvMips", 1, [100.0, 0.0, 0.0]);
        let mut s = with_objects(at(0.0), vec![m]);
        assert!(k("MipsCatch", "{}", &s).is_empty());
        s.stars = 15;
        assert!(!k("MipsCatch", "{}", &s).is_empty());
        // Koopa: star byte0 = 1 of course 1 collected -> silent.
        let mut ko = obj("bhvKoopa", 1, [100.0, 0.0, 0.0]);
        ko.beh_params = (1 << 24) | (2 << 16);
        let mut s = with_objects(at(0.0), vec![ko]);
        s.course = 1;
        assert!(!k("KoopaRaceClaim", "{}", &s).is_empty());
        s.course_stars[0] |= 1 << 1;
        assert!(k("KoopaRaceClaim", "{}", &s).is_empty());
    }

    #[test]
    fn distance_buckets_and_phase_and_hold() {
        let mut y = obj("bhvYoshi", 1, [350.0, 0.0, 0.0]);
        let a = k("YoshiTalk", "{}", &with_objects(at(0.0), vec![y.clone()]));
        assert_eq!(a[0], (0, 0));
        y.pos[0] = 450.0;
        assert_eq!(k("YoshiTalk", "{}", &with_objects(at(0.0), vec![y.clone()]))[0], (0, 1));
        y.action = 3; // phase change is a different cell
        assert_ne!(a, k("YoshiTalk", "{}", &with_objects(at(0.0), vec![y.clone()])));
        // Held vs not.
        let mut s = with_objects(at(0.0), vec![y.clone()]);
        let free = k("YoshiTalk", "{}", &s);
        s.held_slot = Some(1);
        assert_ne!(free, k("YoshiTalk", "{}", &s));
    }

    #[test]
    fn dialog_state_distinguishes() {
        let y = obj("bhvYoshi", 1, [100.0, 0.0, 0.0]);
        let a = with_objects(at(0.0), vec![y]);
        let mut b = a.clone();
        b.action = ACT_READING_NPC_DIALOG;
        assert_ne!(k("YoshiTalk", "{}", &a), k("YoshiTalk", "{}", &b));
        // The sign's reading action is a third value.
        let sg = obj("bhvSignOnWall", 2, [100.0, 0.0, 0.0]);
        let mut c = with_objects(at(0.0), vec![sg]);
        let n = k("SignRead", "{}", &c);
        c.action = ACT_READING_SIGN;
        let r = k("SignRead", "{}", &c);
        assert_ne!(n, r);
        assert!(r.contains(&(3, 2)));
    }

    #[test]
    fn nearest_object_is_chosen() {
        let a = obj("bhvYoshi", 1, [1000.0, 0.0, 0.0]);
        let b = obj("bhvYoshi", 2, [100.0, 0.0, 0.0]);
        let s = with_objects(at(0.0), vec![a.clone(), b.clone()]);
        let one = with_objects(at(0.0), vec![b]);
        assert_eq!(k("YoshiTalk", "{}", &s), k("YoshiTalk", "{}", &one));
    }

    #[test]
    fn ukiki_needs_cage_and_hoot_flags() {
        let u = obj("bhvUkiki", 1, [100.0, 0.0, 0.0]);
        let s = with_objects(at(0.0), vec![u.clone()]);
        assert!(k("UkikiCatch", "{}", &s).is_empty(), "no cage loaded");
        let mut cage = obj("bhvUkikiCage", 2, [0.0, 0.0, 0.0]);
        cage.beh_params = 3 << 24;
        let mut s = with_objects(at(0.0), vec![u, cage]);
        s.course = 5;
        assert!(!k("UkikiCatch", "{}", &s).is_empty());
        s.course_stars[4] |= 1 << 3;
        assert!(k("UkikiCatch", "{}", &s).is_empty());

        let mut h = obj("bhvHoot", 1, [100.0, 0.0, 0.0]);
        let awake = k("HootWake", "{}", &with_objects(at(0.0), vec![h.clone()]));
        h.intangible_timer = 10;
        h.pos = [h.home[0] - 800.0, h.home[1] + 150.0, h.home[2] - 300.0];
        let asleep = k("HootWake", "{}", &with_objects(at(0.0), vec![h]));
        assert!(asleep.contains(&(5, 9)));
        assert_ne!(awake, asleep);
    }

    #[test]
    fn koopa_race() {
        let mut ko = obj("bhvKoopa", 1, [3000.0, 0.0, 0.0]);
        ko.beh_params = 2 << 16;
        ko.action = 3;
        let flag = obj("bhvKoopaRaceEndpoint", 2, [0.0, 0.0, 0.0]);
        let s = with_objects(at(2000.0), vec![ko.clone(), flag.clone()]);
        let far = k("KoopaRaceReachFlag", "{}", &s);
        assert_eq!(far, vec![(0, 5), (1, 2), (2, 0)]);
        let won = k("KoopaRaceReachFlag", "{}", &with_objects(at(300.0), vec![ko.clone(), flag.clone()]));
        assert_eq!(won[2], (2, 1));
        // Not racing: silent.
        ko.action = 2;
        assert!(k("KoopaRaceReachFlag", "{}", &with_objects(at(2000.0), vec![ko, flag])).is_empty());
    }

    #[test]
    fn tuxie_and_snowman() {
        let mut m = obj("bhvTuxiesMother", 1, [0.0, 0.0, 0.0]);
        m.beh_params = 1 << 24;
        let baby = obj("bhvSmallPenguin", 2, [800.0, 0.0, 0.0]);
        let s = with_objects(at(400.0), vec![m.clone(), baby]);
        let free = k("TuxieReturnBaby", "{}", &s);
        assert_eq!(free, vec![(0, 0), (1, 0), (2, 3)]); // 400 + 800 = 1200
        let mut c = s.clone();
        c.held_slot = Some(2);
        assert_eq!(k("TuxieReturnBaby", "{}", &c), vec![(0, 0), (1, 1), (2, 1)]);
        m.action = 1;
        let b2 = obj("bhvSmallPenguin", 2, [800.0, 0.0, 0.0]);
        assert_ne!(k("TuxieReturnBaby", "{}", &with_objects(at(400.0), vec![m, b2])), free);

        let mut body = obj("bhvSnowmansBottom", 1, [-4230.0, -1344.0, 3813.0]);
        body.action = 1;
        let mut head = obj("bhvSnowmansHead", 2, [0.0, 0.0, 0.0]);
        head.beh_params = 0;
        let s = with_objects(at(0.0), vec![body.clone(), head.clone()]);
        let a = k("SnowmanLeadBody", "{}", &s);
        assert_eq!(a[1], (1, 2));
        body.action = 3;
        assert_ne!(a, k("SnowmanLeadBody", "{}", &with_objects(at(0.0), vec![body, head])));
    }

    #[test]
    fn missions() {
        // BOB (level 9, course 1): the nearest uncollected spot in the area.
        let s = st([1500.0, 1200.0, 300.0], 0.0, 9, 1, 0);
        let a = k("MissionReach", "{}", &s);
        assert_eq!(a[0], (0, 8 + 5)); // BEHIND CHAIN CHOMP'S GATE
        assert_eq!(a[1], (1, 0));
        assert_eq!(a[3], (3, 1)); // reached
        let mut c = s.clone();
        c.course_stars[0] |= 1 << 5;
        assert_ne!(k("MissionReach", "{}", &c)[0], a[0]);
        // Red coin star: ready only with 8 red coins.
        let mut r = st([0.0; 3], 0.0, 9, 1, 0);
        let r0 = k("MissionReachRedCoinStar", "{}", &r);
        assert_eq!(r0[0], (0, 8 + 3));
        assert_eq!(r0[2], (2, 0));
        r.red_coins = 8;
        assert_eq!(k("MissionReachRedCoinStar", "{}", &r)[2], (2, 1));
        // Not a course: silent. Counters add fields.
        assert!(k("MissionReach", "{}", &st([0.0; 3], 0.0, 6, 1, 0)).is_empty());
        r.red_coins = 3;
        let cn = k("MissionCounter", "{}", &r);
        assert!(cn.contains(&(4, 3)));
        // Coins: exact count; 100 is silent.
        let mut cs = st([0.0; 3], 0.0, 9, 1, 0);
        cs.coins = 37;
        assert_eq!(k("MissionCoins", "{}", &cs)[0], (0, 37));
        cs.coins = 100;
        assert!(k("MissionCoins", "{}", &cs).is_empty());
        // Loaded star far away and not in the table.
        let mut star = obj("bhvStar", 3, [9000.0, 0.0, 0.0]);
        star.interact_type = INTERACT_STAR_OR_KEY;
        star.beh_params = 1 << 24;
        let k4 = k("MissionLoadedStar", "{}", &with_objects(st([0.0; 3], 0.0, 9, 1, 0), vec![star]));
        assert_eq!(k4[0], (0, 1));
        // PSS.
        let p = st([-6357.0, -4500.0, 3149.0], 0.0, LEVEL_PSS as u16, 1, 0);
        assert_eq!(k("MissionPssUnder21", "{}", &p), vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn bad_params_error() {
        assert!(crate::objectives::build("NpcTalk", &serde_json::json!({})).is_err());
        assert!(crate::objectives::build("NpcTalk", &serde_json::json!({"bhv": "bhvYoshi"})).is_ok());
    }
}
