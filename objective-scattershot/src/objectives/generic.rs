//! The generic kinds: standing still, leaving a course, travelling to a
//! point, dialog and menu options, doors.
//!
//! Sources (objective-rrt/objectives/src): `mario/*.rs`, `menu.rs`, `door.rs`,
//! and the generators in `lib.rs`. An rrt goal becomes a bucketed key field,
//! gated to where the goal applies (PORTING.md).
//!
//! | rrt kind | scattershot |
//! |---|---|
//! | `Mario::Settle` (idle.rs) | [`Still`]: field `still` (the `still()` test: stationary flag, \|fv\| and \|vy\| < 0.1) |
//! | `Mario::Wait(n)` (idle.rs) | [`Still`]: field `wait`, frames spent still in buckets of `wait_step` (from `action_timer`) |
//! | `Mario::GoToCastle` (exit.rs) | [`ExitCourse`]: fields `exit` (action can exit through the pause menu), `pause` (menu cursor line while the pause menu is up) |
//! | `Mario::PosDist(p)` (pos.rs) | [`PosDist`]: field `dist`, straight-line distance bucket to a fixed `point` |
//! | `Mario::PosTime(p)` (time.rs) | not ported: "reach a place sooner" is what the archive does for every cell (the faster state of a key wins), so no objective is needed. Pace ladder / steering hints are search machinery |
//! | `Dialog::Read/Yes/No(id)` (menu.rs) | [`DialogOption`]: while a dialog is open (optionally only listed `ids`): fields `phase`, `last_page`, `line` (yes/no cursor). The dialog id and "a dialog is open" are `Busy`'s (`busy`, `dialog`); which answer was given is just `line` on the last page |
//! | `Menu::SaveAndContinue/ContinueNoSave` (menu.rs) | [`SaveMenuOption`]: in the save menu: `shown`, `line` (1 save and continue, 3 continue without saving), `ready`. That the menu is open is `Busy` (busy=3). Save and Quit is never an option, it is just line 2 |
//! | `Menu::SelectAct` (menu.rs) | [`ActSelectOption`]: on the act select screen: `cursor`, `ready`. Which acts are selectable is the game's own business; the cursor position is the distinction. Open screen is `Busy` (busy=4) |
//! | `Door::Enter` (door.rs) | [`DoorEnter`]: nearest enterable door (by behaviour, or a fixed point): `dist` bucket and `in_door` (a door action within 300). Door animations are `Busy`'s (busy=1); zone change is `Stage`/`Root`'s `level` |
//!
//! Left out on purpose: the rrt inputs (policies, macros, press timing) are
//! search machinery; the engine's `busy` mechanism presses through modals.
//! `Mario::GoToCastle`'s success (castle reached) is `Root`'s `level` field.
//! All distances are straight-line.

use anyhow::Result;
use obrrt_core::layout::{ACT_FLAG_PAUSE_EXIT, ACT_FLAG_STATIONARY, INTERACT_DOOR, INTERACT_WARP_DOOR, LEVEL_CASTLE, LEVEL_CASTLE_COURTYARD, LEVEL_CASTLE_GROUNDS, MENU_MODE_RENDER_PAUSE_SCREEN};
use obrrt_core::{St, bhv_addr};
use serde::Deserialize;
use serde_json::Value;

use super::util::{ActionGroup, bucket, dist, in_box};
use crate::objective::{Cx, Objective};

/// The kinds of this family: `Some(objective)` for a kind name it knows, else `None`.
pub fn build(kind: &str, params: &Value) -> Option<Result<Box<dyn Objective>>> {
    fn mk<T: Objective + for<'a> Deserialize<'a> + 'static>(p: &Value) -> Option<Result<Box<dyn Objective>>> {
        Some(serde_json::from_value::<T>(p.clone()).map(|o| Box::new(o) as Box<dyn Objective>).map_err(Into::into))
    }
    match kind {
        "Still" => mk::<Still>(params),
        "ExitCourse" => mk::<ExitCourse>(params),
        "PosDist" => mk::<PosDist>(params),
        "DialogOption" => mk::<DialogOption>(params),
        "SaveMenuOption" => mk::<SaveMenuOption>(params),
        "ActSelectOption" => mk::<ActSelectOption>(params),
        "DoorEnter" => mk::<DoorEnter>(params),
        "Near" => mk::<Near>(params),
        _ => None,
    }
}

/// `Near`: evaluates its children only while Mario is within `radius` (default
/// 1500) of an object with one of the behaviours `bhv`; writes nothing itself.
/// For objectives about objects that move: their distance to Mario changes
/// independently of where he is, and many of them at once would make nearly
/// every frame a new state. Under a `Near` only the ones he is close to count.
///
/// ```json
/// {"kind": "Near", "params": {"bhv": ["bhvBobomb"], "radius": 1500}, "children": [{"kind": "BobombThrow"}]}
/// ```
#[derive(Deserialize)]
pub struct Near {
    pub bhv: Vec<String>,
    #[serde(default = "near_radius")]
    pub radius: f32,
    #[serde(skip)]
    addrs: std::sync::OnceLock<Vec<u32>>,
}

fn near_radius() -> f32 {
    1500.0
}

impl Objective for Near {
    fn fields(&self) -> &[&'static str] {
        &[]
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        let addrs = self.addrs.get_or_init(|| self.bhv.iter().filter_map(|b| bhv_addr(b)).collect());
        let objs = st.objects.as_deref().map(|v| v.as_slice()).unwrap_or(&[]);
        if objs.iter().any(|o| addrs.contains(&o.bhv) && dist(o.pos, st.pos) <= self.radius) {
            cx.all_children();
        }
    }
}

/// An optional place for an objective to apply: a level (and area), a sphere
/// (`center`/`radius`) or a box (`min`/`max`); all that are given must hold.
#[derive(Deserialize, Default, Clone)]
pub struct Gate {
    pub level: Option<u16>,
    pub area: Option<u8>,
    pub center: Option<[f32; 3]>,
    pub radius: Option<f32>,
    pub min: Option<[f32; 3]>,
    pub max: Option<[f32; 3]>,
}

impl Gate {
    fn holds(&self, st: &St) -> bool {
        self.level.is_none_or(|l| l == st.level)
            && self.area.is_none_or(|a| a == st.area)
            && match (self.center, self.radius) {
                (Some(c), Some(r)) => dist(st.pos, c) < r,
                _ => true,
            }
            && match (self.min, self.max) {
                (Some(lo), Some(hi)) => in_box(st.pos, lo, hi),
                _ => true,
            }
    }
}

// ---------------------------------------------------------------- Still

/// Standing still (`Mario::Settle`, `Mario::Wait`): `still` 0/1, and while
/// still, `wait`, the frames spent still (action timer) in buckets of
/// `wait_step` (0 leaves it out). Within `gate` (anywhere by default).
///
/// `{"kind": "Still", "params": {"level": 9, "wait_step": 30}}`
#[derive(Deserialize)]
pub struct Still {
    #[serde(flatten)]
    pub gate: Gate,
    #[serde(default)]
    pub wait_step: u32,
    /// Largest bucket of `wait` (longer waits share it).
    #[serde(default = "eight")]
    pub wait_max: i32,
}

fn eight() -> i32 {
    8
}

/// rrt `mario/idle.rs::still`.
fn still(st: &St) -> bool {
    st.action & ACT_FLAG_STATIONARY != 0 && st.forward_vel.abs() < 0.1 && st.vel[1].abs() < 0.1
}

impl Objective for Still {
    fn fields(&self) -> &[&'static str] {
        &["still", "wait"]
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if !self.gate.holds(st) {
            return;
        }
        let s = still(st);
        cx.put(0, s as i32);
        if s && self.wait_step > 0 {
            cx.put(1, bucket(st.action_timer as f32, self.wait_step as f32).min(self.wait_max));
        }
        cx.all_children();
    }
}

// ---------------------------------------------------------------- ExitCourse

/// Leaving a course through the pause menu (`Mario::GoToCastle`). Inactive in
/// the castle, its grounds and courtyard. Writes `exit` 1 when Mario's action
/// allows Exit Course (`ACT_FLAG_PAUSE_EXIT`) and, with the pause menu up,
/// `pause` = the cursor line (1 Continue, 2 Exit Course, 3 Camera).
///
/// `{"kind": "ExitCourse", "params": {"min_stars": 1}}`
#[derive(Deserialize, Default)]
pub struct ExitCourse {
    #[serde(flatten)]
    pub gate: Gate,
    /// Only worth it with this many stars (rrt weights it higher after a star).
    #[serde(default)]
    pub min_stars: i16,
}

impl Objective for ExitCourse {
    fn fields(&self) -> &[&'static str] {
        &["exit", "pause"]
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        let l = st.level as u32;
        if l == LEVEL_CASTLE || l == LEVEL_CASTLE_GROUNDS || l == LEVEL_CASTLE_COURTYARD {
            return;
        }
        if st.stars < self.min_stars || !self.gate.holds(st) {
            return;
        }
        if st.menu_mode == MENU_MODE_RENDER_PAUSE_SCREEN as i16 {
            cx.put(1, st.menu_line as i32);
        } else {
            cx.put(0, (st.action & ACT_FLAG_PAUSE_EXIT != 0) as i32);
        }
        cx.all_children();
    }
}

// ---------------------------------------------------------------- PosDist

/// Getting to a point (`Mario::PosDist`): `dist`, the straight-line distance
/// to `point` in buckets of `step` (rrt's `arrival_scale` 500); -1 once within
/// `radius` (rrt 150). Within `gate` (the level/area etc.).
///
/// `{"kind": "PosDist", "params": {"point": [0, 100, 500], "step": 500}}`
#[derive(Deserialize)]
pub struct PosDist {
    pub point: [f32; 3],
    #[serde(default = "d500")]
    pub step: f32,
    #[serde(default = "d150")]
    pub radius: f32,
    /// Distances past this share one bucket (0: no cap).
    #[serde(default)]
    pub max_dist: f32,
    #[serde(flatten)]
    pub gate: Gate,
}

fn d500() -> f32 {
    500.0
}
fn d150() -> f32 {
    150.0
}

impl Objective for PosDist {
    fn fields(&self) -> &[&'static str] {
        &["dist"]
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if !self.gate.holds(st) {
            return;
        }
        let d = dist(st.pos, self.point);
        let d = if self.max_dist > 0.0 { d.min(self.max_dist) } else { d };
        cx.put(0, if d <= self.radius { -1 } else { bucket(d, self.step) });
        cx.all_children();
    }
}

// ---------------------------------------------------------------- Dialog

/// A dialog's answer state (`Dialog::Read/Yes/No`), while one is open and
/// not yet released (`St::in_dialog`), optionally only the dialogs `ids`:
/// `phase` (`gDialogBoxState`: 0 opening, 1 page shown and taking a press,
/// 2 scrolling, 3 closing), `last_page` 0/1 and, on a last page with a
/// choice, `line` (the cursor: 1 yes, 2 no).
///
/// `{"kind": "DialogOption", "params": {"ids": [14]}}`
#[derive(Deserialize, Default)]
pub struct DialogOption {
    #[serde(default)]
    pub ids: Vec<i16>,
}

impl Objective for DialogOption {
    fn fields(&self) -> &[&'static str] {
        &["phase", "last_page", "line"]
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if !st.in_dialog() || (!self.ids.is_empty() && !self.ids.contains(&st.dialog)) {
            return;
        }
        cx.put(0, st.dialog_state as i32);
        cx.put(1, st.dialog_last_page as i32);
        if st.dialog_choice && st.dialog_last_page {
            cx.put(2, st.menu_line as i32);
        }
        cx.all_children();
    }
}

// ---------------------------------------------------------------- Save menu

/// The save menu after an exit star (`Menu::SaveAndContinue/ContinueNoSave`):
/// `shown` (options up: `gDialogBoxState` 1, else coins still counting),
/// `line` (cursor, 1 save and continue, 3 continue without saving) and
/// `ready` (the menu timer > 110: a press is taken).
///
/// `{"kind": "SaveMenuOption"}`
#[derive(Deserialize, Default)]
pub struct SaveMenuOption {}

impl Objective for SaveMenuOption {
    fn fields(&self) -> &[&'static str] {
        &["shown", "line", "ready"]
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if !st.in_save_menu() {
            return;
        }
        let shown = st.dialog_state == 1;
        cx.put(0, shown as i32);
        if shown {
            cx.put(1, st.menu_line as i32);
            cx.put(2, (st.menu_timer > 110) as i32);
        }
        cx.all_children();
    }
}

// ---------------------------------------------------------------- Act select

/// The act select screen (`Menu::SelectAct`): `cursor` (`act_cursor`, the
/// position among the selectable acts) and `ready` (menu timer >= 11: a press
/// is taken).
///
/// `{"kind": "ActSelectOption", "params": {"course": 4}}`
#[derive(Deserialize, Default)]
pub struct ActSelectOption {
    /// Only this course (any by default).
    pub course: Option<i16>,
}

impl Objective for ActSelectOption {
    fn fields(&self) -> &[&'static str] {
        &["cursor", "ready"]
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if !st.act_select || self.course.is_some_and(|c| c != st.course) {
            return;
        }
        cx.put(0, st.act_cursor as i32);
        cx.put(1, (st.menu_timer >= 11) as i32);
        cx.all_children();
    }
}

// ---------------------------------------------------------------- Doors

/// Going through a door (`Door::Enter`): the nearest door Mario can enter
/// gives `dist` (straight line, buckets of `step`, rrt's 400) and `in_door`
/// (in a door action within `door_radius`, rrt 300). Doors are the objects
/// with behaviour `bhv` (one name or a list), or, by default, the ones with
/// door interaction (`INTERACT_DOOR`, `INTERACT_WARP_DOOR`); or a fixed
/// `point`. In the castle a door needs the stars of its first behaviour byte
/// (`bhvDoor`, `bhvStarDoor`) and a key warp door the key its byte names
/// (2: one key, 1: two); with a `point`, `min_stars`/`min_keys` say. Inactive
/// when no enterable door exists.
///
/// `{"kind": "DoorEnter", "params": {"level": 6, "area": 1, "bhv": ["bhvStarDoor"]}}`
#[derive(Deserialize)]
pub struct DoorEnter {
    #[serde(default)]
    pub bhv: Vec<String>,
    pub point: Option<[f32; 3]>,
    #[serde(default)]
    pub min_stars: i16,
    #[serde(default)]
    pub min_keys: u8,
    #[serde(default = "d400")]
    pub step: f32,
    #[serde(default = "d300")]
    pub door_radius: f32,
    /// Distances past this share one bucket.
    #[serde(default = "d4000")]
    pub max_dist: f32,
    #[serde(flatten)]
    pub gate: Gate,
}

fn d400() -> f32 {
    400.0
}
fn d300() -> f32 {
    300.0
}
fn d4000() -> f32 {
    4000.0
}

impl DoorEnter {
    /// The position of the nearest enterable door, if any (door.rs `Enter::new`).
    fn nearest(&self, st: &St) -> Option<[f32; 3]> {
        if let Some(p) = self.point {
            return (st.stars >= self.min_stars && st.keys >= self.min_keys).then_some(p);
        }
        let addrs: Vec<u32> = self.bhv.iter().filter_map(|n| bhv_addr(n)).collect();
        let objs = st.objects.as_deref().map(|v| v.as_slice()).unwrap_or(&[]);
        objs.iter()
            .filter(|o| {
                if self.bhv.is_empty() { o.interact_type & (INTERACT_DOOR | INTERACT_WARP_DOOR) != 0 } else { addrs.contains(&(o.bhv as u32)) }
            })
            .filter(|o| {
                let castle = st.level as u32 == LEVEL_CASTLE;
                let needs = if castle && matches!(o.bhv_name(), "bhvDoor" | "bhvStarDoor") { (o.beh_params >> 24) as i16 } else { 0 };
                let keys = if castle && o.interact_type & INTERACT_WARP_DOOR != 0 {
                    match o.beh_params >> 24 {
                        2 => 1,
                        1 => 2,
                        _ => 0,
                    }
                } else {
                    0
                };
                st.stars >= needs.max(self.min_stars) && st.keys >= keys.max(self.min_keys)
            })
            .map(|o| o.pos)
            .min_by(|a, b| dist(*a, st.pos).total_cmp(&dist(*b, st.pos)))
    }
}

impl Objective for DoorEnter {
    fn fields(&self) -> &[&'static str] {
        &["dist", "in_door"]
    }

    fn needs_objects(&self) -> bool {
        self.point.is_none()
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if !self.gate.holds(st) {
            return;
        }
        let Some(p) = self.nearest(st) else { return };
        let d = dist(st.pos, p);
        cx.put(0, bucket(d.min(self.max_dist), self.step));
        cx.put(1, (ActionGroup::Door.contains(st.action) && d < self.door_radius) as i32);
        cx.all_children();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{key_of, obj, st, with_objects};

    fn k(kind: &str, params: &str, s: &St) -> Vec<(u8, i32)> {
        key_of(&format!(r#"{{"kind":"Root","children":[{{"kind":"{kind}","params":{params}}}]}}"#), 1, s)
    }

    #[test]
    fn still_gate_and_wait() {
        let p = r#"{"level":9,"wait_step":30}"#;
        let mut s = st([0.0; 3], 0.0, 9, 1, 0);
        s.action = 0x0C40_0201; // idle (stationary)
        s.action_timer = 65;
        assert_eq!(k("Still", p, &s), vec![(0, 1), (1, 2)]);
        s.action_timer = 80; // same bucket
        assert_eq!(k("Still", p, &s), vec![(0, 1), (1, 2)]);
        s.action_timer = 95;
        assert_eq!(k("Still", p, &s), vec![(0, 1), (1, 3)]);
        s.forward_vel = 1.0; // moving: not still
        assert_eq!(k("Still", p, &s), vec![(0, 0)]);
        assert!(k("Still", p, &st([0.0; 3], 0.0, 8, 1, 0)).is_empty(), "wrong level");
    }

    #[test]
    fn exit_course() {
        let mut s = st([0.0; 3], 0.0, 9, 1, 0);
        s.action = 0x0C40_0201;
        let idle = k("ExitCourse", "{}", &s);
        s.action = 0x0100_088C; // falling: no pause exit
        let fall = k("ExitCourse", "{}", &s);
        assert_eq!(idle, vec![(0, 1)]);
        assert_eq!(fall, vec![(0, 0)]);
        s.level = 6;
        assert!(k("ExitCourse", "{}", &s).is_empty(), "castle");
        s.level = 9;
        s.menu_mode = 1;
        s.menu_line = 2;
        assert_eq!(k("ExitCourse", "{}", &s), vec![(1, 2)]);
    }

    #[test]
    fn pos_dist_buckets() {
        let p = r#"{"point":[0,0,1000],"step":500,"level":9}"#;
        let at = |z: f32| k("PosDist", p, &st([0.0, 0.0, z], 0.0, 9, 1, 0));
        assert_eq!(at(1000.0 - 149.0), vec![(0, -1)]);
        assert_eq!(at(1000.0 - 151.0), vec![(0, 0)]);
        assert_eq!(at(1000.0 - 499.0), vec![(0, 0)]);
        assert_eq!(at(1000.0 - 501.0), vec![(0, 1)]);
        assert!(k("PosDist", p, &st([0.0; 3], 0.0, 4, 1, 0)).is_empty());
    }

    #[test]
    fn dialog_states() {
        let mut s = st([0.0; 3], 0.0, 9, 1, 0);
        assert!(k("DialogOption", "{}", &s).is_empty(), "no dialog");
        s.dialog = 14;
        s.dialog_state = 1;
        s.dialog_last_page = true;
        s.dialog_choice = true;
        s.menu_line = 1;
        let yes = k("DialogOption", "{}", &s);
        s.menu_line = 2;
        let no = k("DialogOption", "{}", &s);
        assert_eq!(yes, vec![(0, 1), (1, 1), (2, 1)]);
        assert_ne!(yes, no);
        s.dialog_last_page = false; // cursor not meaningful before the last page
        s.menu_line = 1;
        let a = k("DialogOption", "{}", &s);
        s.menu_line = 2;
        assert_eq!(a, k("DialogOption", "{}", &s));
        assert!(k("DialogOption", r#"{"ids":[15]}"#, &s).is_empty(), "other dialog");
        assert!(!k("DialogOption", r#"{"ids":[14]}"#, &s).is_empty());
    }

    #[test]
    fn save_menu_and_act_select() {
        let mut s = st([0.0; 3], 0.0, 9, 1, 0);
        assert!(k("SaveMenuOption", "{}", &s).is_empty());
        s.menu_mode = 20; // MENU_MODE_RENDER_COURSE_COMPLETE_SCREEN
        s.menu_mode = obrrt_core::layout::MENU_MODE_RENDER_COURSE_COMPLETE_SCREEN as i16;
        s.dialog_state = 0;
        assert_eq!(k("SaveMenuOption", "{}", &s), vec![(0, 0)]);
        s.dialog_state = 1;
        s.menu_line = 3;
        s.menu_timer = 111;
        assert_eq!(k("SaveMenuOption", "{}", &s), vec![(0, 1), (1, 3), (2, 1)]);
        s.menu_timer = 100;
        assert_eq!(k("SaveMenuOption", "{}", &s), vec![(0, 1), (1, 3), (2, 0)]);

        let mut a = st([0.0; 3], 0.0, 9, 1, 0);
        assert!(k("ActSelectOption", "{}", &a).is_empty());
        a.act_select = true;
        a.course = 4;
        a.act_cursor = 2;
        a.menu_timer = 11;
        assert_eq!(k("ActSelectOption", "{}", &a), vec![(0, 2), (1, 1)]);
        assert!(k("ActSelectOption", r#"{"course":5}"#, &a).is_empty());
    }

    #[test]
    fn door_nearest_and_requirements() {
        let mut door = obj("bhvDoor", 0, [0.0, 0.0, 1000.0]);
        door.interact_type = INTERACT_DOOR;
        door.beh_params = 3 << 24; // needs 3 stars in the castle
        let far = obj("bhvDoor", 1, [0.0, 0.0, 5000.0]);
        let mut far = far;
        far.interact_type = INTERACT_DOOR;
        let s = |level: u16, stars: i16, z: f32| {
            let mut s = with_objects(st([0.0, 0.0, z], 0.0, level, 1, 0), vec![door.clone(), far.clone()]);
            s.stars = stars;
            s
        };
        // Elsewhere than the castle the star byte is ignored.
        assert_eq!(k("DoorEnter", "{}", &s(9, 0, 0.0)), vec![(0, 2), (1, 0)]);
        // In the castle without the stars the near door doesn't count: the far one does.
        assert_eq!(k("DoorEnter", "{}", &s(6, 0, 0.0)), vec![(0, 10), (1, 0)]); // 5000 capped to 4000 -> 10
        assert_eq!(k("DoorEnter", "{}", &s(6, 3, 0.0)), vec![(0, 2), (1, 0)]);
        // In a door action close to the door.
        let mut near = s(6, 3, 900.0);
        near.action = 0x0000_1320;
        assert_eq!(k("DoorEnter", "{}", &near), vec![(0, 0), (1, 1)]);
        // A fixed point, with a requirement.
        let p = r#"{"point":[0,0,1000],"min_stars":5}"#;
        assert!(k("DoorEnter", p, &s(6, 3, 0.0)).is_empty());
        assert_eq!(k("DoorEnter", p, &s(6, 5, 0.0)), vec![(0, 2), (1, 0)]);
        // No door at all: silent.
        assert!(k("DoorEnter", "{}", &st([0.0; 3], 0.0, 9, 1, 0)).is_empty());
    }

    #[test]
    fn near_gates_its_children_on_distance_to_the_objects() {
        use crate::testutil::{obj, st, with_objects};
        let json = r#"{"kind": "Near", "params": {"bhv": ["bhvBobomb"], "radius": 1000},
            "children": [{"kind": "Still", "params": {}}]}"#;
        let t = crate::objective::Tree::load(json).unwrap();
        let mut s = st([0.0, 0.0, 0.0], 0.0, 9, 1, 0);
        s.action = 0x0C40_0201;
        let near = with_objects(s.clone(), vec![obj("bhvBobomb", 1, [500.0, 0.0, 0.0])]);
        let far = with_objects(s.clone(), vec![obj("bhvBobomb", 1, [5000.0, 0.0, 0.0])]);
        let other = with_objects(s, vec![obj("bhvKoopa", 1, [100.0, 0.0, 0.0])]);
        assert!(!t.key(&near).is_empty(), "the child speaks near a bob-omb");
        assert!(t.key(&far).is_empty() && t.key(&other).is_empty(), "silent when far or another object");
    }
}
