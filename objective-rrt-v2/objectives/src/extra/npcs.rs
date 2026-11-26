//! NPCs and dialog: signs and message panels (read), MIPS the rabbit
//! (catch), Koopa the Quick (race to the flag, then claim the star), Tuxie's
//! mother and baby penguin (carry back), the big penguin race (talk, win,
//! claim), the snowman's head (wake the body, lead it, talk to the head),
//! Hoot the owl (wake, ride), the cage Ukiki (catch, follow to the cage),
//! Yoshi, and the intro Lakitu.
//!
//! Talking to Bob-omb Buddies, Toads and Koopa the Quick before his race is
//! `Npc::Talk` (parent module) and is not repeated here. Star-giving Toads
//! need nothing more: the talk spawns the star (mario_misc.c,
//! toad_message_talking) and `Star::Collect` takes it. Grabbing a holdable
//! NPC is also offered by `Object::Grab` at weight 1; `Mips::Catch` and
//! `Ukiki::Catch` are the same act with the star's weight and a chase.
//!
//! Dialogs are left to `Dialog::*`: an objective here succeeds when its
//! dialog has started (or when the NPC's state shows its effect).
//!
//! Decomp references are to sm64-binds/sm64-port.

use std::f32::consts::TAU;
use std::sync::Arc;

use obrrt_core::buttons::B_BUTTON;
use obrrt_core::layout::*;
use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St, dist, frames_for};

/// include/sm64.h: `ACT_RIDING_HOOT` (0x0A8 | ACT_FLAG_MOVING); not in the
/// generated layout.
const ACT_RIDING_HOOT: u32 = 0x0000_04A8;

// oAction values, include/object_constants.h.
const KTQ_ACT_RACE: i32 = 3;
const KTQ_ACT_AFTER_RACE: i32 = 6;
const RACING_PENGUIN_ACT_WAIT_FOR_MARIO: i32 = 0;
const RACING_PENGUIN_ACT_PREPARE_FOR_RACE: i32 = 2;
const RACING_PENGUIN_ACT_RACE: i32 = 3;
const RACING_PENGUIN_ACT_FINISH_RACE: i32 = 4;
const RACING_PENGUIN_ACT_SHOW_FINAL_TEXT: i32 = 5;
const MIPS_ACT_IDLE: i32 = 4;
const UKIKI_ACT_GO_TO_CAGE: i32 = 4;
const UKIKI_ACT_WAIT_TO_RESPAWN: i32 = 5;
const YOSHI_ACT_IDLE: i32 = 0;
const YOSHI_ACT_WALK: i32 = 1;
const CAMERA_LAKITU_INTRO_ACT_TRIGGER_CUTSCENE: i32 = 0;
/// oBehParams2ndByte of the intro Lakitu (CAMERA_LAKITU_BP_INTRO).
const CAMERA_LAKITU_BP_INTRO: u32 = 1;
/// camera_lakitu.inc.c, camera_lakitu_intro_act_trigger_cutscene: Mario on
/// the bridge (-544 < x < 545, y > 800, -2000 < z < -177) starts the intro.
const LAKITU_BRIDGE: [f32; 3] = [0.0, 850.0, -1100.0];
/// oBehParams2ndByte of the cage Ukiki (UKIKI_CAGE; UKIKI_CAP steals caps).
const UKIKI_CAGE: u32 = 0;

/// snowman.inc.c, snowmans_bottom_act_2: the body stops once within 300 of
/// this point (and the head drops onto it).
const SNOWMAN_SITE: [f32; 3] = [-4230.0, -1344.0, 1813.0];

fn byte0(o: &Obj) -> u8 {
    (o.beh_params >> 24) as u8
}

fn byte1(o: &Obj) -> u32 {
    (o.beh_params >> 16) & 0xFF
}

fn find_bhv<'a>(s: &'a St, name: &str) -> Option<&'a Obj> {
    s.objects().iter().find(|o| o.bhv_name() == name)
}

/// The nearest object with behaviour `name` to `p`, as the game's
/// `cur_obj_nearest_object_with_behavior` would pick it.
fn nearest_bhv<'a>(s: &'a St, name: &str, p: [f32; 3]) -> Option<&'a Obj> {
    s.objects().iter().filter(|o| o.bhv_name() == name).min_by(|a, b| dist(a.pos, p).total_cmp(&dist(b.pos, p)))
}

/// Mario talking or about to (`ACT_WAITING_FOR_DIALOG` / NPC dialog), or a
/// dialog box open.
fn talking(s: &St) -> bool {
    s.in_dialog() || matches!(s.action, ACT_WAITING_FOR_DIALOG | ACT_READING_NPC_DIALOG | ACT_READING_AUTOMATIC_DIALOG)
}

fn star_missing(s: &St, star: Option<(i16, u8)>) -> bool {
    star.is_none_or(|(c, i)| !s.has_star(c, i))
}

// ---------------------------------------------------------------------------
// Approach: go to one NPC (in a given state) until something happens.
// ---------------------------------------------------------------------------

/// What counts as done for an [`Approach`]; any one of a spec's list.
#[derive(Clone, Copy)]
enum Done {
    /// Mario talking (or a dialog open) within this distance of the object.
    DialogNear(f32),
    /// The object's `oAction` differs from the start's.
    ActionLeaves,
    /// Mario within this distance of the object.
    Within(f32),
    /// Mario holds the object.
    Held,
    /// Mario in this action within this distance of the object.
    MarioAction(u32, f32),
}

/// Where the controller-seeded genomes head.
#[derive(Clone, Copy)]
enum Aim {
    /// The object.
    At,
    /// In front of the object (along its facing yaw) at this distance, then
    /// straight at it: for things read or talked to only while facing them.
    Front(f32),
    /// A fixed point.
    Point([f32; 3]),
    /// This far from the object, on Mario's side (it hurts to touch).
    Near(f32),
}

struct Spec {
    kind: &'static str,
    weight: f64,
    value: f64,
    /// The object's `oAction` at the start must be one of these (any if
    /// empty).
    actions: &'static [i32],
    /// Further start condition on the object.
    extra: Option<fn(&Obj) -> bool>,
    min_stars: i16,
    not_holding: bool,
    done: &'static [Done],
    aim: Aim,
    /// Pressed on alternate frames within `button_radius` of the object.
    button: u16,
    button_radius: f32,
    /// v1's GA budget (population, generations, horizon).
    budget: (usize, usize, usize),
}

struct Approach {
    spec: &'static Spec,
    id: ObjId,
    zone: (u16, u8),
    star: Option<(i16, u8)>,
}

impl Approach {
    fn push(out: &mut Vec<Arc<dyn Objective>>, spec: &'static Spec, b: &St, o: &Obj, star: Option<(i16, u8)>) {
        let a = Approach { spec, id: o.id, zone: b.zone(), star };
        if a.startable(o) && b.stars >= spec.min_stars && star_missing(b, star) {
            out.push(Arc::new(a));
        }
    }

    fn startable(&self, o: &Obj) -> bool {
        (self.spec.actions.is_empty() || self.spec.actions.contains(&o.action)) && self.spec.extra.is_none_or(|f| f(o))
    }

    fn target(&self, s: &St, o: &Obj) -> [f32; 3] {
        match self.spec.aim {
            Aim::At => o.pos,
            Aim::Point(p) => p,
            Aim::Near(r) => {
                let (dx, dz) = (s.pos[0] - o.pos[0], s.pos[2] - o.pos[2]);
                let l = (dx * dx + dz * dz).sqrt().max(1.0);
                [o.pos[0] + dx / l * r, o.pos[1], o.pos[2] + dz / l * r]
            }
            Aim::Front(d) => {
                let a = o.yaw as f32 * TAU / 65536.0;
                let (fx, fz) = (a.sin(), a.cos());
                let (rx, rz) = (s.pos[0] - o.pos[0], s.pos[2] - o.pos[2]);
                let along = rx * fx + rz * fz;
                let side = (rx * fz - rz * fx).abs();
                if along > 60.0 && side < along * 0.4 {
                    o.pos
                } else {
                    [o.pos[0] + d * fx, o.pos[1], o.pos[2] + d * fz]
                }
            }
        }
    }

    /// The castle (inside, grounds, courtyard): rooms and corridors, where the straight line to a Toad or MIPS
    /// goes through walls and floors (Toad's upstairs star: 150 M each at t1 steps 188 and 195, none; MIPS 2:
    /// 130 M). With `OBRRT_GEO=1` the ETA is the walking distance and the steering a point along the walk
    /// (as the races do); off, or outside the castle, the straight line as before.
    fn in_castle(s: &St) -> bool {
        matches!(s.zone().0, 6 | 16 | 26)
    }

    fn walk(s: &St, t: [f32; 3]) -> f32 {
        if Self::in_castle(s) { obrrt_core::geo::path_dist(s.zone(), s.pos, t).unwrap_or_else(|| s.dist(t)) } else { s.dist(t) }
    }

    fn met(&self, s0: &St, s: &St) -> bool {
        let Some(o) = s.find(self.id) else { return false };
        let d = s.dist(o.pos);
        self.spec.done.iter().any(|done| match *done {
            Done::DialogNear(r) => d < r && talking(s),
            Done::Within(r) => d < r,
            Done::ActionLeaves => s0.find(self.id).is_some_and(|o0| o0.action != o.action),
            Done::Held => s.held().is_some_and(|h| h.id == self.id),
            Done::MarioAction(a, r) => s.action == a && d < r,
        })
    }
}

impl Objective for Approach {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn precision(&self) -> obrrt_core::Precision {
        if matches!(self.kind(), "KoopaRace::Claim" | "PenguinRace::Claim" | "Mips::Catch" | "Ukiki::Catch") { obrrt_core::Precision::Precise } else { obrrt_core::Precision::Explore }
    }
    fn kind(&self) -> &'static str {
        self.spec.kind
    }
    fn key(&self) -> String {
        format!("{} {:?} {}", self.spec.kind, self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        self.spec.weight
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && s.stars >= self.spec.min_stars
            && !(self.spec.not_holding && s.holding)
            && star_missing(s, self.star)
            && s.find(self.id).is_some_and(|o| self.startable(o))
    }
    /// The way to the aim point, then (for an aim off the object, e.g. in
    /// front of a sign, not one beside it) on to the object; `Done` where v1's success held
    /// (any `done` condition, on a frame after the start); the NPC gone or
    /// the zone left: `Lost`.
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        let (population, generations, horizon) = self.spec.budget;
        Some(obrrt_core::GaBudget { population, generations, horizon })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        // v1 looked from the frame after the start on.
        if s.frame != start.frame && self.met(start, s) {
            return Eta::Done;
        }
        let Some(o) = s.find(self.id) else { return Eta::Lost };
        let t = self.target(s, o);
        // An aim beside a thing that hurts to touch (`Near`) is the goal
        // itself: its done conditions trigger from there.
        let on = if matches!(self.spec.aim, Aim::Near(_)) { 0.0 } else { dist(t, o.pos) };
        Eta::Frames(frames_for(Self::walk(s, t) + on))
    }
    fn success_value(&self) -> f64 {
        self.spec.value
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        obrrt_core::Arrival::Stop
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        s.find(self.id).map(|o| {
            let t = self.target(s, o);
            // Along the walk while far (the aim point itself within 700, where the straight line is the walk).
            if Self::in_castle(s) && s.dist(t) > 700.0 {
                obrrt_core::geo::waypoint(s.zone(), s.pos, t, 700.0).unwrap_or(t)
            } else {
                t
            }
        })
    }
    fn steer_button(&self, s: &St) -> u16 {
        match s.find(self.id) {
            Some(o) if self.spec.button != 0 && s.dist(o.pos) < self.spec.button_radius => {
                // B while holding throws; never for a catch already made.
                if self.spec.not_holding && s.holding { 0 } else { self.spec.button }
            }
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && s.find(self.id).is_some()
    }
}

/// `Sign::Read`: signposts and wall signs (bhvMessagePanel, bhvSignOnWall,
/// INT_SUBTYPE_SIGN). interaction.c, check_read_sign: B (or A) while
/// touching it, Mario facing it and it facing Mario (SIGN_RANGE 0x4000).
/// Hints only.
static SIGN_READ: Spec = Spec {
    kind: "Sign::Read",
    weight: 0.15,
    value: 0.1,
    actions: &[],
    extra: None,
    min_stars: 0,
    not_holding: false,
    done: &[Done::MarioAction(ACT_READING_SIGN, 400.0)],
    aim: Aim::Front(220.0),
    button: B_BUTTON,
    button_radius: 220.0,
    budget: (12, 4, 120),
};

/// `Yoshi::Talk`: Yoshi on the castle roof (120 stars; yoshi.inc.c). An
/// INT_SUBTYPE_NPC like the Bob-omb Buddies; gives 100 lives.
static YOSHI_TALK: Spec = Spec {
    kind: "Yoshi::Talk",
    weight: 1.0,
    value: 1.0,
    actions: &[YOSHI_ACT_IDLE, YOSHI_ACT_WALK],
    extra: None,
    min_stars: 0,
    not_holding: false,
    done: &[Done::DialogNear(400.0)],
    aim: Aim::At,
    button: B_BUTTON,
    button_radius: 200.0,
    budget: (12, 4, 150),
};

/// `Lakitu::Intro`: the castle grounds' intro Lakitu (only before the castle
/// is first entered, camera_lakitu.inc.c bhv_camera_lakitu_init) starts his
/// dialog when Mario stands on the bridge.
static LAKITU_INTRO: Spec = Spec {
    kind: "Lakitu::Intro",
    weight: 0.3,
    value: 0.3,
    actions: &[CAMERA_LAKITU_INTRO_ACT_TRIGGER_CUTSCENE],
    extra: Some(|o| byte1(o) == CAMERA_LAKITU_BP_INTRO),
    min_stars: 0,
    not_holding: false,
    done: &[Done::ActionLeaves],
    aim: Aim::Point(LAKITU_BRIDGE),
    button: 0,
    button_radius: 0.0,
    budget: (12, 4, 150),
};

/// `Mips::Catch`: MIPS in the castle basement (only spawned with 15 / 50
/// stars and his star not yet collected, mips.inc.c bhv_mips_init). He runs
/// at 40-45 u/f, so he is caught by a dive (B at speed; interaction.c
/// able_to_grab_object allows ACT_DIVE). Held, he starts his dialog; after
/// it Mario drops him and he spawns the star (bhv_mips_act_idle).
static MIPS_CATCH: Spec = Spec {
    kind: "Mips::Catch",
    weight: 15.0,
    value: 20.0,
    actions: &[],
    // After the catch he idles (MIPS_ACT_IDLE) and gives nothing more.
    extra: Some(|o| o.action != MIPS_ACT_IDLE && o.held_state == 0),
    min_stars: 15,
    not_holding: true,
    // Met when his dialog opens (Mario holds him, so he is within 400), as Toad's talk is: a state that only holds
    // him (Done::Held) is controllable and idle, nothing then leads on to the dialog (mips-new, 2026-11-14: 69 of 97
    // catches, none went on; the held state's next 40 frames are the dialog, then `Dialog::Read` takes over).
    done: &[Done::DialogNear(400.0)],
    aim: Aim::At,
    button: B_BUTTON,
    button_radius: 250.0,
    budget: (24, 6, 240),
};

/// `Ukiki::Catch`: the cage Ukiki on TTM's summit (bhvUkiki, 2nd byte
/// UKIKI_CAGE, act 2). Held, he asks to be let go (DIALOG_079, yes/no; yes
/// sends him to the cage), ukiki.inc.c cage_ukiki_held_loop.
static UKIKI_CATCH: Spec = Spec {
    kind: "Ukiki::Catch",
    weight: 10.0,
    value: 8.0,
    actions: &[],
    extra: Some(|o| {
        byte1(o) == UKIKI_CAGE && o.held_state == 0 && !matches!(o.action, UKIKI_ACT_GO_TO_CAGE | UKIKI_ACT_WAIT_TO_RESPAWN)
    }),
    min_stars: 0,
    not_holding: true,
    done: &[Done::Held],
    aim: Aim::At,
    button: B_BUTTON,
    button_radius: 220.0,
    budget: (20, 5, 240),
};

/// `Ukiki::FollowToCage`: after letting him go he walks to the cage and
/// waits; Mario within 200 (laterally) starts DIALOG_080, then he jumps on
/// the cage, it falls and breaks and the star appears (ukiki.inc.c
/// ukiki_act_go_to_cage, ukiki_cage.inc.c).
static UKIKI_FOLLOW: Spec = Spec {
    kind: "Ukiki::FollowToCage",
    weight: 12.0,
    value: 10.0,
    actions: &[UKIKI_ACT_GO_TO_CAGE],
    extra: Some(|o| byte1(o) == UKIKI_CAGE),
    min_stars: 0,
    not_holding: false,
    done: &[Done::DialogNear(400.0)],
    aim: Aim::At,
    button: 0,
    button_radius: 0.0,
    budget: (16, 5, 300),
};

/// `Hoot::Wake`: Hoot sleeps in his WF tree, invisible, until Mario is
/// within 50 of him (hoot.inc.c bhv_hoot_loop, HOOT_AVAIL_ASLEEP_IN_TREE);
/// then he talks (DIALOG_044) as soon as Mario can, and becomes tangible.
static HOOT_WAKE: Spec = Spec {
    kind: "Hoot::Wake",
    weight: 3.0,
    value: 2.0,
    actions: &[],
    extra: Some(hoot_asleep),
    min_stars: 0,
    not_holding: false,
    done: &[Done::Within(50.0), Done::DialogNear(600.0)],
    aim: Aim::At,
    button: 0,
    button_radius: 0.0,
    budget: (16, 5, 240),
};

/// `Hoot::Ride`: once awake and talked to he is tangible
/// (cur_obj_become_tangible: oIntangibleTimer 0); jumping into him (an
/// airborne action 0x080-0x097) grabs on, interaction.c interact_hoot.
static HOOT_RIDE: Spec = Spec {
    kind: "Hoot::Ride",
    weight: 4.0,
    value: 2.0,
    actions: &[],
    extra: Some(|o| o.intangible_timer == 0),
    min_stars: 0,
    not_holding: true,
    done: &[Done::MarioAction(ACT_RIDING_HOOT, 400.0)],
    aim: Aim::At,
    button: 0,
    button_radius: 0.0,
    budget: (20, 5, 240),
};

/// Asleep in the tree: still where he spawned. bhv_hoot_init sets oHome to
/// pos + (800, -150, 300); awake, he flies around that home.
fn hoot_asleep(o: &Obj) -> bool {
    o.intangible_timer != 0 && dist(o.pos, [o.home[0] - 800.0, o.home[1] + 150.0, o.home[2] - 300.0]) < 2.0
}

/// `KoopaRace::Claim`: after the race Koopa the Quick waits
/// (KOOPA_THE_QUICK_ACT_AFTER_RACE) for Mario within 400 (no button,
/// cur_obj_can_mario_activate_textbox_2); if Mario won, the star spawns
/// after the dialog (koopa.inc.c koopa_the_quick_act_after_race).
static KTQ_CLAIM: Spec = Spec {
    kind: "KoopaRace::Claim",
    weight: 12.0,
    value: 5.0,
    actions: &[KTQ_ACT_AFTER_RACE],
    extra: None,
    min_stars: 0,
    not_holding: false,
    done: &[Done::DialogNear(600.0)],
    aim: Aim::At,
    button: 0,
    button_radius: 0.0,
    budget: (16, 5, 240),
};

/// `PenguinRace::Talk`: the big penguin atop CCM's slide proposes the race
/// when Mario is within 400 and not below him (racing_penguin.inc.c
/// racing_penguin_act_wait_for_mario); `Dialog::Yes` starts it.
static PENGUIN_TALK: Spec = Spec {
    kind: "PenguinRace::Talk",
    weight: 8.0,
    value: 3.0,
    actions: &[RACING_PENGUIN_ACT_WAIT_FOR_MARIO],
    extra: None,
    min_stars: 0,
    not_holding: false,
    // Not `ActionLeaves`: the penguin leaves WAIT_FOR_MARIO (-> SHOW_INIT_TEXT) several frames BEFORE Mario's
    // dialog opens, so a success there was a walking state and `Dialog::Yes` (only offered from a state with
    // the dialog open) had no base: ccm2-a, 17 Talk successes in 1500 nodes, 0 Dialog::Yes attempts.
    done: &[Done::DialogNear(600.0)],
    aim: Aim::At,
    button: 0,
    button_radius: 0.0,
    budget: (12, 4, 150),
};

/// The racing penguin's oRacingPenguinMarioWon and oRacingPenguinMarioCheated
/// (racing_penguin.inc.c): Mario got past the finish line first, and was
/// more than 60 frames in the air while racing or near the shortcut
/// (bhv_penguin_race_shortcut_check_update). A cheated win only gets
/// DIALOG_132 and no star.
fn penguin_won(o: &Obj) -> bool {
    o.raw_1ac[1] != 0
}

fn penguin_cheated(o: &Obj) -> bool {
    o.raw_1ac[3] != 0
}

/// `PenguinRace::Claim`: at the bottom he waits (SHOW_FINAL_TEXT) for Mario
/// within 400; if Mario won fairly the star spawns after the dialog. Only
/// offered after a fair win: in v32-chomp-s3-1b all 7 claims were of cheated
/// wins (DIALOG_132).
static PENGUIN_CLAIM: Spec = Spec {
    kind: "PenguinRace::Claim",
    weight: 12.0,
    value: 5.0,
    actions: &[RACING_PENGUIN_ACT_SHOW_FINAL_TEXT],
    extra: Some(|o| penguin_won(o) && !penguin_cheated(o)),
    min_stars: 0,
    not_holding: false,
    done: &[Done::DialogNear(600.0)],
    aim: Aim::At,
    button: 0,
    button_radius: 0.0,
    budget: (16, 5, 240),
};

/// `Snowman::WakeBody`: the snowman's body (bhvSnowmansBottom, CCM act 5)
/// talks (DIALOG_110) when Mario is within 400, then starts rolling
/// (snowman.inc.c bhv_snowmans_bottom_loop, action 0 -> 1). Touching it
/// hurts (sRollingSphereHitbox, INTERACT_DAMAGE 3), so aim beside it.
static SNOWMAN_WAKE: Spec = Spec {
    kind: "Snowman::WakeBody",
    weight: 8.0,
    value: 5.0,
    actions: &[0],
    extra: None,
    min_stars: 0,
    not_holding: false,
    done: &[Done::ActionLeaves, Done::DialogNear(600.0)],
    aim: Aim::Near(300.0),
    button: 0,
    button_radius: 0.0,
    budget: (12, 4, 150),
};

/// `Snowman::Finish`: with the body under it the head waits (action 4) for
/// Mario within 700, in front of it and facing it
/// (obj_behaviors.c trigger_obj_dialog_when_facing); after DIALOG_111 the
/// star spawns (bhv_snowmans_head_loop).
static SNOWMAN_FINISH: Spec = Spec {
    kind: "Snowman::Finish",
    weight: 15.0,
    value: 10.0,
    // 2 and 3 too: the body's success state (it is under the head, action 3/4) is when the head is still
    // flying (action 2, then 3 on the body) and reaches action 4 some 30-60 frames later; `ActionLeaves`
    // makes those steps bases, so a state with the head at 4 exists for the dialog.
    actions: &[2, 3, 4],
    extra: None,
    min_stars: 0,
    not_holding: false,
    done: &[Done::ActionLeaves, Done::DialogNear(900.0)],
    aim: Aim::Front(500.0),
    button: 0,
    button_radius: 0.0,
    budget: (16, 5, 200),
};

// ---------------------------------------------------------------------------
// Races: reach the goal before the racer.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum RaceKind {
    /// Koopa the Quick (BoB, THI) to bhvKoopaRaceEndpoint.
    Koopa,
    /// The big penguin down CCM's slide to bhvPenguinRaceFinishLine.
    Penguin,
}

/// racing_penguin.inc.c racing_penguin_act_race: Mario in an air action
/// with the penguin's oTimer (reset on every grounded frame) past 60 marks
/// the race cheated. Counted here from the start (frames airborne before it
/// unseen, so never more than the game's), plus 2 frames of margin for the
/// timer's off-by-one: `Lost` never comes before the game's flag.
const PENGUIN_AIR_LIMIT: f64 = 62.0;

/// bhvPenguinRaceShortcutCheck (levels/ccm/script.c): Mario within 500 of it
/// cheats (racing_penguin.inc.c bhv_penguin_race_shortcut_check_update).
const PENGUIN_SHORTCUT: [f32; 3] = [-6393.0, -716.0, 7503.0];

/// Slide speed (units a frame) the penguin ETA assumes for Mario; the
/// penguin's own is 70..150 (racing_penguin_act_race).
const PENGUIN_SLIDE_SPEED: f32 = 80.0;
/// The finish line wins at distance < 990 AND z below the line's (-6400: `won`); the path's last leg runs
/// along x -6488 from z -6088, so only its last ~310 units are still needed (was 990: the ETA reached 0 ~700
/// units before the win).
const PENGUIN_WIN_RADIUS: f32 = 310.0;

/// `mem.f[0]` of a penguin race: consecutive airborne frames while racing.
const AIR_FRAMES: usize = 0;

struct Race {
    race: RaceKind,
    racer: ObjId,
    goal: ObjId,
    zone: (u16, u8),
    star: Option<(i16, u8)>,
}

impl Race {
    /// Racing, and (the penguin) not yet cheated: a cheated race can't be
    /// won (v32-chomp-s3-1b "won" 96% of its penguin races, all cheated).
    fn racing(&self, r: &Obj) -> bool {
        match self.race {
            RaceKind::Koopa => r.action == KTQ_ACT_RACE,
            RaceKind::Penguin => {
                matches!(r.action, RACING_PENGUIN_ACT_PREPARE_FOR_RACE | RACING_PENGUIN_ACT_RACE) && !penguin_cheated(r)
            }
        }
    }

    fn both<'a>(&self, s: &'a St) -> Option<(&'a Obj, &'a Obj)> {
        Some((s.find(self.racer)?, s.find(self.goal)?))
    }

    /// Mario has won in `s`.
    fn won(&self, s: &St) -> bool {
        let Some((r, g)) = self.both(s) else { return false };
        if !self.racing(r) {
            return false;
        }
        match self.race {
            // koopa.inc.c bhv_koopa_race_endpoint_update: Mario within 400 of
            // the endpoint before KtQ is (bhv_koopa_update marks him finished
            // within 400 of it). Using the cannon makes it a cheat.
            RaceKind::Koopa => s.dist(g.pos) < 390.0 && dist(r.pos, g.pos) >= 400.0,
            // racing_penguin.inc.c bhv_penguin_race_finish_line_update: Mario
            // within 1000 and past it (z below it) before the penguin reaches
            // the end of his path (he then leaves the race actions). More than
            // 60 frames airborne or the shortcut make it a cheat.
            RaceKind::Penguin => s.dist(g.pos) < 990.0 && s.pos[2] < g.pos[2],
        }
    }
}

impl Objective for Race {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        match self.race {
            RaceKind::Koopa => "KoopaRace::ReachFlag",
            RaceKind::Penguin => "PenguinRace::Win",
        }
    }
    fn key(&self) -> String {
        format!("{} {:?} {}", self.kind(), self.zone, self.racer)
    }
    /// The penguin race is a long chain (a descent of 80000 units, ~1000 frames, seen only through 600-900 frame
    /// attempts) that the tree must keep extending: 5x the Koopa race's weight (ccm2-f: at 20 only 29 Win
    /// attempts in 40 M frames against 170 Talk; from a half-way state a win took 31 attempts, ccm2-e).
    fn obj_weight(&self, _b: &St) -> f64 {
        match self.race {
            RaceKind::Koopa => 20.0,
            RaceKind::Penguin => 100.0,
        }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && star_missing(s, self.star)
            // Already won (the flag set): `PenguinRace::Await` is the next step, not the race again.
            && self.both(s).is_some_and(|(r, _)| self.racing(r) && !(matches!(self.race, RaceKind::Penguin) && penguin_won(r)))
    }
    /// Frames to the goal; `Done` on a won frame after the start (v1's
    /// success); the race over (the racer out of his race action: he got
    /// there first, or the penguin's cheat flag set), racer or goal gone, or
    /// the zone left: `Lost`. The penguin race also keeps its airborne window
    /// in `mem` ([`PENGUIN_AIR_LIMIT`]): past it the race is cheated.
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: if matches!(self.race, RaceKind::Penguin) { 900 } else { 600 } })
    }
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        if let RaceKind::Penguin = self.race {
            let in_race = self.both(s).is_some_and(|(r, _)| r.action == RACING_PENGUIN_ACT_RACE);
            mem.f[AIR_FRAMES] = if in_race && s.action_has(ACT_FLAG_AIR) { mem.f[AIR_FRAMES] + 1.0 } else { 0.0 };
            if mem.f[AIR_FRAMES] > PENGUIN_AIR_LIMIT {
                return Eta::Lost;
            }
        }
        if s.frame != start.frame && self.won(s) {
            return Eta::Done;
        }
        if let RaceKind::Penguin = self.race {
            // The shortcut cheats the race.
            if dist(s.pos, PENGUIN_SHORTCUT) < 500.0 {
                return Eta::Lost;
            }
            // On the slide: the length left along the penguin's own path (the walkable mesh doesn't
            // cover the steep slide floors, so the walking distance gave 54 frames of progress).
            if let Some((r, _)) = self.both(s).filter(|(r, _)| self.racing(r)) {
                // Behind the penguin counts: once he is at the bottom unbeaten the race is lost, and every unit Mario
                // trails him is paid (t1 step 838: 0 wins in 213; its best 'ETA 15' state had the penguin 47 past
                // the finish line, MarioWon 0, and the tree kept building on lost races).
                let pp = super::slides::ccm().progress(r.pos);
                let pr = super::slides::ccm().progress(s.pos);
                if pp.remaining < PENGUIN_WIN_RADIUS && !penguin_won(r) {
                    return Eta::Lost;
                }
                let behind = (pr.remaining - pp.remaining).max(0.0);
                return Eta::Frames((((pr.remaining - PENGUIN_WIN_RADIUS).max(0.0) + 2.0 * pr.off + behind) / PENGUIN_SLIDE_SPEED) as f64);
            }
            if false {
                // Always the path, never the walking distance: at the top plateau farther than
                // 1500 units (the old reach) from the path's start (x -6298) the geodesic gave ETA 423 against 1000 a few
                // steps east (and 580 for a state halfway down the slide), so the best-ETA bases were idle states at
                // the top (ccm2-b: Win attempts from them made 1-8 progress; the halfway states were never started).
                let pr = super::slides::ccm().progress(s.pos);
                return Eta::Frames((((pr.remaining - PENGUIN_WIN_RADIUS).max(0.0) + 2.0 * pr.off) / PENGUIN_SLIDE_SPEED) as f64);
            }
        }
        match self.both(s) {
            // OBRRT_GEO=1: the walking distance (KtQ's route spirals up BoB's mountain; the straight line
            // to the flag rated the foot of the mountain close, and t1's BoB trees never won in 8 checkpoints).
            Some((r, g)) if self.racing(r) => {
                Eta::Frames(frames_for(obrrt_core::geo::path_dist(s.zone(), s.pos, g.pos).unwrap_or_else(|| s.dist(g.pos))))
            }
            _ => Eta::Lost,
        }
    }
    fn success_value(&self) -> f64 {
        20.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // The penguin race, on the slide: 700 units along the penguin's path.
        if let RaceKind::Penguin = self.race {
            return Some(super::slides::ccm().ahead(s.pos, 700.0));
        }
        // OBRRT_GEO=1: a point 700 units along the walk to the flag, else the flag.
        let g = s.find(self.goal)?.pos;
        Some(obrrt_core::geo::waypoint(s.zone(), s.pos, g, 700.0).unwrap_or(g))
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && self.both(s).is_some()
    }
    fn kinship(&self) -> u32 {
        10
    }
}

/// `bhvKoopaRaceEndpoint`'s `value` once Mario has won and the star is still to come (core/src/read.rs).
const KTQ_WON: i32 = 11;

/// `KoopaRace::Await`: Mario has won (the endpoint's status 1), and Koopa the Quick is still running up: stay
/// alive and in the course until he gets to the flag and waits (KOOPA_THE_QUICK_ACT_AFTER_RACE), where
/// `KoopaRace::Claim` takes over (the dialog, then the star: koopa_the_quick_act_after_race). Without it the
/// trees won the race (zt-ktq-geo-1, 293 wins) but every claim came from a branch that had lost.
struct KtqAwait {
    racer: ObjId,
    goal: ObjId,
    zone: (u16, u8),
    star: Option<(i16, u8)>,
}

impl Objective for KtqAwait {
    fn kind(&self) -> &'static str {
        "KoopaRace::Await"
    }
    fn key(&self) -> String {
        format!("KoopaRace::Await {:?} {}", self.zone, self.racer)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        20.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && star_missing(s, self.star)
            && s.find(self.goal).is_some_and(|g| g.value == KTQ_WON)
            && s.find(self.racer).is_some_and(|r| r.action != KTQ_ACT_AFTER_RACE)
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 8, generations: 3, horizon: 600 })
    }
    /// `Done` when KtQ waits at the flag; `Lost` out of the zone or with the win gone; else his distance to the
    /// flag in Mario's frames (he runs it, so Mario only has to survive).
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let (Some(r), Some(g)) = (s.find(self.racer), s.find(self.goal)) else { return Eta::Lost };
        if r.action == KTQ_ACT_AFTER_RACE {
            return Eta::Done;
        }
        if g.value != KTQ_WON {
            return Eta::Lost;
        }
        Eta::Frames(frames_for(dist(r.pos, g.pos)))
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // Stay by the flag, where he stops.
        s.find(self.goal).map(|g| g.pos)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && s.find(self.goal).is_some_and(|g| g.value == KTQ_WON)
    }
    fn kinship(&self) -> u32 {
        10
    }
}

/// `PenguinRace::Await`: Mario got past the finish line first (oRacingPenguinMarioWon, not cheated) and the
/// penguin is still on his way down (RACE, then FINISH_RACE: he slams into the wall). Stay alive, near the
/// finish, until he turns up and waits (SHOW_FINAL_TEXT, racing_penguin.inc.c), where `PenguinRace::Claim`
/// takes over. The race's own `Done` is the win frame, so without this the claim only ever came from a branch
/// that had no penguin waiting yet (the KoopaRace::Await situation, zt-ktq-geo-1).
struct PenguinAwait {
    racer: ObjId,
    goal: ObjId,
    zone: (u16, u8),
    star: Option<(i16, u8)>,
}

impl PenguinAwait {
    fn pending(r: &Obj) -> bool {
        penguin_won(r) && !penguin_cheated(r) && r.action != RACING_PENGUIN_ACT_SHOW_FINAL_TEXT
    }
}

impl Objective for PenguinAwait {
    fn kind(&self) -> &'static str {
        "PenguinRace::Await"
    }
    fn key(&self) -> String {
        format!("PenguinRace::Await {:?} {}", self.zone, self.racer)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        20.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && star_missing(s, self.star) && s.find(self.racer).is_some_and(Self::pending)
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 8, generations: 3, horizon: 600 })
    }
    /// `Done` when the penguin waits at the bottom with the win intact; `Lost` out of the zone or once the
    /// win is gone (cheated, or the flag cleared); else his path left in frames (he slides 70..150 a frame)
    /// plus Mario's way to within 300 of the finish.
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let (Some(r), Some(g)) = (s.find(self.racer), s.find(self.goal)) else { return Eta::Lost };
        if penguin_cheated(r) || !penguin_won(r) {
            return Eta::Lost;
        }
        if r.action == RACING_PENGUIN_ACT_SHOW_FINAL_TEXT {
            return Eta::Done;
        }
        let penguin = super::slides::ccm().progress(r.pos).remaining as f64 / PENGUIN_SLIDE_SPEED as f64;
        Eta::Frames(penguin + frames_for((s.dist(g.pos) - 300.0).max(0.0)))
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        s.find(self.goal).map(|g| g.pos)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && s.find(self.racer).is_some_and(Self::pending)
    }
    fn kinship(&self) -> u32 {
        10
    }
}

/// Where the race star comes to rest (it spawns at (-7339, -5700, -6774) + 200 up and falls: racing_penguin.inc.c
/// `cur_obj_spawn_star_at_y_offset`; a replayed star was taken at (-7356, -5725, -6771)).
const PENGUIN_STAR_REST: [f32; 3] = [-7339.0, -5725.0, -6774.0];

/// `PenguinRace::Star`: the penguin waits at SHOW_FINAL_TEXT after a fair win. From the claim's dialog (read by
/// `Dialog::Read`) the star spawns a few frames after the last page (FinalTextbox -1, MarioWon still set), and
/// Mario has to walk the ~600 units to it. Without this the only base that offered `Star::Collect` was a lucky
/// dense node later on (v5e from the first win route: all 19 attempts at the spawned star started from one
/// `Tech::SlideKick~d` node, none from the dialog's own end), and a 30 M run with 33 claims got no star.
/// `Done` when the star is collected.
struct PenguinStar {
    racer: ObjId,
    zone: (u16, u8),
    star: (i16, u8),
}

impl PenguinStar {
    fn pending(r: &Obj) -> bool {
        r.action == RACING_PENGUIN_ACT_SHOW_FINAL_TEXT && penguin_won(r) && !penguin_cheated(r)
    }

    /// The star object once spawned (its rest position), else the spot.
    fn star_pos(&self, s: &St) -> Option<[f32; 3]> {
        s.objects().iter().find(|o| crate::star::spawned_star(o) && (o.beh_params >> 24) as u8 == self.star.1).map(crate::star::rest_pos)
    }
}

impl Objective for PenguinStar {
    fn kind(&self) -> &'static str {
        "PenguinRace::Star"
    }
    fn key(&self) -> String {
        format!("PenguinRace::Star {:?} {}", self.zone, self.racer)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        50.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && !s.has_star(self.star.0, self.star.1) && s.find(self.racer).is_some_and(Self::pending)
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 8, generations: 3, horizon: 300 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        if s.has_star(self.star.0, self.star.1) {
            return Eta::Done;
        }
        let Some(r) = s.find(self.racer) else { return Eta::Lost };
        let star = self.star_pos(s);
        // The star not out and the win gone (cheated, or used up): nothing left to wait for.
        if star.is_none() && !Self::pending(r) {
            return Eta::Lost;
        }
        Eta::Frames(frames_for(s.dist(star.unwrap_or(PENGUIN_STAR_REST))))
    }
    fn success_value(&self) -> f64 {
        20.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        Some(self.star_pos(s).unwrap_or(PENGUIN_STAR_REST))
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && s.find(self.racer).is_some_and(Self::pending)
    }
    fn kinship(&self) -> u32 {
        10
    }
}

// ---------------------------------------------------------------------------
// Tuxie: carry the lost baby back to its mother.
// ---------------------------------------------------------------------------

/// tuxie.inc.c tuxies_mother_act_0: Mario holding the baby within 300 of her
/// switches her to action 1.
const MOTHER_RADIUS: f32 = 300.0;
/// Picking the baby up: the punch-grab and act_picking_up (~10 frames), as
/// kinghits' GRAB_FRAMES without the lining up.
const BABY_GRAB_FRAMES: f64 = 12.0;

/// `Tuxie::ReturnBaby`: Mario holding the right baby (bhvSmallPenguin with
/// the mother's 2nd behaviour byte) within 300 of Tuxie's mother switches
/// her to action 1 and DIALOG_058; after it Mario drops the baby and the
/// star spawns (tuxie.inc.c tuxies_mother_act_0 / _act_1). The wrong baby
/// gets DIALOG_059 and nothing.
///
/// The baby is found by its home: delivered or rejected it changes
/// behaviour (bhvUnused20E0 / bhvPenguinBaby), which changes its ObjId.
struct ReturnBaby {
    baby_home: [i32; 3],
    mother: ObjId,
    zone: (u16, u8),
    star: (i16, u8),
}

impl ReturnBaby {
    fn baby<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        s.objects()
            .iter()
            .find(|o| o.id.home == self.baby_home && matches!(o.bhv_name(), "bhvSmallPenguin" | "bhvPenguinBaby"))
    }

    fn holding_baby(&self, s: &St) -> bool {
        s.held().is_some_and(|h| h.id.home == self.baby_home)
    }
}

impl Objective for ReturnBaby {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Tuxie::ReturnBaby"
    }
    fn key(&self) -> String {
        format!("Tuxie::ReturnBaby {:?} {:?} {}", self.zone, self.baby_home, self.mother)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        15.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && !s.has_star(self.star.0, self.star.1)
            && s.find(self.mother).is_some_and(|m| m.action == 0)
            && self.baby(s).is_some()
            && (!s.holding || self.holding_baby(s))
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 20, generations: 5, horizon: 300 })
    }
    /// `Done` when the mother is in action 1 on a frame after the start
    /// (v1's success). Holding the baby: the way to within
    /// [`MOTHER_RADIUS`] of her; else to the baby, [`BABY_GRAB_FRAMES`], and
    /// from it to her. Mother or baby gone, or the zone left: `Lost`.
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let Some(m) = s.find(self.mother) else { return Eta::Lost };
        if s.frame != start.frame && m.action == 1 {
            return Eta::Done;
        }
        if self.holding_baby(s) {
            return Eta::Frames(frames_for((s.dist(m.pos) - MOTHER_RADIUS).max(0.0)));
        }
        let Some(b) = self.baby(s) else { return Eta::Lost };
        Eta::Frames(frames_for(s.dist(b.pos)) + BABY_GRAB_FRAMES + frames_for((dist(b.pos, m.pos) - MOTHER_RADIUS).max(0.0)))
    }
    fn success_value(&self) -> f64 {
        20.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        obrrt_core::Arrival::Stop
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        if self.holding_baby(s) { s.find(self.mother).map(|m| m.pos) } else { self.baby(s).map(|b| b.pos) }
    }
    fn steer_button(&self, s: &St) -> u16 {
        match self.baby(s) {
            Some(b) if !s.holding && s.dist(b.pos) < 180.0 => B_BUTTON,
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && s.find(self.mother).is_some() && self.baby(s).is_some()
    }
    fn kinship(&self) -> u32 {
        10
    }
}

// ---------------------------------------------------------------------------
// Snowman: lead the rolling body to the head.
// ---------------------------------------------------------------------------

/// snowmans_bottom_act_2: within this of [`SNOWMAN_SITE`] it jumps under the
/// head (action 3).
const SNOWMAN_SITE_RADIUS: f32 = 300.0;
/// The body's rolling speed: 10 u/f at the start of action 1, capped at 70
/// (snowman.inc.c); a rough middle for a slope run.
const SNOWMAN_BODY_PACE: f32 = 30.0;
/// The body's speed after it leaves the path: it turns to Mario, caps at 70 and rolls the flat run to the site.
const SNOWMAN_ROLL_PACE: f32 = 50.0;
/// ccm_seg7_trajectory_snowman (levels/ccm/areas/1/trajectory.inc.c): the body's own path (action 1).
const SNOWMAN_PATH: &[[f32; 3]] = &[
    [2501.0, 2662.0, -975.0], [2533.0, 2560.0, -800.0], [2566.0, 2300.0, -500.0], [2600.0, 1884.0, 733.0],
    [2466.0, 1647.0, 1835.0], [2000.0, 1483.0, 2233.0], [766.0, 1321.0, 2400.0], [-872.0, 1190.0, 2033.0],
    [-3072.0, 1097.0, 1300.0], [-3805.0, 882.0, -366.0], [-3758.0, 819.0, -1513.0], [-3145.0, 786.0, -2426.0],
    [-1658.0, 546.0, -2853.0], [-138.0, 300.0, -3000.0], [1966.0, -192.0, -2800.0], [3066.0, -456.0, -2800.0],
    [3933.0, -461.0, -2999.0], [4266.0, -188.0, -3433.0], [3901.0, -402.0, -3800.0], [3133.0, -579.0, -3866.0],
    [2033.0, -855.0, -3800.0], [766.0, -1073.0, -3633.0], [-1100.0, -1142.0, -3744.0], [-2318.0, -1188.0, -3658.0],
    [-3318.0, -1228.0, -3375.0], [-4010.0, -1267.0, -2802.0], [-4470.0, -1368.0, -2151.0], [-4679.0, -1358.0, -1321.0],
    [-4770.0, -1333.0, -648.0], [-4847.0, -1351.0, 40.0],
];

fn snowman_path() -> &'static super::slides::SlidePath {
    static P: std::sync::OnceLock<super::slides::SlidePath> = std::sync::OnceLock::new();
    P.get_or_init(|| super::slides::SlidePath::new(SNOWMAN_PATH))
}
/// snowmans_bottom_act_2: at oTimer 200 it gives up and respawns.
const SNOWMAN_GIVE_UP: f64 = 200.0;
/// `mem` of LeadBody: bit 0 set once the body was seen in action 2, `f[0]`
/// the frame that was.
const ROLLING: u32 = 1;

/// `Snowman::LeadBody`: the woken body follows its path down the mountain
/// (action 1), then rolls straight (action 2) — towards Mario if he passed
/// within 800 of the checkpoint (bhvSnowmansBodyCheckpoint, gone once
/// passed) and is within 0x2000 of its heading. Within 300 of
/// [`SNOWMAN_SITE`] it jumps under the head (action 3, then 4). After 200
/// frames of action 2 it gives up and respawns at its start. snowman.inc.c.
struct LeadBody {
    body: ObjId,
    zone: (u16, u8),
    star: (i16, u8),
}

impl LeadBody {
    fn checkpoint(s: &St) -> Option<&Obj> {
        find_bhv(s, "bhvSnowmansBodyCheckpoint")
    }
}

impl Objective for LeadBody {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Snowman::LeadBody"
    }
    fn key(&self) -> String {
        format!("Snowman::LeadBody {:?} {}", self.zone, self.body)
    }
    /// 100: a 700-frame chain the tree must keep extending; at 15 the cheap `Snowman::WakeBody` (80 of 85
    /// attempts) took all the attempts (ccm2-j: 5 LeadBody attempts in 12 M frames).
    fn obj_weight(&self, _b: &St) -> f64 {
        100.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && !s.has_star(self.star.0, self.star.1) && s.find(self.body).is_some_and(|o| matches!(o.action, 1 | 2))
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 700 })
    }
    /// `Done` when the body is in action 3 or 4 on a frame after the start
    /// (v1's success). Else the body's roll to within
    /// [`SNOWMAN_SITE_RADIUS`] of the site at [`SNOWMAN_BODY_PACE`], plus,
    /// while the checkpoint is still there, Mario's way to it (he must pass
    /// it for the body to follow him). Body gone (it gave up), back out of
    /// its rolling actions, or its 200-frame action-2 window
    /// ([`SNOWMAN_GIVE_UP`], from the first frame seen in it) used up:
    /// `Lost`.
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let Some(o) = s.find(self.body) else { return Eta::Lost };
        if s.frame != start.frame && matches!(o.action, 3 | 4) {
            return Eta::Done;
        }
        if !matches!(o.action, 1 | 2) {
            return Eta::Lost;
        }
        if o.action == 2 {
            if mem.flags & ROLLING == 0 {
                mem.flags |= ROLLING;
                mem.f[0] = s.frame as f64;
            } else if s.frame as f64 - mem.f[0] >= SNOWMAN_GIVE_UP {
                return Eta::Lost;
            }
        }
        let roll = |from: [f32; 3]| ((dist(from, SNOWMAN_SITE) - SNOWMAN_SITE_RADIUS).max(0.0) / SNOWMAN_ROLL_PACE) as f64;
        if o.action == 2 {
            // Rolling straight on the yaw fixed at the path's end (oSnowmansBottomUnkF8, `value`): Mario can't
            // change it now. It reaches the site only if that line passes within the site's radius ahead of it.
            let a = (o.value as u16) as f32 * std::f32::consts::TAU / 65536.0;
            let (dx, dz) = (SNOWMAN_SITE[0] - o.pos[0], SNOWMAN_SITE[2] - o.pos[2]);
            let (ahead, side) = (dx * a.sin() + dz * a.cos(), (dx * a.cos() - dz * a.sin()).abs());
            if ahead < 0.0 || side > SNOWMAN_SITE_RADIUS {
                return Eta::Lost;
            }
            return Eta::Frames(roll(o.pos));
        }
        // Action 1: the body's path to its end at SNOWMAN_BODY_PACE; at that frame Mario must be (after the
        // checkpoint, while it is still there) standing in the cone towards the site, which the body then rolls
        // at (ccm2-i: a state with the checkpoint passed and Mario away from the site when the body reached the
        // path's end: it rolled straight on, 644 units west of the site, ETA stuck at 11).
        let end = *SNOWMAN_PATH.last().unwrap();
        let path = snowman_path().progress(o.pos).remaining as f64 / SNOWMAN_BODY_PACE as f64;
        let mario = match Self::checkpoint(s) {
            Some(c) => frames_for(s.dist(c.pos) + dist(c.pos, SNOWMAN_SITE)),
            None => frames_for(s.dist(SNOWMAN_SITE)),
        };
        Eta::Frames(path.max(mario) + roll(end))
    }
    fn success_value(&self) -> f64 {
        15.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        Some(Self::checkpoint(s).map_or(SNOWMAN_SITE, |c| c.pos))
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && s.find(self.body).is_some()
    }
}

// ---------------------------------------------------------------------------
// Generation.
// ---------------------------------------------------------------------------

/// Objectives `o` offers, seen from base state `b` (called for every
/// object within range of `b`; `b` is never a dialog or menu state).
pub fn generate(b: &St, o: &Obj, out: &mut Vec<Arc<dyn Objective>>) {
    let star_here = |i: u8| Some((b.course, i));
    match o.bhv_name() {
        "bhvMessagePanel" | "bhvSignOnWall" => Approach::push(out, &SIGN_READ, b, o, None),
        "bhvYoshi" => Approach::push(out, &YOSHI_TALK, b, o, None),
        // With `lakituskip` on the intro is what Tech::LakituSkip avoids:
        // not offered (walks onto the bridge still play it).
        "bhvCameraLakitu" if byte1(o) == CAMERA_LAKITU_BP_INTRO && !super::tech::enabled("lakituskip") => {
            Approach::push(out, &LAKITU_INTRO, b, o, None)
        }
        // Castle secret stars aren't in St; MIPS only exists while one of his
        // is uncollected.
        "bhvMips" => Approach::push(out, &MIPS_CATCH, b, o, None),
        "bhvUkiki" if byte1(o) == UKIKI_CAGE => {
            // The cage's first behaviour byte is the star (TTM: 1).
            let Some(cage) = find_bhv(b, "bhvUkikiCage") else { return };
            let star = star_here(byte0(cage));
            if !star_missing(b, star) {
                return;
            }
            Approach::push(out, &UKIKI_CATCH, b, o, star);
            Approach::push(out, &UKIKI_FOLLOW, b, o, star);
        }
        "bhvHoot" => {
            Approach::push(out, &HOOT_WAKE, b, o, None);
            Approach::push(out, &HOOT_RIDE, b, o, None);
        }
        // Koopa the Quick (2nd byte KOOPA_BP_KOOPA_THE_QUICK_BOB / _THI); his
        // first byte is the star (BoB 1, THI 2).
        "bhvKoopa" if matches!(byte1(o), 2 | 3) => {
            let star = star_here(byte0(o));
            if star_missing(b, star) {
                Approach::push(out, &KTQ_CLAIM, b, o, star);
            }
        }
        "bhvRacingPenguin" => {
            let star = star_here(byte0(o));
            if star_missing(b, star) {
                Approach::push(out, &PENGUIN_TALK, b, o, star);
                Approach::push(out, &PENGUIN_CLAIM, b, o, star);
            }
        }
        "bhvSnowmansBottom" => {
            if let Some(head) = find_bhv(b, "bhvSnowmansHead") {
                let star = star_here(byte0(head));
                if star_missing(b, star) {
                    Approach::push(out, &SNOWMAN_WAKE, b, o, star);
                }
            }
        }
        "bhvSnowmansHead" => {
            let star = star_here(byte0(o));
            if star_missing(b, star) {
                Approach::push(out, &SNOWMAN_FINISH, b, o, star);
            }
        }
        _ => {}
    }
}

/// Objectives that come from the level rather than one object (called once
/// per `b`): the multi-object missions, whose parts can be far apart.
pub fn generate_level(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    if !(1..=15).contains(&b.course) {
        return;
    }
    for o in b.objects() {
        match o.bhv_name() {
            "bhvKoopa" if matches!(byte1(o), 2 | 3) && o.action != KTQ_ACT_AFTER_RACE => {
                // koopa_the_quick_act_show_init_text: the nearest endpoint.
                let star = Some((b.course, byte0(o)));
                let Some(g) = nearest_bhv(b, "bhvKoopaRaceEndpoint", o.pos) else { continue };
                if !star_missing(b, star) {
                    continue;
                }
                if g.value == KTQ_WON {
                    // Won, KtQ still on his way: wait for him (KoopaRace::Await), then KTQ_CLAIM.
                    out.push(Arc::new(KtqAwait { racer: o.id, goal: g.id, zone: b.zone(), star }));
                } else if o.action == KTQ_ACT_RACE {
                    out.push(Arc::new(Race { race: RaceKind::Koopa, racer: o.id, goal: g.id, zone: b.zone(), star }));
                }
            }
            "bhvRacingPenguin"
                if matches!(o.action, RACING_PENGUIN_ACT_PREPARE_FOR_RACE | RACING_PENGUIN_ACT_RACE) =>
            {
                let star = Some((b.course, byte0(o)));
                if let Some(g) = nearest_bhv(b, "bhvPenguinRaceFinishLine", o.pos)
                    && star_missing(b, star)
                {
                    out.push(Arc::new(Race { race: RaceKind::Penguin, racer: o.id, goal: g.id, zone: b.zone(), star }));
                    // Won already, the penguin still racing: wait for him at the bottom.
                    if PenguinAwait::pending(o) {
                        out.push(Arc::new(PenguinAwait { racer: o.id, goal: g.id, zone: b.zone(), star }));
                    }
                }
            }
            // Waiting at the bottom after a fair win: the star to walk to once the dialog is read.
            "bhvRacingPenguin" if o.action == RACING_PENGUIN_ACT_SHOW_FINAL_TEXT && penguin_won(o) && !penguin_cheated(o) => {
                let star = (b.course, byte0(o));
                if !b.has_star(star.0, star.1) {
                    out.push(Arc::new(PenguinStar { racer: o.id, zone: b.zone(), star }));
                }
            }
            // The penguin slamming into the wall at the bottom (FINISH_RACE), the win in hand.
            "bhvRacingPenguin" if o.action == RACING_PENGUIN_ACT_FINISH_RACE => {
                let star = Some((b.course, byte0(o)));
                if let Some(g) = nearest_bhv(b, "bhvPenguinRaceFinishLine", o.pos)
                    && star_missing(b, star)
                    && PenguinAwait::pending(o)
                {
                    out.push(Arc::new(PenguinAwait { racer: o.id, goal: g.id, zone: b.zone(), star }));
                }
            }
            "bhvTuxiesMother" if o.action == 0 => {
                let star = (b.course, byte0(o));
                if b.has_star(star.0, star.1) {
                    continue;
                }
                // Only the baby whose 2nd byte matches hers earns the star.
                for baby in b.objects().iter().filter(|p| p.bhv_name() == "bhvSmallPenguin" && byte1(p) == byte1(o)) {
                    out.push(Arc::new(ReturnBaby { baby_home: baby.id.home, mother: o.id, zone: b.zone(), star }));
                }
            }
            "bhvSnowmansBottom" if matches!(o.action, 1 | 2) => {
                if let Some(head) = find_bhv(b, "bhvSnowmansHead") {
                    let star = (b.course, byte0(head));
                    if !b.has_star(star.0, star.1) {
                        out.push(Arc::new(LeadBody { body: o.id, zone: b.zone(), star }));
                    }
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::tech::testkit::{ACT_WALKING, obj, st, with};
    use obrrt_core::fold;

    const ACT_FREEFALL: u32 = 0x0100_088C;

    #[test]
    fn return_baby_phases() {
        let mother = |act: i32| obj("bhvTuxiesMother", 3, [0.0, 0.0, 0.0], act, 0);
        let baby = obj("bhvSmallPenguin", 5, [1000.0, 0.0, 0.0], 0, 0);
        let at = |frame: u32, p: [f32; 3], act: i32, held: bool| {
            let mut s = with(st(LEVEL_CCM, 1, p, ACT_WALKING), vec![mother(act), baby.clone()]);
            s.holding = held;
            s.held_slot = held.then_some(5);
            s.frame = frame;
            s
        };
        let m = mother(0);
        let obj = ReturnBaby { baby_home: baby.id.home, mother: m.id, zone: (LEVEL_CCM as u16, 1), star: (4, 2) };
        let traj = vec![
            at(0, [2000.0, 0.0, 0.0], 0, false),
            at(30, [1100.0, 0.0, 0.0], 0, false),
            at(45, [1000.0, 0.0, 0.0], 0, true),
            at(62, [500.0, 0.0, 0.0], 0, true),
            at(70, [250.0, 0.0, 0.0], 1, true),
        ];
        let (met, etas) = fold(&obj, &traj);
        assert_eq!(met, Some(4), "{etas:?}");
        let f: Vec<f64> = etas.iter().map(|e| e.frames()).collect();
        assert!(f.windows(2).all(|w| w[1] < w[0]), "ETA not decreasing: {f:?}");
    }

    #[test]
    fn penguin_race_air_window() {
        let racer = obj("bhvRacingPenguin", 3, [0.0, 0.0, 0.0], RACING_PENGUIN_ACT_RACE, 0);
        let goal = obj("bhvPenguinRaceFinishLine", 4, [0.0, -4000.0, -6000.0], 0, 0);
        let race = Race { race: RaceKind::Penguin, racer: racer.id, goal: goal.id, zone: (LEVEL_CCM as u16, 2), star: None };
        let at = |frame: u32, a: u32| {
            let mut s = with(st(LEVEL_CCM, 2, [0.0, 0.0, 500.0], a), vec![racer.clone(), goal.clone()]);
            s.frame = frame;
            s
        };
        let s0 = at(0, ACT_WALKING);
        let mut mem = Mem::default();
        race.eta(&s0, &s0, &mut mem);
        // A grounded frame resets the count.
        for i in 1..=50 {
            assert!(race.eta(&at(i, ACT_FREEFALL), &s0, &mut mem) != Eta::Lost);
        }
        race.eta(&at(51, ACT_WALKING), &s0, &mut mem);
        let lost_at = (52..200).find(|&i| race.eta(&at(i, ACT_FREEFALL), &s0, &mut mem) == Eta::Lost);
        assert_eq!(lost_at, Some(52 + PENGUIN_AIR_LIMIT as u32));
    }

    #[test]
    fn lead_body_checkpoint_and_give_up() {
        let body = |act: i32, x: f32| {
            let mut o = obj("bhvSnowmansBottom", 3, [-1000.0, -1000.0, 1813.0], act, 0);
            o.pos[0] = x;
            // Rolling west, straight at the site (the heading in `value`, oSnowmansBottomUnkF8).
            o.value = 0xC000;
            o
        };
        let cp = obj("bhvSnowmansBodyCheckpoint", 4, [-500.0, -1000.0, 1813.0], 0, 0);
        let b0 = body(1, -1000.0);
        let lead = LeadBody { body: b0.id, zone: (LEVEL_CCM as u16, 1), star: (4, 4) };
        let at = |frame: u32, objs: Vec<Obj>| {
            let mut s = with(st(LEVEL_CCM, 1, [-600.0, -1000.0, 1813.0], ACT_WALKING), objs);
            s.frame = frame;
            s
        };
        let traj = vec![
            at(0, vec![body(1, -1000.0), cp.clone()]),
            at(10, vec![body(1, -1500.0)]),
            at(20, vec![body(2, -2000.0)]),
            at(100, vec![body(2, -3000.0)]),
            at(110, vec![body(3, -4000.0)]),
        ];
        let (met, etas) = fold(&lead, &traj);
        assert_eq!(met, Some(4), "{etas:?}");
        let f: Vec<f64> = etas.iter().map(|e| e.frames()).collect();
        assert!(f.windows(2).all(|w| w[1] < w[0]), "ETA not decreasing: {f:?}");
        // 200 frames of action 2 after the first seen: it gives up.
        let late = vec![traj[2].clone(), at(220, vec![body(2, -3000.0)])];
        let mut mem = Mem::default();
        let e: Vec<Eta> = late.iter().map(|s| lead.eta(s, &traj[0], &mut mem)).collect();
        assert!(e[0] != Eta::Lost && e[1] == Eta::Lost, "{e:?}");
    }
}
