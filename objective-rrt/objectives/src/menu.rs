//! Dialog and menu options, each its own objective (one per option), all
//! [`Objective::modal`]: attempted only from a state in that dialog or menu,
//! and the only thing attempted there.
//!
//! - `Dialog::Read(id)`: a dialog with no choice, read to its end.
//! - `Dialog::Yes(id)` / `Dialog::No(id)`: a dialog ending in a yes/no choice.
//! - `Menu::SaveAndContinue` / `Menu::ContinueNoSave`: the save menu after
//!   an exit star (Save and Quit is never an option).
//! - `Menu::SelectAct(course, act)`: an act on the act select screen whose
//!   star is missing. The game loads at most the first missing act, so that
//!   is the only one with a missing star (or, with only the 100-coin star
//!   missing, any act).
//!
//! Each is a [`Budget::Policy`] (state feedback) that presses on the first
//! frame the game takes the press, once the cursor is on the option
//! (`OBRRT_FAST_MODAL`, on by default; see [`set_fast`]). The rules, from the
//! decomp (src/game/ingame_menu.c, src/menu/star_select.c):
//!
//! - Presses count on their first frame only (`buttonPressed`, the new-press
//!   edge: game_init.c:574). Two buttons that do the same thing, alternated,
//!   give a new press every frame; one held gives none after the first.
//! - Dialogs (render_dialog_entries, ingame_menu.c:1716-1792) take A or B
//!   (1742-1743; B confirms like A, it doesn't cancel) only while a page is
//!   shown (`gDialogBoxState` VERTICAL). Opening takes 12 frames (rotating
//!   box: 90° at 7.5°/frame; 9 for the zooming box), each page scroll 8
//!   frames (HORIZONTAL), closing 9 (gDialogID stays set until it's done).
//! - A yes/no choice's cursor moves only on the last page while VERTICAL,
//!   after the press check of the same frame (render_dialog_triangle_choice,
//!   called at 1828 after the switch), so it must be right a frame before the
//!   press; it starts at 1 = yes (1733). The stick moves a cursor on a new
//!   push only (handle_menu_scrolling, 708-758: `gMenuHoldKeyIndex` remembers
//!   the last push, even across menus), then repeats after 10 frames.
//! - The save menu after an exit star (render_course_complete_screen,
//!   2990-3044) shows its options once the coins are counted and
//!   gCourseDoneMenuTimer > 100, and takes A or Start from timer > 110; its
//!   cursor (1 save and continue, 3 continue without saving) moves before
//!   the press check of the same frame.
//! - The act select screen (lvl_update_obj_and_load_act_button_actions,
//!   star_select.c:416-446) takes A, B or Start from sActSelectorMenuTimer
//!   >= 11, before the cursor moves that frame (in area_update_objects).
//!
//! `OBRRT_FAST_MODAL=0` restores the old fixed macros: the stick held toward
//! the option and A every 4th frame.

use std::sync::atomic::{AtomicBool, Ordering};

use obrrt_core::buttons::{A_BUTTON, B_BUTTON, START_BUTTON};
use obrrt_core::layout::{MENU_OPT_CONTINUE_DONT_SAVE, MENU_OPT_SAVE_AND_CONTINUE};
use obrrt_core::{Budget, GamePad, Objective, PadPolicy, Progress, St};

/// Most frames a dialog or menu is given to close.
const LIMIT: usize = 900;

/// `gDialogBoxState` while a page (or a menu's options) is shown: the only
/// state that takes a press (ingame_menu.c:41-46).
const DIALOG_STATE_VERTICAL: i8 = 1;

static FAST: AtomicBool = AtomicBool::new(true);

/// Frame-optimal policies (true, the default) or the old every-4th-frame
/// macros (false) for dialogs and menus.
pub fn set_fast(on: bool) {
    FAST.store(on, Ordering::Relaxed);
    obrrt_core::set_early_dialog_end(on);
}

pub(crate) fn fast() -> bool {
    FAST.load(Ordering::Relaxed)
}

/// A new press of `a` or `b` (whichever the last frame didn't hold).
pub(crate) fn press(played: &[GamePad], a: u16, b: u16) -> u16 {
    if played.last().is_some_and(|p| p.button & a != 0) { b } else { a }
}

/// The stick toward a cursor's target `dir` when it needs to move: pushed,
/// then let go for a frame, so every push is new (a held stick moves a menu
/// cursor once, then only every few frames).
pub(crate) fn nudge(played: &[GamePad], dir: (i8, i8)) -> (i8, i8) {
    if played.last().is_some_and(|p| (p.stick_x, p.stick_y) == dir) { (0, 0) } else { dir }
}

/// The stick held at (x, y), A pressed every 4th frame.
fn press_through(x: i8, y: i8, frames: usize) -> Vec<GamePad> {
    (0..frames).map(|f| GamePad::new(if f % 4 == 3 { A_BUTTON } else { 0 }, x, y)).collect()
}

/// The first index after which `open` stopped holding, if the cursor on its
/// last open frame was `line` (any line if None).
fn closed_on(traj: &[St], open: impl Fn(&St) -> bool, line: Option<i8>) -> Option<usize> {
    (1..traj.len()).find(|&i| !open(&traj[i]) && open(&traj[i - 1])).filter(|&i| line.is_none_or(|l| traj[i - 1].menu_line == l))
}

fn frames_progress(traj: &[St]) -> Progress {
    Progress { score: 0.0, at: traj.len() - 1 }
}

#[derive(Clone, Copy, PartialEq)]
pub enum Answer {
    Read,
    Yes,
    No,
}

pub struct Dialog {
    id: i16,
    answer: Answer,
}

impl Dialog {
    /// The options of the dialog open in `b`.
    pub fn options(b: &St) -> Vec<Dialog> {
        let answers: &[Answer] = if b.dialog_choice { &[Answer::Yes, Answer::No] } else { &[Answer::Read] };
        answers.iter().map(|&answer| Dialog { id: b.dialog, answer }).collect()
    }
}

impl Objective for Dialog {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): dialogs and menus: only the end matters.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn kind(&self) -> &'static str {
        match self.answer {
            Answer::Read => "Dialog::Read",
            Answer::Yes => "Dialog::Yes",
            Answer::No => "Dialog::No",
        }
    }
    fn key(&self) -> String {
        format!("{} {}", self.kind(), self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        1.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.dialog == self.id && s.dialog_choice == (self.answer != Answer::Read) && !s.dialog_released()
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.frame as f64 / 30.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let line = match self.answer {
            Answer::Read => None,
            Answer::Yes => Some(1),
            Answer::No => Some(2),
        };
        // Done once Mario plays again (fast: possibly while the box is still
        // closing, St::dialog_released), else once gDialogID clears.
        closed_on(traj, |s| s.dialog == self.id && !s.dialog_released(), line)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        frames_progress(traj)
    }
    fn budget(&self) -> Budget {
        let x = match self.answer {
            Answer::Read => 0,
            Answer::Yes => -127,
            Answer::No => 127,
        };
        if !fast() {
            return Budget::Macro(press_through(x, 0, LIMIT));
        }
        let line = match self.answer {
            Answer::Read => None,
            Answer::Yes => Some(1),
            Answer::No => Some(2),
        };
        let policy = PadPolicy::new(move |traj, played| {
            let s = traj.last().unwrap();
            if s.dialog_state != DIALOG_STATE_VERTICAL {
                // Opening, scrolling or closing: no press counts. The stick
                // is held for the choice from the start, so the cursor moves
                // on the last page's first frame.
                return GamePad::new(0, if s.dialog_state == 3 { 0 } else { x }, 0);
            }
            match line {
                Some(l) if s.dialog_last_page && s.menu_line != l => {
                    // Cursor not there yet (the stick was already pushed in
                    // the last menu): push again.
                    let (sx, sy) = nudge(played, (x, 0));
                    GamePad::new(0, sx, sy)
                }
                _ => GamePad::new(press(played, A_BUTTON, B_BUTTON), x, 0),
            }
        });
        Budget::Policy { limit: LIMIT, policy }
    }
    fn modal(&self) -> bool {
        true
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}

pub struct SaveMenu {
    option: u32,
}

impl SaveMenu {
    pub fn options() -> Vec<SaveMenu> {
        vec![SaveMenu { option: MENU_OPT_SAVE_AND_CONTINUE }, SaveMenu { option: MENU_OPT_CONTINUE_DONT_SAVE }]
    }
}

impl Objective for SaveMenu {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): dialogs and menus: only the end matters.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn kind(&self) -> &'static str {
        if self.option == MENU_OPT_SAVE_AND_CONTINUE { "Menu::SaveAndContinue" } else { "Menu::ContinueNoSave" }
    }
    fn key(&self) -> String {
        self.kind().into()
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        1.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.in_save_menu()
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.frame as f64 / 30.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        closed_on(traj, St::in_save_menu, Some(self.option as i8))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        frames_progress(traj)
    }
    fn budget(&self) -> Budget {
        // The cursor runs 1 (top) to 3, clamped: up reaches 1, down 3.
        let y = if self.option == MENU_OPT_SAVE_AND_CONTINUE { 127 } else { -127 };
        if !fast() {
            return Budget::Macro(press_through(0, y, LIMIT));
        }
        let target = self.option as i8;
        let policy = PadPolicy::new(move |traj, played| {
            let s = traj.last().unwrap();
            if s.dialog_state != DIALOG_STATE_VERTICAL {
                // Counting coins: the options aren't up.
                return GamePad::new(0, 0, 0);
            }
            if s.menu_line != target {
                let (sx, sy) = nudge(played, (0, y));
                return GamePad::new(0, sx, sy);
            }
            // The press check sees the timer before this frame's increment.
            let button = if s.menu_timer > 110 { press(played, A_BUTTON, START_BUTTON) } else { 0 };
            GamePad::new(button, 0, y)
        });
        Budget::Policy { limit: LIMIT, policy }
    }
    fn modal(&self) -> bool {
        true
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}

pub struct SelectAct {
    course: i16,
    act: u8,
    /// Cursor position (among selectable acts) of `act`, and the right (+)
    /// or left (-) taps from where the cursor starts.
    target: i8,
    taps: i32,
}

impl SelectAct {
    /// The acts worth selecting on the act select screen open in `b`: those
    /// it can load whose star is missing. The game loads min(selected act,
    /// first missing act), so that is the first missing act; when only the
    /// 100-coin star is missing, every act (it can be collected in any).
    pub fn options(b: &St) -> Vec<SelectAct> {
        let course = b.course;
        let acts_done = (0..6).all(|i| b.has_star(course, i));
        let mut acts: Vec<u8> = Vec::new();
        if acts_done {
            if !b.has_star(course, 6) {
                acts.extend(1..=6);
            }
        } else if (1..=6).contains(&b.act_init) && !b.has_star(course, b.act_init - 1) {
            acts.push(b.act_init);
        }
        if acts.is_empty() {
            // Course done: still leave the screen (the first missing act or 1).
            acts.push(b.act_init.clamp(1, 6));
        }
        // Selectable acts, in cursor order: collected ones and the first
        // missing one (all six once the acts are done).
        let selectable: Vec<u8> = (1..=6u8).filter(|&a| acts_done || b.has_star(course, a - 1) || a == b.act_init).collect();
        acts.into_iter()
            .map(|act| {
                let target = selectable.iter().position(|&a| a == act).unwrap_or(0) as i32;
                SelectAct { course, act, target: target as i8, taps: target - b.act_cursor as i32 }
            })
            .collect()
    }
}

impl SelectAct {
    /// Where `self.act` is in the cursor order on the act select screen of
    /// `s`, if it is one of the options there.
    fn target_in(&self, s: &St) -> Option<i8> {
        SelectAct::options(s).into_iter().find(|o| o.course == self.course && o.act == self.act).map(|o| o.target)
    }
}

impl Objective for SelectAct {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): dialogs and menus: only the end matters.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn kind(&self) -> &'static str {
        "Menu::SelectAct"
    }
    fn key(&self) -> String {
        format!("SelectAct {} {}", self.course, self.act)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        1.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        // Fast: only while this act is still an option here (the pool keeps
        // the options of earlier screens, whose act may be done by now).
        s.act_select && s.course == self.course && (!fast() || self.target_in(s).is_some())
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.frame as f64 / 30.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        // The act loaded is the one under the cursor (it is at most the
        // first missing act, so the game doesn't lower it).
        let target = if fast() { self.target_in(&traj[0]).unwrap_or(-1) } else { self.target };
        (1..traj.len()).find(|&i| !traj[i].act_select && traj[i - 1].act_select).filter(|&i| traj[i - 1].act_cursor == target)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        frames_progress(traj)
    }
    fn budget(&self) -> Budget {
        if fast() {
            let (course, act) = (self.course, self.act);
            let policy = PadPolicy::new(move |traj, played| {
                let s = traj.last().unwrap();
                // The cursor position of the act on this screen.
                let target = SelectAct { course, act, target: 0, taps: 0 }.target_in(&traj[0]).unwrap_or(0);
                if s.act_cursor != target {
                    let dir = (if target > s.act_cursor { 127 } else { -127 }, 0);
                    let (sx, sy) = nudge(played, dir);
                    return GamePad::new(0, sx, sy);
                }
                // The stick stays centred: the cursor moves after the press
                // check of the same frame.
                let button = if s.menu_timer >= 11 { press(played, A_BUTTON, B_BUTTON) } else { 0 };
                GamePad::new(button, 0, 0)
            });
            return Budget::Policy { limit: LIMIT, policy };
        }
        // Wait for the screen to accept input (11 frames), tap the stick one
        // act at a time, then A.
        let mut pads = vec![GamePad::new(0, 0, 0); 14];
        let x = if self.taps > 0 { 127 } else { -127 };
        for _ in 0..self.taps.unsigned_abs() {
            pads.push(GamePad::new(0, x, 0));
            pads.push(GamePad::new(0, 0, 0));
            pads.push(GamePad::new(0, 0, 0));
        }
        pads.extend(press_through(0, 0, LIMIT));
        Budget::Macro(pads)
    }
    fn modal(&self) -> bool {
        true
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}
