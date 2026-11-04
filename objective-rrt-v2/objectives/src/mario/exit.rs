//! `Mario::GoToCastle`: leave a course through the start menu (START, Exit
//! Course, confirm). A fixed macro, no search.
//!
//! Exit Course only works while Mario's action has `ACT_FLAG_PAUSE_EXIT`
//! (sm64.h): ~40 stationary actions such as idle, crouching, landing stops,
//! water idle, holding a pole and ledge grab.

use obrrt_core::layout::*;
use obrrt_core::buttons::{A_BUTTON, START_BUTTON};
use obrrt_core::{Budget, Eta, GamePad, Mem, Objective, PadPolicy, St};

/// Frames from START to the castle: the pause menu opening, the cursor to
/// Exit Course and the press (~10 with the fast policy, ~35 with the macro),
/// then the fade and warp (~40). Rough; the objective is played by
/// [`Objective::fixed`], so the ETA only ranks starts and marks the end.
const EXIT_FRAMES: f64 = 60.0;
/// Start bias per star had: v1 ranked GoToCastle starts by
/// `-(stars * 10 + coins * 0.1)` (the latest progress first) at temperature
/// 0.5; in frames that is 10 v1 units ([`obrrt_core::V1_UNIT_FRAMES`]) a
/// star...
const STAR_BIAS_FRAMES: f64 = 10.0 * obrrt_core::V1_UNIT_FRAMES;
/// ... and 0.1 unit a coin...
const COIN_BIAS_FRAMES: f64 = 0.1 * obrrt_core::V1_UNIT_FRAMES;
/// ... at v1's 0.5 units of temperature, so the start distribution is v1's
/// (every start's ETA is the same EXIT_FRAMES).
const EXIT_TEMPERATURE: f64 = 0.5 * obrrt_core::V1_UNIT_FRAMES;

fn castle(level: u16) -> bool {
    let l = level as u32;
    l == LEVEL_CASTLE || l == LEVEL_CASTLE_GROUNDS || l == LEVEL_CASTLE_COURTYARD
}

pub struct GoToCastle {
    level: u16,
}

impl GoToCastle {
    pub fn applies(b: &St) -> bool {
        !castle(b.level)
    }

    pub fn new(b: &St) -> GoToCastle {
        GoToCastle { level: b.level }
    }
}

impl Objective for GoToCastle {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): a menu macro: nothing to start from along the way.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn kind(&self) -> &'static str {
        "Mario::GoToCastle"
    }
    fn key(&self) -> String {
        format!("GoToCastle {}", self.level)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        // Worth most right after a course gave a star. OBRRT_PAINTING_FAST:
        // not boosted (8 of 10 no-star course visits in the hy/dn/mg routes
        // were pass-throughs: re-enter the same painting, exit, 179-234 f,
        // no faster than walking to the next painting).
        if b.stars > 0 && !crate::painting::painting_fast() { 2.0 } else { 0.5 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.level == self.level && s.action_has(ACT_FLAG_PAUSE_EXIT)
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if castle(s.level) {
            return Eta::Done;
        }
        Eta::Frames((EXIT_FRAMES - s.frame.saturating_sub(start.frame) as f64).max(1.0))
    }
    fn start_bias(&self, s: &St) -> f64 {
        // The latest progress first: most stars, then most coins.
        -(s.stars as f64 * STAR_BIAS_FRAMES + s.coins as f64 * COIN_BIAS_FRAMES)
    }
    fn fixed(&self) -> Option<Budget> {
        if crate::menu::fast() {
            return Some(fast_exit());
        }
        let n = GamePad::new(0, 0, 0);
        let mut pads = vec![GamePad::new(START_BUTTON, 0, 0)];
        pads.extend(std::iter::repeat_n(n, 20));
        pads.extend(std::iter::repeat_n(GamePad::new(0, 0, -100), 3));
        pads.extend(std::iter::repeat_n(n, 10));
        pads.push(GamePad::new(A_BUTTON, 0, 0));
        // Then A every few frames through the fade and the castle's dialog.
        for i in 0..240 {
            pads.push(if i % 6 == 0 { GamePad::new(A_BUTTON, 0, 0) } else { n });
        }
        Some(Budget::Macro(pads))
    }
    fn temperature(&self) -> f64 {
        EXIT_TEMPERATURE
    }
}

/// Pause menu's Exit Course line (`gDialogLineNum`; 1 Continue, 3 Camera).
const EXIT_COURSE_LINE: i8 = 2;

/// State feedback (`OBRRT_FAST_MODAL`, the default): START, then, once the
/// pause menu shows its options (`gDialogBoxState` VERTICAL,
/// render_pause_courses_and_castle, ingame_menu.c:2602-2650), a fresh push
/// down until the cursor is on Exit Course, then A on the next frame (the
/// cursor moves before the press check of the same frame, but a push only
/// counts if it is new, `gMenuHoldKeyIndex`, so A waits until the cursor is
/// seen there; A on Continue would unpause). Nothing after that: the fade
/// and warp take no input, and a castle dialog is a modal objective of its
/// own.
fn fast_exit() -> Budget {
    let policy = PadPolicy::new(|traj, played| {
        let s = traj.last().unwrap();
        let paused = s.menu_mode == MENU_MODE_RENDER_PAUSE_SCREEN as i16;
        let was_paused = traj.iter().any(|t| t.menu_mode == MENU_MODE_RENDER_PAUSE_SCREEN as i16);
        if !paused {
            // Before the menu: START (a new press). After it: wait.
            let start = !was_paused && !played.last().is_some_and(|p| p.button & START_BUTTON != 0);
            return GamePad::new(if start { START_BUTTON } else { 0 }, 0, 0);
        }
        if s.dialog_state != 1 {
            return GamePad::new(0, 0, 0);
        }
        if s.menu_line != EXIT_COURSE_LINE {
            let (sx, sy) = crate::menu::nudge(played, (0, -127));
            return GamePad::new(0, sx, sy);
        }
        GamePad::new(crate::menu::press(played, A_BUTTON, START_BUTTON), 0, 0)
    });
    Budget::Policy { limit: 300, policy }
}

