//! Door animations, dialogs and menus as states of their own.
//!
//! Without this a state in the middle of a door animation or with a dialog
//! open has the same key as a faster state at the same spot, and is thrown
//! away: nothing can then be expanded from it to see the animation or the
//! dialog through. `objective-rrt` handles these the same way round: a door
//! action counts as having entered the door (`door.rs`, `in_door`), and a
//! dialog or menu is "modal", where the only inputs tried are the ones that
//! press through it (`menu.rs`).

use obrrt_core::St;
use serde::Deserialize;

use super::util::ActionGroup;
use crate::objective::{Cx, Objective};

/// Writes `busy` (1 a door animation, 2 a dialog, 3 the save menu, 4 the act
/// select) and, in a dialog, which one, while Mario is in one of them;
/// nothing otherwise.
#[derive(Deserialize, Default)]
pub struct Busy {}

fn kind(st: &St) -> i32 {
    if st.in_save_menu() {
        3
    } else if st.act_select {
        4
    } else if st.in_dialog() {
        2
    } else if ActionGroup::Door.contains(st.action) {
        1
    } else {
        0
    }
}

impl Objective for Busy {
    fn fields(&self) -> &[&'static str] {
        &["busy", "dialog"]
    }

    fn busy(&self, st: &St) -> bool {
        kind(st) != 0
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        match kind(st) {
            0 => {}
            2 => {
                cx.put(0, 2);
                cx.put(1, st.dialog as i32);
            }
            k => cx.put(0, k),
        }
    }
}
