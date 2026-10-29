//! The coarse layers: overall progress, and one stage of the route.

use obrrt_core::St;
use serde::Deserialize;

use super::util::pack_pos;
use crate::objective::{Cx, Objective};

/// Level (level * 256 + area), keys, stars and a coarse position; then
/// every child.
#[derive(Deserialize)]
pub struct Root {
    #[serde(default = "ten")]
    pub grid_log2: u32,
}

fn ten() -> u32 {
    10
}

impl Objective for Root {
    fn fields(&self) -> &[&'static str] {
        &["level", "keys", "stars", "pos"]
    }

    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        cx.put(0, st.level as i32 * 256 + st.area as i32);
        cx.put(1, st.keys as i32);
        cx.put(2, st.stars as i32);
        cx.put(3, pack_pos(st.pos, self.grid_log2));
        cx.all_children();
    }
}

/// Active in one level (and area), optionally only with a number of keys:
/// a finer position if `grid_log2` is given, then every child.
#[derive(Deserialize)]
pub struct Stage {
    pub level: u16,
    pub area: Option<u8>,
    pub min_keys: Option<u8>,
    pub max_keys: Option<u8>,
    pub grid_log2: Option<u32>,
}

impl Stage {
    fn inside(&self, st: &St) -> bool {
        st.level == self.level
            && self.area.is_none_or(|a| a == st.area)
            && self.min_keys.is_none_or(|k| st.keys >= k)
            && self.max_keys.is_none_or(|k| st.keys <= k)
    }
}

impl Objective for Stage {
    fn is_group(&self) -> bool {
        true
    }

    fn in_group(&self, st: &St) -> bool {
        self.inside(st)
    }

    fn fields(&self) -> &[&'static str] {
        &["pos"]
    }

    fn pos_grid(&self) -> Option<u32> {
        self.grid_log2
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if !self.inside(st) {
            return;
        }
        if let Some(k) = self.grid_log2 {
            cx.put(0, pack_pos(st.pos, k));
        }
        cx.all_children();
    }
}
