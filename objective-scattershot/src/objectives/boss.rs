//! A boss arena: Mario's position, whether he holds the boss, how far it is,
//! and the boss's own state (position, hit points, action, facing). While
//! Mario holds the boss, his facing and spin rate (the throw's direction and
//! power when swinging Bowser) are keyed too.

use obrrt_core::{St, bhv_addr};
use serde::Deserialize;

use super::util::{bucket, dist, pack_pos, yaw_sector};
use crate::objective::{Cx, Objective};

#[derive(Deserialize)]
pub struct Boss {
    pub bhv: String,
    #[serde(default = "step")]
    pub step: f32,
    #[serde(default = "eight")]
    pub grid_log2: u32,
    /// Angle sectors for the boss's facing and Mario's facing while holding.
    #[serde(default = "sectors")]
    pub sectors: u32,
    /// Mario's angular speed (`angleVel` yaw) per bucket while holding.
    #[serde(default = "spin_step")]
    pub spin_step: f32,
    /// Mario's height above the boss per bucket.
    #[serde(default = "dy_step")]
    pub dy_step: f32,
}

fn sectors() -> u32 {
    16
}
fn spin_step() -> f32 {
    256.0
}
fn dy_step() -> f32 {
    100.0
}

fn step() -> f32 {
    250.0
}
fn eight() -> u32 {
    8
}

impl Objective for Boss {
    /// pos, held, dist: Mario and the boss. boss_pos, health, boss_action,
    /// boss_yaw: the boss (-1 when it isn't loaded). dy: Mario's height above
    /// it. yaw, spin: Mario's facing and spin rate, only while he holds it.
    fn fields(&self) -> &[&'static str] {
        &["pos", "held", "dist", "boss_pos", "health", "boss_action", "boss_yaw", "dy", "yaw", "spin"]
    }

    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        cx.put(0, pack_pos(st.pos, self.grid_log2));
        let bhv = bhv_addr(&self.bhv);
        let objs = st.objects.as_deref().map(|v| v.as_slice()).unwrap_or(&[]);
        let mut bosses = objs.iter().filter(|o| Some(o.bhv) == bhv);
        // The held one, else the nearest.
        let held = bosses.clone().find(|o| Some(o.slot) == st.held_slot);
        let boss = held.or_else(|| bosses.by_ref().min_by(|a, b| dist(a.pos, st.pos).total_cmp(&dist(b.pos, st.pos))));
        cx.put(1, held.is_some() as i32);
        match boss {
            Some(b) => {
                cx.put(2, bucket(dist(b.pos, st.pos), self.step));
                cx.put(3, pack_pos(b.pos, self.grid_log2));
                cx.put(4, b.health);
                cx.put(5, b.action);
                cx.put(6, yaw_sector(b.yaw as i16, self.sectors));
                cx.put(7, bucket(st.pos[1] - b.pos[1], self.dy_step));
            }
            None => (2..8).for_each(|f| cx.put(f, -1)),
        }
        if held.is_some() {
            cx.put(8, yaw_sector(st.yaw, self.sectors));
            cx.put(9, bucket(st.angle_vel[1] as f32, self.spin_step));
        } else {
            cx.put(8, -1);
            cx.put(9, 0);
        }
        cx.all_children();
    }
}
