//! Technique lenses: fine resolution in a small region, and the side of a
//! door or gate.

use obrrt_core::{St, bhv_addr};
use serde::Deserialize;

use super::util::{ActionGroup, action_class, bucket, dist, in_box, pack_pos, side, yaw_bucket, yaw_sector};
use crate::objective::{Cx, Objective};

/// A region (a sphere `center`/`radius`, or a box `min`/`max`) where the
/// key gets fine: position, forward speed (buckets of `speed_step`), facing
/// (`yaw_sectors`) and action (classes, see `actions`), and the children. Outside it, while |forward speed| > `payoff_speed` (the trick's
/// speed being spent), a `payoff` tag and the speed; otherwise nothing.
#[derive(Deserialize)]
pub struct Lens {
    pub center: Option<[f32; 3]>,
    pub radius: Option<f32>,
    pub min: Option<[f32; 3]>,
    pub max: Option<[f32; 3]>,
    #[serde(default = "six")]
    pub grid_log2: u32,
    /// Forward speed per bucket; 0 leaves it out of the key.
    #[serde(default = "four")]
    pub speed_step: f32,
    /// Vertical speed per bucket; 0 (the default) leaves it out of the key.
    #[serde(default)]
    pub vspeed_step: f32,
    #[serde(default = "sixteen")]
    pub yaw_sectors: u32,
    pub payoff_speed: Option<f32>,
    /// Only active while an object with this behaviour (e.g. `bhvCameraLakitu`)
    /// is loaded, and, with `while_param2`, whose second behaviour-parameter
    /// byte (`oBehParams2ndByte`) is that. Otherwise the lens writes nothing.
    pub while_object: Option<String>,
    pub while_param2: Option<u32>,
    /// Mario is "caught" (e.g. by Lakitu's intro dialog) when his action is in
    /// one of `caught_actions` or the open dialog is `caught_dialog`. The key
    /// is then just `caught`, whatever else is going on: all caught states
    /// (with the same level, keys and stars) share one cell.
    #[serde(default)]
    pub caught_actions: Vec<ActionGroup>,
    pub caught_dialog: Option<i16>,
    /// A direction on the ground (x, z) to face: with it, `yaw` is fine within
    /// `yaw_fine_deg` of that direction (`yaw_fine_sectors` buckets, odd so the direction is mid-bucket) and coarse
    /// elsewhere (`yaw_coarse_sectors` buckets, odd, for the rest of the circle),
    /// instead of `yaw_sectors` equal ones.
    pub yaw_toward: Option<[f32; 2]>,
    #[serde(default = "forty_five")]
    pub yaw_fine_deg: f32,
    #[serde(default = "seven")]
    pub yaw_fine_sectors: u32,
    #[serde(default = "three")]
    pub yaw_coarse_sectors: u32,
    /// The actions told apart, in class order (class 0 is every other one):
    /// `action` is 1 + the index of the first group Mario's action is in.
    #[serde(default = "default_actions")]
    pub actions: Vec<ActionGroup>,
}

fn forty_five() -> f32 {
    45.0
}
fn seven() -> u32 {
    7
}
fn three() -> u32 {
    3
}

fn default_actions() -> Vec<ActionGroup> {
    vec![ActionGroup::Walking, ActionGroup::LongJump]
}

fn six() -> u32 {
    6
}
fn four() -> f32 {
    4.0
}
fn sixteen() -> u32 {
    16
}

impl Lens {
    /// Whether the `while_object` condition holds (always, without one).
    fn active(&self, st: &St) -> bool {
        let Some(name) = &self.while_object else { return true };
        let Some(bhv) = bhv_addr(name) else { return false };
        let objs = st.objects.as_deref().map(|v| v.as_slice()).unwrap_or(&[]);
        objs.iter().any(|o| o.bhv == bhv && self.while_param2.is_none_or(|p| (o.beh_params >> 16) & 0xFF == p))
    }

    pub fn inside(&self, p: [f32; 3]) -> bool {
        let sphere = match (self.center, self.radius) {
            (Some(c), Some(r)) => dist(p, c) < r,
            _ => false,
        };
        let boxed = match (self.min, self.max) {
            (Some(lo), Some(hi)) => in_box(p, lo, hi),
            _ => false,
        };
        sphere || boxed
    }
}

impl Objective for Lens {
    fn fields(&self) -> &[&'static str] {
        &["pos", "hspeed", "yaw", "action", "payoff", "vspeed", "caught"]
    }

    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }

    fn needs_objects(&self) -> bool {
        self.while_object.is_some()
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if !self.active(st) {
            return;
        }
        if self.caught_dialog == Some(st.dialog) || self.caught_actions.iter().any(|g| g.contains(st.action)) {
            cx.absorb(6, 1);
            return;
        }
        if self.inside(st.pos) {
            cx.put(0, pack_pos(st.pos, self.grid_log2));
            if self.speed_step > 0.0 {
                cx.put(1, bucket(st.forward_vel, self.speed_step));
            }
            if self.vspeed_step > 0.0 {
                cx.put(5, bucket(st.vel[1], self.vspeed_step));
            }
            cx.put(2, match self.yaw_toward {
                // sm64 yaw 0 faces +z and grows towards +x.
                Some([x, z]) => {
                    let target = (x.atan2(z) / std::f32::consts::TAU * 65536.0).round() as i32 as i16;
                    let range = (self.yaw_fine_deg / 360.0 * 65536.0) as i32;
                    yaw_bucket(st.yaw, target, range, self.yaw_fine_sectors, self.yaw_coarse_sectors)
                }
                None => yaw_sector(st.yaw, self.yaw_sectors),
            });
            cx.put(3, action_class(st.action, &self.actions));
            cx.all_children();
        } else if self.payoff_speed.is_some_and(|v| st.forward_vel.abs() > v) {
            cx.put(4, 1);
            cx.put(1, bucket(st.forward_vel, if self.speed_step > 0.0 { self.speed_step } else { 4.0 }));
        }
    }
}

/// Within `radius` of `point`: which side of the plane through it (normal
/// `normal`) Mario is on, 1 on the side the normal points to.
#[derive(Deserialize)]
pub struct Side {
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub radius: f32,
}

impl Objective for Side {
    fn fields(&self) -> &[&'static str] {
        &["side"]
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if dist(st.pos, self.point) < self.radius {
            cx.put(0, side(st.pos, self.point, self.normal));
        }
    }
}
