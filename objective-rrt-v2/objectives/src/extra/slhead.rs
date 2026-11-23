//! Snowman's Land star 1, "Snowman's Big Head" (course 10, star index 0; the star hangs at (700, 4500, 690)).
//!
//! The way up, from the level's collision (levels/sl/areas/1/collision.inc.c; a Dijkstra over its floors with the
//! rules of `obrrt_core::geo`, 100-unit cells, 300 jump, started at the spawn floor (5500, 1024, 400)):
//! 1. the mountain's spiral on its east side, climbing from the floor at y 1024 to (1800, 3242, 200), 15.4 k units;
//! 2. the ice bridge (y 3328, from (1725, 260) to (63, 1704), 460 wide), up the ramp to the 3584 platform west of
//!    the snowman, and the 3738 / 3840 steps to the ledge at (-200, 3840, 1000). This is the only way: the west
//!    slopes of the body (the slippery spiral to (-608, 3564, 1786)) are not joined to the floor below;
//! 3. the ledge is 512 under the head's brim (the floor at y 4352 round the star: (623, 4352, 623), (0, 4352, 1024)).
//!    A wall 768 high (the head, (0,3584,1024)-(0,4352,1024)) stands between: a wall-kick climb, the one place a walk
//!    mesh can't follow (`geo::JUMP` 300), so the walking distance to the star is `UNREACHABLE` from every state.
//!
//! The snowman's wind (src/game/behaviors/sl_snowman_wind.inc.c) blows over the bridge. The object at (700, 3428,
//! 700) is idle until Mario stands on the ground within 1000 of the bridge's centre (1100, 3328, 1164) (the
//! textbox, DIALOG_153; `cur_obj_can_mario_activate_textbox`: any ground action, so a walk or a landing on the bridge
//! does it, a jump over it does not); after the dialog, while Mario is within 1500 and 500 in y of its home it
//! spawns `bhvStrongWindParticle`s every frame (`cur_obj_spawn_strong_wind_particles(12, 3.0, ...)`) aimed at Mario
//! (yaw to Mario, clamped to +-0x1500 round the object's yaw 30 degrees), spread +-6000 units (about 33 degrees),
//! 100 units per frame for 15 frames. A touch is `interact_strong_wind` (interaction.c): `ACT_GETTING_BLOWN`,
//! backwards at 24 and then up to 60, off the bridge sideways (the wind runs from (700, 700) to the north-east, the
//! bridge to the north-west). Who is in the cone: the bridge from 440 units after its start (angle 93 degrees)
//! to its end, the north-west half more; the 3584 platform is outside (its angle to the snowman is -53 or less).
//! The shield is the walking penguin (`bhvSLWalkingPenguin`, sl_walking_penguin.inc.c): a particle dies once within
//! 300 of the point 60 behind the penguin and 100 to the north-east of it (strong_wind_particle.inc.c). The
//! penguin walks the bridge at 2.25 units a frame on average (60 f at 6, 30 f at 12, 30 f at -6, three stops of 30),
//! to x < 300 and back at 12, so Mario crosses at its pace behind the point, or he is blown off.
//!
//! The objectives are one chain, each a stage of that route with the remaining length of it as the ETA (so they
//! work with `OBRRT_GEO` off, where the walk helper is the straight distance):
//! - `SlHead::Climb`: the spiral, to the bridge's start;
//! - `SlHead::Bridge`: the bridge and its ramp, with a penalty while the wind would reach Mario (see [`exposed`]);
//! - `SlHead::Ledge`: the platform and the steps to the ledge;
//! - `SlHead::Brim`: from the ledge or anywhere on the head's wall, to the floor of the brim (wall kicks, the tech
//!   objectives do the kicks; this is the height and the direction).
//! Generated in SL area 1 while the star is missing; nothing elsewhere.

use std::sync::{Arc, OnceLock};

use obrrt_core::layout::LEVEL_SL;
use obrrt_core::{Eta, Mem, Objective, St, dist, frames_for};

/// The route, a polyline over the floors (see the module comment); `ROUTE[BRIDGE]` is the bridge's south-east
/// end, `ROUTE[PLATFORM]` the foot of the 3584 platform, the last is the ledge.
const ROUTE: [[f32; 3]; 49] = [
    [5500.0, 1024.0, 400.0], [5300.0, 1024.0, 400.0], [5100.0, 1024.0, 400.0], [4900.0, 1023.0, 500.0],
    [4700.0, 1018.0, 500.0], [4500.0, 1013.0, 500.0], [4300.0, 1008.0, 500.0], [4100.0, 1004.0, 500.0],
    [3900.0, 999.0, 500.0], [3700.0, 994.0, 500.0], [3500.0, 989.0, 500.0], [3300.0, 984.0, 500.0],
    [3100.0, 963.0, 700.0], [2900.0, 963.0, 900.0], [2700.0, 943.0, 1100.0], [2500.0, 912.0, 1300.0],
    [2300.0, 880.0, 1500.0], [2200.0, 1132.0, 1600.0], [2100.0, 1434.0, 1400.0], [2000.0, 1637.0, 1300.0],
    [1900.0, 1834.0, 1200.0], [1900.0, 1860.0, 1000.0], [1800.0, 2108.0, 800.0], [1600.0, 2108.0, 600.0],
    [1400.0, 2108.0, 400.0], [1400.0, 2251.0, 200.0], [1400.0, 2311.0, 0.0], [1400.0, 2372.0, -200.0],
    [1500.0, 2455.0, -400.0], [1600.0, 2538.0, -600.0], [1600.0, 2816.0, -700.0], [1500.0, 2959.0, -600.0],
    [1500.0, 3084.0, -400.0], [1500.0, 3110.0, -200.0], [1700.0, 3159.0, 0.0], [1800.0, 3242.0, 200.0],
    [1600.0, 3328.0, 400.0], [1400.0, 3328.0, 600.0], [1200.0, 3328.0, 800.0], [1000.0, 3328.0, 900.0],
    [800.0, 3328.0, 1100.0], [600.0, 3328.0, 1300.0], [400.0, 3328.0, 1500.0], [200.0, 3328.0, 1600.0],
    [0.0, 3355.0, 1700.0], [-200.0, 3474.0, 1600.0], [-300.0, 3584.0, 1400.0], [-300.0, 3584.0, 1200.0],
    [-200.0, 3840.0, 1000.0],
];
const BRIDGE: usize = 36;
const PLATFORM: usize = 46;
const LEDGE: usize = 48;

/// The brim's floor under the star (collision: HARD (623, 4352, 623), (768, 4352, 768)).
const BRIM: [f32; 3] = [700.0, 4352.0, 690.0];
/// Where the wind object stands (script.c) and its yaw (30 degrees, the script's angle).
const WIND: [f32; 3] = [700.0, 3428.0, 700.0];
const WIND_YAW_DEG: f32 = 30.0;
/// A particle dies within this of the penguin's shield point (strong_wind_particle.inc.c).
const SHIELD_R: f32 = 300.0;
/// Shield margin: particles start up to 100 off the object, Mario is 60 wide, the shield is judged on the
/// centre line.
const SHIELD_MARGIN: f32 = 80.0;
/// Frames a state in the wind costs on top of the route (a blow-off is a fall to the mountain, tens of seconds).
const WIND_COST: f64 = 600.0;

/// Cumulative length along [`ROUTE`] at each vertex.
fn cum() -> &'static [f32; 49] {
    static C: OnceLock<[f32; 49]> = OnceLock::new();
    C.get_or_init(|| {
        let mut c = [0.0; 49];
        for i in 1..49 {
            c[i] = c[i - 1] + dist(ROUTE[i - 1], ROUTE[i]);
        }
        c
    })
}

/// Where along the route `p` is within the stage `lo..=hi`: (distance from `p` to the route, length of the route
/// left to `hi` from the nearest point on it).
fn along(p: [f32; 3], lo: usize, hi: usize) -> (f32, f32) {
    let c = cum();
    let mut best = (f32::MAX, 0.0);
    for i in lo..hi {
        let (a, b) = (ROUTE[i], ROUTE[i + 1]);
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let l2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
        let t = if l2 > 0.0 {
            (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1] + (p[2] - a[2]) * ab[2]) / l2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let q = [a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]];
        let d = dist(p, q);
        if d < best.0 {
            best = (d, c[hi] - (c[i] + t * l2.sqrt()));
        }
    }
    best
}

/// The penguin's shield point: 60 behind it and 100 to the north-east (sl_walking_penguin.inc.c, the facing is the
/// script's -51 degrees, perpendicular 39 degrees).
fn shield_point(s: &St) -> Option<[f32; 3]> {
    let p = s.objects().iter().find(|o| o.bhv_name() == "bhvSLWalkingPenguin")?;
    Some([p.pos[0] + 60.0 * (-0.777_f32) + 100.0 * 0.629, p.pos[1], p.pos[2] + 60.0 * 0.629 + 100.0 * 0.777])
}

/// Whether the snowman's wind would reach Mario at `m` with the penguin's shield at `shield`: Mario within 1500 of
/// the wind object and 500 of its height, inside the cone (the wind's yaw is the angle to Mario clamped to 30
/// degrees either side of 30 degrees, then +-33 degrees of spread), and the shield not on the line from the object
/// to him (every particle that reaches him passes within about 100 of it).
fn exposed(m: [f32; 3], shield: Option<[f32; 3]>) -> bool {
    let (dx, dz) = (m[0] - WIND[0], m[2] - WIND[2]);
    if dist(m, WIND) >= 1500.0 || (m[1] - WIND[1]).abs() >= 500.0 {
        return false;
    }
    let wrap = |a: f32| (a + 540.0).rem_euclid(360.0) - 180.0;
    let ang = dx.atan2(dz).to_degrees();
    let yaw = WIND_YAW_DEG + wrap(ang - WIND_YAW_DEG).clamp(-30.0, 30.0);
    if wrap(ang - yaw).abs() > 33.0 + 4.0 {
        return false;
    }
    let Some(p) = shield else { return false };
    // Distance from the shield point to the segment object -> Mario, in the plane.
    let (sx, sz) = (p[0] - WIND[0], p[2] - WIND[2]);
    let l2 = dx * dx + dz * dz;
    let t = if l2 > 0.0 { ((sx * dx + sz * dz) / l2).clamp(0.0, 1.0) } else { 0.0 };
    let d = ((sx - t * dx).powi(2) + (sz - t * dz).powi(2)).sqrt();
    d > SHIELD_R - SHIELD_MARGIN
}

/// The courses's star: index 0 of course 10.
fn missing(s: &St) -> bool {
    s.zone() == (LEVEL_SL as u16, 1) && !s.has_star(10, 0)
}

fn grounded(s: &St) -> bool {
    s.pos[1] - s.floor_y < 5.0
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stage {
    Climb,
    Bridge,
    Ledge,
    Brim,
}

impl Stage {
    fn range(self) -> (usize, usize) {
        match self {
            Stage::Climb => (0, BRIDGE),
            Stage::Bridge => (BRIDGE, PLATFORM),
            Stage::Ledge => (PLATFORM - 1, LEDGE),
            Stage::Brim => (LEDGE, LEDGE),
        }
    }
}

pub struct Head {
    stage: Stage,
}

impl Head {
    /// On the head's brim: a floor at 4340 or higher (the head's tiers are 4352, the cap 4864).
    fn on_brim(s: &St) -> bool {
        grounded(s) && s.floor_y > 4340.0
    }

    /// Frames left in the stage (Done at its end).
    fn left(&self, s: &St) -> Option<f64> {
        let (lo, hi) = self.stage.range();
        if self.stage == Stage::Brim {
            if Self::on_brim(s) {
                return None;
            }
            // From the ledge to the brim: straight, and the 512 of wall as wall-kick climbing, which is slower
            // than the same height in the air: its height counts twice.
            let up = (BRIM[1] - s.pos[1]).max(0.0);
            return Some(frames_for(s.dist(BRIM) + up));
        }
        let (d, rest) = along(s.pos, lo, hi);
        if rest < 60.0 && d < 150.0 && (self.stage != Stage::Climb || grounded(s)) {
            return None;
        }
        Some(frames_for(d + rest))
    }
}

impl Objective for Head {
    fn kind(&self) -> &'static str {
        match self.stage {
            Stage::Climb => "SlHead::Climb",
            Stage::Bridge => "SlHead::Bridge",
            Stage::Ledge => "SlHead::Ledge",
            Stage::Brim => "SlHead::Brim",
        }
    }
    fn key(&self) -> String {
        format!("{}", self.kind())
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        match self.stage {
            Stage::Climb => 25.0,
            _ => 30.0,
        }
    }
    /// Near the stage's stretch of the route (500 at most; 250 around the bridge's wind), not already past it.
    fn base_criteria(&self, s: &St) -> bool {
        if !missing(s) {
            return false;
        }
        match self.stage {
            Stage::Brim => s.pos[1] > 3700.0 && s.pos[1] < 5000.0 && !Self::on_brim(s),
            _ => {
                let (lo, hi) = self.stage.range();
                let (d, rest) = along(s.pos, lo, hi);
                d < 500.0 && rest > 100.0
            }
        }
    }
    fn precision(&self) -> obrrt_core::Precision {
        match self.stage {
            Stage::Climb => obrrt_core::Precision::Explore,
            _ => obrrt_core::Precision::Precise,
        }
    }
    fn needs_objects(&self) -> bool {
        matches!(self.stage, Stage::Bridge | Stage::Ledge)
    }
    /// Frames of route left (distance at `PACE`), plus [`WIND_COST`] while the wind would reach Mario; `Done` at
    /// the stage's end (`Brim`: on a floor of the brim); `Lost` out of SL area 1 or with the star had.
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.has_star(10, 0) || s.zone() != (LEVEL_SL as u16, 1) {
            return Eta::Lost;
        }
        let Some(mut f) = self.left(s) else { return Eta::Done };
        if self.needs_objects() && exposed(s.pos, shield_point(s)) {
            f += WIND_COST;
        }
        Eta::Frames(f)
    }
    fn success_value(&self) -> f64 {
        5.0
    }
    /// The route ahead (500 units), or in the bridge's wind the penguin's shield point when Mario would be
    /// exposed there.
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let (lo, hi) = self.stage.range();
        if self.stage == Stage::Brim {
            return Some(BRIM);
        }
        let c = cum();
        let (_, rest) = along(s.pos, lo, hi);
        let want = (c[hi] - rest + 500.0).min(c[hi]);
        let i = (lo..=hi).find(|&i| c[i] >= want).unwrap_or(hi);
        let ahead = ROUTE[i];
        if self.needs_objects() {
            let sh = shield_point(s);
            if exposed(ahead, sh) {
                return sh;
            }
        }
        Some(ahead)
    }
    fn kinship(&self) -> u32 {
        10
    }
    fn exists_in(&self, s: &St) -> bool {
        s.level as u32 == LEVEL_SL && !s.has_star(10, 0)
    }
}

/// Once per `b`: the four stages while the star is missing in SL area 1.
pub fn generate_level(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    if !missing(b) {
        return;
    }
    for stage in [Stage::Climb, Stage::Bridge, Stage::Ledge, Stage::Brim] {
        let h = Head { stage };
        if h.base_criteria(b) {
            out.push(Arc::new(h));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_lengths() {
        let c = cum();
        assert!(c[48] > 11000.0 && c[48] < 13000.0, "{}", c[48]);
        assert!(c[BRIDGE] > c[BRIDGE - 1]);
    }

    #[test]
    fn along_decreases_up_the_route() {
        let (_, a) = along([5000.0, 1024.0, 450.0], 0, BRIDGE);
        let (_, b) = along([2000.0, 1637.0, 1300.0], 0, BRIDGE);
        assert!(a > b);
        assert!(along(ROUTE[BRIDGE], 0, BRIDGE).1 < 1.0);
    }

    #[test]
    fn wind_cone_and_shield() {
        // Mid bridge, no penguin: not shielded by anything, but only "exposed" when a penguin is there to be judged
        // (None = no shield known = no penalty).
        let mid = [1100.0, 3328.0, 1164.0];
        assert!(!exposed(mid, None));
        // A penguin far down the bridge: exposed; the shield next to Mario: not.
        assert!(exposed(mid, Some([1700.0, 3328.0, 700.0])));
        assert!(!exposed(mid, Some([1090.0, 3328.0, 1170.0])));
        // The 3584 platform is outside the cone; so is the bridge's start.
        assert!(!exposed([-300.0, 3584.0, 1400.0], Some([1700.0, 3328.0, 700.0])));
        assert!(!exposed([1876.0, 3328.0, 434.0], Some([300.0, 3328.0, 1700.0])));
        // Above the wind's 500.
        assert!(!exposed([1100.0, 3960.0, 1164.0], Some([1700.0, 3328.0, 700.0])));
    }
}
