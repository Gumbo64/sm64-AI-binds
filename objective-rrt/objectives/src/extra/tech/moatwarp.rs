//! `Tech::MoatWarpBlj` (`moatwarp`, and `OBRRT_MOATWARP=1`; default off):
//! the 2026 120-star TAS's opening (ref/tas/120_first2_phases.txt, frames
//! 1756-1852). From the moat bank's 71.6-degree NOT_SLIPPERY slope (floor
//! 0x15, the TAS stops at (3442, -166, 787)) Mario BLJs backwards up the
//! bank (71.6 -> 35 -> 14 degrees) until a wall stalls him near (932, 717,
//! -71); stalled, each re-jump is x1.5 in 2 frames (-1211 -> -6098); then
//! one WALKING frame's quarter steps (~|fv|/4 = 1500 units each) put him
//! over the moat bottom behind the grills, onto the Vanish Cap Under the
//! Moat warp (castle_grounds/script.c bhvWarp at (-3379, -815, -2025),
//! behaviour byte 0x3C: radius 600, height 50), normally closed until the
//! moat is drained. VCutM's star follows (TAS: star 1 at f2569).
//!
//! Rough location only (technique objectives give a start region and an
//! endpoint; the general BLJ code finds the setup):
//! - **Start:** controllable on the castle grounds within [`REGION_R`] of
//!   [`BANK`], below y 100 (the bank), or already mid-chain (long jump or its
//!   landing, fv <= -30) on the grounds' east side (the climb up to the
//!   stall), VCutM's star not yet collected.
//! - **Success:** in VCutM (level 18), or ACT_DISAPPEARED within 700 of the
//!   warp on the grounds (the warp's fade), after a BLJ (the chain metric
//!   saw a re-jump): the trick itself, not swimming in after a drain.
//! - **Progress:** the graded BLJ chain metric (super::blj::metric_at: x1.5
//!   steps of the peak backward speed), and once the chain has passed
//!   [`FAST`], the closest horizontal approach to the warp after it (the
//!   aim of the walking frame).
//! - **Inputs:** the GA with the BLJ chain overlay (super::blj, free exit:
//!   after the chain's last press the genome steers, so it picks the walking
//!   frame's stick), facing down the bank, presses until [`Variant::stop`]
//!   backward speed; with `OBRRT_PAUSE_BUF=1` presses after a same-frame
//!   landing are pause-buffered (super::pausebuf).
//!
//! Findability: see run/examples/moatwarp_lab.rs and the report in
//! ref/tas/slope_techniques.md's moat-warp section.

use std::sync::{Arc, OnceLock};

use obrrt_core::{Budget, Objective, PadOverlay, Precision, Progress, St};

use super::{blj, enabled};

const LEVEL_GROUNDS: u16 = 16;
const LEVEL_VCUTM: u16 = 18;
/// gLevelToCourseNumTable: VCutM is course 22 (secret), star 0.
const COURSE_VCUTM: i16 = 22;
const ACT_DISAPPEARED: u32 = 0x0000_1300;
const ACT_LONG_JUMP: u32 = 0x0300_0888;
const ACT_LONG_JUMP_LAND: u32 = 0x0000_0479;

/// The TAS's stop on the moat bank (f1756, FREEFALL_LAND_STOP).
pub const BANK: [f32; 3] = [3442.0, -166.0, 787.0];
pub const REGION_R: f32 = 700.0;
/// The VCutM warp in the moat (radius 600).
pub const WARP: [f32; 3] = [-3379.0, -815.0, -2025.0];
/// The TAS's wall stall at the top of the climb.
pub const STALL: [f32; 3] = [932.0, 717.0, -71.0];
/// Downhill of the bank's floor at the TAS's stop (normal (0.895, 0.316,
/// -0.316)), when the base state isn't on it.
const BANK_DOWN_FALLBACK: i32 = 0x4dd1;
/// Past this backward speed the chain is fast enough to reach the moat in
/// one walking frame (|fv| ~ the 4700 units to the warp).
pub const FAST: f32 = 1000.0;

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("OBRRT_MOATWARP").is_ok_and(|v| v == "1")) && enabled("moatwarp")
}

fn hd(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn mid_chain(s: &St) -> bool {
    matches!(s.action, ACT_LONG_JUMP | ACT_LONG_JUMP_LAND) && s.forward_vel <= -30.0
}

/// The grounds' east side: the bank, the climb and the stall's terrace.
fn east_side(s: &St) -> bool {
    s.pos[0] > 600.0 && s.pos[0] < 4200.0 && s.pos[2] > -600.0 && s.pos[2] < 1500.0
}

fn on_bank(s: &St) -> bool {
    s.hdist(BANK) < REGION_R && s.pos[1] < 100.0
}

/// Downhill yaw of `s`'s floor (the normal's horizontal part points down).
fn downhill(s: &St) -> i32 {
    (s.floor_normal[0].atan2(s.floor_normal[2]) * 32768.0 / std::f32::consts::PI) as i32
}

/// Chain variants: the side of downhill the setup faces (straight down the
/// bank walks into the moat: it is a few units wide; the TAS faces 42
/// degrees off downhill, 0x3003 vs 0x4dd1), and the backward speed past
/// which the overlay stops pressing and the genome steers (the walking
/// frame). In the lab only the TAS's -6097 of its stall's x1.5 ladder
/// (-1211, -1815, -2718, -4070, -6097) walks into the warp: `stop` 5000
/// stops there, 9000 one press later (-9147).
#[derive(Debug, Clone, Copy)]
pub struct Variant {
    pub side: i32,
    pub stop: f32,
}

const VARIANTS: [Variant; 4] =
    [Variant { side: -0x2000, stop: 5000.0 }, Variant { side: 0x2000, stop: 5000.0 }, Variant { side: -0x2000, stop: 9000.0 }, Variant { side: 0x2000, stop: 9000.0 }];

/// The objective's inputs: a quick setup on the narrow bank (walk one frame
/// toward `yaw` plus the genome's offset, up to 0x1000 by its stick x;
/// crouch slide; long jump with the stick back), the BLJ chain rule
/// (super::blj::overlay_exit, pause-buffered under OBRRT_PAUSE_BUF=1) up to
/// `stop`, then the genome's stick alone on the landing (the walking frame;
/// key frames). blj's own setup waits for 10.5 u/f before the crouch slide,
/// which on this bank is a step into the water.
pub fn overlay(yaw: i32, stop: f32) -> PadOverlay {
    use obrrt_core::GamePad;
    use obrrt_core::buttons::{A_BUTTON, Z_TRIG};
    const ACT_WALKING: u32 = 0x0400_0440;
    const ACT_CROUCH_SLIDE: u32 = 0x0480_8459;
    let rule = blj::overlay_exit(BANK, yaw, 0.0, None, stop, true);
    let along = |s: &St, y: i32| {
        let rel = (y - s.cam_yaw as i32) as f32 * std::f32::consts::TAU / 65536.0;
        ((127.0 * rel.sin()).round() as i8, (-127.0 * rel.cos()).round() as i8)
    };
    PadOverlay::keyed(move |traj, played, genome| {
        let s = traj.last().unwrap();
        let begun = mid_chain(&traj[0]) || traj.iter().any(|t| t.action == ACT_LONG_JUMP);
        if begun && s.action == ACT_LONG_JUMP && !traj.iter().any(|t| t.action == ACT_LONG_JUMP_LAND) {
            // The first (forward) long jump: stick fully back until it lands
            // going backwards (blj's free exit would hand this to the genome).
            let b = along(s, s.yaw as i32 + 0x8000);
            return (GamePad::new(Z_TRIG, b.0, b.1), false);
        }
        if begun {
            let prev_a = played.last().is_some_and(|p| p.button & A_BUTTON != 0);
            if s.action == ACT_LONG_JUMP_LAND && -s.forward_vel >= stop && !prev_a && s.play_mode == 0 {
                return (GamePad::new(0, genome.stick_x, genome.stick_y), true);
            }
            return rule.pad_keyed(traj, played, genome);
        }
        match s.action {
            ACT_CROUCH_SLIDE => {
                let b = along(s, s.yaw as i32 + 0x8000);
                (GamePad::new(A_BUTTON | Z_TRIG, b.0, b.1), false)
            }
            ACT_WALKING if traj.len() > 1 => {
                let f = along(s, s.yaw as i32);
                (GamePad::new(Z_TRIG, f.0, f.1), false)
            }
            _ if s.controllable() && traj.len() == 1 => {
                let f = along(s, yaw + genome.stick_x as i32 * 0x1000 / 128);
                (GamePad::new(0, f.0, f.1), true)
            }
            // Anything else (a fall, a bonk): the genome.
            _ => (genome, false),
        }
    })
}

pub struct MoatWarpBlj {
    start: [f32; 3],
    yaw: i32,
    v: Variant,
    cell: [i32; 2],
}

impl MoatWarpBlj {
    pub fn generate(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
        if !on() || !Self::base(b) {
            return;
        }
        // On the steep bank: start where Mario is, facing down its floor; mid-chain: as is.
        let (start, yaw) = if b.floor_normal[1] < 0.5 && on_bank(b) {
            (b.pos, downhill(b))
        } else if mid_chain(b) {
            (b.pos, b.yaw as i32)
        } else {
            (BANK, BANK_DOWN_FALLBACK)
        };
        let cell = [(start[0] / 400.0).round() as i32, (start[2] / 400.0).round() as i32];
        for v in VARIANTS {
            // Mid-chain the side is moot: one per stop.
            if mid_chain(b) && v.side > 0 {
                continue;
            }
            out.push(Arc::new(MoatWarpBlj { start, yaw: yaw + v.side, v, cell }));
        }
    }

    fn base(s: &St) -> bool {
        s.level == LEVEL_GROUNDS
            && !s.has_star(COURSE_VCUTM, 0)
            && !s.holding
            && ((s.controllable() && on_bank(s)) || (mid_chain(s) && east_side(s) && s.play_mode == 0))
    }
}

impl Objective for MoatWarpBlj {
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn kind(&self) -> &'static str {
        "Tech::MoatWarpBlj"
    }
    fn key(&self) -> String {
        format!("Tech::MoatWarpBlj {:?} side {:#x} stop {}", self.cell, self.v.side, self.v.stop)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        if Self::base(b) { 6.0 } else { 0.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        Self::base(s) && s.hdist(self.start) < 600.0
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        // Mid-chain states first (by speed), then nearness to the start.
        if mid_chain(s) { -(s.forward_vel.abs() as f64).ln() } else { s.dist(self.start) as f64 / 400.0 }
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let sp = blj::chain_speed(traj);
        let mut chained = mid_chain(&traj[0]);
        for (i, s) in traj.iter().enumerate() {
            chained |= sp[i] > 0.0;
            let warped = s.level == LEVEL_VCUTM || (s.level == LEVEL_GROUNDS && s.action == ACT_DISAPPEARED && s.hdist(WARP) < 700.0);
            if warped {
                return chained.then_some(i);
            }
        }
        None
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // The chain metric (x1.5 steps), plus, once past FAST, the walk's
        // aim: the closest approach to the warp after that (up to +3 steps).
        let (m, peak, at, _) = blj::metric_at(traj);
        let start_fast = mid_chain(&traj[0]) && -traj[0].forward_vel > FAST;
        if peak < FAST && !start_fast {
            return Progress { score: m, at };
        }
        let from = if start_fast { 0 } else { at };
        let (i, d) = traj
            .iter()
            .enumerate()
            .skip(from)
            .filter(|(_, s)| s.level == LEVEL_GROUNDS && s.play_mode == 0)
            .map(|(i, s)| (i, hd(s.pos, WARP)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((at, f32::MAX));
        let aim = (1.0 - d as f64 / 5000.0).max(0.0) * 3.0;
        if aim > 0.0 && d < hd(traj[from].pos, WARP) { Progress { score: m + aim, at: i } } else { Progress { score: m, at } }
    }
    fn budget(&self) -> Budget {
        // Setup (~10) + climb (~40) + stall cycles (~10) + walk, room for buffering.
        Budget::Ga { population: 32, generations: 6, horizon: 140 }
    }
    fn precision(&self) -> Precision {
        Precision::Precise
    }
    fn success_value(&self) -> f64 {
        // A course entry (VCutM: its star is ~700 frames on).
        6.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        if mid_chain(s) || s.forward_vel < -FAST { Some(WARP) } else { Some(blj::steer(s, STALL)) }
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        Some(overlay(self.yaw, self.v.stop))
    }
    fn exists_in(&self, s: &St) -> bool {
        s.level == LEVEL_GROUNDS && !s.has_star(COURSE_VCUTM, 0)
    }
}
