//! Pause buffering for BLJ chains (`OBRRT_PAUSE_BUF=1`, default off): a
//! structured overlay on a chain overlay, not a genome bit.
//!
//! Mechanism (game_init.c read_controller_inputs: `buttonPressed = button &
//! (button ^ buttonDown)`, polled every frame, paused or not): a chain press
//! that lands in its own jump frame leaves Mario in ACT_LONG_JUMP_LAND with
//! A held, so the next game frame can't press A again (no fresh press); it
//! is a ground frame that moves him the full |fv| (common_landing_action ->
//! perform_ground_step) and adds a frame to the cycle. Pressing START on the
//! press frame pauses the game at the end of that frame (play_mode_normal);
//! during the pause A is released, START (or A) unpauses, and the first game
//! frame after sees a fresh A: one press per game frame, no ground frame
//! between. Each pause costs real frames (the TAS's pattern, 2026 120-star
//! f1804-1808 / f1810-1814: press+START, Z(+L), Z+START(+L), Z, press: 3
//! frozen frames per buffered press), so it is slower in real time than the
//! 2-frame cycle (4 frames vs 2 per x1.5); it pays only where the ground
//! frame's travel would carry Mario off the spot that ends each jump early
//! (a short slope, a ledge before a wall stall). Our clocks are emulator
//! frames, so the frozen frames count, as they do in the real game.
//!
//! [`wrap`]: on a chain press (A on ACT_LONG_JUMP_LAND) at a backward speed
//! of at least [`min_speed`] when Mario isn't wall-stalled, add START; then
//! play [`TAIL`] while paused; then hand back to the chain overlay, which
//! sees A released and presses at once.

use std::sync::OnceLock;

use obrrt_core::buttons::{A_BUTTON, START_BUTTON, Z_TRIG};
use obrrt_core::{GamePad, PadOverlay, St};

const ACT_LONG_JUMP_LAND: u32 = 0x0000_0479;

/// The frames after a buffered press (press + START): Z, Z + START
/// (unpause), Z. Verified in run/examples/moatwarp_lab.rs `pause`.
pub const TAIL: [u16; 3] = [Z_TRIG, Z_TRIG | START_BUTTON, Z_TRIG];

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("OBRRT_PAUSE_BUF").is_ok_and(|v| v == "1"))
}

/// Position in an open pause sequence: `Some(k)` if the next pad is TAIL[k].
fn open_seq(played: &[GamePad]) -> Option<usize> {
    let j = played.iter().rposition(|p| p.button & (A_BUTTON | START_BUTTON) == A_BUTTON | START_BUTTON)?;
    let k = played.len() - 1 - j;
    (k < TAIL.len() && played[j + 1..].iter().zip(TAIL).all(|(p, t)| p.button == t)).then_some(k)
}

/// The backward speed from which a chain press is buffered
/// (`OBRRT_PAUSE_BUF_MIN`, default 200): a ground frame then moves Mario at
/// least that far, off most BLJ spots (the JRB dock's TAS chain buffers
/// from -246, the moat bank's from -539).
fn min_speed() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| std::env::var("OBRRT_PAUSE_BUF_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(200.0))
}

/// Whether Mario is wall-stalled: his last frame (a landing, or a jump that
/// came straight down) didn't move him (a
/// ground frame there costs no distance, so no buffer: the TAS's moat-bank
/// stall re-jumps unbuffered, f1814-1820).
fn stalled(traj: &[St]) -> bool {
    let n = traj.len();
    // Paused / unpausing frames repeat the state: not a stall.
    n >= 2 && traj[n - 2].play_mode == 0 && traj[n - 1].action == ACT_LONG_JUMP_LAND && {
        let (a, b) = (traj[n - 2].pos, traj[n - 1].pos);
        (a[0] - b[0]).abs() + (a[2] - b[2]).abs() < 1.0
    }
}

/// `inner` with pause-buffered presses (see the module doc).
pub fn wrap(inner: PadOverlay) -> PadOverlay {
    PadOverlay::keyed(move |traj, played, genome| {
        if let Some(k) = open_seq(played) {
            return (GamePad::new(TAIL[k], 0, 0), false);
        }
        let (pad, key) = inner.pad_keyed(traj, played, genome);
        let s = traj.last().unwrap();
        let press = pad.button & A_BUTTON != 0 && s.action == ACT_LONG_JUMP_LAND && s.play_mode == 0;
        if press && -s.forward_vel >= min_speed() && !stalled(traj) {
            return (GamePad::new(pad.button | START_BUTTON, pad.stick_x, pad.stick_y), key);
        }
        (pad, key)
    })
}

/// [`wrap`] when `OBRRT_PAUSE_BUF=1`, else `inner` unchanged.
pub fn maybe(inner: PadOverlay) -> PadOverlay {
    if on() { wrap(inner) } else { inner }
}
