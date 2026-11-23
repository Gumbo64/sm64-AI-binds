//! Playing a controller from a start state, and the fitness of one play: the
//! GA's (search/src/ga.rs `play`): reward eta_{t-1} − eta_t, a bonus on Done, a
//! penalty on death or Lost, fitness = the best discounted running sum.

use anyhow::Result;
use obrrt_core::{Eta, GamePad, Mem, St, read_state};
use sm64_binds::{SM64Game, SaveState};

use crate::lang::{Ctx, MAX_POINTS, Prog, RefNow, Trace};

/// How much history a case carries (for frames in action and last buttons).
pub const PREFIX: usize = 24;

pub struct Case {
    pub name: String,
    pub train: bool,
    /// Validation: not trained on, picks best-of-run. Neither `train` nor `val`: held out.
    pub val: bool,
    pub movie: usize,
    /// Pads played from power-on to the start (noise included).
    pub frame: usize,
    /// Pads played after loading `save` (the state at frame − noise.len()).
    pub noise: Vec<GamePad>,
    pub save: std::sync::Arc<SaveState>,
    /// The last states before the start, the start last.
    pub pre_states: Vec<St>,
    /// The pads between them.
    pub pre_pads: Vec<GamePad>,
    pub start: St,
    pub task: usize,
    /// The inputs that followed in the movie (as the movie's own baseline).
    pub ref_pads: Vec<GamePad>,
}

/// What is to be done from a case: the fitness's ETA and the points a program
/// can see.
pub trait Task: Send + Sync {
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta;
    /// Points: 0 the current target, 1 the next one, 2.. landmarks.
    fn points(&self, s: &St, mem: &Mem) -> [[f32; 3]; MAX_POINTS];
    fn target_for_baseline(&self, s: &St) -> [f32; 3] {
        self.points(s, &Mem::default())[0]
    }
    fn overlay(&self) -> Option<obrrt_core::PadOverlay> {
        None
    }
    fn objective(&self) -> Option<std::sync::Arc<dyn obrrt_core::Objective>> {
        None
    }
    /// Frames this task may play, if it sets its own limit.
    fn horizon(&self) -> Option<usize> {
        None
    }
    /// The reference's pace into the current target, for `Pace` tests.
    fn pace(&self, _mem: &Mem) -> f32 {
        0.0
    }
    /// The reference at its aligned frame (followers).
    fn reference(&self, _s: &St, _mem: &Mem) -> RefNow {
        RefNow::default()
    }
    /// The task's waypoints as text, for the independent check.
    fn way_text(&self) -> String {
        String::new()
    }
    /// A task's own report line for a finished play.
    fn report(&self, _mem: &Mem, _frames: usize, _end: End) -> String {
        String::new()
    }
}

/// The frame's controller.
pub trait Ctrl {
    fn pad(&mut self, traj: &[St], played: &[GamePad], pts: &[[f32; 3]; MAX_POINTS], pace: f32, rf: RefNow) -> GamePad;
}

/// Frames the last state has been in its action, restarted by a B press
/// (the hand-written overlays' in_action).
pub fn in_action(traj: &[St], played: &[GamePad]) -> u32 {
    let a = traj.last().map_or(0, |s| s.action);
    let off = traj.len().saturating_sub(played.len() + 1);
    let mut n = 0;
    for i in (0..traj.len()).rev() {
        if traj[i].action != a {
            break;
        }
        n += 1;
        if i >= 1 + off && played.get(i - 1 - off).is_some_and(|p| p.button & obrrt_core::buttons::B_BUTTON != 0) {
            break;
        }
    }
    n
}

pub struct ProgCtrl<'a> {
    pub prog: &'a Prog,
    pub trace: Option<&'a Trace>,
}

impl Ctrl for ProgCtrl<'_> {
    fn pad(&mut self, traj: &[St], played: &[GamePad], pts: &[[f32; 3]; MAX_POINTS], pace: f32, rf: RefNow) -> GamePad {
        let s = traj.last().unwrap();
        let c = Ctx { s, last: played.last().copied().unwrap_or(GamePad::new(0, 0, 0)), in_act: in_action(traj, played), pts: *pts, pace, rf };
        self.prog.pad(&c, self.trace)
    }
}

/// Inputs recorded elsewhere.
pub struct PadsCtrl<'a> {
    pub pads: &'a [GamePad],
    pub at: usize,
}

impl Ctrl for PadsCtrl<'_> {
    fn pad(&mut self, _: &[St], _: &[GamePad], _: &[[f32; 3]; MAX_POINTS], _: f32, _: RefNow) -> GamePad {
        let p = self.pads.get(self.at).copied().unwrap_or(GamePad::new(0, 0, 0));
        self.at += 1;
        p
    }
}

/// A hand-written overlay on a "genome" that only steers at the target.
pub struct OverlayCtrl {
    pub overlay: obrrt_core::PadOverlay,
    pub steer: Box<dyn Fn(&St) -> [f32; 3] + Send + Sync>,
}

impl Ctrl for OverlayCtrl {
    fn pad(&mut self, traj: &[St], played: &[GamePad], _: &[[f32; 3]; MAX_POINTS], _: f32, _: RefNow) -> GamePad {
        let s = traj.last().unwrap();
        let t = (self.steer)(s);
        let yaw = ((t[0] - s.pos[0]).atan2(t[2] - s.pos[2]) * 65536.0 / std::f32::consts::TAU) as i32 as i16;
        let (x, y) = crate::lang::stick(s.cam_yaw, yaw, 1.0);
        self.overlay.pad(traj, played, GamePad::new(0, x, y))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Roll {
    pub horizon: usize,
    pub gamma: f64,
    pub bonus: f64,
    pub death: f64,
    /// Ends a play this many frames after its fitness last rose.
    pub stall: usize,
}

impl Default for Roll {
    fn default() -> Self {
        Roll { horizon: 300, gamma: 0.99, bonus: 300.0, death: 600.0, stall: 60 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    Done,
    Died,
    Lost,
    Horizon,
    Stall,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub fitness: f64,
    pub done: Option<usize>,
    pub end: End,
    pub frames: u64,
    pub best_eta: f64,
    pub start_eta: f64,
    pub pads: Vec<GamePad>,
    /// The task's report line.
    pub report: String,
}

pub fn play(game: &mut SM64Game, case: &Case, task: &dyn Task, ctrl: &mut dyn Ctrl, r: &Roll, keep: bool) -> Result<Outcome> {
    game.load_state(&case.save)?;
    for p in &case.noise {
        game.step_game(*p)?;
    }
    let mut traj = case.pre_states.clone();
    let mut played = case.pre_pads.clone();
    let mut mem = Mem::default();
    let start = &case.start;
    let e0 = task.eta(start, start, &mut mem).frames();
    let mut out = Outcome {
        fitness: f64::NEG_INFINITY,
        done: None,
        end: End::Horizon,
        frames: 0,
        best_eta: e0,
        start_eta: e0,
        pads: Vec::new(),
        report: String::new(),
    };
    let (mut sum, mut disc, mut prev) = (0.0, 1.0, e0);
    let mut best_t = 0;
    let base = case.frame as u32;
    let horizon = task.horizon().unwrap_or(r.horizon);
    for t in 0..horizon {
        let cur = traj.last().unwrap();
        let pts = task.points(cur, &mem);
        let pad = ctrl.pad(&traj, &played, &pts, task.pace(&mem), task.reference(cur, &mem));
        game.step_game(pad)?;
        out.frames += 1;
        if keep {
            out.pads.push(pad);
        }
        played.push(pad);
        let st = read_state(game, base + t as u32 + 1, false);
        let mut rw;
        let eta = if !st.alive() || st.lives < start.lives { Eta::Lost } else { task.eta(&st, start, &mut mem) };
        let lost = match eta {
            Eta::Lost => true,
            Eta::Frames(x) => !x.is_finite(),
            Eta::Done => false,
        };
        if lost {
            rw = -r.death;
            out.end = if st.alive() { End::Lost } else { End::Died };
        } else {
            rw = prev - eta.frames();
            prev = eta.frames();
            if eta.done() {
                rw += r.bonus;
                out.done = Some(t + 1);
                out.end = End::Done;
            }
        }
        out.best_eta = out.best_eta.min(prev);
        sum += disc * rw;
        disc *= r.gamma;
        if sum > out.fitness {
            out.fitness = sum;
            best_t = t;
        }
        if lost || eta.done() {
            break;
        }
        traj.push(st);
        if t.saturating_sub(best_t) > r.stall {
            out.end = End::Stall;
            break;
        }
    }
    out.report = task.report(&mem, out.frames as usize, out.end);
    Ok(out)
}

/// `f` on (i, game) over `0..n`, in order, on one thread per game.
pub fn par<R: Send>(games: &mut [SM64Game], n: usize, f: impl Fn(&mut SM64Game, usize) -> Result<R> + Sync) -> Result<Vec<R>> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Option<Result<R>>>> = (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for g in games.iter_mut() {
            let (next, out, f) = (&next, &out, &f);
            sc.spawn(move || {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    *out[i].lock().unwrap() = Some(f(g, i));
                }
            });
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
}
