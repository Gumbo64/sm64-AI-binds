//! Random inputs for genetic searches, with a choice of how frames correlate.
//!
//! Every kind keeps the per-frame distribution of [`crate::util::random_gamepad`]
//! (stick axes uniform in -128..=127, A, B and Z each pressed 4% of frames);
//! they differ only in how one frame relates to the next.

use rand::prelude::*;
use rand::rngs::SmallRng;
use sm64_binds::GamePad;
use sm64_binds::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};

/// [`crate::util::random_gamepad`] on a seeded generator.
pub fn random_gamepad(rng: &mut SmallRng) -> GamePad {
    let stick_x = rng.random_range(-128..=127);
    let stick_y = rng.random_range(-128..=127);
    let mut button = 0;
    if rng.random_bool(0.04) {
        button |= A_BUTTON;
    }
    if rng.random_bool(0.04) {
        button |= B_BUTTON;
    }
    if rng.random_bool(0.04) {
        button |= Z_TRIG;
    }
    GamePad::new(button, stick_x, stick_y)
}

fn gauss(rng: &mut SmallRng) -> f32 {
    let u1: f32 = rng.random_range(1e-7..1.0);
    let u2: f32 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Streaming unit-variance Gaussian 1/f^beta noise: the sum of AR(1)
/// processes with time constants 2, 4, ..., 2^octaves frames. Octave k gets
/// variance proportional to tau_k^(beta - 1), which makes the spectrum 1/f^beta
/// between those time scales (beta = 1: equal power per octave, pink; beta = 2:
/// brown, flattening out beyond the longest time constant). Stationary from
/// the first sample.
#[derive(Clone)]
struct Colored {
    rho: Vec<f32>,
    gain: Vec<f32>,
    x: Vec<f32>,
}

impl Colored {
    fn new(rng: &mut SmallRng, beta: f32, octaves: usize) -> Self {
        let tau: Vec<f32> = (0..octaves).map(|k| 2f32.powi(k as i32 + 1)).collect();
        let w: Vec<f32> = tau.iter().map(|t| t.powf(beta - 1.0)).collect();
        let total: f32 = w.iter().sum();
        Colored {
            rho: tau.iter().map(|t| 1.0 - 1.0 / t).collect(),
            gain: w.iter().map(|w| (w / total).sqrt()).collect(),
            x: (0..octaves).map(|_| gauss(rng)).collect(),
        }
    }

    fn next(&mut self, rng: &mut SmallRng) -> f32 {
        let mut sum = 0.0;
        for k in 0..self.x.len() {
            let r = self.rho[k];
            self.x[k] = r * self.x[k] + (1.0 - r * r).sqrt() * gauss(rng);
            sum += self.gain[k] * self.x[k];
        }
        sum
    }
}

impl Colored {
    /// Replaces the octave states by a sample from their posterior given
    /// `obs` ((value, variance) of the summed signal, oldest first), starting
    /// from the stationary prior (Kalman filter, then a Cholesky draw).
    fn fit(&mut self, rng: &mut SmallRng, obs: &[(f32, f32)]) {
        let n = self.x.len();
        let rho: Vec<f64> = self.rho.iter().map(|&r| f64::from(r)).collect();
        let h: Vec<f64> = self.gain.iter().map(|&g| f64::from(g)).collect();
        let mut x = vec![0.0f64; n];
        let mut p = vec![0.0f64; n * n];
        for i in 0..n {
            p[i * n + i] = 1.0;
        }
        let mut ph = vec![0.0f64; n];
        for (t, &(y, r)) in obs.iter().enumerate() {
            if t > 0 {
                for i in 0..n {
                    x[i] *= rho[i];
                    for j in 0..n {
                        p[i * n + j] *= rho[i] * rho[j];
                    }
                    p[i * n + i] += 1.0 - rho[i] * rho[i];
                }
            }
            for i in 0..n {
                ph[i] = (0..n).map(|j| p[i * n + j] * h[j]).sum();
            }
            let s: f64 = (0..n).map(|i| h[i] * ph[i]).sum::<f64>() + f64::from(r);
            let innovation = f64::from(y) - (0..n).map(|i| h[i] * x[i]).sum::<f64>();
            for i in 0..n {
                x[i] += ph[i] * innovation / s;
                for j in 0..n {
                    p[i * n + j] -= ph[i] * ph[j] / s;
                }
            }
        }
        // x + L z, with L L' = P (a little jitter keeps it positive definite).
        let mut l = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = p[i * n + j] + if i == j { 1e-9 } else { 0.0 };
                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = if i == j { sum.max(1e-12).sqrt() } else { sum / l[j * n + j] };
            }
        }
        let z: Vec<f64> = (0..n).map(|_| f64::from(gauss(rng))).collect();
        for i in 0..n {
            self.x[i] = (x[i] + (0..=i).map(|k| l[i * n + k] * z[k]).sum::<f64>()) as f32;
        }
    }
}

/// Inverse of [`phi`] (Acklam's rational approximation, relative error 1e-9).
fn inv_phi(p: f32) -> f32 {
    let p = f64::from(p).clamp(1e-9, 1.0 - 1e-9);
    const A: [f64; 6] = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
    const B: [f64; 5] = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const C: [f64; 6] = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const D: [f64; 4] = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    let tail = |q: f64| (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    (if p < 0.02425 {
        tail((-2.0 * p.ln()).sqrt())
    } else if p > 1.0 - 0.02425 {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }) as f32
}

/// Standard normal CDF (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7).
fn phi(z: f32) -> f32 {
    let x = (z as f64) / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    let erf = if x >= 0.0 { y } else { -y };
    (0.5 * (1.0 + erf)) as f32
}

/// z with P(N(0,1) > z) = 0.04.
const PRESS_Z: f32 = 1.750_686_1;

/// How one input channel evolves over time.
#[derive(Clone, Copy, PartialEq)]
pub enum Noise {
    White,
    /// 1/f^beta Gaussian noise over 2..2^octaves frames.
    Colored {
        beta: f32,
        octaves: usize,
    },
    /// Brownian motion of the stick position: a random walk with N(0, sigma)
    /// steps per frame, reflected at the edges of the stick range (its
    /// stationary distribution is uniform). Sticks only.
    Walk {
        sigma: f32,
    },
}

/// The noise of the stick and of the buttons, and whether crossover copies
/// blocks of frames (about 30 long) rather than single frames.
#[derive(Clone, Copy, PartialEq)]
pub struct Variant {
    pub stick: Noise,
    pub buttons: Noise,
    pub block: bool,
}

impl Variant {
    /// `white` (the original curiosity run), `pink`, `pink-block`,
    /// `white-block`, `brown`, `brown-stick` or `walk`.
    pub fn named(name: &str) -> Option<Variant> {
        const PINK: Noise = Noise::Colored {
            beta: 1.0,
            octaves: 16,
        };
        const BROWN: Noise = Noise::Colored {
            beta: 2.0,
            octaves: 9,
        };
        const W: Noise = Noise::White;
        Some(match name {
            "white" => Variant {
                stick: W,
                buttons: W,
                block: false,
            },
            "pink" => Variant {
                stick: PINK,
                buttons: PINK,
                block: false,
            },
            "pink-block" => Variant {
                stick: PINK,
                buttons: PINK,
                block: true,
            },
            "white-block" => Variant {
                stick: W,
                buttons: W,
                block: true,
            },
            "brown" => Variant {
                stick: BROWN,
                buttons: BROWN,
                block: false,
            },
            "brown-stick" => Variant {
                stick: BROWN,
                buttons: W,
                block: false,
            },
            "walk" => Variant {
                stick: Noise::Walk { sigma: 10.0 },
                buttons: W,
                block: false,
            },
            _ => return None,
        })
    }
}

/// One genome slot's input streams.
#[derive(Clone)]
pub struct Slot {
    v: Variant,
    stick: [Option<Colored>; 2],
    walk: [f32; 2],
    buttons: [Option<Colored>; 3],
}

impl Slot {
    pub fn new(rng: &mut SmallRng, v: Variant) -> Self {
        let colored = |rng: &mut SmallRng, n: Noise| match n {
            Noise::Colored { beta, octaves } => Some(Colored::new(rng, beta, octaves)),
            _ => None,
        };
        Slot {
            v,
            stick: [colored(rng, v.stick), colored(rng, v.stick)],
            walk: match v.stick {
                Noise::Walk { .. } => [
                    rng.random_range(-128.0..128.0),
                    rng.random_range(-128.0..128.0),
                ],
                _ => [0.0; 2],
            },
            buttons: [
                colored(rng, v.buttons),
                colored(rng, v.buttons),
                colored(rng, v.buttons),
            ],
        }
    }

    /// A slot fitted to inputs just played, then sampled: its noise is a
    /// random draw consistent with `history` (oldest first), so the inputs
    /// that follow carry on from those rather than starting from scratch.
    ///
    /// Each coloured channel is a linear-Gaussian model (the AR(1) octave
    /// states, observed through their weighted sum), so a Kalman filter over
    /// the history gives the exact posterior of the octave states, which is
    /// then sampled. The sticks are observed as the z-scores their values
    /// came from (with the quantisation as the observation noise); a button
    /// press is a z above [`PRESS_Z`] (taken as 2.15), a release a z below
    /// it (taken as -0.07). Octaves slower than the history stay near their
    /// prior, i.e. random. A walk starts where the stick was; white noise has
    /// nothing to fit.
    pub fn fit(rng: &mut SmallRng, v: Variant, history: &[GamePad]) -> Self {
        let mut slot = Slot::new(rng, v);
        let Some(last) = history.last() else { return slot };
        if matches!(v.stick, Noise::Walk { .. }) {
            slot.walk = [f32::from(last.stick_x) + 0.5, f32::from(last.stick_y) + 0.5];
        }
        let z_of = |s: i8| {
            let z = inv_phi((f32::from(s) + 128.5) / 256.0);
            let pdf = (-0.5 * z * z).exp() / (2.0 * std::f32::consts::PI).sqrt();
            let width = 1.0 / (256.0 * pdf.max(1e-4));
            (z, (width * width / 12.0).max(1e-4))
        };
        for (i, c) in slot.stick.iter_mut().enumerate() {
            if let Some(c) = c {
                let obs: Vec<(f32, f32)> = history.iter().map(|p| z_of(if i == 0 { p.stick_x } else { p.stick_y })).collect();
                c.fit(rng, &obs);
            }
        }
        for (i, b) in [A_BUTTON, B_BUTTON, Z_TRIG].into_iter().enumerate() {
            if let Some(c) = &mut slot.buttons[i] {
                let obs: Vec<(f32, f32)> = history.iter().map(|p| if p.button & b != 0 { (2.15, 0.09) } else { (-0.07, 0.9) }).collect();
                c.fit(rng, &obs);
            }
        }
        slot
    }

    /// The next frame's input.
    pub fn next(&mut self, rng: &mut SmallRng) -> GamePad {
        if self.v.stick == Noise::White && self.v.buttons == Noise::White {
            return random_gamepad(rng);
        }
        let to_stick = |z: f32| ((phi(z) * 256.0).floor() as i32 - 128).clamp(-128, 127) as i8;
        let mut xy = [0i8; 2];
        for (i, axis) in xy.iter_mut().enumerate() {
            *axis = match self.v.stick {
                Noise::White => rng.random_range(-128..=127),
                Noise::Colored { .. } => to_stick(self.stick[i].as_mut().unwrap().next(rng)),
                Noise::Walk { sigma } => {
                    let mut p = self.walk[i] + sigma * gauss(rng);
                    // Reflect into [-128, 128).
                    loop {
                        if p < -128.0 {
                            p = -256.0 - p;
                        } else if p >= 128.0 {
                            p = 256.0 - p - 1e-3;
                        } else {
                            break;
                        }
                    }
                    self.walk[i] = p;
                    (p.floor() as i32).clamp(-128, 127) as i8
                }
            };
        }
        let mut button = 0;
        for (i, b) in [A_BUTTON, B_BUTTON, Z_TRIG].into_iter().enumerate() {
            let pressed = match self.v.buttons {
                Noise::Colored { .. } => self.buttons[i].as_mut().unwrap().next(rng) > PRESS_Z,
                _ => rng.random_bool(0.04),
            };
            if pressed {
                button |= b;
            }
        }
        GamePad::new(button, xy[0], xy[1])
    }
}

#[cfg(test)]
mod fit_tests {
    use super::*;

    fn run(v: Variant, rng: &mut SmallRng, n: usize) -> Vec<GamePad> {
        let mut slot = Slot::new(rng, v);
        (0..n).map(|_| slot.next(rng)).collect()
    }

    #[test]
    fn inverse_cdf_inverts() {
        for z in [-3.0f32, -1.0, 0.0, 0.5, 2.0] {
            assert!((inv_phi(phi(z)) - z).abs() < 1e-3, "{z}");
        }
    }

    /// Pink inputs continue from a fit with far less jump at the join than a
    /// fresh start, and the fit is random (two draws differ).
    #[test]
    fn fitted_noise_follows_on() {
        let v = Variant::named("pink").unwrap();
        let (mut fit_jump, mut fresh_jump) = (0i64, 0i64);
        for seed in 0..40 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let history = run(v, &mut rng, 300);
            let last = history.last().unwrap();
            let mut fitted = Slot::fit(&mut rng, v, &history);
            let mut fresh = Slot::new(&mut rng, v);
            fit_jump += (i64::from(fitted.next(&mut rng).stick_x) - i64::from(last.stick_x)).abs();
            fresh_jump += (i64::from(fresh.next(&mut rng).stick_x) - i64::from(last.stick_x)).abs();
        }
        assert!(fit_jump * 3 < fresh_jump, "fit {fit_jump} vs fresh {fresh_jump}");
        let mut rng = SmallRng::seed_from_u64(5);
        let history = run(v, &mut rng, 300);
        let a = Slot::fit(&mut rng, v, &history).next(&mut rng);
        let b = Slot::fit(&mut rng, v, &history).next(&mut rng);
        assert!(a.stick_x != b.stick_x || a.stick_y != b.stick_y);
        // White noise and no history are fine.
        let w = Variant::named("white").unwrap();
        Slot::fit(&mut rng, w, &history).next(&mut rng);
        Slot::fit(&mut rng, v, &[]).next(&mut rng);
    }
}
