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
