//! Pink (1/f) noise inputs, as the pre-tree GAs used (copied from
//! sm64-AI-binds/rust/src/noise.rs, variant "pink"): each stick axis and each
//! of A, B, Z is its own 1/f Gaussian stream over 2..2^16 frames; the stick
//! maps it to a uniform -128..127, a button is pressed while its stream is in
//! the top 4%.

use rand::Rng;
use rand::rngs::SmallRng;
use sm64_binds::GamePad;
use sm64_binds::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};

const OCTAVES: usize = 16;
/// z with P(N(0,1) > z) = 0.04.
const PRESS_Z: f32 = 1.750_686_1;

fn gauss(rng: &mut SmallRng) -> f32 {
    let u1: f32 = rng.random_range(1e-7..1.0);
    let u2: f32 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Sum of AR(1) processes with time constants 2, 4, ..., 2^OCTAVES frames,
/// equal power per octave (beta = 1); unit variance, stationary from the start.
struct Pink {
    rho: [f32; OCTAVES],
    gain: f32,
    x: [f32; OCTAVES],
}

impl Pink {
    fn new(rng: &mut SmallRng) -> Pink {
        let mut rho = [0.0; OCTAVES];
        let mut x = [0.0; OCTAVES];
        for k in 0..OCTAVES {
            rho[k] = 1.0 - 1.0 / 2f32.powi(k as i32 + 1);
            x[k] = gauss(rng);
        }
        Pink { rho, gain: (1.0 / OCTAVES as f32).sqrt(), x }
    }

    fn next(&mut self, rng: &mut SmallRng) -> f32 {
        let mut sum = 0.0;
        for k in 0..OCTAVES {
            let r = self.rho[k];
            self.x[k] = r * self.x[k] + (1.0 - r * r).sqrt() * gauss(rng);
            sum += self.x[k];
        }
        sum * self.gain
    }
}

/// Standard normal CDF (Abramowitz & Stegun 7.1.26).
fn phi(z: f32) -> f32 {
    let x = z as f64 / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    (0.5 * (1.0 + if x >= 0.0 { y } else { -y })) as f32
}

/// `n` frames of pink-noise input.
pub fn pink_inputs(rng: &mut SmallRng, n: usize) -> Vec<GamePad> {
    let mut ch: Vec<Pink> = (0..5).map(|_| Pink::new(rng)).collect();
    let stick = |z: f32| ((phi(z) * 256.0).floor() as i32 - 128).clamp(-128, 127) as i8;
    (0..n)
        .map(|_| {
            let (x, y) = (stick(ch[0].next(rng)), stick(ch[1].next(rng)));
            let mut button = 0;
            for (i, b) in [A_BUTTON, B_BUTTON, Z_TRIG].into_iter().enumerate() {
                if ch[2 + i].next(rng) > PRESS_Z {
                    button |= b;
                }
            }
            GamePad::new(button, x, y)
        })
        .collect()
}
