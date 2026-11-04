//! `full`: the reference method. Every trial re-solves the whole route: all
//! the waypoints the best route reached, as one chain from the root, seeded
//! with the best route's stream.

use rand::rngs::SmallRng;

use crate::route::{Ctx, GaSpec, Method, Trial};

pub struct Full;

impl Method for Full {
    fn name(&self) -> &'static str {
        "full"
    }
    fn propose(&mut self, ctx: &Ctx, _rng: &mut SmallRng) -> Trial {
        Trial {
            from: 0,
            stages: ctx.reached().into_iter().filter(|&i| i > 0).collect(),
            spec: GaSpec::default(),
            seed: None,
            mutation: None,
            tag: "full".into(), ..Default::default() }
    }
}
