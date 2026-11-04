//! Route optimisation methods, one per file. Each implements
//! [`super::Method`]: propose trials (which waypoints to re-solve, from where,
//! with what GA settings and seed), hear how they went.

use super::Method;

pub mod beam;
pub mod direct;
pub mod essential;
pub mod full;
pub mod longjump;
pub mod perturb;
pub mod shortcut;
pub mod trim;
pub mod window;
pub mod window_delete;

/// All methods by name.
pub fn make(name: &str) -> Option<Box<dyn Method>> {
    Some(match name {
        "full" => Box::new(full::Full),
        "direct" => Box::new(direct::Direct::default()),
        "shortcut" => Box::new(shortcut::Shortcut::default()),
        "essential" => Box::new(essential::Essential::default()),
        "window" => Box::new(window::Window::default()),
        "window_delete" => Box::new(window_delete::WindowDelete::default()),
        "trim" => Box::new(trim::Trim::default()),
        "perturb" => Box::new(perturb::Perturb::default()),
        "beam" => Box::new(beam::Beam::default()),
        "longjump" => Box::new(longjump::LongJump::default()),
        _ => return None,
    })
}

pub const NAMES: &[&str] = &["full", "direct", "shortcut", "essential", "window", "window_delete", "trim", "perturb", "beam", "longjump"];
