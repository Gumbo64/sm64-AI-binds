//! The process's [`SearchConfig`]: one typed value the search, the objectives and this crate read
//! their settings from (instead of each reading `OBRRT_*` variables itself).
//!
//! A program builds a config (from the environment with [`from_env`], from a file, or in code) and
//! [`install`]s it before the search starts. A program that never does gets the environment's, read
//! once at the first use, so the legacy `OBRRT_*` switches still work for every tool.
use std::sync::atomic::{AtomicPtr, Ordering};

pub use obrrt_env::*;

static CURRENT: AtomicPtr<SearchConfig> = AtomicPtr::new(std::ptr::null_mut());

/// The installed config; before any, the environment's (read once).
pub fn current() -> &'static SearchConfig {
    let p = CURRENT.load(Ordering::Acquire);
    if !p.is_null() {
        // SAFETY: only ever set to a leaked Box (never freed).
        return unsafe { &*p };
    }
    let cfg = from_env(SearchConfig::default()).unwrap_or_else(|e| {
        eprintln!("OBRRT_*: {e:#}; using the defaults for the lazily read switches");
        SearchConfig::default()
    });
    let fresh = Box::into_raw(Box::new(cfg));
    match CURRENT.compare_exchange(std::ptr::null_mut(), fresh, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => unsafe { &*fresh },
        Err(other) => {
            // SAFETY: `fresh` was never shared.
            drop(unsafe { Box::from_raw(fresh) });
            unsafe { &*other }
        }
    }
}

/// Makes `cfg` the process's config (replacing an earlier one; readers that already hold the old one keep it).
pub fn install(cfg: SearchConfig) -> &'static SearchConfig {
    let p = Box::into_raw(Box::new(cfg));
    CURRENT.store(p, Ordering::Release);
    unsafe { &*p }
}

/// `base` with the `OBRRT_*` variables of this process (the browser build's per-instance map there) applied.
pub fn from_env(base: SearchConfig) -> anyhow::Result<SearchConfig> {
    obrrt_env::from_env_with(base, &|k| crate::sys::env::var(k).ok())
}
