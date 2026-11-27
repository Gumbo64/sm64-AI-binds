//! The process's [`SearchConfig`]: one typed value the search, the objectives and this crate read
//! their settings from (instead of each reading `OBRRT_*` variables itself).
//!
//! A program builds a config (from the environment with [`from_env`], from a file, or in code) and
//! [`install`]s it before the search starts. A program that never does gets the environment's, read
//! once at the first use, so the legacy `OBRRT_*` switches still work for every tool.
//!
//! One process can run several searches with different configs: each engine [`scope`]s its own config
//! on every thread it runs code on (its main thread and its attempts' worker threads), and
//! [`current`] answers with the scoped config there, the installed one elsewhere.
use std::cell::Cell;
use std::sync::atomic::{AtomicPtr, Ordering};

pub use obrrt_env::*;

static CURRENT: AtomicPtr<SearchConfig> = AtomicPtr::new(std::ptr::null_mut());

thread_local! {
    static SCOPED: Cell<Option<&'static SearchConfig>> = const { Cell::new(None) };
}

/// While alive, [`current`] on this thread answers with the config given to [`scope`].
pub struct Scope(Option<&'static SearchConfig>);

impl Drop for Scope {
    fn drop(&mut self) {
        SCOPED.with(|s| s.set(self.0));
    }
}

/// Makes `cfg` this thread's config until the guard drops (scopes nest).
pub fn scope(cfg: &'static SearchConfig) -> Scope {
    Scope(SCOPED.with(|s| s.replace(Some(cfg))))
}

/// A config that lives for the rest of the process (an engine keeps its config this way; they are few and small).
pub fn leak(cfg: SearchConfig) -> &'static SearchConfig {
    Box::leak(Box::new(cfg))
}

/// This thread's scoped config; else the installed one; before any, the environment's (read once).
pub fn current() -> &'static SearchConfig {
    if let Some(c) = SCOPED.with(|s| s.get()) {
        return c;
    }
    global()
}

/// The installed config; before any, the environment's (read once).
pub fn global() -> &'static SearchConfig {
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
