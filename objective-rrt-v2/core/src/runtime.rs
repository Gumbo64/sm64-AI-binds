//! The run-time state an engine owns, as opposed to its settings ([`crate::config`]): the position
//! goal the web miner's bubble sets ([`crate::posgoal`]), the zones and epoch the mover's caches
//! are published at ([`crate::mover`]) and typed slots for the caches of the crates above this one
//! (doors, floor grids, the move GA's tallies: [`Runtime::ext`]).
//!
//! A program that never asks gets one process-wide runtime ([`Runtime::global`]), which is what
//! everything shared one global of before. An engine given its own ([`Runtime::new`], the search's
//! `Config::runtime`) runs under it on every thread it uses ([`scope`]), so two engines in one
//! process keep their goals, publications and caches apart. Not here: the static floor snapshots of
//! [`crate::geo`] (a function of the ROM and the zone, filled from whichever engine saw it first,
//! and read by the mover only through what a runtime has published) and the steering library
//! itself (one stateless controller, registered once per process).
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::posgoal::GoalState;

type Zone = (u16, u8);

pub struct Runtime {
    id: usize,
    /// The position goal ([`crate::posgoal`]).
    pub goal: GoalState,
    pub(crate) published: RwLock<HashSet<Zone>>,
    pub(crate) epoch: AtomicU64,
    ext: Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>,
}

impl Runtime {
    /// A fresh runtime: no goal, nothing published, no caches.
    pub fn new() -> Arc<Runtime> {
        static NEXT: AtomicUsize = AtomicUsize::new(1);
        Arc::new(Runtime {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            goal: GoalState::default(),
            published: RwLock::new(HashSet::new()),
            epoch: AtomicU64::new(0),
            ext: Mutex::new(HashMap::new()),
        })
    }

    /// The process-wide runtime, which a thread outside any engine's [`scope`] uses.
    pub fn global() -> &'static Arc<Runtime> {
        static G: OnceLock<Arc<Runtime>> = OnceLock::new();
        G.get_or_init(Runtime::new)
    }

    /// Distinguishes runtimes (for a cache keyed by thread that has to notice a switch).
    pub fn id(&self) -> usize {
        self.id
    }

    /// This runtime's one `T` (made with `T::default()` on first use): the home of a cache that
    /// belongs to the engine instead of the process.
    pub fn ext<T: Default + Send + Sync + 'static>(&self) -> Arc<T> {
        let mut m = self.ext.lock().unwrap();
        let v = m.entry(TypeId::of::<T>()).or_insert_with(|| Arc::new(T::default()) as Arc<dyn Any + Send + Sync>).clone();
        v.downcast::<T>().expect("ext slot holds its own type")
    }
}

thread_local! {
    static SCOPED: RefCell<Option<Arc<Runtime>>> = const { RefCell::new(None) };
}

/// While alive, [`current`] on this thread answers with the runtime given to [`scope`]. Not `Send`: it
/// restores the thread it was made on.
pub struct Scope(Option<Arc<Runtime>>, std::marker::PhantomData<*const ()>);

impl Drop for Scope {
    fn drop(&mut self) {
        SCOPED.with(|s| *s.borrow_mut() = self.0.take());
    }
}

/// Makes `rt` this thread's runtime until the guard drops (scopes nest).
pub fn scope(rt: Arc<Runtime>) -> Scope {
    Scope(SCOPED.with(|s| s.borrow_mut().replace(rt)), std::marker::PhantomData)
}

/// Calls `f` with this thread's runtime: the scoped one, else the process-wide one.
pub fn with<R>(f: impl FnOnce(&Runtime) -> R) -> R {
    SCOPED.with(|s| match s.borrow().as_ref() {
        Some(rt) => f(rt),
        None => f(Runtime::global()),
    })
}

/// This thread's runtime.
pub fn current() -> Arc<Runtime> {
    SCOPED.with(|s| s.borrow().clone()).unwrap_or_else(|| Runtime::global().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scopes_nest_and_keep_state_apart() {
        let a = Runtime::new();
        let b = Runtime::new();
        a.goal.set_weight(7.0);
        {
            let _sa = scope(a.clone());
            assert_eq!(with(|r| r.id()), a.id());
            {
                let _sb = scope(b.clone());
                assert_eq!(with(|r| r.id()), b.id());
                assert_eq!(with(|r| r.goal.weight()), 10.0);
            }
            assert_eq!(with(|r| r.goal.weight()), 7.0);
        }
        assert_eq!(with(|r| r.id()), Runtime::global().id());
        #[derive(Default)]
        struct Slot(std::sync::atomic::AtomicU32);
        a.ext::<Slot>().0.store(5, Ordering::Relaxed);
        assert_eq!(a.ext::<Slot>().0.load(Ordering::Relaxed), 5);
        assert_eq!(b.ext::<Slot>().0.load(Ordering::Relaxed), 0);
    }
}
