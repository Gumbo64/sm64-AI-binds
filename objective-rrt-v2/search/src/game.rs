//! The game type the rolling-horizon path plays. By default plain
//! `SM64Game`; with feature `hooked`, [`HookedGame`]: the game plus a
//! [`StepHook`] that decides the pad each frame actually plays (e.g. a
//! consensus perturbation supplied by a downstream crate), so every
//! step/copy/sync/save/load a worker does goes through the hook. Passing a
//! hooked game where `&mut SM64Game` is typed would silently step it
//! unhooked, so the RH path uses these.
//!
//! This crate knows nothing about what the hook does: downstream crates
//! implement [`StepHook`] and build a [`HookedGame`].

#[cfg(not(feature = "hooked"))]
pub type Game = sm64_binds::SM64Game;
#[cfg(not(feature = "hooked"))]
pub type GameSave = sm64_binds::SaveState;
#[cfg(feature = "hooked")]
pub type Game = HookedGame;
#[cfg(feature = "hooked")]
pub type GameSave = HookedSave;

#[cfg(feature = "hooked")]
pub use hooked::*;

#[cfg(feature = "hooked")]
mod hooked {
    use anyhow::Result;
    use sm64_binds::{GamePad, SM64Game, SM64GameGenerator, SaveState};
    use std::any::Any;

    /// Plays one frame of a game. The hook's own state is part of the game
    /// state: it is cloned with forks and saved with save states.
    pub trait StepHook: Send + Sync + 'static {
        /// Plays one frame of `game`, wanting `wanted`; returns the pad played.
        fn step(&mut self, game: &mut SM64Game, wanted: GamePad) -> Result<GamePad>;
        fn clone_box(&self) -> Box<dyn StepHook>;
        /// The hook's state as bytes, for a save that crosses to another worker (see
        /// [`set_pert_decoder`]); empty when it has none.
        fn export(&self) -> Vec<u8> {
            Vec::new()
        }
        /// For downstream crates to recover their concrete hook.
        fn as_any(&self) -> &dyn Any;
    }

    impl Clone for Box<dyn StepHook> {
        fn clone(&self) -> Self {
            self.clone_box()
        }
    }

    type PertDecoder = fn(&[u8]) -> Result<Box<dyn StepHook>>;
    static PERT_DECODER: std::sync::OnceLock<PertDecoder> = std::sync::OnceLock::new();

    /// Registers how a hook's [`StepHook::export`] bytes become a hook again (once per process).
    pub fn set_pert_decoder(f: PertDecoder) {
        let _ = PERT_DECODER.set(f);
    }

    pub fn pert_bytes(p: &Box<dyn StepHook>) -> Vec<u8> {
        p.export()
    }

    pub fn pert_from_bytes(b: &[u8]) -> Result<Box<dyn StepHook>> {
        (PERT_DECODER.get().ok_or_else(|| anyhow::anyhow!("no perturbation decoder registered (game::set_pert_decoder)"))?)(b)
    }

    /// A game whose every frame goes through `hook`.
    pub struct HookedGame {
        pub game: SM64Game,
        pub hook: Box<dyn StepHook>,
        /// Makes forks (a hooked game can't come from a generator alone).
        pub generator: &'static SM64GameGenerator,
    }

    /// A save state of a [`HookedGame`]: the memory and the hook's state.
    #[derive(Clone)]
    pub struct HookedSave {
        pub game: SaveState,
        pub hook: Box<dyn StepHook>,
    }

    impl HookedGame {
        pub fn new(game: SM64Game, hook: Box<dyn StepHook>, generator: &'static SM64GameGenerator) -> HookedGame {
            HookedGame { game, hook, generator }
        }

        /// Plays one frame through the hook.
        pub fn step_game(&mut self, wanted: GamePad) -> Result<()> {
            self.step(wanted).map(|_| ())
        }

        /// Plays one frame through the hook; returns the pad played.
        pub fn step(&mut self, wanted: GamePad) -> Result<GamePad> {
            self.hook.step(&mut self.game, wanted)
        }

        /// A copy (a new game instance with the same memory and hook state).
        pub fn fork(&self) -> Result<HookedGame> {
            Ok(HookedGame { game: self.generator.copy_from(&self.game)?, hook: self.hook.clone(), generator: self.generator })
        }

        pub fn copy_from(&mut self, source: &HookedGame) -> Result<()> {
            self.game.copy_from(&source.game)?;
            self.hook = source.hook.clone();
            Ok(())
        }

        /// Becomes a copy of `base` (cheap when repeated from the same base).
        pub fn sync_from(&mut self, base: &HookedGame) -> Result<()> {
            self.game.sync_from(&base.game)?;
            self.hook = base.hook.clone();
            Ok(())
        }

        pub fn save_state(&self) -> HookedSave {
            HookedSave { game: self.game.save_state(), hook: self.hook.clone() }
        }

        pub fn load_state(&mut self, s: &HookedSave) -> Result<()> {
            self.game.load_state(&s.game)?;
            self.hook = s.hook.clone();
            Ok(())
        }
    }

    impl std::ops::Deref for HookedGame {
        type Target = SM64Game;
        fn deref(&self) -> &SM64Game {
            &self.game
        }
    }

    impl std::ops::DerefMut for HookedGame {
        fn deref_mut(&mut self) -> &mut SM64Game {
            &mut self.game
        }
    }
}

/// The plain save inside a [`GameSave`] (for the default-only state tree; it
/// is never reached with `hooked`, whose `Worker::new` fails).
#[cfg(not(feature = "hooked"))]
pub fn plain(s: &GameSave) -> &sm64_binds::SaveState {
    s
}
#[cfg(feature = "hooked")]
pub fn plain(s: &GameSave) -> &sm64_binds::SaveState {
    &s.game
}

/// The plain save of a [`GameSave`], by value.
#[cfg(not(feature = "hooked"))]
pub fn into_plain(s: GameSave) -> sm64_binds::SaveState {
    s
}
#[cfg(feature = "hooked")]
pub fn into_plain(s: GameSave) -> sm64_binds::SaveState {
    s.game
}
/// The hook state a save carries (with `hooked`; nothing otherwise).
#[cfg(not(feature = "hooked"))]
pub type Pert = ();
#[cfg(feature = "hooked")]
pub type Pert = Box<dyn StepHook>;
#[cfg(not(feature = "hooked"))]
pub fn pert_of(_: &GameSave) -> Pert {}
#[cfg(feature = "hooked")]
pub fn pert_of(s: &GameSave) -> Pert {
    s.hook.clone()
}
#[cfg(not(feature = "hooked"))]
pub fn wrap(s: sm64_binds::SaveState, _: &Pert) -> GameSave {
    s
}
#[cfg(feature = "hooked")]
pub fn wrap(s: sm64_binds::SaveState, p: &Pert) -> GameSave {
    HookedSave { game: s, hook: p.clone() }
}

#[cfg(not(feature = "hooked"))]
pub fn pert_bytes(_: &Pert) -> Vec<u8> {
    Vec::new()
}
#[cfg(not(feature = "hooked"))]
pub fn pert_from_bytes(_: &[u8]) -> anyhow::Result<Pert> {
    Ok(())
}
