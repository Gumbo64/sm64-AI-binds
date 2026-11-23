//! Node save states, kept as the 4 KiB pages that differ from a base state,
//! XORed with it and LZ4-compressed (~20 KB instead of 8.6 MB). Each zone
//! (level, area) has its own base, the first state reached there: a state in
//! another level differs from the castle grounds in almost every page.

use anyhow::Result;
use crate::game::{GameSave, Pert};

const PAGE: usize = 4096;

#[derive(Default, Clone)]
pub struct Delta {
    /// Index of the base state this is relative to.
    pub base: u16,
    pages: Vec<u32>,
    xor: Vec<u8>,
    /// The perturbation state saved with the game (feature `hooked`; the
    /// root's default delta has none and is set when the engine is built).
    pert: Option<Pert>,
}

impl Delta {
    pub fn new(base: u16, root: &[u8], state: &GameSave) -> Delta {
        let bytes = crate::game::plain(state).as_bytes();
        let mut pages = Vec::new();
        let mut xor = Vec::new();
        for i in 0..bytes.len().div_ceil(PAGE) {
            let r = i * PAGE..((i + 1) * PAGE).min(bytes.len());
            if bytes[r.clone()] != root[r.clone()] {
                pages.push(i as u32);
                xor.extend(bytes[r.clone()].iter().zip(&root[r]).map(|(a, b)| a ^ b));
            }
        }
        Delta { base, pages, xor: lz4_flex::compress_prepend_size(&xor), pert: Some(crate::game::pert_of(state)) }
    }

    /// No save at all (sparse saves, `OBRRT_REPLAY_MAX`): the state is
    /// rebuilt from its saved ancestor (`Tree::load_into`). `apply` fails.
    pub fn none() -> Delta {
        Delta { base: u16::MAX, pages: Vec::new(), xor: Vec::new(), pert: None }
    }

    pub fn is_none(&self) -> bool {
        self.base == u16::MAX
    }

    pub fn bytes(&self) -> usize {
        self.xor.len() + 4 * self.pages.len()
    }

    /// Heap bytes held (capacities), for OBRRT_MEM_DIAG.
    pub fn heap_bytes(&self) -> usize {
        self.xor.capacity() + 4 * self.pages.capacity()
    }

    /// The full save state: its base's bytes with these pages XORed back in.
    pub fn apply(&self, bases: &[Vec<u8>]) -> Result<GameSave> {
        anyhow::ensure!(!self.is_none(), "no save state (sparse saves: load through Tree::load_into)");
        let mut bytes = bases[self.base as usize].clone();
        if !self.pages.is_empty() {
            let xor = lz4_flex::decompress_size_prepended(&self.xor)?;
            let mut rest = xor.as_slice();
            for &i in &self.pages {
                let at = i as usize * PAGE;
                let len = PAGE.min(bytes.len() - at);
                for (b, x) in bytes[at..at + len].iter_mut().zip(&rest[..len]) {
                    *b ^= x;
                }
                rest = &rest[len..];
            }
        }
        let save = sm64_binds::SaveState::from_bytes(bytes)?;
        #[cfg(feature = "hooked")]
        let pert = self.pert.as_ref().ok_or_else(|| anyhow::anyhow!("delta without its perturbation"))?;
        #[cfg(not(feature = "hooked"))]
        let pert = &();
        Ok(crate::game::wrap(save, pert))
    }
}
