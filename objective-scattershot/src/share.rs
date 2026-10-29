//! Input sharing: replay the inputs one cell took since entering its zone
//! from another state that entered the same zone.

use std::collections::HashSet;

use anyhow::Result;
use rand::prelude::*;
use rand::rngs::SmallRng;
use sm64_binds::GamePad;

use crate::archive::Archive;
use crate::search::Search;

pub struct Sharer {
    tried: HashSet<(usize, usize)>,
    rng: SmallRng,
    pub attempts: u64,
    /// Cells added or improved by shared inputs.
    pub changed: u64,
}

/// The inputs from `node`'s zone entry to `node`.
pub fn since_entry(a: &Archive, node: usize) -> Vec<GamePad> {
    a.since_entry(node)
}

impl Sharer {
    pub fn new(seed: u64) -> Sharer {
        Sharer { tried: HashSet::new(), rng: SmallRng::seed_from_u64(seed), attempts: 0, changed: 0 }
    }

    pub fn share(&mut self, s: &mut Search) -> Result<()> {
        let a = &s.archive;
        // The donor: of 8 random cells, the one furthest into its zone.
        let (donor, inputs) = (0..8)
            .map(|_| a.cells[self.rng.random_range(0..a.cells.len())].node as usize)
            .map(|n| (n, a.since_entry(n)))
            .max_by_key(|d| d.1.len())
            .unwrap();
        let (zone, entry) = (a.nodes[donor].zone, a.nodes[donor].entry as usize);
        // Other entries of that zone that are still some cell's state.
        let live: HashSet<u32> = a.cells.iter().map(|c| c.node).collect();
        let others: Vec<usize> = (a.entries(zone).iter())
            .map(|&n| n as usize)
            .filter(|&n| n != entry && live.contains(&(n as u32)) && !self.tried.contains(&(entry, n)))
            .collect();
        if inputs.is_empty() || others.is_empty() {
            return Ok(());
        }
        let to = others[self.rng.random_range(0..others.len())];
        let Some(cell) = a.cells.iter().position(|c| c.node as usize == to) else { return Ok(()) };
        self.tried.insert((entry, to));
        self.attempts += 1;
        self.changed += s.play_from(cell, inputs, true)? as u64;
        Ok(())
    }
}
