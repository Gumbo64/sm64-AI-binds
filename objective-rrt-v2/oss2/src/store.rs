//! Save states kept small: the 4 KB pages that differ from a reference state,
//! each as its index and the XOR, LZ4-compressed. Each area (level * 256 +
//! area) has its own reference, the first state saved there: a state in
//! another level differs from the castle grounds in almost every page
//! (objective-rrt's `store.rs` does the same).
//!
//! A packed save: u16 reference index, u32 uncompressed length, LZ4 data.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::{Result, bail};
use sm64_binds::SaveState;

const PAGE: usize = 4096;

thread_local! {
    // Per-thread buffers: big short-lived allocations fragment the heap
    // (RSS was 4x the saves without them).
    static SCRATCH: RefCell<(Vec<u8>, Vec<u8>)> = Default::default();
}

pub struct Packer {
    refs: RwLock<(Vec<Arc<Vec<u8>>>, HashMap<u32, u16>)>,
}

impl Packer {
    /// `reference` is reference 0, used for `zone` (and for nothing else).
    pub fn new(reference: &SaveState, zone: u32) -> Packer {
        let refs = (vec![Arc::new(reference.as_bytes().to_vec())], HashMap::from([(zone, 0)]));
        Packer { refs: RwLock::new(refs) }
    }

    /// The reference for `zone`, made from `s` if the area has none yet.
    fn reference(&self, zone: u32, s: &SaveState) -> (u16, Arc<Vec<u8>>) {
        {
            let r = self.refs.read().unwrap();
            if let Some(&i) = r.1.get(&zone) {
                return (i, r.0[i as usize].clone());
            }
        }
        let mut r = self.refs.write().unwrap();
        if let Some(&i) = r.1.get(&zone) {
            return (i, r.0[i as usize].clone());
        }
        let i = r.0.len() as u16;
        r.0.push(Arc::new(s.as_bytes().to_vec()));
        r.1.insert(zone, i);
        (i, r.0[i as usize].clone())
    }

    /// Number of references held (one per area seen), and their bytes.
    pub fn references(&self) -> (usize, usize) {
        let r = self.refs.read().unwrap();
        (r.0.len(), r.0.iter().map(|b| b.len()).sum())
    }

    pub fn pack(&self, s: &SaveState, zone: u32) -> Box<[u8]> {
        let (id, reference) = self.reference(zone, s);
        SCRATCH.with_borrow_mut(|(out, z)| {
            out.clear();
            for (i, (a, b)) in s.as_bytes().chunks(PAGE).zip(reference.chunks(PAGE)).enumerate() {
                if a != b {
                    out.extend((i as u32).to_le_bytes());
                    out.extend(a.iter().zip(b).map(|(a, b)| a ^ b));
                }
            }
            z.resize(6 + lz4_flex::block::get_maximum_output_size(out.len()), 0);
            z[..2].copy_from_slice(&id.to_le_bytes());
            z[2..6].copy_from_slice(&(out.len() as u32).to_le_bytes());
            let n = lz4_flex::block::compress_into(out, &mut z[6..]).unwrap();
            z[..6 + n].into()
        })
    }

    pub fn unpack(&self, save: &[u8]) -> Result<SaveState> {
        let id = u16::from_le_bytes(save[..2].try_into()?) as usize;
        let reference = {
            let r = self.refs.read().unwrap();
            let Some(b) = r.0.get(id) else { bail!("save refers to reference {id}, which does not exist") };
            b.clone()
        };
        let mut bytes = reference.as_ref().clone();
        SCRATCH.with_borrow_mut(|(x, _)| -> Result<()> {
            x.resize(u32::from_le_bytes(save[2..6].try_into()?) as usize, 0);
            lz4_flex::block::decompress_into(&save[6..], x)?;
            let mut rest = x.as_slice();
            while !rest.is_empty() {
                let at = u32::from_le_bytes(rest[..4].try_into()?) as usize * PAGE;
                let len = PAGE.min(bytes.len() - at);
                for (b, x) in bytes[at..at + len].iter_mut().zip(&rest[4..4 + len]) {
                    *b ^= x;
                }
                rest = &rest[4 + len..];
            }
            Ok(())
        })?;
        SaveState::from_bytes(bytes)
    }
}
