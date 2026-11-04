//! Writes the inputs from power-on to one node of a run's tree as an .m64.
//!
//! cargo run --release --bin nodepath -- <run_dir> <node id> [out.m64]

use anyhow::Result;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let id: usize = args[2].parse()?;
    let out = args.get(3).cloned().unwrap_or(format!("{dir}/node-{id}.m64"));
    let bytes = std::fs::read(format!("{dir}/tree.bin"))?;
    let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
    let count = u32_at(0) as usize;
    // Records: 0 is the boot, node n is record n + 1.
    let mut records = Vec::with_capacity(count);
    let mut at = 4;
    for _ in 0..count {
        let parent = u32_at(at) as i32;
        let n = u32_at(at + 4) as usize;
        records.push((parent, &bytes[at + 8..at + 8 + 4 * n]));
        at += 8 + 4 * n;
    }
    let mut chain = vec![id + 1];
    let mut node = id as i32;
    while node >= 0 {
        let parent = records[node as usize + 1].0;
        chain.push(if parent >= 0 { parent as usize + 1 } else { 0 });
        if parent < 0 {
            break;
        }
        node = parent;
    }
    let mut m64 = Vec::new();
    for r in chain.iter().rev() {
        m64.extend_from_slice(records[*r].1);
    }
    std::fs::write(&out, &m64)?;
    println!("{out}: {} frames", m64.len() / 4);
    Ok(())
}
