//! Checks the steering controller's conventions in the emulator: replays a
//! run's tree to one node, then plays phases of fixed or steered inputs and
//! prints Mario every frame.
//!
//! cargo run --release --bin steertest -- <rom> <run_dir> <node id> <phase>...
//!
//! Phases, played in order:
//! - `hold,N,BUTTONS,SX,SY`: N frames of one pad (BUTTONS in hex, e.g. 8000
//!   for A; SX/SY raw stick, y positive up);
//! - `steer,N,X,Y,Z`: N frames of `obrrt_search::steer::steer` towards X,Y,Z;
//! - `steer-rel,N,DX,DY,DZ`: the same towards the phase's start position plus
//!   DX,DY,DZ;
//! - `wingcap`: gives Mario the wing cap (sets MARIO_WING_CAP in his flags and
//!   a 1800-frame cap timer in memory), to test flying without a state that
//!   has it.
//!
//! Each steered phase ends with the closest distance it got to its target.

use anyhow::{Result, bail};
use obrrt_core::{Budget, GamePad, Objective, Progress, St, check_layout, dist, read_state};
use obrrt_search::route::read_tree_bin;
use obrrt_search::steer::steer;
use sm64_binds::SM64GameGenerator;

/// An objective that only names a point to steer to.
struct Point([f32; 3]);

impl Objective for Point {
    fn kind(&self) -> &'static str {
        "Steer::Test"
    }
    fn key(&self) -> String {
        format!("SteerTest {:?}", self.0)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        0.0
    }
    fn base_criteria(&self, _s: &St) -> bool {
        false
    }
    fn base_heuristic(&self, _s: &St) -> f64 {
        0.0
    }
    fn success(&self, _traj: &[St]) -> Option<usize> {
        None
    }
    fn progress(&self, _traj: &[St]) -> Progress {
        Progress { score: 0.0, at: 0 }
    }
    fn budget(&self) -> Budget {
        Budget::Macro(Vec::new())
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.0)
    }
}

fn print(f: usize, s: &St, pad: GamePad, target: Option<[f32; 3]>) {
    let d = target.map_or(String::new(), |t| format!(" d {:6.0} hd {:6.0}", dist(s.pos, t), s.hdist(t)));
    println!(
        "{f:4} pad {:04x} {:4} {:4} | pos {:7.0} {:7.0} {:7.0} vel {:6.1} {:6.1} {:6.1} fv {:5.1} yaw {:#06x} pitch {:6} av {:5} {:5} act {:#010x} wl {:6}{d}",
        pad.button, pad.stick_x, pad.stick_y, s.pos[0], s.pos[1], s.pos[2], s.vel[0], s.vel[1], s.vel[2], s.forward_vel,
        s.yaw as u16, s.pitch, s.angle_vel[0], s.angle_vel[1], s.action, s.water_level,
    );
}

/// MARIO_WING_CAP (include/sm64.h).
const MARIO_WING_CAP: u32 = 0x8;

fn wing_cap(game: &mut sm64_binds::SM64Game) -> Result<()> {
    use obrrt_core::layout::{ADDR_gMarioStates, MARIO_capTimer, MARIO_flags};
    let flags_at = ADDR_gMarioStates as u64 + MARIO_flags as u64;
    let mut buf = [0u8; 4];
    game.read_memory(flags_at as u32, &mut buf)?;
    let flags = u32::from_le_bytes(buf) | MARIO_WING_CAP;
    let (store, _, memory) = game.wasm_parts();
    let view = memory.view(store);
    view.write(flags_at, &flags.to_le_bytes())?;
    view.write(ADDR_gMarioStates as u64 + MARIO_capTimer as u64, &1800u16.to_le_bytes())?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        bail!("usage: steertest <rom> <run_dir> <node id> <phase>...");
    }
    let game_gen = SM64GameGenerator::from_file(&args[1])?;
    let node: usize = args[3].parse()?;
    // tree.bin: record 0 is the boot, node n is record n + 1.
    let records = read_tree_bin(&format!("{}/tree.bin", args[2]))?;
    let mut chain = vec![node + 1];
    while let Some(&r) = chain.last() {
        let parent = records[r].0;
        if r == 0 {
            break;
        }
        chain.push(if parent >= 0 { parent as usize + 1 } else { 0 });
    }
    let mut game = game_gen.create_game()?;
    let mut frame = 0u32;
    for r in chain.iter().rev() {
        for p in &records[*r].1 {
            game.step_game(*p)?;
            frame += 1;
        }
    }
    check_layout(&game)?;
    let mut s = read_state(&game, frame, false);
    println!("node {node}: frame {frame} level {} area {} cam_yaw {:#06x}", s.level, s.area, s.cam_yaw as u16);
    print(0, &s, GamePad::new(0, 0, 0), None);
    let mut f = 0;
    for phase in &args[4..] {
        let p: Vec<&str> = phase.split(',').collect();
        if p[0] == "wingcap" {
            wing_cap(&mut game)?;
            println!("-- wingcap");
            continue;
        }
        let n: usize = p[1].parse()?;
        let num = |i: usize| -> Result<f32> { Ok(p[i].parse()?) };
        let target = match p[0] {
            "hold" => None,
            "steer" => Some([num(2)?, num(3)?, num(4)?]),
            "steer-rel" => Some([s.pos[0] + num(2)?, s.pos[1] + num(3)?, s.pos[2] + num(4)?]),
            other => bail!("unknown phase {other}"),
        };
        println!("-- {phase} target {target:?}");
        let fixed = if target.is_none() {
            GamePad::new(u16::from_str_radix(p[2], 16)?, p[3].parse()?, p[4].parse()?)
        } else {
            GamePad::new(0, 0, 0)
        };
        let mut best = (f32::MAX, 0);
        for k in 0..n {
            let pad = match target {
                Some(t) => steer(&Point(t), &s, k, 4),
                None => fixed,
            };
            game.step_game(pad)?;
            frame += 1;
            f += 1;
            s = read_state(&game, frame, false);
            print(f, &s, pad, target);
            if let Some(t) = target {
                if dist(s.pos, t) < best.0 {
                    best = (dist(s.pos, t), f);
                }
            }
        }
        if target.is_some() {
            println!("-- closest {:.0} at frame {}", best.0, best.1);
        }
    }
    Ok(())
}
