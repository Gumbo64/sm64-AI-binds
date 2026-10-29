//! Movement trace of a TAS (or any raw replay): one CSV row per frame with
//! inputs, Mario's kinematics, action, floor/wall geometry and the intro
//! Lakitu's state, for movement analysis (ref/tas/slope_techniques.md).
//!
//! tas_trace <rom> <movie.m64|raw> <out.csv> [from] [to] [--save N,N,... <dir>]
//!
//! Rows are written for frames from..=to (default all); the movie is played
//! from power-on either way. Frame N is the state after N input samples
//! (same numbering as `tasplay`). `--save` writes, for each N, the first N
//! inputs as a headerless raw file `<dir>/<movie stem>-f<N>.m64`: replaying
//! it (e.g. as `obrrt <rom> ... <start.m64>` with OBRRT_BOOT_NOWAIT=1, or the
//! labs' `<movie|raw>` argument) reproduces the state at frame N.
//!
//! Columns: frame, buttons (hex), sx, sy (raw stick), level, area, x, y, z,
//! vx, vy, vz, fvel, hspeed, yaw, action (hex), act_name (from the decomp's
//! include/sm64.h when found), act_timer, grounded (1 when the action is not
//! an air action), floor_y, nx, ny, nz (floor normal), floor_type (hex),
//! slope_deg, downhill_dyaw (floor's downhill yaw minus facing yaw; 0 =
//! facing straight downhill), wall_nx, wall_nz, air_peak (highest y since the
//! last ground frame), h_floor (y - floor_y), stars, lakitu (intro Lakitu's
//! oAction, -1 when it doesn't exist), in_box (Mario inside the Lakitu trigger
//! box: -544 < x < 545, y > 800, -2000 < z < -177), caught (in_box and the
//! action passes mario_ready_to_speak: the intro Lakitu, when it exists,
//! starts its cutscene on such a frame), prev_action, act_state, act_arg,
//! input (MarioState input flags, hex), int_yaw, int_mag (intended yaw and
//! magnitude 0..32), cam_yaw.
use anyhow::{Result, bail};
use obrrt_core::layout::*;
use obrrt_core::{GamePad, read_state};
use sm64_binds::SM64GameGenerator;
use std::collections::HashMap;
use std::io::Write;

const ACT_WAITING_FOR_DIALOG: u32 = 0x0000_130A;
const ACT_FIRST_PERSON: u32 = 0x0C00_0227;

fn action_names() -> HashMap<u32, String> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../sm64-binds/sm64-port/include/sm64.h");
    let mut out = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("no {path}: action names left as hex");
        return out;
    };
    for line in text.lines() {
        let mut it = line.split_whitespace();
        if it.next() != Some("#define") {
            continue;
        }
        let (Some(name), Some(val)) = (it.next(), it.next()) else { continue };
        if !name.starts_with("ACT_") || name.starts_with("ACT_FLAG") || name.starts_with("ACT_GROUP") || name.starts_with("ACT_ID") {
            continue;
        }
        if let Some(h) = val.strip_prefix("0x")
            && let Ok(v) = u32::from_str_radix(h, 16)
        {
            out.entry(v).or_insert_with(|| name[4..].to_string());
        }
    }
    out
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        bail!("usage: tas_trace <rom> <movie.m64|raw> <out.csv> [from] [to] [--save N,N,... <dir>]");
    }
    let bytes = std::fs::read(&args[2])?;
    let body: &[u8] = if bytes.len() >= 0x400 && &bytes[..4] == b"M64\x1a" {
        let samples = u32::from_le_bytes(bytes[0x18..0x1C].try_into().unwrap()) as usize;
        let data = &bytes[0x400..];
        &data[..(samples * 4).min(data.len() / 4 * 4)]
    } else {
        &bytes[..bytes.len() / 4 * 4]
    };
    let pads: Vec<GamePad> =
        body.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect();
    let mut pos_args = Vec::new();
    let mut saves: Vec<usize> = Vec::new();
    let mut save_dir = String::new();
    let mut i = 4;
    while i < args.len() {
        if args[i] == "--save" {
            saves = args[i + 1].split(',').map(|s| s.parse().unwrap()).collect();
            save_dir = args[i + 2].clone();
            i += 3;
        } else {
            pos_args.push(args[i].clone());
            i += 1;
        }
    }
    let from: usize = pos_args.first().map_or(1, |s| s.parse().unwrap());
    let to: usize = pos_args.get(1).map_or(pads.len(), |s| s.parse::<usize>().unwrap().min(pads.len()));
    let stem = std::path::Path::new(&args[2]).file_stem().unwrap().to_string_lossy().to_string();
    for &n in &saves {
        if n > pads.len() {
            bail!("save frame {n} past the movie's end");
        }
        let raw: Vec<u8> = body[..n * 4].to_vec();
        let p = format!("{save_dir}/{stem}-f{n}.m64");
        std::fs::write(&p, raw)?;
        eprintln!("wrote {p}");
    }
    let names = action_names();
    let lakitu_bhv = obrrt_core::bhv_addr("bhvCameraLakitu");

    let gg = SM64GameGenerator::from_file(&args[1])?;
    let mut game = gg.create_game()?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[3])?);
    writeln!(
        out,
        "frame,buttons,sx,sy,level,area,x,y,z,vx,vy,vz,fvel,hspeed,yaw,action,act_name,act_timer,grounded,floor_y,nx,ny,nz,floor_type,slope_deg,downhill_dyaw,wall_nx,wall_nz,air_peak,h_floor,stars,lakitu,in_box,caught,prev_action,act_state,act_arg,input,int_yaw,int_mag,cam_yaw"
    )?;
    let mut peak = f32::MIN;
    for (k, p) in pads[..to].iter().enumerate() {
        game.step_game(*p)?;
        let f = k + 1;
        let want = f >= from;
        let grounds = want && game.with_memory(|m| i16::from_le_bytes([m[ADDR_gCurrLevelNum as usize], m[ADDR_gCurrLevelNum as usize + 1]])) == 16;
        let s = read_state(&game, f as u32, grounds);
        let air = s.action & ACT_FLAG_AIR != 0;
        peak = if air { peak.max(s.pos[1]) } else { s.pos[1] };
        if !want {
            continue;
        }
        let lakitu = s
            .objects
            .as_ref()
            .and_then(|objs| {
                objs.iter().find(|o| Some(o.bhv) == lakitu_bhv && (o.beh_params >> 16) & 0xFF == 1).map(|o| o.action)
            })
            .unwrap_or(-1);
        let [nx, ny, nz] = s.floor_normal;
        let slope = ny.clamp(-1.0, 1.0).acos().to_degrees();
        // Downhill direction in SM64 yaw (atan2s(z, x) convention: yaw 0 = +z).
        let dyaw = if nx.abs() + nz.abs() > 1e-4 {
            let down = (nx.atan2(nz) / std::f32::consts::PI * 32768.0) as i32;
            ((down - s.yaw as i32) as i16) as i32
        } else {
            0
        };
        let (wx, wz) = s.wall_normal.map_or((0.0, 0.0), |w| (w[0], w[2]));
        let in_box = s.pos[0] > -544.0 && s.pos[0] < 545.0 && s.pos[1] > 800.0 && s.pos[2] > -2000.0 && s.pos[2] < -177.0;
        let name = names.get(&s.action).cloned().unwrap_or_else(|| format!("{:#x}", s.action));
        let hs = (s.vel[0] * s.vel[0] + s.vel[2] * s.vel[2]).sqrt();
        let ms = ADDR_gMarioStates as usize;
        let (input, prev, astate, aarg, imag, iyaw) = game.with_memory(|m| {
            let u16_ = |a: usize| u16::from_le_bytes([m[a], m[a + 1]]);
            let u32_ = |a: usize| u32::from_le_bytes(m[a..a + 4].try_into().unwrap());
            (u16_(ms + 2), u32_(ms + 0x10), u16_(ms + 0x18), u32_(ms + 0x1C), f32::from_bits(u32_(ms + 0x20)), u16_(ms + 0x24) as i16)
        });
        // mario_ready_to_speak (mario.c): what the intro Lakitu's
        // set_mario_npc_dialog needs; in the box, such a frame is caught.
        let group = s.action & ACT_GROUP_MASK;
        let ready = (group == 0 || group == 0x40 || s.action == ACT_WAITING_FOR_DIALOG)
            && s.action & (ACT_FLAG_INVULNERABLE | ACT_FLAG_RIDING_SHELL) == 0
            && s.action != ACT_FIRST_PERSON;
        let prev_name = names.get(&prev).cloned().unwrap_or_else(|| format!("{prev:#x}"));
        writeln!(
            out,
            "{f},{:#06x},{},{},{},{},{:.1},{:.1},{:.1},{:.2},{:.2},{:.2},{:.2},{:.2},{},{:#x},{name},{},{},{:.1},{nx:.3},{ny:.3},{nz:.3},{:#x},{slope:.1},{dyaw},{wx:.2},{wz:.2},{:.1},{:.1},{},{lakitu},{},{},{prev_name},{astate},{aarg},{input:#06x},{iyaw},{imag:.1},{}",
            p.button,
            p.stick_x,
            p.stick_y,
            s.level,
            s.area,
            s.pos[0],
            s.pos[1],
            s.pos[2],
            s.vel[0],
            s.vel[1],
            s.vel[2],
            s.forward_vel,
            hs,
            s.yaw,
            s.action,
            s.action_timer,
            !air as u8,
            s.floor_y,
            s.floor_type,
            if air { peak } else { s.pos[1] },
            s.pos[1] - s.floor_y,
            s.stars,
            in_box as u8,
            (in_box && ready) as u8,
            s.cam_yaw,
        )?;
    }
    eprintln!("{}: frames {from}..={to}", args[3]);
    Ok(())
}
