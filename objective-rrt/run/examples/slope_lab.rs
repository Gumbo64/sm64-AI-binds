//! Lab for slope-speed techniques and the Lakitu skip from TAS states
//! (ref/tas/slope_techniques.md): plays a movie to frame N, then a scripted
//! variant, and reports every air phase (takeoff/landing position and speed,
//! peak height), Lakitu-box ground frames and when Mario reaches the door.
//!
//! slope_lab <rom> <movie.m64|raw> <N> "<script>" [--graft M] [--quiet]
//!
//! Script tokens (space separated), run in order:
//!   tas:K        play the movie's next K inputs unchanged (from frame N on)
//!   tasw:K       likewise, but with --graft the stick is rotated so it
//!                steers in the same world direction as in the movie
//!   yaw:V        hold the stick toward world yaw V (decimal or 0x hex; SM64
//!                yaw, 0 = +z, 0x8000 = -z, toward the castle door)
//!   aim:X,Z      hold the stick toward the point (X, Z), re-aimed each frame
//!   along        hold the stick along Mario's horizontal velocity
//!   tilt:F       stick tilt 0..1 (default 1)
//!   run:K        K frames, stick only
//!   hold:K:BTN   K frames, stick plus buttons BTN (letters A B Z, `-` none)
//!   land:BTN     stick only until a frame ends in a ground action, then one
//!                frame with BTN (so BTN acts on the landing action)
//!   air:BTN      likewise until a frame ends airborne
//!   flat:BTN     likewise until grounded on a floor with normal y > 0.99
//!   ffl:BTN      likewise until FREEFALL_LAND with actionTimer 3 (its last
//!                frame: the next frame's input acts on FREEFALL_LAND_STOP)
//!   land:W:BTN   (also air/flat/ffl:W:BTN) hold buttons W while waiting
//!   ptilt:F      stick tilt on the press frame of land/air/until (0 = tilt;
//!                0.38 keeps the stick under the speed-kick threshold so
//!                A-held + B on a landing is a jump kick, not a dive)
//!   until:Z:BTN  stick only until z < Z, then one frame with BTN
//!   savestate:P  write an emulator snapshot of the current state to P
//!   end:K        stick only for up to K more frames (default end: 90)
//!   setfv:V setx:V setz:V setyaw:V   overwrite Mario's forward speed, x, z
//!                or facing yaw now (sensitivity probes: the next frame
//!                starts from the edited state)
//! A button letter in lower case (`a`) means held from the previous frame
//! (no new press needs it released first: the lab releases A for one frame
//! before an upper-case `A` if A was down).
//!
//! --save-inputs P: write the movie's first N inputs plus the run's inputs
//! to P (headerless raw .m64; a prefix of k*4 bytes is the state at frame k).
//!
//! --load-state P: start from a snapshot written by the `savestate:P` token
//! (frame numbers then count from N).
//!
//! --graft M: copy Mario's movement fields (MarioState action..slideVelZ and
//! peakHeight) from the frame-N state into a second game played to frame M of
//! the same movie, and run the script there. With M on the first grounds visit
//! (frames 1646..1760 of the 2026 120-star TAS) the intro Lakitu exists, so
//! "caught" is the real cutscene (Mario in ACT_READING_NPC_DIALOG), not just
//! the box predicate.
use anyhow::{Result, bail};
use obrrt_core::layout::*;
use obrrt_core::{GamePad, St, read_state};
use sm64_binds::{SM64Game, SM64GameGenerator};

const ACT_READING_NPC_DIALOG: u32 = 0x2000_1306;
const ACT_WAITING_FOR_DIALOG: u32 = 0x0000_130A;
const ACT_FIRST_PERSON: u32 = 0x0C00_0227;
const ACT_PUSHING_DOOR: u32 = 0x0000_1321;
const ACT_FREEFALL_LAND: u32 = 0x0400_0471;

fn pads_of(path: &str) -> Result<Vec<GamePad>> {
    let bytes = std::fs::read(path)?;
    let body = if bytes.len() >= 0x400 && &bytes[..4] == b"M64\x1a" {
        let n = u32::from_le_bytes(bytes[0x18..0x1C].try_into().unwrap()) as usize;
        let d = &bytes[0x400..];
        d[..(n * 4).min(d.len() / 4 * 4)].to_vec()
    } else {
        bytes[..bytes.len() / 4 * 4].to_vec()
    };
    Ok(body.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect())
}

fn stick(s: &St, yaw: i32, tilt: f32) -> (i8, i8) {
    let rel = (yaw - s.cam_yaw as i32) as f32 * std::f32::consts::TAU / 65536.0;
    let t = tilt * 121.0;
    let raw = |v: f32| if v.abs() < 0.5 { 0 } else { (v + 6.0 * v.signum()).round().clamp(-127.0, 127.0) as i8 };
    (raw(t * rel.sin()), raw(-t * rel.cos()))
}

fn buttons(s: &str) -> (u16, bool) {
    let mut b = 0;
    let mut fresh_a = false;
    for c in s.chars() {
        match c {
            'A' => {
                b |= 0x8000;
                fresh_a = true;
            }
            'a' => b |= 0x8000,
            'B' | 'b' => b |= 0x4000,
            'Z' | 'z' => b |= 0x2000,
            _ => {}
        }
    }
    (b, fresh_a)
}

fn names() -> std::collections::HashMap<u32, String> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../sm64-binds/sm64-port/include/sm64.h");
    let mut out = std::collections::HashMap::new();
    for line in std::fs::read_to_string(path).unwrap_or_default().lines() {
        let mut it = line.split_whitespace();
        if it.next() != Some("#define") {
            continue;
        }
        if let (Some(n), Some(v)) = (it.next(), it.next())
            && n.starts_with("ACT_")
            && !n.starts_with("ACT_FLAG")
            && !n.starts_with("ACT_GROUP")
            && !n.starts_with("ACT_ID")
            && let Some(h) = v.strip_prefix("0x")
            && let Ok(v) = u32::from_str_radix(h, 16)
        {
            out.entry(v).or_insert_with(|| n[4..].to_string());
        }
    }
    out
}

fn in_box(s: &St) -> bool {
    s.pos[0] > -544.0 && s.pos[0] < 545.0 && s.pos[1] > 800.0 && s.pos[2] > -2000.0 && s.pos[2] < -177.0
}

fn ready_to_speak(a: u32) -> bool {
    let g = a & ACT_GROUP_MASK;
    (g == 0 || g == 0x40 || a == ACT_WAITING_FOR_DIALOG)
        && a & (ACT_FLAG_INVULNERABLE | ACT_FLAG_RIDING_SHELL) == 0
        && a != ACT_FIRST_PERSON
}

fn air(s: &St) -> bool {
    s.action & ACT_FLAG_AIR != 0
}

struct Run {
    game: SM64Game,
    frame: u32,
    st: St,
    prev_btn: u16,
    log: Vec<(GamePad, St)>,
    quiet: bool,
    names: std::collections::HashMap<u32, String>,
}

impl Run {
    fn step(&mut self, mut p: GamePad, fresh_a: bool) -> Result<()> {
        if fresh_a && self.prev_btn & 0x8000 != 0 {
            // A must be released for a frame to count as a new press.
            let q = GamePad::new(p.button & !0x8000, p.stick_x, p.stick_y);
            self.step(q, false)?;
            p = GamePad::new(p.button, p.stick_x, p.stick_y);
        }
        self.game.step_game(p)?;
        self.frame += 1;
        self.prev_btn = p.button;
        self.st = read_state(&self.game, self.frame, false);
        if !self.quiet {
            let s = &self.st;
            let n = self.names.get(&s.action).cloned().unwrap_or_else(|| format!("{:#x}", s.action));
            println!(
                "{:6} {:#06x} {:4} {:4} | {:7.1} {:6.1} {:7.1} vy {:6.1} fv {:6.1} {:<20} fy {:6.1} ny {:.3}{}{}",
                self.frame,
                p.button,
                p.stick_x,
                p.stick_y,
                s.pos[0],
                s.pos[1],
                s.pos[2],
                s.vel[1],
                s.forward_vel,
                n,
                s.floor_y,
                s.floor_normal[1],
                if in_box(s) { " BOX" } else { "" },
                if in_box(s) && ready_to_speak(s.action) { " CAUGHT?" } else { "" }
            );
        }
        self.log.push((p, self.st.clone()));
        Ok(())
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        bail!("usage: slope_lab <rom> <movie> <N> \"<script>\" [--graft M] [--quiet]");
    }
    let pads = pads_of(&args[2])?;
    let n: usize = args[3].parse()?;
    let script = args[4].clone();
    let graft: Option<usize> = args.iter().position(|a| a == "--graft").map(|i| args[i + 1].parse().unwrap());
    let quiet = args.iter().any(|a| a == "--quiet");
    let gg = SM64GameGenerator::from_file(&args[1])?;
    let mut game = gg.create_game()?;
    for p in &pads[..n] {
        game.step_game(*p)?;
    }
    // With --graft, the source game keeps playing the movie for `tasw`.
    let mut src: Option<SM64Game> = None;
    if let Some(m) = graft {
        let mut g2 = gg.create_game()?;
        for p in &pads[..m] {
            g2.step_game(*p)?;
        }
        let ms = ADDR_gMarioStates as usize;
        let (fields, peak) = game.with_memory(|mem| (mem[ms + 0x0C..ms + 0x60].to_vec(), mem[ms + 0xBC..ms + 0xC0].to_vec()));
        {
            let (store, _, memory) = g2.wasm_parts();
            let view = memory.view(store);
            view.write((ms + 0x0C) as u64, &fields)?;
            view.write((ms + 0xBC) as u64, &peak)?;
        }
        src = Some(std::mem::replace(&mut game, g2));
    }
    // --load-state P: start from a `savestate:` snapshot instead (its frame
    // is taken as N; `tas`/`tasw` then read the movie from N).
    if let Some(i) = args.iter().position(|a| a == "--load-state") {
        let st = sm64_binds::SaveState::from_bytes(std::fs::read(&args[i + 1])?)?;
        game.load_state(&st)?;
    }
    let frame0 = graft.unwrap_or(n) as u32;
    let st = read_state(&game, frame0, false);
    let mut r = Run { game, frame: frame0, st, prev_btn: pads[n - 1].button, log: Vec::new(), quiet, names: names() };
    let start = r.st.clone();
    let mut yaw: Option<i32> = None;
    let mut aim: Option<(f32, f32)> = None;
    let mut tilt = 1.0f32;
    let mut ptilt = 0.0f32;
    let mut tas_i = n;
    let stick_now = |r: &Run, yaw: Option<i32>, aim: Option<(f32, f32)>, tilt: f32| -> (i8, i8) {
        let y = match (aim, yaw) {
            (Some((x, _)), _) if x.is_nan() => (r.st.vel[0].atan2(r.st.vel[2]) / std::f32::consts::PI * 32768.0) as i32,
            (Some((x, z)), _) => {
                let dx = x - r.st.pos[0];
                let dz = z - r.st.pos[2];
                (dx.atan2(dz) / std::f32::consts::PI * 32768.0) as i32
            }
            (None, Some(y)) => y,
            (None, None) => r.st.yaw as i32,
        };
        stick(&r.st, y, tilt)
    };
    let mut toks: Vec<String> = script.split_whitespace().map(String::from).collect();
    if !toks.iter().any(|t| t.starts_with("end")) {
        toks.push("end:90".into());
    }
    'script: for t in &toks {
        let parts: Vec<&str> = t.split(':').collect();
        let num = |s: &str| -> i32 {
            if let Some(h) = s.strip_prefix("0x") { i32::from_str_radix(h, 16).unwrap() } else { s.parse().unwrap() }
        };
        match parts[0] {
            "tasw" => {
                // The movie's next K inputs with the stick rotated so that
                // Mario's intended yaw (world direction) matches the source
                // game's on the same input: a camera-yaw guess, then up to 3
                // corrections from the intended-yaw error (reloading the
                // frame each time).
                let ms = ADDR_gMarioStates as usize;
                let iyaw = |g: &SM64Game| g.with_memory(|m| i16::from_le_bytes([m[ms + 0x24], m[ms + 0x25]]) as i32);
                for _ in 0..num(parts[1]) {
                    let p = pads[tas_i];
                    tas_i += 1;
                    let Some(g) = src.as_mut() else {
                        r.step(p, false)?;
                        continue;
                    };
                    let cs = read_state(g, 0, false).cam_yaw as i32;
                    g.step_game(p)?;
                    let want = iyaw(g);
                    let rot = |p: GamePad, d: i32| {
                        let d = d as f32 * std::f32::consts::TAU / 65536.0;
                        let (x, y) = (p.stick_x as f32, p.stick_y as f32);
                        let (c, sn) = (d.cos(), d.sin());
                        let cl = |v: f32| v.round().clamp(-128.0, 127.0) as i8;
                        GamePad::new(p.button, cl(x * c + y * sn), cl(-x * sn + y * c))
                    };
                    let mut d = r.st.cam_yaw as i32 - cs;
                    let snap = r.game.save_state();
                    let mut q = rot(p, d);
                    r.game.step_game(q)?;
                    for _ in 0..3 {
                        if p.stick_x == 0 && p.stick_y == 0 {
                            break;
                        }
                        let err = ((want - iyaw(&r.game)) as i16) as i32;
                        if err.abs() <= 8 {
                            break;
                        }
                        d -= err;
                        r.game.load_state(&snap)?;
                        q = rot(p, d);
                        r.game.step_game(q)?;
                    }
                    r.game.load_state(&snap)?;
                    r.step(q, false)?;
                }
            }
            "tas" => {
                for _ in 0..num(parts[1]) {
                    let p = pads[tas_i];
                    tas_i += 1;
                    r.step(p, false)?;
                }
            }
            "yaw" => {
                yaw = Some(num(parts[1]));
                aim = None;
            }
            "along" => {
                // Stick along Mario's current horizontal velocity, re-aimed
                // every frame (sliding without the stick's loss or turn).
                aim = Some((f32::NAN, f32::NAN));
            }
            "aim" => {
                let v: Vec<f32> = parts[1].split(',').map(|x| x.parse().unwrap()).collect();
                aim = Some((v[0], v[1]));
            }
            "tilt" => tilt = parts[1].parse()?,
            "ptilt" => ptilt = parts[1].parse()?,
            "run" | "end" => {
                let k = parts.get(1).map_or(90, |s| num(s));
                for _ in 0..k {
                    let (sx, sy) = stick_now(&r, yaw, aim, tilt);
                    r.step(GamePad::new(0, sx, sy), false)?;
                    if r.st.action == ACT_PUSHING_DOOR || r.st.action == ACT_READING_NPC_DIALOG {
                        break 'script;
                    }
                }
            }
            "hold" => {
                let (b, fa) = buttons(parts[2]);
                for i in 0..num(parts[1]) {
                    let (sx, sy) = stick_now(&r, yaw, aim, tilt);
                    r.step(GamePad::new(b, sx, sy), fa && i == 0)?;
                }
            }
            "land" | "air" | "until" | "flat" | "ffl" => {
                let (b, fa) = buttons(parts.last().unwrap());
                let zlim = if parts[0] == "until" { parts[1].parse::<f32>()? } else { 0.0 };
                // land:W:BTN / air:W:BTN hold W while waiting.
                let wait = if parts[0] != "until" && parts.len() == 3 { buttons(parts[1]).0 } else { 0 };
                let pt = if ptilt > 0.0 { ptilt } else { tilt };
                for _ in 0..200 {
                    let done = match parts[0] {
                        "land" => !air(&r.st),
                        "air" => air(&r.st),
                        "flat" => !air(&r.st) && r.st.floor_normal[1] > 0.99,
                        "ffl" => r.st.action == ACT_FREEFALL_LAND && r.st.action_timer == 3,
                        _ => r.st.pos[2] < zlim,
                    };
                    let (sx, sy) = stick_now(&r, yaw, aim, tilt);
                    if done {
                        let (px, py) = stick_now(&r, yaw, aim, pt);
                        r.step(GamePad::new(b, px, py), fa)?;
                        break;
                    }
                    r.step(GamePad::new(wait, sx, sy), false)?;
                }
            }
            "savestate" => {
                // An emulator snapshot of the current state (sm64-binds
                // SaveState bytes; load with --load-state). Grafted states
                // (intro Lakitu present) can't be expressed as an input file.
                let path = t["savestate:".len()..].to_string();
                std::fs::write(&path, r.game.save_state().as_bytes())?;
                eprintln!("wrote {path} at frame {}", r.frame);
            }
            "setfv" | "setx" | "setz" | "setyaw" => {
                let ms = ADDR_gMarioStates as usize;
                let (off, bytes) = match parts[0] {
                    "setfv" => (0x54, parts[1].parse::<f32>()?.to_le_bytes().to_vec()),
                    "setx" => (0x3C, parts[1].parse::<f32>()?.to_le_bytes().to_vec()),
                    "setz" => (0x44, parts[1].parse::<f32>()?.to_le_bytes().to_vec()),
                    _ => (0x2E, (num(parts[1]) as i16).to_le_bytes().to_vec()),
                };
                let (store, _, memory) = r.game.wasm_parts();
                memory.view(store).write((ms + off) as u64, &bytes)?;
                r.st = read_state(&r.game, r.frame, false);
            }
            _ => bail!("bad token {t}"),
        }
        if r.st.action == ACT_READING_NPC_DIALOG {
            break;
        }
    }
    // --save-inputs <path>: the movie's first N inputs plus every input this
    // run played, as a headerless raw file (frame k = first k inputs).
    if let Some(i) = args.iter().position(|a| a == "--save-inputs") {
        if graft.is_some() {
            bail!("--save-inputs needs a run without --graft");
        }
        let mut raw = Vec::new();
        for p in pads[..n].iter().chain(r.log.iter().map(|(p, _)| p)) {
            raw.extend_from_slice(&p.button.to_be_bytes());
            raw.push(p.stick_x as u8);
            raw.push(p.stick_y as u8);
        }
        std::fs::write(&args[i + 1], raw)?;
        eprintln!("wrote {} ({} frames)", args[i + 1], n + r.log.len());
    }
    // Summary: air phases and box frames.
    let mut phases = Vec::new();
    let mut prev = start.clone();
    let mut cur: Option<(u32, St, f32)> = None;
    for (_, s) in &r.log {
        if air(s) && !air(&prev) {
            cur = Some((s.frame, prev.clone(), s.pos[1]));
        }
        if let Some((_, _, pk)) = cur.as_mut() {
            *pk = pk.max(s.pos[1]);
        }
        if !air(s) && air(&prev)
            && let Some((f0, take, pk)) = cur.take()
        {
            phases.push(format!(
                "air f{}-{} {} from ({:.0},{:.0},{:.0}) fv {:.1} -> land ({:.0},{:.0},{:.0}) fv {:.1}, peak {:.0} (+{:.0}), dist {:.0}",
                f0,
                s.frame,
                r.names.get(&prev.action).cloned().unwrap_or_default(),
                take.pos[0],
                take.pos[1],
                take.pos[2],
                take.forward_vel,
                s.pos[0],
                s.pos[1],
                s.pos[2],
                s.forward_vel,
                pk,
                pk - take.pos[1],
                ((s.pos[0] - take.pos[0]).powi(2) + (s.pos[2] - take.pos[2]).powi(2)).sqrt()
            ));
        }
        prev = s.clone();
    }
    for p in &phases {
        println!("  {p}");
    }
    let box_ground: Vec<u32> = r.log.iter().filter(|(_, s)| in_box(s) && ready_to_speak(s.action)).map(|(_, s)| s.frame).collect();
    let real = r.log.iter().find(|(_, s)| s.action == ACT_READING_NPC_DIALOG).map(|(_, s)| s.frame);
    let door = r.log.iter().find(|(_, s)| s.action == ACT_PUSHING_DOOR || s.pos[2] < -2950.0).map(|(_, s)| s.frame);
    let last = &r.st;
    println!(
        "SUMMARY start f{frame0} script \"{script}\" | box-ground frames {:?} | real lakitu caught {:?} | door-or-z<-2950 at {:?} ({} frames) | end ({:.0},{:.0},{:.0}) {}",
        box_ground,
        real,
        door,
        door.map_or(-1, |d| (d - frame0) as i32),
        last.pos[0],
        last.pos[1],
        last.pos[2],
        r.names.get(&last.action).cloned().unwrap_or_default()
    );
    Ok(())
}
