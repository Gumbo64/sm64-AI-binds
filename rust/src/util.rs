use std::fs::{self, File};
use std::path::Path;
use rand::prelude::*;
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use sm64_binds::{GamePad, buttons};

pub fn write_gamepads_to_file(gamepads: &[GamePad], filename: &str) -> std::io::Result<()> {
    if let Some(parent) = Path::new(filename).parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(filename)?;
    for gp in gamepads {
        file.write_u16::<BigEndian>(gp.button)?;
        file.write_i8(gp.stick_x)?;
        file.write_i8(gp.stick_y)?;
    }
    Ok(())
}

pub fn read_gamepads_from_file(filename: &str) -> std::io::Result<Vec<GamePad>> {
    let mut file = File::open(filename)?;
    let mut gamepads = Vec::new();

    loop {
        // Try to read the button; if we hit EOF cleanly, we're done
        let button = match file.read_u16::<BigEndian>() {
            Ok(v) => v,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        };

        let stick_x = file.read_i8()?;
        let stick_y = file.read_i8()?;

        gamepads.push(GamePad {
            button,
            stick_x,
            stick_y,
        });
    }

    Ok(gamepads)
}

pub fn random_gamepad(rng: &mut ThreadRng) -> GamePad {
    let stick_x = rng.random_range(-128..=127);
    let stick_y = rng.random_range(-128..=127);

    let mut button = 0;
    if rng.random_bool(0.04) { button |= buttons::A_BUTTON; }
    if rng.random_bool(0.04) { button |= buttons::B_BUTTON; }
    if rng.random_bool(0.04) { button |= buttons::Z_TRIG; }
    // if rng.random_bool(0.0001) {button |= buttons::START_BUTTON; }
    return GamePad::new(button, stick_x, stick_y)
}

pub fn skip_menu_pads() -> Vec<GamePad> {
    let mut path = Vec::new();
    for i in 0..1680 {
        let mut button = 0;
        if (150..160).contains(&i) || (200..300).contains(&i) {
            button = buttons::START_BUTTON;
        }
        if i % 2 == 0 {
            button = buttons::A_BUTTON; 
        }
        let pad = GamePad::new(button, 0, 0);
        path.push(pad);
    }
    path
}
/// The ROM, read at run time: `$SM64_ROM`, else `baserom.us.z64` at the
/// repository root (it used to be built in with `include_bytes!`, so the
/// crate couldn't compile without a ROM, e.g. as a git dependency).
pub fn rom_bytes() -> std::io::Result<Vec<u8>> {
    let path = std::env::var("SM64_ROM").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../baserom.us.z64").to_string());
    std::fs::read(&path).map_err(|e| std::io::Error::new(e.kind(), format!("{path}: {e}")))
}
