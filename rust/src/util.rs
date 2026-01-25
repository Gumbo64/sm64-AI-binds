use std::fs::{self, File};
use std::path::Path;
use rand::prelude::*;
use byteorder::{BigEndian, WriteBytesExt};
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

pub fn random_gamepad(rng: &mut ThreadRng) -> GamePad {
    let stick_x = rng.random_range(-128..=127);
    let stick_y = rng.random_range(-128..=127);

    let mut button = 0;
    if rng.random_bool(0.04) { button |= buttons::A_BUTTON; }
    if rng.random_bool(0.04) { button |= buttons::B_BUTTON; }
    if rng.random_bool(0.04) { button |= buttons::Z_TRIG; }
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