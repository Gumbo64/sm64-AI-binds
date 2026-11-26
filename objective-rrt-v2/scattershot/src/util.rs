use anyhow::Result;
use sm64_binds::GamePad;

/// Inputs from an m64 file: a Mupen64 movie (magic `M64\x1a`, a 0x400-byte
/// header) or a raw file with no header. Either way 4 bytes a frame: u16
/// big-endian buttons, i8 stick x, i8 stick y.
pub fn load_m64(path: &str) -> Result<Vec<GamePad>> {
    let bytes = obrrt_core::sys::fs::read(path)?;
    Ok(parse_m64(&bytes))
}

pub fn parse_m64(bytes: &[u8]) -> Vec<GamePad> {
    let body = if bytes.len() >= 0x400 && &bytes[..4] == b"M64\x1a" { &bytes[0x400..] } else { bytes };
    body.chunks_exact(4)
        .map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8))
        .collect()
}

/// Power-on to Mario: 1680 frames, A on even frames, START at 150..160 and 200..300
/// (moved here from the retired curiosity-GA crate, unchanged).
pub fn skip_menu_pads() -> Vec<sm64_binds::GamePad> {
    use sm64_binds::buttons;
    (0..1680)
        .map(|i| {
            let mut button = 0;
            if (150..160).contains(&i) || (200..300).contains(&i) {
                button = buttons::START_BUTTON;
            }
            if i % 2 == 0 {
                button = buttons::A_BUTTON;
            }
            sm64_binds::GamePad::new(button, 0, 0)
        })
        .collect()
}
