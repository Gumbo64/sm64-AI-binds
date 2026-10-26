//! Check example for St's read-only collision fields (floor_y, ceil_y,
//! floor type/normal/object, action_timer, wall) and the static queries
//! `geo::floor_at` / `geo::ceil_at`: replays a movie and prints them per
//! frame over ranges.
//!
//! core_fields <rom> <movie.m64> <from-to>[,<from-to>...]
use anyhow::Result;
use obrrt_core::{GamePad, geo, read_state};
use sm64_binds::SM64GameGenerator;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let ranges: Vec<(u32, u32)> = a[3]
        .split(',')
        .map(|r| {
            let (x, y) = r.split_once('-').unwrap();
            (x.parse().unwrap(), y.parse().unwrap())
        })
        .collect();
    let end = ranges.iter().map(|r| r.1).max().unwrap();
    let gen_ = SM64GameGenerator::from_file(&a[1])?;
    let mut game = gen_.create_game()?;
    let bytes = std::fs::read(&a[2])?;
    let n = u32::from_le_bytes(bytes[0x18..0x1C].try_into().unwrap()) as usize;
    let pads: Vec<GamePad> = bytes[0x400..]
        .chunks_exact(4)
        .take(n)
        .map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8))
        .collect();
    for (i, p) in pads.iter().enumerate().take(end as usize) {
        game.step_game(*p)?;
        let f = i as u32 + 1;
        if !ranges.iter().any(|r| (r.0..=r.1).contains(&f)) {
            continue;
        }
        let s = read_state(&game, f, true);
        let (sf, sc) = game.with_memory(|m| {
            (geo::floor_at(m, s.pos[0], s.pos[1], s.pos[2]), geo::ceil_at(m, s.pos[0], s.floor_y + 80.0, s.pos[2]))
        });
        let obj = s.floor_obj.map_or(String::new(), |slot| {
            let name = s.objects().iter().find(|o| o.slot == slot).map_or("?", |o| o.bhv_name());
            format!(" obj#{slot} {name} vy {:.1} y {:.1}", s.floor_obj_vy.unwrap(), s.floor_obj_y.unwrap())
        });
        println!(
            "{f:6} L{}a{} act {:#010x} t{:<3} pos {:8.1} {:7.1} {:8.1} fv {:7.1} | floor {:7.1} type {:#04x} ny {:.4}{obj} | ceil {:7.1} | wall {} | static floor {} ceil {}",
            s.level,
            s.area,
            s.action,
            s.action_timer,
            s.pos[0],
            s.pos[1],
            s.pos[2],
            s.forward_vel,
            s.floor_y,
            s.floor_type,
            s.floor_normal[1],
            s.ceil_y,
            s.wall_normal.map_or("-".into(), |n| format!("{:.2},{:.2},{:.2}", n[0], n[1], n[2])),
            sf.map_or("none".into(), |h| format!("{:.1}/{:#04x}/{:.4}", h.y, h.kind, h.normal[1])),
            sc.map_or("none".into(), |h| format!("{:.1}", h.y)),
        );
    }
    Ok(())
}
