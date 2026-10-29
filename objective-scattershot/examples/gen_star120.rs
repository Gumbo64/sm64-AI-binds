//! `cargo run --release --example gen_star120 -- [tools/analysis/levels.json] [schemas/star120.json]`
//!
//! Builds a 120-star schema from the level entry-script data
//! (`tools/parse_levels.py` -> levels.json): one Stage per level area, and in
//! each Stage the objective kinds that apply there. Which kinds apply is asked
//! of the kinds themselves, not hand-tabled:
//! - a kind belongs to a Stage if it writes a key for a state with Mario in
//!   that area and no objects (table-driven kinds: warp pads, paintings,
//!   missions), or
//! - for a state with one of the area's objects loaded next to Mario where it
//!   wrote nothing without it (stars, bosses, NPCs, switches, enemies ...).
//!
//! Technique kinds (tech_* families: they need a hand-placed region) and
//! kinds that answer in every area (they'd be in every Stage for no reason)
//! are left out, apart from the menu/dialog kinds, which go once at the root.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use anyhow::{Context, Result};
use objective_scattershot::objective::Tree;
use obrrt_core::{Obj, ObjId, St, bhv_addr};
use serde_json::{Value, json};

/// Kinds that answer in nearly every area but still belong in every course's
/// Stage (leaving a course, the red-coin and counter missions).
const COURSE_KINDS: &[&str] = &["ExitCourse", "MissionReachRedCoinStar", "MissionCounter"];
/// Bowser fights: Bowser is spawned by code, not placed in the level script.
const BOWSER_KINDS: &[&str] = &["BowserGrabTail", "BowserThrowAtBomb"];
/// Kinds whose objects only exist once something spawns them, or that need a
/// state the probe doesn't make (racing, in a cannon): added to an area that
/// has any of these placed behaviours.
const COMPANIONS: &[(&str, &[&str])] = &[
    ("KoopaRaceReachFlag", &["bhvKoopaRaceEndpoint"]),
    ("PenguinRaceWin", &["bhvRacingPenguin"]),
    ("SnowmanLeadBody", &["bhvSnowmansBottom"]),
    ("EyerokPunchEye", &["bhvEyerokBoss"]),
    ("ShellRide", &["bhvKoopa", "bhvKoopaShellUnderwater"]),
    ("WaterRingSwimThrough", &["bhvMantaRay", "bhvJetStreamRingSpawner"]),
    ("CannonFire", &["bhvCannon", "bhvWaterBombCannon", "bhvCannonClosed", "bhvBobombBuddyOpensCannon"]),
    ("CannonFlight", &["bhvCannon", "bhvWaterBombCannon", "bhvCannonClosed", "bhvBobombBuddyOpensCannon"]),
    ("CannonEnter", &["bhvCannon", "bhvWaterBombCannon", "bhvCannonClosed", "bhvBobombBuddyOpensCannon"]),
    ("PostPound", &["bhvChainChomp"]),
    ("MoneybagStomp", &["bhvMoneybagHidden"]),
    ("BookPunch", &["bhvHauntedBookshelfManager", "bhvHauntedBookshelf"]),
    ("BbhEntranceEnter", &["bhvBooWithCage"]),
    ("SunLookUp", &[]),
];
/// Kinds added wherever another kind is (pairs that work together).
const WITH_KIND: &[(&str, &str)] = &[("SunLookUp", "SunSpotStand")];
/// Radius of the `Near` gate around object-driven kinds.
const NEAR_RADIUS: f32 = 1500.0;
/// Kinds that go once under the root rather than per Stage.
const ROOT_KINDS: &[&str] = &["Busy", "DialogOption", "SaveMenuOption", "ActSelectOption"];
/// Families not mapped automatically.
const SKIP_FAMILIES: &[&str] = &["tech_blj", "tech_kick", "tech_misc", "lens", "progress", "boss", "busy"];
/// Kinds that are search scaffolding or need hand-set params.
const SKIP_KINDS: &[&str] = &["Still", "PosDist", "Side", "Lens", "Boss", "Stage", "Root", "NpcTalk", "EnemyHit", "WarpEnter", "CapCollect", "ChestOpen"];

/// Probe variants: (stars, keys, Mario's health, the object's action).
fn variants() -> Vec<(i16, u8, i16, i32)> {
    let mut v = Vec::new();
    for (stars, keys) in [(0, 0), (120, 2)] {
        for health in [0x880, 0x300] {
            for action in 0..5 {
                v.push((stars, keys, health, action));
            }
        }
    }
    v
}

fn state(level: u16, area: u8, course: i16, pos: [f32; 3]) -> St {
    St {
        frame: 0, pos, vel: [0.0; 3], forward_vel: 0.0, yaw: 0, pitch: 0, angle_vel: [0; 3], water_level: -11000,
        cam_yaw: 0, action: 0x0C40_0201, mario_flags: 0, holding: false, held_slot: None, red_coins: 0, level, area,
        act: 1, course, stars: 0, keys: 0, coins: 0, lives: 4, health: 0x880, play_mode: 0, transition: 0, dialog: -1,
        dialog_choice: false, menu_line: 0, menu_mode: -1, act_select: false, act_init: 0, act_cursor: 0,
        dialog_state: 0, dialog_last_page: false, menu_timer: 0, floor_y: pos[1], ceil_y: 20000.0, floor_type: 0,
        floor_normal: [0.0, 1.0, 0.0], floor_obj: None, floor_obj_vy: None, floor_obj_y: None, action_timer: 0,
        wall_normal: None, course_stars: [0; 25], save_flags: 0, objects: Some(Arc::new(Vec::new())),
    }
}

fn obj(bhv: u32, slot: u16, pos: [f32; 3], beh_params: u32, interact_type: u32, action: i32) -> Obj {
    Obj {
        id: ObjId { bhv, home: [pos[0] as i32, pos[1] as i32, pos[2] as i32] }, slot, bhv, pos, home: pos, interact_type,
        interact_status: 0, action, health: 3, value: 0, loot_coins: 0, beh_params, held_state: 0, intangible_timer: 0, yaw: 0,
        raw_1ac: [0; 4],
    }
}

/// Kind names registered in each family file's `build`, by family.
fn kinds() -> Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir("src/objectives")? {
        let path = entry?.path();
        let family = path.file_stem().unwrap().to_string_lossy().to_string();
        let src = std::fs::read_to_string(&path)?;
        let Some(start) = src.find("pub fn build(") else { continue };
        let body = &src[start..];
        let body = &body[..body.find("\n}\n").unwrap_or(body.len())];
        let mut rest = body;
        while let Some(i) = rest.find('"') {
            rest = &rest[i + 1..];
            let Some(j) = rest.find('"') else { break };
            let name = rest[..j].to_string();
            rest = &rest[j + 1..];
            let after = rest.trim_start();
            if name.starts_with(|c: char| c.is_ascii_uppercase()) && name.chars().all(|c| c.is_ascii_alphanumeric()) && (after.starts_with("=>") || after.starts_with('|')) {
                out.push((family.clone(), name));
            }
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

fn writes(tree: &Tree, s: &St) -> bool {
    !tree.key(s).is_empty()
}

fn camel(dir: &str) -> String {
    dir.split('_').map(|w| w[..1].to_uppercase() + &w[1..]).collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let input = args.get(1).map_or("tools/analysis/levels.json", |s| s.as_str());
    let output = args.get(2).map_or("schemas/star120.json", |s| s.as_str());
    let data: Value = serde_json::from_str(&std::fs::read_to_string(input).with_context(|| format!("reading {input}"))?)?;
    // Interaction types by behaviour (tools/parse_interact.py), so probe objects look as the game makes them.
    let interact: BTreeMap<String, u32> = serde_json::from_str(&std::fs::read_to_string("tools/analysis/interact.json").context("tools/analysis/interact.json: run tools/parse_interact.py")?)?;

    // Every candidate kind, loaded once with its default params.
    let mut trees = Vec::new();
    for (family, kind) in kinds()? {
        if SKIP_FAMILIES.contains(&family.as_str()) || SKIP_KINDS.contains(&kind.as_str()) || ROOT_KINDS.contains(&kind.as_str()) {
            continue;
        }
        match Tree::load(&format!(r#"{{"kind": "{kind}", "params": {{}}}}"#)) {
            Ok(t) => trees.push((family, kind, t)),
            Err(e) => eprintln!("skip {kind}: {e}"),
        }
    }

    // Probe every level area.
    struct Area {
        level: u16,
        area: u8,
        dir: String,
        course: i16,
        areas_in_level: usize,
        kinds: BTreeMap<String, BTreeSet<String>>, // kind -> reasons
    }
    let mut areas = Vec::new();
    let mut unknown_bhv = BTreeMap::<String, usize>::new();
    for lv in data["levels"].as_array().context("levels")? {
        let (level, course, dir) = (lv["id"].as_u64().unwrap() as u16, lv["course"].as_i64().unwrap_or(0) as i16, lv["dir"].as_str().unwrap().to_string());
        let list = lv["areas"].as_array().unwrap();
        for a in list {
            let area = a["id"].as_u64().unwrap() as u8;
            let objects = a["objects"].as_array().unwrap();
            if objects.is_empty() && a["warps"].as_array().is_none_or(|w| w.is_empty()) {
                continue;
            }
            let spawn = a["mario_pos"]["pos"].as_array().map(|p| [p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32, p[2].as_f64().unwrap() as f32]);
            let first = objects.first().map(|o| {
                let p = o["pos"].as_array().unwrap();
                [p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32, p[2].as_f64().unwrap() as f32]
            });
            let home = spawn.or(first).unwrap_or([0.0; 3]);
            let mut found: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
            let vary = |mut s: St, (stars, keys, health, _): (i16, u8, i16, i32)| {
                (s.stars, s.keys, s.health) = (stars, keys, health);
                s
            };
            for (_, kind, t) in &trees {
                if variants().into_iter().any(|v| writes(t, &vary(state(level, area, course, home), v))) {
                    found.entry(kind.clone()).or_default().insert("the area".into());
                }
            }
            for o in objects {
                let name = o["bhv"].as_str().unwrap();
                let Some(b) = bhv_addr(name) else {
                    *unknown_bhv.entry(name.to_string()).or_default() += 1;
                    continue;
                };
                let p = o["pos"].as_array().unwrap();
                let pos = [p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32, p[2].as_f64().unwrap() as f32];
                let mario = [pos[0] + 150.0, pos[1], pos[2]];
                let it = interact.get(name).copied().unwrap_or(0);
                let bp = o["beh_param"].as_u64().unwrap_or(0) as u32;
                for (_, kind, t) in &trees {
                    let hit = variants().into_iter().any(|v| {
                        let plain = vary(state(level, area, course, mario), v);
                        let mut with = plain.clone();
                        with.objects = Some(Arc::new(vec![obj(b, 1, pos, bp, it, v.3)]));
                        writes(t, &with) && !writes(t, &plain)
                    });
                    if hit {
                        found.entry(kind.clone()).or_default().insert(format!("{name} at ({}, {}, {})", pos[0], pos[1], pos[2]));
                    }
                }
            }
            for (kind, bhvs) in COMPANIONS {
                let placed: Vec<String> = objects.iter().filter_map(|o| o["bhv"].as_str()).filter(|b| bhvs.contains(b)).map(String::from).collect();
                if let Some(b) = placed.first() {
                    found.entry(kind.to_string()).or_default().insert(format!("{b} (spawns it or needs a state the probe can't make)"));
                }
            }
            for (kind, with) in WITH_KIND {
                if found.contains_key(*with) {
                    found.entry(kind.to_string()).or_default().insert(format!("goes with {with}"));
                }
            }
            if [30, 33, 34].contains(&level) {
                for k in BOWSER_KINDS {
                    found.entry(k.to_string()).or_default().insert("Bowser (spawned by the level, not in its script)".into());
                }
            }
            areas.push(Area { level, area, dir: dir.clone(), course, areas_in_level: list.len(), kinds: found });
        }
    }

    // Kinds that answer "the area" nearly everywhere say nothing about the area: drop them.
    let mut everywhere = BTreeMap::<String, usize>::new();
    for a in &areas {
        for (k, why) in &a.kinds {
            if why.contains("the area") {
                *everywhere.entry(k.clone()).or_default() += 1;
            }
        }
    }
    let ubiquitous: BTreeSet<String> = everywhere.iter().filter(|(_, n)| **n * 10 >= areas.len() * 8).map(|(k, _)| k.clone()).collect();

    // The schema.
    let mut stages = Vec::new();
    let mut used = BTreeMap::<String, usize>::new();
    let mut names = BTreeSet::new();
    for a in &areas {
        let mut children = Vec::new();
        for (k, why) in &a.kinds {
            let course_kind = COURSE_KINDS.contains(&k.as_str()) && (1..=15).contains(&a.course);
            if ubiquitous.contains(k) && why.len() == 1 && why.contains("the area") && !course_kind {
                continue;
            }
            *used.entry(k.clone()).or_default() += 1;
            let reasons: Vec<&String> = why.iter().take(4).collect();
            let more = if why.len() > 4 { format!(" and {} more", why.len() - 4) } else { String::new() };
            let stage = format!("{}{}", camel(&a.dir), if a.areas_in_level > 1 { a.area.to_string() } else { String::new() });
            let node = json!({"kind": k, "name": format!("{stage}{k}"),
                "doc": format!("For {}{}.", reasons.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("; "), more), "params": {}});
            // Object-driven kinds speak only near their objects (see the `Near`
            // kind): distances to objects that move would otherwise multiply
            // into a new state nearly every frame. Area-wide kinds stay as they are.
            let bhvs: BTreeSet<String> = why.iter().filter(|r| r.starts_with("bhv")).map(|r| r.split(' ').next().unwrap().to_string()).collect();
            if why.contains("the area") || bhvs.is_empty() {
                children.push(node);
            } else {
                children.push(json!({"kind": "Near", "name": format!("{stage}{k}Near"), "doc": format!("Within {NEAR_RADIUS} of: {}.", bhvs.iter().cloned().collect::<Vec<_>>().join(", ")),
                    "params": {"bhv": bhvs, "radius": NEAR_RADIUS}, "children": [node]}));
            }
        }
        // Bowser fights: the generic Boss kind needs its behaviour.
        if [30, 33, 34].contains(&a.level) {
            children.push(json!({"kind": "Boss", "name": format!("{}Boss", camel(&a.dir)), "doc": "Bowser: held, distance, his position, hp, action, facing; Mario's facing and spin while holding.", "params": {"bhv": "bhvBowser"}}));
        }
        let mut name = format!("{}{}", camel(&a.dir), if a.areas_in_level > 1 { a.area.to_string() } else { String::new() });
        while !names.insert(name.clone()) {
            name.push('_');
        }
        let mut params = json!({"level": a.level, "grid_log2": 8});
        if a.areas_in_level > 1 {
            params["area"] = a.area.into();
        }
        stages.push(json!({"kind": "Stage", "name": name, "doc": format!("Level {} ({}) area {}: {} objectives from its objects.", a.level, a.dir, a.area, children.len()), "params": params, "children": children}));
    }
    let mut root_children: Vec<Value> = ROOT_KINDS.iter().map(|k| json!({"kind": k, "name": k, "params": {}})).collect();
    root_children.extend(stages);
    let schema = json!({"kind": "Root", "name": "Route120", "doc": "120 stars: one Stage per level area from the level entry scripts (examples/gen_star120.rs), each with the objective kinds its objects call for.", "params": {"grid_log2": 10}, "children": root_children});
    let text = serde_json::to_string_pretty(&schema)?;
    let tree = Tree::load(&text)?; // it must load
    std::fs::write(output, text + "\n")?;

    println!("wrote {output}: {} objectives in {} stages", tree.insts.len(), areas.len());
    println!("dropped as answering in nearly every area: {ubiquitous:?}");
    println!("kinds used (in how many stages): {used:?}");
    let unused: Vec<&String> = trees.iter().map(|t| &t.1).filter(|k| !used.contains_key(*k)).collect();
    println!("kinds never used: {unused:?}");
    println!("behaviours with no address in obrrt_core (skipped): {unknown_bhv:?}");
    for a in &areas {
        println!("  {}:{} {} -> {}", a.level, a.area, a.dir, a.kinds.keys().filter(|k| !ubiquitous.contains(*k) || a.kinds[*k].len() > 1).cloned().collect::<Vec<_>>().join(" "));
    }
    Ok(())
}
