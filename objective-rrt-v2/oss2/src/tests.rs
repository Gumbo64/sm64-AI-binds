use std::collections::HashSet;

use obrrt_core::St;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use sm64_binds::SaveState;

use crate::archive::{Archive, Found, Rollout};
use crate::objective::{Cx, Inst, Key, Objective, Tree};

fn st(pos: [f32; 3], fv: f32, level: u16, area: u8, keys: u8) -> St {
    St {
        frame: 0, pos, vel: [0.0; 3], forward_vel: fv, yaw: 0, pitch: 0, angle_vel: [0; 3], water_level: -11000,
        cam_yaw: 0, action: 0x0400_0440, mario_flags: 0, holding: false, held_slot: None, red_coins: 0, level, area,
        act: 0, course: 0, stars: 0, keys, coins: 0, lives: 4, health: 0x880, play_mode: 0, transition: 0, dialog: -1,
        dialog_choice: false, menu_line: 0, menu_mode: -1, act_select: false, act_init: 0, act_cursor: 0,
        dialog_state: 0, dialog_last_page: false, menu_timer: 0, floor_y: 0.0, ceil_y: 20000.0, floor_type: 0,
        floor_normal: [0.0, 1.0, 0.0], floor_obj: None, floor_obj_vy: None, floor_obj_y: None, action_timer: 0,
        wall_normal: None, course_stars: [0; 25], save_flags: 0, objects: None,
    }
}

const LOBBY: &str = r#"{"kind": "Root", "params": {"grid_log2": 10}, "children": [
    {"kind": "Stage", "name": "Lobby", "params": {"level": 6, "area": 1, "grid_log2": 8}, "children": [
        {"kind": "Lens", "name": "Steps", "params": {"center": [-2230, 0, 303], "radius": 500, "payoff_speed": 60},
         "children": [{"kind": "Side", "params": {"point": [0, 0, 0], "normal": [1, 0, 0], "radius": 1e9}}]},
        {"kind": "Side", "params": {"point": [-2652, 512, -1463], "normal": [0.7, 0, 0.7], "radius": 3000}}]},
    {"kind": "Stage", "name": "BitDW", "params": {"level": 17}}]}"#;

#[test]
fn dfs_ids() {
    let t = Tree::load(LOBBY).unwrap();
    let kids: Vec<Vec<usize>> = t.insts.iter().map(|i| i.children.clone()).collect();
    assert_eq!(kids, vec![vec![1, 5], vec![2, 4], vec![3], vec![], vec![], vec![]]);
    let ends: Vec<usize> = t.insts.iter().map(|i| i.end).collect();
    assert_eq!(ends, vec![6, 5, 4, 4, 5, 6]);
    assert_eq!(t.insts[2].name.as_deref(), Some("Steps"));
}

/// Fields of objective `id` in `key`.
fn of(key: &Key, id: u16) -> Vec<(u8, i32)> {
    key.iter().filter(|e| e.0 == id).map(|e| (e.1, e.2)).collect()
}

#[test]
fn lens_cases() {
    let t = Tree::load(LOBBY).unwrap();
    // Inside: pos, hspeed, yaw, action, and the lens's child.
    let k = t.key(&st([-2230.0, 0.0, 303.0], -9.0, 6, 1, 0));
    let f: Vec<u8> = of(&k, 2).iter().map(|e| e.0).collect();
    assert_eq!(f, vec![0, 1, 2, 3]);
    assert_eq!(of(&k, 2)[1].1, -3);
    // Walking is class 1 of the default [walking, long_jump].
    assert_eq!(of(&k, 2)[3], (3, 1));
    assert_eq!(of(&k, 3), vec![(0, 0)]);
    // Outside but fast: payoff and hspeed only, no children.
    let k = t.key(&st([-2652.0, 512.0, -1900.0], -300.0, 6, 1, 0));
    assert_eq!(of(&k, 2), vec![(4, 1), (1, -75)]);
    assert!(of(&k, 3).is_empty());
    // Outside and slow: nothing.
    let k = t.key(&st([0.0, 0.0, 0.0], 10.0, 6, 1, 0));
    assert!(of(&k, 2).is_empty());
    // Another level: the lobby subtree is silent, BitDW writes nothing but is active.
    let k = t.key(&st([0.0, 0.0, 0.0], 10.0, 17, 1, 0));
    assert_eq!(k.len(), 4);
}

/// Writes after evaluating its child, to check the header is restored.
struct After;
impl Objective for After {
    fn fields(&self) -> &[&'static str] {
        &["x"]
    }
    fn eval(&self, _: &St, cx: &mut Cx) {
        cx.all_children();
        cx.put(0, 7);
    }
}

#[test]
fn headers_owned_by_core() {
    let inst = |id, children, end| Inst {
        id, kind: "After".into(), name: None, doc: None, params: serde_json::Value::Null, children,
        fields: vec!["x"], end, obj: Box::new(After),
    };
    let t = Tree { insts: vec![inst(0, vec![1], 3), inst(1, vec![2], 3), inst(2, vec![], 3)], objects: false, groups: vec![] };
    let k = t.key(&st([0.0; 3], 0.0, 6, 1, 0));
    assert_eq!(k, vec![(2, 0, 7), (1, 0, 7), (0, 0, 7)]);
}

#[test]
fn nested_grids_add_no_keys() {
    let t = Tree::load(LOBBY).unwrap();
    let mut keys = HashSet::new();
    let mut fine = HashSet::new();
    for i in 0..4000 {
        let p = [(i * 37 % 4096) as f32 - 6000.0, 100.0, (i * 91 % 4096) as f32 + 2000.0];
        keys.insert(t.key(&st(p, 0.0, 6, 1, 0)));
        fine.insert(((p[0] / 256.0).floor() as i32, (p[2] / 256.0).floor() as i32));
    }
    assert_eq!(keys.len(), fine.len());
}

fn archive() -> Archive {
    let mut b = b"SM64SAVE".to_vec();
    b.extend(1u32.to_le_bytes());
    b.extend([0u8; 20]);
    b.extend(0u32.to_le_bytes());
    b.extend(16u64.to_le_bytes());
    b.extend([5u8; 16]);
    let s = SaveState::from_bytes(b).unwrap();
    Archive::new(&s, &s, vec![(0, 0, 0)], 0)
}

fn rollout(src: usize, a: &Archive, keys: &[i32]) -> Rollout {
    let c = &a.cells[src];
    let found = keys.iter().enumerate().map(|(t, &v)| Found { key: vec![(0, 0, v)], t, save: None, group: 0, busy: false, zone: 0, entered: false, pos: [0; 3] }).collect();
    let inputs = vec![sm64_binds::GamePad::new(0, 0, 0); keys.len()];
    Rollout { src, node: c.node as usize, frames: c.frames, p: c.p(), inputs, found }
}

#[test]
fn bayes() {
    let mut a = archive();
    // A new cell: success, and the new cell's prior comes from the source.
    (a.cells[0].visits, a.cells[0].succ) = (10.0, 8.0);
    assert!(a.merge(rollout(0, &a, &[1])).changed > 0);
    assert_eq!((a.cells[0].visits, a.cells[0].succ), (11.0, 9.0));
    assert_eq!((a.cells[1].a0, a.cells[1].b0), (1.5, 0.5));
    // Improving the source cell itself is no success.
    a.cells[1].frames = 100;
    let mut r = rollout(1, &a, &[1]);
    r.frames = 0;
    assert_eq!(a.merge(r).changed, 0);
    let c = &a.cells[1];
    assert_eq!((c.visits, c.succ, c.frames), (1.0, 0.0, 1));
    // Improving another cell is.
    a.cells[0].frames = 50;
    assert!(a.merge(rollout(1, &a, &[0])).changed > 0);
}

#[test]
fn thompson_prefers_success() {
    let mut a = archive();
    a.merge(rollout(0, &a, &[1, 2]));
    (a.cells[0].visits, a.cells[0].succ) = (20.0, 0.0);
    a.cells[1].visits = 20.0;
    a.cells[1].succ = 18.0;
    a.cells[2].visits = 20.0;
    a.cells[2].succ = 1.0;
    let mut rng = SmallRng::seed_from_u64(1);
    let n = (0..1000).filter(|_| a.select(&mut rng, 256) == 1).count();
    assert!(n > 900, "{n}");
}

#[test]
fn replacement_resets_evidence() {
    let mut a = archive();
    a.merge(rollout(0, &a, &[1]));
    let (v0, s0) = (a.cells[0].visits, a.cells[0].succ);
    // Creation shrinks nothing: the source keeps its evidence (plus this visit).
    a.merge(rollout(0, &a, &[2]));
    assert_eq!((a.cells[0].visits, a.cells[0].succ), (v0 + 1.0, s0 + 1.0));
    // Cell 1 gathers evidence, then cell 2 finds a faster path to it.
    (a.cells[1].visits, a.cells[1].succ, a.cells[1].frames) = (8.0, 4.0, 100);
    (a.cells[2].visits, a.cells[2].succ) = (6.0, 6.0);
    assert_eq!(a.replaced, 0);
    assert!(a.merge(rollout(2, &a, &[1])).changed > 0);
    let c = &a.cells[1];
    assert_eq!((c.visits, c.succ, c.frames, a.replaced), (2.0, 1.0, 2, 1));
    // The prior comes from cell 2's posterior mean at the time.
    let m = (a.cells[2].a0 + 6.0) / (a.cells[2].a0 + a.cells[2].b0 + 6.0);
    assert!((c.a0 - 2.0 * m).abs() < 1e-6 && (c.b0 - 2.0 * (1.0 - m)).abs() < 1e-6);
}

#[test]
fn zone_entries() {
    let mut a = archive();
    let found = |v, t, zone, entered| Found { key: vec![(0, 0, v)], t, save: Some(vec![1].into()), group: 0, busy: false, zone, entered, pos: [0; 3] };
    let mut r = rollout(0, &a, &[]);
    r.inputs = vec![sm64_binds::GamePad::new(0, 0, 0); 10];
    r.found = vec![found(1, 2, 5, true), found(2, 4, 5, false)];
    a.merge(r);
    // Chained: node 2 hangs under node 1 with only its own inputs.
    assert_eq!((a.nodes[1].parent, a.inputs(1).len(), a.nodes[1].entry), (0, 3, 1));
    assert_eq!((a.nodes[2].parent, a.inputs(2).len(), a.nodes[2].entry), (1, 2, 1));
    let mut r = rollout(2, &a, &[]);
    r.inputs = vec![sm64_binds::GamePad::new(0, 0, 0); 10];
    r.found = vec![found(3, 0, 5, false), found(4, 1, 6, true), found(5, 2, 6, false)];
    a.merge(r);
    let entries: Vec<u32> = a.nodes.iter().map(|n| n.entry).collect();
    assert_eq!(entries, vec![0, 1, 1, 1, 4, 4]);
    assert_eq!(crate::share::since_entry(&a, 3).len(), 2 + 1);
    assert_eq!(a.cells[3].frames, 6);
}

#[test]
fn pack_round_trip() {
    let state = |fill: u8, n: usize| {
        let mut b = b"SM64SAVE".to_vec();
        b.extend(1u32.to_le_bytes());
        b.extend([0u8; 20]);
        b.extend(0u32.to_le_bytes());
        b.extend((n as u64).to_le_bytes());
        b.extend((0..n).map(|i| if i % 5000 == 7 { fill } else { 3 }));
        SaveState::from_bytes(b).unwrap()
    };
    let a = Archive::new(&state(3, 10000), &state(3, 10000), vec![], 0);
    let s = state(9, 10000);
    assert_eq!(a.unpack(&a.pack(&s, 0)).unwrap(), s);
    // Another area gets its own reference: the first state saved there.
    let (other, again) = (state(4, 10000), state(5, 10000));
    let first = a.pack(&other, 7);
    assert!(first.len() < a.pack(&other, 0).len(), "packed against itself it is nearly empty");
    assert_eq!(a.unpack(&first).unwrap(), other);
    assert_eq!(a.unpack(&a.pack(&again, 7)).unwrap(), again);
    assert_eq!(a.unpack(&a.pack(&s, 0)).unwrap(), s, "area 0 still uses the first reference");
    assert_eq!(a.references().0, 2);
}

#[test]
fn boss_keys_the_boss_and_the_hold() {
    use obrrt_core::{Obj, ObjId, bhv_addr};
    let bhv = bhv_addr("bhvBowser").expect("bhvBowser");
    let bowser = |slot: u16, pos: [f32; 3], health: i32| Obj {
        id: ObjId { bhv, home: [0; 3] }, slot, bhv, pos, home: [0.0; 3], interact_type: 0, interact_status: 0, action: 7,
        health, value: 0, loot_coins: 0, beh_params: 0, held_state: 0, intangible_timer: 0, yaw: 0x4000, raw_1ac: [0; 4],
    };
    let t = Tree::load(r#"{"kind": "Boss", "params": {"bhv": "bhvBowser"}}"#).unwrap();
    let mut s = st([0.0, 100.0, 0.0], 0.0, 30, 1, 0);
    s.objects = Some(std::sync::Arc::new(vec![bowser(3, [500.0, 0.0, 0.0], 2), bowser(4, [5000.0, 0.0, 0.0], 3)]));
    let f = |k: &Key| -> Vec<i32> { of(k, 0).iter().map(|e| e.1).collect() };
    let k = of(&t.key(&s), 0);
    let get = |k: &Vec<(u8, i32)>, i: u8| k.iter().find(|e| e.0 == i).unwrap().1;
    // Not held: the nearest Bowser (500 away, 2 hit points, action 7, facing 0x4000 = sector 4).
    assert_eq!((get(&k, 1), get(&k, 2), get(&k, 4), get(&k, 5), get(&k, 6), get(&k, 7)), (0, 2, 2, 7, 4, 1));
    assert_eq!((get(&k, 8), get(&k, 9)), (-1, 0));
    assert_eq!(f(&t.key(&s)).len(), 10);
    // Held: that one, plus Mario's facing and spin.
    s.holding = true;
    s.held_slot = Some(4);
    s.yaw = 0x8000u16 as i16;
    s.angle_vel = [0, 1024, 0];
    let k = of(&t.key(&s), 0);
    assert_eq!((get(&k, 1), get(&k, 4), get(&k, 8), get(&k, 9)), (1, 3, 8, 4));
}

#[test]
fn lens_vertical_speed_and_optional_hspeed() {
    let make = |extra: &str| {
        Tree::load(&format!(
            r#"{{"kind": "Lens", "params": {{"center": [0, 0, 0], "radius": 100{extra}}}}}"#
        ))
        .unwrap()
    };
    let mut s = st([0.0, 0.0, 0.0], 30.0, 6, 1, 0);
    s.vel = [0.0, 20.0, 0.0];
    // Default: hspeed in steps of 4, no vspeed.
    let k = of(&make("").key(&s), 0);
    assert!(k.iter().any(|e| e.0 == 1 && e.1 == 7) && !k.iter().any(|e| e.0 == 5));
    // hspeed off, vspeed in steps of 8.
    let k = of(&make(r#", "speed_step": 0, "vspeed_step": 8"#).key(&s), 0);
    assert!(!k.iter().any(|e| e.0 == 1));
    assert_eq!(k.iter().find(|e| e.0 == 5).unwrap().1, 2);
}

#[test]
fn lens_only_while_its_object_exists() {
    use obrrt_core::{Obj, ObjId, bhv_addr};
    let bhv = bhv_addr("bhvCameraLakitu").expect("bhvCameraLakitu");
    let lakitu = |param2: u32| Obj {
        id: ObjId { bhv, home: [0; 3] }, slot: 1, bhv, pos: [0.0; 3], home: [0.0; 3], interact_type: 0, interact_status: 0,
        action: 0, health: 0, value: 0, loot_coins: 0, beh_params: param2 << 16, held_state: 0, intangible_timer: 0, yaw: 0,
        raw_1ac: [0; 4],
    };
    let t = Tree::load(r#"{"kind": "Lens", "params": {"center": [0, 0, 0], "radius": 100, "while_object": "bhvCameraLakitu", "while_param2": 1}}"#).unwrap();
    assert!(t.objects, "the tree asks for objects");
    let mut s = st([0.0, 0.0, 0.0], 0.0, 16, 1, 0);
    // No Lakitu (after the intro): nothing is written, even inside the region.
    s.objects = Some(std::sync::Arc::new(vec![]));
    assert!(of(&t.key(&s), 0).is_empty());
    // A Lakitu with the wrong parameter byte is not the intro one.
    s.objects = Some(std::sync::Arc::new(vec![lakitu(0)]));
    assert!(of(&t.key(&s), 0).is_empty());
    // The intro Lakitu: the lens is on.
    s.objects = Some(std::sync::Arc::new(vec![lakitu(1)]));
    assert!(!of(&t.key(&s), 0).is_empty());
    // Outside the region it is silent as usual.
    s.pos = [5000.0, 0.0, 0.0];
    assert!(of(&t.key(&s), 0).is_empty());
}

#[test]
fn caught_states_share_one_cell() {
    let t = Tree::load(
        r#"{"kind": "Root", "params": {"grid_log2": 10}, "children": [
            {"kind": "Stage", "params": {"level": 16, "area": 1, "grid_log2": 8}, "children": [
                {"kind": "Lens", "name": "Skip", "params": {"min": [-1000, 0, -3000], "max": [1000, 2000, 1000], "caught_actions": ["dialog"], "caught_dialog": 34}},
                {"kind": "Side", "name": "Door", "params": {"point": [0, 0, -3000], "normal": [0, 0, -1], "radius": 1e9}}]}]}"#,
    )
    .unwrap();
    // Free states in different places keep their positions apart.
    let a = t.key(&st([100.0, 900.0, -500.0], 10.0, 16, 1, 0));
    let b = t.key(&st([-300.0, 900.0, -1500.0], 10.0, 16, 1, 0));
    assert_ne!(a, b);
    // Caught by the dialog action, or by dialog 34: one key, wherever they are,
    // with the progress fields only and nothing from the objectives after it.
    let mut c = st([100.0, 900.0, -500.0], 10.0, 16, 1, 0);
    c.action = 0x0000_130A;
    let mut d = st([-300.0, 900.0, -1500.0], 0.0, 16, 1, 0);
    d.dialog = 34;
    let (kc, kd) = (t.key(&c), t.key(&d));
    assert_eq!(kc, kd);
    let fields: Vec<(u16, u8)> = kc.iter().map(|e| (e.0, e.1)).collect();
    assert_eq!(fields, vec![(0, 0), (0, 1), (0, 2), (2, 6)], "level, keys, stars, then caught: no positions, no Side");
    // Another level's progress stays separate.
    let mut e = st([0.0; 3], 0.0, 16, 1, 1);
    e.action = 0x0000_130A;
    assert_ne!(t.key(&e), kc);
}

// ---------- hierarchical sampling ----------

const STAGES: &str = r#"{"kind": "Root", "params": {"grid_log2": 10}, "children": [
    {"kind": "Stage", "name": "Grounds", "params": {"level": 16, "area": 1, "grid_log2": 8}},
    {"kind": "Stage", "name": "Lobby", "params": {"level": 6, "area": 1, "max_keys": 0}},
    {"kind": "Stage", "name": "KeyLobby", "params": {"level": 6, "area": 1, "min_keys": 1, "max_keys": 1}}]}"#;

#[test]
fn groups_follow_the_stages() {
    let t = Tree::load(STAGES).unwrap();
    assert_eq!(t.groups, vec![1, 2, 3]);
    assert_eq!(t.rank(), vec![0, 1, 2, 3]);
    // With a grid the stage also writes a position; without one it writes nothing,
    // and the group still comes from the state.
    assert_eq!(t.group(&st([0.0; 3], 0.0, 16, 1, 0)), 1);
    assert_eq!(t.group(&st([0.0; 3], 0.0, 6, 1, 0)), 2);
    assert_eq!(t.group(&st([0.0; 3], 0.0, 6, 1, 1)), 3);
    assert_eq!(t.group(&st([0.0; 3], 0.0, 17, 1, 0)), 0, "no stage: the root");
    assert_eq!(t.group(&st([0.0; 3], 0.0, 6, 2, 0)), 0, "wrong area");
}

/// An archive with `n` cells in group 1 and `m` in group 2 (never expanded).
fn two_groups(n: usize, m: usize, grouped: bool) -> Archive {
    let mut a = archive().with_groups(vec![0, 1, 2], grouped);
    for (group, count) in [(1u16, n), (2, m)] {
        for _ in 0..count {
            let i = a.cells.len();
            a.cells.push(crate::archive::Cell { key: vec![(0, 1, i as i32)].into(), node: 0, frames: 10, visits: 0.0, succ: 0.0, a0: 1.0, b0: 1.0, group, busy: false, pos: [0; 3] });
            a.groups[group as usize].cells.push(i as u32);
        }
    }
    a
}

#[test]
fn selection_is_two_level() {
    // 100,000 cells in one group and 10 in another: flat sampling almost never
    // reaches the small group, group sampling gives it about its share of groups.
    let share = |grouped: bool| {
        let a = two_groups(100_000, 10, grouped);
        let mut rng = SmallRng::seed_from_u64(3);
        (0..2000).filter(|_| a.cells[a.select(&mut rng, 256)].group == 2).count() as f64 / 2000.0
    };
    let (flat, grouped) = (share(false), share(true));
    assert!(flat < 0.10, "flat {flat}");
    assert!(grouped > 0.30, "grouped {grouped}");
    // The same seed picks the same cells.
    let a = two_groups(1000, 10, true);
    let run = |seed| {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..50).map(|_| a.select(&mut rng, 256)).collect::<Vec<_>>()
    };
    assert_eq!(run(9), run(9));
}

#[test]
fn group_reward_needs_new_ground() {
    let mut a = archive().with_groups(vec![0, 1, 2], true);
    let found = |key: i32, group: u16, t: usize| Found { key: vec![(0, 0, key)], t, save: None, group, busy: false, zone: 0, entered: false, pos: [0; 3] };
    let roll = |a: &Archive, src: usize, found: Vec<Found>| {
        let c = &a.cells[src];
        let n = found.last().map_or(1, |f| f.t + 1);
        Rollout { src, node: c.node as usize, frames: c.frames, p: c.p(), inputs: vec![sm64_binds::GamePad::new(0, 0, 0); n], found }
    };
    // Root (group 0) reaches a new cell in group 1: a later group, so progress.
    let r = roll(&a, 0, vec![found(10, 1, 0)]);
    a.merge(r);
    assert_eq!((a.groups[0].visits, a.groups[0].succ), (1.0, 1.0));
    assert_eq!(a.groups[1].cells, vec![1]);
    // From that cell (group 1): a new cell in its own group counts.
    let r = roll(&a, 1, vec![found(11, 1, 0)]);
    a.merge(r);
    assert_eq!((a.groups[1].visits, a.groups[1].succ), (1.0, 1.0));
    // A new cell in an earlier group does not.
    let r = roll(&a, 1, vec![found(12, 0, 0)]);
    a.merge(r);
    assert_eq!((a.groups[1].visits, a.groups[1].succ), (2.0, 1.0));
    // Nor does only improving a known cell, though the cell itself counts that as a success.
    let (cell, frames) = (a.find(&[(0, 0, 11)]).unwrap(), 1);
    a.cells[cell].frames = 1000;
    let mut r = roll(&a, 1, vec![found(11, 1, 0)]);
    r.frames = 0;
    let before = a.cells[1].succ;
    a.merge(r);
    assert!(a.cells[cell].frames <= frames + 1);
    assert_eq!((a.groups[1].visits, a.groups[1].succ), (3.0, 1.0));
    assert_eq!(a.cells[1].succ, before + 1.0);
    // best_frames tracks the fewest in the group.
    assert_eq!(a.groups[1].best_frames, a.cells.iter().filter(|c| c.group == 1).map(|c| c.frames).min().unwrap());
}

/// `cargo test --release -- --ignored --nocapture selection_cost`
#[test]
#[ignore]
fn selection_cost() {
    for grouped in [false, true] {
        let a = two_groups(1_000_000, 1000, grouped);
        let mut rng = SmallRng::seed_from_u64(1);
        let n = 20_000;
        let t = std::time::Instant::now();
        let mut sink = 0;
        for _ in 0..n {
            sink ^= a.select(&mut rng, 256);
        }
        let per = t.elapsed().as_secs_f64() / n as f64;
        println!("{} select over 1M cells: {:.1} us each ({:.0} per second) [{sink}]", if grouped { "grouped" } else { "flat   " }, per * 1e6, 1.0 / per);
    }
}

/// Every kind a family registers: loads with its default params (or says what it needs) and
/// writes a key without panicking, on a plain state and on states with objects loaded.
#[test]
fn every_kind_builds_and_evaluates() {
    let mut kinds = std::collections::BTreeSet::new();
    for entry in std::fs::read_dir("../scattershot/src/objectives").unwrap() {
        let path = entry.unwrap().path();
        let src = std::fs::read_to_string(&path).unwrap();
        let Some(start) = src.find("pub fn build(") else { continue };
        let body = &src[start..];
        let body = &body[..body.find("\n}\n").unwrap_or(body.len())];
        let mut rest = body;
        while let Some(i) = rest.find('"') {
            rest = &rest[i + 1..];
            let Some(j) = rest.find('"') else { break };
            let name = &rest[..j];
            rest = &rest[j + 1..];
            let after = rest.trim_start();
            if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) && name.chars().all(|c| c.is_ascii_alphanumeric()) && (after.starts_with("=>") || after.starts_with('|')) {
                kinds.insert(name.to_string());
            }
        }
    }
    assert!(kinds.len() > 100, "found only {} kinds", kinds.len());
    let objects = |s: obrrt_core::St| {
        use crate::testutil::{obj, with_objects};
        let all: Vec<_> = ["bhvBowser", "bhvStar", "bhvDoor", "bhvCannon", "bhvKoopaShell", "bhvCoinFormation", "bhvBobomb", "bhvChainChompGate"]
            .iter()
            .enumerate()
            .filter_map(|(i, b)| obrrt_core::bhv_addr(b).map(|_| obj(b, i as u16 + 1, [100.0 * i as f32, 50.0, -200.0])))
            .collect();
        with_objects(s, all)
    };
    let mut needs_params = Vec::new();
    for kind in &kinds {
        let json = format!(r#"{{"kind": "{kind}", "params": {{}}}}"#);
        let tree = match Tree::load(&json) {
            Ok(t) => t,
            Err(e) => {
                needs_params.push(format!("{kind}: {e}"));
                continue;
            }
        };
        for (level, area) in [(16u16, 1u8), (6, 1), (9, 1), (24, 1), (36, 2), (1, 0)] {
            let plain = st([0.0, 100.0, 0.0], 20.0, level, area, 0);
            let _ = tree.key(&plain);
            let _ = tree.key(&objects(plain));
        }
    }
    println!("{} kinds; {} need params: {:#?}", kinds.len(), needs_params.len(), needs_params);
}

// ---------- the rewritten archive ----------

/// A rollout from `src` whose `found` are (key value, frame, checkpoint?).
fn roll_from(a: &Archive, src: usize, found: &[(i32, usize, bool)]) -> Rollout {
    let c = &a.cells[src];
    let n = found.iter().map(|f| f.1 + 1).max().unwrap_or(1);
    let found = found
        .iter()
        .map(|&(v, t, save)| Found { key: vec![(0, 0, v)], t, save: save.then(|| vec![7u8].into()), group: 0, busy: false, zone: 0, entered: false, pos: [0; 3] })
        .collect();
    Rollout { src, node: c.node as usize, frames: c.frames, p: c.p(), inputs: vec![sm64_binds::GamePad::new(0, 0, 0); n], found }
}

#[test]
fn a_rollout_from_a_source_replaced_in_the_same_batch() {
    // Cell 1 (key 1) at frame 50, a checkpoint. In one batch, rollout A (from
    // the root) finds key 1 faster; rollout B was planned from cell 1's old
    // node and finds key 2 without a checkpoint.
    let setup = |keep: bool| {
        let mut a = archive();
        a.drop_stale = !keep;
        a.merge(roll_from(&a, 0, &[(1, 49, true)]));
        a.free_pending();
        let old = a.cells[1].node as usize;
        assert!(a.has_save(old));
        let b = roll_from(&a, 1, &[(2, 3, false)]);
        let ra = roll_from(&a, 0, &[(1, 9, false)]);
        assert!(a.merge(ra).changed > 0);
        assert_ne!(a.cells[1].node as usize, old, "cell 1 moved to the faster node");
        (a, b, old)
    };
    // Default: B is dropped (it started from the slower state), with no credit,
    // and the old node's save, now unused, is freed with the batch.
    let (mut a, b, old) = setup(false);
    let visits = a.cells[1].visits;
    assert_eq!(a.merge(b).changed, 0);
    assert_eq!((a.stale, a.cells[1].visits), (1, visits));
    assert_eq!(a.find(&[(0, 0, 2)]), None);
    a.free_pending();
    assert!(!a.has_save(old));
    // Kept (OSS_KEEP_STALE): B's cell is rebuilt from the old node's save, which
    // therefore survives the end of the batch (the old engine dropped such a
    // rollout by accident when the save had been freed mid-batch).
    let (mut a, b, old) = setup(true);
    assert!(a.merge(b).changed > 0);
    let k2 = a.find(&[(0, 0, 2)]).expect("B's cell was kept");
    assert_eq!(a.nodes[a.cells[k2].node as usize].base as usize, old);
    a.free_pending();
    assert!(a.has_save(old));
}

#[test]
fn an_unused_replaced_save_is_freed_after_the_batch() {
    let mut a = archive();
    a.merge(roll_from(&a, 0, &[(1, 49, true)]));
    a.free_pending();
    let (old, bytes) = (a.cells[1].node as usize, a.save_bytes());
    a.merge(roll_from(&a, 0, &[(1, 9, true)]));
    assert!(a.has_save(old), "kept until the batch ends");
    a.free_pending();
    assert!(!a.has_save(old));
    assert_eq!(a.save_bytes(), bytes, "one save replaced by another of the same size");
    assert_eq!(a.checkpoints(), 2, "power-on and the new node");
}

#[test]
fn keys_sharing_a_hash_slot_are_told_apart() {
    let mut a = archive();
    a.merge(roll_from(&a, 0, &[(1, 0, false), (2, 1, false)]));
    let (k1, k2) = (a.find(&[(0, 0, 1)]).unwrap(), a.find(&[(0, 0, 2)]).unwrap());
    a.force_slot(&[(0, 0, 1)], vec![k2 as u32, k1 as u32]);
    assert_eq!(a.find(&[(0, 0, 1)]), Some(k1));
    assert_eq!(a.find(&[(0, 0, 2)]), Some(k2));
    assert_eq!(a.find(&[(0, 0, 3)]), None);
}

#[test]
fn best_frames_per_objective_match_a_full_scan() {
    use rand::Rng;
    let mut a = archive().with_groups(vec![0, 0, 0, 0], false);
    let mut rng = SmallRng::seed_from_u64(4);
    for _ in 0..300 {
        let src = rng.random_range(0..a.cells.len());
        let c = &a.cells[src];
        let n = rng.random_range(1..6);
        let found: Vec<Found> = (0..n)
            .map(|t| {
                let key = (0..rng.random_range(1..4)).map(|o| (o as u16, 0u8, rng.random_range(0..20))).collect();
                Found { key, t, save: None, group: 0, busy: false, zone: 0, entered: false, pos: [0; 3] }
            })
            .collect();
        let r = Rollout { src, node: c.node as usize, frames: c.frames, p: c.p(), inputs: vec![sm64_binds::GamePad::new(0, 0, 0); n], found };
        a.merge(r);
        a.free_pending();
    }
    for o in 0..3usize {
        let scan = a.cells.iter().filter(|c| c.key.iter().any(|e| e.0 as usize == o)).map(|c| c.frames).min();
        assert_eq!(a.best_in(o..o + 1), scan, "objective {o}");
    }
    assert_eq!(a.best_in(1..3), [1, 2].iter().filter_map(|&o| a.best_in(o..o + 1)).min());
}

/// The generated 120-star schema (examples/gen_star120.rs) loads, has a Stage
/// for every level area, unique names, and evaluates at each Stage.
#[test]
fn star120_schema_loads_and_evaluates() {
    let Ok(text) = std::fs::read_to_string("schemas/star120.json") else { return };
    let t = Tree::load(&text).unwrap();
    let mut names = std::collections::HashSet::new();
    for i in &t.insts {
        if let Some(n) = &i.name {
            assert!(names.insert(n.clone()), "duplicate name {n}");
        }
    }
    let stages: Vec<&crate::objective::Inst> = t.insts.iter().filter(|i| i.kind == "Stage").collect();
    assert!(stages.len() >= 40, "{} stages", stages.len());
    assert!(t.insts.iter().filter(|i| i.kind == "StarCollect").count() >= 15);
    for s in stages {
        let level = s.params["level"].as_u64().unwrap() as u16;
        let area = s.params.get("area").and_then(|a| a.as_u64()).unwrap_or(1) as u8;
        let state = st([0.0, 0.0, 0.0], 0.0, level, area, 0);
        assert_eq!(t.group(&state) as usize, s.id, "{:?} is the group of its own state", s.name);
        let _ = t.key(&state);
    }
}

/// `write` produces a cells.meta whose offsets point at each cells.jsonl line
/// and whose parents, areas and frames match the lines, and a summary.json
/// with the totals; both stamped with the size of the cells.jsonl.
#[test]
fn write_indexes_cells_jsonl() {
    let mut a = archive();
    a.merge(roll_from(&a, 0, &[(1, 3, false), (2, 7, true)]));
    a.merge(roll_from(&a, 1, &[(3, 2, false)]));
    let dir = std::env::temp_dir().join(format!("oss-write-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let d = dir.to_str().unwrap();
    a.write(d).unwrap();
    let jsonl = std::fs::read(dir.join("cells.jsonl")).unwrap();
    let meta = std::fs::read(dir.join("cells.meta")).unwrap();
    assert_eq!(&meta[..4], b"OSSM");
    let n = u64::from_le_bytes(meta[8..16].try_into().unwrap()) as usize;
    assert_eq!(n, a.cells.len());
    assert_eq!(u64::from_le_bytes(meta[16..24].try_into().unwrap()) as usize, jsonl.len());
    let arr = |base: usize, i: usize, w: usize| &meta[base + i * w..base + (i + 1) * w];
    let (o_off, o_par, o_zone, o_fr) = (24, 24 + 8 * n, 24 + 12 * n, 24 + 16 * n);
    assert_eq!(meta.len(), 24 + 26 * n, "v2: offsets, parents, zones, frames, positions");
    for i in 0..n {
        let off = u64::from_le_bytes(arr(o_off, i, 8).try_into().unwrap()) as usize;
        let end = jsonl[off..].iter().position(|&b| b == b'\n').unwrap() + off;
        let line: serde_json::Value = serde_json::from_slice(&jsonl[off..end]).unwrap();
        assert_eq!(line["id"].as_u64().unwrap() as usize, i);
        assert_eq!(line["parent"].as_i64().unwrap(), i32::from_le_bytes(arr(o_par, i, 4).try_into().unwrap()) as i64);
        assert_eq!(line["frames"].as_u64().unwrap(), u32::from_le_bytes(arr(o_fr, i, 4).try_into().unwrap()) as u64);
        assert_eq!(u32::from_le_bytes(arr(o_zone, i, 4).try_into().unwrap()), 0);
    }
    let s: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("summary.json")).unwrap()).unwrap();
    assert_eq!(s["cells_jsonl_bytes"].as_u64().unwrap() as usize, jsonl.len());
    assert_eq!(s["areas"][0]["n"].as_u64().unwrap() as usize, n);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn write_grids_cells_by_area_and_position() {
    let mut a = archive();
    a.merge(roll_from(&a, 0, &[(1, 0, false), (2, 1, false), (3, 2, false), (4, 3, false)]));
    let poses = [[0, 0, 0], [10, 20, 30], [-1, 0, 0], [300, -700, 5000], [255, 256, -256]];
    for (c, p) in a.cells.iter_mut().zip(poses) {
        c.pos = p;
    }
    let dir = std::env::temp_dir().join(format!("oss-grid-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    a.write(dir.to_str().unwrap()).unwrap();
    let g = std::fs::read(dir.join("cells.grid")).unwrap();
    let u32_at = |o: usize| u32::from_le_bytes(g[o..o + 4].try_into().unwrap());
    let i16_at = |o: usize| i16::from_le_bytes(g[o..o + 2].try_into().unwrap());
    assert_eq!(&g[..4], b"OSSG");
    let (size, nb, n) = (u32_at(8) as i32, u32_at(12) as usize, u64::from_le_bytes(g[16..24].try_into().unwrap()) as usize);
    assert_eq!((size, n), (256, a.cells.len()));
    let ids = 32 + 20 * nb;
    let mut seen = vec![0; n];
    let mut prev = None;
    for b in 0..nb {
        let o = 32 + 20 * b;
        let (z, bx, by, bz, start, count) = (u32_at(o), i16_at(o + 4), i16_at(o + 6), i16_at(o + 8), u32_at(o + 12) as usize, u32_at(o + 16) as usize);
        assert!(prev < Some((z, bx, by, bz)), "sorted");
        prev = Some((z, bx, by, bz));
        for k in start..start + count {
            let i = u32_at(ids + 4 * k) as usize;
            seen[i] += 1;
            let p = a.cells[i].pos;
            assert_eq!((bx, by, bz), ((p[0] as i32).div_euclid(size) as i16, (p[1] as i32).div_euclid(size) as i16, (p[2] as i32).div_euclid(size) as i16));
        }
    }
    assert!(seen.iter().all(|&s| s == 1), "every cell once: {seen:?}");
    assert_eq!(nb, 4, "(0,0,0) and (10,20,30) share a bucket; -1 is its own");
    std::fs::remove_dir_all(&dir).unwrap();
}

// ---------- oss2 ----------

#[test]
fn merge_in_parts_credits_once() {
    // Two parts from one source = one expansion: one visit, summed counts.
    let mut a = archive();
    a.merge(rollout(0, &a, &[1]));
    let (v0, s0) = (a.cells[0].visits, a.cells[0].succ);
    a.cells[1].frames = 100;
    let mut m = crate::archive::Merged::default();
    m.add(&a.merge_part(rollout(0, &a, &[2, 3])).unwrap());
    m.add(&a.merge_part(rollout(0, &a, &[1])).unwrap());
    a.credit(0, &m);
    assert_eq!((m.added, m.improved, m.changed), (2, 1, 3));
    assert_eq!((a.cells[0].visits, a.cells[0].succ), (v0 + 1.0, s0 + 1.0));
    assert_eq!(a.replaced, 1);
}

#[test]
fn coarsen_moves_grids_and_sets_params() {
    let schema = LOBBY;
    // Off: the schema as is.
    assert_eq!(crate::run::coarsen(schema).unwrap(), schema);
    unsafe {
        std::env::set_var("OSS_GRID_ADD", "2");
        std::env::set_var("OSS_PARAMS", r#"{"Lens": {"speed_step": 0, "yaw_sectors": 4}}"#);
    }
    let out = crate::run::coarsen(schema);
    unsafe {
        std::env::remove_var("OSS_GRID_ADD");
        std::env::remove_var("OSS_PARAMS");
    }
    let t = Tree::load(&out.unwrap()).unwrap();
    let grids: Vec<Option<u32>> = t.insts.iter().map(|i| i.obj.pos_grid()).collect();
    // Root 10, Stage Lobby 8, Lens (default 6), Side, Side, Stage BitDW (none).
    assert_eq!(grids, vec![Some(12), Some(10), Some(8), None, None, None]);
    assert_eq!(t.insts[2].params["speed_step"], 0);
    assert_eq!(t.insts[2].params["yaw_sectors"], 4);
    // Coarser bins: a state 100 units over is the same Lens cell now.
    let k = |x: f32| t.key(&st([-2230.0 + x, 0.0, 303.0], 33.0, 6, 1, 0));
    assert_eq!(k(0.0), k(100.0));
}

#[test]
fn kind_rate_is_a_posterior_mean() {
    let k = crate::ga::KindStats { tried: 8, succ: 2, ..Default::default() };
    assert!((k.rate() - 0.3).abs() < 1e-12);
    assert_eq!(crate::ga::KindStats::default().rate(), 0.5);
}
