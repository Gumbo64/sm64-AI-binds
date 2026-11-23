//! Speedrun-technique objectives: the registry and the switch that turns
//! them on and off (`OBRRT_TECH`, parsed in run/src/main.rs and handed to
//! [`configure`]).
//!
//! Each technique is one entry in [`TECHNIQUES`]: a short name, what it is,
//! its tags and the course it belongs to, if any. Its objective (and any
//! softened rule that exists only for it) checks [`enabled`] with its name.
//! To add one, add a file next to its related code (or here), an entry in
//! [`TECHNIQUES`], and an `enabled("name")` check where it's generated.
//!
//! Tags ([`Tag`]):
//! - `movement`: general speedrun movement, usable anywhere (dive
//!   rollouts, long-jump chains...);
//! - `location`: tied to a place but not to one star (a wall clip, a door
//!   skip);
//! - `star`: a strategy for one specific star (plus `location` if it fits).
//!
//! `OBRRT_TECH` is a comma list read left to right, starting from nothing
//! (from everything if the first entry is a removal, so `-star` = `all,-star`):
//! `all` (the default when unset) adds every technique, `none` clears the
//! set, a name or a tag adds those techniques, and `-name` / `-tag` removes
//! them. Examples: `none`, `gateclip,blj`, `movement`, `all,-star`,
//! `location,gateclip`. An unknown entry is an error.

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use obrrt_core::{Obj, Objective, St};

mod wallkick;
mod wfcless;
mod wftop;
mod ccmslideskip;
mod travel;
mod kicks;
mod flip;
mod islandhop;
mod owlless;
mod jrbcless;
mod jrbcaveclip;
mod kinghits;
mod whomphits;
mod lakituskip;
mod openers;
mod wfclimb;
pub mod blj;
pub mod lblj;
pub mod bljany;
pub mod bljfast;
mod poleredirect;
mod hyperclimb;
mod speedchain;
mod hyperwalk;
pub mod bombclip;
pub mod moatwarp;
pub mod pausebuf;
mod fastexit;
pub mod waterwarp;
pub mod freesetup;

/// Objectives from the technique files in this directory, per object near
/// `b`. Each technique adds one line here, guarded by `enabled("name")`.
#[allow(unused_variables)]
pub fn generate(b: &St, o: &Obj, out: &mut Vec<Arc<dyn Objective>>) {
    if enabled("kinghits") { kinghits::generate(b, o, out); }
    if enabled("whomphits") { whomphits::generate(b, o, out); }
    if enabled("lakituskip") { lakituskip::generate(b, o, out); }
    if enabled("fastexit") { fastexit::generate(b, o, out); }
}

/// Objectives from the technique files in this directory, once per `b`.
/// Each technique adds one line here, guarded by `enabled("name")`.
#[allow(unused_variables)]
pub fn generate_level(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    if enabled("wftop") { out.extend(wftop::TopKick::new(b).map(|x| Arc::new(x) as Arc<dyn Objective>)); }
    if enabled("ccmslideskip") { out.extend(ccmslideskip::SlideSkip::new(b).map(|x| Arc::new(x) as Arc<dyn Objective>)); }
    if enabled("wfcless") { out.extend(wfcless::Cless::new(b).map(|x| Arc::new(x) as Arc<dyn Objective>)); }
    if enabled("wallkick") { wallkick::Climb::generate(b, wallkick::Mode::WallKick, out); }
    if enabled("tjwk") { wallkick::Climb::generate(b, wallkick::Mode::Tjwk, out); }
    if enabled("wksideflip") { wallkick::Climb::generate(b, wallkick::Mode::SideFlip, out); }
    if enabled("wkredirect") { wallkick::Redirect::generate(b, out); }
    if enabled("wkchain") { wallkick::Chain::generate(b, out); }
    if enabled("diverollout") && travel::Dash::applies(b) { out.push(Arc::new(travel::Dash::new(b, travel::Move::DiveRollout))); }
    if enabled("ljchain") && travel::Dash::applies(b) { out.push(Arc::new(travel::Dash::new(b, travel::Move::LongJumpChain))); }
    if enabled("flipheight") && flip::FlipHeight::applies(b) { out.push(Arc::new(flip::FlipHeight::new(b))); }
    if enabled("islandhop") { islandhop::generate_level(b, out); }
    if enabled("owlless") { owlless::generate_level(b, out); }
    if enabled("jrbcless") { out.extend(jrbcless::PillarCannonless::new(b)); }
    if enabled("jrbcaveclip") { out.extend(jrbcaveclip::CaveClip::new(b)); }
    if enabled("jrbclesshyper") { out.extend(jrbcless::PillarHyper::new(b)); }
    if enabled("lblj") && lblj::Lblj::start(b) { out.push(Arc::new(lblj::Lblj)); }
    if enabled("blj") || enabled("bljspots") { blj::Blj::generate(b, out); }
    if enabled("poleredirect") { poleredirect::PoleRedirect::generate(b, out); }
    if enabled("hyperclimb") { hyperclimb::HyperClimb::generate(b, out); }
    if enabled("speedchain") { speedchain::SpeedChain::generate(b, out); }
    if enabled("bljany") { bljany::BljAny::generate(b, out); }
    if enabled("hyperwalk") { hyperwalk::generate(b, out); }
    if enabled("bombclip") { bombclip::BombClip::generate(b, out); }
    if enabled("moatwarp") { moatwarp::MoatWarpBlj::generate(b, out); }
    if enabled("lakituskip") { lakituskip::generate_level(b, out); }
    if enabled("wfopener") { openers::generate_level(b, openers::Which::Wf, out); }
    if enabled("bobopener") { openers::generate_level(b, openers::Which::Bob, out); }
    if enabled("wfclimb") && std::env::var("OBRRT_WFCLIMB").is_ok_and(|v| v == "1") { wfclimb::generate_level(b, out); }
    if enabled("slidekick") && kicks::KickRun::applies(b) { out.push(Arc::new(kicks::KickRun::new(b, kicks::Kick::Slide))); }
    if enabled("jumpkick") && kicks::KickRun::applies(b) { out.push(Arc::new(kicks::KickRun::new(b, kicks::Kick::Jump))); }
    if travel::aim_on() { travel::generate_aim(b, out); }
    waterwarp::generate(b, out); // OBRRT_WATERWARP=1 only
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tag {
    Movement,
    Location,
    Star,
}

impl Tag {
    pub fn name(self) -> &'static str {
        match self {
            Tag::Movement => "movement",
            Tag::Location => "location",
            Tag::Star => "star",
        }
    }
    const ALL: [Tag; 3] = [Tag::Movement, Tag::Location, Tag::Star];
}

pub struct Technique {
    pub name: &'static str,
    pub description: &'static str,
    pub tags: &'static [Tag],
    /// The course it belongs to (gLevelToCourseNumTable numbering), if any.
    pub course: Option<i16>,
}

pub const TECHNIQUES: &[Technique] = &[Technique {
    name: "wftop",
    description: "Wf::TopKick: WF star 2 from the fortress top, jump onto a tower platform, wall-kick off the tower at (569, 4715, -227) and ledge-grab the top ring (378, 5376, -334) as the TAS does; A hints",
    tags: &[Tag::Star],
    course: Some(2),
}, Technique {
    name: "ccmslideskip",
    description: "Ccm::SlideSkip: CCM star 1, wall-kick out of the slide's tube at the top (-4659, 6614, -6518) and fall ~12,500 straight to the exit room (y -5836); A hints",
    tags: &[Tag::Star],
    course: Some(4),
}, Technique {
    name: "wfcless",
    description: "Wf::Cless: WF star 6 without the cannon: fall off the walkway by the corner and ledge-grab into the wall beside the star (v19 found it)",
    tags: &[Tag::Star, Tag::Location],
    course: Some(2),
}, Technique {
    name: "gateclip",
    description: "Bobomb::GateClip: a bob-omb shoves Mario through the Chain Chomp's gate to BoB star 6 (an instance of bombclip's overlay and signature-gated progress, gate as the wall); \
                  also keeps star 6 open (soft star_blocked) for clip setups",
    tags: &[Tag::Star],
    course: Some(1),
}, Technique {
    name: "wallkick",
    description: "WallKick::Climb: reach a floor or ledge high above b (objects mark the high places), \
                  past a jump's reach, so it takes wall kicks; A hints in the kick window",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "tjwk",
    description: "TripleJumpWallKick::Climb: a triple jump then a wall kick to a floor or ledge 650-1150 above b",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "wksideflip",
    description: "WallKick::SideFlip: run at a wall, stick back (turning around), A with the stick at the wall (side flip), A on contact: a floor or ledge 350-1150 above b (objects, or +650 at b's wall); stick hints, A + stick forced on the turn and in the kick window",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "wkredirect",
    description: "WallKick::Redirect: from b at a wall, a kick as a 180° turn at 32+ keeping 0.9x the speed, towards a goal 1000 out along the wall normal; A forced in the kick window",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "wkchain",
    description: "WallKick::Chain: from b at a wall, +700 / +1500 over 3+ kicks with two wall normals 60°+ apart within 60 frames (corridor, chimney, corner); A forced in the kick window",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "diverollout",
    description: "Tech::DiveRollout: reach a point 900-2000 from b at 30+ u/frame straight-line, with a dive rollout; B/A hints",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "ljchain",
    description: "Tech::LongJumpChain: reach a point 900-2000 from b at 33+ u/frame straight-line, with 2+ long jumps; Z/A hints",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "flipheight",
    description: "Tech::FlipHeight: stand 250+ above the start within 60 frames, with a side flip or backflip; A/Z hints",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "islandhop",
    description: "IslandHop::Jump: BoB star 3 without cannon or wing cap, jump-dive off the TAS ledge (3341, 3007, -1564) or long jump off the summit corner (3072, 4097, -2559) onto the floating island; Z/A/B hints",
    tags: &[Tag::Star],
    course: Some(1),
}, Technique {
    name: "owlless",
    description: "Owlless::Cage: WF star 5 without Hoot, triple jump / double jump + wall kick from the high ground into the cage (y 4096); A hints",
    tags: &[Tag::Star, Tag::Location],
    course: Some(2),
}, Technique {
    name: "jrbcless",
    description: "Jrb::PillarCannonless: JRB star 5 without the cannon, out of the water onto the stone spire (or the ship's stern in acts 2-6), jump + wall kick off the cliff, ledge-grab the platform (y 1946); A hints",
    tags: &[Tag::Star],
    course: Some(3),
}, Technique {
    name: "jrbcaveclip",
    description: "Jrb::CaveClip: full-speed swim into the steep (floor-type) faces above the cave tunnel's mouth clips into the rock and downwarps into the tunnel/cave",
    tags: &[Tag::Location],
    course: Some(3),
}, Technique {
    name: "lblj",
    description: "Tech::Lblj: lobby BLJ past the 8-star door into the BitDW corridor with < 8 stars; stick-back + A/Z hints in long jumps; \
                  also opens the BitDW trap-floor warp from behind the door (soft door gating)",
    tags: &[Tag::Location],
    course: None,
}, Technique {
    name: "blj",
    description: "Tech::Blj / Tech::BljRun: BLJ at TAS spots (BoB elevator to the summit and lobby left steps into CCM: rule overlay on the GA; WF stairs to the Wild Blue ledge: stick-back + A/Z hints); graded BLJ metric as progress",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "bljany",
    description: "Tech::BljAny: BLJ anywhere blj_ok holds (rising object floor, low ceiling, steep non-slippery slope, or already <= -53; ref/blj-conditions.md), one per 400 cell; GA + BLJ overlay; graded BLJ metric; success = landing <= -100 800+ away or 2500 behind by dist/60",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "moatwarp",
    description: "Tech::MoatWarpBlj (also needs OBRRT_MOATWARP=1): the 2026 TAS opening, BLJ up the castle-grounds moat bank (3442, -166, 787) to a wall stall, x1.5 per 2 frames, then one walking frame into the VCutM warp (-3379, -815, -2025) r 600; BLJ overlay + free exit; success = VCutM after a BLJ",
    tags: &[Tag::Location, Tag::Star],
    course: Some(22),
}, Technique {
    name: "bljkinds",
    description: "bljany's extra BLJ kinds (CATALOGUE G02): rising object floor without oVelY (tilted object floor or a known platform), shallow-water slope (water-100 < y < water-40, no slide test), blocked quarter step (LONG_JUMP at the floor, vy -2/-4, fv <= -3), step riser (LONG_JUMP_LAND at -53 < fv <= -20 on a flat floor: growth seen, not just a wall behind); water gate y > water-100",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "hyperwalk",
    description: "Tech::HyperWalk: steer a hyperspeed walk (|fv| >= 100) to a star / warp / door / painting / holdable / far point (stick away from the goal on the ground, at it in the air; A held + B hops; Z/B stops near it); heuristic dist/|fv|; success within 300 at |fv| > 60, star, zone change, or 3000+ away and 1500+ closer; with it on, bljany succeeds on speed only (LJ chain at fv <= -150: the hand-off); HyperWalk success and distance credit only while |fv| > 60 holds",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "kinghits",
    description: "Boss::GrabRelease: King Bob-omb (BoB star 1) hit by a punch-grab from behind released at once (any drop puts him in his thrown act 4; he loses health on landing on the summit); B hints",
    tags: &[Tag::Star],
    course: Some(1),
}, Technique {
    name: "lakituskip",
    description: "Tech::LakituSkip: castle grounds before the intro, reach the castle door without dialog 34 (the trigger box catches any ground frame in it; every landing in its z range must be on the ~10-wide bridge-edge strip outside it); keys edge (onto and along the strip) and air (crouch-slide long jump onto the strip, Z+A long-jump chain, dives); progress = depth through the box over safe states only; Lakitu::Intro not offered while on",
    tags: &[Tag::Location],
    course: None,
}, Technique {
    name: "slidekick",
    description: "Tech::SlideKick: reach a point 900-2400 from b, 250-800 below at 40+ u/frame or level at 27+ u/frame straight-line, with a slide kick (crouch slide, B, rollout); Z/B/A hints",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "jumpkick",
    description: "Tech::JumpKick: reach a level point 700-1600 from b at 28+ u/frame straight-line, with a jump kick (B in the air, or punch-kick on a landing frame); B/A hints",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "wfopener",
    description: "Opener::Wf: from the WF painting landing, the high ground near (3300, 1600, 2000) within 6 s (TAS: jump kick, long jump up the ramp, wall kick at (3276, 740, 3890), dive rollouts; ~2.8 s); Z/A/B hints",
    tags: &[Tag::Location],
    course: Some(2),
}, Technique {
    name: "bobopener",
    description: "Opener::Bob: from the BoB painting landing, the Chain Chomp plateau near (1600, 768, 4150) within 6 s (TAS 5.1 s with slide-kick / dive-rollout chains); A/B hints",
    tags: &[Tag::Location],
    course: Some(1),
}, Technique {
    name: "wfclimb",
    description: "Wf::FortressClimb: from the WF painting landing, the fortress top (y 3584, within 1450 of the Whomp King: a whomphits start) within 8 / 11 / 15 s (distance to the arena disc; TAS 175 frames: long jumps, wall kicks, dive rollouts, steep jumps, side-flip wall kick); progress = distance gained on long-jump / wall-kick / flip / steep-jump / dive / rollout frames only, every loss counted, minus 1 a frame (no walk credit); success needs one in the segment; Z/A/B hints; experimental: also needs OBRRT_WFCLIMB=1 (0 successes in 4 test trees)",
    tags: &[Tag::Location],
    course: Some(2),
}, Technique {
    name: "poleredirect",
    description: "Tech::PoleRedirect: hyperspeed (|vel_xz| >= 150) into a tree/pole low on its trunk keeps the speed in vel; on the IDLE frame stick + Z + C-up crouch-slides at +|v| along the stick, BRAKING keeps it, A+B jump-kicks (TAS BoB tree +380); success fv >= 0.8 stored within 10 frames; also makes blj bob-elevator succeed at the tree (300+ toward it)",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "speedchain",
    description: "Tech::SpeedChain: BoB's spawn slope into about 90 speed, kept to the elevator ramp with the Lakitu hill's chain (TAS 2026 f22397-22548).",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "hyperclimb",
    description: "Tech::HyperClimb: from fv >= 150 on the ground (crouch slide, braking, landing), jump / double jump (vy 42/52 + fv/4) to a target 300+ higher within 6000 xz (stars, ! boxes, King, BoB summit / island) or +1000; A to launch, A held rising, A on wall hit, B dive near the target; success standing on it or 1000+ up within 90 frames",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "bljspots",
    description: "Tech::BljRun at the catalogue's extra TAS BLJ spots with their exit goals: lobby centre steps -> WF, WF-room steps -> BoB, lobby left steps -> PSS, castle 1F seam -> PSS, BoB steep slope -> red-coin posts, CCM east -> chimney warp, CCM plateau -> bottom warp pad; genome steers the exit",
    tags: &[Tag::Location],
    course: None,
}, Technique {
    name: "jrbcavewide",
    description: "Jrb::CaveClip wide mode (needs jrbcaveclip): any submerged action at hs > 26 counts as in the bay, and states faster than 60 aim at y -850..950 (the TAS's metal surface clip at y 753, f81720); swimming keeps the -850..-150 band (a swim can't pass the wall above y 0)",
    tags: &[Tag::Location],
    course: Some(3),
}, Technique {
    name: "jrbclesshyper",
    description: "Jrb::PillarHyper: JRB star 5 the TAS way (f82451-82613): dock-slope BLJ under the rock overhang (TAS run-up rule overlay, GA press sticks; pin not reproduced in the lab), wall reversal, dive + backward rollout over the water, braking run up the spire's floor faces, steep jump onto the platform; progress = BLJ metric, then nearness to the spire at hyperspeed, then height on the spire; B/A forced in the dive/rollout windows",
    tags: &[Tag::Star],
    course: Some(3),
}, Technique {
    name: "whomphits",
    description: "Boss::WhompFast: WF star 1, the Whomp King in the minimum 114-frame cycle (TAS f16212-16510): in his cone the first frame he can attack, in the air above his back as he lands, GP on his first flat frame; progress = hits (real hp drops), then minus the predicted frame of the next hit (absolute, so idling never scores); staged through DIALOG_115 and the star; WhompKing::Pound keeps only the approach (act 0)",
    tags: &[Tag::Star],
    course: Some(2),
}, Technique {
    name: "chestfacing",
    description: "Chest::Open facing mode: progress adds the chest's facing test (Mario's yaw + 0x8000 within 0x3000 of the chest's, treasure_chest.inc.c:70), and steering detours sideways past closed wrong chests the way would face",
    tags: &[Tag::Location],
    course: None,
}, Technique {
    name: "bombclip",
    description: "Tech::BombClip: at a wall with a floor seen 150-300 behind it, air-throw a bob-omb towards the wall, regrab it as it bloats (hitbox 65->117), carry it facing away (it pushes Mario back ~49/frame) to the wall and release it at the hand (push up to ~110) through the wall; keyed rule overlay (release/regrab windows); progress 0 until a bloated-bob-omb push frame, then depth through the plane; success past the plane on a floor (or a star) with the bob-omb/explosion near",
    tags: &[Tag::Movement],
    course: None,
}, Technique {
    name: "fastexit",
    description: "Tech::FastExit (OBRRT_FASTEXIT=1): grab a star with a death warp already queued (below a death plane's line, 20 frames, or out of bounds, 48) so the death exit replaces the star dance, kick-out and save menu (~160 frames); progress tiers doomed > over a death plane > rest",
    tags: &[Tag::Location],
    course: None,
}];

static ENABLED: OnceLock<HashSet<&'static str>> = OnceLock::new();

/// Parses an `OBRRT_TECH` value into the set of enabled technique names.
pub fn parse(spec: &str) -> Result<HashSet<&'static str>, String> {
    let words: Vec<&str> = spec.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    // A list that starts with a removal starts from everything.
    let mut set: HashSet<&'static str> = match words.first() {
        Some(w) if w.starts_with('-') => TECHNIQUES.iter().map(|t| t.name).collect(),
        _ => HashSet::new(),
    };
    for raw in words {
        let (remove, word) = match raw.strip_prefix('-') {
            Some(w) => (true, w),
            None => (false, raw),
        };
        let names: Vec<&'static str> = match word {
            "all" => TECHNIQUES.iter().map(|t| t.name).collect(),
            "none" => {
                set.clear();
                continue;
            }
            _ => {
                if let Some(tag) = Tag::ALL.iter().find(|t| t.name() == word) {
                    TECHNIQUES.iter().filter(|t| t.tags.contains(tag)).map(|t| t.name).collect()
                } else if let Some(t) = TECHNIQUES.iter().find(|t| t.name == word) {
                    vec![t.name]
                } else {
                    return Err(format!("OBRRT_TECH: unknown technique or tag {word:?}"));
                }
            }
        };
        for n in names {
            if remove {
                set.remove(n);
            } else {
                set.insert(n);
            }
        }
    }
    Ok(set)
}

/// Sets the enabled techniques once, before the search starts. Without a
/// call every technique is on (`all`).
pub fn configure(spec: &str) -> Result<(), String> {
    let set = parse(spec)?;
    ENABLED.set(set).map_err(|_| "techniques already configured".to_string())
}

/// The enabled techniques' names (for the run's log).
pub fn enabled_names() -> Vec<&'static str> {
    TECHNIQUES.iter().map(|t| t.name).filter(|n| enabled(n)).collect()
}

/// Whether technique `name` is on.
pub fn enabled(name: &str) -> bool {
    match ENABLED.get() {
        Some(set) => set.contains(name),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs() {
        assert!(parse("all").unwrap().contains("gateclip"));
        assert!(parse("none").unwrap().is_empty());
        assert!(!parse("all,-star").unwrap().contains("gateclip"));
        assert!(parse("star").unwrap().contains("gateclip"));
        assert!(!parse("movement").unwrap().contains("gateclip"));
        assert!(parse("location,gateclip").unwrap().contains("gateclip"));
        assert!(!parse("all,-gateclip").unwrap().contains("gateclip"));
        assert!(!parse("-gateclip").unwrap().contains("gateclip"));
        assert_eq!(parse("-gateclip").unwrap().len(), TECHNIQUES.len() - 1);
        assert!(parse("-movement").unwrap().contains("gateclip"));
        assert!(parse("gateclip,none").unwrap().is_empty());
        assert!(parse("bogus").is_err());
    }
}

/// Synthetic states for the technique and extra objectives' unit tests.
#[cfg(test)]
pub(crate) mod testkit {
    use std::sync::Arc;

    use obrrt_core::{Obj, ObjId, St, course_of_level};

    /// Walking (moving, on the ground).
    pub const ACT_WALKING: u32 = 0x0400_0440;

    /// Mario alive at `pos` in `(level, area)`, in `action`, frame 0, no
    /// stars, objects read (empty).
    pub fn st(level: u32, area: u8, pos: [f32; 3], action: u32) -> St {
        St {
            frame: 0,
            pos,
            vel: [0.0; 3],
            forward_vel: 0.0,
            yaw: 0,
            pitch: 0,
            angle_vel: [0; 3],
            water_level: -11000,
            cam_yaw: 0,
            action,
            mario_flags: 0,
            holding: false,
            held_slot: None,
            red_coins: 0,
            level: level as u16,
            area,
            act: 1,
            course: course_of_level(level),
            stars: 0,
            keys: 0,
            coins: 0,
            lives: 4,
            health: 0x880,
            play_mode: 0,
            transition: 0,
            dialog: -1,
            dialog_choice: false,
            menu_line: 0,
            menu_mode: -1,
            act_select: false,
            act_init: 0,
            act_cursor: 0,
            dialog_state: 0,
            dialog_last_page: false,
            menu_timer: 0,
            floor_y: pos[1],
            ceil_y: 20000.0,
            floor_type: 0,
            floor_normal: [0.0, 1.0, 0.0],
            floor_obj: None,
            floor_obj_vy: None,
            floor_obj_y: None,
            action_timer: 0, cap_timer: 0,
            wall_normal: None,
            course_stars: [0; 25],
            save_flags: 0,
            objects: Some(Arc::new(Vec::new())),
        }
    }

    /// An object `bhv` (by name; 1 if the table doesn't know it) at `pos`,
    /// home there, in `action` with `health`.
    pub fn obj(bhv: &str, slot: u16, pos: [f32; 3], action: i32, health: i32) -> Obj {
        let b = obrrt_core::bhv_addr(bhv).unwrap_or(1);
        Obj {
            id: ObjId { bhv: b, home: [pos[0] as i32, pos[1] as i32, pos[2] as i32] },
            slot,
            bhv: b,
            pos,
            home: pos,
            interact_type: 0,
            interact_status: 0,
            action,
            health,
            value: 0,
            loot_coins: 0,
            beh_params: 0,
            held_state: 0,
            intangible_timer: 0,
            yaw: 0,
            raw_1ac: [0; 4],
        }
    }

    /// `s` with `objs` as its objects.
    pub fn with(mut s: St, objs: Vec<Obj>) -> St {
        s.objects = Some(Arc::new(objs));
        s
    }
}
