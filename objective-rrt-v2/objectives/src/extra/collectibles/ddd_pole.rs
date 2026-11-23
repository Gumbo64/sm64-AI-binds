//! `DddPole::Hop`: Dire, Dire Docks star 3 ("Pole-Jumping for Red Coins", course 9 index 2).
//!
//! The mechanism (decomp): the eight red coins hang at y 1015-1288 in area 2 (levels/ddd/areas/2/macro.inc.c),
//! over platforms whose tops are 878-929 (areas/2/collision.inc.c; the ceiling is 1544). Nine `bhvDDDPole`s
//! (levels/ddd/script.c) hang from that ceiling, their lower end at y 1005, hitbox 80 radius, 800 high, 100 of it
//! below the origin (behavior_data.c bhvDDDPole). Each slides along its yaw: `oPosX = home.x + off * coss(yaw)`,
//! `oPosZ = home.z + off * sins(yaw)` (obj_behaviors_2.c obj_set_dist_from_home; x and z, not the usual
//! sin/cos), `off` going 0 -> 100 * (2nd behaviour byte) and back at `oDDDPoleVel` 10 a frame, and standing 21
//! frames at each end (ddd_pole.inc.c: it moves only while oTimer > 20, the clamp resets oTimer). A jump into a
//! pole (`interact_pole`, airborne actions) grabs it; Mario is then carried with it (set_pole_position uses the
//! pole's position), climbs at stickY / 8 up to `hitboxHeight - 100` = 700 above its origin (act_holding_pole),
//! and A lets go (act_holding_pole: a wall kick) towards the next pole or the coin.
//! The coins are at the poles' ends or along their tracks (coin 3525,-2250 at pole 9's start; 3300,-1900 at
//! pole 8's; 4680,-2200 where pole 9 passes at offset 1130; the z 3550-3595 row along pole 1; ...), so the
//! straight line to the nearest coin (`RedCoin::Next`) ignores that the way there is a ride on a moving pole.
//!
//! The floors: the water in area 2 is at y 0 (ground at 110 near the sign); the red coins' platforms are floating
//! boxes (areas/2/collision.inc.c: tops 878 in the south row, 929 in the middle and the west, 725 in the north)
//! that Mario only reaches from the poles or by a climb from the water (the main tree does that by itself, ~1700
//! frames from the entry). The coins form two groups: the south row z 3550-3595 along pole 1 (coins 5550, 4400,
//! 3850, 2974; the outer two over the platforms' tops, the middle two only from the pole), and the north four
//! (poles 8 and 9 and the platform 1674-2319). The way between them is a chain: pole 1 -> the south platform ->
//! pole 2 (x 5605, z 3380 -> 1280) -> the middle platform (z 413-1437) -> pole 7 (x 5760, z 360 -> -1940) -> the
//! north platform -> poles 8 and 9; the west poles 6, 4, 3 and 5 link the middle platform to the north-west one.
//! Air and health (mario.c update_mario_health): swimming below y = waterLevel - 140 = -140 costs 1 health a frame
//! (full 0x880 = 2176, so ~2000 frames; +0x1A a frame above that line), so the water under the platforms is a
//! detour that heals near the surface and drains below; the tests' health stayed 0x880 because the rides are dry.
//! The star (5513, 1200, 900) appears after the 8th coin (bhv_bowser_course_red_coin_star_loop) over the middle platform.
//!
//! The ETA plans the chain: Mario reaches pole p at time T[p] (its position over a frame-by-frame track up to
//! 2000 frames; `Pole::track`, equal to the closed form `Pole::at`), jumping into it from where he is; T[q] is
//! then relaxed over hops (a pole to another within 350 units: one dive from a pole's height) and over walks (let
//! go over a platform, walk to where the next pole will be), up to four chained moves. The coin plan rides a pole
//! to the coin and lets go. Straight lines to a coin only count on a platform or in the fall to its height (the
//! first version let a pole rider "dive" 630 units at ETA 22 and fell into the water). From below y 700 every
//! plan costs 1500 more, or a state that fell back to the water (the no-plan fallback 2500 + the line to the
//! nearest coin) beat the one riding the pole. Steering: the next pole where Mario will meet it, or on a pole its
//! own position at the coin's height. `RedCoin::Next` is not generated beside this from 4 coins on (collectibles.rs).
//!
//! Tests (agentws ddd2, side trees from route-0609 of t1, OBRRT_GEO=1, 2 threads; start = the entry, or a node of an
//! earlier test): the old objective: 80 M from the entry, 2 red coins (one pole ride), `DddPole::Hop` 38 attempts /
//! 7 successes, `RedCoin::Next` 219/0; from a 2-coin pole state, 40 M: 4 coins (the south row), Hop 28/17, Next 158/23.
//! Then from the 4-coin state at pole 1's end: with the old plan (horizon 600, free hops) 0 more coins in 40 M; with
//! this plan (horizon 2000, hops of at most 350, platform walks, low penalty, weight 12, GA 32 x 8) the rides pole 2
//! and 7 worked: from a 4-coin state on pole 2, 40 M: **7 coins** (Hop 30 attempts / 10 successes, 4 -> 5 -> 6 -> 7).
//! The 8th coin (the north-west platform, ETA 1623 from pole 9) and the star: 0 in 40 M (Hop 18 attempts).

use obrrt_core::layout::{ACT_FLAG_AIR, ACT_FLAG_ON_POLE, LEVEL_DDD};
use obrrt_core::{Eta, Mem, Obj, Objective, St, dist, frames_for};

const STEP: u32 = 8;
const HORIZON: u32 = 2000;
/// Mario's pace in the air after letting go of a pole (a wall kick's 24).
const HOP_PACE: f32 = 24.0;
/// Grabbing a pole after the jump reaches it, and letting go of one.
const GRAB_FRAMES: f32 = 8.0;
const LEAVE_FRAMES: f32 = 6.0;
/// Coin reach from a pole's axis (Mario 37 + coin 100).
const COIN_REACH: f32 = 110.0;
const CLIMB_PACE: f32 = 8.0;
/// Poles are grabbed from this near their axis horizontally.
const POLE_REACH: f32 = 100.0;
/// Getting up to a pole's height from below (platforms, wall kicks): units a frame and a fixed cost.
const LOW_PACE: f32 = 3.0;
const LOW_FRAMES: f32 = 100.0;
/// Added to every plan from below the platforms (y < 700, the water and the ground at 110): the climb up to the
/// 725-929 tops is the slow part, and in test d a state that fell back to the water (ETA 840 via the no-plan
/// fallback) beat the one riding pole 1 (ETA 975), so the tree kept going back down and swimming around.
const LOW_PENALTY: f64 = 1500.0;
const NO_PLAN: f64 = 2500.0;
const OLD_STEP: u32 = 6;
const OLD_HORIZON: u32 = 600;
const OLD_NO_PLAN: f64 = 800.0;
/// Below this many red coins the entry plan (`plan_old`) and its own pool entry apply.
const EARLY_COINS: i16 = 2;
/// From this many red coins on: the last stage.
const LAST_COINS: i16 = 6;
/// Horizontal reach of a walk/jump straight at a coin from the ground (one platform's width).
const STRAIGHT_WALK: f32 = 450.0;
/// The farthest a hop between two poles (or from a pole to a coin) can be without a platform to stand on:
/// the fall from a pole's height to the platforms' 878-929 tops is under 250 units (a dive: 30 a frame for
/// ~10 frames). The first plan let Mario "hop" 3000 units at pace 24 and the tree could not follow (ETA 214 at
/// the end of pole 1 with the coins 5000 away).
const HOP_MAX: f32 = 350.0;
/// In the air up on the platforms' level (y >= 700) a pole is "near" this far: the east wall's wall kicks reach
/// y 970 at x 6900, 1000 units from the middle platform, and test l's tree (reach 100 + the fall) rated those
/// states at the no-plan ETA, worse than the swimmers below them, and found no coin at all in 45 M.
const AIR_REACH: f32 = 1300.0;
/// Walking on a platform between where a pole lets him off and where another one is (run pace).
const WALK_PACE: f32 = 28.0;
/// How far past a platform's edge a pole still counts as over it (the pole's 80 radius plus Mario's 37 and a jump).
const PLAT_MARGIN: f32 = 130.0;
/// The platforms the poles hang over (x0, x1, z0, z1; levels/ddd/areas/2/collision.inc.c: the floating boxes
/// whose tops are 878 (south row) and 929 (the middle and west ones); the north one's top is 725).
const PLATS: [[f32; 4]; 6] = [
    [2688.0, 3302.0, 3433.0, 3741.0],
    [4941.0, 5837.0, 3075.0, 3792.0],
    [4941.0, 5862.0, 413.0, 1437.0],
    [1254.0, 1971.0, 361.0, 1488.0],
    [1674.0, 2319.0, -1883.0, -1279.0],
    [3098.0, 5555.0, -2454.0, -2044.0],
];
/// Mario is up on the platforms from this height (the lowest top is 725).
const HIGH_Y: f32 = 700.0;

/// The platform whose top Mario is over (within the pole margin), if any.
fn plat_of(p: [f32; 2]) -> Option<usize> {
    PLATS.iter().position(|r| p[0] > r[0] - PLAT_MARGIN && p[0] < r[1] + PLAT_MARGIN && p[1] > r[2] - PLAT_MARGIN && p[1] < r[3] + PLAT_MARGIN)
}

#[derive(Clone, Copy)]
struct Pole {
    home: [f32; 3],
    dir: [f32; 2],
    max: f32,
    off: f32,
    vel: f32,
    timer: u32,
    y: f32,
}

impl Pole {
    fn new(o: &Obj) -> Pole {
        let a = o.yaw as f32 * std::f32::consts::TAU / 65536.0;
        let dir = [a.cos(), a.sin()];
        // `value` (core/src/read.rs): sign = direction of oDDDPoleVel, magnitude = oTimer + 1 (<= 31).
        let (vel, timer) = if o.value < 0 { (-10.0, (-o.value - 1) as u32) } else { (10.0, (o.value - 1).max(0) as u32) };
        let off = (o.pos[0] - o.home[0]) * dir[0] + (o.pos[2] - o.home[2]) * dir[1];
        Pole { home: o.home, dir, max: 100.0 * ((o.beh_params >> 16) & 0xFF) as f32, off, vel, timer, y: o.pos[1] }
    }

    /// Horizontal position `t` frames on.
    fn at(&self, t: u32) -> [f32; 2] {
        let (mut off, mut vel, mut tm, mut rem) = (self.off, self.vel, self.timer, t);
        loop {
            if tm <= 20 {
                let w = (21 - tm).min(rem);
                tm += w;
                rem -= w;
            }
            if rem == 0 {
                break;
            }
            let to_end = if vel > 0.0 { self.max - off } else { off };
            let n = (to_end / vel.abs()).floor() as u32 + 1;
            if rem < n {
                off += vel * rem as f32;
                break;
            }
            off = if vel > 0.0 { self.max } else { 0.0 };
            vel = -vel;
            tm = 0;
            rem -= n;
        }
        [self.home[0] + off * self.dir[0], self.home[2] + off * self.dir[1]]
    }

    /// Horizontal position at every frame 0..=span, one frame at a time (the same state machine as `at`: it
    /// stands while oTimer <= 20 and moves after, the clamp at an end flips the speed and resets oTimer).
    fn track(&self, span: usize) -> Vec<[f32; 2]> {
        let (mut off, mut vel, mut tm) = (self.off, self.vel, self.timer);
        let mut v = Vec::with_capacity(span + 1);
        for _ in 0..=span {
            v.push([self.home[0] + off * self.dir[0], self.home[2] + off * self.dir[1]]);
            if tm <= 20 {
                tm += 1;
            } else {
                off += vel;
                if off < 0.0 || off > self.max {
                    off = off.clamp(0.0, self.max);
                    vel = -vel;
                    tm = 0;
                }
            }
        }
        v
    }

    /// Mario's height where he grabs it: inside its hitbox (905..1705).
    fn grab_y(&self, s_y: f32) -> f32 {
        s_y.clamp(self.y - 60.0, self.y + 600.0)
    }
}

fn hd(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// The stage by red coins held: 3 all 8 (the star: it spawns at (5513, 1200, 900) over the middle platform and the
/// plan rides the same chain to it, `DddPole::Star`), 0 entry (fewer than 2: `plan_old`), 1 the rides (`plan_new`), 2 the last coins (6+:
/// own pool entry and kind `DddPole::Last`, so the west route to the north-west platform gets its own attempts).
/// Each stage is its own kind (`Hop`, `Ride`, `Last`): the search learns a start-ETA cutoff and a yield multiplier
/// per kind, and the entry stage's successes (start ETA 5-300) would cut off the rides' starts (ETA 1600+).
fn stage_of(coins: i16) -> u8 {
    if coins < EARLY_COINS { 0 } else if coins < LAST_COINS { 1 } else if coins < 8 { 2 } else { 3 }
}

pub struct Hop {
    zone: (u16, u8),
    stage: u8,
}

impl Hop {
    pub fn new(b: &St) -> Option<Hop> {
        if b.level as u32 != LEVEL_DDD || b.area != 2 || b.has_star(9, 2) {
            return None;
        }
        let h = Hop { zone: b.zone(), stage: stage_of(b.red_coins) };
        (h.poles(b).len() > 0 && h.coins(b).next().is_some()).then_some(h)
    }

    /// The stage's plan: the first version's for the entry, the chained one after.
    fn plan(&self, s: &St) -> (f64, [f32; 3]) {
        if self.stage == 0 { self.plan_old(s) } else { self.plan_new(s) }
    }

    fn poles(&self, s: &St) -> Vec<Pole> {
        s.objects().iter().filter(|o| o.bhv_name() == "bhvDDDPole").map(Pole::new).collect()
    }

    fn coins<'a>(&self, s: &'a St) -> impl Iterator<Item = [f32; 3]> + 'a {
        // The targets: the red coins; once all 8 are held, the star they spawn (spawn_no_exit_star: bhv
        // bhvSpawnedStarNoLevelExit, hidden_star.inc.c; before it spawns, the level's `bhvHiddenRedCoinStar` at the same spot (5513, 1200, 900), which is all the 8-coin
        // nodes have for the first frames; the level's own `bhvStar`s are ghost stars of collected
        // stars, (3900, 850, -600) and (2030, -3700, -2780): the first stage-3 test used them, planned a flight to
        // the second at ETA 160 and made no progress).
        let star = self.stage == 3;
        s.objects().iter().filter(move |o| matches!(o.bhv_name(), "bhvSpawnedStarNoLevelExit" | "bhvSpawnedStar" | "bhvHiddenRedCoinStar") == star && (star || o.bhv_name() == "bhvRedCoin")).map(|o| o.pos)
    }

    /// The first version's plan (horizon 600, free hops, straight lines), kept for the entry stage (fewer than 2 red coins):
    /// ddd2's tests found the chained plan's low-state penalty and hop limits worse than this one up to the first coins.
    fn plan_old(&self, s: &St) -> (f64, [f32; 3]) {
        let poles = self.poles(s);
        let coins: Vec<[f32; 3]> = self.coins(s).collect();
        let sp = [s.pos[0], s.pos[2]];
        let n = poles.len();
        let steps = (OLD_HORIZON / OLD_STEP) as usize;
        let on = s.action_has(ACT_FLAG_ON_POLE);
        // T[p]: earliest time Mario is on pole p (from where he is, jumping into it).
        let mut t_on = vec![f32::INFINITY; n];
        let mut first = vec![[0.0f32; 3]; n];
        for (i, p) in poles.iter().enumerate() {
            if on && hd(sp, p.at(0)) < POLE_REACH {
                t_on[i] = 0.0;
                first[i] = [p.at(0)[0], s.pos[1], p.at(0)[1]];
                continue;
            }
            let lead = if on { LEAVE_FRAMES } else { 0.0 };
            for k in 0..=steps {
                let t = (k as u32 * OLD_STEP) as f32;
                let q = p.at(k as u32 * OLD_STEP);
                let g = [q[0], p.grab_y(s.pos[1]), q[1]];
                // Below the pole's hitbox Mario can't jump into it (he must be at 745+, 905 - his 160 height;
                // the first test's tree flew to the water under the coins, 1000 units "closer" at pace 30).
                let gap = (p.y - 260.0 - s.pos[1]).max(0.0);
                let climb = if gap > 0.0 { gap / LOW_PACE + LOW_FRAMES } else { 0.0 };
                let tr = frames_for(hd(sp, q)) as f32 + frames_for((g[1] - s.pos[1]).abs()) as f32 + climb + lead;
                if tr <= t {
                    t_on[i] = t + GRAB_FRAMES;
                    first[i] = g;
                    break;
                }
            }
        }
        // Through another pole: leave p at t' >= T[p], hop to q arriving at t'+h.
        let t1 = t_on.clone();
        for (qi, q) in poles.iter().enumerate() {
            for (pi, p) in poles.iter().enumerate() {
                if pi == qi || !t1[pi].is_finite() {
                    continue;
                }
                let mut tp = (t1[pi] / OLD_STEP as f32).ceil() as u32;
                while tp <= steps as u32 {
                    let t0 = (tp * OLD_STEP) as f32;
                    let from = p.at(tp * OLD_STEP);
                    // hop time to where q will be: fixed point over the arrival time
                    let mut h = hd(from, q.at(tp * OLD_STEP)) / HOP_PACE;
                    for _ in 0..3 {
                        h = hd(from, q.at((t0 + LEAVE_FRAMES + h) as u32)) / HOP_PACE;
                    }
                    let arrive = t0 + LEAVE_FRAMES + h + GRAB_FRAMES;
                    if arrive < t_on[qi] {
                        t_on[qi] = arrive;
                        first[qi] = first[pi];
                    }
                    tp += 2;
                }
            }
        }
        // Each coin: ride a pole to it and let go, or walk/jump straight.
        let mut best = (OLD_NO_PLAN + frames_for(coins.iter().map(|c| dist(s.pos, *c)).fold(f32::MAX, f32::min)), coins.first().copied().unwrap_or(s.pos));
        for c in &coins {
            let cp = [c[0], c[2]];
            if c[1] - s.pos[1] < 150.0 && hd(sp, cp) < 800.0 {
                let f = frames_for(dist(s.pos, *c));
                if f < best.0 {
                    best = (f, *c);
                }
            }
            for (i, p) in poles.iter().enumerate() {
                if !t_on[i].is_finite() {
                    continue;
                }
                let climb = ((c[1] - p.y).max(0.0) / CLIMB_PACE).min(OLD_HORIZON as f32);
                let mut k = (t_on[i] / OLD_STEP as f32).ceil() as u32;
                while k <= steps as u32 {
                    let t = (k * OLD_STEP) as f32;
                    let reach = (hd(p.at(k * OLD_STEP), cp) - COIN_REACH).max(0.0);
                    let total = t.max(climb + t_on[i]) + reach / HOP_PACE + if reach > 0.0 { LEAVE_FRAMES } else { 0.0 };
                    if (total as f64) < best.0 {
                        let target = if t_on[i] <= 0.0 { [p.at(0)[0], c[1], p.at(0)[1]] } else { first[i] };
                        best = (total as f64, target);
                    }
                    k += 1;
                }
            }
        }
        best
    }

    /// The best plan from `s`: frames, and where to steer first.
    fn plan_new(&self, s: &St) -> (f64, [f32; 3]) {
        let poles = self.poles(s);
        let coins: Vec<[f32; 3]> = self.coins(s).collect();
        let sp = [s.pos[0], s.pos[2]];
        let n = poles.len();
        let steps = (HORIZON / STEP) as usize;
        let on = s.action_has(ACT_FLAG_ON_POLE);
        let airborne = s.action_has(ACT_FLAG_AIR);
        // Every pole's position at every frame up to the horizon and a hop beyond it (one closed-form walk each).
        let span = HORIZON as usize + 120;
        let tbl: Vec<Vec<[f32; 2]>> = poles.iter().map(|p| p.track(span)).collect();
        // Which poles can lead to which: tracks within a hop's reach of each other or over the same platform
        // (9 samples along each track; the poles' tracks are fixed, only their timing moves).
        let samples: Vec<Vec<[f32; 2]>> = poles
            .iter()
            .map(|p| (0..=8).map(|k| [p.home[0] + p.max * p.dir[0] * k as f32 / 8.0, p.home[2] + p.max * p.dir[1] * k as f32 / 8.0]).collect())
            .collect();
        let adj = |a: usize, b: usize| {
            samples[a].iter().any(|x| samples[b].iter().any(|y| hd(*x, *y) <= HOP_MAX + 60.0 || (plat_of(*x).is_some() && plat_of(*x) == plat_of(*y))))
        };
        let at = |i: usize, t: f32| tbl[i][(t.max(0.0) as usize).min(span)];
        // T[p]: earliest time Mario is on pole p (from where he is, jumping into it).
        let mut t_on = vec![f32::INFINITY; n];
        let mut first = vec![[0.0f32; 3]; n];
        for (i, p) in poles.iter().enumerate() {
            if on && hd(sp, at(i, 0.0)) < POLE_REACH {
                t_on[i] = 0.0;
                first[i] = [at(i, 0.0)[0], s.pos[1], at(i, 0.0)[1]];
                continue;
            }
            let lead = if on { LEAVE_FRAMES } else { 0.0 };
            let high = s.pos[1] >= HIGH_Y;
            for k in 0..=steps {
                let t = (k as u32 * STEP) as f32;
                let q = at(i, t);
                let g = [q[0], p.grab_y(s.pos[1]), q[1]];
                let d = hd(sp, q);
                // From up on the platforms Mario can only walk to a pole over the same platform, or dive to
                // one within the fall's reach; from below (the water, the ground at 110) the old estimate.
                let tr = if high && !on {
                    let reach = if airborne { AIR_REACH + 30.0 * ((s.pos[1] - g[1]).max(0.0) / 2.0).sqrt() } else { 0.0 };
                    if d > reach && !(plat_of(sp).is_some() && plat_of(sp) == plat_of(q) && !airborne) {
                        f32::INFINITY
                    } else {
                        frames_for(d) as f32 + frames_for((g[1] - s.pos[1]).abs()) as f32
                    }
                } else if on && d > HOP_MAX {
                    f32::INFINITY
                } else {
                    // Below the pole's hitbox Mario can't jump into it (he must be at 745+, 905 - his 160 height;
                    // the first test's tree flew to the water under the coins, 1000 units "closer" at pace 30).
                    let gap = (p.y - 260.0 - s.pos[1]).max(0.0);
                    let climb = if gap > 0.0 { gap / LOW_PACE + LOW_FRAMES } else { 0.0 };
                    frames_for(d) as f32 + frames_for((g[1] - s.pos[1]).abs()) as f32 + climb + lead
                };
                if tr <= t {
                    t_on[i] = t + GRAB_FRAMES;
                    first[i] = g;
                    break;
                }
            }
        }
        // Through other poles (the chain pole 2 -> platform -> pole 7 -> the north poles needs two or three
        // hops): leave p at t' >= T[p], hop to q arriving at t'+h; relaxed until nothing improves.
        for _ in 0..4 {
            let mut changed = false;
            for qi in 0..n {
                for pi in 0..n {
                    if pi == qi || !t_on[pi].is_finite() || !adj(pi, qi) {
                        continue;
                    }
                    let mut tp = (t_on[pi] / STEP as f32).ceil() as u32;
                    while tp <= steps as u32 {
                        let t0 = (tp * STEP) as f32;
                        let from = at(pi, t0);
                        // (a) a hop through the air: fixed point over the arrival time
                        let mut h = hd(from, at(qi, t0)) / HOP_PACE;
                        for _ in 0..2 {
                            h = hd(from, at(qi, t0 + LEAVE_FRAMES + h)) / HOP_PACE;
                        }
                        let mut arrive = f32::INFINITY;
                        if hd(from, at(qi, t0 + LEAVE_FRAMES + h)) <= HOP_MAX {
                            arrive = t0 + LEAVE_FRAMES + h + GRAB_FRAMES;
                        }
                        // (b) over the same platform: let go, walk to where q is
                        if let Some(k) = plat_of(from) {
                            let mut w = hd(from, at(qi, t0)) / WALK_PACE;
                            for _ in 0..2 {
                                w = hd(from, at(qi, t0 + LEAVE_FRAMES + w)) / WALK_PACE;
                            }
                            if plat_of(at(qi, t0 + LEAVE_FRAMES + w)) == Some(k) {
                                arrive = arrive.min(t0 + LEAVE_FRAMES + w + GRAB_FRAMES);
                            }
                        }
                        if arrive + 1.0 < t_on[qi] {
                            t_on[qi] = arrive;
                            first[qi] = first[pi];
                            changed = true;
                        }
                        tp += 3;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        // Each coin: ride a pole to it and let go, or (on the ground or in a short fall) reach it directly.
        let mut best = (NO_PLAN + frames_for(coins.iter().map(|c| dist(s.pos, *c)).fold(f32::MAX, f32::min)), coins.first().copied().unwrap_or(s.pos));
        for c in &coins {
            let cp = [c[0], c[2]];
            // A straight line is only real where Mario can get there without a pole: a walk/jump on a platform
            // (or a short stretch), or a dive during the fall to the coin's height (2 units/frame^2 of the
            // 4 gravity, 30 a frame). The first test's tree took coin 2 on a pole and then dived from the pole
            // at the next coin 630 away (ETA 22), fell into the water, and never rode the pole on.
            let hreach = if on {
                0.0
            } else if airborne {
                COIN_REACH + 30.0 * ((s.pos[1] - c[1]).max(0.0) / 2.0).sqrt()
            } else if plat_of(sp).is_some() && plat_of(sp) == plat_of(cp) && s.pos[1] >= HIGH_Y {
                STRAIGHT_WALK
            } else if s.pos[1] < HIGH_Y {
                STRAIGHT_WALK
            } else {
                0.0
            };
            if c[1] - s.pos[1] < 150.0 && hd(sp, cp) < hreach {
                let f = frames_for(dist(s.pos, *c));
                if f < best.0 {
                    best = (f, *c);
                }
            }
            for (i, p) in poles.iter().enumerate() {
                if !t_on[i].is_finite() {
                    continue;
                }
                let climb = ((c[1] - p.y).max(0.0) / CLIMB_PACE).min(HORIZON as f32);
                let mut k = (t_on[i] / STEP as f32).ceil() as u32;
                while k <= steps as u32 {
                    let t = (k * STEP) as f32;
                    let reach = (hd(at(i, t), cp) - COIN_REACH).max(0.0);
                    // beyond the pole's own reach: a hop (short) or a walk over the same platform
                    if reach > HOP_MAX - COIN_REACH && !(plat_of(at(i, t)).is_some() && plat_of(at(i, t)) == plat_of(cp)) {
                        k += 1;
                        continue;
                    }
                    let total = t.max(climb + t_on[i]) + reach / HOP_PACE + if reach > 0.0 { LEAVE_FRAMES } else { 0.0 };
                    if (total as f64) < best.0 {
                        let target = if t_on[i] <= 0.0 { [at(i, 0.0)[0], c[1], at(i, 0.0)[1]] } else { first[i] };
                        best = (total as f64, target);
                    }
                    k += 1;
                }
            }
        }
        if s.pos[1] < HIGH_Y {
            best.0 += LOW_PENALTY;
        }
        best
    }
}

impl Objective for Hop {
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        match self.stage {
            0 => "DddPole::Hop",
            1 => "DddPole::Ride",
            2 => "DddPole::Last",
            _ => "DddPole::Star",
        }
    }
    fn key(&self) -> String {
        format!("DddPole {:?} stage {}", self.zone, self.stage)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        // The pool samples a kind by its entries' mean weight x decay x the kind's multiplier: with a pool of ~40
        // kinds `DddPole::Last` got 2% of the picks (17 attempts in 20 M frames) at weight 12 although one attempt
        // from the pole 9 state took the ETA from 1623 to 94 and the next start (ETA 94) was tried once.
        [20.0, 40.0, 60.0, 60.0][self.stage as usize]
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && !s.has_star(9, 2) && self.coins(s).next().is_some() && stage_of(s.red_coins) == self.stage
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        let (population, generations) = if self.stage == 0 { (20, 5) } else { (32, 8) };
        Some(obrrt_core::GaBudget { population, generations, horizon: 600 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        if s.alive() && (s.red_coins > start.red_coins || (self.stage == 3 && s.stars > start.stars)) {
            return Eta::Done;
        }
        Eta::Frames(self.plan(s).0)
    }
    fn start_bias(&self, s: &St) -> f64 {
        -(s.red_coins.clamp(0, 8) as f64) * 2.0 * obrrt_core::V1_UNIT_FRAMES
    }
    fn success_value(&self) -> f64 {
        3.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        (s.zone() == self.zone).then(|| self.plan(s).1)
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frame-by-frame track agrees with the closed form (poles 1 and 9 of levels/ddd/script.c, mid-motion
    /// and in the end pause).
    #[test]
    fn track_matches_closed_form() {
        for (off, vel, timer, max) in [(0.0, 10.0, 0u32, 3000.0), (1130.0, -10.0, 30, 1300.0), (1295.0, 10.0, 25, 1300.0), (0.0, -10.0, 5, 700.0)] {
            let p = Pole { home: [3550.0, 1005.0, -2250.0], dir: [1.0, 0.0], max, off, vel, timer, y: 1005.0 };
            let tr = p.track(900);
            for t in 0..=900u32 {
                let a = p.at(t);
                assert!((a[0] - tr[t as usize][0]).abs() < 0.01 && (a[1] - tr[t as usize][1]).abs() < 0.01, "t {t}: {a:?} vs {:?}", tr[t as usize]);
            }
        }
    }
}
