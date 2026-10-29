//! Two stars per course visit (`OBRRT_TWOSTAR=1`, default off): the
//! 100-coin star plus an act star, as the 120-star TAS does in all 15 main
//! courses (ref/tas/star_strategies.md, §2 item 4).
//!
//! Why it works in the game: the 100-coin star doesn't end the visit. When
//! the coin count passes 100 in a main course, `interact_coin` (interaction.c)
//! calls `bhv_spawn_star_no_level_exit(6)` (sparkle_spawn_star.inc.c), which
//! sets INT_SUBTYPE_NO_EXIT on the star; `interact_star_or_key` then picks
//! ACT_STAR_DANCE_NO_EXIT and Mario plays on in the course. Any other star
//! ends it, and the coin count restarts at 0 on the next entry. So the
//! coins must come before the act star, in one visit.
//!
//! Why the search didn't: once any branch grabs the act star, every state at
//! the old star count is behind the frontier and retired, the coin-collecting
//! ones too (the most coins in any course in the hy/dn runs: 39). The
//! generic `OBRRT_KEEP_WIP` keeps states with work under way the frontier
//! lacks (`St::wip`); with this switch the coin count toward a missing
//! 100-coin star is such work (`wip_coins`, installed as the core's wip
//! extra). The 100-coin star's own grab then moves the frontier while Mario
//! is still in the course, so its act stars stay open from there.
//!
//! Also, with the switch:
//! - `Coins::Milestone` / `Coins::Reach100` weigh more (`milestone_weight`)
//!   and steer by `coin_target` instead of to the nearest coin.
//! - `TwoStar::CoinRun`: 10 more coins in one attempt, steering to the
//!   densest nearby coins (coin-dropping enemies count, by their loot),
//!   biased toward an act star still missing (the coins along the way).

use std::sync::OnceLock;

use obrrt_core::layout::INTERACT_COIN;
use obrrt_core::{Budget, Objective, Progress, St, course_of_level};

/// `OBRRT_TWOSTAR=1`.
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("OBRRT_TWOSTAR").is_ok_and(|v| v == "1"))
}

/// Installs `wip_coins` as the core's wip extra (with the switch on; once).
pub fn install() {
    static DONE: OnceLock<()> = OnceLock::new();
    DONE.get_or_init(|| {
        if on() {
            obrrt_core::set_wip_extra(wip_coins);
        }
    });
}

/// From this many coins a state's coins count as work under way.
const WIP_FROM: i16 = 20;

/// The main course `s` is in, when its 100-coin star is still missing.
fn hundred_course(s: &St) -> Option<i16> {
    let course = course_of_level(s.level as u32);
    ((1..=15).contains(&course) && !s.has_star(course, 6)).then_some(course)
}

/// Work toward the 100-coin star: the coin count, in a main course whose
/// 100-coin star is missing, from `WIP_FROM` coins up to 99.
pub fn wip_coins(s: &St) -> i64 {
    match hundred_course(s) {
        Some(_) if (WIP_FROM..100).contains(&s.coins) => s.coins as i64,
        _ => 0,
    }
}

/// `Coins` objective weight with the switch on (None: the default).
pub fn milestone_weight(target: i16) -> Option<f64> {
    on().then(|| if target >= 100 { 30.0 } else { 6.0 + target as f64 / 5.0 })
}

/// A coin's cluster: coins within this many units count toward its value.
const CLUSTER: f32 = 700.0;
/// Coins gained for a `CoinRun` success.
const RUN_COINS: i16 = 10;

/// Coin sources in `s`: (position, coins): loaded coins (`oDamageOrCoinValue`,
/// 5 for a blue coin) and objects that drop coins when beaten
/// (`oNumLootCoins`: goombas, piranha plants' blue coin, ...).
fn coin_sources(s: &St) -> Vec<([f32; 3], f32)> {
    s.objects()
        .iter()
        .filter_map(|o| {
            if o.interact_type == INTERACT_COIN {
                Some((o.pos, o.value.max(1) as f32))
            } else if o.loot_coins > 0 && o.health >= 0 {
                Some((o.pos, o.loot_coins as f32))
            } else {
                None
            }
        })
        .collect()
}

/// Where to steer for coins from `s`: the source with the most coins near
/// it over how far it is, a little toward `star` (the missing act star).
pub fn coin_target(s: &St, star: Option<[f32; 3]>) -> Option<[f32; 3]> {
    let coins = coin_sources(s);
    let d2 = |a: [f32; 3], b: [f32; 3]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    // Candidates: the nearest few (the cost is candidates x coins a frame).
    let mut cand: Vec<(f32, [f32; 3])> = coins.iter().map(|&(c, _)| (s.dist(c), c)).collect();
    cand.sort_by(|a, b| a.0.total_cmp(&b.0));
    cand.truncate(CANDIDATES);
    let far = (3.0 * CLUSTER).powi(2);
    cand.into_iter()
        .map(|(dc, c)| {
            let cluster: f32 = coins.iter().map(|&(o, v)| (d2(c, o), v)).filter(|&(q, _)| q < far).map(|(q, v)| v * (-q.sqrt() / CLUSTER).exp()).sum();
            let along = star.map_or(0.0, |p| 0.3 * (d2(c, p).sqrt() - s.dist(p)).max(0.0));
            (c, cluster / (1.0 + (dc + along) / 1000.0))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|x| x.0)
}

/// Coins considered as steering targets (the nearest ones).
const CANDIDATES: usize = 16;

/// `TwoStar::CoinRun`: `RUN_COINS` more coins (or 100) in one attempt, in a
/// main course whose 100-coin star is missing.
pub struct CoinRun {
    level: u16,
    course: i16,
    /// A missing act star's spot in this zone and act, if any (for the bias).
    star: Option<[f32; 3]>,
}

impl CoinRun {
    pub fn new(b: &St) -> Option<CoinRun> {
        if !on() || b.coins >= 100 {
            return None;
        }
        let course = hundred_course(b)?;
        let star = super::table::SPOTS
            .iter()
            .filter(|p| p.level == b.level as u32 && p.area == b.area && !b.has_star(p.course, p.index) && p.act_ok(b))
            .map(|p| p.pos)
            .min_by(|a, c| b.dist(*a).total_cmp(&b.dist(*c)));
        Some(CoinRun { level: b.level, course, star })
    }

    fn target(&self, s: &St) -> Option<[f32; 3]> {
        coin_target(s, self.star)
    }
}

impl Objective for CoinRun {
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "TwoStar::CoinRun"
    }
    fn key(&self) -> String {
        format!("TwoStar coins {}", self.course)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        8.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.level == self.level && s.coins < 100 && !s.has_star(self.course, 6)
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        // More coins first (a start 10 coins further is worth 5 units).
        let near = self.target(s).map_or(10.0, |c| s.dist(c) as f64 / 400.0);
        (100 - s.coins.clamp(0, 100)) as f64 / 2.0 + near
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let c0 = traj[0].coins;
        let goal = (c0 + RUN_COINS).min(100);
        traj.iter().position(|s| s.level == self.level && s.alive() && s.coins >= goal)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let c0 = traj[0].coins;
        let (at, score) = traj
            .iter()
            .enumerate()
            .filter(|(_, s)| s.level == self.level && s.alive())
            .map(|(i, s)| {
                let d = self.target(s).map_or(5000.0, |c| s.dist(c) as f64);
                (i, (s.coins - c0) as f64 * 1000.0 - d)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, f64::MIN));
        Progress { score, at }
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 360 }
    }
    fn success_value(&self) -> f64 {
        2.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.target(s)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.level == self.level && !s.has_star(self.course, 6)
    }
}
