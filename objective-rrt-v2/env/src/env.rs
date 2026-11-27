//! The legacy adapter: `OBRRT_*` variables to a [`SearchConfig`].
//!
//! Each variable is read exactly the way the code that used to read it did (`== "1"`,
//! `!= "0"`, strict or lenient number parsing), so a config built here behaves like the
//! environment it came from.
use anyhow::{Context, Result};

use crate::config::*;

/// Where variables come from: the process environment, or any other lookup (the browser
/// build's per-instance map).
pub type Lookup<'a> = &'a dyn Fn(&str) -> Option<String>;

/// `base` with the process environment's `OBRRT_*` switches applied.
pub fn from_env(base: SearchConfig) -> Result<SearchConfig> {
    from_env_with(base, &|k| std::env::var(k).ok())
}

struct E<'a>(Lookup<'a>);

impl E<'_> {
    fn get(&self, k: &str) -> Option<String> {
        (self.0)(k)
    }
    /// When set, `value == "1"`.
    fn eq1(&self, k: &str, f: &mut bool) {
        if let Some(v) = self.get(k) {
            *f = v == "1";
        }
    }
    /// A number that must parse when set.
    fn strict<T: std::str::FromStr>(&self, k: &str, f: &mut T) -> Result<()>
    where
        T::Err: std::error::Error + Send + Sync + 'static,
    {
        if let Some(v) = self.get(k) {
            *f = v.parse().with_context(|| format!("{k}={v}"))?;
        }
        Ok(())
    }
    fn strict_opt<T: std::str::FromStr>(&self, k: &str, f: &mut Option<T>) -> Result<()>
    where
        T::Err: std::error::Error + Send + Sync + 'static,
    {
        if let Some(v) = self.get(k) {
            *f = Some(v.parse().with_context(|| format!("{k}={v}"))?);
        }
        Ok(())
    }
    /// A number that is ignored when it does not parse.
    fn lenient<T: std::str::FromStr>(&self, k: &str, f: &mut T) {
        if let Some(v) = self.get(k).and_then(|v| v.parse().ok()) {
            *f = v;
        }
    }
    fn lenient_opt<T: std::str::FromStr>(&self, k: &str, f: &mut Option<T>) {
        if let Some(v) = self.get(k).and_then(|v| v.parse().ok()) {
            *f = Some(v);
        }
    }
    fn string(&self, k: &str, f: &mut Option<String>) {
        if let Some(v) = self.get(k) {
            *f = Some(v);
        }
    }
}

/// `base` with the switches `get` knows applied.
pub fn from_env_with(base: SearchConfig, get: Lookup) -> Result<SearchConfig> {
    let e = E(get);
    let mut c = base;
    let SearchConfig { budget, sampler, dedup, zone, goal_focus, kin, ga, priors, objectives, stop, research } = &mut c;

    e.strict_opt("OBRRT_MAX_FRAMES", &mut budget.max_frames)?;
    e.strict_opt("OBRRT_MEM_MB", &mut budget.mem_mb)?;
    e.strict("OBRRT_ROUND", &mut budget.round_plans)?;

    e.eq1("OBRRT_PICK_BY_B", &mut sampler.pick_by_b);
    e.eq1("OBRRT_NEW_ELIGIBLE", &mut sampler.new_eligible);


    let d = &mut dedup.dense;
    if let Some(a) = e.get("OBRRT_ADD") {
        d.add = (!a.is_empty() && a != "none").then_some(a);
    }
    e.strict("OBRRT_DENSE_GRID", &mut d.grid)?;
    e.strict_opt("OBRRT_DENSE_CAP", &mut d.cap)?;
    if let Some(v) = e.get("OBRRT_DENSE_NE") {
        d.ne = v != "0";
    }
    match e.get("OBRRT_DENSE_EVERY").as_deref() {
        None => {}
        Some("obj") => {
            d.every = 15;
            d.per_obj = true;
        }
        Some(v) => {
            d.per_obj = v.starts_with("obj:");
            d.every = v.strip_prefix("obj:").unwrap_or(v).parse().with_context(|| format!("OBRRT_DENSE_EVERY={v}"))?;
        }
    }
    if let Some(v) = e.get("OBRRT_DENSE_AIR") {
        d.air = v != "0";
    }

    // zone
    e.eq1("OBRRT_ZONE_LOCK", &mut zone.lock);
    if let Some(v) = e.get("OBRRT_ZONE_PATH") {
        zone.path = v;
    }

    // goal focus
    e.lenient("OBRRT_GOAL_FOCUS", &mut goal_focus.share);
    e.eq1("OBRRT_GOAL_EXPLORE", &mut goal_focus.explore);
    e.eq1("OBRRT_GOAL_WIDE", &mut goal_focus.wide);
    if let Some(f) = e.get("OBRRT_STALL_ETA").and_then(|v| v.parse::<f64>().ok()).filter(|&f| f > 0.0) {
        goal_focus.stall_eta = Some(f);
    }

    // kin
    e.strict("OBRRT_KINSHIP_MULT", &mut kin.mult)?;

    e.strict("OBRRT_PINK", &mut ga.pink)?;
    e.lenient("OBRRT_GA_GAMMA", &mut ga.gamma);
    e.lenient("OBRRT_GA_REGRET", &mut ga.regret);
    e.eq1("OBRRT_GA_KEEP_SEED", &mut ga.keep_seed);
    e.eq1("OBRRT_CALM", &mut ga.calm);
    e.eq1("OBRRT_MOVE_GA", &mut ga.move_ga);
    e.eq1("OBRRT_MOVE_GA_EXTRA", &mut ga.move_ga_extra);

    // priors
    e.string("OBRRT_REACH_PRIOR", &mut priors.reach_prior);
    e.string("OBRRT_RESCUE_PRIOR", &mut priors.rescue_prior);
    e.string("OBRRT_ROAD", &mut priors.road);

    // stop
    e.eq1("OBRRT_ANY_EXIT", &mut stop.any_exit);
    e.eq1("OBRRT_FINAL", &mut stop.final_search);
    e.eq1("OBRRT_EXIT_NOSTOP", &mut stop.exit_nostop);
    if let Some(v) = e.get("OBRRT_EXIT_SKIP") {
        stop.exit_skip = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    }
    if let Some(f) = e.get("OBRRT_STOP_FLAG").and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok()) {
        stop.flag = f;
    }
    e.eq1("OBRRT_STOP_PROGRESS", &mut stop.progress);
    e.lenient_opt("OBRRT_STOP_STALL", &mut stop.stall);
    e.lenient_opt("OBRRT_STOP_EXIT_GRACE", &mut stop.exit_grace);
    e.lenient_opt("OBRRT_GOOD_MAX_FRAME", &mut stop.good_max_frame);

    // objectives
    if let Some(v) = e.get("OBRRT_TECH") {
        objectives.tech = v;
    }
    e.eq1("OBRRT_MOVES", &mut objectives.moves);
    e.eq1("OBRRT_MOVES_EXTRA", &mut objectives.moves_extra);
    e.eq1("OBRRT_MOVES_MODES", &mut objectives.moves_modes);
    e.eq1("OBRRT_SWIM_HOLD", &mut objectives.swim_hold);
    e.eq1("OBRRT_SWIM_TARGETS", &mut objectives.swim_targets);
    e.eq1("OBRRT_GP_CATCH", &mut objectives.gp_catch);
    e.eq1("OBRRT_LAKITU_ARC", &mut objectives.lakitu_arc);
    e.eq1("OBRRT_LAKITU_THROUGH_DOOR", &mut objectives.lakitu_through_door);
    e.eq1("OBRRT_PAINTING_FAST", &mut objectives.painting_fast);
    e.eq1("OBRRT_GROUNDS_FIX", &mut objectives.grounds_fix);
    if let Some(v) = e.get("OBRRT_MOVE_WEIGHT") {
        objectives.move_weight = Some(v.parse().with_context(|| "OBRRT_MOVE_WEIGHT: a number")?);
    }
    e.eq1("OBRRT_KING_LANDING", &mut objectives.king_landing);
    e.string("OBRRT_BLJ_DETECTOR", &mut objectives.blj_detector);
    e.eq1("OBRRT_BLJ_FAST", &mut objectives.blj_fast);
    e.eq1("OBRRT_BLJ_RISERS", &mut objectives.blj_risers);
    e.eq1("OBRRT_BLJ_SCAN", &mut objectives.blj_scan);
    e.eq1("OBRRT_BLJ_SPOTS", &mut objectives.blj_spots);
    e.string("OBRRT_BLJ_SPOTS_CATALOG", &mut objectives.blj_spots_catalog);
    e.eq1("OBRRT_BLJ_WALLS", &mut objectives.blj_walls);
    e.eq1("OBRRT_FASTEXIT", &mut objectives.fastexit);
    e.eq1("OBRRT_MOATWARP", &mut objectives.moatwarp);
    e.eq1("OBRRT_PAUSE_BUF", &mut objectives.pause_buf);
    e.eq1("OBRRT_WATERWARP", &mut objectives.waterwarp);
    e.eq1("OBRRT_TWOSTAR", &mut objectives.twostar);
    e.eq1("OBRRT_STATIC_ID", &mut objectives.static_id);
    e.eq1("OBRRT_GEO", &mut objectives.geo);

    let rh = &mut research.rh;
    e.eq1("OBRRT_RH", &mut rh.on);
    e.lenient("OBRRT_RH_POP", &mut rh.pop);
    e.lenient("OBRRT_RH_H", &mut rh.h);
    e.lenient("OBRRT_RH_K", &mut rh.k);
    e.lenient("OBRRT_RH_G", &mut rh.g);
    e.lenient("OBRRT_RH_G_PRECISE", &mut rh.g_precise);
    e.lenient("OBRRT_RH_GAMMA", &mut rh.gamma);
    e.lenient("OBRRT_RH_BONUS", &mut rh.bonus);
    e.lenient("OBRRT_RH_DEATH", &mut rh.death);
    e.lenient("OBRRT_RH_STALL", &mut rh.stall);
    e.lenient("OBRRT_RH_HOT", &mut rh.hot);
    e.lenient("OBRRT_RH_MAXLEN", &mut rh.maxlen);
    e.eq1("OBRRT_GA_QD", &mut research.qd);
    e.string("OBRRT_GA_GAMMAS", &mut research.qd_gammas);
    Ok(c)
}
