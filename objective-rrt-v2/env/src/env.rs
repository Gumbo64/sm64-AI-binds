//! The legacy adapter: `OBRRT_*` variables to a [`SearchConfig`].
//!
//! Each variable is read exactly the way the code that used to read it did (`== "1"`,
//! `!= "0"`, strict or lenient number parsing), so a config built here behaves like the
//! environment it came from.
use anyhow::{Context, Result, bail};

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
    /// Set to exactly "1".
    fn is1(&self, k: &str) -> bool {
        self.get(k).is_some_and(|v| v == "1")
    }
    /// `true` when "1"; otherwise untouched.
    fn on(&self, k: &str, f: &mut bool) {
        if self.is1(k) {
            *f = true;
        }
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
    let SearchConfig { budget, sampler, chain, dedup, zone, goal_focus, kin, ga, priors, objectives, stop, research } = &mut c;

    // budget
    e.strict("OBRRT_GA_SCALE", &mut budget.ga_scale)?;
    e.strict_opt("OBRRT_MAX_FRAMES", &mut budget.max_frames)?;
    e.strict_opt("OBRRT_MEM_MB", &mut budget.mem_mb)?;
    e.strict("OBRRT_ROUND", &mut budget.round_plans)?;

    // sampler
    match e.get("OBRRT_SAMPLER").as_deref() {
        Some("rate") => sampler.rate_sampler = true,
        Some("decay") | None => {}
        Some(v) => bail!("OBRRT_SAMPLER: rate or decay, not {v}"),
    }
    let r = &mut sampler.rate;
    e.strict("OBRRT_RATE_CAP", &mut r.cap)?;
    e.strict("OBRRT_RATE_TAU", &mut r.tau)?;
    e.strict("OBRRT_RATE_PRIOR", &mut r.prior_frames)?;
    e.strict("OBRRT_RATE_NOSTART", &mut r.nostart_frames)?;
    e.strict("OBRRT_RATE_KIND_SCALE", &mut r.kind_scale)?;
    e.strict("OBRRT_RATE_ROUTE", &mut r.route)?;
    e.strict("OBRRT_RATE_NOVEL", &mut r.novel)?;
    match e.get("OBRRT_RATE_MODE").as_deref() {
        Some("hier") => r.hier = true,
        Some("flat") | None => {}
        Some(v) => bail!("OBRRT_RATE_MODE: hier or flat, not {v}"),
    }
    e.eq1("OBRRT_RATE_NOSTART_SKIP", &mut r.nostart_skip);
    e.strict("OBRRT_RATE_SKIP_BUDGET", &mut r.skip_budget)?;
    e.eq1("OBRRT_PICK_BY_B", &mut sampler.pick_by_b);
    e.eq1("OBRRT_NEW_ELIGIBLE", &mut sampler.new_eligible);
    e.on("OBRRT_STAR", &mut sampler.rrt_star);
    let a = &mut sampler.arrival;
    e.on("OBRRT_ARRIVAL", &mut a.on);
    e.strict("OBRRT_ARRIVAL_PACE", &mut a.pace)?;
    e.strict_opt("OBRRT_ARRIVAL_BAND", &mut a.band)?;

    // chain
    match e.get("OBRRT_CHAIN").as_deref() {
        Some("route") => chain.route = true,
        Some("stack") | None => {}
        Some(v) => bail!("OBRRT_CHAIN: route or stack, not {v}"),
    }

    // dedup
    e.on("OBRRT_SOFT_DEDUP", &mut dedup.soft_dedup);
    e.on("OBRRT_STARSET_KEY", &mut dedup.starset_key);
    // (a value that does not parse is ignored, as the tree's setting always did)
    e.lenient_opt("OBRRT_STAR_WINDOW", &mut dedup.star_window);
    e.on("OBRRT_KEEP_WIP", &mut dedup.keep_wip);
    e.on("OBRRT_CELL_MOTION", &mut dedup.cell_motion);
    match e.get("OBRRT_REPLAY_MAX").as_deref() {
        Some("off") => dedup.replay_max = ReplayMax::Off,
        Some(v) => dedup.replay_max = ReplayMax::Frames(v.parse().with_context(|| format!("OBRRT_REPLAY_MAX={v}"))?),
        None => {}
    }
    e.strict("OBRRT_SPARSE_CHECK", &mut dedup.sparse_check)?;
    let d = &mut dedup.dense;
    e.string("OBRRT_ADD", &mut d.add);
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
    let n = &mut dedup.near;
    e.on("OBRRT_NEAR_SEED", &mut n.seed);
    e.strict("OBRRT_NEAR_R", &mut n.radius)?;
    e.strict("OBRRT_NEAR_P", &mut n.p)?;

    // zone
    e.eq1("OBRRT_ZONE_LOCK", &mut zone.lock);
    if let Some(v) = e.get("OBRRT_ZONE_PATH") {
        zone.path = v;
    }
    e.on("OBRRT_ZONE_SHARE", &mut zone.share);
    zone.donor_sample = e.get("OBRRT_ZONE_SHARE_DONOR").is_some_and(|v| v == "sample");
    e.eq1("OBRRT_ZONE_SHARE_STATS", &mut zone.share_stats);
    e.on("OBRRT_REROUTE", &mut zone.reroute);
    e.string("OBRRT_REROUTE_SCHEMA", &mut zone.reroute_schema);
    e.strict("OBRRT_POLISH", &mut zone.polish)?;
    if let Some(v) = e.get("OBRRT_REPLAY_BUDGET") {
        let f: f64 = v.parse().with_context(|| format!("OBRRT_REPLAY_BUDGET={v}"))?;
        if f > 0.0 {
            zone.replay_budget = Some(f);
        }
    }

    // goal focus
    e.lenient("OBRRT_GOAL_FOCUS", &mut goal_focus.share);
    e.eq1("OBRRT_GOAL_EXPLORE", &mut goal_focus.explore);
    e.eq1("OBRRT_GOAL_WIDE", &mut goal_focus.wide);
    if let Some(f) = e.get("OBRRT_STALL_ETA").and_then(|v| v.parse::<f64>().ok()).filter(|&f| f > 0.0) {
        goal_focus.stall_eta = Some(f);
    }
    let p = &mut goal_focus.parent_window;
    if let Some(v) = e.get("OBRRT_PARENT_WINDOW").and_then(|v| v.parse::<f64>().ok()) {
        p.share = Some(v);
    }
    e.lenient("OBRRT_PARENT_WINDOW_MAX", &mut p.max);
    e.lenient("OBRRT_PARENT_WINDOW_FRAMES", &mut p.frames);
    e.lenient("OBRRT_PARENT_WINDOW_POP", &mut p.pop);
    e.lenient("OBRRT_PARENT_WINDOW_HORIZONS", &mut p.horizons);

    // kin
    e.strict("OBRRT_KINSHIP_MULT", &mut kin.mult)?;
    match e.get("OBRRT_KIN_UNIT").as_deref() {
        None | Some("") | Some("node") => {}
        Some("attempt") => kin.unit = KinUnit::Attempt,
        Some("frames") => kin.unit = KinUnit::Frames,
        Some(v) => bail!("OBRRT_KIN_UNIT={v}: expected node, attempt or frames"),
    }
    e.strict_opt("OBRRT_KIN_FRAMES", &mut kin.frames)?;
    if let Some(v) = e.get("OBRRT_KIN_SOFT") {
        let l: f64 = v.parse().with_context(|| format!("OBRRT_KIN_SOFT={v}"))?;
        anyhow::ensure!(l > 0.0 && l <= 1.0, "OBRRT_KIN_SOFT: 0 < λ <= 1");
        kin.soft = Some(l);
    }
    e.strict_opt("OBRRT_KIN_CAP", &mut kin.cap)?;

    // ga
    match e.get("OBRRT_FITNESS").as_deref() {
        Some("met") => ga.fitness = FitnessKind::Met,
        Some("steps") => ga.fitness = FitnessKind::Steps,
        _ => {}
    }
    if let Some(v) = e.get("OBRRT_GA_NEW") {
        ga.new_ops = v == "1";
    }
    e.strict("OBRRT_PINK", &mut ga.pink)?;
    e.on("OBRRT_GA_BUDGET", &mut ga.per_objective_budget);
    e.lenient("OBRRT_GA_GAMMA", &mut ga.gamma);
    e.lenient("OBRRT_GA_REGRET", &mut ga.regret);
    e.lenient("OBRRT_GA_CBUTTONS", &mut ga.cbuttons);
    e.eq1("OBRRT_GA_KEEP_SEED", &mut ga.keep_seed);
    e.eq1("OBRRT_TAIL", &mut ga.tail);
    e.eq1("OBRRT_CALM", &mut ga.calm);
    e.eq1("OBRRT_MOVE_GA", &mut ga.move_ga);
    e.eq1("OBRRT_MOVE_GA_EXTRA", &mut ga.move_ga_extra);

    // priors
    e.string("OBRRT_REACH_PRIOR", &mut priors.reach_prior);
    e.string("OBRRT_RESCUE_PRIOR", &mut priors.rescue_prior);
    e.string("OBRRT_KIND_WEIGHTS", &mut priors.kind_weights);
    e.string("OBRRT_ROAD", &mut priors.road);
    e.eq1("OBRRT_ETA_LEARN", &mut priors.eta_learn);
    e.lenient("OBRRT_ETA_LEARN_N0", &mut priors.eta_learn_n0);
    if let Some(s) = e.get("OBRRT_MODEL").filter(|s| !s.is_empty()) {
        priors.model = Some(s);
    }
    e.lenient("OBRRT_MODEL_START_T", &mut priors.model_start_t);
    e.lenient("OBRRT_MODEL_SERVERS", &mut priors.model_servers);
    e.string("OBRRT_MODEL_ZONES", &mut priors.model_zones);

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
    if let Some(v) = e.get("OBRRT_FAST_MODAL") {
        objectives.fast_modal = Some(v != "0");
    }
    e.eq1("OBRRT_MOVES", &mut objectives.moves);
    e.eq1("OBRRT_MOVES_EXTRA", &mut objectives.moves_extra);
    e.eq1("OBRRT_MOVES_MODES", &mut objectives.moves_modes);
    match e.get("OBRRT_MOVES_UPHILL").as_deref() {
        Some("jump") => objectives.moves_uphill = Uphill::Jump,
        Some("airkick") => objectives.moves_uphill = Uphill::AirKick,
        _ => {}
    }
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
    e.eq1("OBRRT_TALK_NIL", &mut objectives.talk_nil);
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

    // research
    e.eq1("OBRRT_FASTER_ONLY", &mut research.speed.faster_only);
    e.eq1("OBRRT_BEAT", &mut research.speed.beat);
    e.eq1("OBRRT_BNB", &mut research.speed.bnb);
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
    match e.get("OBRRT_PICK") {
        None => {}
        Some(v) => research.pick = Some(v),
    }
    e.string("OBRRT_PICK_SCHEMA", &mut research.pick_schema);
    e.strict("OBRRT_PICK_RAND", &mut research.pick_rand)?;
    e.strict("OBRRT_PICK_KIN_W", &mut research.pick_kin_w)?;
    e.eq1("OBRRT_SAMPLE_CHECK", &mut research.sample_check);
    Ok(c)
}
