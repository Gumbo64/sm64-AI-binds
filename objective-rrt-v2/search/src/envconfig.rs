//! The tree's settings as one typed [`SearchConfig`] (the `obrrt-env` crate): [`from_env`] reads
//! the `OBRRT_*` variables, as `obrrt2` does (run/src/main.rs), for every program that runs the
//! engine (obrrt2, sm64-crypto3's miner); [`apply_globals`] puts a config into the engine's
//! [`Config`] and the search's remaining process-wide switches; [`after_root`] does the part
//! that needs the root state.
use anyhow::Result;
use obrrt_core::St;
use obrrt_core::config::{FitnessKind, KinUnit, ReplayMax, SearchConfig};
use std::sync::atomic::Ordering::Relaxed;

use crate::Config;

/// Sets `cfg` and the search's global switches from `OBRRT_*`, and installs the environment's
/// config as the process's ([`obrrt_core::config::install`]).
pub fn from_env(cfg: &mut Config) -> Result<()> {
    let sc = obrrt_core::config::from_env(SearchConfig::default())?;
    apply_globals(obrrt_core::config::install(sc), cfg)
}

/// Puts `sc` into `cfg` and the search's global switches (tree and engine statics, dense adds,
/// near seeds, backward chaining). Call before the engine is built.
pub fn apply_globals(sc: &SearchConfig, cfg: &mut Config) -> Result<()> {
    let (b, s, ch, d, z, g, k, ga, p) = (&sc.budget, &sc.sampler, &sc.chain, &sc.dedup, &sc.zone, &sc.goal_focus, &sc.kin, &sc.ga, &sc.priors);
    let _ = (g, p);

    // budget
    cfg.ga_scale = b.ga_scale;
    cfg.max_frames = b.max_frames;
    if let Some(mb) = b.mem_mb {
        cfg.mem_budget = mb * 1_000_000;
    }
    cfg.round_plans = b.round_plans;
    cfg.tune_frames = b.tune_frames;

    // GA
    cfg.fitness = match ga.fitness {
        FitnessKind::Met => crate::ga::Fitness::Met,
        FitnessKind::Steps => crate::ga::Fitness::Steps,
    };
    cfg.ga_new_ops = ga.new_ops;
    cfg.ga_pink = ga.pink;
    cfg.ga_per_objective_budget = ga.per_objective_budget;

    // sampler
    cfg.sampler_rate = s.rate_sampler;
    cfg.rate = crate::pool::RateParams {
        floor: s.rate.floor,
        cap: s.rate.cap,
        tau: s.rate.tau,
        prior_frames: s.rate.prior_frames,
        nostart_frames: s.rate.nostart_frames,
        hier: s.rate.hier,
        kind_scale: s.rate.kind_scale,
        nostart_skip: s.rate.nostart_skip,
        skip_budget: s.rate.skip_budget,
        route: s.rate.route,
        novel: s.rate.novel,
    };
    cfg.node_prior = s.rate.node_prior;
    cfg.node_cap = s.rate.node_cap;
    cfg.kind_cap = s.kind_cap;
    cfg.kind_bandit = s.kind_bandit;
    cfg.kind_gamma = s.kind_gamma;
    cfg.entry_bandit = s.entry_bandit;
    cfg.zone_bandit = s.zone_bandit;
    cfg.value_zones = s.value_zones;
    cfg.start_wear = s.start_wear;
    cfg.decay_route = s.decay_route;
    cfg.route_share = s.route_share;
    cfg.pick_by_b = s.pick_by_b;
    cfg.new_eligible = s.new_eligible;
    cfg.rrt_star = s.rrt_star;
    cfg.arrival = s.arrival.on;
    cfg.arrival_pace = s.arrival.pace;
    cfg.arrival_t = s.arrival.t;
    cfg.arrival_band = s.arrival.band;
    cfg.arrival_decay_either = s.arrival.decay_either;
    cfg.arrival_mix = s.arrival.mix;
    cfg.earliest = s.earliest.on;
    cfg.earliest_cap = s.earliest.cap;
    cfg.earliest_share = s.earliest.share;
    cfg.earliest_cell = s.earliest.cell;

    // chain
    cfg.chain_route = ch.route;
    cfg.chain_follow = ch.follow;
    cfg.chain_fixed = ch.fixed;

    // dedup, kinship
    cfg.soft_dedup = d.soft_dedup;
    cfg.rebuild_check = d.sparse_check;
    cfg.kinship_mult = k.mult;

    // zones, grafts, replays
    cfg.reroute = z.reroute;
    cfg.reroute_schema = z.reroute_schema.clone();
    cfg.reroute_share = z.reroute_share;
    cfg.reroute_margin = z.reroute_margin;
    cfg.polish = z.polish;
    cfg.zone_share = z.share;
    cfg.zone_share_k = z.share_k;
    cfg.zone_share_cap = z.share_cap;
    cfg.zone_share_share = z.share_share;
    cfg.zone_share_bg = z.share_bg;
    cfg.replay_budget = z.replay_budget;

    // The tree's and the engine's process-wide switches.
    crate::engine::SETTLE_RANK.store(d.settle_rank, Relaxed);
    crate::engine::FINGERPRINT.store(d.fingerprint, Relaxed);
    crate::engine::STARSET.store(d.starset_key, Relaxed);
    if let Some(w) = d.star_window {
        crate::tree::STAR_WINDOW.store(w.max(1), Relaxed);
    }
    crate::tree::RETIRE_SUBSET.store(d.retire_subset, Relaxed);
    crate::tree::RETIRE_SUBSET_CAP.store(d.retire_subset_cap, Relaxed);
    crate::tree::KEEP_WIP.store(d.keep_wip, Relaxed);
    crate::tree::KEEP_WIP_N.store(d.keep_wip_n, Relaxed);
    crate::tree::KEEP_WIP_FRAMES.store(d.keep_wip_frames, Relaxed);

    // Dense adds and sparse saves (search/src/dense.rs, tree.rs).
    let modes = match &d.dense.add {
        Some(v) => crate::dense::Modes::parse(v).map_err(anyhow::Error::msg)?,
        None => Default::default(),
    };
    let replay = match d.replay_max {
        ReplayMax::Off => None,
        ReplayMax::Frames(n) => Some(n),
        ReplayMax::Auto if modes.any() => Some(150),
        ReplayMax::Auto => None,
    };
    if let Some(r) = replay {
        crate::tree::REPLAY_MAX.store(r, Relaxed);
    }
    if modes.any() {
        anyhow::ensure!(replay.is_some(), "OBRRT_ADD needs sparse saves (OBRRT_REPLAY_MAX=0 saves every node)");
        let grid = d.dense.grid;
        anyhow::ensure!(grid >= 64 && grid % 64 == 0, "OBRRT_DENSE_GRID: a multiple of 64");
        crate::tree::DENSE_GRID.store(grid, Relaxed);
        crate::dense::configure(Some(crate::dense::Settings {
            modes,
            grid: grid as f32,
            cap: d.dense.cap.unwrap_or(if modes.all { 256 } else { 0 }),
            ne: d.dense.ne,
            mem_objects: d.dense.mem_objects,
            every: d.dense.every,
            per_obj: d.dense.per_obj,
            air: d.dense.air,
        }));
    }

    // Kinship (tree.rs).
    match k.unit {
        KinUnit::Node => {}
        KinUnit::Attempt => crate::tree::KIN_ATTEMPT.store(true, Relaxed),
        KinUnit::Frames => crate::tree::KIN_FRAMES_UNIT.store(true, Relaxed),
    }
    if let Some(f) = k.frames {
        crate::tree::KIN_FRAMES.store(f, Relaxed);
    }
    if let Some(l) = k.soft {
        crate::tree::KIN_SOFT.store(l.to_bits(), Relaxed);
    }
    if let Some(c) = k.cap {
        crate::tree::KIN_CAP.store(c, Relaxed);
    }
    if d.near.seed {
        crate::near::configure(Some(crate::near::Settings { radius: d.near.radius, p: d.near.p, per_key: d.near.keep }));
    }
    if z.backchain.on {
        let bc = &z.backchain;
        crate::backchain::configure(Some(crate::backchain::Settings {
            share: bc.share,
            kmin: bc.kmin,
            kmax: bc.kmax,
            cap: bc.cap,
            per_target: bc.per_target,
            min_weight: bc.min_weight,
        }));
    }
    Ok(())
}

/// The switches that depend on the root: the walkable mesh (`OBRRT_GEO`) and
/// the zone lock (`OBRRT_ZONE_LOCK`).
pub fn after_root(root: &St) {
    let sc = obrrt_core::config::current();
    // OBRRT_GEO=1: walkable distances over the level's floors (core/src/geo.rs),
    // for the goal focus's ranking and its approach moves (search/src/engine.rs).
    if sc.objectives.geo {
        obrrt_core::geo::set_enabled(true);
    }
    // OBRRT_ZONE_LOCK=1: the tree stays in the root's level (search/src/tree.rs).
    // Every star count stays open unless OBRRT_STAR_WINDOW says otherwise: a
    // star that doesn't end the visit (100 coins) mustn't close the others.
    if sc.zone.lock {
        crate::tree::ZONE_LOCK.store(root.level as i32, Relaxed);
        if sc.dedup.star_window.is_none() {
            crate::tree::STAR_WINDOW.store(99, Relaxed);
        }
        println!("zone lock: level {}", root.level);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `SearchConfig::default()` carries the engine's defaults: applying it changes nothing.
    #[test]
    fn default_config_is_the_engines_default() {
        let before = Config::default();
        let mut cfg = Config::default();
        apply_globals(&SearchConfig::default(), &mut cfg).unwrap();
        // Config has no PartialEq: compare field by field through a debug-like projection.
        macro_rules! same { ($($f:ident),*) => { $( assert_eq!(format!("{:?}", before.$f), format!("{:?}", cfg.$f), stringify!($f)); )* } }
        same!(ga_scale, chain_fixed, ga_new_ops, ga_pink, ga_per_objective_budget, rrt_star, arrival, arrival_pace, arrival_t, arrival_band,
            arrival_decay_either, arrival_mix, route_share, value_zones, round_plans, max_frames, kind_cap, zone_bandit, soft_dedup, start_wear,
            decay_route, entry_bandit, bandit_gamma, kind_bandit, kind_gamma, mem_budget, pick_by_b, new_eligible, earliest, earliest_cap,
            earliest_share, earliest_cell, tune_frames, reroute, reroute_schema, reroute_share, polish, reroute_margin, zone_share, zone_share_k,
            zone_share_cap, zone_share_share, zone_share_bg, replay_budget, chain_route, chain_follow, sampler_rate, node_prior, node_cap,
            kinship_mult, rebuild_check);
        assert_eq!(format!("{:?}", before.fitness), format!("{:?}", cfg.fitness));
        let (a, b) = (before.rate, cfg.rate);
        assert_eq!((a.floor, a.cap, a.tau, a.prior_frames, a.nostart_frames, a.hier, a.kind_scale, a.nostart_skip, a.skip_budget, a.route, a.novel),
            (b.floor, b.cap, b.tau, b.prior_frames, b.nostart_frames, b.hier, b.kind_scale, b.nostart_skip, b.skip_budget, b.route, b.novel));
        let d = crate::backchain::Settings::default();
        let bc = &SearchConfig::default().zone.backchain;
        assert_eq!((d.share, d.kmin, d.kmax, d.cap, d.per_target, d.min_weight), (bc.share, bc.kmin, bc.kmax, bc.cap, bc.per_target, bc.min_weight));
    }
}
