//! The objective-RRT search's settings as one typed value, and the adapters that fill it.
//!
//! [`SearchConfig`] is plain data (serde, `Default` = no switch set). The search crates take it
//! as a value; reading the process environment is this crate's job, at the edge:
//! [`from_env`] (the legacy `OBRRT_*` variables) and [`from_toml`] (a file).
mod config;
mod env;

pub use config::*;
pub use env::{Lookup, from_env, from_env_with};

/// Parses a TOML document into a config; fields it leaves out keep their defaults.
pub fn from_toml(text: &str) -> anyhow::Result<SearchConfig> {
    Ok(toml::from_str(text)?)
}

/// The config as TOML (what [`from_toml`] reads back).
pub fn to_toml(cfg: &SearchConfig) -> anyhow::Result<String> {
    Ok(toml::to_string_pretty(cfg)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_round_trips() {
        let mut c = SearchConfig::default();
        c.sampler.pick_by_b = true;
        c.dedup.dense.add = Some("all".into());
        c.stop.exit_skip = vec![2, 3];
        let back = from_toml(&to_toml(&c).unwrap()).unwrap();
        assert_eq!(c, back);
        assert_eq!(from_toml("").unwrap(), SearchConfig::default());
    }

    #[test]
    fn env_lookup_applies_switches() {
        let vars = [("OBRRT_ROUND", "32"), ("OBRRT_EXIT_SKIP", "2, 3"), ("OBRRT_STOP_FLAG", "0x200"), ("OBRRT_PICK_BY_B", "0")];
        let get = |k: &str| vars.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string());
        let c = from_env_with(SearchConfig::default(), &get).unwrap();
        assert_eq!(c.budget.round_plans, 32);
        assert!(!c.sampler.pick_by_b);
        assert_eq!(c.stop.exit_skip, vec![2, 3]);
        assert_eq!(c.stop.flag, 0x200);
        let bad = |k: &str| (k == "OBRRT_ROUND").then(|| "x".to_string());
        assert!(from_env_with(SearchConfig::default(), &bad).is_err());
        assert_eq!(from_env_with(SearchConfig::default(), &|_| None).unwrap(), SearchConfig::default());
    }
}
