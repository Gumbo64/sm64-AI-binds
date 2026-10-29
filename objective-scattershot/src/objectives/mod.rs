//! The objective kinds a schema can use, by name.

pub mod generic;
pub mod stars;
pub mod interact;
pub mod enemies;
pub mod bosses;
pub mod npcs;
pub mod warps;
pub mod tech_blj;
pub mod tech_kick;
pub mod tech_misc;
pub mod boss;
pub mod busy;
pub mod lens;
pub mod progress;
pub mod util;

use anyhow::{Result, bail};
use serde_json::Value;

use crate::objective::Objective;

pub fn build(kind: &str, params: &Value) -> Result<Box<dyn Objective>> {
    let p = if params.is_null() { Value::Object(Default::default()) } else { params.clone() };
    Ok(match kind {
        "Root" => Box::new(serde_json::from_value::<progress::Root>(p)?),
        "Stage" => Box::new(serde_json::from_value::<progress::Stage>(p)?),
        "Lens" => Box::new(serde_json::from_value::<lens::Lens>(p)?),
        "Side" => Box::new(serde_json::from_value::<lens::Side>(p)?),
        "Busy" => Box::new(serde_json::from_value::<busy::Busy>(p)?),
        "Boss" => Box::new(serde_json::from_value::<boss::Boss>(p)?),
        _ => {
            // The ported families, each with its own kinds.
            let families: [fn(&str, &Value) -> Option<Result<Box<dyn Objective>>>; 10] = [
                generic::build, stars::build, interact::build, enemies::build, bosses::build,
                npcs::build, warps::build, tech_blj::build, tech_kick::build, tech_misc::build,
            ];
            for build in families {
                if let Some(obj) = build(kind, &p) {
                    return obj;
                }
            }
            bail!("unknown objective kind {kind}")
        }
    })
}
