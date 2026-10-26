//! Collectibles: coins of every kind (blue coins and the blue coin switch, coin rings and formations, red coins, coin-dropping objects), 1-up mushrooms, caps (wing/metal/vanish) and the objects that give them, secret-star triggers.
//!
//! Generated objectives for this category; see `extra/mod.rs`. What the
//! parent module already covers (single coins via `Coin::Collect`, ! boxes,
//! breakable boxes, grabbing) is not repeated; these add the sequences and
//! gating it lacks:
//!
//! - `BlueCoinSwitch::Press`: ground pound the blue coin switch.
//! - `HiddenBlueCoin::Collect`: a switch's blue coin, only while it is out.
//! - `Cap::Collect`: wing, metal, vanish caps and Mario's own lost cap.
//! - `OneUp::Collect`: every 1-up mushroom variant, once it is revealed.
//! - `SecretTrigger::Touch` / `OneUpTrigger::Touch`: the invisible secrets
//!   (5 make a star) and the triggers that reveal hidden 1-ups.
//! - `RedCoin::Next`: the course's next red coin, wherever it is.
//! - `CoinFormation::Sweep`: every coin of a line, ring or arrow at once.
//! - `CrazyBox::Bounce`: ride a jumping box until it breaks into 5 coins.
//! - `Heart::Spin`: run through a spinning heart to heal.

use std::sync::Arc;

use obrrt_core::{Obj, Objective, St};

mod blue;
mod caps;
mod coins;
mod common;
mod heart;
mod oneup;
mod triggers;

/// Objectives `o` offers, seen from base state `b` (called for every
/// object within range of `b`; `b` is never a dialog or menu state).
pub fn generate(b: &St, o: &Obj, out: &mut Vec<Arc<dyn Objective>>) {
    match o.bhv_name() {
        "bhvBlueCoinSwitch" if o.action == blue::BLUE_COIN_SWITCH_ACT_IDLE => out.push(Arc::new(blue::Press::new(b, o))),
        "bhvHiddenBlueCoin" if o.action == blue::HIDDEN_BLUE_COIN_ACT_ACTIVE => {
            out.push(Arc::new(blue::Collect::new(b, o)))
        }
        "bhvWingCap" | "bhvMetalCap" | "bhvVanishCap" | "bhvNormalCap" => {
            if let Some(c) = caps::Collect::new(b, o) {
                out.push(Arc::new(c));
            }
        }
        name if oneup::is_one_up(name) => out.push(Arc::new(oneup::Collect::new(b, o))),
        "bhvHiddenStarTrigger" => out.push(Arc::new(triggers::Touch::secret(b, o))),
        "bhvHidden1upTrigger" | "bhvHidden1upInPoleTrigger" => out.push(Arc::new(triggers::Touch::one_up(b, o))),
        "bhvCoinFormation" => out.push(Arc::new(coins::Sweep::new(b, o))),
        "bhvJumpingBox" => out.push(Arc::new(coins::CrazyBox::new(b, o))),
        "bhvRecoveryHeart" if b.wedges() < 8 => out.push(Arc::new(heart::Spin::new(b, o))),
        _ => {}
    }
}

/// Objectives that come from the level rather than one object (called once
/// per `b`).
pub fn generate_level(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    if let Some(r) = coins::RedNext::new(b) {
        out.push(Arc::new(r));
    }
}
