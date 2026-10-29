//! Mario's self objectives: generated from every base state, whatever objects
//! are around.

mod exit;
mod idle;
mod pos;
mod time;

pub use exit::GoToCastle;
pub use idle::{Settle, Wait};
pub use pos::{PosDist, swim_targets, underwater_point};
pub use time::PosTime;
