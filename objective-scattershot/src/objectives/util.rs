//! Small quantization helpers shared by objectives.

/// A position on a 2^k grid packed into one i32: each of qx, qy, qz =
/// floor(p / 2^k) keeps its low 10 bits (signed, -512..511), as
/// `(qx << 20) | (qy << 10) | qz`. The GUI decodes it with 10-bit sign
/// extension. Every field named "pos" (or ending in "pos") uses this.
pub fn pack_pos(p: [f32; 3], grid_log2: u32) -> i32 {
    let q = |v: f32| (v / (1u32 << grid_log2) as f32).floor() as i32 & 0x3FF;
    (q(p[0]) << 20) | (q(p[1]) << 10) | q(p[2])
}

pub fn unpack_pos(v: i32) -> [i32; 3] {
    let s = |x: i32| ((x & 0x3FF) << 22) >> 22;
    [s(v >> 20), s(v >> 10), s(v)]
}

pub fn bucket(v: f32, step: f32) -> i32 {
    (v / step).floor() as i32
}

/// A kind of Mario action, for bucketing the many action values: a lens lists
/// the groups it cares about and every other action is "other" (class 0).
#[derive(serde::Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum ActionGroup {
    Idle,
    /// Walking, running, braking, decelerating and turning round.
    Walking,
    /// Jumps, double and triple jumps, flips and wall kicks.
    Jump,
    LongJump,
    /// Free fall and rollouts.
    Freefall,
    /// The dive in the air and the slides it ends in (dive slide, belly slide).
    Dive,
    /// The slide kick (B while crouch-sliding): the ground slide and the kick in the air.
    SlideKick,
    /// Going through a door: pulling or pushing it open, unlocking a key or
    /// star door, going in a star door, and appearing on the other side of a
    /// warp door (the set `objective-rrt`'s `door.rs` calls `in_door`, plus the spawn).
    Door,
    /// Waiting for, or reading, an NPC's dialog (`ACT_WAITING_FOR_DIALOG`, `ACT_READING_NPC_DIALOG`).
    Dialog,
}

impl ActionGroup {
    /// By the action's id (its low 9 bits), as in sm64.h.
    pub fn contains(self, action: u32) -> bool {
        let id = action & 0x1FF;
        match self {
            ActionGroup::Idle => id == 0x001,
            ActionGroup::Walking => matches!(id, 0x040 | 0x043 | 0x044 | 0x045 | 0x04A),
            ActionGroup::Jump => matches!(id, 0x080..=0x083 | 0x086 | 0x087),
            ActionGroup::LongJump => id == 0x088,
            ActionGroup::Freefall => matches!(id, 0x08C | 0x0A6 | 0x0AD),
            ActionGroup::Dive => matches!(id, 0x08A | 0x056 | 0x053),
            ActionGroup::SlideKick => matches!(id, 0x05A | 0x0AA),
            ActionGroup::Door => matches!(action, 0x0000_1320 | 0x0000_1321 | 0x0000_1322 | 0x0000_132E | 0x0000_132F | 0x0000_1331),
            ActionGroup::Dialog => matches!(action, 0x0000_130A | 0x2000_1306),
        }
    }
}

/// The inputs for a state that only needs time: a door animation, a dialog
/// or a menu. As `objective-rrt`'s `press_through`: the stick held at
/// (x, y), A pressed on every 4th frame, which advances a dialog and does
/// nothing in an animation. `x` is -127 to answer yes, 127 for no, 0 to just read.
pub fn press_through(x: i8, y: i8, frames: usize) -> Vec<sm64_binds::GamePad> {
    (0..frames).map(|f| sm64_binds::GamePad::new(if f % 4 == 3 { sm64_binds::buttons::A_BUTTON } else { 0 }, x, y)).collect()
}

/// 1 + the index of the first of `groups` containing `action`, else 0 (other).
pub fn action_class(action: u32, groups: &[ActionGroup]) -> i32 {
    groups.iter().position(|g| g.contains(action)).map_or(0, |i| i as i32 + 1)
}

/// Mario's yaw (an s16 angle) as one of `n` sectors, 0..n.
pub fn yaw_sector(yaw: i16, n: u32) -> i32 {
    ((yaw as u16 as u32 * n) >> 16) as i32
}

/// Mario's yaw relative to `target` (both s16 angles) in uneven buckets: within
/// `fine_range` (s16 units) either side, `fine_n` equal buckets (0..fine_n,
/// left to right); the rest of the circle in `coarse_n` equal buckets after
/// them (odd counts put the target, and straight away from it, in the middle of a bucket). Most of the circle is facing the wrong way and needs no precision.
pub fn yaw_bucket(yaw: i16, target: i16, fine_range: i32, fine_n: u32, coarse_n: u32) -> i32 {
    let d = i32::from(yaw.wrapping_sub(target)); // -32768..=32767
    if d.abs() <= fine_range {
        let i = ((d + fine_range) as i64 * i64::from(fine_n) / (2 * i64::from(fine_range) + 1)) as i32;
        return i.min(fine_n as i32 - 1);
    }
    let rest = 65536 - 2 * fine_range;
    let e = (d - fine_range).rem_euclid(65536) - 1; // position along the rest, from the fine zone's right edge
    fine_n as i32 + ((e.clamp(0, rest - 1) as i64 * i64::from(coarse_n)) / i64::from(rest)) as i32
}

pub fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// 1 on the side `normal` points to, else 0.
pub fn side(p: [f32; 3], point: [f32; 3], normal: [f32; 3]) -> i32 {
    let d: f32 = (0..3).map(|i| (p[i] - point[i]) * normal[i]).sum();
    (d > 0.0) as i32
}

pub fn in_box(p: [f32; 3], min: [f32; 3], max: [f32; 3]) -> bool {
    (0..3).all(|i| min[i] <= p[i] && p[i] <= max[i])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pos_round_trip() {
        for p in [[0.0, 0.0, 0.0], [-2230.0, 0.0, 303.0], [8000.0, -8000.0, -1.0], [-5513.0, 717.0, -4324.0]] {
            for k in [6, 8, 10] {
                let q = unpack_pos(pack_pos(p, k));
                for i in 0..3 {
                    assert_eq!(q[i], (p[i] / (1 << k) as f32).floor() as i32);
                }
            }
        }
    }

    #[test]
    fn action_classes() {
        let g = [ActionGroup::Walking, ActionGroup::LongJump];
        assert_eq!(action_class(0x0400_0440, &g), 1); // ACT_WALKING
        assert_eq!(action_class(0x0400_0445, &g), 1); // ACT_BRAKING
        assert_eq!(action_class(0x0300_0888, &g), 2); // ACT_LONG_JUMP
        assert_eq!(action_class(0x0300_0880, &g), 0); // ACT_JUMP: other
        assert_eq!(action_class(0x0C40_0201, &g), 0); // ACT_IDLE: other
        assert_eq!(action_class(0x0300_0880, &[ActionGroup::Jump]), 1);
        // The slide kick, on the ground (ACT_SLIDE_KICK_SLIDE) and in the air (ACT_SLIDE_KICK).
        let k = [ActionGroup::Walking, ActionGroup::LongJump, ActionGroup::SlideKick];
        assert_eq!((action_class(0x0080_045A, &k), action_class(0x0180_08AA, &k)), (3, 3));
        assert_eq!(action_class(0x0480_8459, &k), 0); // ACT_CROUCH_SLIDE itself is other
        // Dives: the dive, the dive slide and the belly slide; not butt or crouch slides.
        let d = [ActionGroup::Walking, ActionGroup::LongJump, ActionGroup::SlideKick, ActionGroup::Dive];
        assert_eq!((action_class(0x0188_088A, &d), action_class(0x0088_0456, &d), action_class(0x008C_0453, &d)), (4, 4, 4));
        assert_eq!((action_class(0x0084_0452, &d), action_class(0x0480_8459, &d)), (0, 0));
    }

    #[test]
    fn press_through_presses_every_fourth_frame() {
        let p = press_through(-127, 0, 9);
        let a: Vec<bool> = p.iter().map(|g| g.button != 0).collect();
        assert_eq!(a, vec![false, false, false, true, false, false, false, true, false]);
        assert!(p.iter().all(|g| g.stick_x == -127 && g.stick_y == 0));
    }

    #[test]
    fn door_actions() {
        for a in [0x0000_1320, 0x0000_1321, 0x0000_1322, 0x0000_132E, 0x0000_132F, 0x0000_1331] {
            assert!(ActionGroup::Door.contains(a), "{a:#x}");
        }
        assert!(!ActionGroup::Door.contains(0x0400_0440), "walking");
        assert_eq!(action_class(0x0000_1320, &[ActionGroup::Walking, ActionGroup::Door]), 2);
    }

    #[test]
    fn uneven_yaw_buckets() {
        // Facing 0x8000 (-z); fine within 45 degrees (0x2000), 7 + 3 buckets.
        let b = |yaw: i16| yaw_bucket(yaw, i16::MIN, 0x2000, 7, 3);
        assert_eq!(b(i16::MIN), 3); // straight at it: a middle fine bucket
        assert_eq!(b(i16::MIN + 0x2000), 6);
        assert_eq!(b(i16::MIN.wrapping_sub(0x2000)), 0);
        let all: std::collections::BTreeSet<i32> = (0..65536).map(|y| b((y - 32768) as i16)).collect();
        assert_eq!(all.into_iter().collect::<Vec<_>>(), (0..10).collect::<Vec<_>>());
        // Facing straight away is a coarse bucket, and nearby angles share it.
        assert_eq!(b(0), b(300));
        assert!(b(0) >= 7);
    }

    #[test]
    fn yaw_sectors() {
        assert_eq!(yaw_sector(0, 16), 0);
        assert_eq!(yaw_sector(-1, 16), 15);
        assert_eq!(yaw_sector(i16::MIN, 16), 8);
    }
}
