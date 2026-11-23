//! Slide paths: progress along a winding slide, for the two stars a straight
//! line to the end ignores (PSS "under 21 seconds", CCM's Big Penguin Race).
//!
//! Both courses are `TERRAIN_SLIDE` areas: every floor not flat to ~1 degree
//! is slippery (mario.c `mario_floor_is_slippery`), so Mario slides at once
//! and `update_sliding` (mario_actions_moving.c) only loses speed by the
//! `lossFactor` (0.98 on very slippery floor, up to 1.0 with the stick held
//! along the slide direction: `intendedMag / 32 * forward * 0.02 + 0.98`),
//! while `update_sliding_angle` adds `accel * steepness` (10 for the slide
//! class). Holding the stick down the slide is then the fastest, and
//! steering sideways turns the velocity (`slideVel` rotation) at the cost
//! of nothing but direction.
//!
//! A path is a polyline; progress is the remaining length from the point
//! of the path nearest to Mario (the vertical axis counted double, the
//! slide's layers lie above each other). The PSS path is derived here from
//! the slide's floor triangles (levels/pss/areas/1/collision.inc.c: the
//! triangles' centroids, shortest downhill walk to the end quad); PSS has
//! no path object. CCM's is the racing penguin's own trajectory
//! (levels/ccm/areas/2/trajectory.inc.c).

use std::sync::OnceLock;

use obrrt_core::dist;

/// PSS: Mario's entry point, then the greedy downhill walk over the slide's floor triangles' centroids (re-sampled every 400 units), then the end quad's centre.
pub const PSS_PATH: &[[f32; 3]] = &[
    [5632.0, 6656.0, -5631.0],
    [3005.0, 6129.0, -5460.0],
    [594.0, 5571.0, -5460.0],
    [-3764.0, 4574.0, -5274.0],
    [-4340.0, 4477.0, -5199.0],
    [-4861.0, 4379.0, -5017.0],
    [-5316.0, 4282.0, -4691.0],
    [-5645.0, 4179.0, -4216.0],
    [-5819.0, 4072.0, -3573.0],
    [-5856.0, 3862.0, -2080.0],
    [-5814.0, 3541.0, 186.0],
    [-5609.0, 3403.0, 900.0],
    [-5173.0, 3259.0, 1393.0],
    [-4509.0, 3111.0, 1703.0],
    [-3551.0, 2947.0, 1844.0],
    [-2678.0, 2816.0, 2017.0],
    [-2055.0, 2797.0, 2018.0],
    [-983.0, 2756.0, 2019.0],
    [1431.0, 1320.0, 2023.0],
    [2812.0, 549.0, 2024.0],
    [3516.0, 372.0, 2073.0],
    [4349.0, 231.0, 2302.0],
    [5009.0, 124.0, 2786.0],
    [5489.0, 17.0, 3428.0],
    [5715.0, -87.0, 4226.0],
    [5624.0, -191.0, 5058.0],
    [5295.0, -294.0, 5787.0],
    [4722.0, -396.0, 6305.0],
    [3970.0, -499.0, 6519.0],
    [3182.0, -596.0, 6477.0],
    [2498.0, -692.0, 6174.0],
    [2006.0, -789.0, 5628.0],
    [1779.0, -887.0, 4913.0],
    [1729.0, -956.0, 3988.0],
    [2002.0, -970.0, 3029.0],
    [1729.0, -977.0, 2689.0],
    [1729.0, -1208.0, 1965.0],
    [1729.0, -1225.0, 1523.0],
    [1729.0, -1233.0, 845.0],
    [1729.0, -1447.0, 319.0],
    [1729.0, -1481.0, -178.0],
    [1729.0, -1492.0, -997.0],
    [1729.0, -1754.0, -1358.0],
    [1729.0, -1856.0, -1921.0],
    [1727.0, -1996.0, -3216.0],
    [1711.0, -2129.0, -4261.0],
    [1355.0, -2250.0, -5097.0],
    [1170.0, -2339.0, -5559.0],
    [842.0, -2429.0, -5848.0],
    [437.0, -2526.0, -5944.0],
    [11.0, -2616.0, -5885.0],
    [-327.0, -2709.0, -5626.0],
    [-812.0, -2897.0, -4771.0],
    [-1371.0, -2978.0, -4389.0],
    [-1801.0, -3069.0, -4157.0],
    [-2348.0, -3251.0, -3884.0],
    [-3035.0, -3408.0, -3984.0],
    [-3796.0, -3516.0, -4476.0],
    [-4550.0, -3600.0, -4231.0],
    [-5165.0, -3709.0, -3788.0],
    [-5651.0, -3807.0, -3408.0],
    [-5867.0, -3908.0, -2842.0],
    [-5863.0, -3972.0, -2339.0],
    [-6050.0, -4186.0, -170.0],
    [-6186.0, -4496.0, 3115.0],
    [-6357.0, -4500.0, 3149.0],
];

/// CCM: ccm_seg7_trajectory_penguin_race (levels/ccm/areas/2/trajectory.inc.c), the racing penguin's own path, top to the finish line.
pub const CCM_PATH: &[[f32; 3]] = &[
    [-4762.0, 6660.0, -6143.0],
    [-4133.0, 6455.0, -6100.0],
    [-2000.0, 6100.0, -5944.0],
    [-1200.0, 6033.0, -5833.0],
    [1022.0, 5611.0, -6033.0],
    [3833.0, 5033.0, -6233.0],
    [6055.0, 4598.0, -5766.0],
    [6677.0, 4462.0, -4877.0],
    [6277.0, 4417.0, -3344.0],
    [4788.0, 4280.0, -1844.0],
    [2211.0, 4086.0, -555.0],
    [522.0, 3687.0, -222.0],
    [-724.0, 3443.0, -466.0],
    [-1350.0, 3302.0, -1288.0],
    [-1255.0, 3039.0, -3000.0],
    [-2233.0, 2785.0, -4533.0],
    [-3288.0, 2622.0, -4820.0],
    [-4266.0, 2480.0, -4555.0],
    [-4900.0, 2333.0, -3944.0],
    [-5066.0, 2175.0, -2977.0],
    [-4833.0, 2018.0, -1999.0],
    [-4122.0, 1866.0, -1366.0],
    [-3200.0, 1736.0, -1088.0],
    [-222.0, 1027.0, -1200.0],
    [1333.0, 761.0, -1733.0],
    [2488.0, 562.0, -2944.0],
    [2977.0, 361.0, -4988.0],
    [3754.0, 329.0, -5689.0],
    [5805.0, 86.0, -5980.0],
    [6566.0, -449.0, -4133.0],
    [6689.0, -1119.0, -888.0],
    [6688.0, -2127.0, 1200.0],
    [6666.0, -2573.0, 3555.0],
    [6600.0, -2667.0, 4333.0],
    [6366.0, -2832.0, 5722.0],
    [5844.0, -3021.0, 6355.0],
    [2955.0, -3394.0, 6255.0],
    [1788.0, -3512.0, 5988.0],
    [-89.0, -3720.0, 5188.0],
    [-732.0, -3910.0, 4144.0],
    [-722.0, -4095.0, 2688.0],
    [-1333.0, -4198.0, 1255.0],
    [-2377.0, -4302.0, 788.0],
    [-4500.0, -4684.0, 277.0],
    [-5466.0, -4790.0, 11.0],
    [-6044.0, -4860.0, -333.0],
    [-6388.0, -5079.0, -1155.0],
    [-6510.0, -5389.0, -2666.0],
    [-6476.0, -5555.0, -3622.0],
    [-6488.0, -5684.0, -4777.0],
    [-6488.0, -5829.0, -6088.0],
    [-6507.0, -5841.0, -6400.0],
];

/// Weight of the vertical axis when finding the nearest path point.
const Y_WEIGHT: f32 = 2.0;

pub struct SlidePath {
    pts: &'static [[f32; 3]],
    /// Path length from each point to the end.
    rest: Vec<f32>,
}

/// Where Mario is on a path.
pub struct Progress {
    /// Path length left from the nearest point to the end.
    pub remaining: f32,
    /// Distance from the path (the weighted one, a measure of straying).
    pub off: f32,
    /// Index of the segment's end point.
    pub next: usize,
}

impl SlidePath {
    pub fn new(pts: &'static [[f32; 3]]) -> SlidePath {
        let mut rest = vec![0.0; pts.len()];
        for i in (0..pts.len() - 1).rev() {
            rest[i] = rest[i + 1] + dist(pts[i], pts[i + 1]);
        }
        SlidePath { pts, rest }
    }

    pub fn total(&self) -> f32 {
        self.rest[0]
    }

    /// The nearest point of the path to `p` and the length beyond it.
    pub fn progress(&self, p: [f32; 3]) -> Progress {
        let w = |a: [f32; 3], b: [f32; 3]| {
            let d = [a[0] - b[0], (a[1] - b[1]) * Y_WEIGHT, a[2] - b[2]];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        let mut best = Progress { remaining: self.total(), off: f32::INFINITY, next: 1 };
        for i in 0..self.pts.len() - 1 {
            let (a, b) = (self.pts[i], self.pts[i + 1]);
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
            let t = if len2 > 0.0 {
                (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1] + (p[2] - a[2]) * ab[2]) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let q = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
            let off = w(p, q);
            if off < best.off {
                best = Progress { remaining: self.rest[i + 1] + dist(q, b), off, next: i + 1 };
            }
        }
        best
    }

    /// A point `ahead` units further along the path from Mario's nearest point.
    pub fn ahead(&self, p: [f32; 3], ahead: f32) -> [f32; 3] {
        let pr = self.progress(p);
        let want = (pr.remaining - ahead).max(0.0);
        // The first point with at most `want` left; interpolate in its segment.
        let mut i = pr.next;
        while i + 1 < self.pts.len() && self.rest[i] > want {
            i += 1;
        }
        if i == 0 || self.rest[i] >= want {
            return self.pts[i];
        }
        let (a, b) = (self.pts[i - 1], self.pts[i]);
        let seg = self.rest[i - 1] - self.rest[i];
        let t = if seg > 0.0 { (self.rest[i - 1] - want) / seg } else { 1.0 };
        [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
    }
}

pub fn pss() -> &'static SlidePath {
    static P: OnceLock<SlidePath> = OnceLock::new();
    P.get_or_init(|| SlidePath::new(PSS_PATH))
}

pub fn ccm() -> &'static SlidePath {
    static P: OnceLock<SlidePath> = OnceLock::new();
    P.get_or_init(|| SlidePath::new(CCM_PATH))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_falls_along_path() {
        for path in [pss(), ccm()] {
            let mut last = f32::INFINITY;
            for p in path.pts.iter() {
                let pr = path.progress(*p);
                assert!(pr.remaining <= last + 1.0, "{} after {last}", pr.remaining);
                last = pr.remaining;
            }
            let a = path.ahead(path.pts[0], 1000.0);
            assert!(path.progress(a).remaining < path.total() - 900.0);
        }
    }
}
