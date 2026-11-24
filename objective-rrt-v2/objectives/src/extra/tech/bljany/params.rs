//! The BLJ spot detectors' thresholds as data (`OBRRT_BLJ_DETECTOR=<file>`,
//! e.g. ref/blj_detector.json; tools/analysis/blj_tune.py writes new ones).
//! Without the switch every value is the code default, which is the
//! constant the detectors used before, so nothing changes.
//!
//! The file is a flat JSON object of `"name": number` (other keys, such as
//! a `"changes"` log, are ignored). Names: the fields of [`Detector`].

use std::sync::OnceLock;

/// Every detector threshold (defaults: the former constants).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detector {
    /// Slope clause: ny range (rise tan >= 1.25 up to the wall limit) and
    /// the smallest triangle (projected), static and on objects.
    pub slope_ny_min: f32,
    pub slope_ny_max: f32,
    pub slope_area: f32,
    pub obj_slope_area: f32,
    /// Slope starts: also grid points this far apart over the triangle (0:
    /// the centroid only).
    pub slope_spacing: f32,
    /// Riser clause: height and length range; the floor in front within
    /// `front_lo..front_hi` of the foot, the one behind up to `behind_extra`
    /// above the top.
    pub riser_h_min: f32,
    pub riser_h_max: f32,
    pub riser_min_len: f32,
    pub riser_front_lo: f32,
    pub riser_front_hi: f32,
    pub riser_behind_extra: f32,
    /// Ceiling clause: headroom range over 160.
    pub ceil_head_lo: f32,
    pub ceil_head_hi: f32,
    /// Corner rules: a riser end within `corner_dist` of a wall taller
    /// than `corner_tall`; the pinned BLJ only on risers up to `pin_h`.
    pub corner_dist: f32,
    pub corner_tall: f32,
    pub pin_h: f32,
    /// The scan radius and spots per base, and the lab's match distance.
    pub scan_r: f32,
    pub scan_n: f32,
    pub match_dist: f32,
}

impl Default for Detector {
    fn default() -> Self {
        Detector {
            slope_ny_min: 0.2924,
            slope_ny_max: 0.6247,
            slope_area: 20000.0,
            obj_slope_area: 5000.0,
            slope_spacing: 0.0,
            riser_h_min: 15.0,
            riser_h_max: 60.0,
            riser_min_len: 100.0,
            riser_front_lo: -60.0,
            riser_front_hi: 8.0,
            riser_behind_extra: 60.0,
            ceil_head_lo: 0.0,
            ceil_head_hi: 50.0,
            corner_dist: 25.0,
            corner_tall: 150.0,
            pin_h: 37.5,
            scan_r: 2500.0,
            scan_n: 3.0,
            match_dist: 300.0,
        }
    }
}

/// `"name": number` pairs of a flat JSON object.
pub fn parse_numbers(text: &str) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'"' {
            let j = i + 1 + text[i + 1..].find('"').unwrap_or(0);
            let key = &text[i + 1..j];
            let mut k = j + 1;
            while k < b.len() && (b[k] == b' ' || b[k] == b'\n' || b[k] == b'\t' || b[k] == b'\r') {
                k += 1;
            }
            if k < b.len() && b[k] == b':' {
                k += 1;
                while k < b.len() && b[k].is_ascii_whitespace() {
                    k += 1;
                }
                let e = text[k..].find(|c: char| !(c.is_ascii_digit() || "+-.eE".contains(c))).map_or(b.len(), |e| k + e);
                if let Ok(v) = text[k..e].parse::<f64>() {
                    out.push((key.to_string(), v));
                }
                i = e.max(j + 1);
                continue;
            }
            i = j + 1;
            continue;
        }
        i += 1;
    }
    out
}

impl Detector {
    /// Applies `name = v` pairs (unknown names ignored); returns the names
    /// applied.
    pub fn apply(&mut self, pairs: &[(String, f64)]) -> Vec<String> {
        let mut used = Vec::new();
        for (k, v) in pairs {
            let v = *v as f32;
            let f = match k.as_str() {
                "slope_ny_min" => &mut self.slope_ny_min,
                "slope_ny_max" => &mut self.slope_ny_max,
                "slope_area" => &mut self.slope_area,
                "obj_slope_area" => &mut self.obj_slope_area,
                "slope_spacing" => &mut self.slope_spacing,
                "riser_h_min" => &mut self.riser_h_min,
                "riser_h_max" => &mut self.riser_h_max,
                "riser_min_len" => &mut self.riser_min_len,
                "riser_front_lo" => &mut self.riser_front_lo,
                "riser_front_hi" => &mut self.riser_front_hi,
                "riser_behind_extra" => &mut self.riser_behind_extra,
                "ceil_head_lo" => &mut self.ceil_head_lo,
                "ceil_head_hi" => &mut self.ceil_head_hi,
                "corner_dist" => &mut self.corner_dist,
                "corner_tall" => &mut self.corner_tall,
                "pin_h" => &mut self.pin_h,
                "scan_r" => &mut self.scan_r,
                "scan_n" => &mut self.scan_n,
                "match_dist" => &mut self.match_dist,
                _ => continue,
            };
            *f = v;
            used.push(k.clone());
        }
        used
    }

    /// The parameters as the flat JSON the loader reads.
    pub fn to_json(&self) -> String {
        format!(
            "{{\"slope_ny_min\": {}, \"slope_ny_max\": {}, \"slope_area\": {}, \"obj_slope_area\": {}, \"slope_spacing\": {}, \"riser_h_min\": {}, \"riser_h_max\": {}, \"riser_min_len\": {}, \"riser_front_lo\": {}, \"riser_front_hi\": {}, \"riser_behind_extra\": {}, \"ceil_head_lo\": {}, \"ceil_head_hi\": {}, \"corner_dist\": {}, \"corner_tall\": {}, \"pin_h\": {}, \"scan_r\": {}, \"scan_n\": {}, \"match_dist\": {}}}",
            self.slope_ny_min, self.slope_ny_max, self.slope_area, self.obj_slope_area, self.slope_spacing, self.riser_h_min, self.riser_h_max, self.riser_min_len, self.riser_front_lo, self.riser_front_hi, self.riser_behind_extra, self.ceil_head_lo, self.ceil_head_hi, self.corner_dist, self.corner_tall, self.pin_h, self.scan_r, self.scan_n, self.match_dist
        )
    }
}

/// The run's thresholds (OBRRT_BLJ_DETECTOR, else the defaults).
pub fn get() -> &'static Detector {
    static D: OnceLock<Detector> = OnceLock::new();
    D.get_or_init(|| {
        let mut d = Detector::default();
        if let Ok(path) = obrrt_core::sys::env::var("OBRRT_BLJ_DETECTOR") {
            match obrrt_core::sys::fs::read_to_string(&path) {
                Ok(t) => {
                    let used = d.apply(&parse_numbers(&t));
                    eprintln!("blj detector: {path}: {}", used.join(", "));
                }
                Err(e) => eprintln!("blj detector: {path}: {e}"),
            }
        }
        d
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_round_trip_through_json() {
        let d = Detector::default();
        let mut e = Detector { slope_area: 1.0, ..Detector::default() };
        e.apply(&parse_numbers(&d.to_json()));
        assert_eq!(d, e);
        let mut f = Detector::default();
        let used = f.apply(&parse_numbers("{\"slope_area\": 1.5e4, \"changes\": [{\"x\": \"y\"}], \"riser_h_max\": 70}"));
        assert_eq!(used, vec!["slope_area", "riser_h_max"]);
        assert_eq!((f.slope_area, f.riser_h_max), (15000.0, 70.0));
    }
}
