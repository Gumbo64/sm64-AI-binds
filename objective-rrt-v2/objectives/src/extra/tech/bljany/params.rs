//! The BLJ spot registry's clause thresholds as data (`OBRRT_BLJ_DETECTOR=<file>`;
//! tools/analysis/blj_tune.py writes such files). Without the switch every
//! value is the code default.
//!
//! The file is a flat JSON object of `"name": number` (other keys, such as
//! a `"changes"` log, and names the registry no longer reads are ignored).
//! Names: the fields of [`Detector`].

/// The thresholds [`super::spots`] classifies observed chain starts with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detector {
    /// Slope clause: ny range (rise tan >= 1.25 up to the wall limit).
    pub slope_ny_min: f32,
    pub slope_ny_max: f32,
    /// Ceiling clause: headroom range over 160.
    pub ceil_head_lo: f32,
    pub ceil_head_hi: f32,
}

impl Default for Detector {
    fn default() -> Self {
        Detector { slope_ny_min: 0.2924, slope_ny_max: 0.6247, ceil_head_lo: 0.0, ceil_head_hi: 50.0 }
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
                "ceil_head_lo" => &mut self.ceil_head_lo,
                "ceil_head_hi" => &mut self.ceil_head_hi,
                _ => continue,
            };
            *f = v;
            used.push(k.clone());
        }
        used
    }
}

/// The run's thresholds (OBRRT_BLJ_DETECTOR, else the defaults).
pub fn get() -> &'static Detector {
    obrrt_core::per_config!(Detector, |cfg| {
        let mut d = Detector::default();
        if let Some(path) = &cfg.objectives.blj_detector {
            match obrrt_core::sys::fs::read_to_string(path) {
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
    fn apply_reads_known_names_only() {
        let mut f = Detector::default();
        let used = f.apply(&parse_numbers("{\"ceil_head_hi\": 1.5e1, \"changes\": [{\"x\": \"y\"}], \"riser_h_max\": 70}"));
        assert_eq!(used, vec!["ceil_head_hi"]);
        assert_eq!(f.ceil_head_hi, 15.0);
    }
}
