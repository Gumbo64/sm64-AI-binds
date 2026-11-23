//! Compact text for the JSON the other functions return.

use std::fmt::Write;

use serde_json::Value;

/// A number without trailing noise.
fn n(v: &Value) -> String {
    match v {
        Value::Number(x) => match x.as_f64() {
            Some(f) if f.fract() == 0.0 && f.abs() < 1e15 => format!("{}", f as i64),
            Some(f) => format!("{f:.1}"),
            None => x.to_string(),
        },
        Value::Null => "-".into(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        other => other.to_string(),
    }
}

fn pt(v: &Value) -> String {
    match v.as_array() {
        Some(a) => format!("({})", a.iter().map(n).collect::<Vec<_>>().join(", ")),
        None => "-".into(),
    }
}

fn list(v: &Value) -> String {
    v.as_array().map_or(String::new(), |a| a.iter().map(n).collect::<Vec<_>>().join(","))
}

fn s(v: &Value) -> &str {
    v.as_str().unwrap_or("?")
}

pub fn render(kind: &str, v: &Value) -> String {
    let mut o = String::new();
    match kind {
        "state" => state(v, &mut o),
        "progress" => progress(v, &mut o),
        "objects" => objects(v, &mut o),
        "geo" => geo(v, &mut o),
        "behavior" => behavior(v, &mut o),
        "level" => level(v, &mut o),
        "objectives" => objectives(v, &mut o),
        "diff" => diff(v, &mut o),
        _ => o = serde_json::to_string_pretty(v).unwrap_or_default(),
    }
    o.trim_end().to_string()
}

fn state(v: &Value, o: &mut String) {
    let (z, m) = (&v["zone"], &v["mario"]);
    let _ = writeln!(o, "frame {}  {} area {} act {}  (level {}, course {} {})", n(&v["frame"]), n(&z["level_name"]), n(&z["area"]), n(&z["act"]), n(&z["level"]), n(&z["course"]), n(&z["course_name"]));
    let _ = writeln!(o, "pos {}  vel {}  fvel {}  yaw {} deg (raw {})  pitch {}", pt(&m["pos"]), pt(&m["vel"]), n(&m["forward_vel"]), n(&m["yaw_deg"]), n(&m["yaw"]), n(&m["pitch"]));
    let a = &m["action"];
    let _ = writeln!(o, "action {} ({}) timer {}  [{}]  controllable: {}", n(&a["name"]), n(&a["value"]), n(&m["action_timer"]), list(&a["flags"]), n(&v["controllable"]));
    let _ = writeln!(o, "health {}/8 wedges (raw {})  coins {}  lives {}  stars {}  keys {}", n(&m["health"]["wedges"]), n(&m["health"]["raw"]), n(&m["coins"]), n(&m["lives"]), n(&m["stars"]), n(&m["keys"]));
    let held = if m["held"].is_null() { "nothing".to_string() } else { format!("{} (slot {})", n(&m["held"]["bhv"]), n(&m["held"]["slot"])) };
    let _ = writeln!(o, "cap [{}] timer {}  invincible {}  hurt {} heal {}  holding {}", list(&m["cap"]["flags"]), n(&m["cap"]["timer"]), n(&m["invincibility"]), n(&m["hurt_counter"]), n(&m["heal_counter"]), held);
    let _ = writeln!(o, "mario_flags: {}", list(&m["mario_flags"]));
    let f = &m["floor"];
    if f["none"].as_bool() == Some(true) {
        let _ = write!(o, "floor: none (y {})", n(&f["y"]));
    } else {
        let _ = write!(o, "floor y {} {} normal {} class {} slope {} steep {}", n(&f["y"]), n(&f["type"]), pt(&f["normal"]), n(&f["class"]), n(&f["slope"]), n(&f["steep"]));
        if !f["object"].is_null() {
            let _ = write!(o, " on object {} (slot {})", n(&f["object"]["bhv"]), n(&f["object"]["slot"]));
        }
    }
    let w = &m["water"];
    let _ = writeln!(
        o,
        "\nceil {}  wall normal {}  water {}",
        n(&m["ceil_y"]),
        if m["wall_normal"].is_null() { "none".into() } else { pt(&m["wall_normal"]) },
        if w.is_null() { "none".into() } else { format!("level {} (Mario {} below surface)", n(&w["level"]), n(&w["depth"])) }
    );
    let md = &v["modal"];
    let modal = if md.is_null() {
        "none".to_string()
    } else {
        md.as_object().map_or(String::new(), |m| m.iter().map(|(k, x)| format!("{k}={}", n(x))).collect::<Vec<_>>().join(" "))
    };
    let _ = write!(o, "camera yaw {} deg  play_mode {} transition {}  modal: {}", n(&v["camera_yaw_deg"]), n(&v["play_mode"]), n(&v["transition"]), modal);
}

fn progress(v: &Value, o: &mut String) {
    let _ = writeln!(o, "stars {} (flags: {} course + {} castle)  keys {}", n(&v["stars"]), n(&v["course_stars"]), n(&v["castle_stars"]), n(&v["keys"]));
    if let Some(c) = v["courses"].as_object() {
        for (k, x) in c {
            let cannon = if x["cannon_open"].as_bool() == Some(true) { "  cannon open" } else { "" };
            let _ = writeln!(o, "  {k:<34} stars {}{cannon}", list(&x["stars"]));
        }
    }
    let f = &v["save_flags"];
    let _ = writeln!(o, "save flags {}: {}", n(&f["raw"]), list(&f["set"]));
    let clear: Vec<&str> = f["clear"].as_array().map_or(vec![], |a| a.iter().filter_map(|x| x.as_str()).filter(|x| !x.starts_with("COLLECTED_") && !x.starts_with("CAP_ON_") && *x != "FILE_EXISTS").collect());
    let _ = writeln!(o, "not set: {}", clear.join(","));
    let r = &v["red_coins"];
    let _ = write!(o, "red coins in {}: {} collected, {} loaded now", n(&r["in"]), n(&r["collected"]), n(&r["loaded_now"]));
}

fn objects(v: &Value, o: &mut String) {
    let _ = writeln!(o, "frame {}  {} of {} active objects{}{}  (Mario {})", n(&v["frame"]), n(&v["shown"]), n(&v["active_total"]),
        if v["near"].is_null() { String::new() } else { format!(", within {}", n(&v["near"])) },
        if v["bhv"].is_null() { String::new() } else { format!(", bhv ~ {}", n(&v["bhv"])) }, pt(&v["mario"]));
    let _ = writeln!(o, "{:>3} {:<28} {:>6} {:>22} {:>5} {:>4} {:>4} {:>3} {:>4} {:<10}  details", "slot", "bhv", "dist", "pos", "act", "sub", "hp", "tmr", "fvel", "params");
    for x in v["objects"].as_array().map_or(&[][..], |a| a.as_slice()) {
        let mut d = vec![];
        if !x["interact"]["type"].as_array().is_none_or(|a| a.is_empty()) {
            d.push(format!("int:{}", list(&x["interact"]["type"])));
        }
        if !x["interact"]["status"].as_array().is_none_or(|a| a.is_empty()) {
            d.push(format!("status:{}", list(&x["interact"]["status"])));
        }
        if x["held_state"].as_i64().unwrap_or(0) != 0 {
            d.push(format!("held{}", n(&x["held_state"])));
        }
        if x["intangible"].as_i64().unwrap_or(0) != 0 {
            d.push(format!("intang{}", n(&x["intangible"])));
        }
        if x["prev_action"] != x["action"] {
            d.push(format!("prev{}", n(&x["prev_action"])));
        }
        let vel = x["vel"].as_array().map_or(0.0, |a| a.iter().map(|c| c.as_f64().unwrap_or(0.0).powi(2)).sum::<f64>().sqrt());
        if vel > 0.5 {
            d.push(format!("vel{}", pt(&x["vel"])));
        }
        d.push(format!("hb{}x{}", n(&x["hitbox"][0]), n(&x["hitbox"][1])));
        let _ = writeln!(o, "{:>3} {:<28} {:>6} {:>22} {:>5} {:>4} {:>4} {:>3} {:>4} {:<10}  {}", n(&x["slot"]), s(&x["bhv"]), n(&x["dist"]), pt(&x["pos"]), n(&x["action"]), n(&x["sub_action"]), n(&x["health"]), n(&x["timer"]), n(&x["forward_vel"]), s(&x["params"]), d.join(" "));
    }
}

fn tri_line(t: &Value) -> String {
    format!("d{:>5} {:<26} {} {} {} n{}", n(&t["d"]), s(&t["type"]), pt(&t["v"][0]), pt(&t["v"][1]), pt(&t["v"][2]), pt(&t["n"]))
}

fn geo(v: &Value, o: &mut String) {
    let _ = writeln!(o, "geo around {} radius {}  (Mario water level {})", pt(&v["center"]), n(&v["radius"]), n(&v["mario_water_level"]));
    let a = &v["atlas"];
    let _ = writeln!(o, "atlas: {}", if a["map"].is_null() { format!("none ({})", n(&a["reason"])) } else { format!("{} / {}", n(&a["map"]), n(&a["variant"])) });
    for (k, label) in [("floors", "floors"), ("walls", "walls"), ("ceilings", "ceilings")] {
        let g = &v[k];
        let by = g["by_type"].as_object().map_or(String::new(), |m| m.iter().map(|(t, c)| format!("{t}:{}", n(c))).collect::<Vec<_>>().join(" "));
        let _ = writeln!(o, "{label}: {} in range  {by}", n(&g["count"]));
        for t in g["nearest"].as_array().map_or(&[][..], |a| a.as_slice()).iter().take(if k == "ceilings" { 2 } else { 4 }) {
            let _ = writeln!(o, "    {}", tri_line(t));
        }
    }
    let probe = |p: &Value| {
        let tr = &p["atlas"];
        let at = if tr.is_null() {
            String::new()
        } else {
            format!("  atlas[{}{}]", list(&tr["flags"]), if tr["walk_frames"].is_null() { String::new() } else { format!(" walk {}f", n(&tr["walk_frames"])) })
        };
        format!("floor {:>6} (dy {:>6}) {:<22} nY {:>4} ceil {:>6}{at}", n(&p["floor_y"]), n(&p["dy"]), n(&p["type"]), n(&p["normal_y"]), n(&p["ceil_y"]))
    };
    let _ = writeln!(o, "at point: {}", probe(&v["at_point"]));
    let _ = writeln!(o, "probes (world-axis direction, distance):");
    for p in v["probes"].as_array().map_or(&[][..], |a| a.as_slice()) {
        let _ = writeln!(o, "  {:>5} {:>4}: {}", n(&p["dir"]), n(&p["dist"]), probe(p));
    }
    let w = v["water"].as_array().map_or(&[][..], |a| a.as_slice());
    if w.is_empty() {
        let _ = writeln!(o, "water boxes: none in range");
    }
    for b in w {
        let _ = writeln!(o, "{} box id {} x {} z {} y {}{}", n(&b["kind"]), n(&b["id"]), pt(&b["x"]), pt(&b["z"]), n(&b["y"]), if b["contains_point"].as_bool() == Some(true) { "  (contains point)" } else { "" });
    }
    let d = &v["dynamic"];
    let owners = d["owners"].as_array().map_or(&[][..], |a| a.as_slice());
    let _ = writeln!(o, "object collision in range: {} owner(s)", owners.len());
    for x in owners {
        let _ = writeln!(o, "  slot {} {} tris {} nearest {} vel {}", n(&x["slot"]), n(&x["bhv"]), n(&x["tris"]), n(&x["nearest"]), pt(&x["vel"]));
    }
    for t in d["nearest"].as_array().map_or(&[][..], |a| a.as_slice()).iter().take(3) {
        let _ = writeln!(o, "    slot {} {} {}", n(&t["slot"]), n(&t["list"]), tri_line(t));
    }
}

fn behavior(v: &Value, o: &mut String) {
    let _ = writeln!(o, "{} at {}  ({}:{}-{})", n(&v["name"]), n(&v["address"]), n(&v["file"]), n(&v["line"][0]), n(&v["line"][1]));
    let _ = writeln!(o, "{}", s(&v["script"]));
    for x in v["natives"].as_array().map_or(&[][..], |a| a.as_slice()) {
        let d = list(&x["defined"]);
        let _ = writeln!(o, "native {}: {}", n(&x["fn"]), if d.is_empty() { "definition not found".into() } else { d });
    }
    if !v["refs"].as_array().is_none_or(|a| a.is_empty()) {
        let _ = write!(o, "refs: {}", v["refs"].as_array().unwrap().iter().map(n).collect::<Vec<_>>().join("; "));
    }
}

fn level(v: &Value, o: &mut String) {
    let z = &v["zone"];
    if v["available"].as_bool() != Some(true) {
        let _ = write!(o, "no atlas data for {} area {}: {}", n(&z["level_name"]), n(&z["area"]), n(&v["reason"]));
        return;
    }
    let a = &v["atlas"];
    let _ = writeln!(o, "{} area {} act {}: atlas map {} variant {} ({})", n(&z["level_name"]), n(&z["area"]), n(&z["act"]), n(&a["map"]), n(&a["variant"]), n(&a["label"]));
    let c = v["category_counts"].as_object().map_or(String::new(), |m| m.iter().map(|(k, x)| format!("{k}:{}/{}", n(&x["this_act"]), n(&x["total"]))).collect::<Vec<_>>().join(" "));
    let _ = writeln!(o, "objects this act/total by category: {c}");
    for x in v["objects"].as_array().map_or(&[][..], |a| a.as_slice()) {
        let acts = if x["acts"].is_null() { String::new() } else { format!(" acts[{}]", list(&x["acts"])) };
        let note = if x["note"].is_null() { String::new() } else { format!("  # {}", n(&x["note"])) };
        let dist = if x["dist"].is_null() { String::new() } else { format!(" d{}", n(&x["dist"])) };
        let _ = writeln!(o, "  {:>3} {:<9} {:<30} {:>22}{dist} {}{acts}{note}", n(&x["i"]), n(&x["cat"]), n(&x["bhv"]), pt(&x["pos"]), if n(&x["toggle"]) == "always" { String::new() } else { format!("[{}]", n(&x["toggle"])) });
    }
    if let Some(g) = v["others"].as_object().filter(|g| !g.is_empty()) {
        let _ = writeln!(o, "other objects (use `level all` for positions):");
        for (cat, m) in g {
            let _ = writeln!(o, "  {cat}: {}", m.as_object().map_or(String::new(), |m| m.iter().map(|(b, c)| format!("{b} x{}", n(c))).collect::<Vec<_>>().join(", ")));
        }
    }
    let warps = v["warps"].as_array().map_or(&[][..], |a| a.as_slice());
    let _ = writeln!(o, "warps ({}):", warps.len());
    for w in warps {
        let _ = writeln!(o, "  id {:>3} {:<9} -> {} area {} node {}", n(&w["id"]), n(&w["kind"]), super::level_name(w["dest_level"].as_u64().unwrap_or(0) as u16), n(&w["dest_area"]), n(&w["dest_node"]));
    }
    let iw = v["instant_warps"].as_array().map_or(0, |a| a.len());
    let wt = v["water"].as_array().map_or(&[][..], |a| a.as_slice());
    let _ = write!(o, "instant warps: {iw}  water boxes: {}", wt.len());
    for b in wt {
        let _ = write!(o, "\n  water y {} x {}..{} z {}..{} {}", n(&b["y"]), n(&b["x1"]), n(&b["x2"]), n(&b["z1"]), n(&b["z2"]), n(&b["kind"]));
    }
}

fn objectives(v: &Value, o: &mut String) {
    let _ = writeln!(o, "frame {}: {} offered, {} shown{}", n(&v["frame"]), n(&v["total_offered"]), n(&v["shown"]), if v["kind_filter"].is_null() { String::new() } else { format!(" (kind ~ {})", n(&v["kind_filter"])) });
    let _ = writeln!(o, "{:>7} {:<26} {:<44} {:<5} {:<6} {:<8} {:<13} target", "eta", "kind", "key", "base", "exists", "prec", "ga pop/gen/hz");
    let rows = v["objectives"].as_array().map_or(&[][..], |a| a.as_slice());
    for x in rows.iter().take(40) {
        let g = &x["ga_budget"];
        let ga = if g.is_null() { "-".to_string() } else { format!("{}/{}/{}", n(&g["population"]), n(&g["generations"]), n(&g["horizon"])) };
        let key: String = s(&x["key"]).chars().take(44).collect();
        let _ = writeln!(o, "{:>7} {:<26} {:<44} {:<5} {:<6} {:<8} {:<13} {}{}", if x["lost"].as_bool() == Some(true) { "lost".into() } else { n(&x["eta"]) }, s(&x["kind"]), key, n(&x["base_criteria"]), n(&x["exists_in"]), s(&x["precision"]), ga, pt(&x["steer_target"]), if x["modal"].as_bool() == Some(true) { " modal" } else { "" });
    }
    if rows.len() > 40 {
        let _ = write!(o, "... {} more (filter with a kind substring, or use --json)", rows.len() - 40);
    }
}

fn diff(v: &Value, o: &mut String) {
    let f = &v["frames"];
    let _ = writeln!(o, "frame {} -> {} ({:+})", n(&f["from"]), n(&f["to"]), f["delta"].as_i64().unwrap_or(0));
    let ft = |k: &str, label: &str, o: &mut String| {
        if !v[k].is_null() {
            let _ = writeln!(o, "{label}: {} -> {}", n(&v[k]["from"]), n(&v[k]["to"]));
        }
    };
    ft("zone", "zone", o);
    let p = &v["pos"];
    let _ = writeln!(o, "pos {} -> {}  delta {}  dist {} (horizontal {})", pt(&p["from"]), pt(&p["to"]), pt(&p["delta"]), n(&p["dist"]), n(&p["hdist"]));
    for (k, l) in [("action", "action"), ("forward_vel", "fvel"), ("health", "health wedges"), ("coins", "coins"), ("stars", "stars"), ("keys", "keys"), ("lives", "lives"), ("red_coins", "red coins"), ("held", "held")] {
        ft(k, l, o);
    }
    if !v["cap"].is_null() {
        let _ = writeln!(o, "cap: [{}] -> [{}]", list(&v["cap"]["from"]), list(&v["cap"]["to"]));
    }
    if !v["save_flags"].is_null() {
        let _ = writeln!(o, "save flags set: [{}] cleared: [{}]", list(&v["save_flags"]["set"]), list(&v["save_flags"]["cleared"]));
    }
    for g in v["course_stars_gained"].as_array().map_or(&[][..], |a| a.as_slice()) {
        let _ = writeln!(o, "star gained: {} star(s) {}{}", n(&g["course"]), list(&g["stars"]), if g["cannon"].as_bool() == Some(true) { " + cannon" } else { "" });
    }
    let ob = &v["objects"];
    if !ob.is_null() {
        let _ = writeln!(o, "objects: {} -> {}", n(&ob["count"][0]), n(&ob["count"][1]));
        let group = |key: &str, label: &str, o: &mut String| {
            let a = ob[key].as_array().map_or(&[][..], |a| a.as_slice());
            if a.is_empty() {
                return;
            }
            let mut by: std::collections::BTreeMap<String, Vec<String>> = Default::default();
            for x in a {
                by.entry(s(&x["bhv"]).to_string()).or_default().push(format!("{}@{}", n(&x["slot"]), pt(&x["pos"])));
            }
            for (bhv, v) in by {
                let more = if v.len() > 4 { format!(" +{} more", v.len() - 4) } else { String::new() };
                let _ = writeln!(o, "{label} {bhv} x{}: {}{more}", v.len(), v[..v.len().min(4)].join(" "));
            }
        };
        group("spawned", "spawned  ", o);
        group("despawned", "despawned", o);
        let ch = ob["changed"].as_array().map_or(&[][..], |a| a.as_slice());
        for x in ch.iter().take(25) {
            let mut d = vec![];
            for (k, l) in [("action", "action"), ("health", "hp"), ("held_state", "held")] {
                if !x[k].is_null() {
                    d.push(format!("{l} {}->{}", n(&x[k][0]), n(&x[k][1])));
                }
            }
            if !x["moved"].is_null() {
                d.push(format!("moved {} to {}", n(&x["moved"]["dist"]), pt(&x["moved"]["to"])));
            }
            let _ = writeln!(o, "changed  {} slot {}: {}", n(&x["bhv"]), n(&x["slot"]), d.join(", "));
        }
        if ch.len() > 25 {
            let _ = writeln!(o, "... {} more changed", ch.len() - 25);
        }
    }
}
