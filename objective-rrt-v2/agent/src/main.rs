//! `ar`: agent-driven routes. See agent/README.md (commands) and AGENT.md
//! (how an agent should drive it).

mod ckpt;
mod commands;
mod explore;
mod inspect;
mod objective;
mod popjournal;
mod selftest;
mod serve;
mod session;
mod util;
mod viewer;

use anyhow::{Context, Result, anyhow, bail};
use obrrt_core::St;
use serde_json::{Value, json};
use session::{Config, Session};
use std::collections::HashMap;
use util::*;

const USAGE: &str = "\
ar — agent-driven SM64 routes (objective-rrt-v2/agent; README.md, AGENT.md)
Session: --s NAME, or $AR_SESSION, or the last `ar new`/`ar use` (agent/sessions/.current).

  new NAME [--start JSON] [--threads N] [--window W] [--set key=value]...
  use NAME                      make NAME the current session
  serve [--fg]                  start the solver daemon (background unless --fg)
  stop                          stop it (checkpoint, flush)
  status [--config]             daemon, position, front objective, speed
  brief                         one LLM-sized summary: status, queue, events, state, objects, offers
  push SEL | --lambda NAME|JSON [--args JSON] | --native KIND [--key S] [--near x,y,z] [--within R] | --goto x,y,z --zone Z [--r R] [--air]
       [--note S] [--rh JSON] [--stall N] [--at POS] [--no-check]
  queue                         the queue and every objective's status
  scrap ID [--note S] | move ID POS | edit ID [--sel JSON] [--rh JSON] [--note S] [--stall N]
  pause [--reason S] | resume | set KEY VALUE   (e.g. set rh.g 4, set window 2, set stall_frames 900)
  rewind TARGET [--drop] [--fresh-pop] [--note S]   TARGET: cNNNN, cNNNN:F, fF, -N
  cut [--note S]                commit the work in progress now
  inject FILE.m64...            add input plans to the live population
  wait [--events a,b] [--since SEQ] [--timeout S]   block until such an event (default: the important ones)
  note TEXT [--by NAME]         a comment in the session's history (events; `ar events`, `ar brief`)
  events [--since SEQ] [--n N]  log [--n N]  branches
  export OUT.m64 [--at TARGET]  inputs from power-on (viewer/probe-ready)
  viewer                        write the viewer data now (the daemon does it every 5 s)
  pop [stats|show|seed] [--seg cNNNN] [--record I] [--at TARGET]   the population journal
  Inspect (--at TARGET, default the path's end; --json):
  state | progress | objects [--near R] [--bhv S] | geo [--r R] [--point x,y,z] | behavior NAME | level [--all]
  objectives [--kind S] | eta SEL | diff TARGET_A TARGET_B | trace --from F --to F [--every N] [--eta ID]
  lib list | lib show NAME | lib add NAME FILE.json [--note S] [--parent NAME] | lib retire NAME [--reason S]
                                lambda-objective templates (agent/lib; retired ones stay on disk)
  selftest [--sel JSON] [--steps N] [--threads T] [--samples K]   checkpoints, determinism, journal, rewind
  Exploration tools (optional; Claude decides):
  fork N SEL [--frames F] [--rh JSON]   N independent controllers from the end on SEL; reports frames, keeps branches
  explore [--minutes M] [--threads T]   a bounded obrrt2 tree from the end; report + `explore import RUN NODE`
";

struct Args {
    pos: Vec<String>,
    flags: HashMap<String, String>,
    multi: HashMap<String, Vec<String>>,
}

impl Args {
    fn parse(v: Vec<String>) -> Args {
        let (mut pos, mut flags, mut multi) = (Vec::new(), HashMap::new(), HashMap::<String, Vec<String>>::new());
        let mut i = 0;
        const BOOL: &[&str] = &["json", "fg", "config", "drop", "fresh-pop", "no-check", "all", "air"];
        while i < v.len() {
            if let Some(k) = v[i].strip_prefix("--") {
                if BOOL.contains(&k) || i + 1 >= v.len() {
                    flags.insert(k.to_string(), "1".into());
                } else {
                    flags.insert(k.to_string(), v[i + 1].clone());
                    multi.entry(k.to_string()).or_default().push(v[i + 1].clone());
                    i += 1;
                }
            } else {
                pos.push(v[i].clone());
            }
            i += 1;
        }
        Args { pos, flags, multi }
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.flags.get(k).map(|s| s.as_str())
    }
    fn has(&self, k: &str) -> bool {
        self.flags.contains_key(k)
    }
    fn num<T: std::str::FromStr>(&self, k: &str) -> Result<Option<T>> {
        self.get(k).map(|s| s.parse::<T>().map_err(|_| anyhow!("--{k}: not a number: {s}"))).transpose()
    }
    fn pos(&self, i: usize, what: &str) -> Result<&str> {
        self.pos.get(i).map(|s| s.as_str()).ok_or_else(|| anyhow!("missing {what}\n\n{USAGE}"))
    }
}

fn current_file() -> String {
    format!("{ROOT}/agent/sessions/.current")
}

fn session_name(a: &Args) -> Result<String> {
    if let Some(s) = a.get("s") {
        return Ok(s.into());
    }
    if let Ok(s) = std::env::var("AR_SESSION") {
        return Ok(s);
    }
    std::fs::read_to_string(current_file()).map(|s| s.trim().to_string()).map_err(|_| anyhow!("no session: pass --s NAME (or `ar use NAME`)"))
}

fn parse_json(s: &str, what: &str) -> Result<Value> {
    serde_json::from_str(s).with_context(|| format!("{what}: not JSON: {s}"))
}

fn vec3(s: &str) -> Result<Value> {
    let v: Vec<f64> = s.split(',').map(|x| x.trim().parse::<f64>()).collect::<Result<_, _>>().map_err(|_| anyhow!("expected x,y,z: {s}"))?;
    anyhow::ensure!(v.len() == 3, "expected x,y,z: {s}");
    Ok(json!(v))
}

fn print(v: &Value, a: &Args, kind: &str) {
    if a.has("json") || kind.is_empty() {
        println!("{}", serde_json::to_string_pretty(v).unwrap());
    } else {
        println!("{}", inspect::render(kind, v));
    }
}

/// Installs the domain set-up obrrt2 and legs use (objectives' controller,
/// move library, techniques), with legs' defaults for unset switches.
fn setup_domain() -> Result<()> {
    const DEFAULTS: &[(&str, &str)] = &[
        ("OBRRT_MOVES", "1"),
        ("OBRRT_CALM", "1"),
        ("OBRRT_LAKITU_ARC", "1"),
        ("OBRRT_PAINTING_FAST", "1"),
        ("OBRRT_GP_CATCH", "1"),
        ("OBRRT_SWIM_HOLD", "1"),
        ("OBRRT_TECH", "all"),
    ];
    for (k, v) in DEFAULTS {
        if std::env::var(k).is_err() {
            // SAFETY: single-threaded at start-up, before any thread reads the environment.
            unsafe { std::env::set_var(k, v) };
        }
    }
    obrrt_objectives::setup::from_env().map_err(anyhow::Error::msg)?;
    Ok(())
}

/// Sends a command to the live daemon and waits for its ack, or applies it
/// directly when none runs.
fn command(s: &mut Session, cmd: Value) -> Result<Value> {
    if s.serve_pid().is_none() {
        let c = cmd["cmd"].as_str().unwrap_or("");
        return match c {
            "rewind" => {
                let l = commands::rewind(s, cmd["target"].as_str().unwrap_or(""), cmd["drop"].as_bool().unwrap_or(false), cmd["note"].as_str().unwrap_or(""))?;
                Ok(json!({"head": l.head, "frame": l.frame, "note": "no daemon: applied to the files; `ar serve` continues from there"}))
            }
            "cut" => {
                let c = s.cut("manual", None, None, cmd["note"].as_str().unwrap_or(""))?;
                Ok(json!({"commit": c.id}))
            }
            "pause" | "resume" | "stop" | "inject" | "pop_seed" | "fresh_pop" => bail!("{c}: no daemon is running (`ar serve`)"),
            _ => commands::apply(s, &cmd),
        };
    }
    let id = s.send(&cmd)?;
    let t0 = std::time::Instant::now();
    loop {
        for e in s.events()?.into_iter().rev().take(200) {
            if e["ev"] == "ack" && e["inbox"] == id.as_str() {
                if let Some(err) = e["error"].as_str() {
                    bail!("{err}");
                }
                return Ok(e["result"].clone());
            }
        }
        if t0.elapsed().as_secs() > 120 {
            bail!("the daemon did not acknowledge within 120 s (is it busy or paused in a long step?)");
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// `--at`: (path head, frame). Default: the end of HEAD's path (with the work in progress).
fn target(s: &Session, at: Option<&str>) -> Result<(String, Option<u32>)> {
    let head = s.head()?;
    let Some(t) = at else { return Ok((head, None)) };
    let (w, wp) = s.wip_pads()?;
    let end = w.from + wp.len() as u32;
    if let Some(n) = t.strip_prefix('-') {
        return Ok((head, Some(end.saturating_sub(n.parse()?))));
    }
    if let Some(f) = t.strip_prefix('f') {
        return Ok((head, Some(f.parse()?)));
    }
    let m = s.commit_map()?;
    let (id, f) = match t.split_once(':') {
        Some((id, f)) => (id, Some(f.parse::<u32>()?)),
        None => (t, None),
    };
    let c = m.get(id).ok_or_else(|| anyhow!("--at {t}: no commit {id} (targets: cNNNN, cNNNN:F, fF, -N)"))?;
    Ok((c.id.clone(), Some(f.unwrap_or(c.to))))
}

struct Loaded {
    game: sm64_binds::SM64Game,
    st: St,
}

fn load(s: &Session, at: Option<&str>) -> Result<Loaded> {
    let gg = sm64_binds::SM64GameGenerator::from_file(&s.cfg.rom)?;
    let mut game = gg.create_game()?;
    let mut ck = ckpt::Store::open(&s.dir)?;
    let (head, frame) = target(s, at)?;
    let st = s.load_at(&mut ck, &mut game, &head, frame)?;
    Ok(Loaded { game, st })
}

fn selector(a: &Args, from: usize) -> Result<Value> {
    if let Some(l) = a.get("lambda") {
        let t = if l.trim_start().starts_with('{') { parse_json(l, "--lambda")? } else { json!(l) };
        let mut v = json!({"lambda": t});
        if let Some(x) = a.get("args") {
            v["args"] = parse_json(x, "--args")?;
        }
        return Ok(v);
    }
    if let Some(k) = a.get("native") {
        let mut v = json!({"native": k});
        if let Some(x) = a.get("key") {
            v["key"] = json!(x);
        }
        if let Some(x) = a.get("near") {
            v["near"] = vec3(x)?;
        }
        if let Some(x) = a.num::<f64>("within")? {
            v["within"] = json!(x);
        }
        return Ok(v);
    }
    if let Some(g) = a.get("goto") {
        let mut v = json!({"goto": vec3(g)?, "zone": a.get("zone").ok_or_else(|| anyhow!("--goto needs --zone (e.g. bob, grounds, castle:1; castle is inside the castle, level 6, and grounds the castle grounds, level 16)"))?});
        if let Some(r) = a.num::<f64>("r")? {
            v["r"] = json!(r);
        }
        if a.has("air") {
            v["grounded"] = json!(false);
        }
        return Ok(v);
    }
    let s = a.pos(from, "an objective selector (JSON), or --native / --goto")?;
    parse_json(s, "selector")
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ar: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv[0] == "help" || argv[0] == "--help" || argv[0] == "-h" {
        print!("{USAGE}");
        return Ok(());
    }
    let cmd = argv.remove(0);
    let a = Args::parse(argv);
    match cmd.as_str() {
        "new" => {
            let name = a.pos(0, "a session name")?;
            let mut cfg = Config::default();
            if let Some(s) = a.get("start") {
                cfg.start = parse_json(s, "--start")?;
            }
            if let Some(t) = a.num("threads")? {
                cfg.threads = t;
            }
            if let Some(w) = a.num("window")? {
                cfg.window = w;
            }
            let mut v = serde_json::to_value(&cfg)?;
            for kv in a.multi.get("set").into_iter().flatten() {
                let (k, x) = kv.split_once('=').ok_or_else(|| anyhow!("--set key=value"))?;
                let slot = v.pointer_mut(&format!("/{}", k.replace('.', "/"))).ok_or_else(|| anyhow!("--set: no setting {k}"))?;
                *slot = serde_json::from_str(x).unwrap_or(json!(x));
            }
            let cfg: Config = serde_json::from_value(v)?;
            if let Some(why) = session::below_floor(&cfg.rh) {
                bail!("--set: {why}");
            }
            let gg = sm64_binds::SM64GameGenerator::from_file(&cfg.rom)?;
            let s = Session::create(name, cfg, &gg)?;
            std::fs::write(current_file(), format!("{}\n", s.dir.display()))?;
            println!("session {} at {} ({}); current", s.cfg.name, s.dir.display(), s.cfg.start_desc);
            return Ok(());
        }
        "use" => {
            let name = a.pos(0, "a session name")?;
            let s = Session::open(name)?;
            std::fs::write(current_file(), format!("{}\n", s.dir.display()))?;
            println!("current session: {}", s.dir.display());
            return Ok(());
        }
        "lib" => return lib(&a),
        "behavior" => {
            let v = inspect::behavior(a.pos(0, "a behaviour name")?)?;
            print(&v, &a, "behavior");
            return Ok(());
        }
        _ => {}
    }
    let name = session_name(&a)?;
    let mut s = Session::open(&name)?;
    match cmd.as_str() {
        "serve" => {
            if let Some(pid) = s.serve_pid() {
                bail!("already serving (pid {pid})");
            }
            if a.has("fg") {
                setup_domain()?;
                return serve::serve(&name);
            }
            let exe = std::env::current_exe()?;
            let log = std::fs::OpenOptions::new().create(true).append(true).open(s.dir.join("serve.log"))?;
            let child = std::process::Command::new("nice")
                .args(["-n", "10"])
                .arg(exe)
                .args(["serve", "--fg", "--s", &s.dir.display().to_string()])
                .stdin(std::process::Stdio::null())
                .stdout(log.try_clone()?)
                .stderr(log)
                .spawn()?;
            for _ in 0..100 {
                std::thread::sleep(std::time::Duration::from_millis(100));
                if s.serve_pid().is_some() {
                    break;
                }
            }
            println!("serving {} (pid {}); log {}/serve.log", s.cfg.name, s.serve_pid().unwrap_or(child.id()), s.dir.display());
        }
        "stop" => {
            let pid = s.serve_pid().ok_or_else(|| anyhow!("no daemon"))?;
            command(&mut s, json!({"cmd": "stop"}))?;
            for _ in 0..600 {
                if !pid_alive(pid) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            println!("stopped");
        }
        "status" => {
            if a.has("config") {
                println!("{}", serde_json::to_string_pretty(&serde_json::to_value(&s.cfg)?)?);
                return Ok(());
            }
            let v = status_value(&s)?;
            print(&v, &a, "");
        }
        "brief" => {
            setup_domain()?;
            println!("{}", brief(&s, &a)?);
        }
        "push" => {
            let sel = selector(&a, 0)?;
            let mut c = json!({"cmd": "push", "sel": sel});
            if let Some(n) = a.get("note") {
                c["note"] = json!(n);
            }
            if let Some(r) = a.get("rh") {
                c["rh"] = parse_json(r, "--rh")?;
            }
            if let Some(n) = a.num::<u32>("stall")? {
                c["stall_frames"] = json!(n);
            }
            if let Some(p) = a.num::<usize>("at")? {
                c["pos"] = json!(p);
            }
            let r = command(&mut s, c)?;
            println!("queued {} at {} (queue {})", r["id"].as_str().unwrap_or("?"), r["pos"], r["queue"]);
            if !a.has("no-check") {
                setup_domain()?;
                let l = load(&s, None)?;
                match objective::resolve(&sel, r["id"].as_str().unwrap_or("check"), &l.st, &[]) {
                    Ok(Some(o)) => println!("  at the path's end (frame {}): {} key {:?}, start ETA {:.0}", l.st.frame, o.kind(), o.key(), obrrt_core::start_eta(o.as_ref(), &l.st)),
                    Ok(None) => println!("  not offered at the path's end (frame {}) yet; it resolves when it reaches the front if offered then (`ar objectives` lists what is)", l.st.frame),
                    Err(e) => println!("  error: {e:#}"),
                }
            }
        }
        "queue" => {
            let q = s.queue()?;
            let objs = s.objectives()?;
            println!("queue (front first): {}", if q.is_empty() { "empty".to_string() } else { q.join(" ") });
            for o in &objs {
                let pos = q.iter().position(|x| x == &o.id).map_or("  ".to_string(), |p| format!("{p:>2}"));
                println!("{pos} {} {:<9} {} {}{}", o.id, o.status, o.sel, if o.note.is_empty() { String::new() } else { format!("— {}", o.note) }, o.met_commit.as_ref().map_or(String::new(), |c| format!(" [{c}]")));
            }
        }
        "scrap" | "move" | "edit" => {
            let id = a.pos(0, "an objective id")?;
            let mut c = json!({"cmd": cmd, "id": id, "note": a.get("note")});
            if cmd == "move" {
                c["pos"] = json!(a.pos(1, "a position")?.parse::<usize>()?);
            }
            if let Some(x) = a.get("sel") {
                c["sel"] = parse_json(x, "--sel")?;
            }
            if let Some(x) = a.get("rh") {
                c["rh"] = parse_json(x, "--rh")?;
            }
            if let Some(n) = a.num::<u32>("stall")? {
                c["stall_frames"] = json!(n);
            }
            println!("{}", command(&mut s, c)?);
        }
        "pause" => println!("{}", command(&mut s, json!({"cmd": "pause", "reason": a.get("reason")}))?),
        "resume" => println!("{}", command(&mut s, json!({"cmd": "resume"}))?),
        "set" => {
            let k = a.pos(0, "a setting")?;
            let v = a.pos(1, "a value")?;
            let v = serde_json::from_str(v).unwrap_or(json!(v));
            println!("{}", command(&mut s, json!({"cmd": "set", "key": k, "value": v}))?);
        }
        "rewind" => {
            let t = a.pos(0, "a target (cNNNN, cNNNN:F, fF, -N)")?;
            let r = command(&mut s, json!({"cmd": "rewind", "target": t, "drop": a.has("drop"), "fresh_pop": a.has("fresh-pop"), "note": a.get("note").unwrap_or("")}))?;
            println!("{r}");
        }
        "cut" => println!("{}", command(&mut s, json!({"cmd": "cut", "note": a.get("note").unwrap_or("")}))?),
        "inject" => {
            let files: Vec<String> = a.pos.iter().map(|f| resolve(f)).collect();
            println!("{}", command(&mut s, json!({"cmd": "inject", "files": files}))?);
        }
        "wait" => wait(&s, &a)?,
        "note" => {
            let text = a.pos.join(" ");
            anyhow::ensure!(!text.trim().is_empty(), "note TEXT");
            s.event("note", json!({"text": text, "by": a.get("by").unwrap_or("agent")}))?;
            println!("noted");
        }
        "events" => {
            let since = a.num::<u64>("since")?.unwrap_or(0);
            let ev: Vec<Value> = s.events()?.into_iter().filter(|e| e["seq"].as_u64().unwrap_or(0) > since).collect();
            let n = a.num::<usize>("n")?.unwrap_or(30);
            for e in ev.iter().skip(ev.len().saturating_sub(n)) {
                println!("{}", event_line(e));
            }
        }
        "log" => log(&s, &a)?,
        "branches" => branches(&s)?,
        "export" => {
            let out = a.pos(0, "an output .m64")?;
            let (head, frame) = target(&s, a.get("at"))?;
            let mut pads = s.path_pads(&head, true)?;
            if let Some(f) = frame {
                pads.truncate(f as usize);
            }
            write_m64(out, &pads)?;
            println!("{out}: {} frames from power-on (raw, 4 bytes a frame)", pads.len());
        }
        "pop" => pop(&s, &a)?,
        "revoke-ckpt" => {
            // Repair: withdraw a segment's checkpoints after a frame.
            let owner = a.pos(0, "a segment id")?;
            let after: u32 = a.pos(1, "a frame")?.parse()?;
            let mut ck = ckpt::Store::open(&s.dir)?;
            let n = ck.revoke(owner, after, a.get("why").unwrap_or(""))?;
            println!("revoked {n} checkpoint(s) of {owner} after frame {after}");
        }
        "viewer" => {
            let gg = sm64_binds::SM64GameGenerator::from_file(&s.cfg.rom)?;
            let mut ck = ckpt::Store::open(&s.dir)?;
            viewer::sync(&s, &gg, &mut ck, u64::MAX)?;
            println!("viewer data written: {}/viewer.json, paths/", s.dir.display());
        }
        "state" | "progress" | "objects" | "geo" | "level" | "objectives" | "eta" => {
            if cmd == "objectives" || cmd == "eta" {
                setup_domain()?;
            }
            let l = load(&s, a.get("at"))?;
            let v = inspect::View { game: &l.game, st: &l.st };
            let out = match cmd.as_str() {
                "state" => inspect::state(&v),
                "progress" => inspect::progress(&l.st),
                "objects" => inspect::objects(&v, a.num("near")?, a.get("bhv")),
                "geo" => inspect::geo(&v, a.num("r")?.unwrap_or(1500.0), a.get("point").map(vec3).transpose()?.map(|p| [p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32, p[2].as_f64().unwrap() as f32])),
                "level" => inspect::level(&l.st, a.has("all"))?,
                "objectives" => inspect::objectives(&l.st, a.get("kind")),
                _ => {
                    let sel = selector(&a, 0)?;
                    match objective::resolve(&sel, "eta", &l.st, &[])? {
                        Some(o) => objective::describe(o.as_ref(), &l.st),
                        None => json!({"resolved": false, "frame": l.st.frame, "hint": "not offered here; see `ar objectives`"}),
                    }
                }
            };
            print(&out, &a, if cmd == "eta" { "" } else { &cmd });
        }
        "diff" => {
            let x = load(&s, Some(a.pos(0, "target A")?))?;
            let y = load(&s, Some(a.pos(1, "target B")?))?;
            let v = inspect::diff(&inspect::View { game: &x.game, st: &x.st }, &inspect::View { game: &y.game, st: &y.st });
            print(&v, &a, "diff");
        }
        "trace" => trace(&s, &a)?,
        "selftest" => {
            setup_domain()?;
            let sel = a.get("sel").map(|x| parse_json(x, "--sel")).transpose()?;
            selftest::run(&s, sel, a.num("steps")?.unwrap_or(30), a.num("threads")?.unwrap_or(4), a.num("samples")?.unwrap_or(20))?;
        }
        "fork" => {
            setup_domain()?;
            explore::fork(&s, &a.pos, &a.flags)?;
        }
        "explore" => explore::explore(&mut s, &a.pos, &a.flags)?,
        _ => bail!("unknown command {cmd}\n\n{USAGE}"),
    }
    Ok(())
}

fn status_value(s: &Session) -> Result<Value> {
    let mut v = read_json(s.dir.join("status.json")).unwrap_or(json!({}));
    let alive = s.serve_pid();
    v["daemon"] = json!(alive.map_or("not running".to_string(), |p| format!("running (pid {p})")));
    if alive.is_none() {
        let (w, wp) = s.wip_pads()?;
        v["head"] = json!(s.head()?);
        v["end_frame"] = json!(w.from + wp.len() as u32);
    }
    v["queue"] = json!(s.queue()?);
    v["last_seq"] = json!(s.events()?.last().map_or(0, |e| e["seq"].as_u64().unwrap_or(0)));
    Ok(v)
}

const IMPORTANT: &[&str] = &["met", "committed", "stalled", "window_low", "window_empty", "waiting", "unresolved", "stuck", "lost", "failed", "error", "crashed", "stopped"];

fn event_line(e: &Value) -> String {
    let mut rest = e.clone();
    let m = rest.as_object_mut().unwrap();
    let seq = m.remove("seq").unwrap_or_default();
    let ev = m.remove("ev").unwrap_or_default();
    m.remove("t");
    format!("#{seq} {} {}", ev.as_str().unwrap_or("?"), serde_json::to_string(&rest).unwrap())
}

fn wait(s: &Session, a: &Args) -> Result<()> {
    let want: Vec<String> = a.get("events").map_or(IMPORTANT.iter().map(|x| x.to_string()).collect(), |e| e.split(',').map(Into::into).collect());
    let since = match a.num::<u64>("since")? {
        Some(x) => x,
        None => s.events()?.last().map_or(0, |e| e["seq"].as_u64().unwrap_or(0)),
    };
    let timeout = a.num::<f64>("timeout")?.unwrap_or(3600.0);
    let t0 = std::time::Instant::now();
    loop {
        let hits: Vec<Value> = s.events()?.into_iter().filter(|e| e["seq"].as_u64().unwrap_or(0) > since && want.iter().any(|w| e["ev"] == w.as_str())).collect();
        if !hits.is_empty() {
            for e in &hits {
                println!("{}", event_line(e));
            }
            return Ok(());
        }
        if s.serve_pid().is_none() {
            println!("no daemon running (last seq {since})");
            std::process::exit(3);
        }
        if t0.elapsed().as_secs_f64() > timeout {
            println!("timeout after {timeout} s (no {} since #{since})", want.join("/"));
            std::process::exit(2);
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}

fn log(s: &Session, a: &Args) -> Result<()> {
    let head = s.head()?;
    let objs: HashMap<String, session::Objective> = s.objectives()?.into_iter().map(|o| (o.id.clone(), o)).collect();
    let chain = s.chain(&head)?;
    let n = a.num::<usize>("n")?.unwrap_or(40);
    println!("HEAD {head}: {} commits, frame {}", chain.len(), chain.last().map_or(0, |c| c.to));
    for c in chain.iter().skip(chain.len().saturating_sub(n)) {
        let o = c.objective.as_ref().and_then(|o| objs.get(o));
        let what = o.map_or(String::new(), |o| format!("{} {}", o.id, o.sel));
        let sm = &c.summary;
        println!(
            "{} {:<9} f{:>6}..{:<6} (+{:>5}) {} | lvl {} area {} pos {} stars {} {}",
            c.id, c.kind, c.from, c.to, c.to - c.from, what, sm["level"], sm["area"], sm["pos"], sm["stars"], if c.note.is_empty() { String::new() } else { format!("— {}", c.note) }
        );
    }
    let (w, wp) = s.wip_pads()?;
    println!("work in progress {}: f{}..{} (+{})", w.id, w.from, w.from + wp.len() as u32, wp.len());
    Ok(())
}

fn branches(s: &Session) -> Result<()> {
    let commits = s.commits()?;
    let parents: std::collections::HashSet<String> = commits.iter().filter_map(|c| c.parent.clone()).collect();
    let head = s.head()?;
    let path = s.on_path(&head)?;
    println!("tips (commits nothing builds on); * on HEAD's path");
    for c in commits.iter().filter(|c| !parents.contains(&c.id)) {
        println!("{} {} {:<9} f{}..{} stars {} {}", if path.contains(&c.id) { "*" } else { " " }, c.id, c.kind, c.from, c.to, c.summary["stars"], c.note);
    }
    Ok(())
}

fn trace(s: &Session, a: &Args) -> Result<()> {
    let (head, _) = target(s, None)?;
    let pads = s.path_pads(&head, true)?;
    let from: u32 = a.num("from")?.ok_or_else(|| anyhow!("--from F"))?;
    let to: u32 = a.num("to")?.unwrap_or(pads.len() as u32).min(pads.len() as u32);
    let every: u32 = a.num("every")?.unwrap_or(1).max(1);
    let mut l = load(s, Some(&format!("f{from}")))?;
    let obj = match a.get("eta") {
        Some(id) => {
            setup_domain()?;
            let o = s.objectives()?.into_iter().find(|o| o.id == id).ok_or_else(|| anyhow!("no objective {id}"))?;
            objective::resolve(&o.sel, id, &l.st, &[])?
        }
        None => None,
    };
    let start = l.st.clone();
    let mut mem = obrrt_core::Mem::default();
    for f in from..to {
        let st = obrrt_core::read_state(&l.game, f, obj.as_ref().is_some_and(|o| o.needs_objects()));
        if (f - from) % every == 0 {
            let p = pads[f as usize];
            let eta = obj.as_ref().map(|o| format!(" eta {:.0}", o.eta(&st, &start, &mut mem.clone()).frames()));
            println!("{} | pad {:04x} {:>4} {:>4}{}", inspect::summary_line(&st), p.button, p.stick_x, p.stick_y, eta.unwrap_or_default());
        }
        if let Some(o) = &obj {
            let _ = o.eta(&st, &start, &mut mem);
        }
        l.game.step_game(pads[f as usize])?;
    }
    Ok(())
}

fn pop(s: &Session, a: &Args) -> Result<()> {
    let sub = a.pos.first().map(|x| x.as_str()).unwrap_or("stats");
    let seg = match a.get("seg") {
        Some(x) => x.to_string(),
        None => s.wip()?.id,
    };
    let p = s.journal_path(&seg);
    if !p.exists() {
        bail!("no population journal for {seg} ({})", p.display());
    }
    let heads = popjournal::index(&p)?;
    let i = match (a.num::<usize>("record")?, a.get("at")) {
        (Some(i), _) => i,
        (None, Some(t)) => {
            let f: u32 = t.trim_start_matches('f').parse()?;
            popjournal::find_frame(&heads, f).ok_or_else(|| anyhow!("no record at frame {f} in {seg}"))?
        }
        _ => heads.len().saturating_sub(1),
    };
    match sub {
        "stats" => {
            let size = std::fs::metadata(&p)?.len();
            println!("{seg}: {} records ({} keyframes), {} bytes, frames {}..{}, generations {}..{}", heads.len(), heads.iter().filter(|h| h.kind == 0).count(), size, heads.first().map_or(0, |h| h.frame), heads.last().map_or(0, |h| h.frame), heads.first().map_or(0, |h| h.generation), heads.last().map_or(0, |h| h.generation));
        }
        "show" => {
            let pop = popjournal::read_at(&p, &heads, i)?;
            let h = heads[i];
            println!("{seg} record {i}: generation {} from frame {}, {} plans (best first)", h.generation, h.frame, pop.len());
            let mut idx: Vec<usize> = (0..pop.len()).collect();
            idx.sort_by(|&x, &y| pop[y].fitness.total_cmp(&pop[x].fitness));
            for &j in idx.iter().take(a.num("n")?.unwrap_or(16)) {
                let q = &pop[j];
                let ends = ["horizon", "done", "died", "lost", "cut"];
                println!(
                    "  #{j:<3} fit {:>8.1} at {:>3} {:<8} done@{:<4} eta {:>6.0} {:<10} gen {:<5} first pads {}",
                    q.fitness,
                    q.at,
                    ends.get(q.end as usize).unwrap_or(&"-"),
                    q.first_done,
                    q.final_eta,
                    obrrt_search::ga::rhstep::origin_name(q.origin),
                    q.generation,
                    q.pads.iter().take(6).map(|p| format!("{:04x}/{}/{}", p.button, p.stick_x, p.stick_y)).collect::<Vec<_>>().join(" ")
                );
            }
        }
        "seed" => {
            let mut s2 = Session::open(&s.dir.display().to_string())?;
            println!("{}", command(&mut s2, json!({"cmd": "pop_seed", "segment": seg, "record": i}))?);
        }
        _ => bail!("pop stats|show|seed"),
    }
    Ok(())
}

fn brief(s: &Session, a: &Args) -> Result<String> {
    let mut out = String::new();
    let st_v = status_value(s)?;
    out += &format!("== session {} — daemon {}\n", s.cfg.name, st_v["daemon"].as_str().unwrap_or("?"));
    if let Some(p) = st_v["paused"].as_str() {
        out += &format!("PAUSED: {p}\n");
    }
    if let Some(f) = st_v.get("front").filter(|f| !f.is_null()) {
        out += &format!("front {} {}: eta {:.0}, best {:.0}, {} frames on it, {} since best\n", f["id"].as_str().unwrap_or(""), f["key"].as_str().unwrap_or(""), f["eta"].as_f64().unwrap_or(f64::NAN), f["best"].as_f64().unwrap_or(f64::NAN), f["frames_on_it"], f["since_best"]);
    }
    if let Some(sp) = st_v.get("speed").filter(|x| !x.is_null()) {
        out += &format!("speed: {:.2} frontier frames/s, {:.0} emulated frames/s; generation {}\n", sp["frontier_fps"].as_f64().unwrap_or(0.0), sp["emulated_fps"].as_f64().unwrap_or(0.0), st_v["generation"]);
    }
    if let Some(b) = st_v.get("base_frame").filter(|x| !x.is_null()) {
        out += &format!("window: base (last frozen) frame {b}, frontier {}, {} frames behind may still change; tentatively met {}\n", st_v["frontier_frame"], st_v["behind"], st_v["tentatively_met"]);
    }
    let q = s.queue()?;
    let objs = s.objectives()?;
    out += "\n== queue (front first)\n";
    for id in &q {
        if let Some(o) = objs.iter().find(|o| &o.id == id) {
            out += &format!("  {} {} {}\n", o.id, o.sel, o.note);
        }
    }
    if q.is_empty() {
        out += "  (empty: push objectives)\n";
    }
    out += "\n== recent events\n";
    let ev = s.events()?;
    for e in ev.iter().filter(|e| e["ev"] != "ack" && e["ev"] != "resolved").rev().take(a.num("events")?.unwrap_or(8)).collect::<Vec<_>>().into_iter().rev() {
        let mut l = event_line(e);
        l.truncate(300);
        out += &format!("  {l}\n");
    }
    out += "\n== recent commits\n";
    let chain = s.chain(&s.head()?)?;
    for c in chain.iter().skip(chain.len().saturating_sub(5)) {
        out += &format!("  {} {:<9} f{}..{} (+{}) {}\n", c.id, c.kind, c.from, c.to, c.to - c.from, c.objective.clone().unwrap_or_default());
    }
    let l = load(s, a.get("at"))?;
    let v = inspect::View { game: &l.game, st: &l.st };
    out += &format!("\n== state (frame {})\n{}\n", l.st.frame, inspect::render("state", &inspect::state(&v)));
    out += &format!("\n== progress\n{}\n", inspect::render("progress", &inspect::progress(&l.st)));
    let r: f32 = a.num("near")?.unwrap_or(2500.0);
    out += &format!("\n== objects within {r}\n{}\n", inspect::render("objects", &inspect::objects(&v, Some(r), None)));
    let offers = inspect::objectives(&l.st, None);
    let mut text = inspect::render("objectives", &offers);
    let lines: Vec<&str> = text.lines().take(a.num("offers")?.unwrap_or(30)).collect();
    text = lines.join("\n");
    out += &format!("\n== objectives offered here (by start ETA)\n{text}\n");
    Ok(out)
}

/// The lambda-objective template library (agent/lib/NAME.json): templates
/// are kept forever; `retire` only flags them.
fn lib(a: &Args) -> Result<()> {
    let dir = objective::lib_dir();
    std::fs::create_dir_all(&dir)?;
    let sub = a.pos(0, "list | show NAME | add NAME FILE | retire NAME")?;
    let path = |n: &str| format!("{dir}/{n}.json");
    match sub {
        "list" => {
            let mut names: Vec<_> = std::fs::read_dir(&dir)?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
            names.sort();
            for p in names {
                let v = read_json(&p)?;
                let st = v["status"].as_str().unwrap_or("active");
                if st == "retired" && !a.has("all") {
                    continue;
                }
                println!("{:<24} {:<8} {}", p.file_stem().unwrap().to_string_lossy(), st, v["note"].as_str().unwrap_or(""));
            }
        }
        "show" => println!("{}", serde_json::to_string_pretty(&read_json(path(a.pos(1, "NAME")?))?)?),
        "add" => {
            let name = a.pos(1, "NAME")?;
            anyhow::ensure!(name.chars().all(|c| c.is_alphanumeric() || "_-.".contains(c)), "a name of letters, digits, _ - .");
            anyhow::ensure!(!std::path::Path::new(&path(name)).exists(), "{name} exists: add a revision under a new name with --parent {name}");
            let t = read_json(resolve(a.pos(2, "FILE.json")?))?;
            let sel = json!({"lambda": t.clone()});
            objective::validate(&sel)?;
            let v = json!({"template": t, "note": a.get("note").unwrap_or(""), "parent": a.get("parent"), "status": "active", "created": now()});
            write_json(path(name), &v)?;
            println!("added {name}; use it with `ar push --lambda {name} [--args JSON]`");
        }
        "retire" => {
            let name = a.pos(1, "NAME")?;
            let mut v = read_json(path(name))?;
            v["status"] = json!("retired");
            v["retired_reason"] = json!(a.get("reason").unwrap_or(""));
            write_json(path(name), &v)?;
            println!("retired {name} (kept on disk; `lib list --all` shows it)");
        }
        _ => bail!("lib list | show | add | retire"),
    }
    Ok(())
}
