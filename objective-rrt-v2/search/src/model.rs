//! The learned backbone (backbone/, sm64_AI_backbone.md) during a search, on
//! the GPU through backbone/serve_model.py (a Unix socket). Off unless
//! `OBRRT_MODEL` names the socket; then:
//!
//! - `OBRRT_MODEL_START_T=<frames>` (default 300; 0 = off): start states are
//!   weighted by `exp(-(score - min) / T)` in `pick_start`, where a state's
//!   score is the model's frames to the nearest star (or key) it hasn't
//!   collected, among the 2026 TAS's collection states (the server's goal
//!   bank). Attempt ends are scored in the worker while their game is live;
//!   states without a score of their own (dense adds, unless
//!   `OBRRT_MODEL_DENSE=1`) take their parent's.
//! - `OBRRT_MODEL_POLICY=<n>` (default 0): n extra first-generation GA
//!   rollouts from the goal-conditioned policy, for objectives with a steer
//!   target (the target position is the goal): every 32 frames the policy's
//!   stick and button summary is sampled and turned into inputs.
//! - `OBRRT_MODEL_REPAIR=<n>` (default 0): seeded attempts (polish) whose
//!   recorded route has model states get n extra rollouts built from
//!   16-frame inverse-dynamics chunks toward the route's states, for the
//!   GA to mix with the seed's own variants.
//!
//! - `OBRRT_MODEL_LEARNED=<n>` (default 0; v6 checkpoints): each scored state
//!   also gets n learned objectives sampled from the model (learned.rs: a
//!   style - reach, exit, event, speed, or a verb done to an object the
//!   model points at - with its target, budget and the star it is for),
//!   added to the pool when the state is expanded. `OBRRT_MODEL_LEARNED_MAX`
//!   (default 20000) caps how many states propose.
//!
//! The server answers one request at a time with batch 1, so a given input
//! always gets the same answer. `OBRRT_MODEL_SERVERS=<n>` (default 1) spreads
//! the worker threads over n server processes on sockets `<socket>.0` ..
//! `<socket>.<n-1>` (each thread always uses the same one).

use std::cell::RefCell;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result};
use obrrt_core::features::{self, ModelInput};
use obrrt_core::{GamePad, St};
use rand::Rng;
use sm64_binds::SM64Game;

/// A state's model data: its embedding and its frames to the nearest
/// uncollected goal.
#[derive(Debug)]
pub struct NodeModel {
    pub z: Box<[f32]>,
    pub score: f32,
    pub goal: i32,
    /// Learned objectives proposed from this state (OBRRT_MODEL_LEARNED).
    pub proposals: Vec<Arc<crate::learned::Learned>>,
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub socket: String,
    pub start_t: f64,
    pub dense: bool,
    pub policy: usize,
    pub repair: usize,
    pub learned: u32,
    pub learned_max: usize,
    pub d: usize,
    pub servers: usize,
}

/// The settings, or None when the model is off.
pub fn settings() -> Option<&'static Settings> {
    static S: OnceLock<Option<Settings>> = OnceLock::new();
    S.get_or_init(|| {
        let socket = std::env::var("OBRRT_MODEL").ok().filter(|s| !s.is_empty())?;
        let num = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
        let mut s = Settings {
            socket,
            start_t: num("OBRRT_MODEL_START_T", 300.0),
            dense: std::env::var("OBRRT_MODEL_DENSE").is_ok_and(|v| v == "1"),
            policy: num("OBRRT_MODEL_POLICY", 0.0) as usize,
            repair: num("OBRRT_MODEL_REPAIR", 0.0) as usize,
            learned: num("OBRRT_MODEL_LEARNED", 0.0) as u32,
            learned_max: num("OBRRT_MODEL_LEARNED_MAX", 20000.0) as usize,
            d: 256,
            servers: (num("OBRRT_MODEL_SERVERS", 1.0) as usize).max(1),
        };
        // Zone tables the model was trained with (backbone/data/*/zones).
        if let Ok(dirs) = std::env::var("OBRRT_MODEL_ZONES") {
            let dirs: Vec<&str> = dirs.split(':').collect();
            let n = features::preload_zones(&dirs);
            eprintln!("model: {n} zone tables preloaded");
        }
        match call(&s.socket, 4, &[]) {
            Ok(info) => {
                let v: serde_json::Value = serde_json::from_slice(&info).unwrap_or_default();
                s.d = v["d"].as_u64().unwrap_or(256) as usize;
                eprintln!("model: {} (checkpoint step {}, {} goals)", s.socket, v["step"], v["goals"]);
            }
            Err(e) => panic!("OBRRT_MODEL={}: server not reachable: {e:#}", s.socket),
        }
        Some(s)
    })
    .as_ref()
}

/// Server count, read straight from the environment (`call` runs while
/// `settings` is still being built, for the INFO request).
fn settings_servers() -> usize {
    std::env::var("OBRRT_MODEL_SERVERS").ok().and_then(|v| v.parse().ok()).unwrap_or(1usize).max(1)
}

thread_local! {
    static CONN: RefCell<Option<UnixStream>> = const { RefCell::new(None) };
}

fn call(socket: &str, op: u32, payload: &[u8]) -> Result<Vec<u8>> {
    CONN.with(|c| {
        let mut c = c.borrow_mut();
        if c.is_none() {
            let n = settings_servers();
            let path = if n > 1 {
                let i = rayon::current_thread_index().unwrap_or(0) % n;
                format!("{socket}.{i}")
            } else {
                socket.to_string()
            };
            *c = Some(UnixStream::connect(&path).with_context(|| format!("connecting to {path}"))?);
        }
        let s = c.as_mut().unwrap();
        let mut head = Vec::with_capacity(8 + payload.len());
        head.extend(op.to_le_bytes());
        head.extend((payload.len() as u32).to_le_bytes());
        head.extend_from_slice(payload);
        s.write_all(&head)?;
        let mut n = [0u8; 4];
        s.read_exact(&mut n)?;
        let mut out = vec![0u8; u32::from_le_bytes(n) as usize];
        s.read_exact(&mut out)?;
        Ok(out)
    })
}

fn input(game: &SM64Game, st: &St, goal: Option<[f32; 3]>, pad_prev: GamePad) -> ModelInput {
    game.with_memory(|m| features::live_input(m, st, goal, features::pack_pad(pad_prev)))
}

/// Scores the live state of `game` (`st` read from it with objects).
pub fn node(game: &SM64Game, st: &St, pad_prev: GamePad) -> Option<Arc<NodeModel>> {
    let s = settings()?;
    let mut b = features::input_bytes(&input(game, st, None, pad_prev));
    b.extend_from_slice(&st.course_stars);
    b.extend(st.save_flags.to_le_bytes());
    b.push(st.keys);
    let out = call(&s.socket, 1, &b).expect("model server: node");
    let f = |i: usize| f32::from_le_bytes(out[4 * i..4 * i + 4].try_into().unwrap());
    let z: Box<[f32]> = (0..s.d).map(f).collect();
    let score = f(s.d);
    let goal = i32::from_le_bytes(out[4 * s.d + 4..4 * s.d + 8].try_into().unwrap());
    let proposals = if s.learned > 0 && PROPOSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < s.learned_max {
        propose(&b, st, s.learned)
    } else {
        Vec::new()
    };
    Some(Arc::new(NodeModel { z, score, goal, proposals }))
}

static PROPOSED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// n learned objectives from the state whose NODE request is `node_req`
/// (model input + progress), seeded by the state itself.
fn propose(node_req: &[u8], st: &St, n: u32) -> Vec<Arc<crate::learned::Learned>> {
    let s = settings().unwrap();
    let mut b = node_req.to_vec();
    b.extend(n.to_le_bytes());
    let seed = (st.frame as u64) << 32 ^ (st.pos[0].to_bits() as u64) << 16 ^ st.pos[2].to_bits() as u64;
    b.extend(seed.to_le_bytes());
    let out = call(&s.socket, 5, &b).expect("model server: propose");
    out.chunks_exact(36).filter_map(|c| crate::learned::from_bytes(c, st)).map(Arc::new).collect()
}

/// `n` sampled stick / button summaries for the next 16 frames toward `goal`.
fn policy(game: &SM64Game, st: &St, pad_prev: GamePad, goal: [f32; 3], budget: f32, n: u32, seed: u64) -> Vec<[f32; 6]> {
    let s = settings().unwrap();
    let mut b = features::input_bytes(&input(game, st, None, pad_prev));
    for v in goal.into_iter().chain([budget]) {
        b.extend(v.to_le_bytes());
    }
    b.extend(n.to_le_bytes());
    b.extend(seed.to_le_bytes());
    let out = call(&s.socket, 2, &b).expect("model server: policy");
    out.chunks_exact(24).map(|c| std::array::from_fn(|k| f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap()))).collect()
}

/// Inputs for the next `k` frames toward the state with embedding `z`.
pub fn inverse(game: &SM64Game, st: &St, pad_prev: GamePad, z: &[f32], k: u32) -> Vec<GamePad> {
    let s = settings().unwrap();
    let mut b = features::input_bytes(&input(game, st, None, pad_prev));
    b.extend(z.iter().flat_map(|v| v.to_le_bytes()));
    b.extend(k.to_le_bytes());
    let out = call(&s.socket, 3, &b).expect("model server: inverse");
    out.chunks_exact(4).map(|c| features::unpack_pad(u32::from_le_bytes(c.try_into().unwrap()))).collect()
}

/// Turns a policy summary into `n` frames of inputs: the stick at the
/// summary's yaw relative to Mario's facing (camera-relative on the stick,
/// as the game reads it: intendedYaw = atan2s(-stickY, stickX) + camera yaw)
/// and magnitude; A, B and Z each held on a random share of frames equal to
/// their fraction.
fn theta_inputs(th: [f32; 6], st: &St, n: usize, rng: &mut impl Rng) -> Vec<GamePad> {
    use obrrt_core::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
    let rel = th[0].atan2(th[1]);
    let world = st.yaw as f32 * std::f32::consts::TAU / 65536.0 + rel;
    let cam = st.cam_yaw as f32 * std::f32::consts::TAU / 65536.0;
    let a = world - cam;
    let r = (th[2].clamp(0.0, 1.0) * 70.0).min(80.0);
    let (sx, sy) = ((r * a.sin()).round() as i8, (-r * a.cos()).round() as i8);
    (0..n)
        .map(|_| {
            let mut b = 0u16;
            for (frac, bit) in [(th[3], A_BUTTON), (th[4], B_BUTTON), (th[5], Z_TRIG)] {
                if rng.random::<f32>() < frac.clamp(0.0, 1.0) {
                    b |= bit;
                }
            }
            GamePad::new(b, sx, sy)
        })
        .collect()
}

/// A rollout genome from the policy toward `goal`, played on `game` from its
/// current state (which it changes): re-sampled every 32 frames.
pub fn policy_genome(game: &mut crate::game::Game, start: &St, goal: [f32; 3], horizon: usize, rng: &mut impl Rng) -> Result<Vec<GamePad>> {
    let mut out: Vec<GamePad> = Vec::with_capacity(horizon);
    let mut frame = start.frame;
    while out.len() < horizon {
        let st = obrrt_core::read_state(game, frame, true);
        let prev = out.last().copied().unwrap_or_default();
        let left = (horizon - out.len()) as f32;
        let th = policy(game, &st, prev, goal, left, 1, rng.random())[0];
        let chunk = theta_inputs(th, &st, 32.min(horizon - out.len()), rng);
        for p in &chunk {
            game.step_game(*p)?;
        }
        frame += chunk.len() as u32;
        out.extend(chunk);
    }
    Ok(out)
}

/// A genome toward a recorded route's states (`targets`: frames after the
/// start and their embeddings, in order), played on `game` from its current
/// state: 16-frame inverse-dynamics chunks toward the next target ahead.
pub fn repair_genome(game: &mut crate::game::Game, start: &St, targets: &[(u32, Arc<NodeModel>)], horizon: usize) -> Result<Vec<GamePad>> {
    let mut out: Vec<GamePad> = Vec::with_capacity(horizon);
    let mut frame = start.frame;
    for (at, m) in targets {
        while (out.len() as u32) < *at && out.len() < horizon {
            let st = obrrt_core::read_state(game, frame, true);
            let prev = out.last().copied().unwrap_or_default();
            let ahead = (*at - out.len() as u32).min(64);
            let pads = inverse(game, &st, prev, &m.z, ahead);
            let take = pads.len().min(16).min(horizon - out.len());
            for p in &pads[..take] {
                game.step_game(*p)?;
            }
            frame += take as u32;
            out.extend_from_slice(&pads[..take]);
        }
    }
    Ok(out)
}
