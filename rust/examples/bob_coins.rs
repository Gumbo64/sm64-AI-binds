//! From power-on, go straight into Bob-omb Battlefield and collect `coins`
//! coins, with every candidate played in the game.
//!
//! cargo run --release --example bob_coins -- <rom> [out=bob-coins.m64]
//!     [seed=1] [variant=pink] [minutes=30] [coins=50] [<coins.m64> <frame>]
//!
//! Phases:
//!
//! 1. Boot: the curiosity run's menu and intro inputs (`skip_menu_pads`),
//!    then waits until Mario stands still.
//! 2. Route, five searches: the grounds' centre line, the top of the slope
//!    past the moat's bridge, the castle door, the BoB room door and the BoB
//!    painting (GA, population 512, 150-frame genomes, 4 iterations per 10
//!    committed frames). Each is scored by reaching its goal (earlier is
//!    better), 2x closeness to its target at the end, 0.2x new 200-unit
//!    voxels, and -5000 for a death.
//! 3. Presses A through the act select, then waits for Mario to land in BoB
//!    and stand still.
//! 4. Coins. Given `<coins.m64> <frame>`, plays that file's inputs from
//!    `frame` (where its Mario had just landed in BoB) instead of searching.
//!    Otherwise searches with the curiosity run's settings: population 128,
//!    600-frame horizon, 1 frame committed per iteration after 10 warm-up
//!    iterations, and a per-frame reward discounted by 0.99 (best prefix): 30
//!    per coin + the curiosity novelty term; -5000 for a death.
//!
//! Writes the full input file, from power-on.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{Result, bail};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use sm64_ai::noise::{Slot, Variant, random_gamepad};
use sm64_ai::util::{read_gamepads_from_file, skip_menu_pads, write_gamepads_to_file};
use sm64_binds::buttons::A_BUTTON;
use sm64_binds::{GamePad, GameState, SM64Game, SM64GameGenerator};

const LEVEL_CASTLE: i32 = 6;
const LEVEL_BOB: i32 = 9;
const LEVEL_GROUNDS: i32 = 16;
/// The castle grounds' centre line (x = 0), which leads over the moat's
/// bridge: the route of TASVideos' 16-star TAS (6943M).
const CENTRE_LINE: [f32; 3] = [0.0, 700.0, 2950.0];
/// The top of the slope in front of the castle, past the bridge (where Mario
/// respawns after dying on the grounds).
const CASTLE_FRONT: [f32; 3] = [0.0, 873.0, -1710.0];
/// The castle's front door, from the grounds.
const CASTLE_DOOR: [f32; 3] = [-67.0, 803.0, -3418.0];
/// Just inside the BoB room's door, off the lobby; Mario is in the room once
/// x is below `BOB_ROOM_X`.
const BOB_ROOM: [f32; 3] = [-2100.0, 0.0, -650.0];
const BOB_ROOM_X: f32 = -2000.0;
/// Inside the BoB painting, over its warp floor.
const PAINTING: [f32; 3] = [-5290.0, 445.0, -278.0];
const GOAL_BONUS: f32 = 1.0e6;
const MUTATION: f64 = 0.05;
/// Below this, Mario has no health left.
const DEAD_HEALTH: i32 = 0x100;

type Genome = Vec<GamePad>;

/// One thread's games: a copy of the main game and one to play candidates on.
struct Worker {
    base: SM64Game,
    tmp: SM64Game,
}

/// The main game plus a worker per thread, kept on the same frame.
struct Games {
    main: SM64Game,
    workers: Vec<Worker>,
    played: Vec<GamePad>,
}

impl Games {
    fn step(&mut self, pad: GamePad) -> Result<GameState> {
        self.main.step_game(pad)?;
        self.workers
            .par_iter_mut()
            .map(|w| w.base.step_game(pad))
            .collect::<Result<Vec<_>, _>>()?;
        self.played.push(pad);
        Ok(self.main.get_game_state()?)
    }

    /// Every genome's states, played from the current frame.
    fn rollouts(&mut self, pop: &[Genome]) -> Result<Vec<Vec<GameState>>> {
        let chunk = pop.len().div_ceil(self.workers.len()).max(1);
        let mut out: Vec<(usize, Vec<GameState>)> = self
            .workers
            .par_iter_mut()
            .enumerate()
            .map(|(w, worker)| -> Result<Vec<(usize, Vec<GameState>)>> {
                let (lo, hi) = ((w * chunk).min(pop.len()), ((w + 1) * chunk).min(pop.len()));
                (lo..hi)
                    .map(|i| {
                        worker.tmp.sync_from(&worker.base)?;
                        let mut states = Vec::with_capacity(pop[i].len());
                        for pad in &pop[i] {
                            worker.tmp.step_game(*pad)?;
                            states.push(worker.tmp.get_game_state()?);
                        }
                        Ok((i, states))
                    })
                    .collect()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        out.sort_by_key(|(i, _)| *i);
        Ok(out.into_iter().map(|(_, s)| s).collect())
    }
}

/// The curiosity run's GA (`src/population.rs`), with seeded noise.
struct Ga {
    pop: Vec<Genome>,
    slots: Vec<Slot>,
    rng: SmallRng,
}

impl Ga {
    fn new(size: usize, len: usize, variant: Variant, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut slots: Vec<Slot> = (0..size).map(|_| Slot::new(&mut rng, variant)).collect();
        let pop = (0..size)
            .map(|i| (0..len).map(|_| slots[i].next(&mut rng)).collect())
            .collect();
        Ga { pop, slots, rng }
    }

    /// Breeds the next population; returns the best genome and its score.
    fn breed(&mut self, scores: &[f32]) -> (Genome, f32) {
        let (size, len) = (self.pop.len(), self.pop[0].len());
        let mut order: Vec<usize> = (0..size).collect();
        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let best = (self.pop[order[0]].clone(), scores[order[0]]);
        let elite = size / 5 + 1;
        let mut next: Vec<Genome> = order
            .iter()
            .take(elite)
            .map(|&i| self.pop[i].clone())
            .collect();
        let seeds: Vec<u64> = (0..size - elite).map(|_| self.rng.random()).collect();
        let pop = &self.pop;
        let offspring: Vec<Genome> = seeds
            .par_iter()
            .map(|&s| {
                let mut rng = SmallRng::seed_from_u64(s);
                let p1 = &pop[rng.random_range(0..size)];
                let p2 = &pop[rng.random_range(0..size)];
                (0..len)
                    .map(|j| {
                        let r: f64 = rng.random();
                        let mut gene = if r < 0.4 {
                            p1[j]
                        } else if r < 0.8 {
                            p2[j]
                        } else if r < 0.9 {
                            GamePad {
                                button: p2[j].button,
                                ..p1[j]
                            }
                        } else {
                            GamePad {
                                button: p1[j].button,
                                ..p2[j]
                            }
                        };
                        if rng.random::<f64>() < MUTATION {
                            gene = random_gamepad(&mut rng);
                        }
                        gene
                    })
                    .collect::<Genome>()
            })
            .collect();
        next.extend(offspring);
        self.pop = next;
        best
    }

    /// Drops the first `n` frames of every genome (they were played) and
    /// refills the tails.
    fn shift(&mut self, n: usize) {
        for (slot, g) in self.pop.iter_mut().enumerate() {
            g.drain(..n);
            for _ in 0..n {
                g.push(self.slots[slot].next(&mut self.rng));
            }
        }
    }
}

/// 1 for a death, 0.25 for ending with less health than at the start.
fn death(states: &[GameState], start: &GameState) -> f32 {
    if states
        .iter()
        .any(|s| s.health < DEAD_HEALTH || s.lives < start.lives)
    {
        1.0
    } else if states.last().is_some_and(|s| s.health < start.health) {
        0.25
    } else {
        0.0
    }
}

/// GOAL_BONUS - 10 per frame for the first frame satisfying `goal`.
fn goal(states: &[GameState], goal: impl Fn(&GameState) -> bool) -> Option<(usize, f32)> {
    states
        .iter()
        .position(goal)
        .map(|i| (i, GOAL_BONUS - 10.0 * i as f32))
}

type Voxel = (i32, i32, i32, i32);

fn voxel(s: &GameState, size: f32) -> Voxel {
    let v = |x: f32| (x / size).floor() as i32;
    (s.level_num, v(s.pos[0]), v(s.pos[1]), v(s.pos[2]))
}

/// sm64-AI-binds `CuriosityScoring`'s counts: 333-unit cubes, each visit also
/// counting for the 26 neighbours, and visits per level.
#[derive(Default)]
struct Curiosity {
    buckets: HashMap<Voxel, u32>,
    levels: HashMap<i32, u32>,
}

impl Curiosity {
    const STEP: f32 = 1000.0 / 3.0;
    const MIN: f32 = -8002.0;

    fn key(s: &GameState) -> Voxel {
        let d = |v: f32| ((v - Self::MIN) / Self::STEP).floor() as i32;
        (s.level_num, d(s.pos[0]), d(s.pos[1]), d(s.pos[2]))
    }

    fn add(buckets: &mut HashMap<Voxel, u32>, levels: &mut HashMap<i32, u32>, k: Voxel) {
        *levels.entry(k.0).or_insert(0) += 1;
        for i in -1..=1 {
            for j in -1..=1 {
                for l in -1..=1 {
                    *buckets.entry((k.0, k.1 + i, k.2 + j, k.3 + l)).or_insert(0) += 1;
                }
            }
        }
    }

    /// Discounted (0.99), best prefix: 30 per coin + the novelty term.
    fn coins_reward(&self, states: &[GameState], start: &GameState) -> f32 {
        let (mut lb, mut ll) = (HashMap::new(), HashMap::new());
        let (mut sum, mut discount, mut best) = (0.0f32, 1.0f32, f32::NEG_INFINITY);
        let mut coins = start.coins;
        for s in states {
            let k = Self::key(s);
            let visits =
                self.buckets.get(&k).copied().unwrap_or(0) + lb.get(&k).copied().unwrap_or(0);
            let level =
                self.levels.get(&k.0).copied().unwrap_or(0) + ll.get(&k.0).copied().unwrap_or(0);
            let novelty = 1.0 / (1.0 + visits as f32).sqrt() + 1.0 / (1.0 + level as f32).sqrt();
            let r = 30.0 * (s.coins - coins).max(0) as f32 + novelty;
            coins = s.coins;
            Self::add(&mut lb, &mut ll, k);
            sum += r * discount;
            discount *= 0.99;
            best = best.max(sum);
        }
        best
    }
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// One leg of the route: search until a candidate satisfies `done`, then play
/// it up to that frame.
struct Leg {
    name: &'static str,
    done: fn(&GameState) -> bool,
    target: [f32; 3],
}

const ROUTE: [Leg; 5] = [
    Leg {
        name: "centre line",
        done: |s| s.level_num == LEVEL_GROUNDS && s.pos[0].abs() < 300.0 && s.pos[2] < 3000.0,
        target: CENTRE_LINE,
    },
    Leg {
        name: "castle front",
        done: |s| s.level_num == LEVEL_GROUNDS && s.pos[2] < -1500.0 && s.pos[1] > 700.0,
        target: CASTLE_FRONT,
    },
    Leg {
        name: "castle door",
        done: |s| s.level_num == LEVEL_CASTLE,
        target: CASTLE_DOOR,
    },
    Leg {
        name: "BoB room",
        done: |s| s.level_num == LEVEL_CASTLE && s.pos[0] < BOB_ROOM_X,
        target: BOB_ROOM,
    },
    Leg {
        name: "painting",
        done: |s| s.level_num == LEVEL_BOB,
        target: PAINTING,
    },
];

/// Plays `leg`'s search; returns the iterations it took.
fn reach(
    games: &mut Games,
    st: &mut GameState,
    leg: &Leg,
    variant: Variant,
    seed: u64,
) -> Result<usize> {
    let t0 = Instant::now();
    let mut ga = Ga::new(512, 150, variant, seed);
    let mut archive: HashSet<Voxel> = HashSet::new();
    let mut iteration = 0;
    loop {
        let start_state = st.clone();
        let rollouts = games.rollouts(&ga.pop)?;
        let scores: Vec<f32> = rollouts
            .par_iter()
            .map(|s| {
                let reached = goal(s, leg.done).map_or(-GOAL_BONUS / 2.0, |(_, b)| b);
                let mut seen = HashSet::new();
                let novel = s.iter().filter(|g| {
                    !archive.contains(&voxel(g, 200.0)) && seen.insert(voxel(g, 200.0))
                });
                reached - 2.0 * distance(s.last().unwrap().pos, leg.target)
                    + 0.2 * novel.count() as f32
                    - 5000.0 * death(s, &start_state)
            })
            .collect();
        let best_rollout = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap();
        if let Some((at, _)) = goal(&rollouts[best_rollout], leg.done) {
            // Every candidate ran in the game: play the winner up to the goal.
            for &pad in &ga.pop[best_rollout][..=at] {
                *st = games.step(pad)?;
            }
            return Ok(iteration);
        }
        ga.breed(&scores);
        iteration += 1;
        if iteration % 4 == 0 {
            let commit: Vec<GamePad> = ga.pop[0][..10].to_vec();
            for pad in commit {
                *st = games.step(pad)?;
                archive.insert(voxel(st, 200.0));
            }
            ga.shift(10);
        }
        if iteration % 40 == 0 {
            println!(
                "    {}: {iteration} iterations, frame {} level {} pos {:?}",
                leg.name,
                games.played.len(),
                st.level_num,
                st.pos
            );
        }
        if t0.elapsed().as_secs_f64() > 300.0 {
            bail!("didn't reach the {} in 300 s", leg.name);
        }
    }
}

/// Waits with no input until Mario has stood still for 10 frames.
fn settle(games: &mut Games, st: &mut GameState) -> Result<()> {
    let mut still = 0;
    for _ in 0..3000 {
        if still >= 10 {
            return Ok(());
        }
        let before = st.pos;
        *st = games.step(GamePad::new(0, 0, 0))?;
        still = if st.pos == before { still + 1 } else { 0 };
    }
    bail!("Mario didn't stand still")
}

/// Presses A every `every` frames (no input between) until `done`.
fn press_a_until(
    games: &mut Games,
    st: &mut GameState,
    every: usize,
    done: impl Fn(&GameState) -> bool,
) -> Result<()> {
    let mut i = 0;
    while !done(st) {
        *st = games.step(if i % every == 0 {
            GamePad::new(A_BUTTON, 0, 0)
        } else {
            GamePad::new(0, 0, 0)
        })?;
        i += 1;
        if i > 5000 {
            bail!(
                "stuck pressing A at frame {}: level {} pos {:?}",
                games.played.len(),
                st.level_num,
                st.pos
            );
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        bail!(
            "usage: bob_coins <rom> [out] [seed] [variant] [minutes] [coins] [<coins.m64> <frame>]"
        );
    }
    let out = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "bob-coins.m64".into());
    let seed: u64 = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(1);
    let name = args.get(3).map(String::as_str).unwrap_or("pink");
    let Some(variant) = Variant::named(name) else {
        bail!("unknown variant {name}")
    };
    let minutes: f64 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(30.0);
    let target: i32 = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(50);
    let coin_inputs = match (args.get(6), args.get(7)) {
        (Some(file), Some(frame)) => {
            let pads = read_gamepads_from_file(file)?;
            let frame: usize = frame.parse()?;
            if frame > pads.len() {
                bail!("{file} has only {} frames", pads.len());
            }
            Some(pads[frame..].to_vec())
        }
        (None, None) => None,
        _ => bail!("give both <coins.m64> and <frame>"),
    };
    let t0 = Instant::now();
    let deadline = minutes * 60.0;

    let generator = SM64GameGenerator::from_file(&args[0])?;
    let threads = rayon::current_num_threads();
    let workers = (0..threads)
        .into_par_iter()
        .map(|_| -> Result<Worker> {
            Ok(Worker {
                base: generator.create_game()?,
                tmp: generator.create_game()?,
            })
        })
        .collect::<Result<_>>()?;
    let mut games = Games {
        main: generator.create_game()?,
        workers,
        played: Vec::new(),
    };
    let mut st = games.main.get_game_state()?;

    // 1. Boot and the intro.
    for pad in skip_menu_pads() {
        st = games.step(pad)?;
    }
    settle(&mut games, &mut st)?;
    println!(
        "[1] controllable at frame {}: pos {:?} ({} backend, {:.0}s)",
        games.played.len(),
        st.pos,
        games.main.backend(),
        t0.elapsed().as_secs_f64()
    );

    // 2. The route.
    for (i, leg) in ROUTE.iter().enumerate() {
        let iterations = reach(&mut games, &mut st, leg, variant, seed + i as u64)?;
        println!(
            "[2] {} at frame {}: level {} pos {:?}, {iterations} iterations ({:.0}s)",
            leg.name,
            games.played.len(),
            st.level_num,
            st.pos,
            t0.elapsed().as_secs_f64()
        );
    }

    // 3. Act select and the level entry: A every 4th frame until in BoB (the
    // level number changes as the painting warp starts, so also away from the
    // castle's painting), then wait until Mario stands still.
    press_a_until(&mut games, &mut st, 4, |s| {
        s.level_num == LEVEL_BOB && distance(s.pos, PAINTING) >= 2000.0
    })?;
    settle(&mut games, &mut st)?;
    let landed = games.played.len();
    println!(
        "[3] in BoB at frame {landed}: coins {} pos {:?} ({:.0}s)",
        st.coins,
        st.pos,
        t0.elapsed().as_secs_f64()
    );

    // 4. Coins.
    if let Some(pads) = coin_inputs {
        for pad in pads {
            st = games.step(pad)?;
        }
        println!(
            "[4] played {} given frames: coins {} level {} lives {} health {:#x}",
            games.played.len() - landed,
            st.coins,
            st.level_num,
            st.lives,
            st.health
        );
    } else {
        let mut ga = Ga::new(128, 600, variant, seed + ROUTE.len() as u64);
        let mut curiosity = Curiosity::default();
        let (mut iterations, mut committed) = (0usize, 0usize);
        while st.coins < target && t0.elapsed().as_secs_f64() < deadline {
            let start_state = st.clone();
            let rollouts = games.rollouts(&ga.pop)?;
            let scores: Vec<f32> = rollouts
                .par_iter()
                .map(|s| {
                    let reached = goal(s, |g| g.coins >= target).map_or(0.0, |(_, b)| b);
                    reached + curiosity.coins_reward(s, &start_state)
                        - 5000.0 * death(s, &start_state)
                })
                .collect();
            ga.breed(&scores);
            iterations += 1;
            if iterations <= 10 {
                continue;
            }
            st = games.step(ga.pop[0][0])?;
            let k = Curiosity::key(&st);
            Curiosity::add(&mut curiosity.buckets, &mut curiosity.levels, k);
            ga.shift(1);
            committed += 1;
            if committed % 600 == 0 {
                println!(
                    "    {committed} frames: coins {} level {} ({:.1} fr/s)",
                    st.coins,
                    st.level_num,
                    committed as f64 / t0.elapsed().as_secs_f64()
                );
            }
        }
        println!(
            "[4] coins {} after {committed} frames in BoB, {iterations} iterations ({:.0}s)",
            st.coins,
            t0.elapsed().as_secs_f64()
        );
    }
    write_gamepads_to_file(&games.played, &out)?;
    println!(
        "wrote {out} ({} frames; in BoB from frame {landed})",
        games.played.len()
    );
    Ok(())
}
