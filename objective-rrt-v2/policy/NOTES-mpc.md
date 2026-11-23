# mpc: derivative-free stick refinement (notes)

Code: `src/mpc.rs` (optimisers, scorers, `Refiner`), `src/bin/mpc_check.rs` (experiments; auto-discovered bin).
Raw outputs: `runs/mpc/*.txt`. Run: `mpc_check <rom> curve|smooth|loop|info <problem> ...` from the objective-rrt-v2 root.

## What it is
A plan is x in [-1,1]^D: `knots` piecewise-linear knots over the horizon, each an angle offset from a reference direction
(`Aim`: target / face / velocity / world yaw, up to `ang_max` rad) and a tilt (`mag_lo..mag_hi`, `moves::arcs::stick`'s scale).
Buttons are fixed by the caller (`BtnFn`). Score = `Scorer` over the rolled-out states (`Reach`, `HoldSpeed`, `LandAt` with a forbidden disc,
`TaskFit` = the gp rollout's discounted-ETA fitness). `optimize(algo, ev, x0, budget, rng)` always evaluates x0 first (never worse than it).
`Refiner::decide` is the receding-horizon wrapper (re-plan every k frames, warm start = previous plan time-shifted, `replay` between
re-plans costs no emulator frames). Result -> `Decision { fine_ang (-1..1), fine_mag (0..1), plan }` = `types::Choice`'s fine part;
`Spec::stick_for(st, fine_ang, fine_mag)` makes the stick.

## Cost (one thread)
step_game 14.4 us (69k frames/s), load_state 251 us, save_state 458 us, read_state ~0.
An evaluation of a 24-frame plan (load + 24 steps + state reads + scorer) measured about 1.8 ms end to end: 256 evals in 0.45 s (570 evals/s per thread).
So 32 evals x 24 frames ~ 55 ms per decision.
Closed loop over 48 frames: replan every frame 18-37k emulator frames (2-3 s); every 4 frames 5-10k (0.5-0.9 s).

## Open-loop comparison (`curve`): 8 starts x 3-4 seeds, mean gain over x0 (aim, mid tilt) after N evals; knots 3, horizon 24, problem `speed`
Starts are states along the case movies (fv 35-95), the movie's own buttons held.

| kind (unit) | random | cem | es | cma | spsa | fd | grid |
|---|---|---|---|---|---|---|---|
| Steer to a point (distance, 64 evals) | 123 | 118 | **146** | 142 | 122 | 60 | 110 |
| Hold speed (u/f, 64 evals) | **19.9** | 19.0 | 14.7 | 16.0 | 6.7 | 2.0 | 15.8 |
| Land beside, not on, natural landing (64 evals) | 518 | 429 | 526 | **527** | 478 | 305 | **563** |
| Task fitness (ETA frames, 64 evals) | 4.2 | **4.8** | 4.2 | 4.2 | 3.4 | 3.2 | 3.0 |

At 256 evals Steer: es 151, cma 149, random 140, cem 126, spsa 137, grid 122, fd 61. Same ranking on `lakitu_chain` (cma/random/grid best, spsa/fd poor on Hold and Task).
Differences among random/es/cma/cem are within ~10-15% with this sample (not significant); fd and spsa are clearly worse.
Knots: Steer needs several knots at horizon 24 (k=1: +71, k=3: +149, k=4 at horizon 48: +379); Hold/Land/Task barely care (k=1 is as good).
Most of the gain comes early: 16-32 evals reach ~70-85% of the 256-eval gain (random shooting included).
Against the movie's own stick (a TAS): Hold/Task stay 2-8% below it even at 256 evals (a 3-knot plan cannot match frame-perfect sticks); Steer/Land are
tasks the movie's stick never solved, so refinement wins big there.

## Smoothness (`smooth`): where gradients work
Random-direction steps of 0.03 / 0.1 plan units from x0 (40 dirs x 8 starts): fraction flat (|d|<1e-4 scale) / usable-smooth / cliff (>5% scale)

| kind | eps .03 flat/smooth/cliff | eps .1 flat/smooth/cliff |
|---|---|---|
| Steer | .04 / .96 / 0 | .02 / .88 / .10 |
| Hold speed | **.72** / .28 / 0 | .18 / .71 / .11 |
| Land | .17 / .83 / 0 | .08 / .65 / .27 |
| Task | .04 / .96 / 0 | .01 / .90 / .09 |

Steer and Task surfaces are smooth at small scale, so finite differences get a gradient there; but Hold is mostly *flat* (speed does not depend on the stick in
many ground/slide states), where FD/SPSA see zero gradient and stop (fd used 1/3 to 1/10 of its budget: it terminates when the step shrinks to nothing).
Cliffs are real: scans show a jump of 25-120 score units at **tilt 0.398 (raw stick magnitude 48: jump kick vs speed-kick dive)** in 6 of 16 scans, and
angle cliffs where the plan changes which action happens (wall hit, landing/forbidden disc, rollout cancel), often +-1000 for Land (landing vs not).
=> Gradients are valid only inside one contact/class; expect cliffs at stick-class boundaries.

## Closed loop (`loop`, receding horizon, 48 frames, 6 starts, movie buttons): mean value, emulator frames/run
| controller | Steer (dist, higher better) | Hold speed | Task | emu frames/run |
|---|---|---|---|---|
| fixed aim, mid tilt | -1992 | 86.3 | 18.4 | 0 |
| movie's stick | -2634 (target is off its path) | **119.8** | **32.0** | 0 |
| cma h12 replan 1 | **-1800** | 103.5 | 29.5 | 18k |
| cma h12 replan 4 | -1869 | 91.7 | 22.8 | 4.6k |
| cma h24 replan 1 | -1929 | 99.6 | 26.5 | 37k |
| cma h24 replan 4 | -1952 | 111.6 | 28.6 | 9k |
| cma h24 replan 8 (48 evals) | -1953 | 108.4 | 29.6 | 6.9k |
| cma h48 replan 8 (48 evals) | -1881 | 107.1 | 30.5 | 13.8k |
| cem h24 r4 | -1977 | **111.8** | 27.7 | 10k |
| es h24 r4 | -1915 | 98.1 | 28.7 | 10k |
| random h24 r4 | -1888 | 108.8 | 27.3 | 10k |
| grid h24 r4 | -1979 | 106.2 | **30.7** | 10k |
| fd h24 r4 | -1922 | 88.3 | 19.6 | 10k |
| spsa h24 r4 | -2107 | 90.4 | 19.7 | 10k |

Replanning every frame is not better than every 4-8 (Hold, Task) and costs 4-8x; precision steering to a point is the exception (short horizon, replan 1-2 helps).

## Recommendation
* Optimiser: **(1+lambda)-ES or separable CMA** (lambda 6-8) for steering/landing; **random shooting** (or CEM) is a strong cheap default and best on the flat Hold surface;
  grid(7-9 angles x 3-5 tilts, constant plan) + pattern search is the best at tiny budgets for D=2 (Land 563 at 64 evals). Do not use FD/SPSA as the main method
  (flat plateaus and cliffs); an FD/SPSA polish is only sensible inside a stick class on a Steer/Task-type smooth surface, from an ES/CMA result.
* Budget: 32-64 evals per decision; most of the value arrives by 32.
* Horizon/knots: **horizon 24, 3 knots, replan every 4 frames** as default (about 5-10k emulator frames per 48 frames of play, 0.5-0.9 s); for landing/steering at a point use horizon 12-24
  with replan every 1-2 frames; long (48) horizons need 4 knots and replan 8.
* Search the stick **per class**: never let one plan cross the tilt 0.398 boundary (48 magnitude): give each class (`Short`: tilt <= 0.39, `Full`: >= 0.40) its own range
  (`mag_lo/mag_hi`), and treat the class as the discrete choice made by the policy/GA. The angle is continuous inside a class, cliffs only come from scorer-visible events.
* Cost: MPC at every frame inside a GA fitness evaluation is too expensive (a 300-frame play would cost ~100-300k emulator frames). Use it (a) for final polish of the
  best policies, (b) at a few decision points (landing, takeoff, slide exit), (c) as a teacher: record MPC decisions and distil them into the policy's fine parameters.
