# Papers → sm64-AI-binds: proposals (2026-11-21)

Papers read (from `/workspaces/sm64-combined/papers`; two files there are duplicates):

| Paper | One line |
|---|---|
| **SuperMimic** (2610.28480) | One learned, differentiable world model (WM), fit on the policy's own sim data; the policy is trained by backpropagation through time (BPTT) over 32 imagined steps plus a critic. It matched a 21k-GPU-hour PPO tracker in 3 GPU-hours |
| Code as Policies (2209.07753) | An LLM writes policy code against named APIs. Hierarchical generation of undefined helpers scores 95 vs 81 flat. It has no parameter tuning from feedback |
| MAESTRO (2511.00917) | A VLM coding agent over a curated tool library. The harness and the tools only help together (33% → 77%). Cross-trial code-plus-root-cause refinement |
| CaP-X (2603.22435) | Benchmark for coding agents. A low-level API plus multi-turn matches a high-level API. A text diff of what changed beats raw images. A 9-sample ensemble plus a synthesis pass adds 7–9 points. GRPO on simulator success takes a task from 25% to 80% |
| ASPIRE (2607.00272) | A growing skill library: failure signature, guard, fix and provenance, admitted only when a fresh verifier reproduces it. Dense per-primitive traces take success from 14% to 62%. Library size predicts zero-shot success (4.7% → 30.5%) |
| RAX (rax.pdf) | Claude explores by hand, then compiles a staged program that must pass a rehearsal on a fresh scene with no agent help. Per-stage outcome tables and a review model. 4 parallel agents score 92.2 vs 63.3 for 1 |

Where we are (checked in code): objective-rrt-v2 objectives are `eta(s, start, mem) -> Done|Frames|Lost` (`core/src/lib.rs:702`). GA fitness is the max over t of Σγ^t(eta_{t−1} − eta_t), with γ=0.99, +300 on Done and −600 on death or Lost (`search/src/ga.rs`). The rolling-horizon (RH) GA gold standard is in `search/src/ga/rhstep.rs`. The policy-list genome is `policy/src/plga.rs`, and the inventory holds 482 entries (95 active). The backbone (`backbone/sm64bb/model.py`) has a token encoder plus jump-dynamics heads (k ∈ {4,16,64} from z(s)), an inverse model, events, an IQE quasimetric, reachability and policy heads. It is served to Rust over a Unix socket, not through ONNX. Learned gains so far are within noise.

---

## 1. A SuperMimic-style world model for SM64

### 1.1 What SuperMimic actually does, and what carries over

| SuperMimic ingredient | SM64 counterpart | Carries over? |
|---|---|---|
| One WM shared across 129k motions, because the body is the same | One WM shared across all zones, because Mario's physics is the same; the zone varies only as *context* | **Yes.** This is the core bet: Mario's movement code is one function, and only the terrain and objects vary |
| Deterministic WM; known randomness is passed in as sim context ξ | The emulator is fully deterministic. ξ = zone, local geometry, nearby object state, camera yaw, RNG index | **Yes, and easier** (there is no sensor noise) |
| Fit on k=8 *autoregressive* windows, so long rollouts stay stable | The same. Our current dyn heads are one-shot k-jumps from z(s), which is not a per-step model you can roll out or differentiate step by step | **New work**: it is a different model from today's backbone heads |
| Replay buffer of the current policy's transitions with action noise ε | **The `ar` population journal**: every generation of 128 plans × 600 frames around the trajectory we care about, which is near-miss data by construction. The attempt archive and tree nodes add more | **Yes, and better.** We already log the "noise around the policy" data they had to generate |
| BPTT through the frozen WM over H=32, plus a critic for the return beyond H | BPTT over H≈32–64 frames, with **the critic = learned ETA** (the IQE time-distance head, trained on hindsight frames-to-goal) | **Yes.** Our γ=0.99 ETA-difference fitness is already exactly their return (§3.1) |
| Soft survival weight c_t from a predicted termination probability | Termination = death or `Lost`. The WM predicts P(death) and P(zone left), so the −600 penalty becomes differentiable | **Yes** |
| Return-based reset sampling (50% uniform / 40% softmax on low return / 10% history) | Start states: tree nodes, funnels and `ar` commits, oversampling those where the GA does worst | **Yes.** This is the rate sampler's idea applied to *training* starts |
| Goal interfaces (motion tracking, velocity) with their own encoders over one shared decoder | Goal interfaces = `goto`, an ETA objective (kind embedding plus parameters), and **tracking a reference state sequence** | **Yes** (§1.4) |
| Ablation: first-order gradients need 64× fewer imagined steps than PPO on the same WM | This is the main reason to do it. Our emulator is already fast, so a speed gain alone will not matter | Must be measured |

**Their stated limitation is our whole problem.** SuperMimic trains on flat ground only, with "no terrain or object interaction". SM64 is terrain and objects, plus a ~230-state action machine (jump, long jump, wall kick, ledge grab) with hard mode switches, quarter-step collision and s16 angles. So the WM must be built around those discontinuities, not just made bigger.

### 1.2 Proposed WM: Mario-centric, per-frame, geometry by lookup

The design principle is to let the network learn only what is hard, and to look up or compute everything that is exact.

- **State ŝ_t** (about 40–60 dims). This is not the full token set:
  - pos, vel, forward_vel, yaw (sin/cos), action-ID distribution (a softmax over about 230 IDs, embedded as p·E, so it stays differentiable), action_timer, the action's flag bits;
  - wall/double-jump timers, frames since A/B, peak height, held state, cap timer, health.
  - All of these are already read by `core/src/features.rs`.
- **Action a_t** in the *game's processed* form: intended magnitude and **world-frame intended yaw** (from `intendedYaw = atan2s(-y, x) + camYaw`, mario.c:1309), plus A/B/Z as Bernoulli logits with a "fresh press" bit.
  - The WM is then camera-invariant. Raw int8 stick values are recovered only at export, and the emulator re-simulates them exactly anyway.
- **Geometry by differentiable lookup, not by attention over triangles.** sm64-atlas already has exact ports of `find_floor`, `find_ceil`, `find_wall_collisions` and `find_water_level`, partitioned into the game's 16×16 cells.
  - Bake each zone into dense per-cell fields: floor height, floor normal, surface class, ceiling height, a distance-to-wall field with its gradient, and water level.
  - The WM samples these fields at its *predicted* position by bilinear interpolation, which is differentiable in position. Terrain therefore enters as a smooth function of ŝ.
  - This is the piece SuperMimic did not have.
- **Objects as context ξ** (the nearest 8–16 non-cosmetic objects: relative position, velocity, behaviour embedding, hitbox), predicted forward by a small per-object head. Start with geometry-only zones, as the twin work did, and add moving platforms next.
- **Network:** a residual MLP (SuperMimic used 3×2048) predicting Δŝ, the action-ID logits and event logits (death, damage, ledge grab, wall-kick window, landing, zone change, coin or star touched).
  - Optionally conditioned on a z(zone) taken once from the existing encoder.
- **Loss:** SuperMimic's Eq. 1 over k=8 autoregressive windows (raise it to 16–32 later), with Huber loss on the scaled Δ, cross-entropy on the action ID and binary cross-entropy (BCE) on events.

### 1.3 Data

- **Primary: population journals** (`agent/sessions/*/pop/*.pj`). They are exact, about 300 KB per game second, and dense around good trajectories.
- **Also:**
  - the attempt archive (`backbone/attempts`);
  - the tree datasets (`backbone/data/v7`, 1.3 GB);
  - pink-noise and random-policy rollouts from tree nodes, for coverage and to give the model counterfactuals.
- **TAS:** fine for *training* the model, which is physics knowledge. It is never used as inputs in a result, per the own-120-star rule.
- **Collection:** we cannot store every frame of a 500k fps emulator, so the rule is to keep states, re-simulate on demand, and stream windows. The `ar` checkpoints and journal replay already make this exact.

### 1.4 What we use the WM for, in order of risk

1. **Gradient immigrants into the RH GA (lowest risk).**
   - Every few generations, take the elite plan and run 5–20 Adam steps on its stick sequence through the WM, against the differentiable objective (§3) over H frames with the ETA critic at the end.
   - Insert the result as one population member. The emulator scores it exactly as usual, so a bad gradient costs one slot out of 128.
   - The GA keeps the discrete structure (when to press A, which mode). The gradient refines continuous yaw and magnitude *within* the mode. This keeps "the GA stays the controller".
2. **Gradient tuning of policy-list lambda args.** This keeps the genome a policy list.
   - Relax the `.pol` DSL during the backward pass only: `(if (> F F) …)` becomes a sigmoid blend, `cell` angle and magnitude classes become soft, and `p0..p7` get gradients.
   - Execution is always hard and exact. This gives the GA a Lamarckian/memetic operator, "refine the args of segment i", and never flattens to pads.
3. **SuperMimic proper: one amortized goal-conditioned controller** π(ŝ, goal), trained by BPTT (H=32 plus the critic) through the frozen WM, in a Dyna loop: collect, refit the WM, train the policy.
   - Use it as the **mean of the gen-0 / refill distribution** (policy rollout plus pink noise instead of pure pink), and as one more `Gene` kind in the policy-list genome.
   - The twin project found that approximate proposers are fine when the real game verifies everything. This is the same role, learned.
   - Goal interfaces, as in SuperMimic App. B, each with its own encoder over a shared decoder:
     - (a) `goto` point plus grounded flag;
     - (b) an ETA objective: kind embedding plus its parameters;
     - (c) **"Mimic"**: track a reference *state* sequence, i.e. the next K Mario states. References come from LLM landmark polylines, coarse tree paths, or the atlas `moves` graph. Train on TAS and tree state sequences, but at run time track only our own sketches.
4. **Planning-time critic.** V(ŝ, goal) = learned ETA becomes a second opinion for the tree's start ranking and for `ar`'s "is this objective reachable" check. The reachability head already exists; this makes it horizon-aware.

### 1.5 Gates (we measure before claiming anything, since past learned gains were noise)

| Gate | Pass bar |
|---|---|
| G0 data | ≥ 50M per-frame transitions from journals and archive, ≥ 10 zones, 2 zones fully held out |
| G1 WM accuracy | 8- and 32-frame open-loop position error on held-out data, per action class (ground, air, wall, water), and action-ID accuracy ≥ 95% at k=8. Report the *chaotic* classes (wall kicks, ledges) separately |
| G2 gradient usefulness | On 200 (state, goto) pairs, does one gradient-refined plan beat the GA elite *in the emulator* more than 20% of the time? If it doesn't, stop here |
| G3 in-loop A/B | RH GA with immigrants vs without, at equal emulator frames, on a fixed zone benchmark (BoB, WF, castle legs): frames-to-Done and stars per 100M frames, 3 seeds |
| G4 amortized policy | Gen-0 seeding from π vs pink, same benchmark |

**Throughput note.** These are estimates, not measurements. A 3×1024 MLP is roughly 6 MFLOP per step, so a 4080 should manage around 10⁶–10⁷ forward steps/s at large batch, and about a third of that with the backward pass. That is the same order as the emulator's ~500k frames/s. SuperMimic's >10× rollout speed-up will therefore mostly *not* carry over. The case for the WM is their *64× fewer samples* from first-order gradients. G2 tests exactly that, so measure throughput at G1.

---

## 2. Objectives system: proposals from the coding-agent papers

v2 already does much of what these papers describe:
- exact Done/Lost, with no learned success monitor (MAESTRO needed a VLM at 89.8% accuracy for this);
- dense shaping through ETA (the papers only had binary success);
- free rewind, fork and backward re-solve;
- retire-never-delete libraries;
- LANDMARK_TIPS as an ASPIRE-style skill-card file.

What they show matters, and v2 lacks:

1. **An admission gate for LLM-written objectives and templates** (ASPIRE verifier; RAX rehearsal).
   - An `ar lib` or spec entry starts as `candidate`. It becomes `verified` only when a *fresh* agent, given the entry alone, reproduces Done from N jittered starts: perturbed position and yaw, and ±frame offsets of the funnel.
   - Store its **preconditions** (a machine-checkable guard, e.g. zone, `floor_y` band, holding nothing) and **evidence** (sessions, success rate, median frames).
   - Evidence: ASPIRE promotes only reproduced entries. RAX's rehearsal gate sits behind its 92.2 result. LANDMARK_TIPS #22 (8 of 8 good Ramp ends rejected) shows unvetted objectives cost us real runs.
2. **Per-stage outcome tables** (RAX).
   - Aggregate `ar` events by (objective key, template version, zone, phase reached): met, stalled at phase p, lost, or died.
   - The ETA phases already exist, so log *which phase* each attempt ended in.
   - This is the table that showed RAX that its DoubleDesk failures were mostly at the grasp. It is cheap to build.
3. **A "since last brief" text diff in `ar brief`** (CaP-X: a text diff beats multi-turn without it, and raw images do worse).
   - Include position and speed deltas, the action-transition string, the ETA curve summary (best, plateau frame), what lies between Mario and the target (from `geo`), and population diversity from the journal.
4. **A review pass on every stall or loss** (RAX's BEHAVIOR reviewer).
   - A second, cheap Sonnet call gets the trace plus the objective and returns: missed condition, likely cause, suggested edit (landmark, template change, or budget).
   - Claude stays the planner, and the reviewer only reads.
5. **K candidates plus synthesis for objectives** (CaP-X 55 → 66–68; ASPIRE evolution 0 → 42% on a stuck task).
   - When an objective stalls twice, generate 3–9 alternative landmarks or templates. Run each with the RH GA from the same state at equal budget.
   - Score each by **the downstream true goal's ETA after it** (not its own ETA, which it could game), then keep or merge.
   - This is the in-zone version of route credit.
6. **Compile solved legs into parameterized templates with preconditions** (RAX: tokens per success fall 50–100×).
   - After a zone is done, Claude turns the landmark sequence into a lambda template keyed by funnel, plus a policy list for the hard parts. Verify it per item 1.
   - The next attempt at that zone, for example after an upstream change, replays the template instead of re-planning.
7. **One `ar` session per zone in parallel, with complementary goals** (RAX 4 agents: 92.2 in 4.9 h vs 63.3 in 13.7 h for 1).
   - Funnels make zones independent, so share one lib, one tips file and one budget ledger.
8. **Backward-transfer pass** (ASPIRE +7.6 points suite-wide, +52 and +72 on single tasks): rerun the stars that failed earlier with the final library and tips.
9. **Documentation hygiene** (CaP-X: usage examples add 20%; MAESTRO: tools without a harness add nothing; CaP: buggy or inconsistent examples cause bugs).
   - Every native kind in OBJECTIVES.md and every lib template needs 1–2 *worked, succeeded* examples with args.
   - Untested skills are hidden from Claude until they have one.
   - Use one angle convention in the DSL (it mixes degrees and 0x800 yaw units today).
   - Tips need two failing runs before they are added (ASPIRE's 2-trial rule).
10. **Learned ETA residual instead of hand phase constants.** Write `eta = hand structure + c_kind,phase`, with c fit online per (kind, phase) from realised frames. This extends `etalearn.rs` from per-kind to per-phase, and from start ranking to fitness. It feeds §3 directly.

---

## 3. Differentiable objectives with the learned WM

### 3.1 The key identity

v2's GA fitness is already SuperMimic's objective (their Eq. 2):

```
J = Σ_{t<H} γ^t · ĉ_t · (eta(ŝ_{t-1}) − eta(ŝ_t))  +  γ^H · ĉ_H · V(ŝ_H, goal)   (+ bonus·P(Done), − 600·P(death ∨ Lost))
```

- The reward is potential-based shaping with the ETA as the potential.
- ĉ_t is the survival weight from the WM's death and Lost heads.
- V is the critic: the learned time-distance to the goal. Because the shaping is potential-based, V ≈ −γ·ETA-to-go is the natural critic, which is exactly the IQE head the backbone already has.

So **we don't need a new reward.** We need `eta` itself in a differentiable form. There are three routes, used together:

### 3.2 Route A: compile declarative objectives to torch

`goto`, spec and lambda objectives (tools/spec/FORMAT.md: phases, targets, predicates, penalties, latches) are declarative. Write one compiler, spec → torch:

- `frames_for(dist)`, which is d/30, is already smooth.
- Predicates become **margins**: `in_sphere` gives r − |p − c|, and `action_class ∈ S` gives Σ_{a∈S} p(a) from the WM's action distribution. Done is then σ(margin/τ).
- Phases: eta = Σ_k w_k·(eta_k + remaining_k), where w_k is the soft probability of being in phase k. The phase latches in `mem` become soft latches: m_t = max-ish(m_{t−1}, σ(margin_t)), through a smooth OR.
- `Lost`, meaning the zone was left, becomes the WM's zone-change event probability, which feeds ĉ_t.
- τ is annealed. During the backward pass everything is soft. In the emulator, the original hard Rust/spec evaluator scores the result, so the soft version only has to point the right way.

This gives gradients for every LLM-written objective for free, because Claude keeps writing JSON and the compiler does the rest.

### 3.3 Route B: distil native Rust kinds into a learned ETA head

For the 140-odd native kinds we won't hand-port, train η_φ(s, kind, params) to regress `eta` on logged states, where the label is exact and computed on any state for free. Then use ∇η_φ.

- Start with the kinds that dominate campaign time: Star::*, Coin::*, Painting::Enter, boss kinds and Tech::*.
- The "goal tokens" path in the backbone is the natural place for this.
- Combine it with Route C: η_φ handles the short-range shaped part and the IQE handles long range.

### 3.4 Route C: the learned quasimetric as the long-horizon objective

d(ŝ, g) from the IQE head is differentiable in both arguments. That gives three uses:

- **The critic V** beyond the horizon H (above).
- **Differentiable landmark placement.** For a stalled objective g, optimise a landmark position l to minimise d(s, l) + d(l, g), with l constrained to standable atlas cells (project onto the pole grid after each step).
  - Claude proposes the landmark *type* and the gradient refines *where*.
  - This targets the main known weakness: "badly made objectives" and the GA's ~100-frame myopia.
- **Objective-parameter gradients.** The same trick tunes lambda-template args (the 600 in `=x+600*sin(yaw)`, sphere radius, phase constants) against downstream ETA, instead of by hand.

### 3.5 What gradients are taken with respect to

| Variable | Use | Keeps project rules? |
|---|---|---|
| Stick yaw and magnitude sequence of a GA plan | Gradient immigrants (§1.4.1) | Yes: the GA stays the controller, and the emulator verifies |
| Policy-list lambda args p0..p7 and soft rule thresholds | Memetic operator (§1.4.2) | Yes: the genome stays a policy list |
| Amortized policy weights | SuperMimic BPTT (§1.4.3) | Yes: used as a proposer only |
| Landmark position and template args | §3.4 | Yes: Claude still picks the objectives |
| Per-phase ETA constants | Calibration (§2.10) | Yes: learned in-run, not hand-tuned |

### 3.6 Failure modes to plan for

- **Mode boundaries.** Gradients across jump / no-jump or ledge / no-ledge are zero or misleading. The GA handles these, and the gradient only acts within a fixed button pattern. Reset Adam whenever the WM's predicted action sequence changes class.
- **Model exploitation.** The optimiser finds WM errors, for example clipping through a wall the WM doesn't model. Mitigations: the exact geometry lookup (§1.2), short H with the critic, an ensemble-disagreement penalty, and always scoring in the emulator. SuperMimic avoided reward hacking because its policy is deterministic, and our final scoring is exact, which is stronger.
- **Chaos.** SM64 trajectories near walls and slopes diverge in a few frames. Report G1 per class. If wall and ledge classes fail, limit gradient refinement to ground and air segments, which are still most of the frames.

---

## Recommended order

1. **WM v0 on geometry-only zones** (§1.2, §1.3) plus the **spec → torch compiler for `goto`** (§3.2). Then gates G1 and G2.
   - About one week of work, and it decides whether everything differentiable is worth pursuing.
2. **Cheap objective-system wins that need no learning** (§2 items 2, 3, 9, and the admission gate of item 1). These help `ar` immediately, whatever G2 shows.
3. **If G2 passes:**
   - gradient immigrants in the RH GA, with A/B per G3;
   - then policy-arg gradients;
   - then the SuperMimic amortized controller (G4);
   - then Routes B and C (distilled native ETA, landmark placement).
4. **Parallel per-zone `ar` agents** (§2.7) with candidate ensembles (§2.5), once admission and stage tables exist to keep them honest.
