# Hybrid round: the policy inventory inside a per-start GA

Files: `src/hybrid.rs`, `src/bin/hybrid.rs` (feature `ga`), `runs/hybrid/` (queues, logs, JSON, `compare.py`, `RESULTS.md` = all tables, `lib/` = frozen copy of the library used).
Run from objective-rrt-v2: `hybrid <rom> <problem> pink|seeded|repolicy|market|polish|policy|wseed|wplain [--seeds 3] [--threads 2] [--budgets ...] [--scale s --pink p]`.

## What was built
My own anytime GA over open-loop pad sequences (steady state, pop 16, exact emulator-frame accounting, snapshots at 20k/50k/150k/560k frames per start,
so one run gives all four budgets), with variants:
- `pink`    equal-machinery baseline: pink-noise population; ops jitter / button span / drop-insert / resplice / crossover.
- `seeded`  (a) gen 0 = realised pads of every candidate policy (default instance + 1 sampled lambda draw), rest pink mutants.
- `repolicy`(b) + operator: keep a parent's pads up to a random frame, then a policy drives reactively (favoured: policies that did well from this start).
- `market`  (c) + operator probabilities adapted by credit (score gain per frame spent, EMA, floor 4%).
- `polish`  (d) + trim operator on the best individual (delete one frame, carry the rest).
- `policy`  every candidate once per start, no search.
- `wseed`   the v2 Worker GA (obrrt_search, `attempt_seeded`) seeded with the best policy rollout (pink 0 = overlay GA, 1 = pure pink).
Score = 1e6 - frames if Done, else -best_eta. 12 candidates for `speed` (ground-m02-gate, f06/f07 frozen, gate-speed-8, X00003/5/6, speed-chain-3, g08-g11), 6 for `lakitu_chain`.
3 seeds, 15 train + 21 held-out starts (speed) so n = 45 / 63 per cell.

## speed (success/n, mean frames-to-Done over successes; emu = emulator frames per start incl. policy rollouts)
| method | emu | train | test |
|---|---|---|---|
| policy only (12 candidates, nothing else) | 0.002 M | 39/45 71f | 52/63 101f |
| own GA, pink population (equal machinery) | 0.02 / 0.15 / 0.56 M | 11/45, 23/45, 29/45 | 11/63, 28/63, 39/63 |
| own GA seeded | 0.02 / 0.15 / 0.56 M | 40/45 72f, 41/45 69f, 43/45 71f | 57/63 96f, 57/63 89f, 57/63 86f |
| **own GA + repolicy** | 0.02 M | 45/45 82f | **63/63 102f** |
| **own GA + repolicy + market** | 0.02 / 0.15 / 0.56 M | 45/45 84f, 80f, 77f | **63/63 103f, 98f, 93f** |
| own GA + repolicy + market + trim | 0.02 / 0.15 / 0.56 M | 45/45 84f, 79f, 77f | 63/63 102f, 97f, 93f |
| overlay GA (Worker), scale 0.027 / 0.067 / 0.2 | 0.05 / 0.07 / 0.14 M | 45/45 82f, 78f, 72f | 61/63 109f, 62/63 105f, 62/63 92f |
| pink-noise GA (Worker), scale 0.027 / 0.067 / 0.2 / 0.75 | 0.06 / 0.09 / 0.19 / 0.56 M | 42, 43, 44, 44 /45 (94, 87, 77, 71f) | 53, 59, 60, 61 /63 (142, 136, 106, 96f) |
| overlay GA + policy seed (Worker), 0.027 / 0.067 / 0.2 / 0.75 | 0.023 / 0.046 / 0.129 / 0.455 M | 43, 43, 45, 45 /45 | 59/63 96f, 60/63 96f, 60/63 89f, 61/63 88f |
| pink GA + policy seed (Worker), 0.027 / 0.067 / 0.2 / 0.75 | 0.031 / 0.058 / 0.161 / 0.562 M | 39, 39, 42, 45 /45 | 57/63 99f, 57/63 95f, 59/63 95f, 61/63 90f |

Reading:
1. The `repolicy` operator is what matters: **63/63 held-out and 45/45 train at 0.02 M frames per start**, against 61/63 at 0.05 M (overlay GA) and 53/63 at 0.06 M (pure pink Worker GA, 94-142 frames). That is 2.5x-3x fewer frames and 100 % instead of 84-97 %. Seeding alone (a) plateaus at 57/63: the six hard starts never get solved because the population fills with policy descendants and has no reactive recovery.
2. Final route length: at 0.56 M the hybrid reaches 93 test frames; the overlay GA with 0.43 M gets 85.6 and with a policy seed 88. The hybrid wins on success per frame at low budgets and ties on frames at about 0.15 M; it does not beat the Worker GA's best frames at large budgets (its mutation operators are weaker: my own pink GA at 0.56 M solves 39/63 where the Worker's pink GA solves 61/63).
3. The trim operator (d) changed nothing measurable (97 vs 98 frames at 0.15 M; equal at 0.56 M). The market (c) gave 1-2 frames over fixed weights at 0.15-0.56 M (98 vs 97-98, 93 vs 96): within noise for 3 seeds.
4. Seeding the Worker GA with the best policy rollout (`wseed`) halves its frames (it stops early when the seed already finishes) and shortens routes (pink: 142 -> 99 test frames at 0.066 -> 0.036 M; overlay GA: 109 -> 96 at 0.052 -> 0.029 M), at the price of 1-3 held-out starts at the smaller scales for the overlay GA (59-60/63 vs 61-62/63).
5. Amortisation: the candidates are the LLM-authored and portfolio policies. Their authoring cost (about 15-40 evaluated revisions each, a few M frames per revision on 15-44 starts) was not metered; the rule-GA policies cost 67-72 M frames per run (about 2 M per start over 36 starts, more than the whole per-start budget), so the table's frames do not include policy training. The candidate list and the m02-gate region were chosen by looking at start results (held-out starts included), so absolute numbers are optimistic by the selection; the comparison between variants uses the same candidates.

## lakitu_chain (21 train + 8 val + 13 test starts, x3 seeds)
The Worker's "GA" on this problem is not a search: the objective is a fixed play (the hand overlay), so pink and overlay GA are the same 0.005 M-frame replay: 63/63 train, 30/39 test (71 frames). The hybrid does not beat it:
| method | emu | train | test |
|---|---|---|---|
| hand overlay replay (Worker "GA") | 0.005 M | 63/63 76f | 30/39 71f |
| own GA, pink population | 0.56 M | 6/63 | 8/39 |
| own GA seeded | 0.15 / 0.56 M | 33/63, 44/63 | 23/39, 25/39 |
| repolicy / market / trim | 0.56 M | 47-48/63 (81-87f) | 27-28/39 (85-86f) |
The lakitu policies do not reproduce their authors' numbers under the stock reader: `air-lak-2` evaluates to 1/21 train, 2/13 test with policy4.bin and with the hybrid (the authors measured 17/21, 9/13 with their private reader `src/author_env.rs`). They need re-wiring to the real sense layer (landing frames, zone flags) before they help here; until then the lakitu result is the GA discovering the chain from policy prefixes.

## Recommendation (v2 GA, the `policy` origin in search/src/ga.rs)
1. Add an operator `repolicy` in the GA (the origin exists: ORIGIN_POLICY): cut a population member's pads at a random frame, finish with a reactive inventory policy (favour the ones that did best from this start, 50 % sampled lambda args), keep the realised pads as the child. That is the component that took success from 53-61/63 to 63/63 at 5-30 % of the frames.
2. Seed gen 0 with the best-of-inventory rollout (`attempt_seeded` already takes one seed and works: halves frames, shorter routes), cost 0.002 M per start for 12 candidates.
3. Do not bother with the trim operator or the adaptive market for now; keep them off until a larger test shows a gain.
4. obrrt-policy depends on obrrt-search (feature `ga`), so search cannot link the interpreter directly: expose inventory policies to ga.rs through a trait object / the existing `moves::install_controller` hook (callbacks that return a pad from the trajectory), as the 14 hand policies are.
5. Before relying on lakitu policies, re-evaluate them with the real sense layer.
