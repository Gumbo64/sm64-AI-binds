# The `repolicy` breeding operator inside the production Worker GA

## What was built
- `search/src/ga.rs`: `trait BreedHook { fn session() -> Box<dyn BreedSession> }`, `BreedSession::breed(game, start, save, stage, prefix, states, horizon, rng) -> Option<(pads, emu_frames)>`, `set_breed_hook`, `set_policy_repolicy`, env `OBRRT_POLICY_REPOLICY=p` (default 0 = off; per-offspring probability), `OBRRT_POLICY_REPOLICY_MARKET=1` (probability = 2p x its credit share, clamped to [p/4, 4p]; credit = fitness gain over the elite mean per emulator frame, EMA 0.7 per generation, against all other operators pooled), `REPOLICY_STATS` tallies.
  In `attempt_open` (single-stage objectives only): with probability p a child is made by cutting a random elite parent at a uniform cut, the hook replays the prefix and an inventory policy plays on, and the realised pads are re-evaluated by the normal `rollout` (no overlay on them: RAW_PREFIX) as origin `repolicy` (shares the existing origin slot, so it shows in the makeup stats). Not wired into the rolling-horizon path (`child()` at ~l.980).
  Frame charge (honest, double counts): hook frames (prefix replay + policy play) + the GA's re-evaluation of the child. Roughly 180 hook frames + ~100 re-eval frames per child.
- `policy/src/repolicy.rs` (feature ga): `PolicyBreed` over the same 131-policy inventory as seed.rs; per attempt a session that remembers each policy's mean score from this start and picks among its best 4 60 % of the time (else uniform); instance = default or sampled lambda draw (50/50). `policy ga` installs it when the env var is set and prints children/wins/hook frames.
- Default off is byte-identical: `policy ga --pink 0 --scale 0.027` on 6 starts, frozen `runs/clean/bin/policy` vs the new build, identical output (also nothing is drawn from the rng when unset).
- Raw: `runs/repolicy/res/<set>/{rp15,rp15seed,rp15mkt}_overlay_<scale>.log`, `queue.sh`, `summarize.py`, `summary.md`, `bin/policy` (frozen build).

## Measurement
Same protocol as NOTES-deploy.md: `policy ga --pink 0 --scale s --seeds 3 --threads 2` on the clean fresh/field/noise start sets (n = 59/45/55 starts x 3 seeds), baselines = `clean/res/*/wk_overlay_*` and `deploy/res/*/seed_overlay_*`, p = 0.15. Bootstrap over starts (1000 draws), paired = same starts and seeds. Emu frames include hook and rollout-seed frames. Full tables: `runs/repolicy/summary.md` (regenerate with `python3 -I runs/repolicy/summarize.py`).
Done when this was written: scales 0.027 / 0.067 / 0.2 for `+repolicy` and `+repolicy +seed`, all three sets. NOT done yet: scale 0.75, and the market variant (`rp15mkt_*`); they are in the same queue (`runs/repolicy/queue.sh`, pid in queue.pid) and the tables fill in when `summarize.py` is rerun.

Success (paired diff vs plain overlay in points, 95 % CI), emu frames per start:
| set | scale | overlay | +repolicy | +repolicy +seed | (+seed alone) |
|---|---|---|---|---|---|
| fresh | 0.027 | 37.9% 66k | 39.0% +1.1 (0.0..+2.8) 69k | 39.0% +1.1 63k | 37.9% 62k |
| fresh | 0.067 | 38.4% 118k | 40.7% +2.3 (+0.6..+4.5) 125k | 40.1% +1.7 120k | 39.0% |
| fresh | 0.2 | 42.9% 315k | 44.1% +1.1 (-3.4..+5.6) 338k | 45.2% +2.3 (-2.3..+6.8) 339k | 41.2% |
| field | 0.027 | 16.3% 67k | 17.8% +1.5 (0..+4.4) 70k | 15.6% -0.7 66k | 15.6% |
| field | 0.067 | 19.3% 123k | 19.3% +0.0 129k | 15.6% -3.7 124k | 15.6% |
| field | 0.2 | 20.7% 342k | 19.3% -1.5 (-4.4..0) 365k | 15.6% -5.2 (-11.9..0) 360k | 17.0% |
| noise | 0.027 | 42.4% 66k | 41.2% -1.2 (-3.6..+1.2) 69k | 53.3% +10.9 (+3.0..+19.4) 63k | 53.3% |
| noise | 0.067 | 44.8% 118k | 44.2% -0.6 125k | 54.5% +9.7 (+3.0..+17.6) 118k | 53.9% |
| noise | 0.2 | 52.7% 312k | 52.1% -0.6 (-8.5..+7.3) 336k | 58.2% +5.5 (-3.0..+13.9) 326k | 56.4% |

Frames-to-Done on successes: repolicy alone tracks the overlay (fresh 143/139/127 vs 140/134/125; noise 149/146/137 vs 154/152/143); with the seed it tracks the seed alone (fresh 164/155/140 vs seed 164/159/138; noise 177/166/149). Hook frames are about 5-8 % of the total at 0.027-0.2 (e.g. fresh 0.2: 5.8 M over 180 attempts = 32 k/attempt of 338 k).
Wins: the final best came from a repolicy child in 14-35 % of attempts at 0.027-0.067 when seeded (noise 0.067 +seed: 63 of 165) and 11-18 % unseeded/at 0.2; many of these attempts are failures, so winning is not the same as solving.

## Reading (honest)
- On these clean sets the operator does not change success: every paired difference vs plain overlay is within +-2.3 points except fresh 0.067 (+2.3, CI +0.6..+4.5, small and unreplicated elsewhere); field is flat to slightly negative (-1.5 to +1.5 alone). It costs about +4 to +8 % emulator frames (69/125/338 k vs 66/118/315 k on fresh).
- Combined with the seed: the noise gain (+10.9 / +9.7 / +5.5) is the seed's (seed alone +10.9 / +9.1 / +3.6); paired vs seed alone the combination is +0.0 / +0.6 / +1.8 (CIs include 0). On fresh 0.2 it is +4.0 over seed alone (CI +0.6..+7.9), but the overlay-relative +2.3 is inside its CI: treat as noise. Field is again hurt by the seed (-0.7 to -5.2).
- So the 63/63 of the standalone hybrid does not transfer here. Hypotheses (not tested): (1) that result was on the gate-region `speed` starts for which the candidate policies were written (and chosen with held-out starts visible); the clean fresh/field/noise starts are mostly places no inventory policy handles, so policy tails rarely help; (2) the Worker's overlay GA already has a strong closed-loop mover (the steered rollouts) that plays the role of the policy tail; (3) the hybrid's baseline was a weak pink GA, where a good tail adds a lot; the overlay Worker is already much stronger at equal frames.
- Not measured: other p (only 0.15; a 0.1 and 0.3 smoke on 6 starts ran fine), the market variant and 0.75 (queued), other start sets such as the gate-region train/test starts of NOTES-hybrid, lakitu.
- Recommendation: leave `OBRRT_POLICY_REPOLICY` off (default). The mechanism works and is cheap, so it can be re-tried on the gate-region sets where the standalone result came from.
