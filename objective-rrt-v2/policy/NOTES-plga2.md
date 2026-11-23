# plga2: policy-level fine operators for the policy-list GA

Genome unchanged in kind: `[(driver, hold, fine)]`, driver = library policy + lambda args/seed or pink seed; never flattened during search (only `rollout(keep)` for CLEAN_DUMP export). Results: runs/plga2/RESULTS.md. Code: `src/plga.rs`, `src/bin/plga.rs`, `src/eval.rs` (`Fine`, `BtnSpan`, `Seg.fine`, applied in `play_segs`). Binary: `runs/plga2/bin/plga_final`; queue: `runs/plga2/queue.sh`, `queue2.sh`; tuning runs `runs/plga2/tune/`; staged-policy copies in `runs/plga2/lib2/` (inventory untouched).

## What was added
1. Fine genes per segment (`eval::Fine`), applied closed-loop on whatever the segment's policy outputs: `dang` / `dmag` added to the Choice's fine_ang (1.0 = 11.25 deg) / fine_mag (acts only on Short-magnitude cells, so on Full cells `dmag` is inert), a button-override span (force/suppress A or B, `at,len` inside the hold; also valid on pink segments), and an optional MPC wrap `(H,N,range)` that reuses `mpc_pick` (stick only, its frames are counted in `mpc_frames`; when it fires it picks the stick yaw so `dang` is moot). Ops: nudge, btnspan, mpcwrap (flag `--mpc`), hold1 (+-1..3 hold shift).
2. `repol` op (not asked for, found necessary): hybrid's repolicy as a list op: keep the list up to a random time, a probe-ranked pool policy (60% top-4) takes over, no pink in prefix or tail. The old `handoff` spent ~65% of its children on pink prefixes/tails.
3. Gen-0 from the portfolio: pool probes (default args), one lambda draw per pool policy, 4 two-segment lists from ordered pairs of the best 3 (random hold 8..60); population not truncated (as in hybrid).
4. Credit: epsilon-greedy (25% uniform over active ops), floor 3%, credits pulled halfway to their mean at each checkpoint; fine-tune phase = last 30% of every checkpoint window (anytime-valid), fine ops only, hill-climbing one of the top 3 elites (neutral moves accepted). Guard against an infinite loop when no fine op applies.
Flags: `--no-fine --mpc --no-seed0 --no-eps --no-tune --no-repol --lib2 dir`.

## Findings
See RESULTS.md. Short: fresh 11/27/37/43% vs old plga 12/16/27/37 and hybrid 37/37/39/42 at 20k/60k/200k/560k; the gap to hybrid is closed from 200k (paired CI includes 0, routes 55-63 frames shorter), not at 20k (-26 pts fresh, -16 noise). MPC wrap hurts. Staged policies: no significant effect.
Why 20k stays open (hypothesis, untested): gen 0 alone (24-30 rollouts) uses 7-9k of the 20k frames, leaving ~40 children, and hybrid's pad-level children (jitter windows, button spans on realised pads, drop/insert) hit a successful repolicy parent edit much more often per rollout than closed-loop gene edits do.
