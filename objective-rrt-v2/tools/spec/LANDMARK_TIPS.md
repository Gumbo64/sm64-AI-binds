# Tips for making landmarks (and objectives)

Lessons from the `legs` runs (BoB: bob-chomp, bob-king, fast-king), so that landmark routes, whoever writes them (a person or the LLM loop), avoid the mistakes already made. `pack.py` includes this file in its briefs. Add a tip whenever a route fails for a reason a better landmark would have avoided, and say which run showed it.

## Where a landmark ends

1. **End on stable ground.** Make the target a spot where Mario was grounded on flat, non-slippery floor in a reference route: the top of a climb, not partway up a slope. A leg ending halfway up a slope backslides, and the next leg can't start. (fast-king walk runs: Ledge1 → Ledge2.)
2. **Done is a state, not a big sphere.** Use a small region (~120) plus "grounded" (stationary or moving action class) plus controllable. A 300-unit sphere fired mid-jump on a trajectory falling off the path. (fast-king walk: Ledge1.)
3. **The next leg must be able to start from the Done state.** `done_requires_next` is on by default. Opt out only for legs that really end mid-air.
4. **No doomed progress.** If the objective's goal can be undone right after it looks met (the King thrown off the tower, then a dialog and a return; a dive that slides back), the ETA must count the undo. Otherwise the GA picks the doomed version. (bob-king: off-stage throws.)

## Between landmarks

5. **Don't join landmarks with a straight line across bad terrain.** If the straight line crosses steep or slippery floor, a drop or a wall, add an intermediate landmark on the reference path's way around. (Ledge1 → Ledge2: the reference goes north-east first, then west.)
6. **Don't over-split.** Every landmark is a stop and a cut; add them only where the terrain or a technique needs one.
7. **Mind triggers on retry.** A leg that restarts next to a sign or NPC re-opens its dialog every time. Place landmarks so retries don't trigger dialogs, or count the dialog in the ETA. (bob-king: BackPath and the sign.)

## Techniques

8. **Use native technique objectives for hard tricks** (`Tech::LakituSkip` hill, `Tech::BljRun`, `Tech::PoleRedirect`, `Tech::HyperClimb`) instead of re-describing them as spec phases. They carry tuned overlays.
9. **Chain techniques through their hand-off state.** A technique leg's Done should require that the next technique can start (BLJ Done only where PoleRedirect can start). Filtering ends afterwards kept 1 of ~35. (fast-king s1b → s2.)
10. **Aim each technique at the next landmark, not "anywhere cheap".** A redirect aimed at the cheapest pole missed the TAS's second tree. (fast-king s3 → s4.)
11. **Seed precise sequences from a reference.** A random or steered GA didn't find the jump-kick-landing + jump at fv 298 to the second tree (~2400 units short); seeding from the 2026 TAS made it. (fast-king s5.) Do the same for the TAS's summit climb (double jump → wall kick → double jump).
12. **Movement per leg:** steep slopes make dives slide back; use kicks or the climb the reference used. Keep the move library on elsewhere: long jumps, dives and kicks shorten most legs.

## Cutscenes, dialogs, menus

13. **Let the press-through handle cutscenes and menus,** and skip the "next leg can start" check while Mario isn't controllable. (Door legs went 44 → 131 frames before this.)
14. **Stop pressing as soon as the dialog is closed and Mario is controllable.** Leftover A presses make him jump in place. (BobPainting leg start.)

## Checking

15. **Validate every landmark against a reference** (`spec-validate`): Done where the reference achieved it, ETA calibration near 1, Done stable in the reference (no slide or fall in the next 20 frames), and no lint warnings (steep target floor, no ground condition, a straight leg over bad terrain).
16. **Honest phase frames:** fixed phase constants should come from the game (animations, timers) or a reference, and be checked with the validator's calibration.

## Added from the runs

17. **Check the floor type, not just the shape.** BoB's east mountain path (PoleBase → Ledge1 → Ledge1b, (7397, 1251, -4564) to (8040, 1876, -5780)) is SURFACE_NOISE_SLIPPERY (0x2A) at ny ~0.82: no point on it is stable ground (`legs --check` finds none within 700 to snap to, and the reference's own Done states there slide within 20 frames). Every walked fast-king run failed there (walk-s1b … s5); the one success (legs-bob-king-qd4-s1) went up on a retry. Prefer a route around it (the BLJ / tree chain), or a leg with its own technique, and don't end a leg on it.
18. **Seeds must start where the leg starts.** A reference leg that began with its own dialog press-through, used as a seed from the leg's start, replayed the dialog's A presses in a controllable state: Mario jumped in place for ~130 frames (fast-king-s6 BobPainting, 1956-2091). `legs` now seeds from the reference's first controllable state of the leg (BobPainting 362 → 304 frames in a leg-only re-run).
19. **Steep target floors move the landmark.** ChompSide (5339, 927, 1098) sits on ny 0.61 floor; the snap moved it 290-630 to flat ground on the reference path.
20. **Keep moving through slippery stretches: waypoints as phases, not as legs.** A leg boundary is a stop, a beam cut and a new attempt; on slippery floor every stop slides. Put the stretch's landmarks in one Spec as latched phases (passing a waypoint counts while moving), with one Done on the first stable ground after it and a horizon for the whole stretch. `legs --check` flags leg ends on slippery or steep floor. (fast-king-walk: PoleBase → Ledge1 → Ledge2 on BoB's NOISE_SLIPPERY east path, merged in fast-king-walk-merged.)
21. **Don't snap or settle a landmark right before a technique.** The BLJ needs Mario's approach; a Ramp landmark snapped to a small grounded circle made it fail 2-13 ETA frames short (blj-tas-r2 w1), while the 400-unit sphere he runs through passed (Blj 123, blj-tas-r2c). Let such a landmark be passed through, as tip 20's waypoints are.
22. **The stability probe can reject good ends.** With its default 20 frames it rejected 8 of 8 Ramp ends and 16 ChompPost ends (r2 runs), legs earlier routes solved. Where a leg fails with Done bins found but none kept, check the "rejected by the stability probe" line, and set `stable_frames: 0` on it if the end is fine.
23. **Don't end a leg where the next one is already decided.** The Hill Lakitu skip ended 350 from the castle door at ~100 u/f: its score stopped there, so a last-moment slide kick looked free, and the door leg started 3 frames from a hit it couldn't avoid (all rollouts dive-slide; rejecting slide states can't help). As one leg through the door (`OBRRT_LAKITU_THROUGH_DOOR=1`): 232 frames vs 184 + 88 (lakitu-through vs lakitu-door-w1). Same idea as tip 20: put the thing that judges the arrival inside the leg.
24. **Never let a side effect count as finishing.** With `"dialogs": "stop"` any dialog is the leg's Done, with its success bonus, and the earliest one is carried forward. So throwing King Bob-omb off the summit, whose return opens a dialog, was the best-paid move: 424 M GA frames, 4 returns, 2739 frames (legs-fast-king-walk-merged-r2-w1). Keep `"dialogs": "stop"` only on a leg whose goal is that dialog (KingIntro), and split a fight into one leg per hit.
25. **A seed used to switch the moves off.** The GA dropped policy rollouts on any seeded attempt, so on every `seed_ref` route no move policy ever played, and `OBRRT_MOVES_MODES` was off too (`move-ga: 0 attempts`, legs-lakitu-through-s1). `legs` now defaults `OBRRT_MOVES_SEEDED`, `_MODES` and `_EXTRA` on, with population / 2 policy rollouts a leg. If a route seems to ignore a technique, read `move_ga.json` first: it says what was simulated and what won.
26. **Kicks climb what dives can't, but dive chains are faster where both work.** Alone on BoB's east slope (kind 0x2A, ny 0.82) `jump-kick` covers 3651 units and +877 in 120 frames at about 30 u/f, where every other policy stalls or slides back (extra_lab). In a whole leg the GA still mostly prefers dives, rollouts and steep jumps at up to 48: EastPath took 21 jump-kick frames in one seed and none in the other, at 350 and 337 frames (before: 338, 526). Don't expect a technique to show in a route just because it is available; it shows where nothing else gets through.
27. **A tie with the reference can break the next leg.** A seeded GA tied the TAS's first leg at 108 frames but ended about 8 units away, and the TAS's frame-exact BLJ then missed the warp for 171 M GA frames (legs-tas-first-star-s1). `legs` now keeps the reference's own end of a leg as a candidate whenever the parent is still exactly on the reference, and `OBRRT_GA_KEEP_SEED=1` keeps the seed in the elite.
28. **Before calling a reference's detour wasted, check what else it does.** The last VCutM leg "beat" the TAS by 39 frames only because its Done asked for the star alone, while the TAS's detour presses the vanish cap switch (legs-tas-last). Compare the save flags and other lasting state at the end of the leg, and put whatever the reference achieved into Done.
29. **Check that a grounded Done can be true at the target.** blj-tas.ramp0's UpperPath sphere (449, 3229, -5721) r400 ended 221 above the only floor there (2608), so Mario sat at ETA 0 in mid-air and never finished: 692.7 M GA frames a run. If a leg fails at ETA 0, read the floor under the target first.
30. **A climb aimed at "up" can overshoot the route.** HyperClimb's generic +1000 landed west of the upper path at (-163, 3036, -8091), with no way back up. Aim it at where the next leg starts.
