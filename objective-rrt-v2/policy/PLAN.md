# Reactive policy stack (goal: much better than a raw pink-noise GA, universal over states)

Layers, one module each; the integrator owns `types.rs`.

1. `menu`  : the reduced input space. Per state: button sets and stick regions that can change what happens next (derived from the guard table in scratchpad graph.json / speedchain_overlay.py, and validated by one-step emulator equivalence). `realize(Choice, Obs) -> GamePad`.
2. `sense` : geometry (floor probes, raycasts, edges, ledges, ceilings, predicted landing + zones), objects (nearest by category), Mario state flags. Fills `types::Sense` each frame.
3. `skills`: library of varied reactive controllers, `fn choose(&self, o:&Obs, params:&[f32]) -> Choice`.
4. `mpc`   : derivative-free refinement (CEM, (1+1)-ES, finite-difference/SPSA, grid) of the continuous stick part (`fine_ang`, `fine_mag`, `yaw`) by short rollouts.
5. `policy`: genome = rule list (observation predicate -> skill + params [+ mpc on/off]), evolved by GA; plays with `obrrt_gp::rollout::play`; `bench` compares with pink-noise GA (obrrt-search, `pink: 1.0`) and the existing GA/GP on `gp/cases/*.txt` (problem `speed` = SpeedChain, the speed-keeping task).

Rules: never edit files outside `policy/` (except reading). Build with your own CARGO_TARGET_DIR=/tmp/claude-1000/target-<name>; experiments in background with nohup, PID files, <=4 threads each, `nice -n 5`. Write results to policy/runs/<name>/ and short notes to policy/NOTES-<name>.md.
