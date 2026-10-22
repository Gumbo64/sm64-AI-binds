# Examples

They use only sm64-binds: every candidate is played in the game. On x86-64
Linux, sm64-binds runs its precompiled build of the game by default.

| Example | What it does |
| --- | --- |
| `curiosity` | The curiosity run, forever: population 128, genome 600, 1-frame commits. Every 100 iterations saves its path to `runs/curiosity-pop128-genlen600-step1.m64` and its scoring to `runs-scoring/`; resumes from the saved path, rebuilding the scoring by replaying it. |
| `main` | The first population run, with the default scoring (population 128, genome 250). |
| `bob_coins <rom> [out] [seed] [variant] [minutes] [coins] [<coins.m64> <frame>]` | From power-on: searches its way to the castle, into BoB, and collects coins with the curiosity run's search settings (or plays another file's inputs from where its Mario landed in BoB). |

## Tools

In `../tools/`, for comparing input noise and looking at replays and saved runs: `cargo run --release --bin <name> -- <args>`.

| Tool | What it does |
| --- | --- |
| `noise_bench <rom> <variant> [frames] [out] [seed] [threads]` | The curiosity run (`curiosity`) with a choice of input noise (`src/noise.rs`) and a seed. Same seed, same frames. |
| `curiosity_data` | Writes the saved curiosity scoring's visit counts to `curiosity_output.csv`. |
| `replay_info <rom> <m64> [frame]` | Mario's state at a frame and at the end of a replay. |
| `level_timeline <rom> <m64> [every]` | Where a replay changes level or area. |
