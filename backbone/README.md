# SM64 state backbone (sm64_AI_backbone.md)

PyTorch implementation of the backbone and heads in `../../sm64_AI_backbone.md`,
trained on the 14-star indefinite tree run (`objective-rrt/runs/indef-mgel-lg-pf-s2`)
and the 2026 120-star TAS (`objective-rrt/ref/tas/full-120star-2026-U.m64`).

## Pipeline

```sh
cd sm64-AI-binds
# 1. Raw features (Rust; reads RAM, no normalisation). Schema = column names.
objective-rrt/target/release/dataset schema backbone/data/schema.json
objective-rrt/target/release/dataset tas baserom.us.z64 objective-rrt/ref/tas/full-120star-2026-U.m64 backbone/data/tas120
backbone/.venv/bin/python backbone/select_nodes.py objective-rrt/runs/indef-mgel-lg-pf-s2 backbone/data/select.bin 1000000
objective-rrt/target/release/dataset tree baserom.us.z64 objective-rrt/runs/indef-mgel-lg-pf-s2 backbone/data/select.bin backbone/data/tree14 8
# 2. Train (MLflow: sqlite backbone/mlflow.db, artifacts backbone/mlartifacts)
cd backbone && .venv/bin/python train.py --steps 80000 --run_name <name>
.venv/bin/mlflow ui --backend-store-uri sqlite:///$PWD/mlflow.db --port 5000
# 3. ONNX (encoder / distance / goal tail), checked against torch with onnxruntime
.venv/bin/python export_onnx.py checkpoints/best.pt onnx
```

## Pieces

- `objective-rrt/core/src/features.rs`: raw feature reader (global int/float
  columns, nearest 64 non-cosmetic objects, ≤32 object triangles, zone static
  triangles). New layout fields in `tools/gen_layout.py` (additive only).
- `objective-rrt/run/src/dataset.rs`: DFS replay of `tree.bin` from power-on
  (savestates at branch points), samples every 4th frame on selected edges
  plus the last 64 frames of their parents; TAS: every frame.
- `sm64bb/schema.py`: behaviour → category, surface classes, RNG index table.
- `sm64bb/data.py`: store, k-step chains, hindsight pairs, labels.
- `sm64bb/model.py`: tokenizers, encoder (d=256, 6 layers, 8 heads), heads.
- `train.py`: losses, phase schedule, offline metrics, MLflow.

## Deviations from the spec

- Dynamics horizons are k ∈ {4, 16, 64}: samples are every 4th frame (k=1 not stored).
- No `NAV` tokens (the geo.rs mesh isn't exported to Python yet).
- `HELD` is the held object's own token with an extra HELD type embedding.
- Policy "controller" = the objective kind that made the edge (objective-rrt's
  `via`, ~100 kinds), θ = 16-frame stick/button summary; the controller
  library of the spec doesn't exist yet. TAS samples train θ only.
- Reachability negatives: goals with fewer stars (unreachable) and TAS pairs
  with < 50% of the TAS's frames. Failed search attempts aren't used (no goal state).
- Embedding (bisimulation) head and phase 3 (expert iteration) not done.
- Validation: tree nodes with id % 20 == 0; TAS levels 11 (WDW) and 22 (LLL)
  are held out entirely (the tree never visits them either).

## Race and follow runs in sm64-viewer

`race.py <ref>` and `follow.py <ref>` write objective-rrt-style run directories
(`objective-rrt/runs/race-*`, `follow-*`), shown live at
`http://localhost:5173/sm64-viewer/?tree=<run>`. The header shows the run
against its reference (hover for per-star deltas); **▶ play best** plays the
current best route; the best route is drawn in gold.

What each point label ("Objectives that made the states here") means in a race:

| Label | The state was reached by |
|---|---|
| `start` | the race's first state: the reference's first frame with Mario in control |
| `reference` | the reference run itself: its state every 16 frames (one chain, not a search) |
| `replay_ref` | 16 frames of the reference's own inputs, played from this branch's parent (same inputs, different state, so a different path) |
| `inv16` | 16 frames of model inputs (inverse dynamics) aimed at the reference state 16 frames past the parent's aligned point: tracking at the reference's pace |
| `inv24`, `inv32` | the same aimed 24 / 32 frames ahead within 16 frames: shortcut attempts |
| `inv64` | the first 16 frames of a 64-frame plan toward the reference state 64 frames ahead |
| `inv_s` | like `inv24` with the model's inputs sampled instead of the most likely ones (variety) |
| `noise_ref`, `noise_inv24` | `replay_ref` / `inv24` with the stick rotated (σ ≈ 20°) and scaled ±15 %, and 3 % of A/B/Z frames flipped |
| `ref_fallback` | the reference's own path, used when a segment found nothing faster |

A follow run has one chain: `inverse` (model inputs) and `reference`
(the reference's inputs, used in dialogs, menus and cutscenes).
