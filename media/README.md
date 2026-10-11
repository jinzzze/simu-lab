# Presentation figures

`task-preview.png` uses recorded frames, with aspect ratio preserved, from X01 `train_P1_G3_01`, the D_seed7 visual policy's first simple-transfer scene and the video-timing policy's first predeclared seed-7 test case. The frames illustrate task settings, not aggregate success.

`study-progression.svg` shows the research questions. The outcome predictor is a supporting branch, not an evaluated planner for the four-goal policy. `method.svg` separates offline demonstration generation from online observations and commands.

[Measured results](../docs/EXPERIMENTS.md) · [Data and source scope](../docs/DATA_MODEL_CARD.md)

## Task schematics

- `task-1-simple-transfer.svg`: editable schematic of the simple-transfer task, based on `docs/studies/SIMPLE_TRANSFER.md` and the frozen success rule in `src/sim/grasp_env.py`.
- `task-2-four-goal.png`: the author-supplied four-goal diagram, preserved byte-for-byte. Its embedded “local review draft” subtitle belongs to the original illustration; current release instructions are in the main README.

Both diagrams explain task content, are not to scale and do not represent measured trajectories or performance. The Task 1 dashed block marks the intended final placement.

## Task 1 supervision and results

`task-1-supervision.png` plots the first actual pretraining sample with a valid object box and all 21 hand landmarks inside the image. It displays the same stored 224 × 224 RGB input in all three panels; overlays come directly from the frozen target arrays. The example illustrates available labels, not independently measured label accuracy. `task-1-results.png` plots recorded counts from the primary, ImageNet-reference and 32-image result files.

`task-1-figure-sources.json` records the sample, normalized targets, input hashes and plotted counts. To rebuild both figures, run `python media/plot_task_1.py --data <path-to-pilot_pretraining_v1.npz>` in an environment with NumPy and Matplotlib. The input NPZ is distributed with the v1.0.0 data package; its hash must match the tracked pretraining manifest. Rebuilding these figures does not run training or robot evaluation.
