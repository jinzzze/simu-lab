# Presentation figures

`task-preview.png` uses recorded frames, with aspect ratio preserved, from X01 `train_P1_G3_01`, the D_seed7 visual policy's first simple-transfer scene and the video-timing policy's first predeclared seed-7 test case. The frames illustrate task settings, not aggregate success.

`study-progression.svg` shows the research questions. The outcome predictor is a supporting branch, not an evaluated planner for the four-goal policy. `method.svg` separates offline demonstration generation from online observations and commands.

[Measured results](../docs/EXPERIMENTS.md) · [Data and source scope](../docs/DATA_MODEL_CARD.md)

## Task schematics

- `task-1-simple-transfer.svg`: editable schematic of the simple-transfer task, based on `docs/studies/SIMPLE_TRANSFER.md` and the frozen success rule in `src/sim/grasp_env.py`.
- `task-2-four-goal.png`: the author-supplied four-goal diagram, preserved byte-for-byte. Its embedded “local review draft” subtitle belongs to the original illustration; current release instructions are in the main README.

Both diagrams explain task content, are not to scale and do not represent measured trajectories or performance. The Task 1 dashed block marks the intended final placement.
