# Stage 1: visual learning for a simple transfer

[Study progression](../OVERVIEW.md) · [All experiments](../EXPERIMENTS.md) · [Reproduce this stage](../SIMPLE_TASK_REPRODUCTION.md)

The robot must grasp a 20 × 20 × 10 mm block and place its entire footprint beyond a black line, then release and withdraw while the block remains stable. A learned XY estimator supplies the initial position to a common fixed grasp controller.

Twelve personal phone clips contain 2,162 frames over 86.48 seconds. The visual pretraining uses 438 shared full RGB images. Object-box and hand-keypoint candidates are generated independently; invalid targets are masked. They are candidate labels, not independently measured annotation accuracy.

## Comparison

All primary groups share ImageNet ResNet-18 initialization, RGB inputs, splits, simulated adaptation labels and the fixed controller. Their auxiliary training targets differ. R is an additional reference with ImageNet initialization and simulator adaptation but no phone-video pretraining.

| Group | Personal-video pretraining | 256 adaptation images | 32 adaptation images |
|---|---|---:|---:|
| A | RGB consistency | 192/192 | 184/192 |
| B | RGB + object boxes | 192/192 | 183/192 |
| C | RGB + hand keypoints | 192/192 | 184/192 |
| D | RGB + both targets | 192/192 | 181/192 |
| R | None | 192/192 | 182/192 |

Each cell combines three model seeds over the same 64 initial scenes. The 256-image primary comparison is saturated, including the later reference R. The 32-image follow-up was designed after seeing that ceiling and reuses the test scenes; it is exploratory.

## Findings

Combined hand/object supervision showed no measured task-success advantage. C had lower mean localization error, but did not improve success and was not better in every seed. The 46 low-data failures involve six shared scenes; two fail for every group/seed combination. Small rate differences do not establish statistically significant improvement or degradation.

This stage demonstrated a working real-video encoder → simulated RGB localization → physical grasp pipeline. It also motivated studying learned control: success here largely depends on a common scripted controller once the estimated position is accurate enough.

## Evidence

- [Primary results](../../artifacts/reports/visual_ablation_v1/results.json)
- [First-scene comparison video](../../artifacts/reports/visual_ablation_v1/first_scene_comparison.mp4), seed 7, scene 30000
- [Visual implementation](../VISUAL_TRAINING.md)
- [Frozen source, inputs and checkpoints](https://github.com/jinzzze/simu-lab/releases/tag/v1.0.0)

Only object position varies in this simulation test. Broad real-scene diversity, independent label accuracy and real-robot transfer were not evaluated.
