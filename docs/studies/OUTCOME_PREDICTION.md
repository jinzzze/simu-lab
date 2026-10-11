# Supporting study: predict a grasp skill's outcome

[Study progression](../OVERVIEW.md) · [All experiments](../EXPERIMENTS.md)

After establishing the simple transfer, this study asked whether a learned model could predict what would happen when the fixed grasp skill received a proposed command.

The input is the initial object XY and an absolute grasp XY command. A small MLP predicts endpoint displacement, qualified-lift probability and success probability. All transition labels come from simulation. Personal video contributes through the upstream visual encoder in the RGB integration examples.

## Data and results

The dataset has 560 transitions from 112 initial scenes: 320 training, 80 validation and 160 test transitions. Each scene has five pre-recorded commands, so the test contains 32 independent initial scenes. Three seeds (41, 43, 47) train for 2,000 updates; the reported ensemble averages their final predictions.

| Predictor with true initial state | Mean endpoint error | P90 error | Correct success/failure labels |
|---|---:|---:|---:|
| Zero displacement | 90.8853 mm | 190.1115 mm | 89/160 |
| Training-mean displacement | 92.0621 mm | 122.1010 mm | 89/160 |
| Learned ensemble | 28.4107 mm | 80.2912 mm | 153/160 |

The 153/160 figure is classification accuracy. The executed test commands actually succeeded in 71/160 cases. Mean prediction error still exceeds the block's 20 mm width.

A preselected RGB integration case succeeded with 0.7228 mm final-position prediction error. A later RGB diagnostic reused the original 160 commands and outcomes, obtaining 28.2960 mm mean error and 153/160 correct labels. It added no physical executions. Those commands were originally generated from true state plus fixed offsets; this diagnostic does not measure RGB-based action selection.

## What this study establishes

The predictor beats simple displacement baselines on this fixed simulated skill. It was not used to choose actions or improve the four-goal policy. Variable goals, new objects, long-horizon prediction, stronger action-aware baselines and real-robot transfer remain untested.

## Evidence and reproduction

- [Configuration](../../configs/grasp_world_model.json)
- [All metrics](../../artifacts/reports/grasp_world_model_v1/results.json)
- [RGB diagnostic](../../artifacts/reports/grasp_world_model_v1/rgb_evaluation/results.json)
- [RGB integration video](../../artifacts/reports/grasp_world_model_v1/integration/rgb_world_model_demo.mp4)
- [Stage 1 environment and package](../SIMPLE_TASK_REPRODUCTION.md)

In a disposable extracted v1.0.0 package, existing checkpoints can regenerate predictions and reports with `scripts/export_grasp_world_model_predictions.py` and `scripts/report_grasp_world_model.py`. These update fixed-path derived files. The original recovery from a native-runtime reporting failure loaded saved checkpoints and took zero additional optimizer steps; the original training source remains in its snapshot.
