# Experiment code

[Project overview](../docs/OVERVIEW.md) · [Results across stages](../docs/EXPERIMENTS.md) · [Code map](../docs/CODE_MAP.md)

These directories contain successive stages of the same personal-video robot-learning project. Their names preserve the frozen source paths used by the released models.

| Directory | Research question | Read first |
|---|---|---|
| `paired_two_board_v2` | How can a learned controller act on four specified goals? | [Method](../docs/METHOD.md) |
| `cross_goal_composition_v3` | Does broader start/goal coverage improve composition? | [Coverage report](cross_goal_composition_v3/COVERAGE_REPORT_EN.md) |
| `personal_video_timing_v4` | Does video timing/progress help beyond shared goal geometry? | [Latest study](personal_video_timing_v4/README.md) |

The earlier simple-transfer implementation lives in `src/`, `scripts/` and `configs/`. [Its study guide](../docs/studies/SIMPLE_TRANSFER.md) explains how it led to these experiments.

Start execution through the [quickstart](../docs/QUICKSTART.md) or [training guide](../docs/TRAINING.md). Individual preparation scripts may require original recordings or refuse an already frozen output.
