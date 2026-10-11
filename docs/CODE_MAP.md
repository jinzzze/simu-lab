# Code map

[Documentation](README.md) · [Method](METHOD.md)

Start from a task entry point, then follow its imports. The current four-goal learner reuses components developed in the earlier paired-task directory; versioned directory names identify that provenance.

| Location | Responsibility |
|---|---|
| `src/data/` | Earlier personal-video labels |
| `src/perception/` | Visual encoders, auxiliary losses and training |
| `src/sim/` | Shared Panda contact simulation |
| `src/world_model/` | Fixed-skill state prediction and evaluation |
| `src/policy/` | Earlier diffusion command learner and runtime bridge |
| `scripts/` | Original-stage training/evaluation and release utilities |
| `configs/` | Original-stage experiment settings and environment locks |
| `experiments/paired_two_board_v2/` | Four-goal environment, scoring, recurrent learners and development revisions |
| `experiments/cross_goal_composition_v3/` | Start/goal coverage comparison and new-video processing |
| `experiments/personal_video_timing_v4/` | Final paired timing/progress experiment |
| `tests/` | Shared runtime, data-boundary and model checks |
| `artifacts/` | Recorded metrics, source snapshots and selected media |

## Latest study: follow one run

1. [Configuration](../experiments/personal_video_timing_v4/configs/study_v1.json) fixes the conditions, pairs and evaluation plan.
2. [Collection](../experiments/personal_video_timing_v4/collect.py) records physically executed teacher demonstrations.
3. [Training](../experiments/personal_video_timing_v4/train.py) fits the recurrent policy under the frozen budget.
4. [Evaluation](../experiments/personal_video_timing_v4/evaluate.py) runs it against the simulator and separate scorer.
5. [Analysis](../experiments/personal_video_timing_v4/analyze_results.py) checks traces and aggregates all seeds.
6. [Replay](../experiments/personal_video_timing_v4/replay.py) executes a fixed recorded case and compares arrays and scores.

Use the packaged [quickstart](QUICKSTART.md) rather than running these stages out of order. Several stages intentionally refuse to overwrite completed output directories. The source checkout does not supply the data and weights needed for execution.

Torch and Bullet are kept in separate processes because their native OpenMP runtimes conflict on the recorded Windows setup. This separation is part of the execution design.
