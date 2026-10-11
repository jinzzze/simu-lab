# Learning robot commands

[Study progression](../OVERVIEW.md) · [All experiments](../EXPERIMENTS.md) · [Current recurrent method](../METHOD.md)

The visual experiment handed a predicted position to a fixed controller. This extension instead trained a compact Diffusion Policy to produce robot command sequences for the same simple-transfer task.

## The first learned controller

DP-D uses the frozen D_seed7 visual model to estimate initial object XY. Its observation also includes robot proprioception, the previous command and an episode clock. The policy sees two recent observations, predicts sixteen commands and executes eight before replanning. It does not relocalize the object from new RGB during the trial.

The action labels come from 128 simulated expert training trajectories plus sixteen validation trajectories. A compact temporal U-Net learns action diffusion; twenty DDIM steps generate commands. Three fixed seeds train for 5,000 updates and use final EMA checkpoints.

| Seed | Strict successes | Qualified lifts |
|---|---:|---:|
| 7 | 27/64 | 33/64 |
| 17 | 25/64 | 30/64 |
| 27 | 32/64 | 39/64 |
| **Total** | **84/192** | **102/192** |

The test reuses 64 scenes from earlier development. Extra demonstrations, different budgets and a different controller prevent interpreting the gap from the fixed controller as an isolated algorithm effect. Two initial test images also occur in training/validation; the original report retains those runs and provides a sensitivity check.

## Toward the four-goal task

The later recurrent policy reads current RGB throughout execution and receives a requested goal. Its causal memory replaces the explicitly supplied episode clock used by DP-D. Development explored action representation, history, coordinate precision and color augmentation before the final paired comparison.

The directory `paired_two_board_v2` retains that development implementation. The combination-coverage and personal-video timing studies build on its recurrent learner. The [code map](../CODE_MAP.md) identifies these dependencies without requiring readers to reconstruct the development filenames.

## Evidence

- [Frozen diffusion configuration](../../configs/diffusion_policy.json)
- [All diffusion results](../../artifacts/reports/diffusion_policy_v1/results.json)
- [First recorded seed-7 case](../../artifacts/runs/diffusion_policy_grasp_v1/DP_seed7/first_scene_demo.mp4)
- [Action-model implementation](../../src/policy/diffusion.py)
- [Four-goal combination coverage](../../experiments/cross_goal_composition_v3/COVERAGE_REPORT_EN.md)

DP-D is a compact state-conditioned extension, not an exact reproduction of the full published Diffusion Policy benchmark. All learned-policy execution in this project is simulation-only.
