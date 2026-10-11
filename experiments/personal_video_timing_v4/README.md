> Release navigation: [v1.1.0 downloads](https://github.com/jinzzze/simu-lab/releases/tag/v1.1.0) · [Quick replay](../../docs/QUICKSTART.md). Experimental status/provenance below records the original run; the release does not retrain or alter it.

# Personal-video timing ablation

This study tests whether coarse transfer timing and image progress from the applicant's videos help a learned simulated pick-and-place policy beyond the videos' shared goal geometry. Both conditions cover the same 12 P/G combinations with 24 paired robot demonstrations. The four cyclic combinations P1–G2, P2–G3, P3–G4 and P4–G1 are withheld from training.

The task is full-footprint placement inside a prescribed 40 mm square, with qualified lifting, release, withdrawal and stability. Crossing the black line alone is insufficient. The desired goal XY is chosen before the episode and provided to the policy together with current RGB, robot proprioception and the previous command. Runtime object truth, expert stages, external clocks and human trajectories are excluded.

## Data and intervention

- Personal training sources: S01_E01–E08 (first two predeclared repeats of each diagonal pair) and all 16 X01 videos. X02 remains development-only; X03 remains outside this training pipeline.
- Both groups use identical goal geometry, averaged from first-frame blue centers in those same 24 training clips. Mapping to the simulated tabletop is engineered, not calibrated real-robot registration.
- `geometry_only_24`: legacy initial-XY privileged teacher with its ordinary transport.
- `video_timing_24`: the same teacher adds the matching clip's coarse transfer duration and dimensionless image-progress constraints during transport. Other stages remain identical before transport. Robot actions are recorded from executed simulation commands.
- RGB candidates have explicit missing masks. The image-progress curve is not recovered human 3D motion. Interpolation creates desired robot setpoints; it does not fill missing human labels. MediaPipe hand candidates serve label inspection here and are not an additional policy input or training loss.

Each learner uses the same v6 recurrent BC architecture, initialization, paired episode sampling and augmentation, 2,500 updates and fixed-final EMA selection for seeds 7, 17 and 27. Equal demonstration counts and update budgets do **not** imply equal frame targets or compute, since trajectory lengths differ. All teacher failures are retained. Training proceeds only if each group attains 24 qualified lifts and at least 20 strict placements out of 24 teacher attempts.

## Status and evidence

Completed: 60 teacher trajectories, six trained models, 72 validation and 192 test executions, all-seed audits, and actual fixed-case reproduction. The test counts are **7/96 for shared geometry and 11/96 with video timing/progress**; held-out combinations are 1/24 and 2/24. This small observed difference does not establish a reliable or statistically significant improvement. Overall success is low and most failures lack a qualified lift.

Four initial test-task images repeat training/development pixels despite disjoint scene seeds. Post-hoc exclusion gives 5/84 versus 8/84; excluding all goals for the two affected physical starts gives 4/72 versus 7/72. The original results are retained. See [full results and budget](RESULTS_EN.md), [overlap sensitivity](artifacts/initial_overlap_audit_v1/REPORT_ZH.md), [Chinese progress](PROGRESS_ZH.md), and the [92-second evidence walkthrough](https://github.com/jinzzze/simu-lab/releases/download/v1.1.0/evidence_walkthrough.mp4).

![Method](artifacts/figures/method_v1/method.png)

![All-seed results](artifacts/figures/personal_timing_v2/results.png)

The immutable plan is `configs/study_v1.json`; human sources and profile constraints are in `data/processed/human_profiles_v1.json`. The two groups each achieved 20/24 strict teacher placements and 24/24 lifts, passing the predeclared collection gate. Teacher success is separate from learned-policy performance.

Teacher collection, training and evaluation each save manifests, SHA-256 provenance and source snapshots. `audit_demonstrations.py` verifies identical pre-transport RGB/state/actions and that the video intervention actually changes robot command labels. `audit_training.py` verifies all 24 clips reached training and that paired random streams match. `analyze_results.py` audits causal inference traces, goals and fresh scene separation before reporting all seeds and failures.

The test suite has 32 tasks over eight physical initial positions, repeated for three policy seeds. It is not 96 independent environments per condition. X03 was previously inspected for administrative naming/QC at the applicant's request. A separate frozen label-coverage check was run only after all checkpoints were fixed: 2,637/3,205 frames (82.3%) have object candidates. This is coverage, not accuracy or robot success. No X03 information updates these policies. See [the separate X03 report](../cross_goal_composition_v3/artifacts/reports/X03_frozen_labels_v1/REPORT_ZH.md).

Two fixed development cases reproduced with exactly matching input/action arrays and scoring traces, including from a relocated directory with a previously independently installed environment. This is a same-host replay check, not full retraining or another-machine validation. [Exact commands and scope](REPRODUCTION_ZH.md) · [Portable receipt](receipts/portable_replay_v1/receipt.json) · [Application narrative draft](../../docs/SUBMISSION_EN.md).

## Reproduce or train

Use the [quick replay guide](../../docs/QUICKSTART.md) for the two frozen seed-7 cases. For a full repeat, use the full evidence ZIP and [fresh-workspace preparation](../../docs/TRAINING.md). The helper preserves frozen configurations and personal inputs while creating no prior output directories. Do not rerun `prepare_study.py` on supplied immutable configurations. Torch inference and Bullet physics remain in separate processes.

This is an exploratory personal-data contribution comparison informed by earlier experiments. Coarse visual brackets were inspected by Codex, not independently annotated by a human. Camera/segmentation errors, exact physical contact times and airborne metric geometry remain unmeasured. All reported policy performance is simulation-only.
