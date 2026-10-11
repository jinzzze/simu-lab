# Personal video timing: audited results

Both conditions share the same training-only goal geometry, 12 training combinations and 24 paired robot initial scenes. The video condition additionally uses coarse transfer time and dimensionless image-progress constraints from S01(8) and X01(16). Human metric actions are not recovered. Robot actions come from executed teacher commands. All teacher failures are retained. Learner, initialization, episode/augmentation streams and update counts match; frame-target counts and compute differ. All three fixed-final EMA models are evaluated on 32 tasks, reusing eight physical starts across goals and seeds. Dots are observed seed rates, not confidence intervals. Four initial test-task images exactly repeat training/development pixels, involving two physical starts. The full predeclared results are retained and duplicate-exclusion sensitivities are reported separately. This is an exploratory simulation-only ablation informed by prior experiments.

|Test category|Shared geometry|+ Video timing / progress|
|---|---:|---:|
|Same-number|0/24 (0.0%)|3/24 (12.5%)|
|Seen cross-goal|6/48 (12.5%)|6/48 (12.5%)|
|Held-out combinations|1/24 (4.2%)|2/24 (8.3%)|
|All combinations|7/96 (7.3%)|11/96 (11.5%)|

|Condition|Seed|Validation strict|Test strict|Test lifts|Mean final center distance (mm)|
|---|---:|---:|---:|---:|---:|
|Shared geometry|7|2/12|2/32|6/32|194.1|
|Shared geometry|17|1/12|2/32|2/32|202.8|
|Shared geometry|27|1/12|3/32|13/32|123.6|
|+ Personal timing / progress|7|2/12|5/32|9/32|167.6|
|+ Personal timing / progress|17|0/12|2/32|4/32|189.7|
|+ Personal timing / progress|27|2/12|4/32|14/32|122.2|

Observed pooled difference (video − geometry): +4.2 percentage points. This does not establish a generalizable or statistically significant benefit.

Teacher quality (not learned-policy results): {'geometry_only_24': {'episodes': 24, 'successes': 20, 'lifts': 24, 'frames': 6444}, 'video_timing_24': {'episodes': 24, 'successes': 20, 'lifts': 24, 'frames': 6601}}

Data, causal input histories, current-goal consistency, all seeds, and paired optimization streams passed the stored audits.

The full result contains failure flags and first-closure alignment diagnostics. Final center distance is simulator scoring truth, never a runtime observation.

![Audited results](artifacts/figures/personal_timing_v2/results.png)

## Training budget actually consumed

|Condition|Seed|Updates|Episode draws|Valid action targets|Training seconds|
|---|---:|---:|---:|---:|---:|
|geometry_only_24|7|2500|5000|1339946|132.6|
|geometry_only_24|17|2500|5000|1341501|126.4|
|geometry_only_24|27|2500|5000|1340914|125.3|
|video_timing_24|7|2500|5000|1372197|128.2|
|video_timing_24|17|2500|5000|1375003|123.0|
|video_timing_24|27|2500|5000|1373828|125.1|

## Failure diagnostics

Flags may co-occur; their counts are not disjoint failure categories. First-closure alignment uses simulator truth for diagnosis only.

|Condition|Failed executions|No qualified lift|Lift but strict placement fails|Median first-closure XY error among failures|
|---|---:|---:|---:|---:|
|geometry_only_24|89|75|14|25.5 mm|
|video_timing_24|85|69|16|27.2 mm|

Initial-scene overlap audit: {'scene_seed_sets_disjoint': True, 'test_physical_starts': 8, 'test_tasks_with_initial_pixels_in_training': ['P3_G2_r1', 'P4_G4_r2'], 'test_tasks_with_initial_pixels_in_development': ['P3_G1_r1', 'P3_G3_r1']}

[Method diagram](artifacts/figures/method_v1/method.svg) · [Actual video-to-command intervention](artifacts/intervention_audit_v1/REPORT_EN.md) · [Reproduction](REPRODUCTION_ZH.md)

## Duplicate-pixel sensitivity (post hoc)

|Exclusion rule|Shared geometry|+ Video timing / progress|
|---|---:|---:|
|Four identical initial task images|5/84|8/84|
|All goals for two affected physical starts|4/72|7/72|

These descriptive exclusions were introduced after the overlap audit, not before the experiment. They add no executions and do not replace the primary table. See [matching cases and all seeds](artifacts/initial_overlap_audit_v1/REPORT_ZH.md).
