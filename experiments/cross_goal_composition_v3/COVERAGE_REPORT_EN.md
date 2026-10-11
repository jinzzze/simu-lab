# Earlier four-goal coverage comparison

This exploratory experiment preceded the personal-video timing comparison. Each condition uses 24 robot demonstrations: either four same-number pairs repeated six times or twelve pairs repeated twice. Initial scenes, start/goal marginal frequencies, recurrent learner and 2,500-update budget are paired across three seeds. Trajectory lengths and frame-target counts differ. Both conditions share geometry from S01; this study does not test the new X01 video timing/progress intervention.

| Test category | Same-number training | 12-combination training |
|---|---:|---:|
| Same-number | 22/24 | 5/24 |
| Additional covered | 0/48 | 8/48 |
| Held-out | 0/24 | 2/24 |
| All | 22/96 | 15/96 |

Increasing coverage did not improve overall success. Each condition has 32 tasks repeated for three seeds, using eight physical starts. One initial test-task image per condition matches development pixels despite different scene IDs; no main outcomes were removed. The small held-out count does not demonstrate reliable compositional generalization.

All six models and 264 validation/test executions are summarized in the [frozen result file](artifacts/results_v1/results.json). This selected historical report is provided alongside v1.1.0; its complete raw trajectory/checkpoint archive is not part of the v1.1.0 download, which supplies the complete later personal-video timing study. Results from these two studies must not be pooled. Teacher-forced errors, pixel-overlap checks, seeds, budgets and source references remain in the recorded result file.

A separate waypoint-label development pilot failed its declared expansion gate (seed 7: 0/12 strict successes, 2/12 lifts) and is not counted in the 264 primary executions. No new training was performed for publication.
