# Evidence index

[Documentation](README.md) · [Experiments](EXPERIMENTS.md) · [Reproduction notes](REPRODUCTION.md)

Read a study report first, then use the records below to inspect its data, commands and scores. The two release versions preserve successive stages of the same project.

## Four-goal personal-video comparison — v1.1.0

| Evidence | Location |
|---|---|
| All seeds, failures and budgets | [Results report](../experiments/personal_video_timing_v4/RESULTS_EN.md) |
| Per-case outcomes and source hashes | [Results JSON](../experiments/personal_video_timing_v4/artifacts/results_v1/results.json) |
| Predeclared comparison | [Study configuration](../experiments/personal_video_timing_v4/configs/study_v1.json) |
| Personal sources and profile constraints | [Human profiles](../experiments/personal_video_timing_v4/data/processed/human_profiles_v1.json) |
| Video information changes executed commands | [Intervention audit](../experiments/personal_video_timing_v4/artifacts/intervention_audit_v1/REPORT_EN.md) |
| Initial-image overlap | [Sensitivity results](../experiments/personal_video_timing_v4/RESULTS_EN.md#duplicate-pixel-sensitivity-post-hoc) |
| Same-host portable replay | [Receipt](../experiments/personal_video_timing_v4/receipts/portable_replay_v1/receipt.json) |
| Anonymous public-download verification | [Public receipt](PUBLIC_DELIVERY_VERIFICATION.json) |
| Raw clips, all models and full traces | [Release assets](https://github.com/jinzzze/simu-lab/releases/tag/v1.1.0) |

The full evidence ZIP contains 52 source clips, six trained models, sixty teacher trajectories and 264 validation/test execution traces. The smaller replay ZIP contains two seed-7 models and two fixed development references. Both include manifests; use `SHA256SUMS.txt` to check the selected download.

X03 is a separate frozen video-label coverage check, not a robot-success benchmark. Object candidates occur in 2,637/3,205 frames (82.3%). Coverage does not measure accuracy. X03 was previously viewed for naming/QC and was not used to train or tune these policies.

## Earlier stages — v1.0.0

| Study | Record |
|---|---|
| Visual auxiliary supervision | [Results](../artifacts/reports/visual_ablation_v1/results.json) and [study explanation](studies/SIMPLE_TRANSFER.md) |
| Fixed-skill outcome prediction | [Results](../artifacts/reports/grasp_world_model_v1/results.json) and [study explanation](studies/OUTCOME_PREDICTION.md) |
| Learned diffusion commands | [Results](../artifacts/reports/diffusion_policy_v1/results.json) and [study explanation](studies/LEARNED_ACTIONS.md) |
| Clean-install and reproduction scope | [Receipt](../artifacts/diagnostics/clean_install_v1/receipt.json) |
| Inputs, checkpoints and frozen source | [Release assets](https://github.com/jinzzze/simu-lab/releases/tag/v1.0.0) |

## Reading historical records

Experimental manifests describe the moment a run was frozen. A field such as `publicly_published: false` in an old video manifest does not describe today's release status. Current access instructions are in the [quickstart](QUICKSTART.md); immutable run files remain useful for provenance.

The current English guides organize the evidence. [Historical documents](archive/README.md) preserve the original wording, development decisions and publication receipts at a pinned commit. Machine-readable experiment results, configurations and frozen executable sources are retained unchanged by this presentation update.

## Presentation-update verification

The [verification receipt](PRESENTATION_VERIFICATION.json) records the documentation audit and three fixed-case executions checked during the English presentation update. These checks add no benchmark samples.
