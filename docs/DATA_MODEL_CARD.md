# Data and model card

## Latest four-goal study (v1.1.0)

The applicant personally recorded moving a block between two paper grids. Red P1–P4 are sources and blue G1–G4 are target centers. The measured grids are 100 × 130 mm with a 130 mm edge-to-edge gap. Camera/scene geometry is only approximately mapped to simulation; airborne metric 3D motion and true physical contact labels are unknown.

| Recording block | Collected clips | Use in latest study |
|---|---:|---|
| S01 | 16 same-number clips | First two predeclared repeats per pair, eight clips total, used for training |
| X01 | 16 cross-goal clips | All sixteen used for training |
| X02 | 12 clips | Development of video labels; robot-policy validation uses a separate fixed simulation suite |
| X03 | 16 clips | Separate frozen video candidate-coverage check after all policies were fixed; no policy training or tuning |

The v1.1.0 full package supplies the **52 raw clips referenced by this study**: selected S01 eight plus new X01/X02/X03 forty-four. It does not silently count unselected S01 clips as training data. Earlier twelve-video visual experiments are distributed in v1.0.0.

The applicant reported X01 on 9 October around 20:00 London time, X02 about half an hour later with a small viewpoint change, and X03 about one hour later with another viewpoint change. The reference point for the last interval is ambiguous (approximately 21:00–21:30); no exact timestamp is invented. The recordings share a scene family and are not broad independent environments.

Object candidates and hand candidates are generated independently; invalid labels are masked. Coarse visible event brackets were inspected with Codex assistance, not independently annotated physical events. X03 object candidate coverage is 2,637/3,205 frames (82.3%), **not detection accuracy or task success**. X03 was previously viewed for applicant-requested naming/QC. No claim of a never-viewed real test set is made.

## Learning and information boundaries

Twenty-four training video profiles provide first-frame target geometry for both conditions. One condition additionally uses matching coarse transfer time and dimensionless image progress to shape teacher transport. Teacher actions are actual executed robot commands from contact-based simulation. Personal profiles are not available to the deployed learner.

Each CNN–GRU policy receives current RGB and 12-dimensional robot/goal/previous-command features. The goal is supplied before an episode; runtime object truth, expert phase and external time are excluded. Simulator truth is used for training labels, scoring and explicitly identified diagnostics. Three fixed seeds and final EMA checkpoints are retained for each condition, including failed trajectories and executions.

Strict test scores are 7/96 and 11/96 over 32 tasks and three model seeds, reusing eight physical starts. Four initial test-task images repeat training/development pixels; exclusions are reported separately. There is no statistically established benefit, real-robot transfer result or language policy. See [all results](../experiments/personal_video_timing_v4/RESULTS_EN.md).

## Distribution and provenance

Personal videos, derived data and trained checkpoints are authorized for public distribution. No blanket source, data or model license has been granted. Third-party software retains its own terms; the pretrained MediaPipe and ImageNet model binaries are omitted and fetched separately from upstream with hashes. See [third-party notices](../THIRD_PARTY_NOTICES.md).

Frozen source/configuration manifests, per-episode input/action arrays, scoring traces and checkpoint hashes accompany the full package. `PACKAGE_MANIFEST.json` and `SHA256SUMS.txt` verify files and ZIP downloads. Timestamps/status fields in frozen experimental manifests record their original stage; later publication does not rewrite those records.

The applicant supplied the recordings; Codex assisted with coding, execution, candidate-label inspection and writing. This is research evidence with known limitations, not a production robot policy.

The [v1.0.0 data/model card](DATA_MODEL_CARD_v1.0.0.md) documents the earlier visual, state-prediction and diffusion experiments separately.
