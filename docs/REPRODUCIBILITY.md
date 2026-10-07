# Reproducing the pilot experiment

This Windows prototype uses a local conda environment and an NVIDIA CUDA GPU. All project files, caches and temporary outputs are under `E:\HumanoidRobotLearning`. A fresh independent environment on the same Windows computer now passes dependency checks, 39 tests, CUDA/vision/physics checks and one fixed-controller grasp. See [clean-install receipt](../artifacts/diagnostics/clean_install_v1/receipt.json). A different machine and full training reproduction remain untested.

## Existing local data and results

In PowerShell:

```powershell
Set-Location -LiteralPath E:\HumanoidRobotLearning
. .\scripts\Enter-Project.ps1
.\.env\python.exe scripts\run_tests.py
.\.env\python.exe scripts\analyze_visual_ablation.py
```

The analysis command requires every primary run to be complete, verifies checkpoint/data/source lineage and paired inputs, and fails instead of quietly reporting partial results. The tests run suites in isolated processes because the full Torch and Panda/Bullet stacks have incompatible OpenMP runtimes on this machine. The live evaluation uses the same process boundary, not an unsafe duplicate-runtime environment override.

To repeat the complete 4 × 3 experiment without overwriting the original results:

```powershell
.\scripts\Run-PilotExperiment.ps1 -RunName repeat01
```

This uses the already prepared shared NPZ files and creates new pretraining, adaptation, physical evaluation and report directories in `artifacts/reproductions/repeat01`. Choose a fresh name for a new experiment. Training does not overwrite an existing completed checkpoint.

The primary experiment is fixed in `configs/development_run_plan.json`: A/B/C/D, seeds 7/17/27, 240 pretraining steps, 300 adaptation steps, batch 16, 64 held-out simulation scenes per model. There are 768 physical episodes, not 768 distinct scene layouts. All models use the same 64 initial-state images. The final checkpoint is evaluated; there is no best-checkpoint selection.

## Data derivation

Read `docs/PILOT_LABEL_REPORT.md` before interpreting pseudo labels. `scripts/inspect_pilot.py` checks video metadata/decoding, ArUco and sampled hands. `scripts/label_pilot.py` constructs per-frame labels and review media. `scripts/prepare_pilot_pretraining.py` exports unannotated full RGB frames and independent auxiliary masks. `scripts/generate_sim_dataset.py` creates distinct seeded train/validation/test scenes.

The current pilot has one conservative real development scene family. Adjacent frames, copies, annotations and re-encodings must not be assigned independent real train/test identities. The duplicate `pilot_originals` directory is not extra data. Sparse visual approval is for exploratory development only, not certified label accuracy.

Human videos were personally supplied by the applicant. No public download is currently available; code alone cannot reproduce these specific results without the exact local input data. Do not substitute synthetic images and present their outcomes as the personal-data experiment. Raw video publication and data licensing remain applicant decisions.

## Environment

See `environment.yml`, `requirements.txt` and `docs/ENVIRONMENT.md`. Install PyBullet with conda-forge; do not replace it with a pip source build. PyTorch/CUDA packages are installed separately from the official cu126 wheel index. Model URLs and hashes are in `configs/model_sources.json`; the moving MediaPipe `latest` URL must still match the recorded hash for exact replication.

The predictor takes uint8 HWC RGB, resizes the entire image with OpenCV INTER_AREA, normalizes using ImageNet statistics, and returns XY metres. Hand/object auxiliary heads are discarded. Simulator truth never enters the prediction worker; only image bytes cross its process boundary. Physics engine state is used for training labels and independent scoring.

## Known reproduction boundaries

- GPU/driver/library differences may change numerical details. Paired fairness within the recorded environment does not guarantee bitwise equality on another platform.
- Native OpenMP isolation is part of the current Windows runtime architecture.
- Only object position varies in the present held-out simulation scenes; camera, lighting, yaw, background and material variation are not evaluated.
- The primary visual ablation uses a fixed controller. A separate learned macro-action outcome predictor is implemented; it does not replace that controller or provide a VLA or demonstrated planning improvement.
- The simulation-data source hash predates a scoring-only strengthening of bilateral contact checks; rendering/configuration and generated RGB stayed unchanged. The analysis independently compares each evaluation RGB hash with its stored dataset image.

## Independent macro-action world-model extension

This extension has its own simulator transitions, seeds, manifests and outputs. It does not modify the 1,920 A/B/C/D/R visual-control executions. [GRASP_WORLD_MODEL.md](GRASP_WORLD_MODEL.md) documents the 32-scene/160-transition prediction test, the preselected RGB execution, and the subsequent paired RGB-input diagnostic.

The original three final checkpoints survived a native runtime failure during reporting. Recovery loaded those saved weights and took **zero additional optimizer steps**. The original training source is preserved at `artifacts/runs/grasp_world_model_v1/source_snapshots/train_grasp_world_model_before_process_fix.py`; the training manifest records that source hash. Current training launches reporting in a separate process, and the report records postprocessing source hashes.

To regenerate predictions and derived metrics from the existing completed checkpoints:

```powershell
.\.env\python.exe scripts\export_grasp_world_model_predictions.py
.\.env\python.exe scripts\report_grasp_world_model.py
```

These are fixed-path commands, with no output-directory option. Export rewrites `torch_predictions.npz` and `run_state.json`, without training or replacing checkpoints. Reporting checks NumPy/Torch agreement and rewrites derived metrics, prediction artifacts and plots. It is therefore an intentional regeneration of derived outputs, not a fresh independent run.

First-run collection, training, demonstration and RGB diagnostic entry points are:

```powershell
.\.env\python.exe scripts\generate_grasp_world_model.py
.\.env\python.exe scripts\train_grasp_world_model.py
.\.env\python.exe scripts\demo_grasp_world_model.py
.\.env\python.exe scripts\evaluate_world_model_rgb.py
```

These commands also use fixed project paths. Collection refuses an existing manifest or split log; training refuses an existing run directory; demo refuses an existing `integration/result.json`; RGB diagnosis refuses an existing `rgb_evaluation/results.json`. Those outputs already exist here. Do not delete recorded evidence to make a rerun work; a fresh-run path interface would need to be added before repeating these stages independently.

Simulation and live RGB inference remain in separate processes. The macro model has a NumPy export usable in the simulator process; its measured maximum differences from Torch were 2.98e-8 m in displacement and 1.19e-7 in probability. No duplicate OpenMP runtime suppression is used.

The RGB batch diagnostic rerenders the original 32 test scenes, verifies exact initial RGB hashes, and reuses all 160 original absolute commands and stored outcomes. Those commands were generated from privileged state plus predefined offsets at collection time. It performs no new training or physical executions and does not evaluate a policy choosing commands from RGB. Its 153/160 success-label accuracy must not be reported as robot success rate.

Public reproduction still requires a release decision for real inputs, processed data and weights, plus public access verification; same-host clean installation now passes the recorded checks. The world-model transition dataset is synthetic and must never be presented as the applicant's personally recorded data.

## Diffusion Policy extension

The separately trained action policy has a fixed plan in `configs/diffusion_policy.json`. Existing demonstrations and final checkpoints are local. `collect_diffusion_demonstrations.py` and `train_diffusion_policy.py` protect existing output directories. `evaluate_diffusion_policy.py` checks source/config/checkpoint identity before resuming missing scene rows; it saves every action chunk and effective command. `analyze_diffusion_policy.py` requires all three complete runs and audits paired scene images and data lineage.

The two additional test suites, `tests/test_diffusion_policy.py` and `tests/test_policy_environment.py`, must be run in separate Python processes due to native-library isolation. `record_diffusion_failures.py` replays the first failed recorded action stream per seed, verifies the physical outcome and does not create additional independent evaluation samples. The stored `--limit` evaluation prefix is diagnostic only and is excluded from the full report.

See [DP-D scope and commands](DIFFUSION_POLICY.md). Recompute its report with `.\.env\python.exe scripts\analyze_diffusion_policy.py`, then rebuild the combined local page with `.\.env\python.exe scripts\build_project_report.py`.


## Release preparation update (2026-10-07)

Independent real-scene validation is deferred by the applicant. Local source/report and input/checkpoint release candidates, an English application draft, a 112-second captioned demonstration and a 48-frame blinded development-label review are prepared. Human box/keypoint accuracy is not yet measured. Same-host clean installation passes the recorded checks; publication, final license/data permissions, applicant identity and the actual application remain pending. See [release checklist](RELEASE_CHECKLIST_ZH.md) and [archive reproduction](RELEASE_REPRODUCTION.md).


## Public release addendum

The applicant approved publication of all original videos, processed data and checkpoints on 2026-10-07. Repository: https://github.com/jinzzze/simu-lab ; artifact release: https://github.com/jinzzze/simu-lab/releases/tag/v1.0.0 . Earlier local-only descriptions record the pre-release state. Final application submission and a blanket original-source license grant remain separate steps.
