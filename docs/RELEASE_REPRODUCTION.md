# Release reproduction

## Candidate archive layout

Extract `simu-lab-source-report.zip` into a new folder. It contains the source, reports, demo, manifests and non-weight run evidence. Extract `simu-lab-inputs-checkpoints.zip` into the **same** folder to supply the exact recorded input data and all saved experiment weights. This second archive is a local publication candidate, not an assertion of public licensing. No installed Python environment is copied into either archive. Use `SHA256SUMS.txt` to check both archives and `scripts/verify_release.py` to verify the extracted files against `release_files.json`.

## Windows 64-bit installation

Use an E-drive working folder and an existing conda executable. `configs/conda-win-64-explicit.txt` pins the original conda builds and checksums; `configs/pip-win-64-lock.txt` pins pip-installed packages. The CUDA packages use the separate official index.

```powershell
# Start in the extracted repository; replace the conda executable if necessary.
. .\scripts\Enter-Project.ps1
conda create --prefix .env --file configs\conda-win-64-explicit.txt -y
.\.env\python.exe -m pip install --only-binary=:all: --no-deps -r configs\pip-win-64-lock.txt
.\.env\python.exe -m pip install --only-binary=:all: --no-deps torch==2.10.0+cu126 torchvision==0.25.0+cu126 --index-url https://download.pytorch.org/whl/cu126
.\.env\python.exe -m pip check
.\.env\python.exe scriptsetch_external_models.py
.\.env\python.exe scriptserify_release.py
.\.env\python.exe scripts
un_tests.py
.\.env\python.exe scripts\check_environment.py --output artifacts\diagnostics
ew_install
```

Conda installation is a prerequisite, not bundled. The locks are Windows-specific; no Linux/macOS installation is claimed. Separate-process tests and inference are required on the tested Windows native library stack. Do not set KMP_DUPLICATE_LIB_OK.

## Verify existing results before retraining

```powershell
.\.env\python.exe scriptsnalyze_visual_ablation.py
.\.env\python.exe scriptsnalyze_reference.py
.\.env\python.exe scriptsnalyze_low_data.py
.\.env\python.exe scriptsnalyze_diffusion_policy.py
.\.env\python.exe scripts\export_grasp_world_model_predictions.py
.\.env\python.exe scripts
eport_grasp_world_model.py
.\.env\python.exe scriptsuild_project_report.py
```

These regenerate derived reports from saved inputs and checkpoints; they do not constitute a fresh training run. The SHA manifest describes the archive at release time, so derived reports may change after regeneration. Historical training-source hashes remain historical and must not be rewritten to make a check pass.

For a new primary A/B/C/D training and simulation run use `scripts/Run-PilotExperiment.ps1 -RunName repeat01`, as described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The other extensions still have documented fixed output paths and refuse existing completed training outputs. The clean-install receipt states exactly which tests, analysis commands and smoke episodes were run; do not interpret a smoke episode as repeating the full 2,112 learned-control evaluations.

## Measured release checks

The fresh same-host environment passed pip check, 39 tests, CUDA/MediaPipe/physics checks, and the fixed-controller first-scene smoke. In a relocated copy, primary/reference/low32/DP reports and world-model predictions were regenerated; recorded metrics matched. DP seed 7 scene 30000 was executed again successfully. The world-model first-training -> verified export -> report chain also passed in a separate folder. Original full visual/DP training was not repeated. See [receipt](../artifacts/diagnostics/clean_install_v1/receipt.json).
