# Quickstart: run the four-goal policy

These commands target the **v1.1.0 four-goal experiment**, not the older black-line task. Run PowerShell from your chosen project directory; no particular drive letter is required.

[Documentation](README.md) · [Method](METHOD.md) · [Troubleshooting](REPRODUCTION.md)

The default path below uses the 12.8 MB replay ZIP. For the earlier transfer task, use the [simple-task guide](SIMPLE_TASK_REPRODUCTION.md).

## 1. Download the replay package

From [release v1.1.0](https://github.com/jinzzze/simu-lab/releases/tag/v1.1.0), download `SHA256SUMS.txt` and either:

- **`simu-lab-v1.1.0-replay.zip`**: two seed-7 checkpoints, the first development case for each condition, required frozen code/configuration, environment files and this guide. Use this for a quick execution check.
- **`simu-lab-v1.1.0-evidence.zip`**: all six checkpoints, 60 teacher trajectories, 264 evaluation traces, 52 source clips (eight selected S01 plus 44 X01/X02/X03 clips), processed labels and audits. Use this for complete inspection or training preparation. Earlier v1.0.0 experiments have separate assets.

Compare the selected ZIP's hash with its exact filename in the checksum file:

```powershell
Get-FileHash -Algorithm SHA256 .\simu-lab-v1.1.0-replay.zip
Expand-Archive -LiteralPath .\simu-lab-v1.1.0-replay.zip -DestinationPath .\simu-lab-replay
Set-Location -LiteralPath .\simu-lab-replay
```

Use a new destination directory. The full evidence ZIP is self-contained for the current study; it need not be merged with the old release. A Git clone contains code and reports, but not the checkpoints/raw data required for replay. The supplied ZIP code has exact historical hashes; preserve its bytes.

## 2. Install the runtime

Prerequisites: Conda, Windows and an NVIDIA CUDA-capable GPU with a compatible driver. The recorded setup used an RTX 3080 Ti Laptop GPU (16 GB), Python 3.11 and CUDA 12.6 wheels. CPU-only inference is not implemented. Installation requires network access; the runtime is not bundled.

```powershell
. ./scripts/Enter-Project.ps1
conda env create --prefix .env --file environment.yml
& ./.env/python.exe -m pip install --only-binary=:all: -r requirements.txt
& ./.env/python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
& ./.env/python.exe -B scripts/verify_timing_package.py
```

If `.env` is already installed, skip the three installation commands. Keep PyBullet owned by conda-forge; do not replace it with a pip source build. Enter-Project changes only the current process's environment and creates local cache/output directories. Never set `KMP_DUPLICATE_LIB_OK`: simulation and Torch inference are already isolated in separate processes.

The package verifier checks every supplied file against `PACKAGE_MANIFEST.json`. It does not load model weights. Extra runtime/output files created after extraction are allowed. An earlier clean installation was checked on this same host; the v1.1.0 replay audit uses an existing independent environment. No new-machine installation is claimed.

## 3. Execute two fixed cases

```powershell
& ./.env/python.exe -B -u experiments/personal_video_timing_v4/replay.py --condition geometry_only_24 --seed 7 --suite validation --tag reviewer_geometry_01
& ./.env/python.exe -B -u experiments/personal_video_timing_v4/replay.py --condition video_timing_24 --seed 7 --suite validation --tag reviewer_video_01
```

Each command executes the suite's first predeclared development case, P1–G1, then checks every input/action array and the entire scoring trace against the original run. It also writes and decodes the replay video. **Expected: `all_arrays_identical: true` and `full_scoring_trace_identical: true`; strict robot success is false for both cases.**

Outputs are under `experiments/personal_video_timing_v4/artifacts/replays/<tag>/`: `manifest.json`, `replay.mp4`, `inputs_actions.npz` and `scoring.json`. Use a new tag on every run; existing output directories intentionally refuse overwrite. Replays add zero samples to the reported benchmark.

Exact simulation equality may depend on the runtime/platform. If it differs, preserve the output for diagnosis; do not silently change the references. A reproducible failure demonstrates the execution path, not reliable control performance.

## Explore further

The standalone [92-second video](https://github.com/jinzzze/simu-lab/releases/download/v1.1.0/evidence_walkthrough.mp4) needs no installation. [Results and limitations](https://github.com/jinzzze/simu-lab/blob/v1.1.0/experiments/personal_video_timing_v4/RESULTS_EN.md) include all seeds and failures. For training, use the full evidence ZIP and [the fresh-workspace recipe](https://github.com/jinzzze/simu-lab/blob/v1.1.0/docs/TRAINING.md).

Neither replay package needs the third-party MediaPipe model or ImageNet checkpoint. To rerun original video candidate extraction from the full package, run `scripts/fetch_external_models.py`; it downloads the documented upstream files and verifies their hashes. Third-party base weights are not bundled in either v1.1.0 ZIP.
