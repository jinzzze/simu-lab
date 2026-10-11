# Reproduce the simple-transfer stage

[Study](studies/SIMPLE_TRANSFER.md) · [Current four-goal quickstart](QUICKSTART.md) · [Reproduction scope](REPRODUCTION.md)

This route uses **v1.0.0**, which contains the visual, fixed-skill prediction and early diffusion-policy experiments. The four-goal packages use a separate route.

## 1. Download the matching files

From [release v1.0.0](https://github.com/jinzzze/simu-lab/releases/tag/v1.0.0), download `simu-lab-source-report.zip`, `simu-lab-inputs-checkpoints.zip` and `SHA256SUMS.txt`. Compare each ZIP's SHA-256 with its entry. Extract both ZIPs into the same new directory. Run the following commands from that directory.

```powershell
Get-FileHash -Algorithm SHA256 ./simu-lab-source-report.zip
Get-FileHash -Algorithm SHA256 ./simu-lab-inputs-checkpoints.zip
Expand-Archive -LiteralPath ./simu-lab-source-report.zip -DestinationPath ./simu-lab-simple
Expand-Archive -LiteralPath ./simu-lab-inputs-checkpoints.zip -DestinationPath ./simu-lab-simple
Set-Location -LiteralPath ./simu-lab-simple
```

## 2. Install the recorded Windows environment

Prerequisites are Conda and an NVIDIA CUDA GPU. The original GPU was an RTX 3080 Ti Laptop GPU with 16 GB VRAM. The environment is not bundled. The lock files are Windows-specific.

```powershell
. ./scripts/Enter-Project.ps1
conda create --prefix .env --file configs/conda-win-64-explicit.txt -y
& ./.env/python.exe -m pip install --only-binary=:all: --no-deps -r configs/pip-win-64-lock.txt
& ./.env/python.exe -m pip install --only-binary=:all: --no-deps torch==2.10.0+cu126 torchvision==0.25.0+cu126 --index-url https://download.pytorch.org/whl/cu126
& ./.env/python.exe -m pip check
& ./.env/python.exe scripts/verify_release.py
```

If the environment already exists, skip installation. Keep Bullet under conda-forge and use separate-process entry points. Original preprocessing additionally requires the documented external model downloads via `scripts/fetch_external_models.py`.

## 3. Run a fixed-controller case

Choose an unused output directory:

```powershell
& ./.env/python.exe scripts/evaluate_visual_grasp.py --groups D --seeds 7 --limit 1 --record-first --output artifacts/demos/reviewer_simple_01
```

This runs the first test scene with the D_seed7 visual model and the fixed controller, records video, and saves an execution summary below the chosen directory. The recorded first-scene reference succeeded. A one-scene replay is not a repeat of the full benchmark; inspect the new summary for the actual outcome.

## Inspect or regenerate results

The archives already include reports. To regenerate the original visual summaries in a disposable extracted copy:

```powershell
& ./.env/python.exe scripts/analyze_visual_ablation.py
& ./.env/python.exe scripts/analyze_reference.py
& ./.env/python.exe scripts/analyze_low_data.py
```

These commands update derived reports, not model weights. Verify the archive before regenerating files. The [outcome-prediction](studies/OUTCOME_PREDICTION.md) and [learned-action](studies/LEARNED_ACTIONS.md) guides identify their own artifacts.

Original full visual/diffusion training was not repeated in the release checks. The earlier same-host clean-install receipt records report regeneration, 39 tests and selected execution checks; [evidence index](EVIDENCE.md) gives the exact scope. The current project is frozen for research; these commands are provided for independent reproduction.
