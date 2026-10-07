# Visual training implementation

This is a development ablation, not evidence yet that auxiliary supervision improves task success. All 12 personal pilot videos are treated as one conservative development scene family. They are used for pretraining, without pretending that randomly separated neighbouring frames form a real held-out evaluation set.

## Shared protocol

`configs/visual_training.json` defines fixed optimization budgets. Each paired seed uses the same ImageNet ResNet-18 initialization, auxiliary-head initialization, frame order, teacher targets and photometric random stream. A/B/C/D instantiate the same architecture. A enables no auxiliary loss, B object only, C hand only, D both. The existing YAML group switches are checked before training.

The frozen initial teacher sees the original full RGB frame. The student sees only mild brightness, contrast and saturation changes; no label-based crop or geometry change is applied. The base loss is squared Euclidean distance between L2-normalized pooled features. Auxiliary losses are masked Smooth L1 on normalized box `(cx, cy, w, h)` and 21 hand XY points. Both weights are initially 1.0 and losses are logged separately. Invalid annotations do not remove the RGB frame; fully masked batches contribute a differentiable zero auxiliary loss. Nonfinite masked labels cannot poison training.

Defaults: 240 pretraining steps, batch 16, AdamW at 1e-4, weight decay 1e-4. The final checkpoint is used, without selecting a favourable validation/test checkpoint. Each manifest records data SHA-256, source code hashes, GPU/library versions, initial-model hash, batch-order hash, augmentation RNG hash and measured run time. Output files refuse to overwrite an existing completed checkpoint.

## Inputs and commands

Run PowerShell from `E:\HumanoidRobotLearning`:

```powershell
. ./scripts/Enter-Project.ps1
./.env/python.exe scripts/run_tests.py
./.env/python.exe scripts/train_visual_ablation.py --data data/processed/pilot_pretraining_v1.npz --seeds 7
./.env/python.exe scripts/adapt_visual_state.py --data data/processed/sim_adaptation_v1.npz --seeds 7
```

Additional paired seeds can be run identically with `--seeds 17 27`. Changing any budget is a new shared development run and should use a new `--output` directory for all groups, with changes reported. Do not rerun only the weakest/strongest group or call a revised development benchmark an untouched final test.

Real NPZ keys: uint8 RGB `images[N,224,224,3]`, float `object_targets[N,4]`, bool `object_mask[N]`, float `hand_targets[N,42]`, bool `hand_mask[N,21]`, string `episode_ids[N]`, integer `frame_indices[N]`. Duplicate episode/frame pairs are rejected.

Simulation NPZ keys: the same RGB image format, float `target_xy_m[N,2]`, `scene_seeds[N]`, string `split[N]` with `train`, `val`, optional `test`. Scene IDs cannot cross splits. Training normalization uses training labels only. Validation and optional test metrics are evaluated only after fixed-budget optimization and are not used to select checkpoints. Source manifests must additionally document split seed ranges, rendering distribution and provenance; the loader cannot certify that distinct IDs correspond to truly independent scenes.

## Simulation adaptation and deployment

Discard both human auxiliary heads. Transfer only the encoder to a model with a newly initialized identical XY head. This head preserves the ResNet layer4 spatial feature map (512×7×7): 1×1 convolution to 32 channels, flatten, linear 128, then XY. This choice was fixed before any simulation metrics were seen, because global pooling discards the spatial layout useful for localization. All groups use the same simulation images and fixed 300-step adaptation budget, with full encoder fine-tuning and mild photometric augmentation. Mean/scale of XY is computed from training positions only. Outputs are world XY metres; no airborne height, yaw or human action is inferred.

```python
from src.perception import load_state_predictor
predict_xy = load_state_predictor(
    r"E:\HumanoidRobotLearning\artifacts\runs\visual_state_v1\A_seed7\checkpoint.pt"
)
xy_metres = predict_xy(rgb_uint8_hwc)
```

The callable accepts RGB only and uses full-frame OpenCV `INTER_AREA` resizing to 224. The simulation renderer/data generator must use the same color and resize convention. Controller proprioception, fixed known object height and task line geometry remain shared. True object coordinates may create labels and score outcomes, but must not replace this prediction during the primary controller run.

Perception metrics (mean, median, p90 XY error in millimetres) are saved separately from robotic task success. A low visual loss or successful code smoke test is not a successful grasp. Small object size after full-frame resizing, heuristic box pseudo labels, only one real scene family and the limited simulation distribution are explicit study limitations.

## Native runtime integration

In this Windows environment, the full Panda/Bullet and Torch stacks together abort with OpenMP Error 15 (`libomp.dll` versus `libiomp5md.dll`). The evaluator uses `src/predictor_process.py` and the persistent `scripts/rgb_predictor_worker.py`. Only uint8 RGB crosses into the worker; only predicted XY returns. No scene identifiers, labels or scoring state enter this interface. Deterministic kernels and disabled TF32 match adaptation evaluation.

Use `scripts/run_tests.py` for separate test processes. The live Bullet RGB → Torch predictor integration test checks equivalence with direct model inference. Do not suppress the runtime conflict with `KMP_DUPLICATE_LIB_OK`.
