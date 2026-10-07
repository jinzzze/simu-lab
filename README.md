# Hand–Object Supervision for Robot Learning

A controlled pilot study: can object-box and hand-keypoint supervision from personally recorded phone videos improve visual localization and physical grasp success in simulation?

**Result:** adding both auxiliary targets did not improve task success in this pilot. All groups saturated the 256-image adaptation benchmark. With 32 adaptation images, the combined group scored 181/192 versus 184/192 for RGB-only. This pilot provides no evidence of a success-rate gain from the combined targets. The small observed gap does not establish statistically significant degradation.

The applicant supplied 12 videos of grasping a 20 × 20 × 10 mm black block and placing it beyond a black line. The system uses those real images to update a visual encoder, adapts it to simulated RGB, and drives a Panda arm through actual PyBullet contact dynamics. It does not attach or teleport the object during execution.

The reported A/B/C/D/R experiments implement visual representation transfer and a fixed grasp controller. They do not train a VLA or imitate human action sequences. A separate macro-action world-model extension is now complete and reported below; it does not change those experiments. The original pushing/MPC proposal remains historical design.

## Repository and recorded artifacts

Public repository: [jinzzze/simu-lab](https://github.com/jinzzze/simu-lab). The complete source/report archive and personal-data/checkpoint archive are provided through [release v1.0.0](https://github.com/jinzzze/simu-lab/releases/tag/v1.0.0). Extract both into one folder to reproduce the recorded study; verify the accompanying SHA256 checksums. [Submission draft](docs/SUBMISSION_EN.md) · [Release reproduction](docs/RELEASE_REPRODUCTION.md) · [Data/model card](docs/DATA_MODEL_CARD.md) · [Release checklist](docs/RELEASE_CHECKLIST_ZH.md) · [Third-party notices](THIRD_PARTY_NOTICES.md).

Independent real-scene validation is deferred at the applicant's request (2026-10-07). All existing phone clips remain development data. Human pseudo-label accuracy is still unmeasured; a blinded review set is prepared. These limits do not change the recorded simulation results.

## Measured results

Three fixed training seeds (7, 17, 27), 64 identical simulation test scenes per seed and group. Each row therefore has 192 executions over 64 shared initial-state scenes, not 192 distinct environments.

| Group | Real-video pretraining | 256 sim images: success | Mean XY error | 32 sim images: success | Mean XY error |
|---|---|---:|---:|---:|---:|
| A | RGB consistency only | 192/192 | 1.774 mm | 184/192 | 4.761 mm |
| B | A + object boxes | 192/192 | 1.790 mm | 183/192 | 5.117 mm |
| C | A + hand keypoints | 192/192 | 1.282 mm | 184/192 | 4.662 mm |
| D | A + both targets | 192/192 | 1.654 mm | 181/192 | 5.217 mm |
| R | ImageNet only; no phone-video pretraining | 192/192 | 1.735 mm | 182/192 | 5.663 mm |

A/B/C/D at 256 images form the original fixed development experiment (768 episodes). R is an additional diagnostic (192 episodes), introduced after some primary localization metrics were visible. The 32-image condition was specified after seeing the reference ceiling and reuses the same test scenes (960 episodes): it is exploratory follow-up, not fresh confirmatory evidence. Original runs were preserved. A separate privileged-position controller diagnostic succeeded on all 64 matched scenes; it is not a learned-policy result or a mathematical performance upper bound.

The hand-only group has lower average localization error across the three seeds, but the pattern is not uniform: at 256 images, seed 7 gives C 1.479 mm versus A 1.458 mm. C matches A in success at both budgets. The combined loss shows no measured success advantage. No group/seed/checkpoint was selected for the main table. No statistical significance or real-world generalization is claimed from three seeds and one real scene family.

![Primary experiment](artifacts/reports/visual_ablation_v1/comparison.png)

- [Local visual report](artifacts/reports/visual_ablation_v1/index.html)
- [Detailed Chinese primary report](artifacts/reports/visual_ablation_v1/REPORT_ZH.md)
- [No-phone-video reference](artifacts/reports/visual_ablation_v1/IMAGENET_REFERENCE_ZH.md)
- [32-image follow-up](artifacts/reports/visual_ablation_v1/LOW_DATA_ZH.md)
- [Metrics and per-seed results](artifacts/reports/visual_ablation_v1/results.json)
- [Primary fairness audit](artifacts/reports/visual_ablation_v1/fairness_audit.json)
- [First-scene comparison video](artifacts/reports/visual_ablation_v1/first_scene_comparison.mp4) — fixed seed 7, first test scene 30000, not a best-of selection.

## Independent macro-action world model

A separate model learns the endpoint displacement and lift/success probabilities of the fixed grasp skill from **simulator-generated transitions**. It leaves the A/B/C/D/R comparison and controller unchanged. Three fixed model seeds use 320 training transitions; the test has 160 transitions sharing 32 initial scenes.

| Input / predictor | Mean endpoint error | P90 endpoint error | Correct success/failure labels |
|---|---:|---:|---:|
| True state, zero-displacement baseline | 90.8853 mm | 190.1115 mm | 89/160 |
| True state, training-mean baseline | 92.0621 mm | 122.1010 mm | 89/160 |
| True state, learned ensemble | **28.4107 mm** | **80.2912 mm** | **153/160** |
| RGB-estimated state, learned ensemble | 28.2960 mm | 77.6660 mm | 153/160 |

153/160 (95.625%) is outcome classification accuracy; only 71/160 recorded test actions actually succeeded. The RGB row is a later diagnostic with the previously selected D_seed7 visual model, the same 32 scene images and the original commands. Those commands were constructed from true positions plus fixed offsets during collection. This diagnostic adds **zero physical executions** and does not measure RGB action selection or planning improvement. Small metric differences do not establish an advantage for estimated state.

One separately preselected RGB integration scene also executes successfully, with a 0.7228 mm final-position prediction error. This single example connects the interfaces; it is not a population success estimate. The model is an action-conditioned fixed-skill outcome predictor, with no language policy, video generation or evaluated long-horizon rollout.

[Model scope, results and recovery record](docs/GRASP_WORLD_MODEL.md) · [True-state metrics](artifacts/reports/grasp_world_model_v1/results.json) · [RGB diagnostic](artifacts/reports/grasp_world_model_v1/rgb_evaluation/REPORT_ZH.md) · [Integration video](artifacts/reports/grasp_world_model_v1/integration/rgb_world_model_demo.mp4)

## Independent Diffusion Policy extension

A user-requested DP-D group now learns future robot commands from **128 additional simulated expert trajectories**, with 16 complete validation trajectories. The frozen D_seed7 model supplies the initial RGB estimate; the action model also sees robot proprioception, the previous effective command and a robot clock. A compact conditional temporal U-Net learns action diffusion; 20-step DDIM predicts 16 commands and executes eight before replanning. Runtime does not invoke the scripted grasp-stage controller.

| Policy seed | Strict success | Qualified lifts | Held-out action XYZ error |
|---|---:|---:|---:|
| 7 | 27/64 | 33/64 | 12.700 mm |
| 17 | 25/64 | 30/64 | 12.454 mm |
| 27 | 32/64 | 39/64 | 11.980 mm |

Combined: **84/192 (43.75%)**, sharing 64 initial scenes. The existing D_seed7 fixed-controller reference is 64/64 on these same images. All three policy seeds use the same frozen visual checkpoint, fixed 5,000 training steps and final EMA weights; no best-seed selection. These extra action labels and different control/training budgets prevent a causal, budget-matched comparison with A/B/C/D.

This is initial-image state-conditioned, clock-conditioned action diffusion, not an end-to-end RGB-history policy or an exact reproduction of the published benchmark setup. Its robot action labels come from simulation, not the phone videos. All predeclared test scenes are retained; two test initial images also occur in training/validation, so the report includes a separate descriptive sensitivity check excluding those images. Existing experiment results remain unchanged.

[Implementation and limits](docs/DIFFUSION_POLICY.md) · [Measured DP report](artifacts/reports/diffusion_policy_v1/REPORT_ZH.md) · [All metrics](artifacts/reports/diffusion_policy_v1/results.json) · [First-scene video, seed 7](artifacts/runs/diffusion_policy_grasp_v1/DP_seed7/first_scene_demo.mp4)

## Method

1. **Collect and audit:** 12 personal phone clips, 720p/25 fps, 86.48 seconds and 2,162 frames. Exact duplicate copies are excluded. Whole episodes are assigned to one conservative development scene family before derived labels. Actual recording sessions remain unknown.
2. **Generate independent targets:** conservative RGB/ArUco object candidates and fixed MediaPipe 2D hand landmarks. The object branch never sees hand labels. Invalid auxiliary labels are masked without deleting RGB frames. Visible-component boxes are not amodal object truth.
3. **Pretrain on the same 438 full RGB images:** identical ImageNet ResNet18 initialization and frozen teacher; feature consistency under mild photometric augmentation. All groups instantiate both auxiliary heads and differ only in loss switches. Fixed 240 steps, batch 16, AdamW 1e-4.
4. **Adapt equally:** discard auxiliary heads; transfer only the encoder into a new spatial XY estimator. Use identical simulation RGB/state pairs, head initialization, sampling, augmentation and fixed 300-step budget. Target normalization uses training labels only. The main data split is 256/64/64 complete seeded initial scenes.
5. **Execute physically:** RGB → predicted world XY → shared approach/grasp/lift/transport/release/withdraw controller. Known object dimensions, common goal and robot proprioception are allowed. Object ground truth is used only for training labels, scoring and explicitly privileged diagnostics.

Success requires a qualified bilateral-contact lift (bottom clearance ≥15 mm for ≥0.12 s), complete crossing of the line, workspace containment, table contact, upright placement, release and withdrawal, and ≥0.5 s of stability. Center crossing alone is insufficient. See [task and scoring](docs/GRASP_SIM_REPORT.md).

## Run locally

All work is stored under `E:\HumanoidRobotLearning`. The tested environment is Python 3.11, PyTorch 2.10.0+cu126, torchvision 0.25.0, conda-forge PyBullet 3.25, PandaGym 3.0.7 and an RTX 3080 Ti Laptop GPU. See [environment instructions](docs/ENVIRONMENT.md). A clean reinstall on a second machine has not been verified.

```powershell
Set-Location -LiteralPath E:\HumanoidRobotLearning
. .\scripts\Enter-Project.ps1
.\.env\python.exe scripts\run_tests.py
.\.env\python.exe scripts\analyze_visual_ablation.py
.\.env\python.exe scripts\analyze_reference.py
.\.env\python.exe scripts\analyze_low_data.py
.\.env\python.exe scripts\build_review_media.py
```

Repeat the complete primary experiment into a fresh output directory:

```powershell
.\scripts\Run-PilotExperiment.ps1 -RunName repeat01
```

Run one existing RGB model locally:

```powershell
.\.env\python.exe scripts\evaluate_visual_grasp.py --groups D --seeds 7 --limit 1 --record-first --output artifacts\demos\local_review
```

The 32-image extension is defined by `configs/low_data_plan.json`; its source is `scripts/run_low_data_adaptation.py`. Training scripts protect completed outputs from accidental overwrite. [Reproducibility details](docs/REPRODUCIBILITY.md) explain inputs, data provenance and boundaries.

## What worked and what did not

- The complete personal-video → trained encoder → RGB localization → physical grasp path runs. Primary provenance/pairing checks pass, including rendered-image identity and online/offline prediction agreement. The recorded 26 core tests passed using isolated processes.
- The 256-image benchmark is too easy to distinguish task success; even ImageNet without personal-video pretraining succeeds on all tested scenes.
- Reducing simulation supervision to 32 images reveals failures and larger localization errors, but does not support the combined-supervision hypothesis. All 46 low-data failures lack a qualified lift; example replays are preserved separately from aggregate evidence.
- The Windows Torch and Panda/Bullet stacks conflict when loaded together (two OpenMP runtimes). A persistent isolated predictor process fixes integration; only RGB bytes enter it. Tests use separate processes. Unsafe duplicate-runtime suppression is not used.
- The original pushing task and MPC proposal were superseded by grasp-and-place. The visual comparison retains its fixed controller. The separate learned macro-action predictor models fixed-skill outcomes and has not been used to improve planning.

## Limits and remaining work

This is a small pilot with one conservative real development scene family. There is no independent real-scene test, dense manually annotated pseudo-label accuracy benchmark, yaw variation, camera/lighting/background randomization, real robot evaluation or language-conditioned policy. Personal video is sampled as frames; the separate DP extension learns temporal robot command chunks from simulated expert demonstrations, not human action sequences. Friction, mass and simulator geometry are stated assumptions rather than measurements from the phone footage. A nominal 50 mm ruler measured 48 mm; uniform print scaling is an assumption, not full 3D calibration.

The reported four-group experiment uses real data for visual learning. The independent macro-action extension supplies a trained outcome model from simulation, with true-state and one-checkpoint RGB diagnostics. There is still no VLA, demonstrated world-model planning benefit or broad dynamics generalization. The separate DP extension learns actions from additional simulated robot demonstrations. Its transition labels do not come from the personal videos.

Raw phone videos, derived NPZs, large weights and the environment remain local and are excluded from Git. Without those exact inputs, the code alone cannot recreate this particular experiment. No public data download, code license, public repository or application submission is asserted. These remain applicant-owned release steps. See the [challenge alignment and delivery gaps](docs/CHALLENGE_ALIGNMENT.md).

[Current task](docs/CURRENT_TASK.md) · [Data and labels](docs/PILOT_LABEL_REPORT.md) · [Training implementation](docs/VISUAL_TRAINING.md) · [Original challenge](https://jobs.ashbyhq.com/humanoid/e1a2a9de-ad23-4632-9d93-ee50fd41a221)
