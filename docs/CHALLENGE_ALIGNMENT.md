# Challenge alignment and limits

The project provides personally collected video inputs, executable models, example outputs, design decisions, successes and failures through a public repository and versioned artifacts. The original challenge permits state-prediction directions; its recorded wording is in the historical verification note. The employer determines whether the submitted work meets its assessment standard.

| Evidence | What this project supplies |
|---|---|
| Personal data contributes | Earlier real-video encoder training; current training-only goal geometry plus matched video timing/progress that changes executed teacher commands |
| State prediction | Action-conditioned fixed-skill endpoint and event prediction, trained on simulation transitions; outcome prediction is separate from execution success |
| Learned control | Earlier compact Diffusion Policy; current goal-conditioned recurrent RGB policy with explicit deployment information boundaries |
| Evaluation | All seeds and failures, fixed budgets, held-out combinations, overlap sensitivity and causal input/action traces |
| Reproduction | Frozen sources/configuration/checkpoints, file hashes, executable fixed-case replay and training recipes |

No language-conditioned VLA, real-robot validation, human metric 3D action recovery or evaluated world-model planning benefit is claimed. The current personal-data comparison has low success and does not establish a reliable positive effect. Earlier tasks/controllers/budgets remain separate.

[Current overview](../README.md) · [Earlier study](../README_v1.0.0.md) · [Run instructions](QUICKSTART.md) · [Results](../experiments/personal_video_timing_v4/RESULTS_EN.md)
