# simu-lab

### Learning robot manipulation from personal video

Can a few phone recordings help a robot learn to pick up and place a block? This project follows that question from a simple transfer task to a task with four start locations and four specified goals. All robot experiments run in a contact-based Panda simulation.

[Overview](docs/OVERVIEW.md) · [Experiments](docs/EXPERIMENTS.md) · [Run a policy](docs/QUICKSTART.md) · [Data & models](docs/DATA_MODEL_CARD.md) · [Downloads](https://github.com/jinzzze/simu-lab/releases/tag/v1.1.0)

## The two tasks

### Task 1 · Simple transfer

Pick up a small block and place it entirely beyond a black line. Release it, withdraw the gripper and leave the block stable on the far side.

![Task 1: grasp and lift a block, carry it across the black line, then place and release it](media/task-1-simple-transfer.svg)

### Task 2 · Four-goal placement

Start at one of P1–P4 and move the block to the specified goal, G1–G4. The start and goal can have different numbers. The entire block must finish inside the selected 40 mm target square after release and withdrawal.

![Task 2: transfer from one of four start positions to a specified goal, with the full block footprint inside its target square](media/task-2-four-goal.png)

The second task adds a specified destination and a tighter placement requirement. Both diagrams show task layouts, not measured trajectories; they are not to scale.

[Watch the 92-second walkthrough](https://github.com/jinzzze/simu-lab/releases/download/v1.1.0/evidence_walkthrough.mp4) · [See recorded human and robot scenes](media/task-preview.png)

## From seeing the block to choosing the next move

The first task asked the robot to lift a block and place it beyond a line. A learned visual model estimated the block's position; a fixed controller carried out the grasp. This made it possible to study what the personal videos contributed to perception.

The next experiments explored skill-outcome prediction and learned robot commands. The task then expanded to four starts and four goals: the policy had to use the requested destination and place the entire block inside a 40 mm square. The final comparison tested whether video-derived transport timing and image progress helped beyond the goal geometry already shared by both groups.

![Research progression from visual learning to goal-conditioned control](media/study-progression.svg)

Each stage asks a different question. The outcome predictor is a supporting study, not a planner used by the four-goal policy. [Read the study progression](docs/OVERVIEW.md).

## How the video is used

In the latest study, 24 personal training clips supply target locations. One condition also uses their coarse transport duration and two-dimensional image progress to shape robot demonstrations in simulation. The learner imitates the robot commands actually executed in those demonstrations.

![Video information shapes offline demonstrations; the policy acts on current observations](media/method.svg)

At execution time, the policy receives the current simulated camera image, robot state, requested goal and previous command. It predicts the next gripper movement and opening. A recurrent model carries information from previous steps. [Method and information boundaries](docs/METHOD.md).

## What the experiments found

| Question | Measured result | What it tells us |
|---|---|---|
| Do hand and object targets improve the visual model? | With 32 simulation adaptation images: RGB-only **184/192**, combined targets **181/192** successful transfers | No measured success advantage from the combined targets |
| Can a model predict a fixed skill's outcome? | Mean endpoint error **28.41 mm**, versus **90.89 mm** for a no-motion baseline | Useful prediction on this simulated distribution; planning benefit was not tested |
| Does video timing help the four-goal learner? | Shared geometry **7/96**; added timing/progress **11/96** strict placements | A small observed difference; overall success is low and a reliable benefit is not established |

The tasks, controllers and data budgets differ across rows. The [experiment guide](docs/EXPERIMENTS.md) includes the learned-action and goal-coverage studies, all model seeds and links to the underlying results.

Grasping is the main bottleneck in the latest study: **75/96** and **69/96** runs did not achieve a qualified lift. Its 96 executions per condition reuse eight physical starts across goals and three model seeds. Four initial test-task images also repeat training/development pixels; [the full results](experiments/personal_video_timing_v4/RESULTS_EN.md) retain the original scores and report exclusions separately.

## Run the recorded policy

Start with the [quick replay guide](docs/QUICKSTART.md). It runs two frozen development cases and compares their observations, actions and scores with the recorded traces.

- **Runtime:** Windows, Conda and an NVIDIA CUDA GPU; Python 3.11.
- **Download:** the **12.8 MB replay ZIP** for the two cases, or the **1.80 GB evidence ZIP** for all six models and the current study's inputs and traces. A source clone alone does not contain the weights.
- **Expected result:** both reference cases fail strict placement. A successful replay check reproduces that recorded behavior.

| I want to… | Start here |
|---|---|
| Understand the project | [Overview](docs/OVERVIEW.md) |
| Run an existing policy | [Quickstart](docs/QUICKSTART.md) |
| Repeat training | [Training guide](docs/TRAINING.md) |
| Revisit the simple task | [Stage 1 reproduction](docs/SIMPLE_TASK_REPRODUCTION.md) |
| Inspect every result | [Experiments](docs/EXPERIMENTS.md) and [evidence index](docs/EVIDENCE.md) |
| Find the implementation | [Code map](docs/CODE_MAP.md) |

## Scope and credits

This is a small simulation study. There is no real-robot evaluation or language-conditioned policy. The recordings share a limited scene family, and the extracted image labels do not recover human metric 3D actions. Exact replay was checked on the original Windows host with an independently installed environment; another-machine compatibility and full-retraining determinism remain unverified.

The author collected the personal recordings and confirmed the task layout. Codex assisted with implementation, experiment execution, candidate-label inspection and writing. Recordings and project checkpoints are public with permission; no blanket source/data/model reuse license has been granted. See [data provenance](docs/DATA_MODEL_CARD.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
