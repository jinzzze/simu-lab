# Challenge alignment and current delivery status

The completed pilot uses the applicant's 12 phone videos to train a visual representation, adapts it with simulated labels, and executes physical Panda grasps. A separate learned macro-action outcome model now predicts fixed-skill state transitions and events. Its evidence remains separate from the original auxiliary-supervision comparison.

## Completed evidence

- Original A/B/C/D experiment: three training seeds, 256 simulator adaptation images, 64 shared test scenes per model, 768 executions. Every group scored 192/192.
- Additional ImageNet reference R: 192/192. It skips personal-video pretraining and retains supervised simulator adaptation.
- Exploratory 32-image follow-up: A 184/192, B 183/192, C 184/192, D 181/192, R 182/192. It reuses the development test scenes and one fixed training subset.
- Independent macro-action model: 560 simulator transitions across 112 initial scenes, including 160 test transitions across 32 scenes; all three fixed-budget model seeds are reported. With true initial state as input, the ensemble achieves 28.4107 mm mean endpoint error and 80.2912 mm P90, and correctly classifies success/failure for 153/160 transitions.
- One preselected RGB integration example: outcome predictions saved before execution, no true object state supplied to the model, physical success, 0.7228 mm final-position prediction error.
- Follow-up RGB input diagnostic: the same 32 test scenes and 160 recorded commands, all rendered RGB hashes matched, zero new physical executions. Endpoint mean error is 28.2960 mm, P90 77.6660 mm, and success/failure classification remains 153/160.
- Independent DP-D learned-action extension: 128 simulated expert training trajectories, 16 validation trajectories, three final EMA policy seeds and 84/192 strict successes on 64 shared test scenes. Same frozen D7 visual model; additional action labels and different training/control budgets, so this is not a matched auxiliary-supervision comparison. See [implementation and caveats](DIFFUSION_POLICY.md).
- Source and data provenance, paired-input audits, recorded core tests, checkpoints, plots, failure replays and local reproduction instructions.

153/160 is outcome classification accuracy, not robot success rate: 71/160 of the test actions actually succeeded. The RGB example is one demonstration, not a population estimate or a policy improvement.

## What can be concluded

The combined auxiliary targets did not show a task-success advantage. C had lower average localization error, but did not improve success and was not better in every seed. The 46 low-data failures involve six shared scenes; two scenes failed for all 15 group/seed combinations. Repeated seeds do not create independent environments. Success saturation does not demonstrate equivalence, and the small observed gap does not establish significant degradation.

The learned macro-action model improves endpoint prediction over zero-displacement and training-mean baselines on its simulated test distribution. All transition labels are simulator generated. The personal videos contribute through the upstream visual model in the separate integration example. The model predicts the result of a fixed skill; it does not select actions, replan, generate video or implement a language-conditioned policy. See [world-model scope and evidence](GRASP_WORLD_MODEL.md).

## Remaining implementation and release work

The original A/B/C/D/R experiment still uses a fixed grasp state machine and frame-level representation learning. The independent outcome predictor adds a trained model component, but does not retroactively convert that comparison into a learned-control or VLA experiment. There is no VLA, imitation of human action sequences, long-horizon learned dynamics validation or demonstrated world-model planning benefit. A separate DP-D policy now imitates simulated expert robot command sequences; its results and extra supervision are reported independently.

Independent real recording sessions, dense pseudo-label accuracy annotation, stronger action-aware world-model baselines, broader RGB uncertainty evaluation, new camera/object/physics distributions and real robot testing remain open. The completed RGB batch diagnostic uses one previously selected visual checkpoint and replays commands originally constructed using true simulator state plus fixed offsets. It is not an RGB policy choosing actions, and the small metric differences do not show that estimated state is superior to true state.

A public GitHub repository, code/data release licensing, public input and weight availability, a clean installation on another machine, and final application submission remain incomplete. Raw recordings and large outputs are currently local and ignored by Git; a public code repository alone would not reproduce these specific results. Publication must also check that linked videos and required inputs are actually accessible. No personal applicant details or public project URLs have been fabricated.

## Review entry points

- [README](../README.md): English overview and full auxiliary-supervision table.
- [Local visual report](../artifacts/reports/visual_ablation_v1/index.html).
- [32-image follow-up](../artifacts/reports/visual_ablation_v1/LOW_DATA_ZH.md).
- [Independent macro-action model](GRASP_WORLD_MODEL.md).
- [Reproduction instructions](REPRODUCIBILITY.md).

These are evidence and release boundaries for the current project, not a claim that the hiring challenge has accepted or approved the submission.


## Release preparation update (2026-10-07)

Independent real-scene validation is deferred by the applicant. Local source/report and input/checkpoint release candidates, an English application draft, a 112-second captioned demonstration and a 48-frame blinded development-label review are prepared. Human box/keypoint accuracy is not yet measured. Same-host clean installation passes the recorded checks; publication, final license/data permissions, applicant identity and the actual application remain pending. See [release checklist](RELEASE_CHECKLIST_ZH.md) and [archive reproduction](RELEASE_REPRODUCTION.md).
