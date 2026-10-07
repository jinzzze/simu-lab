# Submission draft — simu-lab

Applicant: **[ENGLISH NAME REQUIRED]**  
GitHub account: **jinzzze**  
Intended repository: **jinzzze/simu-lab** (not yet confirmed published)  
Challenge: https://jobs.ashbyhq.com/humanoid/e1a2a9de-ad23-4632-9d93-ee50fd41a221

## Short application description

I investigated whether object and hand supervision extracted from personally collected phone videos improves robot learning in simulation. My dataset contains 12 overhead videos of picking up a small black block and placing it beyond a line. A controlled A/B/C/D study compares RGB consistency pretraining with object-box, hand-keypoint and combined auxiliary losses. All groups share frames, initialization, batches, training budgets, simulation splits and a contact-based Panda grasp controller.

The result is negative: combined supervision did not improve success. At 256 simulation adaptation images all groups achieved 192/192 executions; an exploratory 32-image follow-up gave RGB-only 184/192 and combined supervision 181/192. These executions reuse 64 initial scenes across three training seeds. I also implemented an action-conditioned macro-outcome world model and a separate compact Diffusion Policy extension. The world model obtained 28.41 mm mean endpoint error and 153/160 correct outcome labels on 32 held-out simulated initial scenes. The learned action policy achieved 84/192 strict grasp successes using additional simulated expert trajectories.

The deliverable includes source code, data lineage, all-seed results, failure examples, checkpoints and reproduction instructions prepared as local release candidates. There is no claimed improvement in real-world generalization, no real-robot evaluation and no VLA. New independent real recordings are deferred. I used Codex assistance for implementation, experiments and documentation; the personal recordings were supplied by me. I will verify the materials and authorship statements before submission.

## Technical choices to discuss

- Keep human-hand targets as auxiliary perception supervision; do not invent human-to-robot joint action labels from monocular videos.
- Separate Torch inference from physics processes to avoid the observed Windows OpenMP conflict.
- Score real contact, sustained lift, release, full boundary clearance and stability; no attachment or object teleportation.
- Retain negative results, post-hoc study status, all seeds and first-scene/failure selection rules.
- Use the world model for outcome prediction only; no planning improvement was measured.
- Describe DP-D as a compact implementation using initial estimated XY, robot state and time, not an official benchmark reproduction or end-to-end image-history policy.

## Before pasting into the application

Fill the applicant name and real public repository/release/demo URLs. The official page was verified in a browser on 2026-10-07: deadline Friday 9 October 2026, 23:59 BST; a public GitHub repository is required. The current application fields are name, email, CV/resume, education, GitHub repository URL and UK work authorization status. Supply these personally; this project description is supporting README material, not a currently observed separate form field. Public URLs must be opened while logged out after publishing. Do not replace pending URLs with guessed links or claim the challenge has accepted this submission.
