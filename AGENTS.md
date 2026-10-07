# Project development rules

- Work in this repository on E:. Derive paths from the repository root; do not create project datasets, models, environments, logs or caches under the old C: workspace.
- Read `docs/PROJECT_PLAN.md` before changing the research protocol. Current study is the A/B/C/D auxiliary-supervision comparison, not an unspecified VLA project.
- In PowerShell, dot-source `scripts/Enter-Project.ps1` and use `.env/python.exe`. PyBullet is managed by conda-forge; do not replace it with a PyPI source build.
- Keep raw RGB, episode splits and controller identical across the four primary groups. Human hand labels are training-only auxiliary targets. The current grasp prototype uses a shared fixed physical state machine; the old push world model is historical. Read docs/AUTONOMOUS_DEVELOPMENT_PROTOCOL.md for this recorded revision.
- On this Windows environment, run scripts/run_tests.py: Torch and Panda/Bullet native OpenMP runtimes must use separate processes. Use src/predictor_process.py for live RGB inference in the simulator. Never suppress duplicate-runtime errors with KMP_DUPLICATE_LIB_OK.
- True simulator object state may be used for training labels, scoring and explicit oracle diagnostics, never to correct primary-group runtime perception or contact planning.
- Split by complete episode/session before deriving frames or labels. Keep future outcomes out of current inputs and use pre-recorded commands for action labels.
- Distinguish smoke checks, synthetic fixtures, planned sample counts and measured research results. Never fabricate success rates or claim uncollected data exists.
- Keep large videos, model weights and environment folders out of Git. Preserve data provenance and record external model versions and hashes.
- The applicant must personally collect the required real data. Synthetic or third-party images may test tooling but do not satisfy that requirement.
