# Project development rules

- Current scope: English presentation, navigation and reproduction fixes. Experiments are frozen. Preserve model, configuration, machine-readable result and frozen executable bytes; do not start new training or benchmarks.
- The project progresses from simple transfer and visual learning to learned four-goal control. Read `docs/OVERVIEW.md`, `docs/EXPERIMENTS.md` and `docs/CODE_MAP.md` before making structural changes.
- Work in the E: project checkout. Derive runtime paths from the repository root. Keep large recordings, checkpoints, environments, logs and caches out of the old C: workspace and Git.
- Use the released package matching the experiment. The current main branch is a reader-facing source/documentation view; frozen release packages have their own manifests.
- On Windows, dot-source `scripts/Enter-Project.ps1` and use the project environment. PyBullet is managed by conda-forge. Torch and Bullet inference/physics must use separate processes. Do not use `KMP_DUPLICATE_LIB_OK`.
- Keep training-only information out of runtime policy inputs. Simulator truth is allowed for labels, scoring and identified oracle diagnostics, not online correction of primary policies.
- Split by whole episode before deriving frames. Keep future outcomes out of current observations; robot action labels come from pre-recorded executed commands.
- Report every declared seed and failure. Distinguish smoke checks, replay checks, predictions and measured policy success. Do not pool results across task settings or budgets.
- Personal recordings come from the author. Credit Codex assistance accurately. Synthetic data and third-party images do not satisfy personal-data collection.
- Current reader-facing documents and figures use English. Original development records remain linked at immutable commits. Never rewrite frozen evidence merely to update historical status fields.
