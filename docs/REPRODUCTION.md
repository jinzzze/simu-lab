# Reproduction scope and troubleshooting

[Quick replay](QUICKSTART.md) · [Training](TRAINING.md) · [Simple task](SIMPLE_TASK_REPRODUCTION.md)

## Choose the result you want to reproduce

| Route | Input package | Expected output |
|---|---|---|
| Four-goal fixed replay | v1.1.0 replay ZIP | Two executions, replay videos, array/score comparisons |
| Four-goal training repeat | v1.1.0 full evidence ZIP, then a new training workspace | New demonstrations, six models and the frozen evaluation suite |
| Simple-task records and replay | Both v1.0.0 ZIPs, extracted together | Original-stage reports and the documented fixed-controller smoke execution |

The four-goal quickstart is the default. The release ZIPs contain frozen code as well as inputs. Run from the extracted package; overlaying a current source checkout can invalidate package hashes.

## What was actually verified

Two four-goal development cases reproduced identical observations, actions and scoring traces on the same Windows host, including from an anonymous public download in a separate directory using an existing independently installed environment. Both cases fail strict placement. Matching that failure is the expected replay result.

The earlier v1.0.0 checks included a fresh environment on the same host, 39 tests, environment diagnostics, report regeneration and selected physical replays. Neither stage establishes full-training determinism or compatibility with an independently configured second computer. [Recorded receipts](EVIDENCE.md).

## Common problems

| Symptom | Check |
|---|---|
| Missing checkpoint or recording | A Git clone contains source and reports. Download the package required by your route. |
| CUDA unavailable | Current inference requires an NVIDIA CUDA GPU. Confirm the driver and the documented PyTorch wheel installation. |
| PyBullet attempts a source build | Install it through the supplied conda-forge environment; do not replace it through pip. |
| Duplicate OpenMP runtime error | Use the supplied separate-process entry points. Do not set `KMP_DUPLICATE_LIB_OK`. |
| Output tag already exists | Choose a new replay tag; completed outputs are protected. |
| Package hash mismatch | Compare the ZIP checksum, re-extract into a new folder and verify before running. |
| Arrays or scores differ | Preserve the new trace and record OS, GPU, driver and library versions. Exact equality may depend on the runtime. |

Training is intentionally a separate procedure. A low loss, successful file verification or matching replay does not establish reliable task performance.

The presentation update also re-executed both four-goal cases and one original simple-transfer case. See the [verification receipt](PRESENTATION_VERIFICATION.json) for the precise scope.
