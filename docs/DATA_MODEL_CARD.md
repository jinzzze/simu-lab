# Data and model card

## Personal development data

12 canonical phone videos, 2,162 frames / 86.48 seconds, 1280 × 720 at 25 fps. The block is 20 × 20 × 10 mm. The printed nominal 50 mm ruler measured 48 mm; planar calibration assumes uniform 0.96 scaling. All videos belong to one conservative development scene family. Actual independent recording sessions are unknown. Duplicate copies are not extra data. Independent real-scene validation was deferred by the applicant on 2026-10-07 because new filming was unavailable.

The shared pretraining export contains 438 RGB frames, 398 accepted object targets and 223 frames with accepted hand targets. Object labels are visible dark-component boxes, not amodal boxes or true 3D centres. Hand labels are MediaPipe estimates. Confidence masks are not accuracy measurements. Existing assistant visual review inspected 96 regular frames and extra uncertain examples. A new deterministic 48-frame blinded annotation set is prepared; independent human boxes and per-joint accuracy remain unmeasured. No accuracy number is invented for these labels.

## Models and supervision

| Component | Training inputs and targets | Evaluation and limitation |
|---|---|---|
| A/B/C/D visual transfer | Shared real RGB; optional pseudo boxes and hand points; then supervised simulated XY | Fixed controller, 64 shared simulated test scenes; no new real-scene test |
| R reference | ImageNet initialization and simulated XY only | Additional post-hoc diagnostic |
| Macro-outcome ensemble | Simulator initial XY and absolute command; displacement and event labels | 160 transitions share 32 scenes; predictions do not choose actions |
| DP-D | 128 simulated training and 16 validation expert trajectories; frozen D7 initial RGB estimate | 84/192; additional action supervision; 2 rendered-image collisions across train/validation and evaluation; exclusion sensitivity 80/186 |

All visual auxiliary heads are discarded at runtime. Simulator truth is used for collection labels, error measurement and scoring; it is not supplied to runtime perception or the learned action policy. The DP policy has no online object visual feedback after its initial estimate. Success counts should not be confused with the world model's success/failure classification accuracy.

## Access and intended use

Research prototype for a local application challenge. No real-robot deployment or broad scene robustness is established. Canonical videos, processed inputs and checkpoints are staged locally; no public access is asserted before the owner approves publication and the upload is verified. Media, data and derived weights require their own release decision; a source-code license does not automatically cover them. External base weights remain fetched from their original sources and verified by SHA256.
