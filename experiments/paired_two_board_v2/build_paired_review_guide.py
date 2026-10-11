"""Create a concise review packet from complete, audited local evidence."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, shutil

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/policy_recurrent_goal_precision_v5.json')
    args=parser.parse_args();cfg = read(ROOT / args.config)
    reports = ROOT / f'artifacts/reports/{cfg["revision"]}'
    suites = ('validation', *cfg['evaluation']['suites'])
    results = {}; sources = {}
    for suite in suites:
        audit = reports / f'{suite}_validation.json'
        path = reports / f'{suite}_analysis_v1/results.json'
        assert audit.exists() and path.exists(), f'Finish {suite}'
        results[suite] = read(path)
        sources[str(path.relative_to(ROOT))] = sha(path)
        sources[str(audit.relative_to(ROOT))] = sha(audit)
    demo = reports / 'four_goal_nominal_demo_v1/manifest.json'
    assert read(demo)['status'] == 'completed' and read(demo)['visual_review'] != 'pending inspection'
    diagnostic = reports / 'input_diagnostics_v1/results.json'
    history = reports / 'stability_history_v1/results.json'
    for path in (demo, diagnostic, history):
        assert path.exists(); sources[str(path.relative_to(ROOT))] = sha(path)
    paired_audit = reports / ('v5_v6_augmentation_only_audit_v1/results.json' if cfg['revision'].endswith('_v6') else 'v4_v5_loss_only_audit_v1/results.json')
    sources[str(paired_audit.relative_to(ROOT))] = sha(paired_audit)
    progress=read(ROOT/'policy_progress.json')
    prefix=reports.relative_to(ROOT).as_posix()
    recombination=reports/'consistent_goal_recombination_v1/analysis_v1/results.json'
    if cfg['revision'].endswith('_v6'):
        assert recombination.exists(), 'Finish the already scheduled consistent-goal diagnostic'
    receipt = ROOT / ('receipts/review_guide_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    receipt.mkdir(); path = ROOT / 'REVIEW_GUIDE_EN.md'
    if path.exists(): shutil.copy2(path, receipt / 'previous_REVIEW_GUIDE_EN.md')
    lines = ['# Four-goal learned-control review guide', '',
             'This local extension now controls physical simulated pick-and-place with a learned current-RGB/goal recurrent policy. The applicant recorded 16 phone clips. All three fixed training seeds and failures are reported. The added video timing/progress has not shown a consistent success advantage. The measured development process includes coordinate precision weighting and episode-consistent RGB augmentation. The public v1.0.0 release describes earlier work; this extension has not yet been published or submitted.', '',
             '## Start here', '',
             f'1. [Watch the four paired tasks]({prefix}/four_goal_nominal_demo_v1/REPORT_ZH.md). Fixed seed7, first nominal scene of each P1→G1 through P4→G4, both conditions, no outcome filtering. Eight replays exactly match original input/action arrays and the full physical scoring traces. Captions include every shown failure and the all-seed nominal success counts.',
             f'2. [Inspect all measured outcomes and limits](LEARNED_POLICY_REPORT_EN.md). At the recorded refresh, {progress["trained_models_completed"]} models have finished across the retained revisions; the initial zero-update CUDA failure and all weak versions remain recorded.',
             '3. [Run the fixed-scene reproduction](POLICY_REPRODUCTION_ZH.md). The project provides source/config snapshots, dataset/checkpoint SHA256, pre-action causal input arrays, effective commands and separate scoring. Torch and Bullet run in separate processes. Same-host replay is verified; different-machine numerical replication is not established.', '',
             f'## Complete results: {cfg["revision"]}', '',
             '| Suite | Initial scenes reused across 3 models | No added video timing/progress | Added video timing/progress |',
             '|---|---:|---:|---:|']
    for suite, result in results.items():
        a, b = [result['groups'][c] for c in cfg['conditions']]
        suffix = ' (development)' if suite == 'validation' else ''
        lines.append(f'| {suite}{suffix} | {result["independent_initial_scenes"]} | {a["successes"]}/{a["episodes"]} | {b["successes"]}/{b["episodes"]} |')
    lines += ['', 'These are executions, not independent-binomial trials: each initial scene is reused by three training seeds. Validation was reused to develop the method. Appearance changes object/table colors; wider-start increases XY jitter from ±3 to ±12 mm. New simulator scene IDs are not independent personal recordings or real-robot transfer. V4/v5 use different new-scene sets, so their test counts are not a paired same-scene ablation.', '',
              '## What was learned and what the personal data supplies', '',
              'The learner receives current simulator RGB, robot EE XYZ/finger width, a task-provided goal XY, goal-minus-EE XY and the previous effective command. A causal GRU predicts goal-relative XY command targets, absolute Z and gripper width. A shared coordinate decoder, IK and limits execute those learned targets. There is no online expert phase, object-truth correction or externally supplied clock. Recurrent memory may encode elapsed progress. The goal is also visually highlighted, so numeric-goal corruption is an intentionally conflicting-input diagnostic.', '',
              'Personal S01/train clips supply pre-action blue-mark target geometry to both conditions. Only the guided condition additionally uses coarse transport duration and dimensionless image progress when generating its physically executed robot training demonstrations. Robot paths/heights/gripper commands are engineered; phone RGB is not the visual training input of this controller, and monocular lifted XYZ is not recovered. The comparison tests the added timing/progress information, not a fully personal-data-free versus personal-data system.', '',
              'Strict success requires a qualifying bilateral-contact lift, the whole rotated block footprint inside the designated 40 mm square, table/upright/slow placement, release, withdrawal and at least 0.5 s stable final placement. The scorer cannot repair the policy or terminate it. Crossing the black separator alone fails. Same-number source/goal pairs do not establish arbitrary or unseen-goal following.', '',
              '## Design decisions and failure evidence', '',
              'The relative-delta recurrent variant lifted more blocks but accumulated placement errors. Goal-relative absolute commands changed that integration behavior, but broad action scales underweighted millimeter XY precision. V5 retains v4 architecture, labels, normalization, initialization, episode stream and 2,500 updates; only coordinate loss weights change. [Six paired runs were audited](artifacts/reports/paired_recurrent_goal_precision_v5/v4_v5_loss_only_audit_v1/REPORT_ZH.md). Condition pairs share update/episode budgets, while different demonstration lengths produce different frame-target counts and compute. Development gains support this engineering change; they do not demonstrate video benefit.', '',
              f'[Teacher-forced stage errors]({prefix}/stage_error_v1/REPORT_ZH.md) are separated from closed-loop results. [Paired input interventions]({prefix}/input_diagnostics_v1/REPORT_ZH.md) retain every seed7 case and separate numeric decoder translation from the network response. [Stability history]({prefix}/stability_history_v1/REPORT_ZH.md) checks late regressions without redefining success. Exact input-image overlap reports retain all original cases rather than dropping inconvenient ones.', '',
              '## Honest scope and remaining applicant work', '',
              'The original A/B/C/D/R visual auxiliary-supervision results, macro-action outcome model and old clock-conditioned diffusion policy remain separate historical experiments. This extension adds learned image-conditioned action selection; it does not implement a VLA, video generation or demonstrated world-model planning. Neither variant has established a reliable benefit from personal transport timing, arbitrary-goal generalization or real-world robustness.', '',
              'S02 additional training, S03 independent validation and S04 personal test sessions remain uncollected. Human strict-success adjudication, capture timestamps, real board-to-robot registration, real-robot tests, applicant identity/work-authorization details and application submission are pending and were skipped at the applicant’s request. The applicant reports an extension to 16 October 2026; exact closing time is unconfirmed. The [7 October official-page check](../../docs/CHALLENGE_VERIFIED_20261007.md) records the historical requirement for personally collected data, a public repository, run instructions, outputs, design decisions and successes/failures; it is not evidence that the employer has accepted this work.', '',
              'A technically defensible interview discussion is the measured development process: what information reached the policy, why teacher error did not guarantee closed-loop success, how action/loss coordinates affected physical performance, and why a negative video contrast must be retained. The local evidence is ready for review; any new public release remains a separate action.']
    if cfg['revision'].endswith('_v6'):
        at=lines.index('## Honest scope and remaining applicant work')
        lines[at:at]=['V5 failed all24 appearance trials per condition. V6 keeps its model, data, loss and update budget but uses one stronger brightness/contrast/per-channel-gain transform per whole episode. The actual augmentation parameter stream is paired across conditions. The known appearance family informed this exploratory revision; a fresh warm table-color combination and all new scene IDs were frozen before training. [Augmentation/budget audit]('+prefix+'/v5_v6_augmentation_only_audit_v1/REPORT_ZH.md). These tests establish only the reported limited shifts, not broad camera/lighting/physics robustness.', '']
        result=read(recombination);sources[str(recombination.relative_to(ROOT))]=sha(recombination)
        extra=['## Consistent goal recombination diagnostic', '',
               'The prescribed numerical goal, RGB highlight and scoring goal agree from reset. All16 P/G combinations use four new object-start poses, with all three fixed final models and no retraining or selection. Four trained pairings are controls; twelve untrained combinations use the same four known source/goal locations.', '',
               '| Condition | Trained pairings | Untrained combinations of known locations |',
               '|---|---:|---:|---:|']
        for condition,g in result['groups'].items():
            label='Privileged initial-XY teacher (separate feasibility diagnostic)' if condition=='privileged_teacher' else condition
            extra.append(f'| {label} | {g["seen_successes"]}/{g["seen_executions"]} | {g["unseen_successes"]}/{g["unseen_executions"]} |')
        extra+=['',f'[Every case, all three model seeds and the source/goal matrix]({prefix}/consistent_goal_recombination_v1/analysis_v1/REPORT_ZH.md). These96 learned executions and16 teacher executions are separate from primary scores. Reused start poses are not96 independent environments. The teacher uses privileged initial object XY and its own trajectory duration; it is a feasibility reference, not a matched-budget learned policy or a mathematical upper bound. This test differs from numerical-vector corruption and does not establish unseen goal positions, arbitrary continuous goals or real-robot transfer.', '']
        at=lines.index('## Honest scope and remaining applicant work');lines[at:at]=extra
    sensitivity=reports/'nominal_pixel_overlap_sensitivity_v1/results.json'
    if sensitivity.exists():
        overlap=read(sensitivity);sources[str(sensitivity.relative_to(ROOT))]=sha(sensitivity)
        a,b=[overlap['groups'][c] for c in cfg['conditions']]
        lines+=['',f'Exact-initial-pixel overlap sensitivity is explicitly posthoc: original nominal counts remain {a["full_successes"]}/{a["full_executions"]} and {b["full_successes"]}/{b["full_executions"]}; excluding matching initial images yields {a["without_matching_initial_image_successes"]}/{a["without_matching_initial_image_executions"]} and {b["without_matching_initial_image_successes"]}/{b["without_matching_initial_image_executions"]}. [All exclusions and per-seed counts]({prefix}/nominal_pixel_overlap_sensitivity_v1/REPORT_ZH.md). Pixel equality is not evidence of identical physical states.']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sources,
                'guide_sha256': sha(path), 'builder_sha256': sha(__file__),
                'public_release_or_submission_performed': False}
    (receipt / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    shutil.copy2(__file__, receipt / Path(__file__).name)
    print(json.dumps({'guide': str(path), 'receipt': str(receipt)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
