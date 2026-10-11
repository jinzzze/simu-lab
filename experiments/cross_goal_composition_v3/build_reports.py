"""Build a review entry from audited measured results; preserve historical reports."""
from pathlib import Path
import json
import re
import numpy as np
from common import ROOT,OLD,PROJECT,read,write,sha,now


def fraction(a):
    return f"{a['successes']}/{a['executions']}"


def main():
    result=read(ROOT/'artifacts/results_v1/results.json');cfg=read(ROOT/'configs/policy_v1.json')
    assert result['status']=='completed'
    pilot=read(ROOT/'artifacts/waypoint_development_v1/manifest.json')
    replay=read(ROOT/'receipts/portable_replay_v2/receipt.json')
    assert pilot['status']=='completed' and replay['status']=='passed'
    out=ROOT/'artifacts/results_v1'
    table=['| Test task category | Same-number training | 12-combination training |',
           '|---|---:|---:|']
    categories=[('same_number','Original four same-number tasks'),('trained_cross','Eight additional training-covered combinations'),
                ('heldout','Four combinations held out of both training sets'),('all','All 16 P/G combinations')]
    for key,label in categories:
        table.append(f"| {label} | {fraction(result['totals']['test']['paired_24'][key])} | {fraction(result['totals']['test']['coverage_24'][key])} |")
    seed_table=['| Condition | Seed | Same-number | Additional covered | Held-out | All |',
                '|---|---:|---:|---:|---:|---:|']
    for c in cfg['conditions']:
        for seed,metrics in result['totals']['test'][c]['per_seed'].items():
            seed_table.append(f"| {c} | {seed} | {fraction(metrics['same_number'])} | {fraction(metrics['trained_cross'])} | {fraction(metrics['heldout'])} | {fraction(metrics['all'])} |")
    methods=[
        'Both conditions receive 24 physically executed robot teacher trajectories. The paired condition covers 4 combinations 6 times each; the coverage condition covers 12 combinations 2 times each. Start and goal marginal frequencies are identical: 6 each. The 24 initial physical scenes are matched. Goal numeric coordinates, image highlight and scoring region agree.',
        'The unchanged v6 learner uses current 128px RGB and a 12D robot/goal/past-command vector, a 64D CNN feature, a 128D causal GRU and 227,268 parameters. It predicts goal-relative absolute XY, absolute Z and gripper width. The decoder/IK share an 8mm per-step limit. No runtime object truth, external phase or episode clock is supplied.',
        'Each of seeds 7/17/27 trains for 2,500 AdamW updates with batches of 2 whole episodes, coordinate-weighted MSE, episode-consistent RGB augmentation and final EMA weights. Paired initial weights, sampling and augmentation streams passed hash audits. Different trajectory lengths produce different frame-target counts and compute.',
        'All 48 training trajectories attained qualified lifts. Strict teacher successes were 24/24 versus 23/24; the one failure is retained in the coverage training data. Twelve separately seeded developer trajectories are used for teacher-forced reporting and closed-loop development. No model/seed/checkpoint is selected from results.',
        'Before this study, a smooth-transport teacher proposal scored 10/12 versus 11/12 for the old teacher in a development probe and failed its declared adoption gate. The proposal was rejected. The original teacher and strict scorer are retained in both study groups.',
        'The frozen test runs 32 tasks: all 16 P/G combinations at 2 object starts per source, repeated for 3 model seeds per condition. These are 96 executions per condition over 8 physical start poses, not 96 independent environments. The validation suite contributes 36 additional executions per condition. Replays are excluded from all counts.',
        'Both conditions share goal geometry derived from the applicant\'s existing 16 S01 clips. This study uses no phone RGB for policy training and no human transport timing. It tests combination coverage, not a new personal-video benefit. X01/X02/X03 recordings remain pending.',
        'Held-out pairs are P1→G2, P2→G3, P3→G4 and P4→G1. They appear in no new robot training or developer trajectories. Locations are known; only combinations are held out. Earlier v6 all-pair diagnoses informed this exploratory study, while its new simulation scene IDs and frozen models are used for the reported test.'
    ]
    overlap={c:{key:sum(r[key] for r in result['test_initial_pixel_overlap'] if r['condition']==c) for key in ('matches_training','matches_development')} for c in cfg['conditions']}
    held_a=result['totals']['test']['paired_24']['heldout'];held_b=result['totals']['test']['coverage_24']['heldout']
    seen_a=result['totals']['test']['paired_24']['trained_cross'];seen_b=result['totals']['test']['coverage_24']['trained_cross']
    finding=f"On the four held-out combinations, paired-only training scored {fraction(held_a)} and 12-combination training scored {fraction(held_b)}. On the eight newly covered combinations the corresponding counts were {fraction(seen_a)} and {fraction(seen_b)}. These are measured simulation outcomes, not a claim of arbitrary-goal or real-robot generalization."
    overall=f"Overall frozen-test success was {fraction(result['totals']['test']['paired_24']['all'])} versus {fraction(result['totals']['test']['coverage_24']['all'])}. Increasing coverage did not improve overall success in this experiment, and the small held-out count does not establish reliable composition."
    en=['# Training coverage for goal-conditioned pick-and-place','',finding,'',
        overall,'',
        'This is a separate local extension. The earlier v6 results, personal data and source snapshots remain unchanged.','',
        '## Measured results','',*table,'',*seed_table,'',
        '[Full audited results, all 264 executions and raw provenance](artifacts/results_v1/results.json)','',
        '![All-seed coverage comparison](artifacts/figures/coverage_v3/coverage_results.png)','',
        '## What changed and what was controlled','']
    for text in methods:
        en += [text,'']
    en += ['## Evidence and reproduction','',
        '- [Why this experiment was needed: v6 grasp-location diagnosis](artifacts/v6_failure_analysis_v1/REPORT_ZH.md)',
        '- [Initial-command goal coupling on matched training inputs](artifacts/initial_goal_coupling_v1/REPORT_ZH.md)',
        '- [Observed support and missing intervals in the old personal-video profiles](artifacts/s01_profile_support_v1/REPORT_ZH.md)',
        '- [Teacher-forced development action errors](artifacts/development_action_errors_v1/REPORT_ZH.md)',
        '- [Training budget and source audit](artifacts/training_audit_v1/REPORT_ZH.md)',
        '- [Separate waypoint-label development pilot](artifacts/waypoint_development_v1/REPORT_ZH.md)',
        '- [Exact two-case replay from relocated source and a separate environment](receipts/portable_replay_v2/receipt.json)',
        '- [Frozen study and complete scene list](configs/study_v1.json)',
        '- [Dataset provenance and all teacher attempts](artifacts/robot_demos_v1/manifest.json)',
        '- [Commands and limitations](REPRODUCTION_ZH.md)',
        '- [Applicant capture list and upload locations](CAPTURE_GUIDE_ZH.md)',
        '- [New-video intake and the pending matched comparison](VIDEO_INTEGRATION_PLAN_ZH.md)',
        '- [First predeclared test case, paired-only model](artifacts/evaluation/test/paired_24_seed7/first_predeclared_case.mp4)',
        '- [The same predeclared case, coverage model](artifacts/evaluation/test/coverage_24_seed7/first_predeclared_case.mp4)',
        '- [Original four-goal evidence](../paired_two_board_v2/REVIEW_GUIDE_EN.md)','',
        'Exact initial-pixel overlap check (32 test tasks per condition): '+json.dumps(overlap)+'. No cases are removed from the main results. Distinct random scene IDs do not imply distinct rendered pixels or fully independent tasks.','',
        f"A separate development pilot changed action labels from the safety-limited command to the teacher's original requested waypoint, with unchanged RGB and causal robot-state inputs. Seed 7 achieved {pilot['pilot_successes']}/12 strict successes and {pilot['pilot_lifts']}/12 qualified lifts. Its frozen expansion gate required at least 6 successes and 8 lifts; gate passed: {pilot['gate_passed']}. This pilot is excluded from the 264 primary executions.",'',
        'Two fixed development cases (one per condition) were replayed from relocated source using an independently installed October 7 environment on the same Windows host. All eight input/action arrays and the full scoring traces matched exactly; both 193-frame videos decoded. This checks those two cases, not a new installation today, another machine or full retraining.','',
        'This extension is behavior cloning, not a VLA or a world-model planner. The project\'s historical visual auxiliary study and separate macro-action state predictor retain their own datasets, claims and results.','',
        'Remaining applicant work: record X01/X02/X03; supply actual recording dates and human labels; provide application identity/CV information. New-video processing and its matched contribution experiment require the missing recordings. No new public release or application has been submitted.']
    (ROOT/'README.md').write_text('\n'.join(en)+'\n',encoding='utf-8')
    zh=['# 本轮进度与结果','',
        '已完成：原 v6 失败分析、独立补拍协议、教师开发对照、60条仿真示范、六个模型、冻结闭环评测和数据/预算/输入路径审计。','',
        f"留出组合：同编号训练 {fraction(held_a)}；12组合训练 {fraction(held_b)}。新增训练覆盖组合：{fraction(seen_a)} 对 {fraction(seen_b)}。",'',
        f"全部16种组合测试：{fraction(result['totals']['test']['paired_24']['all'])} 对 {fraction(result['totals']['test']['coverage_24']['all'])}。扩大覆盖没有改善本轮总体成功率，少量留出成功不足以证明可靠泛化。",'',
        *table,'',*seed_table,'',
        '每组测试96次执行复用8个物体初态，三个种子不是独立新环境。主结果保留所有失败；测试没有参与选模型。',
        '两组训练分别24/24与23/24教师成功，48条均合格抬升。上述教师成功不等于学得策略成功。',
        '旧v6跨组合闭合位置误差中位数约83mm，同编号约6–8mm。组合覆盖实验检验这个问题，不提前保证泛化。',
        '平缓搬运教师开发测试10/12，原教师11/12，候选未过采用门槛；主实验仍使用原教师。',
        '4项新隔离测试与3项原片接入路径检查分别通过。初次隔离测试因旧同名模块被导入而失败，已改为按明确文件位置加载；初次失败收据保留。',
        f"独立原始目标点标签试验：{pilot['pilot_successes']}/12成功、{pilot['pilot_lifts']}/12抬升；扩展门槛通过：{pilot['gate_passed']}。该开发试验另计，不并入264次正式评测。",
        '独立环境复现通过：迁移源代码后，两组各固定一个开发任务，8组输入/动作数组和完整评分轨迹逐值一致；两个193帧演示视频均可解码。环境在10月7日独立安装，同一台机器复用，不声称跨机器验证或完整重训。',
        '', '待补齐：44条新个人视频、真实拍摄日期/人工标签、新视频作用对照、实机证据及申请者资料。',
        '补拍目录：[训练](data/raw/X01_train)、[验证](data/raw/X02_val)、[封存测试](data/raw/X03_test)。',
        '', '[详细英文报告](README.md) · [完整审计](artifacts/results_v1/results.json) · [复现](REPRODUCTION_ZH.md)']
    (ROOT/'PROGRESS_ZH.md').write_text('\n'.join(zh)+'\n',encoding='utf-8')
    narrative=(
        '# Application project narrative — local draft\n\n'
        'I built a reproducible robot-learning study around personally recorded tabletop pick-and-place videos and a physically simulated Panda arm. '
        'The latest controller is a small image- and goal-conditioned CNN–GRU trained by behavior cloning. It consumes current RGB, robot state, the prescribed goal and the previous command, with no simulator object truth or scripted phase supplied at deployment.\n\n'
        'My 16 new phone demonstrations provide pre-action target geometry. I also tested using their coarse timing and image progress to generate robot demonstrations; the added timing did not yield a consistent policy-success benefit. '
        'Robot commands are physically executed simulated demonstrations, not claimed reconstructions of human 3D actions.\n\n'
        'A source–goal recombination diagnostic exposed an important limitation: the original same-number policy missed the grasp when the destination changed. '
        'I then froze an equal 24-trajectory comparison between four paired tasks and 12 training-covered combinations, keeping the learner, three seeds, update budget and evaluation rules fixed. '
        f'The held-out combination counts were {fraction(held_a)} and {fraction(held_b)}, respectively. '
        f"Overall test success was {fraction(result['totals']['test']['paired_24']['all'])} versus {fraction(result['totals']['test']['coverage_24']['all'])}; broader coverage did not improve overall success. "
        'These executions share physical initial scenes across model seeds, so I report every seed and failure rather than treating them as independent trials.\n\n'
        'The repository contains source/config snapshots, data and checkpoint hashes, causal input/action traces, strict physical scoring and fixed-case reproduction commands. '
        'Earlier visual auxiliary-supervision and macro-action outcome-prediction studies are documented separately. The system does not claim VLA training, evaluated world-model planning or real-robot transfer. '
        'Independent new personal-video sessions remain to be collected.\n\n'
        'This is a project-description draft only. No applicant identity, education, work-authorization information or application form has been completed or submitted.\n')
    (ROOT/'APPLICATION_NARRATIVE_EN.md').write_text(narrative,encoding='utf-8')
    # Only the root navigation marker is updated; historical sections and v6
    # reports/configs/source hashes are not rewritten.
    root_readme=PROJECT/'README.md';before=root_readme.read_text(encoding='utf-8')
    backup=ROOT/'receipts/root_README_before_v3.md'
    if not backup.exists():
        backup.write_text(before,encoding='utf-8')
    marker='<!-- paired-policy-current:start -->'
    end='<!-- paired-policy-current:end -->'
    latest=(marker+'\n> **Latest local extension:** A matched training-coverage study now follows the four-goal v6 controller. '
        'See [current results and limits](experiments/cross_goal_composition_v3/README.md), '
        '[Chinese progress](experiments/cross_goal_composition_v3/PROGRESS_ZH.md) and '
        '[the preserved v6 review](experiments/paired_two_board_v2/REVIEW_GUIDE_EN.md). '
        'The new work is local and is not included in public release v1.0.0.\n'+end)
    assert before.count(marker)==1 and before.count(end)==1
    after=before[:before.index(marker)]+latest+before[before.index(end)+len(end):]
    root_readme.write_text(after,encoding='utf-8')
    write(ROOT/'receipts/reports_v1.json',dict(created_utc=now(),source_results_sha256=sha(out/'results.json'),
        pilot_manifest_sha256=sha(ROOT/'artifacts/waypoint_development_v1/manifest.json'),
        replay_receipt_sha256=sha(ROOT/'receipts/portable_replay_v2/receipt.json'),
        outputs={str(p.relative_to(PROJECT)).replace('\\','/'):sha(p) for p in [ROOT/'README.md',ROOT/'PROGRESS_ZH.md',ROOT/'APPLICATION_NARRATIVE_EN.md',root_readme]}))
    print(finding)


if __name__=='__main__':
    main()
