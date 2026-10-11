"""Refresh navigation/status from actual manifests, preserving previous documents."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
SPECS=[('v1a：绝对命令扩散','policy_v1a.json'),('v2：相对位移扩散','policy_delta_v2.json'),
       ('v3：相对位移循环 BC','policy_recurrent_v3.json'),('v4：目标相对绝对位置循环 BC','policy_recurrent_goal_v4.json'),
       ('v5：物理坐标精度加权','policy_recurrent_goal_precision_v5.json'),
       ('v6：整轨迹颜色/光照增强','policy_recurrent_goal_rgbaug_v6.json')]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
receipt=ROOT/f'receipts/policy_progress_{stamp}';receipt.mkdir()
shutil.copy2(__file__,receipt/Path(__file__).name)
files=[PROJECT/'README.md',PROJECT/'docs/CURRENT_TASK.md',ROOT/'README.md',ROOT/'protocol.json',ROOT/'provenance.json',ROOT/'POLICY_PROGRESS_ZH.md',ROOT/'LEARNED_POLICY_REPORT_EN.md',ROOT/'policy_progress.json']
before={}
for p in files:
    if p.exists():
        name=str(p.relative_to(PROJECT)).replace('/','__').replace('\\','__');shutil.copy2(p,receipt/name);before[str(p.relative_to(PROJECT))]=sha(p)
results=[];total_completed=0;total_rollouts=0
for label,filename in SPECS:
    path=ROOT/'configs'/filename
    if not path.exists():continue
    cfg=read(path);revision=cfg['revision'];models=[]
    for condition in cfg['conditions']:
        for seed in cfg['training']['seeds']:
            folder=ROOT/f'artifacts/runs/{revision}/{condition}_seed{seed}'
            if (folder/'manifest.json').exists():models.append(read(folder/'manifest.json'))
    completed=sum(m['status']=='completed' for m in models);total_completed+=completed
    suites={};audited_suites={}
    for suite in ['validation',*cfg['evaluation']['suites'],*cfg['evaluation']['diagnostics'],*[x+'_control' for x in cfg['evaluation']['diagnostics']]]:
        groups={}
        for condition in cfg['conditions']:
            rows=[];done=0;expected_seeds=cfg['evaluation']['diagnostics'].get(suite.removesuffix('_control'),{}).get('seeds',cfg['training']['seeds'])
            for seed in expected_seeds:
                folder=ROOT/f'artifacts/runs/{revision}_evaluation/{suite}/{condition}_seed{seed}'
                if (folder/'episodes.jsonl').exists():rows +=[json.loads(x) for x in (folder/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
                if (folder/'manifest.json').exists():done+=int(read(folder/'manifest.json')['status']=='completed')
            definition=cfg['evaluation']['validation'] if suite=='validation' else cfg['evaluation']['suites'].get(suite,cfg['evaluation']['diagnostics'].get(suite.removesuffix('_control')))
            groups[condition]={'executions':len(rows),'successes':sum(r['success'] for r in rows),'lifts':sum(r['lifted'] for r in rows),
                               'models_completed':done,'models_expected':len(expected_seeds),'executions_expected':definition['count']*len(expected_seeds)}
        total_rollouts+=sum(x['executions'] for x in groups.values());suites[suite]=groups
        audit=ROOT/f'artifacts/reports/{revision}/{suite}_validation.json'
        audited_suites[suite]={'complete_receipt':audit.exists(),'receipt_sha256':sha(audit) if audit.exists() else None}
    report=ROOT/f'artifacts/reports/{revision}/validation_analysis_v1/REPORT_ZH.md'
    additional={}
    recombination=ROOT/f'artifacts/reports/{revision}/consistent_goal_recombination_v1/analysis_v1/results.json'
    if recombination.exists():
        data=read(recombination)
        additional['consistent_goal_recombination']={'report':recombination.with_name('REPORT_ZH.md').relative_to(ROOT).as_posix(),
            'results_sha256':sha(recombination),'groups':data['groups'],
            'primary_results_unchanged':data['primary_results_unchanged']}
    demo=ROOT/f'artifacts/reports/{revision}/four_goal_nominal_demo_v1/manifest.json'
    if demo.exists():
        data=read(demo)
        additional['four_goal_demo']={'status':data['status'],'manifest_sha256':sha(demo),
            'visual_review':data.get('visual_review'),'report':demo.with_name('REPORT_ZH.md').relative_to(ROOT).as_posix()}
    results.append({'label':label,'revision':revision,'config':str(path.relative_to(ROOT)),
        'config_sha256':sha(path),'training_completed':completed,'suites':suites,'audited_suites':audited_suites,'primary_suites':list(cfg['evaluation']['suites']),
        'validation_report':report.relative_to(ROOT).as_posix() if report.exists() else None,'additional_results':additional})
rows=['# 四目标学习策略：当前进度','',f'按实际工件更新于 {datetime.now(timezone.utc).isoformat()}。训练完成 {total_completed} 个模型；正式 suite 已写入 {total_rollouts} 次执行记录（含未结束批次，调试 smoke 和复跑另存）。',
    '', '16 条 S01 个人视频已用于离线时长/归一化进度约束；两组各 16 条机器人训练示范与 8 条共同仿真验证示范全部保存。手机视频没有提供物理机器人动作真值，不能称作人类三维动作恢复或手机 RGB 直接端到端策略训练。',
    '', '| 修订 | 完成训练 | 开发验证：仿真对照 | 开发验证：视频引导 | 状态 |','|---|---:|---:|---:|---|']
for result in results:
    groups=result['suites']['validation'];a=groups['simulation_only'];b=groups['video_guided']
    complete=all(x['models_completed']==x['models_expected'] for x in groups.values())
    def fmt(x):return f'{x["successes"]}/{x["executions"]}' if x['executions'] else '未执行'
    rows.append(f'| {result["label"]} | {result["training_completed"]}/6 | {fmt(a)} | {fmt(b)} | {"完整批次" if complete else "进行中／未完成，勿作最终结果"} |')
rows+=['', '每组完整开发验证为 3 个模型 × 同一 8 个场景，即 24 次执行，只有 8 个独立初始场景。所有固定最终检查点保留，没有择优种子；这些是开发数据。',
    '', '## 报告与新场景测试','']
for result in results:
    rows +=[f'### {result["label"]}','']
    if result['validation_report']:rows.append(f'[开发验证详细报告]({result["validation_report"]})')
    else:rows.append('详细报告等待该修订全部验证和输入/评分核验完成。')
    rows +=['','| suite | 仿真对照成功/已执行 | 视频引导成功/已执行 | 批次状态 |','|---|---:|---:|---|']
    for suite,groups in result['suites'].items():
        if suite=='validation':continue
        a=groups['simulation_only'];b=groups['video_guided']
        state='完整并核验' if result['audited_suites'][suite]['complete_receipt'] else f'未核验完整批次；每组计划{a["executions_expected"]}次'
        detailed=ROOT/f'artifacts/reports/{result["revision"]}/{suite}_analysis_v1/REPORT_ZH.md'
        suite_link=f'[{suite}]({detailed.relative_to(ROOT).as_posix()})' if detailed.exists() else suite
        rows.append(f'| {suite_link} | {a["successes"]}/{a["executions"]} | {b["successes"]}/{b["executions"]} | {state} |')
    for tag, label in [('input_diagnostics_v1','配对输入干预'),('stability_history_v1','严格稳定窗口历史'),
                       ('stage_error_v1','教师强制分阶段误差'),('v3_v4_matched_budget_v1','v3/v4 预算核验'),
                       ('v4_v5_loss_only_audit_v1','v4/v5 仅损失权重变化核验'),
                       ('v5_v6_augmentation_only_audit_v1','v5/v6 图像增强配对核验'),
                       ('four_goal_nominal_demo_v1','固定场景四目标演示（失败保留，不增加主表）'),
                       ('consistent_goal_recombination_v1/analysis_v1','一致目标 P/G 组合诊断（与主表分开）'),
                       ('nominal_pixel_overlap_sensitivity_v1','原始像素重合敏感性（事后分析）')]:
        report=ROOT/f'artifacts/reports/{result["revision"]}/{tag}/REPORT_ZH.md'
        if report.exists():rows.append(f'\n[{label}]({report.relative_to(ROOT).as_posix()})')
    rows+=['','0/0 表示未运行。新场景只有本修订冻结以后首次执行时才属于该修订的独立仿真场景；同一套开发场景重复用于改方法，不构成新的确认性证据。', '']
rows +=['## 必须保留的解释边界','',
    '`simulation_only` 这个内部名称表示不使用视频搬运时长/进度。两组共享从 S01 动作前蓝点标定得到的目标几何，因此它不是完全不使用个人视频信息的基线。此对照只能检验在共同任务布局之外额外加入搬运时序/进度的作用。',
    '', 'v4/v5/v6 使用不同的新场景种子集合；跨版本测试数字不是逐场景配对比较。两组条件在每个版本内使用相同场景。开发集用于修订方法，不能把复用开发集的提升称为独立确认。v6依据v5外观失败开发；已知颜色类型复测只换初态，另有训练前冻结的新暖色组合。',
    '',
    'v1a/v2 是扩散，v3/v4/v5/v6 是因果循环 BC。每个版本内两组配对；跨架构训练目标数和算力不同。v3/v4 共享模型、示范和整轨迹采样预算，但输出参数化/尺度改变。v5保持v4的表示/尺度与2500步预算，只改坐标损失权重；v6只改v5的颜色/光照增强配方。两组等优化步数/采样轨迹数，因长度不同实际有效帧目标数与计算时间可能不同，训练清单均记录。循环隐藏状态可以编码时序信息，没有外部绝对时钟或专家阶段输入。',
    '', '在线控制只读当前/历史 RGB、机器人自身状态、事先目标 XY 和上一有效命令；物体真值只在示范采集标签和评分/诊断中使用。所有评分仍要求合格抬升、完整轮廓进入指定 40 mm 方框、落桌、释放、撤离与稳定。',
    '', 'P1→G1 至 P4→G4 的起点和目标有关联。数值目标冲突实验只是输入敏感性诊断，不能证明任意目标泛化。没有测到收益时如实保留，不用最好演示代替总体结果。',
    '', '## 需要本人协作、已跳过','',
    '- S02 额外训练 16 条、S03 独立验证 16 条、S04 测试 12 条，仍未采集。','- 真实严格成功判定、实际拍摄日期和纸板到机器人坐标的标定未确认。','- 真实机器人执行、个人申请资料与最终提交没有完成。',
    '', '当前新增实验尚未发布；旧公共 release 与 A/B/C/D/R、宏动作模型和旧 DP 数字保持历史含义。',
    '', '[来源与当前状态](policy_progress.json) · [复现说明](POLICY_REPRODUCTION_ZH.md) · [数据生成说明](SIM_DEMONSTRATIONS.md) · [训练/控制信息路径](TRAINING_RUNTIME_DIAGRAM.md)',
    '', '[v5 固定首场景实际复跑](artifacts/reports/policy_replays/v5_fixed_first_scene_replay_v1/REPORT_ZH.md)：输入/动作数组与物理评分逐值匹配，独立保存完整视频；不增加主评测次数。']
(ROOT/'POLICY_PROGRESS_ZH.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
english=['# Paired four-goal learned-control extension','',f'Local evidence refreshed {datetime.now(timezone.utc).isoformat()}. This extension is separate from the historical A/B/C/D/R, macro-world-model and clock-conditioned DP results.',
    '', 'The applicant recorded 16 new phone clips: four each for P1→G1 through P4→G4. The two grids are 100×130 mm, with a measured 130 mm edge gap. The blue marks define target centers. Real board-to-robot registration and strict human outcome labels remain unverified.',
    '', 'The personal clips contribute offline transport timing and dimensionless image-progress constraints to a robot demonstration generator. Robot XYZ paths, heights and command labels are engineered and physically executed in simulation. The phone frames are not the visual training inputs of these learned controllers; this is not recovered human 3D action, direct human-to-robot retargeting or a VLA.',
    '', 'Sixteen matched simulation-only and sixteen video-guided training episodes share initial images/positions, physics, IK, limits and scoring. Eight common simulator validation demonstrations are retained, including one teacher failure. The dataset has 10,789 pre-action RGB/state/effective-command records. Each condition has three paired final-EMA models; no best seed/checkpoint selection.',
    '', 'The internal simulation_only condition omits video transport timing/progress, but both conditions share task geometry calibrated from pre-action blue marks in S01 phone frames. It is not a wholly personal-data-free baseline. The contrast estimates the added role of timing/progress beyond that common geometry.',
    '', 'At runtime, current/past simulated RGB, robot proprioception, the task-provided goal XY and previous command are permitted. The selected goal is also highlighted in simulated RGB. Recurrent variants maintain learned causal memory, which may encode temporal information; there is no externally supplied absolute clock, expert phase or object truth. Numerical goal corruption deliberately conflicts with the original image/scoring goal, so it is an input-sensitivity diagnostic, not goal-generalization evidence.',
    '', '| Revision | Training complete | Simulation-only development success | Video-guided development success | Status |','|---|---:|---:|---:|---|']
for item in results:
    a=item['suites']['validation']['simulation_only'];b=item['suites']['validation']['video_guided']
    done=all(x['models_completed']==x['models_expected'] for x in (a,b))
    english.append(f'| {item["revision"]} | {item["training_completed"]}/6 | {a["successes"]}/{a["executions"]} | {b["successes"]}/{b["executions"]} | {"complete" if done else "partial; not a final estimate"} |')
english+=['', 'A complete group has 24 executions across three models reusing eight initial scenes, not 24 independent environments. These validation scenes were reused to develop later revisions. Cross-architecture training targets/compute differ; v3/v4 share model/data/episode sampling but change action parameterization and scale. V5 keeps v4 targets/model/data/update budget and changes coordinate loss weights; V6 changes only the training RGB augmentation recipe. Within each condition pair, optimizer updates and sampled episodes match, but different trajectory lengths create different numbers of valid frame targets and compute; these counts are recorded. V4/v5/v6 use distinct new-scene sets, so cross-revision test counts are not a paired same-scene comparison. V6 was developed from v5 known-color-shift failures; the same shift family is not an unseen appearance family. One fresh warm color combination was frozen before v6 training. Descriptive differences do not establish significance, video benefit or real-world transfer.',
    '', '## Frozen new-scene suites', '', '| Revision | Suite | Simulation-only strict success | Video-guided strict success | Status |','|---|---|---:|---:|---|']
for item in results:
    for suite in item['primary_suites']:
        a=item['suites'][suite]['simulation_only'];b=item['suites'][suite]['video_guided']
        if not a['executions'] and not b['executions']:continue
        state='complete and audited' if item['audited_suites'][suite]['complete_receipt'] else f'partial; {a["executions_expected"]} per group planned'
        report=ROOT/f'artifacts/reports/{item["revision"]}/{suite}_analysis_v1/REPORT_ZH.md'
        name=f'[{suite}]({report.relative_to(ROOT).as_posix()})' if report.exists() else suite
        english.append(f'| {item["revision"]} | {name} | {a["successes"]}/{a["executions"]} | {b["successes"]}/{b["executions"]} | {state} |')
english+=['', 'Nominal suites reuse 16 new initial scenes across three models; appearance and wider-start suites each reuse eight. Input-intervention suites use seed7 only, with normal-input paired controls, and are separately reported. Exact RGB overlap audits retain all cases; distinct simulator seeds do not guarantee distinct pixels. Independent here refers to newly generated simulator initial scenes, not independent personal recordings or a new robot/domain.',
    '', 'Strict success requires a qualified bilateral-contact lift, full object-footprint containment in the designated 40 mm square, upright/table placement, release, withdrawal and stability. The learned policy alone chooses actions; scoring does not repair or terminate it. Source/checkpoint hashes, causal input arrays, actual commands, separate physical scoring and all failures are retained.',
    '', 'The v1a fixed first scene was actually replayed: all stored input/action arrays and success/lift/final-condition decisions exactly matched. This is a same-host reproducibility diagnostic; it adds no new benchmark execution. [Replay receipt](artifacts/reports/policy_replays/v1a_first_scene_replay_v1/REPORT_ZH.md).',
    '', 'V5 also replayed fixed development scene61000 with seed7: strict physical success and exact agreement of all eight stored input/action arrays and final scoring. All video frames were decoded and six evenly spaced frames were visually inspected; this is not full-motion human adjudication. This fixed single-scene demonstration is not a success-rate estimate. [Receipt and video](artifacts/reports/policy_replays/v5_fixed_first_scene_replay_v1/REPORT_ZH.md).',
    '', 'The current new-scene suite counts, every completed per-seed report, open cooperation items and reproduction commands are linked in [current progress](POLICY_PROGRESS_ZH.md) and [reproduction](POLICY_REPRODUCTION_ZH.md). A zero/zero suite count means unexecuted. Results remain local; no new application or release was submitted.',
    '', 'S02 additional personal training, S03 independent validation and S04 test recordings remain uncollected. Actual capture timestamps, strict human outcomes, real-robot calibration/testing and applicant-only submission details are skipped pending the applicant. Same-number pairs confound source and goal, so arbitrary/unseen-goal following is not claimed.']
for item in results:
    additional=item['additional_results']
    if additional.get('four_goal_demo',{}).get('status')=='completed':
        english+=['',f'[{item["revision"]}: fixed four-task comparison]({additional["four_goal_demo"]["report"]}). Both conditions, seed7, first four nominal scenes; every shown success/failure retained. These are actual same-host replays, not additional benchmark executions.']
    if 'consistent_goal_recombination' in additional:
        data=additional['consistent_goal_recombination'];groups=data['groups']
        english+=['','## Consistent source/goal recombination diagnostic','',
                 'All16 P/G tasks share four new object-start poses. Numerical goal, RGB highlight and scoring goal agree from reset. All three fixed final models are evaluated; four trained pairings are controls and twelve combinations of known locations were not trained. No model selection/retraining or primary score change follows this diagnostic.', '',
                 '| Condition | Trained pairings | Untrained combinations of known locations |','|---|---:|---:|']
        for condition,g in groups.items():
            english.append(f'| {condition} | {g["seen_successes"]}/{g["seen_executions"]} | {g["unseen_successes"]}/{g["unseen_executions"]} |')
        english+=['',f'[All cases, input/source audit and matrices]({data["report"]}). The96 learned executions and16 privileged initial-XY teacher executions are separate from primary suites. Repeated object starts are not96 independent environments. The teacher uses its own control duration; it is a physical feasibility diagnostic, not a matched-budget learned score or mathematical upper bound. Known-location combinations do not establish arbitrary continuous goals, unseen goal positions or real-robot transfer.']
(ROOT/'LEARNED_POLICY_REPORT_EN.md').write_text('\n'.join(english)+'\n',encoding='utf-8')
write(ROOT/'policy_progress.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),'revisions':results,
      'actual_personal_recordings':16,'real_strict_success_labels':0,'real_independent_sessions_collected':0,
      'trained_models_completed':total_completed,'full_suite_executions_recorded':total_rollouts,'document_before_snapshot':str(receipt.relative_to(ROOT))})
for path,link in [(PROJECT/'README.md','experiments/paired_two_board_v2/POLICY_PROGRESS_ZH.md'),(PROJECT/'docs/CURRENT_TASK.md','../experiments/paired_two_board_v2/POLICY_PROGRESS_ZH.md'),(ROOT/'README.md','POLICY_PROGRESS_ZH.md')]:
    text=path.read_text(encoding='utf-8')
    start='<!-- paired-policy-current:start -->';end='<!-- paired-policy-current:end -->'
    if path==PROJECT/'README.md':
        note=f'{start}\n> **New local extension, 2026-10-08:** Four-goal RGB/goal-conditioned learned control is now trained and physically evaluated. See the [English evidence and limits](experiments/paired_two_board_v2/LEARNED_POLICY_REPORT_EN.md) and [all current results]({link}). Historical results below retain their original scope. Sixteen new personal clips are collected; independent real validation remains open.\n{end}\n'
    else:
        note=f'{start}\n> **2026-10-08 当前新增实验：** 四目标 RGB/目标条件学习策略已进入训练和实际仿真验证；最新全部种子结果见[当前策略进度]({link})。下文旧阶段状态及原 A/B/C/D/R、旧 DP 结果保留其历史含义。S01 共 16 条个人视频；独立真实验证尚未采集。\n{end}\n'
    if start in text:text=text[:text.index(start)]+note+text[text.index(end)+len(end):].lstrip('\n')
    else:
        at=text.find('\n')+1;text=text[:at]+'\n'+note+'\n'+text[at:]
    path.write_text(text,encoding='utf-8')
protocol=read(ROOT/'protocol.json');protocol['status']='S01 candidates and 40 matched robot teacher episodes validated; learned policies training/evaluation with actual status in POLICY_PROGRESS_ZH.md'
protocol['learned_policy_progress']='policy_progress.json';write(ROOT/'protocol.json',protocol)
provenance=read(ROOT/'provenance.json');provenance['policy_status']=f'{total_completed} learned checkpoints completed; see policy_progress.json for partial/full evaluation status'
provenance['learned_policy_progress']='policy_progress.json';provenance['paired_simulator']['learned_policy_trained']=total_completed>0
write(ROOT/'provenance.json',provenance)
write(receipt/'manifest.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),'before_sha256':before,
      'after_sha256':{str(p.relative_to(PROJECT)):sha(p) for p in files if p.exists()},'source_sha256':sha(__file__),
      'historical_numbers_preserved':True,'scoring_and_collected_data_unchanged':True})
print(json.dumps({'completed_models':total_completed,'executions_recorded':total_rollouts,'receipt':str(receipt)},ensure_ascii=False))
