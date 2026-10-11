"""Register the independently versioned simulator and completed datasets."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.metadata,importlib.util,json,platform,shutil,subprocess,sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(name,revision,out):
    script=ROOT/name
    command=[sys.executable,str(script)]
    if revision:command+=['--revision',revision]
    result=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True,encoding='utf-8')
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    payload=json.loads(result.stdout)
    write(out,dict(payload,validator_sha256=sha(script),validated_utc=datetime.now(timezone.utc).isoformat()))
    return payload


def protocol():
    receipt=ROOT/'receipts/paired_sim_revision_20261008'
    if receipt.exists():raise ValueError('Protocol revision already registered')
    receipt.mkdir()
    files=['protocol.json','goal_catalog.json','provenance.json','README.md']
    for file in files:shutil.copy2(ROOT/file,receipt/file)
    task=read(ROOT/'configs/sim_task_v1.json')
    plan=read(ROOT/'protocol.json');catalog=read(ROOT/'goal_catalog.json');provenance=read(ROOT/'provenance.json')
    plan['status']='S01 RGB candidates/coarse review complete; independent four-goal physics and goal-conditioned action interface implemented; matched teacher data being collected; no learned policy trained'
    for goal,definition in task['goals'].items():
        for mapping in (plan['goals'],catalog['goals']):
            mapping[goal]['sim_xy_m']=definition['xy_m']
            mapping[goal]['sim_coordinate_source']=definition['source']
            mapping[goal]['human_grid_center_mm']=definition['human_grid_center_mm']
    for start,definition in task['starts'].items():catalog['starts'][start]['sim_xy_m']=definition['xy_m']
    catalog['coordinate_warning']='UVs remain nominal priors; human labels use per-clip S01_calibration_v2. Active simulator XY is configs/sim_task_v1.json: mean pre-action blue centers mapped into engineering-aligned grids. No real robot registration.'
    catalog['active_sim_config']='configs/sim_task_v1.json'
    plan['goal_input']['implementation_status']='implemented in paired_sim.py: current RGB plus 12D robot/goal/past-command vector; learned policy training pending'
    plan['success']['status']='full independent physical scorer implemented; four oracle target smokes successful and crossed-line wrong-goal negative rejected; no learned-policy result'
    plan['calibration']['status']='board-local plane calibration and image-space candidates available; real board-to-robot registration remains unknown'
    plan['simulator']={'config':'configs/sim_task_v1.json','physics_config':'configs/sim_physics_v1.json',
                       'scope':task['scope'],'layout_assumptions':task['layout_assumptions'],
                       'documentation':'SIM_DEMONSTRATIONS.md','active_code':'paired_sim.py'}
    provenance['paired_simulator']={'config':'configs/sim_task_v1.json','config_sha256':sha(ROOT/'configs/sim_task_v1.json'),
                                   'smoke':'artifacts/reports/paired_sim_smoke_v1/manifest.json',
                                   'dataset':'artifacts/reports/paired_robot_demos_v1/manifest.json',
                                   'before_revision_snapshot':str(receipt.relative_to(ROOT)).replace('\\','/'),
                                   'oracle_initial_xy_for_teacher_only':True,'learned_policy_trained':False}
    write(ROOT/'protocol.json',plan);write(ROOT/'goal_catalog.json',catalog);write(ROOT/'provenance.json',provenance)
    with (ROOT/'README.md').open('a',encoding='utf-8') as handle:
        handle.write('\n\n## 2026-10-08 四目标物理仿真与示范\n\n独立环境已实现目标条件输入、逐步动作接口和指定区域完整物理评分。四目标 oracle 演示通过，错误目标负例失败。正在生成匹配的两组各 16 条训练示范与 8 条共同仿真验证示范。见 [任务和复现](SIM_DEMONSTRATIONS.md)。新学习策略未训练；真实视频严格成功仍未确认。\n')
    validate('validate_protocol.py',None,receipt/'protocol_validation.json')
    validate('validate_paired_demonstrations.py','paired_sim_smoke_v1',ROOT/'artifacts/reports/paired_sim_smoke_v1/validation.json')
    tests=receipt/'isolated_test_run';shutil.copytree(PROJECT/'artifacts/diagnostics/tests',tests)
    summary=read(tests/'summary.json');assert len(summary)==9 and all(s['returncode']==0 for s in summary)
    counts=[sum(int(node.get('tests','0')) for node in ET.parse(path).getroot().iter('testsuite')) for path in tests.glob('*.xml')]
    assert sum(counts)==43
    write(receipt/'receipt.json',{'registered_utc':datetime.now(timezone.utc).isoformat(),
                                 'before_sha256':{file:sha(receipt/file) for file in files},
                                 'after_sha256':{file:sha(ROOT/file) for file in files},
                                 'test_suites':9,'tests_passed':43,
                                 'test_scope':'isolated native-runtime test run preceding provenance-only source hash addition; completed data validator checks added source hashes',
                                 'historical_receipts_preserved':True})
    print('Protocol registered; 43 isolated tests archived; smoke and protocol validated',flush=True)


def complete():
    report=ROOT/'artifacts/reports/paired_robot_demos_v1';manifest=read(report/'manifest.json')
    assert manifest['status']=='completed'
    assert not (report/'REPORT_ZH.md').exists(),'Do not overwrite completed report'
    validation=validate('validate_paired_demonstrations.py','paired_robot_demos_v1',report/'validation.json')
    reviewed=read(report/'video_review_sheets_partial.json')
    assert len(reviewed['sheets'])==4
    for item in reviewed['sheets']:
        assert sha(ROOT/item['review_sheet'])==item['review_sheet_sha256']
        assert sha(ROOT/item['video'])==item['video_sha256']
    write(report/'visual_review.json',{'reviewer':'Codex','reviewed_sheets':reviewed['sheets'],
                                      'tiles_inspected':24,'observations':'all four guides show physical pickup, carry, release and withdrawal into respective designated targets',
                                      'scope':'six sampled frames per preview; not frame-by-frame video certification; strict outcomes determined separately by physical scorer',
                                      'learned_policy_evidence':False})
    packages={}
    for name in ('numpy','panda-gym','imageio-ffmpeg','torch','torchvision','mediapipe'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    import cv2
    conda=[]
    for path in (PROJECT/'.env/conda-meta').glob('*.json'):
        item=read(path)
        if any(word in item['name'] for word in ('bullet','opencv','python')):
            conda.append({key:item[key] for key in ('name','version','build')})
    bullet_binary=Path(importlib.util.find_spec('pybullet').origin)
    write(report/'environment.json',{'recorded_utc':datetime.now(timezone.utc).isoformat(),
                                    'python':sys.version,'platform':platform.platform(),
                                    'packages':packages,'opencv_runtime':cv2.__version__,
                                    'conda_selected_packages':conda,'pybullet_binary_sha256':sha(bullet_binary),
                                    'scope':'environment inventory at completed data validation; no environment changes during collection'})
    pairs=[]
    for seed in range(60000,60016):
        rows=[next(e for e in manifest['episodes'] if e['episode_id']==f'{condition}_{seed}') for condition in ('simulation_only','video_guided')]
        baseline,guided=rows
        meta=read((ROOT/guided['file']).with_suffix('.metadata.json'))
        pairs.append({'scene_seed':seed,'goal_id':baseline['goal_id'],'initial_rgb_sha256':baseline['initial_rgb_sha256'],
                      'simulation_only_steps':baseline['steps'],'video_guided_steps':guided['steps'],
                      'human_episode_id':meta['human_episode_id'],'human_duration_s':meta['human_profile']['duration_s'],
                      'used_rgb_candidate_frames':len(meta['human_profile']['valid_source_frame_indices']),
                      'baseline_success':baseline['success'],'guided_success':guided['success']})
    write(report/'paired_comparison.json',{'pairs':pairs,'all_initial_rgb_identical':True,'all_executed_actions_or_lengths_differ':True,
                                         'validated_by':'validation.json','scope':'teacher datasets, not policy performance'})
    write(report/'training_index.json',{'revision':'paired_robot_demos_v1','dataset_manifest_sha256':sha(report/'manifest.json'),
                                       'allowed_npz_arrays':['rgb','observations','actions'],
                                       'forbidden_model_inputs':['stages','requested_commands','metadata','scoring','human_profile','object truth','absolute clock'],
                                       'rgb_shape':[224,224,3],'observation_dim':12,'action_dim':4,
                                       'episode_boundaries_required':True,'normalization_source':'train only',
                                       'sampling_recommendation':'sample episodes uniformly, then valid time within that episode; equal optimizer steps across conditions despite unequal episode lengths',
                                       'train_groups':{condition:[{key:e[key] for key in ('episode_id','file','dataset_sha256','scene_seed','goal_id','steps')} for e in manifest['episodes'] if e['condition']==condition]
                                                       for condition in ('simulation_only','video_guided')},
                                       'shared_validation':[{key:e[key] for key in ('episode_id','file','dataset_sha256','scene_seed','goal_id','steps')} for e in manifest['episodes'] if e['split']=='val'],
                                       'test':[],'model_trained':False})
    failures=[]
    for episode in manifest['episodes']:
        if episode['success']:continue
        path=ROOT/episode['file'];meta=read(path.with_suffix('.metadata.json'));trace=read(path.with_suffix('.scoring.json'))
        outcome=meta['outcome'];offset=np.array(outcome['final_object_xyz_m'][:2])-outcome['goal_xy_m']
        failures.append({'episode_id':episode['episode_id'],'split':episode['split'],'goal_id':episode['goal_id'],
                         'retained_in_original_dataset':True,'final_center_offset_mm':(offset*1000).tolist(),
                         'final_flags':outcome['final_flags'],'lifted':outcome['lifted'],
                         'stage_end_diagnostics':{stage:next(t for t in reversed(trace) if t['stage']==stage) for stage in ('lift','transport','place','release','withdraw')},
                         'interpretation':'designated-goal containment failed while remaining final physical flags passed; trace is consistent with carried-object lateral offset, not a proven contact-mechanics cause',
                         'no_threshold_changed_after_validation':True})
    write(report/'failure_analysis.json',{'failures':failures,'scope':'privileged diagnostic scoring only, never policy inputs'})
    size=sum(path.stat().st_size for path in (ROOT/'data/processed/paired_robot_demos_v1').iterdir() if path.is_file())
    rows='\n'.join(f"| {c} | {v['attempts']} | {v['successes']} |" for c,v in manifest['summary'].items())
    report_text=f'''# 四目标机器人示范完成报告

两组各 16 条训练示范和 8 条共同仿真验证示范已实际执行、保存并核验。四目标开发 smoke 的 4 条 oracle 正例通过；指定 G1、故意搬到 G2 的负例失败，即越过黑线不能代替指定目标成功。

| 数据条件 | 尝试数 | 专家示范成功数 |
| --- | ---: | ---: |
{rows}

以上是使用初始真值 XY 的专家示范计数，不是学习策略评测，也不是实际人类视频成功率。所有尝试保留，未筛掉失败。当前尚无新学习模型或性能提升结论。

共同验证集唯一失败为 `shared_sim_validation_61002`（P3→G3）：抬升、落桌、直立、低速度、释放和撤爪均合格，但最终中心沿 Y 偏移约 −10.1 mm，完整物体投影没有被目标区域包含，稳定时间为 0。搬运末期已存在约 8.8 mm 横向偏移，放下后约 10.1 mm；轨迹与抓持中的横向偏移一致，接触机理尚未单独验证。见 `failure_analysis.json`。失败回合保留，不重采为成功，也不放宽阈值；未来验证必须报告实际闭环任务成功，动作误差不能代替它。

## 实际产物

- 40 个完整回合，{validation['pre_step_action_records']} 条物理执行前 RGB/状态/动作记录。
- 数据目录：`data/processed/paired_robot_demos_v1/`，总计 {size / 1024**2:.1f} MiB。
- 16 对完全相同初始 RGB、相同场景 seed/目标/物理参数的训练场景；每对的动作序列或长度不同，见 `paired_comparison.json`。
- 视频条件仅使用申请人 S01/train 的 16 条原片；每条 metadata 保存 5 个直接来源文件的 SHA256：原片、接收记录、首帧标定、逐帧候选、粗事件复核。
- `source_snapshot/` 和 `manifest.json` 固定代码、配置、开始完成时间及全部输出哈希。已有 4 条视频引导预览和 5 条 smoke 预览，MP4 全帧解码通过。
- `training_index.json` 固定两组训练/共同验证完整回合索引和模型可读的三个数组；未来训练必须按有效回合长度取样，不能跨回合或用诊断字段。两组回合数相同但帧数可以不同，需固定相同优化步数并按回合均匀取样。
- `validation.json` 核验实际动作时序、当前 RGB、固定目标、历史命令、步幅/宽度限制、双侧抬升和指定目标连续稳定时间、来源及场景配对。
- 9 个隔离进程测试套件共 43 项通过，原日志/测试 XML 保存在 `receipts/paired_sim_revision_20261008/isolated_test_run/`。

## 对视频贡献的边界

视频提供离线搬运时长与可见物体的无量纲图像进度约束。缺失帧不补齐；直线路径、抓取/搬运高度和夹爪规则仍是共享工程设定。这没有恢复人类三维动作；真实板到机器人配准未知。当前只验证视频约束进入实际机器人动作数据，不能据此断言它提高学习效果。

输入接口只用当前 RGB、机器人自身状态、事先指定的目标及上一条有效命令；没有绝对时钟、专家阶段、物体真值或人类时间表。未来策略应只加载 `rgb`、`observations`、`actions`，阶段与真实物体评分禁止进入模型输入。

## 下一步

在固定架构、训练步数、策略种子及完整回合划分下，分别训练 simulation-only 和 video-guided 两组目标条件策略。先在共同仿真验证场景检查闭环执行，再冻结独立评测场景与标准；学习策略在部署时通过 `step_action` 执行动作，不调用专家阶段控制。S02/S03/S04 尚未采集，不能将这 8 条仿真验证示范当成真实独立验证或最终测试。只有同编号源/目标数据，结果不得表述为任意或未见目标泛化。

复现命令与详细定义见 [SIM_DEMONSTRATIONS.md](../../../SIM_DEMONSTRATIONS.md)。旧实验和所有历史接收/标定/轨迹收据保留。
'''
    (report/'REPORT_ZH.md').write_text(report_text,encoding='utf-8')
    for file in ('protocol.json','provenance.json','README.md'):shutil.copy2(ROOT/file,report/f'before_completion_{file}')
    plan=read(ROOT/'protocol.json');plan['status']='S01 candidates/coarse review and 40 matched robot teacher episodes completed/validated; no learned policy trained'
    plan['simulator']['dataset_revision']='paired_robot_demos_v1';plan['simulator']['validation']='artifacts/reports/paired_robot_demos_v1/validation.json'
    provenance=read(ROOT/'provenance.json');provenance['paired_simulator']['dataset_status']='completed and validated'
    provenance['paired_simulator']['dataset_sha256']=sha(report/'manifest.json')
    provenance['paired_simulator']['validation_sha256']=sha(report/'validation.json')
    write(ROOT/'protocol.json',plan);write(ROOT/'provenance.json',provenance)
    overview=(ROOT/'README.md').read_text(encoding='utf-8')
    overview=overview.replace('正在生成匹配的两组各 16 条训练示范与 8 条共同仿真验证示范。','匹配的两组各 16 条训练示范与 8 条共同仿真验证示范已生成并通过核验。')
    (ROOT/'README.md').write_text(overview,encoding='utf-8')
    with (ROOT/'README.md').open('a',encoding='utf-8') as handle:
        handle.write(f'\n\n两组各 16 条训练、8 条共同仿真验证示范已完成，共 {validation["pre_step_action_records"]} 条真实执行动作记录；16 对初始 RGB 一致、动作/长度均有差异，来源哈希和评分核验通过。见 [示范报告](artifacts/reports/paired_robot_demos_v1/REPORT_ZH.md)。这是训练数据准备，尚未测量学习策略成功率。\n')
    validate('validate_protocol.py',None,report/'protocol_validation.json')
    write(report/'completion_receipt.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),
                                          'report_sha256':sha(report/'REPORT_ZH.md'),
                                          'validator_sha256':sha(ROOT/'validate_paired_demonstrations.py'),
                                          'environment_sha256':sha(report/'environment.json'),
                                          'raw_human_modified':False,'policy_trained':False,
                                          'protocol_sha256':sha(ROOT/'protocol.json'),'provenance_sha256':sha(ROOT/'provenance.json')})
    print(json.dumps({'completed':True,'episodes':40,'action_records':validation['pre_step_action_records'],'dataset_mib':round(size/1024**2,1),'summary':manifest['summary']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('protocol','complete'),required=True)
    args=parser.parse_args();protocol() if args.phase=='protocol' else complete()
