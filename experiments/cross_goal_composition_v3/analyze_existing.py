"""Read frozen v6 traces; write a separate post-hoc diagnostic, never alter scores."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'paired_two_board_v2'
SOURCE = OLD / 'artifacts/reports/paired_recurrent_goal_rgbaug_v6/consistent_goal_recombination_v1'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = ROOT / 'artifacts/v6_failure_analysis_v1'
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    cases, comparisons, teachers = [], [], []
    for folder in sorted(SOURCE.iterdir()):
        if not (folder / 'episodes.json').exists():
            continue
        hashes[str((folder/'episodes.json').relative_to(OLD))] = sha(folder/'episodes.json')
        records = read(folder / 'episodes.json')
        initial_commands = {}
        for row in records:
            trace_path = folder / (row['case_id'] + '_scoring.json')
            assert sha(trace_path) == row['scoring_sha256']
            trace = read(trace_path)
            xy = np.array([t['ee_xyz_m'][:2] for t in trace])
            obj = np.array([t['object_xyz_m'][:2] for t in trace])
            z = np.array([t['ee_xyz_m'][2] for t in trace])
            widths = np.array([t['finger_width_m'] for t in trace])
            distances = np.linalg.norm(xy-obj, axis=1)*1000
            closing = np.flatnonzero(widths < .030)
            low = np.flatnonzero(z < .025)
            record = dict(run=folder.name, case=row['case_id'], seen_pair=row['seen_pair'],
                          success=row['success'], lifted=row['lifted'],
                          any_bilateral=any(t['bilateral_grasp_contact'] for t in trace),
                          min_low_ee_object_xy_mm=float(distances[low].min()) if len(low) else None,
                          first_closed_xy_error_mm=float(distances[closing[0]]) if len(closing) else None,
                          first_closed_ee_z_mm=float(z[closing[0]]*1000) if len(closing) else None,
                          first_closed_step=int(closing[0]+1) if len(closing) else None,
                          final_goal_error_mm=float(np.linalg.norm(obj[-1]-row['goal_xy_m'])*1000))
            if row['oracle']:
                record['stages'] = {}
                for stage in ('close','lift','transport','place','release','withdraw'):
                    tt = [t for t in trace if t['stage']==stage]
                    if tt:
                        last = tt[-1]
                        record['stages'][stage] = dict(
                            object_minus_ee_xy_mm=((np.array(last['object_xyz_m'])-last['ee_xyz_m'])[:2]*1000).tolist(),
                            ee_goal_xy_error_mm=float(np.linalg.norm(np.array(last['ee_xyz_m'])[:2]-row['goal_xy_m'])*1000),
                            object_goal_xy_error_mm=float(np.linalg.norm(np.array(last['object_xyz_m'])[:2]-row['goal_xy_m'])*1000),
                            bilateral=last['bilateral_grasp_contact'])
                teachers.append(record)
            else:
                array_path = folder / (row['case_id'] + '_inputs_actions.npz')
                assert sha(array_path) == row['dataset_sha256']
                with np.load(array_path) as arrays:
                    first = arrays['requested_commands'][0,:2].astype(float)
                    vectors = arrays['pre_step_vectors']
                    assert np.allclose(vectors[:,4:6],row['goal_xy_m'])
                    initial_commands[row['case_id']] = (first,np.array(row['goal_xy_m']),vectors[0,:4])
                    record['first_requested_xy_m'] = first.tolist()
                cases.append(record)
        for case,(command,goal,robot) in initial_commands.items():
            p,g = case.split('_')
            if p[1] == g[1]:
                continue
            base,base_goal,base_robot = initial_commands[p+'_G'+p[1]]
            assert np.allclose(robot,base_robot,atol=1e-7)
            dg = goal-base_goal
            relative_change = (command-goal)-(base-base_goal)
            comparisons.append(dict(run=folder.name,case=case,
                initial_absolute_command_shift_mm=float(np.linalg.norm(command-base)*1000),
                goal_shift_mm=float(np.linalg.norm(dg)*1000),
                goal_translation_compensation_fraction=float(-relative_change@dg/(dg@dg))))
    aggregates={}
    for condition in ('simulation_only','video_guided'):
        for seen in (True,False):
            subset=[c for c in cases if c['run'].startswith(condition) and c['seen_pair']==seen]
            errors=[c['first_closed_xy_error_mm'] for c in subset if c['first_closed_xy_error_mm'] is not None]
            aggregates[condition+('_paired' if seen else '_crossed')]={
                'n':len(subset),'successes':sum(c['success'] for c in subset),
                'lifts':sum(c['lifted'] for c in subset),
                'any_bilateral':sum(c['any_bilateral'] for c in subset),
                'ever_closed_below_30mm':len(errors),
                'median_first_closed_xy_error_mm':float(np.median(errors)) if errors else None,
                'low_ee_within_10mm_object':sum(c['min_low_ee_object_xy_mm'] is not None and c['min_low_ee_object_xy_mm']<=10 for c in subset)}
    result=dict(scope='post-hoc scoring-trace diagnosis; thresholds are descriptive, not revised success criteria',
        definitions={'closed':'actual finger width <30mm','low':'EE Z <25mm','near':'horizontal EE/object distance <=10mm',
                     'compensation':'1 means network-relative XY exactly cancels decoder goal translation at the first step; RGB highlight also changes'},
        aggregates=aggregates,cases=cases,first_step_comparisons=comparisons,teacher_cases=teachers,
        source_episode_sha256=hashes,analysis_sha256=sha(Path(__file__)))
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# v6 跨组合失败定位（事后分析）','',
        '使用原来 96 次学习策略执行和 16 次教师执行的完整评分轨迹；没有重新运行或修改成功判据。真值仅用于本页诊断。', '',
        '| 条件 | 成功/执行 | 合格抬升 | 曾双指接触 | 低位接近物体 ≤10mm | 首次夹爪宽度 <30mm 时 XY 误差中位数 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,a in aggregates.items():
        err=a['median_first_closed_xy_error_mm']
        lines.append(f"| {name} | {a['successes']}/{a['n']} | {a['lifts']} | {a['any_bilateral']} | {a['low_ee_within_10mm_object']} | {err:.2f} mm |" if err is not None else f'| {name} | no closure |')
    lines += ['', '这些接近/闭合阈值是解释轨迹的事后诊断，不代替原严格物理成功标准。',
        '同一起点改变目标时，数值目标和视觉高亮同时改变。因此初始动作变化说明任务条件影响了接近过程，不能单独证明某一输入或网络结构是唯一原因。',
        '当前训练只含同编号配对，起点与目标完全相关。下一轮用等轨迹数的同编号对照与 12 组合覆盖组，检验增加组合覆盖的作用。',
        '', '## 教师失败', '']
    for t in teachers:
        if not t['success']:
            lines.append(f"- {t['case']}: lifted={t['lifted']}, final center error={t['final_goal_error_mm']:.2f} mm; placement-stage offset={t['stages'].get('place')}")
    lines += ['', '教师初始物体 XY 为显式特权输入，后续不读取物体修正动作。教师放置偏移应在新增开发场景排查，不能使用新留出组合调参。',
              '', '[全部数值及来源哈希](results.json)']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'aggregates':aggregates,'first_step_shift_median_mm':float(np.median([c['initial_absolute_command_shift_mm'] for c in comparisons])),
                      'compensation_median':float(np.median([c['goal_translation_compensation_fraction'] for c in comparisons])),
                      'teacher_failures':[t for t in teachers if not t['success']]},ensure_ascii=False))


if __name__=='__main__':
    main()
