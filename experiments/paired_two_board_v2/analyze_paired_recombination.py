"""Audit all consistent source/goal cases and separate trained-pair controls."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    plan = read(ROOT / 'configs/consistent_goal_recombination_v1.json')
    cfg = read(ROOT / plan['policy_config']); task = read(ROOT / 'configs/sim_task_v1.json')
    physics = read(ROOT / 'configs/sim_physics_v1.json')
    root = ROOT / f'artifacts/reports/{cfg["revision"]}/consistent_goal_recombination_v1'
    out = root / 'analysis_v1'; assert not out.exists()
    sources = {}; groups = {}; cases = []; matched = {}; matrices = {}
    labels = [(c, seed, False) for c in plan['conditions'] for seed in plan['policy_seeds']]
    labels.append(('privileged_teacher', None, True))
    for condition, seed, oracle in labels:
        name = 'privileged_initial_xy_teacher' if oracle else f'{condition}_seed{seed}'
        folder = root / name; meta = read(folder / 'manifest.json'); rows = read(folder / 'episodes.json')
        assert meta['status'] == 'completed' and meta['plan'] == plan and len(rows) == 16
        assert [r['case_id'] for r in rows] == [c['case_id'] for c in plan['cases']]
        assert sha(folder / 'episodes.json') == meta['episodes_sha256']
        for filename, digest in meta['source_sha256'].items(): assert sha(folder / 'source_snapshot' / filename) == digest
        for filename in ('manifest.json', 'episodes.json'):
            sources[str((folder / filename).relative_to(ROOT))] = sha(folder / filename)
        group = groups.setdefault(condition, {'seen_successes': 0, 'seen_executions': 0, 'unseen_successes': 0, 'unseen_executions': 0,
                                             'seen_lifts': 0, 'unseen_lifts': 0, 'per_model': {}})
        model = {'seen_successes': 0, 'unseen_successes': 0}; group['per_model'][str(seed)] = model
        matrix = matrices.setdefault(condition, np.zeros((4, 4), int))
        for case, row in zip(plan['cases'], rows):
            assert row['oracle'] == oracle and row['seen_pair'] == case['seen_pair'] and row['scene_seed'] == case['scene_seed']
            assert row['start_id'] == case['start_id'] and row['goal_id'] == case['goal_id']
            goal = np.asarray(task['goals'][case['goal_id']]['xy_m']); assert np.allclose(goal, row['goal_xy_m'], atol=1e-7)
            assert row['rgb_and_numerical_goal_consistent'] and not row['object_attached_or_teleported']
            trace_path = folder / f'{case["case_id"]}_scoring.json'
            assert sha(trace_path) == row['scoring_sha256']; trace = read(trace_path)
            sources[str(trace_path.relative_to(ROOT))] = row['scoring_sha256']
            assert len(trace) == row['steps']
            assert row['success'] == bool(row['lifted'] and all(row['final_flags'].values()) and
                row['final_stable_duration_s'] + 1e-9 >= physics['stable_duration_s'])
            assert trace[-1]['flags'] == row['final_flags']
            if not oracle:
                assert not row['runtime_object_truth'] and not row['absolute_clock_input']
                assert row['steps'] == 384 and row['inference_calls'] == 384 and not any(row['observation_interventions'].values())
                path = folder / f'{case["case_id"]}_inputs_actions.npz'; assert sha(path) == row['dataset_sha256']
                sources[str(path.relative_to(ROOT))] = row['dataset_sha256']
                with np.load(path, allow_pickle=False) as data:
                    vectors = data['pre_step_vectors']; applied = data['effective_commands']; requested = data['requested_commands']
                    assert vectors.shape == (384, 12) and applied.shape == requested.shape == (384, 4)
                    assert np.array_equal(data['inference_history_ticks'][:, 0], np.arange(384))
                    assert np.array_equal(data['policy_input_vectors'][:, 0], vectors)
                    assert np.allclose(vectors[:, 4:6], goal, atol=1e-7)
                    assert np.allclose(vectors[:, 6:8], goal - vectors[:, :2], atol=1e-7)
                    assert np.allclose(vectors[1:, 8:12], applied[:-1], atol=1e-7)
                    assert hashlib.sha256(data['policy_input_rgb'][0, 0].tobytes()).hexdigest() == row['initial_rgb_sha256']
                    assert all(t['stage'] == 'learned_action' for t in trace)
                    low = np.asarray(task['safety_limits']['xyz_low']); high = np.asarray(task['safety_limits']['xyz_high'])
                    absolute = requested[:, :3].clip(low, high); delta = absolute - vectors[:, :3]
                    factor = np.minimum(1., task['safety_limits']['max_ee_step_m'] / np.maximum(np.linalg.norm(delta, axis=1), 1e-12))
                    expected = vectors[:, :3] + delta * factor[:, None]
                    assert np.allclose(applied[:, :3], expected, atol=1e-6)
                    assert np.allclose(applied[:, 3], requested[:, 3].clip(0, .065), atol=1e-7)
            else:
                assert row['privileged_initial_object_xy'] and not row['online_object_truth_correction']
            key = case['case_id']; old_sha = matched.setdefault(key, row['initial_rgb_sha256'])
            assert old_sha == row['initial_rgb_sha256'], 'All models/teacher must share initial RGB for each case'
            part = 'seen' if case['seen_pair'] else 'unseen'
            group[part + '_successes'] += int(row['success']); group[part + '_executions'] += 1
            group[part + '_lifts'] += int(row['lifted']); model[part + '_successes'] += int(row['success'])
            matrix[int(case['start_id'][1]) - 1, int(case['goal_id'][1]) - 1] += int(row['success'])
            cases.append({'run': name, 'case_id': key, 'scene_seed': row['scene_seed'], 'seen_pair': row['seen_pair'],
                          'success': row['success'], 'lifted': row['lifted'], 'final_flags': row['final_flags'],
                          'final_object_xyz_m': row['final_object_xyz_m']})
    out.mkdir(); result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'policy_revision': cfg['revision'],
        'plan': plan, 'groups': groups, 'cases': cases, 'success_matrices': {k: v.tolist() for k, v in matrices.items()},
        'source_sha256': sources, 'analysis_sha256': sha(__file__), 'primary_results_unchanged': True,
        'scope': 'four object-start poses, all16 consistent goal tasks; unseen combinations of known P/G locations; all3 seeds retained; oracle privileged, separate duration/controller budget; no broad goal-generalization or real-robot claim'}
    (out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 一致目标输入：源/目标组合诊断', '', '数值目标、图像高亮和评分目标一致，目标均在开始前提供。固定四个新的物体初始位置，执行全部 16 种 P/G 组合；四种同编号组合为已训练配对对照，十二种交叉组合没有用于训练。三个固定最终模型全部保留。', '',
             '| 条件 | 已训练配对严格成功 | 未训练组合严格成功 | 已配对/未配对合格抬升 |', '|---|---:|---:|---:|']
    for condition, g in groups.items():
        lines.append(f'| {condition} | {g["seen_successes"]}/{g["seen_executions"]} | {g["unseen_successes"]}/{g["unseen_executions"]} | {g["seen_lifts"]}/{g["unseen_lifts"]} |')
    lines += ['', '前三个模型重复同一组场景，不能视为独立物体环境。交叉组合由已知的四个起点和四个目标构成，不能称为未见目标位置或任意连续目标泛化。该诊断不改主任务分数，也不触发再训练/选模型。', '',
              '教师明确使用初始物体真值 XY，随后按固定物理抓取轨迹执行，不在线修复物体位置。它仅检验这些目标是否可由该控制器执行，使用自己的示范长度；不等预算比较，也不是数学上界，更不是学习策略结果。', '',
              '与“只改数值向量、保留原目标高亮”的冲突输入实验分开。本页直接检查一致的任务指令；所有学习策略仍只读当前 RGB、机器人状态、上一命令和事前目标。', '',
              '![完整组合结果](success_matrices.png)', '', '[逐模型、逐组合、输入时序和来源哈希](results.json)']
    (out / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, condition in zip(axes, matrices):
        matrix = matrices[condition]; denominator = 1 if condition == 'privileged_teacher' else 3
        ax.imshow(matrix / denominator, vmin=0, vmax=1, cmap='YlGn')
        for i in range(4):
            for j in range(4): ax.text(j, i, f'{matrix[i, j]}/{denominator}', ha='center', va='center', fontsize=12)
            ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, edgecolor='black', linewidth=2))
        ax.set_xticks(range(4), ['G1', 'G2', 'G3', 'G4']); ax.set_yticks(range(4), ['P1', 'P2', 'P3', 'P4'])
        ax.set_xlabel('Consistent prescribed goal'); ax.set_ylabel('Object start')
        ax.set_title(condition, fontsize=10)
    fig.suptitle('All16 combinations / 4 object-start poses; outlined cells are trained pairs')
    fig.tight_layout(); fig.savefig(out / 'success_matrices.png', dpi=140); plt.close(fig)
    shutil.copy2(__file__, out / Path(__file__).name)
    print(json.dumps({'report': str(out / 'REPORT_ZH.md'), 'groups': groups}, ensure_ascii=False))

if __name__ == '__main__':
    main()
