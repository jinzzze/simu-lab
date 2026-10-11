"""Describe paired input interventions without treating them as generalization."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def norm_mm(vector):
    return float(np.linalg.norm(vector) * 1000)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--tag', default='input_diagnostics_v1')
    args = parser.parse_args()
    cfg = read(ROOT / args.config)
    out = ROOT / f'artifacts/reports/{cfg["revision"]}/{args.tag}'
    assert not out.exists(), 'Preserve existing reports; choose a new tag'
    sources = {args.config: sha(ROOT / args.config)}
    cases, groups = [], {}
    # All prespecified seed7 scenes are retained. No successful-case selection.
    for intervention, definition in cfg['evaluation']['diagnostics'].items():
        groups[intervention] = {}
        for suite in (intervention + '_control', intervention):
            receipt = ROOT / f'artifacts/reports/{cfg["revision"]}/{suite}_validation.json'
            assert receipt.exists(), f'First complete and audit {suite}'
            sources[str(receipt.relative_to(ROOT))] = sha(receipt)
        for condition in cfg['conditions']:
            counts = {'control_successes': 0, 'intervention_successes': 0,
                      'control_lifts': 0, 'intervention_lifts': 0, 'pairs': 0}
            for seed in definition['seeds']:
                folders = [ROOT / f'artifacts/runs/{cfg["revision"]}_evaluation/{suite}/{condition}_seed{seed}'
                           for suite in (intervention + '_control', intervention)]
                metas = [read(p / 'manifest.json') for p in folders]
                assert all(m['status'] == 'completed' and m['config'] == cfg for m in metas)
                assert metas[0]['checkpoint_sha256'] == metas[1]['checkpoint_sha256']
                rows = [[json.loads(x) for x in (p / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
                        for p in folders]
                expected = list(range(definition['start'], definition['start'] + definition['count']))
                assert all([r['scene_seed'] for r in rs] == expected for rs in rows)
                for folder in folders:
                    for name in ('episodes.jsonl', 'manifest.json'):
                        sources[str((folder / name).relative_to(ROOT))] = sha(folder / name)
                    for name in ('paired_recurrent_goal_policy.py', 'paired_recurrent_goal_policy_worker.py'):
                        p = folder / 'source_snapshot' / name
                        assert sha(p) == metas[0]['source_sha256'][name]
                        sources[str(p.relative_to(ROOT))] = sha(p)
                for control, changed in zip(*rows):
                    assert control['initial_rgb_sha256'] == changed['initial_rgb_sha256']
                    arrays = []
                    for folder, row in zip(folders, (control, changed)):
                        p = folder / f'scene_{row["scene_seed"]}_inputs_actions.npz'
                        assert sha(p) == row['dataset_sha256']
                        sources[str(p.relative_to(ROOT))] = row['dataset_sha256']
                        with np.load(p, allow_pickle=False) as data:
                            arrays.append({k: data[k] for k in ('requested_commands', 'effective_commands',
                                                                  'policy_input_vectors')})
                    a, b = arrays
                    va, vb = a['policy_input_vectors'][0, 0], b['policy_input_vectors'][0, 0]
                    # The first action has identical pre-physics history. Later
                    # trajectory differences include feedback and physical drift.
                    if intervention == 'blank_rgb':
                        assert np.array_equal(va, vb)
                    else:
                        assert np.array_equal(va[[0, 1, 2, 3, 8, 9, 10, 11]],
                                              vb[[0, 1, 2, 3, 8, 9, 10, 11]])
                        assert np.allclose(vb[6:8], vb[4:6] - vb[:2], atol=1e-7)
                    ca, cb = a['requested_commands'][0], b['requested_commands'][0]
                    relative_a = ca.copy(); relative_a[:2] -= va[4:6]
                    relative_b = cb.copy(); relative_b[:2] -= vb[4:6]
                    row = {'condition': condition, 'policy_seed': seed,
                           'scene_seed': control['scene_seed'], 'intervention': intervention,
                           'control_success': control['success'], 'intervention_success': changed['success'],
                           'control_lift': control['lifted'], 'intervention_lift': changed['lifted'],
                           'initial_rgb_sha256': control['initial_rgb_sha256'],
                           'original_goal_xy_m': va[4:6].tolist(), 'input_goal_xy_m': vb[4:6].tolist(),
                           'first_requested_xyz_change_mm': norm_mm(cb[:3] - ca[:3]),
                           'first_goal_relative_xyz_change_mm': norm_mm(relative_b[:3] - relative_a[:3]),
                           'first_width_change_mm': float(abs(cb[3] - ca[3]) * 1000),
                           'first_decoder_goal_shift_mm': norm_mm(vb[4:6] - va[4:6]),
                           'mean_effective_xyz_trajectory_difference_mm': float(np.linalg.norm(
                               b['effective_commands'][:, :3] - a['effective_commands'][:, :3], axis=1).mean() * 1000),
                           'control_final_object_xyz_m': control['final_object_xyz_m'],
                           'intervention_final_object_xyz_m': changed['final_object_xyz_m']}
                    cases.append(row); counts['pairs'] += 1
                    for prefix, source in (('control', control), ('intervention', changed)):
                        counts[prefix + '_successes'] += int(source['success'])
                        counts[prefix + '_lifts'] += int(source['lifted'])
            selected = [c for c in cases if c['intervention'] == intervention and c['condition'] == condition]
            counts['mean_first_goal_relative_xyz_change_mm'] = float(np.mean([
                c['first_goal_relative_xyz_change_mm'] for c in selected]))
            groups[intervention][condition] = counts
    out.mkdir(parents=True)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'revision': cfg['revision'],
              'groups': groups, 'cases': cases, 'source_sha256': sources,
              'analysis_sha256': sha(__file__),
              'scope': 'all prespecified seed7 paired interventions; descriptive input sensitivity, not arbitrary-goal generalization or independent training-seed evidence',
              'blank_rgb_storage': 'NPZ stores transmitted original RGB; frozen worker/policy replaces resized tensor with constant0.5 before inference',
              'numeric_goal_conflict': 'goal and goal-minus-EE vector change; RGB highlight and scorer goal stay original; absolute XY decoder also adds the corrupted numerical goal',
              'first_relative_output': 'decoded first XYZ minus supplied goal XY, separating direct goal translation from changed network output in physical units',
              'trajectory_difference_scope': 'later differences combine intervention, recurrent state, feedback and physics; not a pure encoder contribution'}
    (out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 配对输入干预诊断', '', '保留事先定义的 seed7 和每类四个场景。每个干预有相同初图、物理初始场景和检查点的独立正常输入对照；本表不增加主测试成功数。', '',
             '| 干预 | 条件 | 对照严格成功 | 干预严格成功 | 对照/干预合格抬升 | 首步相对目标 XYZ 输出变化均值 mm |',
             '|---|---|---:|---:|---:|---:|']
    for intervention, conditions in groups.items():
        for condition, c in conditions.items():
            lines.append(f'| {intervention} | {condition} | {c["control_successes"]}/{c["pairs"]} | {c["intervention_successes"]}/{c["pairs"]} | {c["control_lifts"]}/{c["intervention_lifts"]} | {c["mean_first_goal_relative_xyz_change_mm"]:.4f} |')
    lines += ['', 'RGB 干预在 Torch 工作者内部把缩放后的图像置为常数 0.5；存档 NPZ 仍保留实际传输的原图。首步隐藏状态清空且机器人向量一致，首步输出变化可检验本检查点对图像的敏感性；成功数变化不能单独说明视觉提供了何种信息。', '',
              '数值目标干预将 G1→G2→G3→G4→G1，同时更新 goal-minus-EE。图像中的原目标高亮和评分目标保持原样，属于相互冲突的输入诊断。坐标解码本身也加了被修改的目标，因此另行报告扣除目标平移后的网络输出变化；不能把目标平移造成的动作变化直接说成模型学会了目标指令。', '',
              '完整轨迹差异同时包含循环记忆、后续观测和物理反馈。只测一个训练种子、四个场景，没有独立统计显著性、任意目标泛化或真实机器人迁移结论。', '', '[逐回合数值与来源哈希](results.json)']
    (out / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, intervention in zip(axes, cfg['evaluation']['diagnostics']):
        subset = [c for c in cases if c['intervention'] == intervention]
        x = np.arange(len(subset)); width = .35
        ax.bar(x - width / 2, [c['first_requested_xyz_change_mm'] for c in subset], width, label='Decoded absolute XYZ')
        ax.bar(x + width / 2, [c['first_goal_relative_xyz_change_mm'] for c in subset], width, label='Network goal-relative XYZ')
        ax.set_xticks(x, [f'{c["condition"][:3]}\n{c["scene_seed"]}' for c in subset]); ax.set_ylabel('First-action change (mm)')
        ax.set_title(intervention); ax.legend(fontsize=8)
    fig.suptitle(f'{cfg["revision"]}: paired diagnostics, fixed seed7; not goal-generalization tests')
    fig.tight_layout(); fig.savefig(out / 'first_action_sensitivity.png', dpi=140); plt.close(fig)
    shutil.copy2(__file__, out / Path(__file__).name)
    print(json.dumps({'report': str(out / 'REPORT_ZH.md'), 'groups': groups}, ensure_ascii=False))

if __name__ == '__main__':
    main()
