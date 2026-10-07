"""Audit paired runs, summarize measured outcomes, and render research figures."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#526879', '#007c91', '#c48220', '#8950a1']
LABELS = ['A · RGB', 'B · + object', 'C · + hand', 'D · + both']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, content):
    Path(path).write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def failure_tags(row):
    outcome = row['outcome']
    if row['success']:
        return []
    tags = []
    if 'failure_reason' in outcome:
        return [outcome['failure_reason']]
    if not outcome['lifted']:
        tags.append('no_qualified_bilateral_lift')
    tags.extend('final_' + key for key, value in outcome['final_flags'].items() if not value)
    if outcome['final_stable_duration_s'] < .5:
        tags.append('insufficient_final_stability')
    return tags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'artifacts/runs')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/reports/visual_ablation_v1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = read(ROOT / 'configs/development_run_plan.json')
    groups, seeds, scenes = plan['groups'], plan['training_seeds'], plan['test_scene_seeds']
    checks = []
    def check(name, passed, detail=None):
        checks.append({'check': name, 'passed': bool(passed), 'detail': detail})

    evaluation_plan = read(args.runs / 'visual_grasp_v1/evaluation_plan.json')
    for key, path in [('evaluator_sha256', 'scripts/evaluate_visual_grasp.py'),
                      ('predictor_worker_sha256', 'scripts/rgb_predictor_worker.py'),
                      ('predictor_transport_sha256', 'src/predictor_process.py')]:
        check(f'evaluation implementation/{key}', evaluation_plan[key] == sha(ROOT / path))

    data_hashes = {stage: sha(ROOT / 'data/processed' / filename) for stage, filename in
                   [('pretraining', 'pilot_pretraining_v1.npz'), ('state', 'sim_adaptation_v1.npz')]}
    # Cache compressed arrays once; indexing an NpzFile repeatedly decompresses
    # the entire image array for every audit row.
    with np.load(ROOT / 'data/processed/sim_adaptation_v1.npz', allow_pickle=False) as source:
        sim = {key: source[key] for key in source.files}
    scene_map = {int(s): i for i, s in enumerate(sim['scene_seeds'])}
    records, manifests, per_run = [], {}, []
    success = np.zeros((len(groups), len(seeds), len(scenes)), dtype=float)
    errors = np.zeros_like(success)
    maximum_prediction_difference_m = 0.
    for si, seed in enumerate(seeds):
        for gi, group in enumerate(groups):
            key = f'{group}_seed{seed}'
            pre, state, grasp = [args.runs / f'visual_{part}_v1' / key for part in ['pretraining', 'state', 'grasp']]
            p, s, g = read(pre / 'manifest.json'), read(state / 'manifest.json'), read(grasp / 'summary.json')
            manifests[key] = (p, s, g)
            for part, manifest, budget in [('pretraining', p, plan['pretraining_steps']), ('state', s, plan['adaptation_steps'])]:
                check(f'{key}/{part}/fixed budget', manifest['steps_completed'] == budget
                      and manifest['config']['batch_size'] == plan['batch_size'])
                check(f'{key}/{part}/data', manifest['data_sha256'] == data_hashes[part])
                check(f'{key}/{part}/completed', manifest['status'] == 'completed')
                check(f'{key}/{part}/identity', manifest['group'] == group and manifest['seed'] == seed)
                for file, expected in manifest['code_sha256'].items():
                    check(f'{key}/{part}/source/{file}', sha(ROOT / file) == expected)
            check(f'{key}/pretraining checkpoint lineage', s['pretraining_sha256'] == sha(pre / 'checkpoint.pt'))
            check(f'{key}/deployment checkpoint lineage', g['checkpoint_sha256'] == sha(state / 'checkpoint.pt'))
            check(f'{key}/sim config', g['sim_config_sha256'] == sha(ROOT / 'configs/grasp_sim.json'))
            check(f'{key}/sim source', g['sim_code_sha256'] == sha(ROOT / 'src/sim/grasp_env.py'))
            check(f'{key}/scene order', g['scene_seeds'] == scenes)
            rows = [json.loads(line) for line in (grasp / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
            check(f'{key}/episode keys', [(r['group'], r['training_seed'], r['scene_seed']) for r in rows]
                  == [(group, seed, scene) for scene in scenes])
            offline = np.load(state / 'predictions.npz', allow_pickle=False)
            offline_map = {int(scene): xy for scene, xy in zip(offline['test_scene_seeds'], offline['test_xy_m'])}
            for j, row in enumerate(rows):
                scene = row['scene_seed']; index = scene_map[scene]
                check(f'{key}/{scene}/RGB matches held-out data', row['initial_rgb_sha256'] == hashlib.sha256(sim['images'][index].tobytes()).hexdigest())
                check(f'{key}/{scene}/label matches', np.allclose(row['label_xy_m'], sim['target_xy_m'][index], atol=1e-7))
                difference = float(np.max(np.abs(np.asarray(row['prediction_xy_m']) - offline_map[scene])))
                maximum_prediction_difference_m = max(maximum_prediction_difference_m, difference)
                check(f'{key}/{scene}/online-offline prediction', difference < 1e-5, difference)
                success[gi, si, j] = row['success']; errors[gi, si, j] = row['initial_xy_error_mm']
                row['failure_tags'] = failure_tags(row)
                records.append(row)
            check(f'{key}/summary agrees', g['n'] == len(rows) and g['successes'] == sum(r['success'] for r in rows))
            per_run.append({**g, 'pretraining_seconds': p['elapsed_seconds'], 'adaptation_seconds': s['elapsed_seconds']})
        for phase, index, fields in [('pretraining', 0, ['initial_model_sha256', 'teacher_state_sha256', 'batch_sequence_sha256', 'augmentation_final_rng_sha256', 'config', 'episodes']),
                                     ('state', 1, ['head_initial_sha256', 'batch_sequence_sha256', 'augmentation_final_rng_sha256', 'config', 'target_mean', 'target_scale'])]:
            for field in fields:
                values = [manifests[f'{group}_seed{seed}'][index][field] for group in groups]
                check(f'seed{seed}/{phase}/paired {field}', all(x == values[0] for x in values[1:]))
    check('planned total episode count', len(records) == plan['physical_evaluation_episodes'])
    audit = {'passed': all(c['passed'] for c in checks), 'checks': checks,
             'max_online_offline_prediction_difference_m': maximum_prediction_difference_m,
             'scope': 'hash/identity/budget/input parity checks, not proof of statistical generalization'}
    write(args.output / 'fairness_audit.json', audit)
    if not audit['passed']:
        failed = [c for c in checks if not c['passed']]
        raise RuntimeError(f'Fairness/provenance checks failed: {failed[:8]}; see fairness_audit.json')

    totals = []
    for gi, group in enumerate(groups):
        rows = [r for r in records if r['group'] == group]
        tags = Counter(tag for row in rows for tag in row['failure_tags'])
        totals.append({'group': group, 'n': int(success[gi].size), 'successes': int(success[gi].sum()),
                       'success_rate': float(success[gi].mean()),
                       'success_rate_by_seed': {str(seed): float(success[gi, si].mean()) for si, seed in enumerate(seeds)},
                       'xy_mean_error_mm': float(errors[gi].mean()), 'xy_median_error_mm': float(np.median(errors[gi])),
                       'xy_p90_error_mm': float(np.percentile(errors[gi], 90)),
                       'failure_tags_nonexclusive': dict(tags)})
    rng = np.random.default_rng(20261006)
    seed_resamples = rng.integers(0, len(seeds), size=(10000, len(seeds)))
    scene_resamples = rng.integers(0, len(scenes), size=(10000, len(scenes)))
    comparisons = []
    for left, right in [('D', 'A'), ('D', 'B'), ('D', 'C'), ('B', 'A'), ('C', 'A')]:
        a, b = groups.index(left), groups.index(right)
        delta = success[a] - success[b]
        boot = delta[seed_resamples[:, :, None], scene_resamples[:, None, :]].mean(axis=(1, 2)) * 100
        comparisons.append({'comparison': f'{left}-{right}', 'difference_percentage_points': float(delta.mean() * 100),
            'per_seed_difference_percentage_points': {str(seed): float(delta[si].mean() * 100) for si, seed in enumerate(seeds)},
            'crossed_bootstrap_95_percentile_pp': np.percentile(boot, [2.5, 97.5]).tolist(),
            'bootstrap_degenerate_all_observed_pairs_equal': bool(np.all(delta == delta.flat[0])),
            'paired_left_wins': int((delta > 0).sum()), 'paired_right_wins': int((delta < 0).sum()),
            'paired_ties': int((delta == 0).sum())})
    result = {'scope': 'exploratory pilot development, fixed budgets; no real held-out scene test',
              'training_seeds': seeds, 'scene_seeds': scenes, 'groups': totals, 'comparisons': comparisons,
              'per_run': per_run, 'total_episodes': len(records),
              'uncertainty': 'Descriptive crossed bootstrap: resample 3 training seeds and 64 shared scenes independently, preserving all group pairs; 10000 draws, RNG 20261006. Only 3 training seeds: intervals are unstable and not a robust population significance claim.',
              'limitations': ['One conservative real development scene family; 12 clips are not 12 independent environments.',
                              'Synthetic held-out positions share the same camera, lighting, object, yaw and physics.',
                              'Controller is programmed; neither VLA nor learned dynamics/world model is implemented.',
                              'The primary four-group design has no no-video reference; see the separate post-hoc ImageNet-only diagnostic in imagenet_reference.json, which reuses the development test scenes.',
                              'No manual dense ground-truth accuracy for auxiliary pseudo labels.'],
              'audit_passed': audit['passed']}
    write(args.output / 'results.json', result)
    with (args.output / 'episodes.jsonl').open('w', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    render_figures(args.output, success, errors, records, groups, seeds, args.runs)
    make_markdown(args.output, result, checks)
    print(json.dumps({'groups': totals, 'comparisons': comparisons, 'audit_checks': len(checks)}, indent=2))


def render_figures(out, success, errors, records, groups, seeds, runs):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 130, 'savefig.dpi': 180})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    for gi, group in enumerate(groups):
        axes[0].bar(gi, success[gi].mean() * 100, width=.60, color=COLORS[gi], alpha=.82)
        axes[0].scatter(gi + np.linspace(-.12, .12, len(seeds)), success[gi].mean(axis=1) * 100,
                        color='black', s=22, zorder=3)
        axes[0].text(gi, 103, f'{int(success[gi].sum())}/{success[gi].size}', ha='center', fontsize=10)
        axes[1].boxplot(errors[gi].ravel(), positions=[gi], widths=.5, patch_artist=True,
                        boxprops={'facecolor': COLORS[gi], 'alpha': .6}, medianprops={'color': 'black'},
                        flierprops={'marker': '.', 'markersize': 3})
    axes[0].set(ylabel='Physical task success (%)', ylim=(0, 111), title='Matched physical grasp evaluation')
    axes[1].set(ylabel='Initial XY error (mm)', title='RGB-only localization')
    for ax in axes:
        ax.set_xticks(range(4), LABELS); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('12 phone clips → shared simulation adaptation → 768 physical episodes', fontsize=12)
    fig.supxlabel('Exploratory development · 3 paired training seeds × 64 fixed-camera scenes · dots = seeds', fontsize=9)
    for ext in ('png', 'svg'):
        fig.savefig(out / f'comparison.{ext}')
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.7), sharex=True, sharey=True, layout='constrained')
    for gi, group in enumerate(groups):
        rows = [r for r in records if r['group'] == group]
        delta = np.asarray([np.asarray(r['prediction_xy_m']) - r['label_xy_m'] for r in rows]) * 1000
        passed = np.asarray([r['success'] for r in rows])
        axes[gi].scatter(delta[passed, 0], delta[passed, 1], s=13, alpha=.45, color=COLORS[gi], label='success')
        axes[gi].scatter(delta[~passed, 0], delta[~passed, 1], s=35, marker='x', color='#bd3937', label='failure')
        axes[gi].axhline(0, lw=.7, color='#aaaaaa'); axes[gi].axvline(0, lw=.7, color='#aaaaaa')
        axes[gi].set(title=LABELS[gi], xlabel='X prediction error (mm)', aspect='equal'); axes[gi].legend(fontsize=8)
    axes[0].set_ylabel('Y prediction error (mm)')
    fig.suptitle('Post-evaluation diagnosis: location error and physical outcome', fontsize=12)
    fig.savefig(out / 'error_outcome.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for gi, group in enumerate(groups):
        for pi, (part, field) in enumerate([('pretraining', 'consistency'), ('state', 'xy_scaled_smooth_l1')]):
            losses = []
            for seed in seeds:
                rows = [json.loads(x) for x in (runs / f'visual_{part}_v1' / f'{group}_seed{seed}' / 'losses.jsonl').read_text().splitlines()]
                losses.append([row[field] for row in rows])
            curve = np.asarray(losses).mean(axis=0)
            smooth = np.convolve(curve, np.ones(15)/15, mode='valid')
            axes[pi].plot(np.arange(15, len(curve)+1), smooth, color=COLORS[gi], label=LABELS[gi])
    axes[0].set(title='Shared RGB consistency loss', ylabel='Feature squared distance')
    axes[1].set(title='Shared simulation XY adaptation', ylabel='Normalized Smooth L1 loss')
    for ax in axes:
        ax.set_xlabel('Optimization step'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.supxlabel('Mean over 3 seeds; trailing 15-step average. Auxiliary losses are excluded from the left plot.', fontsize=9)
    fig.savefig(out / 'training_curves.png'); plt.close(fig)


def make_markdown(out, result, checks):
    lines = ['# 四组视觉辅助监督：探索性开发实验', '',
             '固定 3 个训练种子 × 每组 64 个相同场景，共 768 次真实物理仿真执行。下列数值来自逐回合日志，不是计划值或 oracle 结果。', '',
             '| 组别 | 成功次数 | 成功率 | XY 平均误差 / mm | XY P90 / mm |', '|---|---:|---:|---:|---:|']
    for row in result['groups']:
        lines.append(f"| {row['group']} | {row['successes']}/{row['n']} | {row['success_rate']*100:.2f}% | {row['xy_mean_error_mm']:.3f} | {row['xy_p90_error_mm']:.3f} |")
    lines += ['', 'A：共同 RGB 基础训练；B：加物体监督；C：加手关键点监督；D：两者都有。所有组均接受相同仿真 XY 标签适配。', '',
              '![成功率和定位误差](comparison.png)', '', '## 配对差异', '',
              '| 比较 | 差值 / 百分点 | seed 7 | seed 17 | seed 27 | 描述性 95% 区间 / 百分点 |', '|---|---:|---:|---:|---:|---|']
    for row in result['comparisons']:
        lo, hi = row['crossed_bootstrap_95_percentile_pp']
        per = row['per_seed_difference_percentage_points']
        lines.append(f"| {row['comparison']} | {row['difference_percentage_points']:+.2f} | {per['7']:+.2f} | {per['17']:+.2f} | {per['27']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ['', '区间通过分别重采样训练种子和共同场景、保持组间配对计算（10,000 次）。只有 3 个训练种子，区间仅用于描述；不能据此声称稳定的人群泛化或显著性。帧没有被当作独立回合。全部观测差异为零时，bootstrap 会退化成 [0,0]；这不代表总体不确定性为零，也不是等效性证明。', '',
              '## 每个训练种子', '', '| 组别 | 训练种子 | 成功次数 | 定位平均误差 / mm |', '|---|---:|---:|---:|']
    for row in result['per_run']:
        lines.append(f"| {row['group']} | {row['training_seed']} | {row['successes']}/{row['n']} | {row['xy_mean_error_mm']:.3f} |")
    lines += ['', '## 失败与训练曲线', '', '![定位误差与物理结果](error_outcome.png)', '', '失败标记可同时出现，不能把它们相加当作失败回合数。', '']
    for row in result['groups']:
        lines.append(f"- {row['group']}：{json.dumps(row['failure_tags_nonexclusive'], ensure_ascii=False)}")
    lines += ['', '![训练曲线](training_curves.png)', '', '## 数据与证据边界', '',
              '- 12 条本人手机视频，86.48 秒、2,162 帧；四组共同使用 438 张 5 Hz 完整 RGB。物体候选有效 398 张，手关键点输出 223 张。',
              '- 视频视为同一保守开发场景族，没有真实未见场景测试。实际采集批次仍未知。',
              '- 仿真适配 256 训练场景、64 验证场景；64 测试场景只改变初始位置，共用相机、背景、方块姿态和物理参数。',
              '- 每组每种子预训练 240 步、适配 300 步、batch 16；使用最终检查点，没有据测试结果挑选。',
              '- 控制器是固定物理抓取状态机，未实现 VLA、语言条件策略、世界模型或视频时序动作模仿。',
              '- 原定四组本身不能回答真实视频相对不使用视频的收益；已另行追加 [ImageNet-only 参照](IMAGENET_REFERENCE_ZH.md)，复用相同开发测试场景，不能视为新的确认性证据。',
              '- 伪标签候选覆盖率不等于定位准确率；稀疏目视检查不替代人工密集真值。',
              f"- 公平性与来源审计 {sum(c['passed'] for c in checks)}/{len(checks)} 项通过；见 [fairness_audit.json](fairness_audit.json)。", '',
              '完整数字：[results.json](results.json)；所有回合：[episodes.jsonl](episodes.jsonl)。', '']
    (out / 'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
