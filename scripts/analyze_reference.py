"""Separate, explicitly exploratory comparison with the no-phone-video reference."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from src.artifact_paths import historical_checkpoint_path_matches
OUT = ROOT / 'artifacts/reports/visual_ablation_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    primary = read(OUT / 'results.json')
    plan = read(ROOT / 'configs/imagenet_reference_plan.json')
    seeds = plan['training_seeds']; scene_seeds = primary['scene_seeds']
    ref_rows, ref_runs, checks = [], [], []
    def check(name, passed):
        checks.append({'check': name, 'passed': bool(passed)})

    data_hash = sha(ROOT / 'data/processed/sim_adaptation_v1.npz')
    evaluation_plan = read(ROOT / 'artifacts/runs/visual_reference_grasp_v1/evaluation_plan.json')
    for key, filename in [('evaluator_sha256', 'scripts/evaluate_visual_grasp.py'),
                          ('predictor_worker_sha256', 'scripts/rgb_predictor_worker.py'),
                          ('predictor_transport_sha256', 'src/predictor_process.py')]:
        check(f'evaluation implementation/{key}', evaluation_plan[key] == sha(ROOT / filename))
    check('evaluation groups/seeds/scenes', evaluation_plan['groups'] == ['R']
          and evaluation_plan['training_seeds'] == seeds and evaluation_plan['scene_seeds'] == scene_seeds)
    check('primary audit passed', primary['audit_passed'])
    for seed in seeds:
        ref = ROOT / 'artifacts/runs/visual_reference_state_v1' / f'R_seed{seed}'
        state = read(ref / 'manifest.json')
        a = read(ROOT / 'artifacts/runs/visual_state_v1' / f'A_seed{seed}' / 'manifest.json')
        for key in ['data_sha256', 'head_initial_sha256', 'batch_sequence_sha256', 'augmentation_final_rng_sha256',
                    'config', 'target_mean', 'target_scale', 'steps_completed']:
            checks.append({'check': f'seed{seed}/R-vs-A/{key}', 'passed': state[key] == a[key]})
        init_dir = ROOT / 'artifacts/runs/imagenet_reference_initialization' / f'R_seed{seed}'
        init = read(init_dir / 'manifest.json')
        a_pre = read(ROOT / 'artifacts/runs/visual_pretraining_v1' / f'A_seed{seed}' / 'manifest.json')
        check(f'seed{seed}/state complete and identity', state['status'] == 'completed'
              and state['group'] == 'R' and state['seed'] == seed)
        check(f'seed{seed}/actual data source', state['data_sha256'] == data_hash)
        check(f'seed{seed}/same training image count', state['train_frames'] == a['train_frames'] == 256)
        check(f'seed{seed}/initialization identity', init['status'] == 'initialization_only'
              and init['group'] == 'R' and init['seed'] == seed)
        check(f'seed{seed}/original ImageNet encoder', init['initial_encoder_sha256'] == a_pre['teacher_state_sha256'])
        check(f'seed{seed}/initialization checkpoint lineage', state['pretraining_sha256'] == sha(init_dir / 'checkpoint.pt')
              and historical_checkpoint_path_matches(state,(init_dir/'checkpoint.pt').relative_to(ROOT).as_posix()))
        for filename, expected in state['code_sha256'].items():
            check(f'seed{seed}/training source/{filename}', expected == sha(ROOT / filename)
                  and expected == a['code_sha256'].get(filename))
        checks.append({'check': f'seed{seed}/no phone video', 'passed': init['real_data_used'] is False and init['steps_completed'] == 0})
        folder = ROOT / 'artifacts/runs/visual_reference_grasp_v1' / f'R_seed{seed}'
        summary = read(folder / 'summary.json'); observed = rows(folder / 'episodes.jsonl')
        check(f'seed{seed}/episode identities', [(r['group'], r['training_seed'], r['scene_seed']) for r in observed]
              == [('R', seed, scene) for scene in scene_seeds])
        check(f'seed{seed}/summary complete and identity', summary['group'] == 'R'
              and summary['training_seed'] == seed and summary['scene_seeds'] == scene_seeds
              and summary['n'] == len(observed) == len(scene_seeds)
              and summary['successes'] == sum(r['success'] for r in observed))
        check(f'seed{seed}/outcome agreement', all(r['success'] == r['outcome']['success'] for r in observed))
        checks.append({'check': f'seed{seed}/checkpoint', 'passed': summary['checkpoint_sha256'] == hashlib.sha256((ref / 'checkpoint.pt').read_bytes()).hexdigest()})
        a_grasp = read(ROOT / 'artifacts/runs/visual_grasp_v1' / f'A_seed{seed}' / 'summary.json')
        for key in ('sim_code_sha256', 'sim_config_sha256'):
            check(f'seed{seed}/same {key}', summary[key] == a_grasp[key] == evaluation_plan[key])
        a_rows = rows(ROOT / 'artifacts/runs/visual_grasp_v1' / f'A_seed{seed}' / 'episodes.jsonl')
        check(f'seed{seed}/paired A episode identities', [(r['group'], r['training_seed'], r['scene_seed']) for r in a_rows]
              == [('A', seed, scene) for scene in scene_seeds])
        checks.append({'check': f'seed{seed}/same rendered RGB', 'passed': [r['initial_rgb_sha256'] for r in observed] == [r['initial_rgb_sha256'] for r in a_rows]})
        ref_rows.extend(observed); ref_runs.append(summary)
    if not all(c['passed'] for c in checks):
        raise RuntimeError(f'Reference fairness check failed: {[c for c in checks if not c["passed"]]}')
    errors = np.asarray([r['initial_xy_error_mm'] for r in ref_rows])
    total = {'group': 'R', 'n': len(ref_rows), 'successes': sum(r['success'] for r in ref_rows),
             'success_rate': float(np.mean([r['success'] for r in ref_rows])),
             'xy_mean_error_mm': float(errors.mean()), 'xy_median_error_mm': float(np.median(errors)),
             'xy_p90_error_mm': float(np.percentile(errors, 90))}
    paired = []
    all_primary = rows(OUT / 'episodes.jsonl')
    lookup = {(r['training_seed'], r['scene_seed']): r for r in ref_rows}
    for group in 'ABCD':
        subset = [r for r in all_primary if r['group'] == group]
        expected_keys = [(group, seed, scene) for seed in seeds for scene in scene_seeds]
        observed_keys = [(r['group'], r['training_seed'], r['scene_seed']) for r in subset]
        check(f'{group}/paired primary episode identities', observed_keys == expected_keys)
        if observed_keys != expected_keys:
            raise RuntimeError(f'Invalid primary episode identities for {group}')
        delta = [int(r['success']) - int(lookup[r['training_seed'], r['scene_seed']]['success']) for r in subset]
        error_delta = [r['initial_xy_error_mm'] - lookup[r['training_seed'], r['scene_seed']]['initial_xy_error_mm'] for r in subset]
        paired.append({'comparison': group + '-R', 'success_difference_pp': float(np.mean(delta)*100),
                       'xy_mean_error_difference_mm': float(np.mean(error_delta)),
                       'paired_video_group_wins': sum(x>0 for x in delta), 'paired_reference_wins': sum(x<0 for x in delta),
                       'per_seed_success_difference_pp': {str(seed): float(np.mean([int(r['success']) - int(lookup[seed, r['scene_seed']]['success']) for r in subset if r['training_seed'] == seed])*100) for seed in seeds}})
    result = {'scope': plan['scope'], 'reference': total, 'per_seed': ref_runs,
              'comparisons': paired, 'fairness_checks': checks, 'audit_passed': all(c['passed'] for c in checks),
              'runtime_note': 'Some training jobs overlapped in wall time. Elapsed training time is recorded for provenance, not a controlled speed benchmark.',
              'interpretation': 'R skips real-video pretraining but retains ImageNet and all supervised simulator adaptation. Differences do not establish real-world transfer or generalization.'}
    (OUT / 'imagenet_reference.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# 额外诊断：不做真实视频预训练', '',
        'R 直接从 ImageNet ResNet18 初始化，跳过手机视频预训练；仿真适配图像、XY 标签、300 步预算、新头初始化、batch 与增强顺序、物理控制器均与对应种子配对。', '',
        '这项诊断在部分主实验定位指标已可见后追加，使用相同开发测试场景；不是新的未触碰验证。原定四组主结果保持原样。', '',
        f"R：{total['successes']}/{total['n']}，成功率 {total['success_rate']*100:.2f}%，平均 XY 误差 {total['xy_mean_error_mm']:.3f} mm。", '',
        '| 比较 | 成功率差 / 百分点 | 平均误差差 / mm |', '|---|---:|---:|']
    for row in paired:
        lines.append(f"| {row['comparison']} | {row['success_difference_pp']:+.2f} | {row['xy_mean_error_difference_mm']:+.3f} |")
    lines += ['', '误差差为负代表定位误差更小。仅 3 个训练种子、单一真实场景族与固定仿真外观，不能据此推广到真实机器人或其他任务。并行运行过部分训练进程，因此训练耗时仅作来源记录，不作严格速度比较。', '',
              '三个训练种子复用同一批 64 个仿真场景；192 次执行不是 192 个独立环境。成功率饱和只能说明当前样本中未观察到差异，不能证明方法等效。', '',
              f"来源与配对核对 {sum(c['passed'] for c in checks)}/{len(checks)} 项通过，包括初始化检查点、训练代码、完成状态及逐回合组别／种子／场景身份。审计只覆盖已记录的来源字段，不构成统计泛化证明。", '',
              '![额外参照](imagenet_reference.png)', '', '[原始数字与公平性核对](imagenet_reference.json)', '']
    (OUT / 'IMAGENET_REFERENCE_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    groups = [total] + primary['groups']
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    labels = ['R\nNo video','A\nRGB','B\n+ object','C\n+ hand','D\n+ both']
    colors = ['#9b9b9b','#526879','#007c91','#c48220','#8950a1']
    axes[0].bar(labels,[r['success_rate']*100 for r in groups],color=colors)
    axes[0].set(ylabel='Physical success (%)',ylim=(0,110))
    for i, row in enumerate(groups):
        axes[0].text(i,103,f"{row['successes']}/{row['n']}",ha='center',fontsize=9)
    axes[1].bar(labels,[r['xy_mean_error_mm'] for r in groups],color=colors)
    axes[1].set_ylabel('Mean XY error (mm)')
    fig.suptitle('Additional exploratory reference: ImageNet without phone-video pretraining')
    fig.supxlabel('Same 3 seeds, 256 adaptation scenes, 300 steps, 64 test scenes; no new held-out evaluation',fontsize=9)
    fig.savefig(OUT / 'imagenet_reference.png',dpi=180); plt.close(fig)
    print(json.dumps({'reference':total,'comparisons':paired},indent=2))


if __name__ == '__main__':
    main()
