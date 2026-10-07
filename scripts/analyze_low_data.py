"""Report the fixed 32-image extension separately from the original experiment."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from src.artifact_paths import historical_checkpoint_path_matches
OUT = ROOT / 'artifacts/reports/visual_ablation_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    plan = read(ROOT / 'configs/low_data_plan.json')
    groups, seeds = plan['groups'], plan['training_seeds']
    data_hash = hashlib.sha256((ROOT / 'data/processed/sim_adaptation_32_v1.npz').read_bytes()).hexdigest()
    expected_scenes = list(range(30000, 30064))
    manifests, records, checks, per_run = {}, [], [], []
    def check(name, passed):
        checks.append({'check': name, 'passed': bool(passed)})

    data_manifest = read(ROOT / 'data/manifests/sim_adaptation_32_v1.json')
    source_data = ROOT / 'data/processed/sim_adaptation_v1.npz'
    check('subset source hash', data_manifest['source_sha256'] == sha(source_data))
    check('subset dataset hash', data_manifest['dataset_sha256'] == data_hash)
    check('subset plan matches', data_manifest['plan'] == plan)
    with np.load(source_data, allow_pickle=False) as original, np.load(ROOT / 'data/processed/sim_adaptation_32_v1.npz', allow_pickle=False) as subset:
        selected = (original['split'] != 'train') | (original['scene_seeds'] < 10032)
        check('subset arrays match source exactly', set(original.files) == set(subset.files)
              and all(np.array_equal(subset[key], original[key][selected]) for key in original.files))
        check('fixed 32-scene subset', np.array_equal(subset['scene_seeds'][subset['split'] == 'train'],
                                                   np.arange(10000, 10032)))
        check('subset manifest scene identities', subset['scene_seeds'].tolist() == data_manifest['scene_seeds']
              and subset['split'].tolist() == data_manifest['splits'])
    evaluation_plan = read(ROOT / 'artifacts/runs/visual_low32_grasp_v1/evaluation_plan.json')
    for key, filename in [('evaluator_sha256', 'scripts/evaluate_visual_grasp.py'),
                          ('predictor_worker_sha256', 'scripts/rgb_predictor_worker.py'),
                          ('predictor_transport_sha256', 'src/predictor_process.py')]:
        check(f'evaluation implementation/{key}', evaluation_plan[key] == sha(ROOT / filename))
    check('evaluation groups/seeds/scenes', evaluation_plan['groups'] == groups
          and evaluation_plan['training_seeds'] == seeds and evaluation_plan['scene_seeds'] == expected_scenes)
    for seed in seeds:
        for group in groups:
            key = f'{group}_seed{seed}'
            state_dir = ROOT / 'artifacts/runs/visual_low32_state_v1' / key
            grasp_dir = ROOT / 'artifacts/runs/visual_low32_grasp_v1' / key
            state, result = read(state_dir / 'manifest.json'), read(grasp_dir / 'summary.json')
            manifests[key] = state
            rows = [json.loads(line) for line in (grasp_dir / 'episodes.jsonl').read_text().splitlines()]
            primary_name = 'visual_reference_grasp_v1' if group == 'R' else 'visual_grasp_v1'
            primary_rows = [json.loads(line) for line in (ROOT / 'artifacts/runs' / primary_name / key / 'episodes.jsonl').read_text().splitlines()]
            init_name = 'imagenet_reference_initialization' if group == 'R' else 'visual_pretraining_v1'
            init_dir = ROOT / 'artifacts/runs' / init_name / key
            initialization = read(init_dir / 'manifest.json')
            primary_state_name = 'visual_reference_state_v1' if group == 'R' else 'visual_state_v1'
            primary_state = read(ROOT / 'artifacts/runs' / primary_state_name / key / 'manifest.json')
            expected_keys = [(group, seed, scene) for scene in expected_scenes]
            check_values = {
                'state complete and identity': state['status'] == 'completed' and state['group'] == group and state['seed'] == seed,
                'initialization identity': initialization['group'] == group and initialization['seed'] == seed,
                'initialization status': (initialization['status'] == 'initialization_only' and initialization['real_data_used'] is False
                                          and initialization['steps_completed'] == 0) if group == 'R' else initialization['status'] == 'completed',
                'pretraining lineage': state['pretraining_sha256'] == sha(init_dir / 'checkpoint.pt')
                                        and state['pretraining_sha256'] == primary_state['pretraining_sha256']
                                        and historical_checkpoint_path_matches(state,(init_dir/'checkpoint.pt').relative_to(ROOT).as_posix()),
                'same adaptation config across budgets': state['config'] == primary_state['config'],
                'training plan': state['low_data_plan'] == plan,
                'low-data training source': state['extra_source_sha256'] == sha(ROOT / 'scripts/run_low_data_adaptation.py'),
                'episode identities': [(r['group'], r['training_seed'], r['scene_seed']) for r in rows] == expected_keys,
                'primary episode identities': [(r['group'], r['training_seed'], r['scene_seed']) for r in primary_rows] == expected_keys,
                'summary identity': result['group'] == group and result['training_seed'] == seed and result['scene_seeds'] == expected_scenes,
                'outcome agreement': all(r['success'] == r['outcome']['success'] for r in rows),
                'data hash': state['data_sha256'] == data_hash,
                '32 training images': state['train_frames'] == 32,
                'fixed budget': state['steps_completed'] == 300 and state['config']['batch_size'] == 16,
                'scenes': [r['scene_seed'] for r in rows] == expected_scenes,
                'same RGB across budgets': [r['initial_rgb_sha256'] for r in rows] == [r['initial_rgb_sha256'] for r in primary_rows],
                'counts': result['n'] == len(rows) == 64 and result['successes'] == sum(r['success'] for r in rows),
                'deployment lineage': result['checkpoint_sha256'] == hashlib.sha256((state_dir / 'checkpoint.pt').read_bytes()).hexdigest(),
                'sim code': result['sim_code_sha256'] == hashlib.sha256((ROOT / 'src/sim/grasp_env.py').read_bytes()).hexdigest(),
                'sim config': result['sim_config_sha256'] == hashlib.sha256((ROOT / 'configs/grasp_sim.json').read_bytes()).hexdigest()}
            for name, passed in check_values.items():
                checks.append({'check': f'{key}/{name}', 'passed': bool(passed)})
            for filename, expected in state['code_sha256'].items():
                check(f'{key}/training source/{filename}', expected == sha(ROOT / filename)
                      and expected == primary_state['code_sha256'].get(filename))
            records.extend(rows); per_run.append(result)
        for field in ['data_sha256','head_initial_sha256','batch_sequence_sha256','augmentation_final_rng_sha256','config','target_mean','target_scale']:
            values = [manifests[f'{group}_seed{seed}'][field] for group in groups]
            checks.append({'check': f'seed{seed}/paired {field}', 'passed': all(v == values[0] for v in values[1:])})
    check('planned total episode count', len(records) == plan['physical_evaluation_episodes'])
    if not all(c['passed'] for c in checks):
        raise RuntimeError(f'Low-data audit failed: {[c for c in checks if not c["passed"]]}')
    totals = []
    for group in groups:
        rows = [r for r in records if r['group'] == group]
        totals.append({'group':group,'n':len(rows),'successes':sum(r['success'] for r in rows),
            'success_rate':float(np.mean([r['success'] for r in rows])),
            'xy_mean_error_mm':float(np.mean([r['initial_xy_error_mm'] for r in rows])),
            'xy_p90_error_mm':float(np.percentile([r['initial_xy_error_mm'] for r in rows],90))})
    lookup = {(r['group'],r['training_seed'],r['scene_seed']):r for r in records}
    pairs = []
    for a,b in [('D','A'),('D','B'),('D','C'),('A','R'),('D','R')]:
        per_seed = {str(seed): float(np.mean([int(lookup[a,seed,scene]['success'])-int(lookup[b,seed,scene]['success']) for scene in expected_scenes])*100) for seed in seeds}
        pairs.append({'comparison':a+'-'+b,'difference_pp':float(np.mean(list(per_seed.values()))),'per_seed_pp':per_seed})
    failed_rows = [r for r in records if not r['success']]
    scene_failures = Counter(r['scene_seed'] for r in failed_rows)
    scene_executions = Counter(r['scene_seed'] for r in records)
    failure_concentration = {
        'failed_executions': len(failed_rows),
        'unique_test_scenes': len(scene_executions),
        'unique_failed_scenes': len(scene_failures),
        'executions_per_scene': len(groups) * len(seeds),
        'scenes_failed_in_every_group_and_seed': sorted(scene for scene, count in scene_failures.items()
                                                      if count == scene_executions[scene]),
        'failures_by_scene': [
            {'scene_seed': scene, 'failures': scene_failures[scene], 'executions': scene_executions[scene],
             'failed_runs': [f"{r['group']}_seed{r['training_seed']}" for r in failed_rows if r['scene_seed'] == scene]}
            for scene in sorted(scene_failures)],
        'interpretation': 'Training seeds and groups reuse the same scene seeds; these are correlated executions, not independent environments.'}
    report = {'plan':plan,'groups':totals,'per_run':per_run,'comparisons':pairs,'audit':checks,
              'audit_passed': all(c['passed'] for c in checks), 'failure_concentration': failure_concentration,
              'subset_limitation': 'All training seeds reuse the same fixed 32-image subset; no resampling of alternative 32-image subsets was performed.',
              'scope':'Exploratory reused-test follow-up. Equal steps, different epochs. No selection by observed low-data outcomes.'}
    (OUT / 'low32_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (OUT / 'low32_episodes.jsonl').open('w',encoding='utf-8') as stream:
        for row in records:stream.write(json.dumps(row)+'\n')
    primary = read(OUT / 'results.json')['groups']
    primary += [read(OUT / 'imagenet_reference.json')['reference']]
    full = {r['group']:r for r in primary}
    lines = ['# 32 张仿真适配图像的追加实验','','该实验在观察到 256 图像参照的成功率饱和后固定，只运行一次预设预算。训练场景按索引取 10000–10031，不按结果筛选。验证与测试场景复用，因此这是探索性追加分析，不是新的未见测试。','',
             '五组使用相同 32 张图像、相同 300 步和 batch 16；A/B/C/D 复用已完成的真实视频编码器，R 直接从 ImageNet 开始。保持相同步数意味着样本重复次数增加，不能说成计算预算按数据量下降。','',
             '全部训练种子复用同一个固定 32 图像子集，没有重复抽取其他 32 图像子集。不同种子只覆盖训练随机性，因此不能据此宣称普遍的样本效率优势。','',
             '| 组别 | 32 张：成功次数 | 32 张：平均误差 mm | 256 张：成功次数 | 256 张：平均误差 mm |','|---|---:|---:|---:|---:|']
    for row in totals:
        other=full[row['group']]
        lines.append(f"| {row['group']} | {row['successes']}/{row['n']} | {row['xy_mean_error_mm']:.3f} | {other['successes']}/{other['n']} | {other['xy_mean_error_mm']:.3f} |")
    lines += ['','## 逐种子差异','','| 比较 | 平均差 / 百分点 | seed 7 | seed 17 | seed 27 |','|---|---:|---:|---:|---:|']
    for pair in pairs:
        d=pair['per_seed_pp'];lines.append(f"| {pair['comparison']} | {pair['difference_pp']:+.2f} | {d['7']:+.2f} | {d['17']:+.2f} | {d['27']:+.2f} |")
    lines += ['', '## 失败在共同场景中的分布', '',
              f"{len(failed_rows)} 次失败执行集中在 {len(scene_failures)} 个共同场景中；总共只有 {len(scene_executions)} 个不同测试场景，每个场景由 {len(groups)} 组 × {len(seeds)} 个训练种子重复执行。192 次组内执行不是 192 个独立环境。", '',
              '| 场景种子 | 失败执行次数 | 该场景总执行次数 |', '|---|---:|---:|']
    for item in failure_concentration['failures_by_scene']:
        lines.append(f"| {item['scene_seed']} | {item['failures']} | {item['executions']} |")
    always_failed = failure_concentration['scenes_failed_in_every_group_and_seed']
    if always_failed:
        lines += ['', '在全部组别与训练种子中都失败的场景：' + '、'.join(map(str, always_failed)) + '。']
    lines += ['','![样本预算比较](label_budget_comparison.png)','',
              f"来源与配对核对 {sum(c['passed'] for c in checks)}/{len(checks)} 项通过，包括固定子集与原数据逐数组一致性、初始化检查点、训练代码、完成状态及逐回合身份。", '',
              '三个训练种子、同一真实开发场景族、固定相机仿真分布；不据此断言显著性或真实机器人泛化。包括 R 在内共 960 次追加物理仿真执行。审计只覆盖已记录的来源字段，不构成统计泛化证明；见 [low32_results.json](low32_results.json)。','']
    (OUT/'LOW_DATA_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    colors=['#526879','#007c91','#c48220','#8950a1','#999999']
    for i,(row,color) in enumerate(zip(totals,colors)):
        other=full[row['group']]
        axes[0].plot([32,256],[row['success_rate']*100,other['success_rate']*100],marker='o',color=color,label=row['group'])
        axes[1].plot([32,256],[row['xy_mean_error_mm'],other['xy_mean_error_mm']],marker='o',color=color,label=row['group'])
    axes[0].set(ylabel='Physical success (%)',ylim=(0,105));axes[1].set_ylabel('Mean initial XY error (mm)')
    for ax in axes:
        ax.set(xscale='log',xlabel='Labeled simulator training images',xticks=[32,256]);ax.set_xticklabels(['32','256']);ax.xaxis.set_minor_locator(NullLocator());ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Exploratory label-budget comparison: 3 seeds × 64 shared scenes per point')
    fig.savefig(OUT/'label_budget_comparison.png',dpi=180);plt.close(fig)
    print(json.dumps({'groups':totals,'comparisons':pairs},indent=2))


if __name__=='__main__':main()
