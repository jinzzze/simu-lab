"""Paired RGB/true-state outcome prediction on the existing held-out command set.

No new physical actions are executed: outcomes belong to the original collection.
Commands are replayed unchanged, not selected by an RGB policy or this model.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, sys
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sim import GraspEnv
from src.predictor_process import PredictorProcess
from src.world_model import MacroOutcomePredictor
from src.world_model.evaluation import metrics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def evaluate_predictions(predictions, initial, final, targets):
    # Ground truth enters only this scoring function after predictions are saved.
    probabilities = np.column_stack([predictions['lift_probability'], predictions['success_probability']])
    return metrics(predictions['final_xy_m'] - initial, probabilities, final - initial, targets)


def main():
    config_path = ROOT / 'configs/grasp_world_model_rgb_evaluation.json'
    cfg = read(config_path)
    out = ROOT / 'artifacts/reports/grasp_world_model_v1/rgb_evaluation'
    if (out / 'results.json').exists():
        raise FileExistsError('Preserve completed RGB diagnostic; no overwrite')
    out.mkdir(parents=True, exist_ok=True)
    manifest = read(ROOT / 'data/manifests/grasp_world_model_v1.json')
    dataset = ROOT / 'data/processed/grasp_world_model_v1/test.jsonl'
    assert manifest['status'] == 'completed'
    assert sha(dataset) == manifest['jsonl_sha256']['test']
    assert sha(ROOT / 'src/sim/grasp_env.py') == manifest['sim_source_sha256']
    assert sha(ROOT / 'configs/grasp_sim.json') == manifest['sim_config_sha256']
    wm_path, visual_path = ROOT / cfg['world_model'], ROOT / cfg['visual_checkpoint']
    reference = read(ROOT / 'artifacts/reports/grasp_world_model_v1/results.json')
    demo = read(ROOT / 'artifacts/reports/grasp_world_model_v1/integration/result.json')
    assert sha(wm_path) == reference['ensemble_sha256'] == demo['world_model_sha256']
    assert sha(visual_path) == demo['visual_checkpoint_sha256']
    sources = [Path(__file__), config_path, ROOT / 'src/world_model/numpy_model.py',
               ROOT / 'src/world_model/evaluation.py', ROOT / 'src/predictor_process.py',
               ROOT / 'scripts/rgb_predictor_worker.py', ROOT / 'src/sim/grasp_env.py']
    metadata = {'plan': cfg, 'started_utc': datetime.now(timezone.utc).isoformat(),
                'visual_checkpoint_sha256': sha(visual_path), 'world_model_sha256': sha(wm_path),
                'records_sha256': sha(dataset),
                'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources}}
    write(out / 'plan_snapshot.json', metadata)
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    scenes = list(range(cfg['scenes']['start'], cfg['scenes']['start'] + cfg['scenes']['count']))
    expected = [(scene, action) for scene in scenes for action in range(5)]
    assert [(r['scene_seed'], r['action_index']) for r in rows] == expected
    estimated_states, commands, scenes_log = [], [], []
    predictor = MacroOutcomePredictor(wm_path)
    env = GraspEnv(image_size=224)
    try:
        with PredictorProcess(visual_path, out / 'predictor_stderr.log') as visual:
            for scene in scenes:
                records = [r for r in rows if r['scene_seed'] == scene]
                rgb = env.reset(scene)['rgb']
                rgb_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
                assert all(r['initial_rgb_sha256'] == rgb_hash for r in records), 'Re-rendered scene differs'
                estimated = visual(rgb)
                cv2.imwrite(str(out / f'scene_{scene}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                scenes_log.append({'scene_seed': scene, 'rgb_sha256': rgb_hash, 'estimated_xy_m': estimated.tolist()})
                estimated_states.extend([estimated.copy() for _ in records])
                commands.extend([r['command_xy_m'] for r in records])
    finally:
        env.close()
    estimated_states, commands = np.asarray(estimated_states), np.asarray(commands)
    predicted = predictor(estimated_states, commands)
    np.savez_compressed(out / 'input_predictions.npz', estimated_xy_m=estimated_states,
                        command_xy_m=commands, **predicted)
    write(out / 'scene_predictions.json', scenes_log)
    # Join outcome labels only after the RGB-based prediction artifact exists.
    initial = np.asarray([r['initial_xy_m'] for r in rows], dtype=np.float32)
    final = np.asarray([r['final_xy_m'] for r in rows], dtype=np.float32)
    targets = np.asarray([[r['lifted'], r['success']] for r in rows], dtype=np.float32)
    oracle_predictions = predictor(initial, commands)
    rgb_metrics = evaluate_predictions(predicted, initial, final, targets)
    true_metrics = evaluate_predictions(oracle_predictions, initial, final, targets)
    assert abs(true_metrics['xy_mean_error_mm'] - reference['evaluations']['test']['ensemble']['xy_mean_error_mm']) < 1e-3
    assert abs(true_metrics['success']['brier'] - reference['evaluations']['test']['ensemble']['success']['brier']) < 1e-6
    visual_errors = np.linalg.norm(estimated_states[::5] - initial[::5], axis=1) * 1000
    result = {**metadata, 'status': 'completed', 'scenes': len(scenes), 'transitions': len(rows),
              'new_physical_executions': 0, 'all_initial_rgb_hashes_match': True,
              'ground_truth_supplied_to_rgb_prediction': False,
              'metrics': {'rgb_estimated_state': rgb_metrics, 'true_initial_state': true_metrics},
              'visual_error_mm': {'mean': float(visual_errors.mean()), 'p90': float(np.percentile(visual_errors, 90)), 'max': float(visual_errors.max())},
              'predictions_sha256': sha(out / 'input_predictions.npz')}
    write(out / 'results.json', result)
    lines = ['# RGB 输入对世界模型预测的影响', '',
        '这是已有测试转换的追加诊断，不训练、不改动作、不追加物理执行。固定使用已预选的 D_seed7 视觉模型，重新渲染 32 个初始场景；每张 RGB 的哈希均与原始转换数据一致。', '',
        '世界模型仅接收 RGB 估计的初始 XY 与已有绝对抓取命令。原始采集命令由仿真真实位置加预定偏移产生；这里重放命令，不能称为 RGB 策略自主选动作。', '',
        '| 输入 | 终点平均误差 mm | P90 mm | 成功标签分类准确率 | 成功 Brier |',
        '|---|---:|---:|---:|---:|']
    for key, label in [('true_initial_state', '真实初始 XY（参照）'), ('rgb_estimated_state', 'RGB 估计 XY')]:
        row = result['metrics'][key]
        lines.append(f"| {label} | {row['xy_mean_error_mm']:.3f} | {row['xy_p90_error_mm']:.3f} | {row['success']['accuracy_at_0_5']*100:.3f}% | {row['success']['brier']:.4f} |")
    lines += ['', f"32 个初始场景的视觉定位平均误差为 {visual_errors.mean():.3f} mm。160 条动作结果共享 32 个场景，实际成功标签 {int(targets[:,1].sum())}/160；表中准确率是结果分类，绝不是机器人任务成功率。", '',
        '本诊断在看过真实状态报告和一个联动例子后定义；只有一个视觉检查点，测试外观固定，不证明概率校准、真实环境泛化、策略改进或 A/B/C/D 的世界模型差异。', '',
        '[完整指标与哈希](results.json)']
    (out / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'metrics': result['metrics'], 'visual_error_mm': result['visual_error_mm']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
