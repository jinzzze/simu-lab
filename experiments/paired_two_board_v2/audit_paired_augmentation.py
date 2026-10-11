"""Verify v5/v6 fixed budgets and actual within-v6 color-transform pairing."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    a = read(ROOT / 'configs/policy_recurrent_goal_precision_v5.json')
    b = read(ROOT / 'configs/policy_recurrent_goal_rgbaug_v6.json')
    for key in ('model', 'training_index', 'dataset_manifest', 'normalization', 'action_representation'):
        assert a[key] == b[key]
    for key in ('steps', 'batch_size', 'learning_rate', 'weight_decay', 'grad_clip', 'optimizer', 'coordinate_loss_weights'):
        assert a['training'][key] == b['training'][key]
    source = {}; pairs = []
    for seed in b['training']['seeds']:
        new = {}
        for condition in b['conditions']:
            paths = [ROOT / f'artifacts/runs/{cfg["revision"]}/{condition}_seed{seed}/manifest.json' for cfg in (a, b)]
            old, current = [read(p) for p in paths]
            assert all(m['status'] == 'completed' for m in (old, current))
            for key in ('initial_weights_sha256', 'episode_fraction_stream_sha256', 'valid_action_targets_seen',
                        'steps_completed', 'parameters', 'dataset_manifest_sha256'):
                assert old[key] == current[key], (condition, seed, key)
            for p in paths: source[str(p.relative_to(ROOT))] = sha(p)
            new[condition] = current
            pairs.append({'condition': condition, 'seed': seed, 'initial_weights_sha256': current['initial_weights_sha256'],
                          'episode_stream_sha256': current['episode_fraction_stream_sha256'],
                          'valid_action_targets_seen': current['valid_action_targets_seen'], 'steps': current['steps_completed'],
                          'parameters': current['parameters'], 'augmentation_parameter_stream_sha256': current['augmentation_parameter_stream_sha256']})
        assert new['simulation_only']['augmentation_parameter_stream_sha256'] == new['video_guided']['augmentation_parameter_stream_sha256']
    out = ROOT / f'artifacts/reports/{b["revision"]}/v5_v6_augmentation_only_audit_v1'
    assert not out.exists(); out.mkdir(parents=True)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'old_revision': a['revision'], 'new_revision': b['revision'],
              'pairs': pairs, 'old_augmentation': a['augmentation'], 'new_augmentation': b['augmentation'],
              'source_sha256': source, 'analysis_sha256': sha(__file__),
              'within_revision_color_parameters_exactly_paired': True,
              'scope': 'same architecture/labels/normalizer/loss/update/episode streams; augmentation recipe changed; conditions still have different valid frame counts and compute'}
    (out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# v5/v6 图像增强配对核验', '', '六对运行的完整初始权重、整轨迹采样流、有效目标数量、更新次数、模型参数数量和数据来源均匹配。模型、标签、动作表示、归一化和坐标损失权重保持一致；变化限定为训练 RGB 的颜色/光照增强配方。', '',
             'v6 同种子的两条件使用完全相同的实际增强参数流。每条轨迹的所有帧共享同一组亮度、对比度和通道增益，采样不读标签、视频未来或物理评分，不做几何变换。', '',
             '两条件仍因示范时长不同而有不同有效帧数与实际计算时间。验证集用于方法开发，v5/v6 新测试使用不同场景集合；该核验不提供统计显著性或逐场景配对的跨版本测试结论。', '', '[全部运行和来源收据](results.json)']
    (out / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8'); shutil.copy2(__file__, out / Path(__file__).name)
    print(json.dumps({'report': str(out / 'REPORT_ZH.md'), 'pairs_checked': len(pairs)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
