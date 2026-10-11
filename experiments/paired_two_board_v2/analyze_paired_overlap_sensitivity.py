"""Posthoc descriptive sensitivity to exact initial-frame training overlap."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, shutil

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--suite', default='nominal'); parser.add_argument('--tag', default='pixel_overlap_sensitivity_v1')
    args = parser.parse_args(); cfg = read(ROOT / args.config)
    reports = ROOT / f'artifacts/reports/{cfg["revision"]}'
    path = reports / f'{args.suite}_image_overlap_v1/results.json'; overlap = read(path)
    out = reports / f'{args.suite}_{args.tag}'; assert not out.exists(); out.mkdir()
    groups = {}
    for condition in cfg['conditions']:
        cases = overlap['groups'][condition]['cases']
        included = [c for c in cases if not c['training_any_frame_matches']]
        excluded = [c for c in cases if c['training_any_frame_matches']]
        groups[condition] = {'full_successes': sum(c['success'] for c in cases), 'full_executions': len(cases),
            'without_matching_initial_image_successes': sum(c['success'] for c in included),
            'without_matching_initial_image_executions': len(included),
            'excluded_executions': len(excluded), 'excluded_scene_ids': sorted({c['scene_seed'] for c in excluded}),
            'excluded_successes': sum(c['success'] for c in excluded),
            'unique_included_initial_images': len({c['initial_rgb_sha256'] for c in included}),
            'per_seed': {str(s): {'successes': sum(c['success'] for c in included if c['policy_seed'] == s),
                                 'executions': sum(c['policy_seed'] == s for c in included)} for s in cfg['training']['seeds']}}
    assert groups['simulation_only']['excluded_scene_ids'] == groups['video_guided']['excluded_scene_ids']
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'revision': cfg['revision'],
              'suite': args.suite, 'groups': groups, 'source_sha256': {str(path.relative_to(ROOT)): sha(path)},
              'analysis_sha256': sha(__file__), 'primary_counts_unchanged': True,
              'scope': 'posthoc descriptive sensitivity; exclude only exact full224x224 initial RGB matches to any training frame; does not prove physical-state identity or make remaining episodes independent'}
    (out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 初始图像重复的事后敏感性分析', '', '主表及其全部回合保留。本页另列排除“完整初图像素与任一训练帧完全相同”的回合后的描述数字；规则在看到像素核查后制定，因此不是事先主测试。', '',
             '| 条件 | 原主表严格成功 | 排除重复初图后 | 排除执行数 | 排除场景 |', '|---|---:|---:|---:|---|']
    for condition, g in groups.items():
        lines.append(f'| {condition} | {g["full_successes"]}/{g["full_executions"]} | {g["without_matching_initial_image_successes"]}/{g["without_matching_initial_image_executions"]} | {g["excluded_executions"]} | {g["excluded_scene_ids"]} |')
    lines += ['', '训练与测试场景编号不同，但渲染的物体位置可能在像素量化后相同。完全相同初图不等于数据文件复制或物理状态相同；排除后的数字也不能证明泛化，更不能把三个种子的重复场景当作独立样本。', '',
              '[各种子数字、来源和排除范围](results.json)']
    (out / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8'); shutil.copy2(__file__, out / Path(__file__).name)
    print(json.dumps({'report': str(out / 'REPORT_ZH.md'), 'groups': groups}, ensure_ascii=False))

if __name__ == '__main__':
    main()
