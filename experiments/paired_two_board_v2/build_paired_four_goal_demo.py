"""Replay the first nominal scene of each paired task and caption every result."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, concurrent.futures, hashlib, json, shutil, subprocess, sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def decode(path):
    capture = cv2.VideoCapture(str(path)); frames = []
    fps = capture.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = capture.read()
        if not ok: break
        frames.append(frame)
    capture.release(); assert frames and fps > 0
    return frames, fps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/policy_recurrent_goal_precision_v5.json')
    parser.add_argument('--tag', default='four_goal_nominal_demo_v1')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(); assert 1 <= args.workers <= 2
    cfg = read(ROOT / args.config)
    report_root = ROOT / f'artifacts/reports/{cfg["revision"]}'
    assert (report_root / 'nominal_validation.json').exists(), 'Finish the full audited nominal suite first'
    summary = read(report_root / 'nominal_analysis_v1/results.json')
    start = cfg['evaluation']['suites']['nominal']['start']
    cases = [{'condition': condition, 'seed': 7, 'scene': scene,
              'tag': f'{args.tag}_{condition}_{scene}'}
             for scene in range(start, start + 4) for condition in cfg['conditions']]
    out = report_root / args.tag; assert not out.exists(), 'Preserve old demo; use a new tag'
    out.mkdir()
    meta = {'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'revision': cfg['revision'], 'config': cfg, 'cases': cases,
            'selection': 'first four nominal scene IDs, one each P1G1-P4G4, fixed seed7, both conditions; diagnostic assembly after primary evaluation, no outcome filtering',
            'benchmark_count': 'zero additional primary episodes; exact-reference replays only',
            'source_sha256': {Path(__file__).name: sha(__file__), 'replay_paired_policy.py': sha(ROOT / 'replay_paired_policy.py'),
                              args.config: sha(ROOT / args.config), 'nominal_analysis_v1/results.json': sha(report_root / 'nominal_analysis_v1/results.json'),
                              'nominal_validation.json': sha(report_root / 'nominal_validation.json')}}
    write(out / 'manifest.json', meta)
    shutil.copy2(__file__, out / Path(__file__).name)
    shutil.copy2(ROOT / args.config, out / 'policy_config.json')
    shutil.copy2(ROOT / 'replay_paired_policy.py', out / 'replay_source.py')

    def replay(case):
        command = [sys.executable, '-u', str(ROOT / 'replay_paired_policy.py'), '--config', args.config,
                   '--tag', case['tag'], '--condition', case['condition'], '--seed', str(case['seed']),
                   '--suite', 'nominal', '--scene', str(case['scene'])]
        with (out / f'{case["condition"]}_{case["scene"]}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        assert result.returncode == 0, f'Check {case["condition"]}_{case["scene"]}.log'
        folder = ROOT / f'artifacts/reports/policy_replays/{case["tag"]}'
        receipt = read(folder / 'manifest.json')
        assert receipt['outcome']['scene_seed'] == case['scene']
        comparison = receipt['reference_comparison']
        assert comparison['all_input_action_arrays_identical']
        assert all(comparison[k] for k in ('success_identical', 'lifted_identical', 'final_flags_identical'))
        reference = ROOT / f'artifacts/runs/{cfg["revision"]}_evaluation/nominal/{case["condition"]}_seed7'
        reference_trace = reference / f'scene_{case["scene"]}_scoring.json'
        receipt['builder_trace_check'] = {'reference_scoring_sha256': sha(reference_trace),
                                          'exact': read(folder / 'scoring.json') == read(reference_trace)}
        assert receipt['builder_trace_check']['exact']
        return case, folder, receipt

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            completed = list(pool.map(replay, cases))
        indexed = {(c['scene'], c['condition']): (folder, receipt) for c, folder, receipt in completed}
        path = out / 'four_goal_comparison.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 12.5, (800, 504)); assert writer.isOpened()
        expected_frames = 0; preview_frames = []
        try:
            for goal, scene in enumerate(range(start, start + 4), 1):
                streams = []
                for condition in cfg['conditions']:
                    folder, receipt = indexed[(scene, condition)]
                    assert sha(folder / 'replay.mp4') == receipt['outputs_sha256']['replay.mp4']
                    frames, fps = decode(folder / 'replay.mp4')
                    assert fps == 12.5 and frames[0].shape == (320, 320, 3)
                    assert len(frames) == receipt['outcome']['video_frames_verified']
                    streams.append((frames, receipt['outcome']))
                assert len(streams[0][0]) == len(streams[1][0])
                for tick in range(len(streams[0][0])):
                    canvas = np.full((504, 800, 3), 248, np.uint8)
                    def label(text, position, color=(20, 20, 20), scale=.48):
                        cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
                    label('LEARNED RGB + GOAL POLICY | SIMULATION | FROZEN SEED 7', (24, 24), scale=.57)
                    label(f'P{goal} -> G{goal} | nominal scene {scene} | t={tick / 12.5:05.2f}s | exact-reference replay', (24, 49))
                    for column, (condition, title) in enumerate(zip(cfg['conditions'], ('No video timing/progress', 'Video timing/progress'))):
                        x = 40 + 400 * column; frames, outcome = streams[column]
                        label(title, (x, 80), scale=.54)
                        canvas[96:416, x:x + 320] = frames[tick]
                        color = (35, 130, 35) if outcome['success'] else (30, 30, 190)
                        result = 'PASS' if outcome['success'] else 'FAIL'
                        label(f'Final strict result: {result} | lift: {outcome["lifted"]}', (x, 440), color)
                    a, b = [summary['groups'][c] for c in cfg['conditions']]
                    label(f'All 3 seeds / 16 nominal scenes: {a["successes"]}/{a["episodes"]} vs {b["successes"]}/{b["episodes"]} strict successes.', (24, 466))
                    label('Both share S01 goal geometry. Same-number pairs only. All failures retained.', (24, 489))
                    writer.write(canvas); expected_frames += 1
                    if tick == len(streams[0][0]) - 1: preview_frames.append(canvas.copy())
        finally:
            writer.release()
        verification, fps = decode(path); assert len(verification) == expected_frames and fps == 12.5
        # Contact sheet uses actual final video frames, not synthetic success scenes.
        sheet = np.concatenate([np.concatenate(preview_frames[:2], axis=1),
                                np.concatenate(preview_frames[2:], axis=1)], axis=0)
        assert cv2.imwrite(str(out / 'contact_sheet.png'), sheet)
        meta.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                    frames_decoded=len(verification), fps=fps, duration_s=len(verification) / fps,
                    outputs_sha256={name: sha(out / name) for name in ('four_goal_comparison.mp4', 'contact_sheet.png')},
                    visual_review='pending inspection', replays=[{'case': c, 'folder': folder.relative_to(ROOT).as_posix(),
                           'manifest_sha256': sha(folder / 'manifest.json'), 'strict_success': receipt['outcome']['success'],
                           'comparison': receipt['reference_comparison'],
                           'physical_trace_check': receipt['builder_trace_check']} for c, folder, receipt in completed])
        write(out / 'manifest.json', meta)
        lines = ['# 四目标固定案例演示', '', '按 nominal 的前四个场景、seed7 和两个条件重放，共八个回合；不按结果选择，失败保留。每个回合与既有主评测输入/动作数组及评分完全匹配，不增加主表执行数。', '',
                 '[播放 61.76 秒比较视频](four_goal_comparison.mp4)', '', '画面注明模拟、具体目标、条件、最终严格判定以及三个种子全部 nominal 回合的成功总数；单个种子/四例演示不代表总体性能。两组均使用 S01 蓝点目标几何，只比较额外的视频时序/进度。', '',
                 '| 固定场景 | 任务 | 不加视频时序 | 视频时序/进度 |', '|---|---|---|---|']
        for goal, scene in enumerate(range(start, start + 4), 1):
            statuses = ['成功' if indexed[(scene, c)][1]['outcome']['success'] else '失败' for c in cfg['conditions']]
            lines.append(f'| {scene} | P{goal}→G{goal} | {statuses[0]} | {statuses[1]} |')
        lines += ['', '原始回合视频、动作、评分和源码快照均保留，详见 [组装及逐回合复现收据](manifest.json)。公开发布和真实机器人效果没有由这份本地演示完成。']
        (out / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(json.dumps({'folder': str(out), 'frames': expected_frames,
                          'successes_shown': {c: sum(r['outcome']['success'] for case, _, r in completed if case['condition'] == c) for c in cfg['conditions']}}, ensure_ascii=False))
    except Exception as exc:
        meta.update(status='failed', error=str(exc)); write(out / 'manifest.json', meta); raise

if __name__ == '__main__':
    main()
