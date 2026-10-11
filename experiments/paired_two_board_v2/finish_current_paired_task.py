"""Finish only the already scheduled v6 evaluation and its result documents."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, concurrent.futures, hashlib, json, shutil, subprocess, sys, time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
REVISION = 'paired_recurrent_goal_rgbaug_v6'
CONFIG = 'configs/policy_recurrent_goal_rgbaug_v6.json'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--staging', required=True)
    args = parser.parse_args()
    reports = ROOT / f'artifacts/reports/{REVISION}'
    out = reports / 'current_task_finish_v1'
    assert not out.exists(); out.mkdir()
    shutil.copy2(__file__, out / Path(__file__).name)
    meta = {'status': 'waiting_for_primary_pipeline', 'started_utc': datetime.now(timezone.utc).isoformat(),
            'commands': [], 'new_training_or_experiments': False,
            'scope': 'Finish frozen v6 suites, queued consistent-goal diagnostic, source/input analysis and review documents; then return control for explicit goal pause',
            'source_sha256': sha(__file__)}
    write(out / 'manifest.json', meta)

    def wait_for(path):
        while True:
            data = read(path)
            if data['status'] == 'failed': raise RuntimeError(f'Existing pipeline failed: {path}')
            if data['status'] == 'completed': return
            time.sleep(10)

    def run(item):
        label, script, extra = item
        argv = [sys.executable, '-u', str(ROOT / script), *extra]
        with (out / f'{label}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(argv, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0, f'Inspect {label}.log'
        return {'label': label, 'argv': argv, 'returncode': result.returncode}

    def record(item):
        meta['commands'].append(run(item)); write(out / 'manifest.json', meta)

    try:
        wait_for(reports / 'development_pipeline_v1/manifest.json')
        meta['status'] = 'finishing_existing_analysis'; write(out / 'manifest.json', meta)
        record(('input_diagnostics', 'analyze_paired_input_diagnostics.py', ['--config', CONFIG]))
        record(('stability_history', 'analyze_paired_stability_history.py', ['--config', CONFIG, '--suites',
                    'validation', 'nominal', 'appearance', 'wider_start', 'novel_appearance']))
        items = [(f'{suite}_image_overlap', 'audit_paired_image_overlap.py', ['--config', CONFIG, '--suite', suite])
                 for suite in ('appearance', 'wider_start', 'novel_appearance')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(run, items):
                meta['commands'].append(result); write(out / 'manifest.json', meta)
        meta['status'] = 'waiting_for_already_queued_recombination'; write(out / 'manifest.json', meta)
        wait_for(reports / 'consistent_goal_recombination_v1/orchestration/manifest.json')
        record(('consistent_goal_analysis', 'analyze_paired_recombination.py', []))
        record(('progress', 'update_paired_policy_progress.py', []))
        record(('document_refresh', 'finalize_paired_policy_documents.py', ['--staging', args.staging]))
        record(('review_guide', 'build_paired_review_guide.py', ['--config', CONFIG]))
        results = {suite: read(reports / f'{suite}_analysis_v1/results.json')['groups']
                   for suite in ('validation', 'nominal', 'appearance', 'wider_start', 'novel_appearance')}
        recombination = read(reports / 'consistent_goal_recombination_v1/analysis_v1/results.json')
        handoff = ROOT / 'CURRENT_TASK_HANDOFF_ZH.md'
        if handoff.exists(): shutil.copy2(handoff, out / 'previous_CURRENT_TASK_HANDOFF_ZH.md')
        lines = ['# 当前任务结束交接', '',
                 '已完成当前冻结 v6 的六个模型、闭环评测、输入/来源分析、已排队的 P/G 组合诊断及固定场景演示。用户要求本任务结束后暂停，不再启动新实验。此清单由实际完整报告生成；线程目标的最终暂停状态由 Codex 工具确认。', '',
                 '| 当前版本套件 | 不加视频时序/进度 | 视频时序/进度 |', '|---|---:|---:|']
        for suite, groups in results.items():
            a, b = [groups[c] for c in ('simulation_only', 'video_guided')]
            lines.append(f'| {suite} | {a["successes"]}/{a["episodes"]} | {b["successes"]}/{b["episodes"]} |')
        lines += ['', '三个模型重复相同初始场景，执行次数不等于独立环境数量。个人视频的额外搬运时序没有被证明带来稳定收益；两组共同使用 S01 蓝点目标几何。', '',
                  '| 一致目标组合诊断 | 已训练配对 | 未训练组合 |', '|---|---:|---:|']
        for condition, g in recombination['groups'].items():
            lines.append(f'| {condition} | {g["seen_successes"]}/{g["seen_executions"]} | {g["unseen_successes"]}/{g["unseen_executions"]} |')
        lines += ['', '组合诊断只使用四个物体初态与已知的 P/G 位置；不代表未见目标位置、连续目标或实机泛化。教师参照使用初始物体真值与独立轨迹长度，不混入学习分数。', '',
                  '[英文审阅入口](REVIEW_GUIDE_EN.md) · [全部进度/历史失败](POLICY_PROGRESS_ZH.md) · [复现命令](POLICY_REPRODUCTION_ZH.md)', '',
                  f'[四目标比较视频与八个实际复跑](artifacts/reports/{REVISION}/four_goal_nominal_demo_v1/REPORT_ZH.md)', '',
                  '仍仅收集 S01 的 16 条个人视频；S02/S03/S04、人工严格标签、实际拍摄日期、实机标定/执行与申请者资料仍未完成，按请求跳过。新增实验只在本地，未发布新 release 或提交申请。']
        handoff.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        meta.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                    review_guide_sha256=sha(ROOT / 'REVIEW_GUIDE_EN.md'), handoff_sha256=sha(handoff),
                    all_planned_background_work_terminated=True)
        write(out / 'manifest.json', meta)
        print(json.dumps({'status': 'completed', 'handoff': str(handoff)}, ensure_ascii=False), flush=True)
    except Exception as exc:
        meta.update(status='failed', error=str(exc)); write(out / 'manifest.json', meta); raise

if __name__ == '__main__':
    main()
