"""Diagnose late regressions using recorded scoring; never alter primary outcomes."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, shutil
from collections import Counter

ROOT = Path(__file__).resolve().parent

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--suites', nargs='+', default=['validation', 'nominal', 'appearance', 'wider_start'])
    parser.add_argument('--tag', default='stability_history_v1')
    args = parser.parse_args(); cfg = read(ROOT / args.config)
    out = ROOT / f'artifacts/reports/{cfg["revision"]}/{args.tag}'
    assert not out.exists()
    sources = {args.config: sha(ROOT / args.config)}; groups = {}; cases = []
    for suite in args.suites:
        receipt = ROOT / f'artifacts/reports/{cfg["revision"]}/{suite}_validation.json'
        assert receipt.exists(), f'Complete/audit {suite} first'
        sources[str(receipt.relative_to(ROOT))] = sha(receipt)
        groups[suite] = {}
        for condition in cfg['conditions']:
            counts = Counter()
            for seed in cfg['training']['seeds']:
                folder = ROOT / f'artifacts/runs/{cfg["revision"]}_evaluation/{suite}/{condition}_seed{seed}'
                meta = read(folder / 'manifest.json'); assert meta['status'] == 'completed'
                physics = read(folder / 'source_snapshot/sim_physics_v1.json')
                rows = [json.loads(x) for x in (folder / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
                sources[str((folder / 'episodes.jsonl').relative_to(ROOT))] = sha(folder / 'episodes.jsonl')
                for row in rows:
                    path = folder / f'scene_{row["scene_seed"]}_scoring.json'
                    assert sha(path) == row['scoring_sha256']; trace = read(path)
                    sources[str(path.relative_to(ROOT))] = row['scoring_sha256']
                    lift_run = 0.; max_lift = 0.; lift_qualified = False
                    first_lift = None; first_stable = None; last_stable = None
                    for tick in trace:
                        dt = tick['t_s'] - (trace[tick['step_after_action'] - 2]['t_s'] if tick['step_after_action'] > 1 else 0.)
                        lift_run = lift_run + dt if tick['clearance_m'] >= physics['min_lift_clearance_m'] and tick['bilateral_grasp_contact'] else 0.
                        max_lift = max(max_lift, lift_run)
                        if max_lift + 1e-9 >= physics['min_lift_duration_s']:
                            lift_qualified = True
                            if first_lift is None: first_lift = tick['t_s']
                        # Require the qualifying lift to have happened before
                        # the stable placement; do not reinterpret pre-grasp rest.
                        if lift_qualified and tick['goal_stable_duration_s'] + 1e-9 >= physics['stable_duration_s']:
                            assert all(tick['flags'].values())
                            if first_stable is None: first_stable = tick['t_s']
                            last_stable = tick['t_s']
                    assert lift_qualified == row['lifted']
                    assert abs(max_lift - row['max_lift_duration_s']) < 1e-7
                    assert abs(trace[-1]['goal_stable_duration_s'] - row['final_stable_duration_s']) < 1e-7
                    assert row['success'] == bool(lift_qualified and row['final_stable_duration_s'] + 1e-9 >= physics['stable_duration_s'] and all(trace[-1]['flags'].values()))
                    late_regression = first_stable is not None and not row['success']
                    counts.update(episodes=1, final_success=int(row['success']),
                                  ever_strict_stable_after_lift=int(first_stable is not None),
                                  lost_earlier_strict_stable=int(late_regression))
                    cases.append({'suite': suite, 'condition': condition, 'policy_seed': seed,
                                  'scene_seed': row['scene_seed'], 'final_success': row['success'],
                                  'first_qualified_lift_s': first_lift, 'first_strict_stable_after_lift_s': first_stable,
                                  'last_strict_stable_after_lift_s': last_stable,
                                  'lost_earlier_strict_stable': late_regression,
                                  'final_stable_duration_s': row['final_stable_duration_s'],
                                  'final_missing_flags': [k for k, value in trace[-1]['flags'].items() if not value]})
            groups[suite][condition] = dict(counts)
    out.mkdir(parents=True)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'revision': cfg['revision'],
              'groups': groups, 'cases': cases, 'source_sha256': sources, 'analysis_sha256': sha(__file__),
              'scope': 'posthoc diagnostic on saved scoring; final384-step primary metric unchanged; no truth-based early stopping or action repair'}
    (out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 严格稳定窗口历史诊断', '', '只读取已保存的物理评分，核查是否曾完成合格抬升及完整放置稳定窗口、随后在固定终点丢失成功。所有主结果仍使用预先冻结的 384 步终点；没有改变成功阈值、早停或动作。', '',
             '| suite | 条件 | 最终严格成功 | 曾在抬升后满足严格稳定窗口 | 之后丢失该状态 |', '|---|---|---:|---:|---:|']
    for suite, conditions in groups.items():
        for condition, c in conditions.items():
            lines.append(f'| {suite} | {condition} | {c["final_success"]}/{c["episodes"]} | {c["ever_strict_stable_after_lift"]}/{c["episodes"]} | {c["lost_earlier_strict_stable"]} |')
    lines += ['', '这个事后诊断用于判断末段退化是否值得继续研究，不能作为新的主成功率。若未来研究停机策略，需要先冻结可从 RGB/机器人观测计算的判据，再在新的场景评测；不能用评分真值提前停止当前策略。', '', '[所有回合时间、终点缺失条件与源文件哈希](results.json)']
    (out / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    shutil.copy2(__file__, out / Path(__file__).name)
    print(json.dumps({'report': str(out / 'REPORT_ZH.md'), 'groups': groups}, ensure_ascii=False))

if __name__ == '__main__':
    main()
