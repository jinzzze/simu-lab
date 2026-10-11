"""Freeze consistent goal/start recombination diagnostics outside primary suites."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil

ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    config = 'configs/policy_recurrent_goal_rgbaug_v6.json'
    cfg = json.loads((ROOT / config).read_text(encoding='utf-8'))
    cases = [{'case_id': f'P{i}_G{j}', 'start_id': f'P{i}', 'goal_id': f'G{j}', 'scene_seed': 68000 + i - 1,
              'seen_pair': i == j} for i in range(1, 5) for j in range(1, 5)]
    plan = {'revision': 'consistent_goal_recombination_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'policy_config': config, 'policy_config_sha256': sha(ROOT / config), 'policy_revision': cfg['revision'],
            'cases': cases, 'conditions': cfg['conditions'], 'policy_seeds': cfg['training']['seeds'],
            'jitter_m': .003, 'steps': 384, 'no_new_training_or_selection': True,
            'primary_results_unchanged': True, 'scene_selection': 'four new initial-object seeds, one per P; all16 target/start combinations, no outcome filtering',
            'target_input': 'set before reset/render; numerical goal and RGB highlight agree; scorer uses that same prescribed goal',
            'sample_scope': '16 tasks share4 object-start poses; seen4 versus unseen12 pairings; all3 fixed model seeds; simulation diagnostic, not arbitrary continuous-goal or real-world evidence',
            'oracle': 'all16 same task cases, fixed physical teacher using privileged INITIAL object XY only, no online position repair; feasibility diagnostic, never mixed into learned scores',
            'execution_order': 'wait for full frozen v6 primary/input suites to finish, then run diagnostic with unchanged checkpoints'}
    path = ROOT / 'configs/consistent_goal_recombination_v1.json'; assert not path.exists()
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    source = ROOT / 'evaluate_paired_recurrent_rgbaug_policy.py'
    text = source.read_text(encoding='utf-8')
    old = "obs=env.reset_task(scene,f'P{pairing}',f'G{pairing}')"
    assert old in text
    text = text.replace(old, "obs=env.reset_task(scene,definition['prescribed_start_id'],definition['prescribed_goal_id'])")
    driver = ROOT / 'paired_recombination_driver.py'; assert not driver.exists(); driver.write_text(text, encoding='utf-8')
    out = ROOT / 'receipts/consistent_goal_recombination_v1_plan'; assert not out.exists(); out.mkdir()
    for p in (path, source, driver, Path(__file__)): shutil.copy2(p, out / p.name)
    (out / 'manifest.json').write_text(json.dumps({'frozen_utc': plan['created_utc'], 'plan_sha256': sha(path),
        'original_driver_sha256': sha(source), 'new_driver_sha256': sha(driver),
        'driver_change': 'only replace reset-task IDs with prescribed case IDs; causal observation/action/physics/scoring loop unchanged',
        'cases_not_executed_before_freeze': True}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Frozen16 consistent goal/start diagnostics and privileged teacher feasibility reference')

if __name__ == '__main__':
    main()
