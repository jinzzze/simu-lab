"""Train all fixed seeds, then run separate validation and independent suites."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, subprocess, sys
import concurrent.futures

ROOT = Path(__file__).resolve().parent; PROJECT = ROOT.parents[1]

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    config = 'configs/policy_recurrent_goal_rgbaug_v6.json'
    cfg = json.loads((ROOT / config).read_text(encoding='utf-8'))
    out = ROOT / f'artifacts/reports/{cfg["revision"]}/development_pipeline_v1'
    assert not out.exists(); out.mkdir(parents=True)
    meta = {'status': 'training', 'config': cfg, 'started_utc': datetime.now(timezone.utc).isoformat(),
            'commands': [], 'overlapping_process_timings_not_a_benchmark': True,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(out / 'manifest.json', meta)
    def run(script, arguments, label):
        argv = [sys.executable, '-u', str(ROOT / script), *arguments]
        with (out / (label + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(argv, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0, f'Inspect {label}.log; failed run is retained'
        return {'argv': argv, 'label': label, 'returncode': result.returncode}
    try:
        meta['commands'].append(run('train_paired_recurrent_rgbaug_policy.py', ['--config', config], 'training_all6'))
        meta['status'] = 'development_validation'; write(out / 'manifest.json', meta)
        cases = [(c, s) for c in cfg['conditions'] for s in cfg['training']['seeds']]
        def evaluate(case):
            c, s = case
            return run(cfg['evaluation_script'], ['--config', config, '--suite', 'validation', '--conditions', c,
                    '--seeds', str(s)], f'validation_{c}_seed{s}')
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for result in pool.map(evaluate, cases): meta['commands'].append(result)
        meta['status'] = 'independent_suites'; write(out / 'manifest.json', meta)
        meta['commands'].append(run('run_paired_rgbaug_followup.py', ['--config', config, '--tag', 'independent_suites_v1',
                    '--workers', '3', '--include-diagnostics'], 'independent_suites'))
        meta.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat())
        write(out / 'manifest.json', meta); print('RGB augmentation pipeline completed; all6 models and frozen suites retained', flush=True)
    except Exception as exc:
        meta.update(status='failed', error=str(exc)); write(out / 'manifest.json', meta); raise

if __name__ == '__main__':
    main()
