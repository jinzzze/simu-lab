"""Queue the frozen recombination diagnostic after the full v6 pipeline."""
from pathlib import Path
from datetime import datetime, timezone
import concurrent.futures, hashlib, json, shutil, subprocess, sys, time

ROOT = Path(__file__).resolve().parent; PROJECT = ROOT.parents[1]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    plan = read(ROOT / 'configs/consistent_goal_recombination_v1.json'); rev = plan['policy_revision']
    out = ROOT / f'artifacts/reports/{rev}/consistent_goal_recombination_v1/orchestration'
    assert not out.exists(); out.mkdir(parents=True)
    shutil.copy2(__file__, out / Path(__file__).name)
    meta = {'status': 'waiting_for_primary_pipeline', 'started_utc': datetime.now(timezone.utc).isoformat(),
            'plan': plan, 'commands': [], 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(out / 'manifest.json', meta)
    try:
        while True:
            path = ROOT / f'artifacts/reports/{rev}/development_pipeline_v1/manifest.json'
            value = read(path)
            if value['status'] == 'failed': raise RuntimeError('Primary pipeline failed; do not continue blindly')
            if value['status'] == 'completed': break
            time.sleep(10)
        # A gated zero-success model remains a valid negative diagnostic, but
        # the diagnostic cannot be mistaken for a replacement primary test.
        meta['status'] = 'running'; write(out / 'manifest.json', meta)
        def run(case):
            condition, seed = case; label = f'{condition}_seed{seed}'
            args = ['--condition', condition, '--seed', str(seed)]
            if condition == 'oracle': args = ['--oracle']; label = 'privileged_teacher'
            argv = [sys.executable, '-u', str(ROOT / 'evaluate_paired_recombination.py'), *args]
            with (out / (label + '.log')).open('w', encoding='utf-8') as log:
                result = subprocess.run(argv, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            assert result.returncode == 0, f'Inspect {label}.log'
            return {'label': label, 'argv': argv, 'returncode': result.returncode}
        meta['commands'].append(run(('oracle', None))); write(out / 'manifest.json', meta)
        cases = [(c, s) for c in plan['conditions'] for s in plan['policy_seeds']]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for result in pool.map(run, cases):
                meta['commands'].append(result); write(out / 'manifest.json', meta); print(json.dumps(result), flush=True)
        meta.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat()); write(out / 'manifest.json', meta)
    except Exception as exc:
        meta.update(status='failed', error=str(exc)); write(out / 'manifest.json', meta); raise

if __name__ == '__main__':
    main()
