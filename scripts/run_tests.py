"""Run native-library test suites in separate processes on Windows."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'artifacts/diagnostics/tests'
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for suite in sorted((ROOT / 'tests').glob('test_*.py')):
        start = time.perf_counter()
        result = subprocess.run([sys.executable, '-m', 'pytest', str(suite), '-q',
                                 '--junitxml', str(output / (suite.stem + '.xml'))],
                                cwd=ROOT, text=True, encoding='utf-8', errors='replace',
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / (suite.stem + '.log')).write_text(result.stdout, encoding='utf-8')
        print(result.stdout, flush=True)
        results.append({'suite': suite.name, 'returncode': result.returncode,
                        'seconds': time.perf_counter() - start})
    (output / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return int(any(r['returncode'] for r in results))


if __name__ == '__main__':
    sys.exit(main())
