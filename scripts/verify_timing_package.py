"""Verify a downloaded package using only the Python standard library."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    manifest = ROOT / 'PACKAGE_MANIFEST.json'
    if not manifest.is_file():
        raise SystemExit('Use an extracted v1.1.0 release ZIP; a Git clone alone has no package manifest.')
    files = json.loads(manifest.read_text(encoding='utf-8'))['files']
    problems = []
    for name, expected in files.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            problems.append([name, 'missing or outside package'])
        elif path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            problems.append([name, 'size or SHA-256 differs'])
    print(json.dumps({'checked': len(files), 'passed': not problems, 'failures': problems}, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
