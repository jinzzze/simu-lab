"""Copy frozen inputs into a NEW workspace; never train or remove recorded output."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'experiments/personal_video_timing_v4'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise SystemExit('Destination must not exist. Preserve previous runs and choose a new directory.')
    sources = json.loads((STUDY / 'data/processed/human_profiles_v1.json').read_text(encoding='utf-8'))['source_sha256']
    files = set()
    for name, expected in sources.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise SystemExit('Full evidence package required; missing input: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit('Frozen input differs: ' + name)
        files.add(path)
    for folder in ('src', 'scripts', 'configs'):
        files.update(p for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.ps1'))
    for name in ('paired_two_board_v2', 'cross_goal_composition_v3', 'personal_video_timing_v4'):
        folder = ROOT / 'experiments' / name
        files.update(p for p in folder.iterdir() if p.is_file() and p.suffix in ('.py', '.json'))
        files.update(p for p in (folder / 'configs').rglob('*') if p.is_file())
    files.update(ROOT / name for name in ('environment.yml', 'requirements.txt'))
    files.add(STUDY / 'data/processed/human_profiles_v1.json')
    assert all(p.is_file() for p in files)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for path in sorted(files):
        relative = path.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    study_target = destination / STUDY.relative_to(ROOT)
    (study_target / 'artifacts').mkdir(exist_ok=True)
    (study_target / 'receipts').mkdir(exist_ok=True)
    (destination / 'RERUN_INPUTS.json').write_text(json.dumps({
        'scope': 'Frozen configurations and personal inputs only; no models, robot demonstrations or evaluation outputs.',
        'files_sha256': manifest,
        'training_started': False,
    }, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(destination), 'copied_files': len(files), 'training_started': False}))


if __name__ == '__main__':
    main()
