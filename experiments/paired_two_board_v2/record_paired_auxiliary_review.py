"""Record already performed media inspection and isolated augmentation checks."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil

ROOT = Path(__file__).resolve().parent; PROJECT = ROOT.parents[1]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    path = ROOT / 'artifacts/reports/paired_recurrent_goal_precision_v5/four_goal_nominal_demo_v1/manifest.json'
    data = read(path); assert data['status'] == 'completed' and data['frames_decoded'] == 772
    assert data['visual_review'] == 'pending inspection'
    assert all(r['physical_trace_check']['exact'] and r['comparison']['all_input_action_arrays_identical'] for r in data['replays'])
    receipt = ROOT / ('receipts/auxiliary_review_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')); receipt.mkdir()
    shutil.copy2(path, receipt / 'demo_before_visual_review.json')
    data.update(visual_review='Codex viewed the decoded four-pair final-frame contact sheet: captions are legible, PASS/FAIL labels agree with all eight stored outcomes, and both condition frames are retained; no full-motion human review claimed',
                visual_review_utc=datetime.now(timezone.utc).isoformat(), visual_reviewer='Codex')
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tests = PROJECT / 'tests/test_paired_rgb_augmentation.py'
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(),
               'tests_already_executed_command': '.env/python.exe -m unittest discover -s tests -p test_paired_rgb_augmentation.py -v',
               'tests_passed': ['same transform at every timestep', 'paired parameters independent of episode length/pixels', 'zero recipe identity'],
               'isolated_torch_process_exit_code': 0, 'tests_run': 3,
               'source_sha256': {str(tests.relative_to(PROJECT)): sha(tests), 'paired_rgb_augmentation.py': sha(ROOT / 'paired_rgb_augmentation.py')},
               'test_scope': 'three new augmentation contracts; original57-suite run remains separately archived, not rerun here',
               'demo_manifest_after_review_sha256': sha(path), 'recorder_sha256': sha(__file__)}
    (receipt / 'manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    shutil.copy2(tests, receipt / tests.name); shutil.copy2(__file__, receipt / Path(__file__).name)
    print(json.dumps({'receipt': str(receipt), 'visual_review_recorded': True, 'existing_tests_recorded': 3}, ensure_ascii=False))

if __name__ == '__main__':
    main()
