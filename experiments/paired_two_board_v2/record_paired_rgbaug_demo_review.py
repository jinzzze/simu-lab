"""Record the completed contact-sheet inspection without replacing raw evidence."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil

ROOT = Path(__file__).resolve().parent
folder = ROOT / 'artifacts/reports/paired_recurrent_goal_rgbaug_v6/four_goal_nominal_demo_v1'
path = folder / 'manifest.json'
data = json.loads(path.read_text(encoding='utf-8'))
assert data['status'] == 'completed' and data['visual_review'] == 'pending inspection'
before = folder / 'before_visual_review.json'
assert not before.exists()
shutil.copy2(path, before)
data.update(
    visual_review='Codex inspected the four final paired frames in contact_sheet.png: all eight panels and goal/task labels are legible; simulation-only PASS/PASS/PASS/PASS and video-guided FAIL/PASS/FAIL/PASS agree with the replay report; final object locations are visible. No claim of human full-motion review or visual certification of strict contact/footprint thresholds; those remain from the complete physical scoring trace.',
    visual_review_utc=datetime.now(timezone.utc).isoformat(),
    visual_reviewer='Codex; not applicant human outcome adjudication',
    visual_review_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    contact_sheet_sha256=hashlib.sha256((folder / 'contact_sheet.png').read_bytes()).hexdigest(),
    before_visual_review_sha256=hashlib.sha256(before.read_bytes()).hexdigest(),
)
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
shutil.copy2(__file__, folder / Path(__file__).name)
print(json.dumps({'review': str(path), 'prior_manifest_preserved': str(before)}, ensure_ascii=False))
