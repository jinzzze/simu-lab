"""Read-only local navigation check for the completed current review documents."""
from pathlib import Path
import hashlib, json, re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
names = ('POLICY_PROGRESS_ZH.md', 'LEARNED_POLICY_REPORT_EN.md', 'POLICY_REPRODUCTION_ZH.md',
         'REVIEW_GUIDE_EN.md', 'CURRENT_TASK_HANDOFF_ZH.md', 'TRAINING_RUNTIME_DIAGRAM.md')
sources = {}; checked = []; missing = []
for name in names:
    path = ROOT / name
    sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for target in re.findall(r'!?\[[^\]\n]+\]\(([^)\n]+)\)', path.read_text(encoding='utf-8')):
        if target.startswith(('http:', 'https:', 'mailto:', 'codex:', 'app:', '#')):
            continue
        target = unquote(target.strip('<>').split('#', 1)[0])
        if not target:
            continue
        resolved = (ROOT / target).resolve()
        checked.append({'document': name, 'target': target, 'exists': resolved.exists()})
        if not resolved.exists():
            missing.append({'document': name, 'target': target, 'resolved': str(resolved)})
result = {'status': 'passed' if not missing else 'failed', 'local_links_checked': len(checked),
          'missing': missing, 'document_sha256': sources,
          'validator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'files_modified': False}
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if not missing else 1)
