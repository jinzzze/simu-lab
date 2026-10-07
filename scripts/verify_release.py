"""Check extracted candidate files without importing training or simulation stacks."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-only',action='store_true');args=ap.parse_args()
    rows=json.loads((ROOT/'release_files.json').read_text(encoding='utf-8'))['files'];fail=[];checked=0
    for row in rows:
        if args.source_only and row['archive']!='source-report':continue
        p=(ROOT/row['path']).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:fail.append(row['path'])
        checked+=1
    print(json.dumps({'checked':checked,'passed':not fail,'failures':fail},indent=2))
    if fail:raise SystemExit(1)
if __name__=='__main__':main()
