"""Fetch external base models only from recorded upstream URLs, verify before install."""
from pathlib import Path
import hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    for item in json.loads((ROOT/'configs/model_sources.json').read_text(encoding='utf-8-sig')):
        path=(ROOT/item['path']).resolve()
        if not path.is_relative_to(ROOT):raise ValueError('Model path outside repository')
        if path.exists():
            if digest(path)!=item['sha256']:raise ValueError(f'Existing model mismatch: {path}; preserved')
            print('Verified',item['path']);continue
        path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_suffix(path.suffix+'.download')
        if partial.exists():raise FileExistsError(partial)
        try:
            with urllib.request.urlopen(item['url'],timeout=90) as response,partial.open('xb') as out:
                for b in iter(lambda:response.read(1024*1024),b''):out.write(b)
            if partial.stat().st_size!=item['bytes'] or digest(partial)!=item['sha256']:raise ValueError(f'Upstream bytes changed: {item["url"]}')
            partial.replace(path);print('Downloaded and verified',item['path'])
        finally:
            if partial.exists():partial.unlink()
if __name__=='__main__':main()
