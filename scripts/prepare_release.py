"""Stage and package local submission candidates without publishing or granting licenses."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import argparse,hashlib,json,re,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/release_candidate_v1'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def collect():
    source=set();inputs=set()
    for name in ['README.md','AGENTS.md','THIRD_PARTY_NOTICES.md','environment.yml','requirements.txt','.gitattributes','.gitignore']:
        source.add(ROOT/name)
    for directory in ['src','scripts','tests','configs','docs','assets','data/manifests','artifacts/reports']:
        source.update(f for f in (ROOT/directory).rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ('.pyc','.download'))
    for f in (ROOT/'artifacts/runs').rglob('*'):
        if not f.is_file() or '__pycache__' in f.parts:continue
        (inputs if f.suffix in ('.pt','.pth','.npz') else source).add(f)
    for directory in ['artifacts/diagnostics/environment_current','artifacts/diagnostics/grasp_oracle','artifacts/diagnostics/pilot_20261006','artifacts/diagnostics/pilot_labels_v2','artifacts/diagnostics/label_review_release_v1','artifacts/diagnostics/clean_install_v1','artifacts/diagnostics/tests']:
        source.update(f for f in (ROOT/directory).rglob('*') if f.is_file())
    source.update(f for f in (ROOT/'artifacts/diagnostics').glob('*') if f.is_file() and f.suffix in ('.json','.txt','.png'))
    for directory in ['data/raw/pilot','data/processed/pilot_labels_v2','data/processed/grasp_world_model_v1','data/processed/diffusion_policy_v1']:
        inputs.update(f for f in (ROOT/directory).rglob('*') if f.is_file())
    inputs.update((ROOT/'data/processed').glob('*.npz'))
    # Include local documentary links needed by historical Markdown, without environments/caches.
    pattern=re.compile(r'\]\(([^)]+)\)')
    changed=True
    while changed:
        changed=False
        for f in list(source):
            if f.suffix.lower()!='.md':continue
            for ref in pattern.findall(f.read_text(encoding='utf-8-sig',errors='replace')):
                url=urlsplit(ref.split(' "')[0].strip('<>'))
                if url.scheme or not url.path:continue
                target=(f.parent/unquote(url.path)).resolve()
                if target.is_relative_to(ROOT) and target.is_file() and target not in source|inputs:
                    rel=target.relative_to(ROOT)
                    if any(x.startswith('.') for x in rel.parts):continue
                    (inputs if str(rel).replace('\\','/').startswith(('data/raw/','data/processed/','artifacts/models/')) or target.suffix in ('.pt','.pth') else source).add(target);changed=True
    return sorted(source),sorted(inputs)
class Links(HTMLParser):
    def __init__(self):super().__init__();self.paths=[]
    def handle_starttag(self,tag,attrs):self.paths.extend(v for k,v in attrs if k in ('href','src','poster') and v)
def validate(stage):
    links=[];missing=[];secrets=[];large=[]
    expressions=[re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'),re.compile(r'github_pat_[A-Za-z0-9_]{40,}'),re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),re.compile(r'AKIA[0-9A-Z]{16}')]
    for f in stage.rglob('*'):
        if not f.is_file():continue
        if f.stat().st_size>100*1024*1024:large.append(str(f.relative_to(stage)))
        if f.suffix in ('.py','.ps1','.md','.json','.jsonl','.txt','.yml','.html','.log'):
            text=f.read_text(encoding='utf-8-sig',errors='replace')
            if any(x.search(text) for x in expressions):secrets.append(str(f.relative_to(stage)))
            refs=[]
            if f.suffix=='.html':parsed=Links();parsed.feed(text);refs=parsed.paths
            elif f.suffix=='.md':refs=re.findall(r'\]\(([^)]+)\)',text)
            for ref in refs:
                u=urlsplit(ref.split(' "')[0].strip('<>'))
                if u.scheme or not u.path:continue
                target=(f.parent/unquote(u.path)).resolve();links.append([str(f.relative_to(stage)),ref])
                if not target.is_relative_to(stage) or not target.exists():missing.append(links[-1])
    return {'local_link_count':len(links),'missing_local_links':missing,'credential_pattern_matches':secrets,'source_files_over_100MiB':large,'note':'Credential pattern scan is limited; no guarantee of exhaustive secret detection. Personal video/images are present in this local candidate and need publication review.'}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage-only',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for label,folder,files in [('source-report','repository',collect()[0]),('inputs-checkpoints','inputs',collect()[1])]:
        for f in files:
            rel=f.relative_to(ROOT);dst=OUT/folder/rel;dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists() or sha(dst)!=sha(f):
                if dst.exists():dst.chmod(0o666)
                shutil.copy2(f,dst)
            rows.append({'path':rel.as_posix(),'archive':label,'bytes':f.stat().st_size,'sha256':sha(f)})
    manifest={'status':'local_candidate_not_published','intended_repository':'jinzzze/simu-lab','independent_real_validation':'deferred_by_user','files':rows}
    (OUT/'repository/release_files.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    validation=validate(OUT/'repository')
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    if validation['credential_pattern_matches'] or validation['source_files_over_100MiB']:raise ValueError('Source package review required')
    if validation['missing_local_links']:print('Unresolved local links:',json.dumps(validation['missing_local_links']))
    if args.stage_only:print('Staged',len(rows),'files');return
    archives=[]
    for folder,name in [('repository','simu-lab-source-report.zip'),('inputs','simu-lab-inputs-checkpoints.zip')]:
        target=OUT/name
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for f in sorted((OUT/folder).rglob('*')):
                if f.is_file() and '.git' not in f.relative_to(OUT/folder).parts:z.write(f,f.relative_to(OUT/folder).as_posix())
        with zipfile.ZipFile(target) as z:
            if z.testzip() is not None:raise ValueError('Archive CRC failed')
        archives.append({'name':name,'bytes':target.stat().st_size,'sha256':sha(target),'crc_passed':True})
    (OUT/'SHA256SUMS.txt').write_text(''.join(f'{x["sha256"]}  {x["name"]}\n' for x in archives))
    (OUT/'release_manifest.json').write_text(json.dumps({'status':'local_candidate_not_published','intended_repository':'jinzzze/simu-lab','file_count':len(rows),'archives':archives,'validation':validation},indent=2))
    print(json.dumps(archives,indent=2))
if __name__=='__main__':main()
