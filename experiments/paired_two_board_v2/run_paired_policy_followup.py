"""Complete audited independent suites with separate output folders per model."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,re,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
EXECUTABLES={'paired_goal_image_diffusion_v1':'evaluate_paired_policy.py',
    'paired_goal_image_diffusion_delta_v2':'evaluate_paired_delta_policy.py',
    'paired_goal_recurrent_bc_v3':'evaluate_paired_recurrent_policy.py',
    'paired_goal_relative_recurrent_bc_v4':'evaluate_paired_recurrent_goal_policy.py'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--tag',required=True)
    parser.add_argument('--workers',type=int,default=3);parser.add_argument('--include-diagnostics',action='store_true');args=parser.parse_args()
    assert 1<=args.workers<=6 and re.fullmatch('[a-z0-9][a-z0-9_-]{0,63}',args.tag)
    cfg=read(ROOT/args.config);rev=cfg['revision'];out=ROOT/f'artifacts/reports/{rev}/{args.tag}'
    assert not out.exists(),'Use a new orchestration tag; evaluation resumes preserve prior episodes';out.mkdir(parents=True)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for path in [Path(__file__),ROOT/args.config,ROOT/'validate_paired_policy_runs.py',ROOT/'analyze_paired_policy.py']:
        shutil.copy2(path,snapshot/path.name)
    meta={'status':'waiting_for_development_validation','config':cfg,'workers':args.workers,'started_utc':datetime.now(timezone.utc).isoformat(),
          'source_sha256':{p.name:sha(p) for p in snapshot.iterdir()},'commands':[],'overlapping_workers_timing_not_a_benchmark':True}
    write(out/'manifest.json',meta)
    # Read-only wait: never start a second evaluator against active validation folders.
    while True:
        completed=[]
        for c in cfg['conditions']:
            for s in cfg['training']['seeds']:
                path=ROOT/f'artifacts/runs/{rev}_evaluation/validation/{c}_seed{s}/manifest.json'
                if path.exists():
                    value=read(path)
                    if value['status']=='failed':raise RuntimeError(f'Development evaluation failed: {c} seed{s}')
                    if value['status']=='completed':completed.append(value)
        if len(completed)==6:break
        time.sleep(15)
    def command(parts,label):
        argv=[sys.executable,*[str(x) for x in parts]];meta['commands'].append({'argv':argv,'label':label});write(out/'manifest.json',meta)
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(argv,cwd=ROOT.parents[1],stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:raise RuntimeError(f'{label} failed; inspect archived log')
    def audit_and_analyze(suite):
        flag=[] if (ROOT/f'artifacts/reports/{rev}/{suite}_validation.json').exists() else ['--require-complete']
        command([ROOT/'validate_paired_policy_runs.py','--config',args.config,'--suite',suite,*flag],suite+'_audit')
        if suite in ['validation',*cfg['evaluation']['suites']] and not (ROOT/f'artifacts/reports/{rev}/{suite}_analysis_v1/REPORT_ZH.md').exists():
            command([ROOT/'analyze_paired_policy.py','--config',args.config,'--suite',suite],suite+'_analysis')
    audit_and_analyze('validation')
    totals={c:sum(m['successes'] for m in completed if m['condition']==c) for c in cfg['conditions']}
    meta['development_successes']=totals
    if 'main_suite_gate' in cfg['evaluation'] and min(totals.values())==0:
        meta.update(status='completed_gate_skipped',reason='Predeclared gate: at least one condition has zero development successes')
        write(out/'manifest.json',meta);print(json.dumps(meta),flush=True);return
    suites=list(cfg['evaluation']['suites'])
    if args.include_diagnostics:suites += [name for key in cfg['evaluation']['diagnostics'] for name in (key+'_control',key)]
    for suite in suites:
        seeds=cfg['evaluation']['diagnostics'].get(suite.removesuffix('_control'),{}).get('seeds',cfg['training']['seeds'])
        pending=[(c,s) for c in cfg['conditions'] for s in seeds];active={};last_progress=None
        while pending or active:
            while pending and len(active)<args.workers:
                c,s=pending.pop(0);label=f'{suite}_{c}_seed{s}';path=out/(label+'.log');handle=path.open('w',encoding='utf-8')
                argv=[sys.executable,'-u',str(ROOT/EXECUTABLES[cfg['kind']]),'--config',args.config,'--suite',suite,'--conditions',c,'--seeds',str(s)]
                process=subprocess.Popen(argv,cwd=ROOT.parents[1],stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                active[label]=(process,handle,c,s);meta['commands'].append({'argv':argv,'label':label});meta.update(status='running',current_suite=suite);write(out/'manifest.json',meta)
            for label,(process,handle,c,s) in list(active.items()):
                if process.poll() is not None:
                    handle.close();del active[label]
                    if process.returncode:
                        for other,other_handle,_,_ in active.values():other.terminate();other.wait(timeout=20);other_handle.close()
                        raise RuntimeError(f'{label} failed; preserved partial outputs and logs')
            progress={f'{c}_seed{s}':len(rows(ROOT/f'artifacts/runs/{rev}_evaluation/{suite}/{c}_seed{s}/episodes.jsonl')) for c in cfg['conditions'] for s in seeds}
            if progress!=last_progress:print(json.dumps({'suite':suite,'executed_by_model':progress}),flush=True);last_progress=progress
            if pending or active:time.sleep(5)
        audit_and_analyze(suite)
        command([ROOT/'update_paired_policy_progress.py'],suite+'_progress')
        print(json.dumps({'completed_suite':suite,'revision':rev}),flush=True)
    meta.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat());write(out/'manifest.json',meta)

if __name__=='__main__':main()
