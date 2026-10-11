"""Resume fixed study stages; never run evaluation if the teacher gate failed."""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import ROOT,read,write,sha,check_sources,now


def run(script,*args):
    print('RUN',script,*args,flush=True)
    subprocess.run([sys.executable,'-B','-u',str(ROOT/script),*map(str,args)],cwd=ROOT,check=True)


def run_suite(suite,cfg):
    """Two isolated physics/Torch pairs, as in the preceding coverage study."""
    logs=ROOT/'artifacts/evaluation_logs';logs.mkdir(exist_ok=True)
    def job(condition,seed):
        stem=f'{suite}_{condition}_seed{seed}';attempt=1
        path=logs/f'{stem}_{attempt}.log'
        while path.exists():
            attempt+=1;path=logs/f'{stem}_{attempt}.log'
        command=[sys.executable,'-B','-u',str(ROOT/'evaluate.py'),
                 '--condition',condition,'--seed',str(seed),'--suite',suite]
        with path.open('x',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:raise RuntimeError(f'Evaluation failed; retained log: {path}')
        print(f'Completed {suite} {condition} seed{seed}',flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(job,c,s) for c in cfg['conditions'] for s in cfg['training']['seeds']]
        for job_result in as_completed(jobs):job_result.result()


def main():
    state=ROOT/'artifacts/pipeline_state.json';cfg=read(ROOT/'configs/policy_v1.json')
    def record(stage,status='running',**kwargs):write(state,dict(stage=stage,status=status,updated_utc=now(),**kwargs))
    try:
        m=read(ROOT/'artifacts/robot_demos_v1/manifest.json')
        assert m['status']=='completed','External teacher collection must finish first'
        record('demonstration_audit')
        if not (ROOT/'artifacts/demonstration_audit_v1/receipt.json').exists():run('audit_demonstrations.py')
        assert m['training_gate_passed'],'Predeclared teacher quality gate failed; stop before training, retain all attempts'
        record('training');run('train.py')
        record('training_audit')
        if not (ROOT/'artifacts/training_audit_v1/receipt.json').exists():run('audit_training.py')
        record('validation')
        run_suite('validation',cfg)
        freeze=ROOT/'artifacts/final_evaluation_freeze_v1.json'
        if not freeze.exists():
            models={}
            for c in cfg['conditions']:
                for seed in cfg['training']['seeds']:
                    path=ROOT/f'artifacts/models/{c}_seed{seed}/manifest.json';meta=read(path)
                    val=read(ROOT/f'artifacts/evaluation/validation/{c}_seed{seed}/manifest.json')
                    assert meta['status']==val['status']=='completed';check_sources(meta['source_sha256'])
                    models[f'{c}_seed{seed}']=meta['checkpoint_sha256']
            write(freeze,dict(created_utc=now(),selection='All three fixed-final EMA seeds per condition; no development-outcome selection',
                checkpoints=models,study_sha256=sha(ROOT/'configs/study_v1.json'),policy_sha256=sha(ROOT/'configs/policy_v1.json')))
        record('test')
        run_suite('test',cfg)
        record('results')
        if not (ROOT/'artifacts/results_v1/results.json').exists():run('analyze_results.py')
        record('completed',status='completed',results='artifacts/results_v1/results.json')
    except Exception as exc:
        record('stopped',status='needs_attention',error=str(exc));raise


if __name__=='__main__':main()
