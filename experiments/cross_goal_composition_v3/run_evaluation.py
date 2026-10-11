"""Bounded two-process execution, development before frozen test; all six models."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess
import sys
from common import ROOT,PROJECT,read,write,now,sha


def job(suite,condition,seed):
    logdir=ROOT/'artifacts/evaluation_logs';logdir.mkdir(exist_ok=True)
    log=logdir/f'{suite}_{condition}_seed{seed}.log'
    if log.exists():
        log=logdir/f'{suite}_{condition}_seed{seed}_resume.log'
    with log.open('x',encoding='utf-8') as f:
        command=[sys.executable,'-B','-u',str(ROOT/'evaluate.py'),'--suite',suite,'--condition',condition,'--seed',str(seed)]
        p=subprocess.run(command,cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if p.returncode:
        raise RuntimeError(f'Failed {suite}/{condition}/{seed}; see {log}')
    print(f'Completed {suite}/{condition}/seed{seed}',flush=True)
    return dict(suite=suite,condition=condition,seed=seed,log=str(log.relative_to(ROOT)),returncode=p.returncode)


def main():
    cfg=read(ROOT/'configs/policy_v1.json')
    for c in cfg['conditions']:
        for s in cfg['training']['seeds']:
            folder=ROOT/f'artifacts/models/{c}_seed{s}';meta=read(folder/'manifest.json')
            assert meta['status']=='completed' and meta['checkpoint_sha256']==sha(folder/'checkpoint.pt')
    receipt=ROOT/'artifacts/evaluation_orchestration.json'
    meta=dict(status='running',started_utc=now(),jobs=[],workers=2);write(receipt,meta)
    try:
        for suite in ('validation','test'):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(job,suite,c,s) for c in cfg['conditions'] for s in cfg['training']['seeds']]
                for f in as_completed(futures):
                    meta['jobs'].append(f.result());write(receipt,meta)
        meta.update(status='completed',completed_utc=now());write(receipt,meta)
    except Exception as exc:
        meta.update(status='failed',error=str(exc));write(receipt,meta);raise


if __name__=='__main__':
    main()
