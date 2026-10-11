"""Bounded completion of already-frozen primary evaluation and reporting."""
import json
import subprocess
import sys
import time
from common import ROOT,PROJECT,read,write,now


def main():
    path=ROOT/'artifacts/primary_completion.json'
    if path.exists():
        raise FileExistsError(path)
    meta=dict(status='waiting_for_frozen_evaluation',started_utc=now(),steps=[]);write(path,meta)
    deadline=time.monotonic()+7200
    try:
        while True:
            if time.monotonic()>deadline:
                raise TimeoutError('Frozen evaluation did not complete within two hours')
            try:
                status=read(ROOT/'artifacts/evaluation_orchestration.json')['status']
                if status=='failed':
                    raise RuntimeError('Frozen evaluation failed; inspect orchestration manifest')
                if status=='completed':
                    break
            except (FileNotFoundError,json.JSONDecodeError):
                pass
            time.sleep(10)
        for name in ('analyze_results.py','plot_results.py','build_reports.py'):
            meta['status']='running_'+name;write(path,meta)
            log=ROOT/'artifacts'/(name.replace('.py','')+'_completion.log')
            with log.open('x',encoding='utf-8') as f:
                p=subprocess.run([sys.executable,'-B','-u',str(ROOT/name)],cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            meta['steps'].append(dict(script=name,returncode=p.returncode,log=str(log.relative_to(ROOT))))
            if p.returncode:
                raise RuntimeError(name+' failed')
            write(path,meta)
        meta.update(status='reports_completed_visual_review_pending',completed_utc=now())
        write(path,meta);print(json.dumps(meta,ensure_ascii=False),flush=True)
    except Exception as exc:
        meta.update(status='failed',error=str(exc));write(path,meta);raise


if __name__=='__main__':
    main()
