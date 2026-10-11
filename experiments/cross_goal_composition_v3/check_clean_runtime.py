"""Non-destructive check of an independently installed environment, separate native processes."""
import json
import os
from pathlib import Path
import subprocess
from common import ROOT,PROJECT,now,write


def main():
    prefix=PROJECT/'work/clean_env_v1';python=prefix/'python.exe';assert python.exists()
    out=ROOT/'receipts/clean_runtime_check_v1';out.mkdir(exist_ok=False)
    env=os.environ.copy()
    # Avoid falling back to DLLs from the active training environment.
    old=str(PROJECT/'.env').lower()
    parts=[p for p in env['PATH'].split(os.pathsep) if not p.lower().startswith(old)]
    env.update(CONDA_PREFIX=str(prefix),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    env['PATH']=os.pathsep.join([str(prefix),str(prefix/'Scripts'),str(prefix/'Library/bin')]+parts)
    commands={
        'pip_check':[str(python),'-B','-m','pip','check'],
        'torch_import':[str(python),'-B','-c','import json,sys,torch; print(json.dumps(dict(python=sys.executable,torch=torch.__version__,module=torch.__file__,cuda=torch.cuda.is_available())))'],
        'bullet_import':[str(python),'-B','-c','import json,sys,pybullet,numpy,panda_gym; print(json.dumps(dict(python=sys.executable,bullet=pybullet.__file__,numpy=numpy.__version__,panda=panda_gym.__version__)))']}
    checks=[]
    for name,cmd in commands.items():
        p=subprocess.run(cmd,cwd=out,env=env,text=True,encoding='utf-8',errors='replace',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        (out/(name+'.log')).write_text(p.stdout,encoding='utf-8');checks.append(dict(name=name,returncode=p.returncode))
    result=dict(status='passed' if all(x['returncode']==0 for x in checks) else 'failed',created_utc=now(),
                prefix=str(prefix.relative_to(PROJECT)),checks=checks,
                scope='Reuse the separately installed Oct7 clean prefix; this is an import/package check, not a new installation or robot replay.')
    write(out/'receipt.json',result);print(json.dumps(result,ensure_ascii=False))
    if result['status']!='passed':
        raise SystemExit(1)


if __name__=='__main__':
    main()
