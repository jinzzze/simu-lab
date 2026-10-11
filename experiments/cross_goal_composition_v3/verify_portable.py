"""Relocate a minimal replay capsule and verify it with a separate installed environment."""
import os
import argparse
import re
from pathlib import Path
import shutil
import subprocess
from common import ROOT,PROJECT,read,write,sha,now,check_sources


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',choices=['validation','test'],default='validation')
    parser.add_argument('--revision',default='v2');args=parser.parse_args()
    assert re.fullmatch(r'v[1-9][0-9]*',args.revision)
    out=ROOT/('receipts/portable_replay_'+args.revision);out.mkdir(parents=True,exist_ok=False)
    capsule=PROJECT/('work/cross_goal_replay_capsule_'+args.revision)
    if capsule.exists():
        raise FileExistsError(capsule)
    capsule.mkdir()
    files=set();cfg=read(ROOT/'configs/policy_v1.json');study=read(ROOT/'configs/study_v1.json')
    case=study[args.suite+'_cases'][0]['case_id']
    for condition in cfg['conditions']:
        for kind in ('models','evaluation/'+args.suite):
            folder=ROOT/f'artifacts/{kind}/{condition}_seed7';m=read(folder/'manifest.json')
            assert m['status']=='completed';check_sources(m['source_sha256'])
            files.update(PROJECT/x for x in m['source_sha256'])
            files.add(folder/'manifest.json')
            if kind=='models':
                files.add(folder/'checkpoint.pt')
            else:
                files.update([folder/'episodes.json',folder/(case+'_inputs_actions.npz'),folder/(case+'_scoring.json')])
    files.add(ROOT/'replay.py')
    # Package initializers import additional modules. Include the full small
    # Python source tree rather than relying on the main scripts' import lists.
    files.update((PROJECT/'src').rglob('*.py'))
    files.update([PROJECT/'requirements.txt',PROJECT/'environment.yml'])
    hashes={}
    for file in sorted(files):
        rel=file.relative_to(PROJECT);target=capsule/rel
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
        assert sha(file)==sha(target);hashes[str(rel).replace('\\','/')]=sha(file)
    write(capsule/'CAPSULE_MANIFEST.json',dict(created_utc=now(),source_files_sha256=hashes,
        scope='Two seed7 checkpoints plus first-case references and exact required source; replay only, not full training data'))
    prefix=PROJECT/'work/clean_env_v1';python=prefix/'python.exe'
    env=os.environ.copy();old=str(PROJECT/'.env').lower()
    parts=[p for p in env['PATH'].split(os.pathsep) if not p.lower().startswith(old)]
    env['PATH']=os.pathsep.join([str(prefix),str(prefix/'Scripts'),str(prefix/'Library/bin')]+parts)
    env.update(CONDA_PREFIX=str(prefix),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None)
    checks=[]
    for condition in cfg['conditions']:
        script=capsule/ROOT.relative_to(PROJECT)/'replay.py'
        tag=condition+'_portable_first_case_v1'
        command=[str(python),'-B','-u',str(script),'--condition',condition,'--seed','7','--suite',args.suite,'--tag',tag]
        with (out/(condition+'.log')).open('x',encoding='utf-8') as log:
            p=subprocess.run(command,cwd=capsule,env=env,stdout=log,stderr=subprocess.STDOUT,
                             creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        rp=script.parent/'artifacts/replays'/tag
        checks.append(dict(condition=condition,returncode=p.returncode))
        if p.returncode:
            write(out/'receipt.json',dict(status='failed',created_utc=now(),checks=checks,capsule=str(capsule)))
            raise RuntimeError('Portable replay failed; inspect retained log')
        receipt=read(rp/'manifest.json')
        assert receipt['all_arrays_identical'] and receipt['full_scoring_trace_identical']
        checks[-1].update(all_arrays_identical=True,full_scoring_trace_identical=True,
                          success=receipt['outcome']['success'],case=case,video_frames_verified=receipt['video_frames_verified'])
        shutil.copy2(rp/'manifest.json',out/(condition+'_replay_receipt.json'))
    for rel,digest in hashes.items():
        assert sha(capsule/rel)==digest
    result=dict(status='passed',created_utc=now(),checks=checks,source_files=len(hashes),suite=args.suite,
        capsule=str(capsule.relative_to(PROJECT)),runtime=str(prefix.relative_to(PROJECT)),
        scope='Actual two-case replay from relocated source using an independently installed Oct7 prefix on the same Windows host. Not a fresh installation today, another-machine check or full retraining.',
        capsule_manifest_sha256=sha(capsule/'CAPSULE_MANIFEST.json'))
    write(out/'receipt.json',result);print(__import__('json').dumps(result,ensure_ascii=False,indent=2))
    write(ROOT/'receipts/portable_replay_latest.json',dict(status='passed',
        receipt=str((out/'receipt.json').relative_to(ROOT)).replace('\\','/'),
        receipt_sha256=sha(out/'receipt.json'),retained_first_failure='receipts/portable_replay_v1/receipt.json'))


if __name__=='__main__':
    main()
