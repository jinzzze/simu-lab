"""Local archive with training/evaluation evidence; verify extracted bytes and replays."""
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from common import ROOT,OLD,VIDEOS,PROJECT,read,write,sha,now,check_sources


def main():
    assert read(ROOT/'artifacts/completion_state.json')['status']=='automated_checks_passed'
    assert read(ROOT/'artifacts/results_v1/results.json')['benchmark_executions']==264
    assert read(VIDEOS/'artifacts/reports/X03_frozen_labels_v1/summary.json')['status']=='completed'
    out=PROJECT/'work/personal_timing_release_candidate_v1';out.mkdir(parents=True,exist_ok=False)
    files=set()
    def tree(path):
        files.update(p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    # All recorded v4 checkpoints, 60 teacher trajectories, 264 evaluation
    # input/action traces, source snapshots, replays, reports and figures.
    tree(ROOT)
    for file in (ROOT/'artifacts/robot_demos_v1/manifest.json',*ROOT.glob('artifacts/models/*/manifest.json'),*ROOT.glob('artifacts/evaluation/*/*/manifest.json')):
        meta=read(file);check_sources(meta['source_sha256']);files.update(PROJECT/k for k in meta['source_sha256'])
    profiles=read(ROOT/'data/processed/human_profiles_v1.json');check_sources(profiles['source_sha256'])
    files.update(PROJECT/k for k in profiles['source_sha256'])
    files.update((PROJECT/'src').rglob('*.py'))
    files.update(p for p in VIDEOS.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json','.csv'))
    tree(VIDEOS/'configs');tree(VIDEOS/'receipts')
    for session in ('X01','X02','X03'):
        for p in (VIDEOS/'data/raw').glob(session+'*'):
            if p.is_dir():tree(p)
        for p in (VIDEOS/'data/processed').glob(session+'*'):
            if p.is_dir():tree(p)
        for p in (VIDEOS/'artifacts/reports').glob(session+'*'):
            if p.is_dir():tree(p)
    for name in ('intake','test_intake'):tree(VIDEOS/'data/processed'/name)
    tree(VIDEOS/'artifacts/reports/new_video_labels_audit_v1')
    tree(OLD/'configs')
    files.update(p for p in OLD.iterdir() if p.is_file() and p.suffix=='.py')
    # S01 candidate extraction references the frozen detector configuration.
    for file in (OLD/'artifacts/reports/S01_trajectories_v1/config.json',PROJECT/'artifacts/models/hand_landmarker.task',
        PROJECT/'assets/calibration/board_geometry.json',PROJECT/'scripts/Enter-Project.ps1',
        PROJECT/'environment.yml',PROJECT/'requirements.txt',PROJECT/'docs/ENVIRONMENT.md',PROJECT/'THIRD_PARTY_NOTICES.md'):
        assert file.exists();files.add(file)
    for name in ('LICENSE','LICENSE.md','LICENSE_DATA.md'):
        if (PROJECT/name).is_file():files.add(PROJECT/name)
    for report in ('X01_trajectories_v1','X02_trajectories_v1'):
        sources=read(VIDEOS/f'artifacts/reports/{report}/summary.json')['source_hashes']
        check_sources(sources);files.update(PROJECT/k for k in sources)
    entries={str(p.relative_to(PROJECT)).replace('\\','/'):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)}
    assert all(not k.startswith(('/', '../')) and ':' not in k for k in entries)
    manifest=dict(created_utc=now(),status='local_release_candidate',files=entries,
        scope='Recorded personal_video_timing_v4 study and dependencies, plus X01/X02/X03 video processing. Not the complete history of prior studies. No Python environment or caches.',
        not_publicly_published=True,source_files=len(entries),uncompressed_bytes=sum(v['bytes'] for v in entries.values()))
    readme='''# simu-lab: personal-video timing study — local release candidate

Start with [the current experiment](experiments/personal_video_timing_v4/README.md), [all results and limitations](experiments/personal_video_timing_v4/RESULTS_EN.md), and [the 92-second evidence walkthrough](experiments/personal_video_timing_v4/artifacts/demo_v1/evidence_walkthrough.mp4).

Recorded result: shared geometry 7/96 strict test successes; additional personal timing/progress 11/96. Four initial task images repeat training/development pixels; exclusion gives 5/84 versus 8/84. This is exploratory simulation evidence, with low overall success. No statistical significance or real-robot transfer is claimed.

The archive contains all six checkpoints, all 60 teacher trajectories and all 264 recorded evaluation input/action arrays and scoring traces, frozen code/config snapshots, selected S01 training sources, X01/X02/X03 raw videos and their recorded processing, figures, audits and exact fixed-case references. The package is specific to the current extension; earlier studies are discussed as history but their complete artifacts are not included. Python environments and caches are excluded.

## Verify the files

`PACKAGE_MANIFEST.json` lists every supplied source file with size and SHA-256. The checksum of the ZIP is stored beside it. Verify before running. Torch checkpoints are project-generated and should only be loaded from a trusted archive.

## Run a fixed case

From the extracted root, install the pinned environment if needed using `environment.yml` and `requirements.txt`; the environment notes record the Windows/CUDA setup. The package does not contain `.env`. The archive check uses a previously independent installation on the same Windows host, not a fresh installation on another computer.

```powershell
. ./scripts/Enter-Project.ps1
& ./.env/python.exe -B -u experiments/personal_video_timing_v4/replay.py --condition geometry_only_24 --seed 7 --suite validation --tag reviewer_geometry_v1
& ./.env/python.exe -B -u experiments/personal_video_timing_v4/replay.py --condition video_timing_24 --seed 7 --suite validation --tag reviewer_video_v1
```

Use a new tag for each replay. Output is written under `experiments/personal_video_timing_v4/artifacts/replays/<tag>/`. A successful reproduction check requires identical input/action arrays and complete scoring traces; it does not mean the robot succeeded on that task. The fixed development case fails strict placement for both models.

## Training and recorded evidence

Full recipes are in [the experiment README](experiments/personal_video_timing_v4/README.md) and [reproduction notes](experiments/personal_video_timing_v4/REPRODUCTION_ZH.md). Existing complete outputs are checked and reused. Full retraining requires a new source/input copy without previously generated models, evaluation, result or report output directories. Preserve this recorded archive. Replays do not prove full-training determinism.

The 24 video profiles are training-only. X02 is development-only; X03 was processed with frozen rules after all six models were fixed. X03 was previously viewed for applicant-requested naming/QC, so it is not claimed never viewed. Candidate coverage is not annotation accuracy or robot success.

This is a local review candidate. It has not been published or submitted. The applicant must review the English narrative, data-sharing scope, CV and application information before submission. The personal videos were supplied by the applicant; implementation, experiments and writing used Codex assistance.
'''
    archive=out/'simu_lab_personal_timing_candidate_v1.zip'
    write(out/'PACKAGE_MANIFEST.json',manifest)
    (out/'README.md').write_text(readme,encoding='utf-8')
    print(dict(stage='archive',files=len(entries),uncompressed_bytes=manifest['uncompressed_bytes']),flush=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(PROJECT)).replace('\\','/'))
        z.write(out/'PACKAGE_MANIFEST.json','PACKAGE_MANIFEST.json');z.write(out/'README.md','README.md')
    digest=sha(archive);(out/'SHA256SUMS.txt').write_text(digest+'  '+archive.name+'\n',encoding='ascii')
    unpacked=out/'unpacked';unpacked.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None,'Archive CRC failure'
        assert set(z.namelist())==set(entries)|{'PACKAGE_MANIFEST.json','README.md'}
        z.extractall(unpacked)
    for name,meta in entries.items():
        p=unpacked/name;assert p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256'],name
    print(dict(stage='all_extracted_bytes_verified',files=len(entries)),flush=True)
    prefix=PROJECT/'work/clean_env_v1';env=os.environ.copy()
    parts=[p for p in env['PATH'].split(os.pathsep) if not p.lower().startswith(str(PROJECT/'.env').lower())]
    env['PATH']=os.pathsep.join([str(prefix),str(prefix/'Scripts'),str(prefix/'Library/bin')]+parts)
    env.update(CONDA_PREFIX=str(prefix),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
    checks=[]
    for condition in ('geometry_only_24','video_timing_24'):
        target=unpacked/ROOT.relative_to(PROJECT);tag='release_'+condition+'_v1'
        cmd=[str(prefix/'python.exe'),'-B','-u',str(target/'replay.py'),'--condition',condition,'--suite','validation','--seed','7','--tag',tag]
        with (out/(condition+'.log')).open('x',encoding='utf-8') as log:
            run=subprocess.run(cmd,cwd=unpacked,env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert run.returncode==0,f'Extracted replay failed; retained log {condition}'
        receipt=read(target/f'artifacts/replays/{tag}/manifest.json')
        assert receipt['all_arrays_identical'] and receipt['full_scoring_trace_identical']
        checks.append(dict(condition=condition,arrays_identical=True,scoring_trace_identical=True,
            strict_success=receipt['outcome']['success'],video_frames=receipt['video_frames_verified']))
        print(dict(stage='extracted_replay_passed',condition=condition),flush=True)
    result=dict(status='passed',created_utc=now(),archive=str(archive.relative_to(PROJECT)),archive_sha256=digest,
        archive_bytes=archive.stat().st_size,files=len(entries),all_extracted_hashes_verified=True,checks=checks,
        scope='Actual replay from the extracted archive using an existing independent environment on the same Windows host. No fresh environment installation or full retraining.',
        source_manifest_sha256=sha(out/'PACKAGE_MANIFEST.json'),code_sha256=sha(ROOT/'build_release_candidate.py'),
        published=False,application_submitted=False)
    write(out/'receipt.json',result)
    write(ROOT/'receipts/release_candidate_latest.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
