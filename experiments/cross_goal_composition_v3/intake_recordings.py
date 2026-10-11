"""Inspect new train/validation recordings without renaming raw files or opening test videos."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import cv2
from common import ROOT,OLD,read,write,sha,now


def inside_session(session_root,relative_path):
    path=(session_root/relative_path).resolve()
    if not path.is_relative_to(session_root.resolve()) or path.suffix.lower() not in ('.mp4','.mov','.m4v'):
        raise ValueError('Recording path must stay inside the assigned session folder')
    return path


def inspect_video(path):
    cap=cv2.VideoCapture(str(path));fps=float(cap.get(cv2.CAP_PROP_FPS));declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames=0;shape=None;consistent=True
    while True:
        ok,frame=cap.read()
        if not ok:
            break
        if shape is None:
            shape=list(frame.shape)
        consistent=consistent and list(frame.shape)==shape;frames+=1
    cap.release()
    return dict(decoded_frames=frames,declared_frames=declared,fps=fps,shape=shape,
        duration_s=frames/fps if fps>0 else None,
        technical_pass=frames>0 and frames==declared and fps>0 and consistent,
        content_review='not_performed',strict_success='unknown',action_labels='not_recovered')


def prior_session_hashes(prior_root,current_session):
    hashes=set();sources={}
    for file in sorted(prior_root.glob('*/manifest.json')):
        prior=read(file)
        # Frame-review manifests share this directory but are not raw intakes.
        if prior.get('status')!='inspected':
            continue
        if prior['session_id']!=current_session:
            hashes.update(r['sha256'] for r in prior['recordings'])
            sources[str(file.relative_to(prior_root))]=sha(file)
    return hashes,sources


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--session',choices=['X01','X02'],required=True)
    p.add_argument('--tag',required=True)
    p.add_argument('--mapping',type=Path,help='Optional CSV episode_id,relative_video_path within assigned session; preserves original phone filenames')
    p.add_argument('--captured-at',help='Applicant-supplied actual ISO8601 recording timestamp with offset; never inferred from file times')
    p.add_argument('--setup',default='unknown')
    args=p.parse_args();assert re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',args.tag)
    if args.captured_at:
        parsed=datetime.fromisoformat(args.captured_at)
        if parsed.tzinfo is None:
            raise ValueError('Recording timestamp requires timezone offset')
    protocol=read(ROOT/'protocol.json');session=next(s for s in protocol['new_sessions'] if s['session_id']==args.session)
    assert session['split'] in ('train','val')
    session_root=ROOT/'data/raw'/session['folder']
    with (ROOT/'capture_checklist.csv').open(encoding='utf-8-sig',newline='') as f:
        planned=[r for r in csv.DictReader(f) if r['session_id']==args.session]
    mappings={r['episode_id']:Path(r['planned_video_path']).name for r in planned}
    mapping_hash=None
    if args.mapping:
        with args.mapping.open(encoding='utf-8-sig',newline='') as f:
            mapped=list(csv.DictReader(f))
        if len({r['episode_id'] for r in mapped})!=len(mapped) or not {r['episode_id'] for r in mapped}<={r['episode_id'] for r in planned}:
            raise ValueError('Duplicate or unknown episode IDs in mapping')
        mappings.update({r['episode_id']:r['relative_video_path'] for r in mapped});mapping_hash=sha(args.mapping)
    paths={eid:inside_session(session_root,path) for eid,path in mappings.items()}
    if len(set(paths.values()))!=len(paths):
        raise ValueError('One raw video assigned to multiple planned episodes')
    with (OLD/'capture_checklist.csv').open(encoding='utf-8-sig',newline='') as f:
        old_hashes={r['video_sha256'] for r in csv.DictReader(f) if r['video_sha256']}
    prior_root=ROOT/'data/processed/intake'
    other_hashes,prior_sources=prior_session_hashes(prior_root,args.session)
    records=[];missing=[];seen=set()
    for spec in planned:
        path=paths[spec['episode_id']]
        if not path.exists():
            missing.append(spec['episode_id']);continue
        digest=sha(path)
        if digest in old_hashes or digest in other_hashes or digest in seen:
            raise ValueError('Duplicate raw content across episodes/splits: '+spec['episode_id'])
        seen.add(digest)
        qc=inspect_video(path)
        records.append(dict(episode_id=spec['episode_id'],session_id=args.session,split=session['split'],
            intended_start_id=spec['start_id'],intended_goal_id=spec['goal_id'],
            relative_video_path=str(path.relative_to(ROOT)).replace('\\','/'),sha256=digest,
            captured_at=args.captured_at,capture_time_source='applicant' if args.captured_at else 'unknown',
            setup=args.setup,intended_mapping_source='predeclared manifest plus supplied filename mapping',
            mapping_visually_confirmed=False,qc=qc))
    unmapped=[str(p.relative_to(session_root)) for p in session_root.rglob('*')
              if p.is_file() and p.suffix.lower() in ('.mp4','.mov','.m4v') and p.resolve() not in paths.values()]
    out=ROOT/'data/processed/intake'/args.tag;out.mkdir(parents=True,exist_ok=False)
    result=dict(status='inspected',created_utc=now(),session_id=args.session,split=session['split'],planned=len(planned),
        received_mapped=len(records),missing_episode_ids=missing,unmapped_files=unmapped,recordings=records,
        all_planned_decode_passed=not missing and all(r['qc']['technical_pass'] for r in records),
        ready_for_profile_training=False,
        remaining=['verify pre-action P/G mapping','per-session board calibration','review transport/placement time brackets','preserve occlusion/uncertainty masks'],
        capture_protocol_sha256=sha(ROOT/'protocol.json'),mapping_sha256=mapping_hash,raw_files_modified=False,
        comparison_prior_intakes=prior_sources,other_session_raw_hashes_checked=len(other_hashes),
        legacy_raw_hashes_checked=len(old_hashes),
        test_videos_opened=False,intake_code_sha256=sha(ROOT/'intake_recordings.py'))
    write(out/'manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='recordings'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
