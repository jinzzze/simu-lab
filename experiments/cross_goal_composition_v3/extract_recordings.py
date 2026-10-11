"""Offline RGB candidates for new train/validation recordings; test is excluded.

The conservative object algorithm and annotation drawing are frozen legacy
functions. Only dark thresholds (90/105) were selected on X01 first frames.
Missing labels stay missing. Human physical actions and outcomes remain unknown.
"""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from common import ROOT, OLD, PROJECT, read, write, sha, now
import extract_s01_trajectories as legacy
from calibrate_recordings import BoardDetector


def process(item, cal, cfg, out, report, hand, detector):
    episode=item['episode_id'];source=ROOT/item['relative_video_path']
    assert sha(source)==item['sha256'],'Raw video changed'
    cap=cv2.VideoCapture(str(source));ok,frame=cap.read()
    if not ok:raise ValueError('First-frame decode failed')
    height,width=frame.shape[:2];fps=item['qc']['fps'];expected=item['qc']['decoded_frames']
    ref=legacy.make_reference(frame,cal,cfg)
    output=report/f'{episode}_annotated.mp4'
    writer=imageio_ffmpeg.write_frames(str(output),(width,height),fps=fps,codec='libx264',quality=6,
        pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1)
    writer.send(None)
    rows=[];trail=[];tiles=[];uncertain=[]
    samples=set(np.linspace(0,expected-1,24).astype(int).tolist())
    try:
        with (out/f'{episode}.frames.jsonl').open('w',encoding='utf-8') as stream:
            while ok:
                index=len(rows)
                qc=legacy.reference_qc(frame,cal,detector,cfg['max_reference_marker_center_shift_px']*width/1280)
                obj=legacy.detect_object(frame,ref,cfg,reference_ok=all(q['valid'] for q in qc.values()))
                rgb=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                detected=hand.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
                hands=[[[float(p.x),float(p.y),float(p.z)] for p in points] for points in detected.hand_landmarks]
                inside=[[bool(np.isfinite(p).all() and 0<=p[0]<=1 and 0<=p[1]<=1) for p in points] for points in hands]
                row={'schema_version':'cross-goal-rgb-candidates-v1','episode_id':episode,'session_id':item['session_id'],
                    'split':item['split'],'source_sha256':item['sha256'],'frame_index':index,'timestamp_s':index/fps,
                    'timestamp_basis':'frame_index / container average fps; approximate, not recovered per-frame PTS',
                    'rgb_retained':True,'training_ready':False,'object':obj,
                    'hand':{'landmarks_normalized':hands,'inside_image_mask':inside,'auxiliary_mask_candidate':bool(hands),
                        'per_joint_model_confidence_available':False,'mask_semantics':'finite/in-image, not true visibility',
                        'z_semantics':'MediaPipe relative normalized z; NOT metric height'},
                    'object_on_table':None,'object_grid_uv':None,'object_robot_xyz':None,'robot_action':None,
                    'qc':{'reference':qc,'hand_object_overlap':legacy.hand_object_overlap_qc(hands,obj,frame.shape),
                        'occlusion_status':'unknown','missing_object_is_not_proof_of_occlusion':True}}
                stream.write(json.dumps(row,separators=(',',':'))+'\n');rows.append(row)
                trail.append(obj['center_px'] if obj['auxiliary_mask_candidate'] else None)
                display=legacy.annotate(frame,row,cal,ref,trail[-75:])
                writer.send(np.ascontiguousarray(cv2.cvtColor(display,cv2.COLOR_BGR2RGB)))
                if index in samples:tiles.append(cv2.resize(display,(480,270)))
                if not obj['auxiliary_mask_candidate'] and (not uncertain or index-uncertain[-1][0]>=8) and len(uncertain)<12:
                    uncertain.append((index,cv2.resize(display,(480,270))))
                ok,frame=cap.read()
    finally:cap.release();writer.close()
    assert len(rows)==expected,'Frame count differs from intake'
    # Legacy event proposals use pixel thresholds. Evaluate them in 1280-wide
    # coordinates for consistent resolution; original pixel labels stay intact.
    event_rows=[]
    for row in rows:
        o=dict(row['object'])
        if o['center_px'] is not None:o['center_px']=(np.array(o['center_px'])*1280/width).tolist()
        event_rows.append(dict(row,object=o))
    event_cal=json.loads(json.dumps(cal))
    event_cal['boards']['target_right']['grid_quad_pixels']=(np.array(cal['boards']['target_right']['grid_quad_pixels'])*1280/width).tolist()
    proposals=legacy.event_proposals(event_rows,event_cal,fps)
    proposals.update(episode_id=episode,session_id=item['session_id'],split=item['split'],source_sha256=item['sha256'],
        calibration_id=cal['calibration_id'],code_sha256=sha(Path(__file__)),review_status='unreviewed candidates')
    write(out/f'{episode}.events.json',proposals)
    legacy.contact_sheet(tiles,report/f'{episode}_review.jpg')
    legacy.contact_sheet([t for _,t in uncertain],report/f'{episode}_uncertain.jpg')
    indices=set()
    for event in proposals['events']:
        lo,hi=event['frame_interval']
        indices.update(np.linspace(max(0,lo-5),min(expected-1,hi+5),6).astype(int).tolist())
    event_tiles=[];cap=cv2.VideoCapture(str(source));index=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            if index in indices:event_tiles.append(cv2.resize(legacy.annotate(frame,rows[index],cal,ref,[]),(480,270)))
            index+=1
    finally:cap.release()
    legacy.contact_sheet(event_tiles,report/f'{episode}_event_review.jpg')
    cap=cv2.VideoCapture(str(output));verified=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape[:2]==(height,width)
            verified+=1
    finally:cap.release()
    assert verified==expected,'Derivative video truncated'
    files=[out/f'{episode}.frames.jsonl',out/f'{episode}.events.json',output,report/f'{episode}_review.jpg']
    return {'episode_id':episode,'session_id':item['session_id'],'split':item['split'],'source_sha256':item['sha256'],
        'frames':len(rows),'object_status_counts':dict(Counter(r['object']['status'] for r in rows)),
        'hand_frames':sum(bool(r['hand']['landmarks_normalized']) for r in rows),
        'reference_invalid_frames':sum(not all(q['valid'] for q in r['qc']['reference'].values()) for r in rows),
        'review_frame_indices':sorted(samples),'uncertain_review_indices':[i for i,_ in uncertain],
        'event_review_indices':sorted(indices),'event_proposals':proposals['events'],
        'annotated_output_frames_verified':verified,'strict_success':None,
        'files':{str(f.relative_to(ROOT)).replace('\\','/'):sha(f) for f in files}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--session',choices=['X01','X02'],required=True)
    p.add_argument('--revision',default='v1');args=p.parse_args();assert args.revision.isalnum()
    cv2.setNumThreads(1)
    intake_path=ROOT/f'data/processed/intake/{args.session.lower()}_intake_v1/manifest.json';intake=read(intake_path)
    assert intake['split']==('train' if args.session=='X01' else 'val')
    cfg=read(OLD/'artifacts/reports/S01_trajectories_v1/config.json')
    cfg.update(version='cross_goal_rgb_candidates_v1',scope='train/development only',dark_max_channel=90,comparison_dark_max_channel=105,
        threshold_selection='X01/train first frames only; fixed before X02',
        reference_shift_rule='12 pixels at width 1280, scaled linearly by image width',
        note='Missing remains missing; conservative RGB candidates, not amodal, physical XYZ, actions or success.')
    model=PROJECT/'artifacts/models/hand_landmarker.task'
    paths=[Path(__file__),OLD/'extract_s01_trajectories.py',OLD/'prepare_s01_calibration.py',ROOT/'calibrate_recordings.py',
        PROJECT/'src/data/pilot_labels.py',model]
    sources={str(f.relative_to(PROJECT)).replace('\\','/'):sha(f) for f in paths}
    if args.session=='X02':
        frozen=read(ROOT/'artifacts/reports/X01_trajectories_v1/summary.json')
        assert frozen['source_hashes']==sources,'Train-frozen implementation changed before validation'
        assert frozen['config']==cfg,'Train-frozen configuration changed before validation'
    out=ROOT/f'data/processed/{args.session}_trajectories_{args.revision}'
    report=ROOT/f'artifacts/reports/{args.session}_trajectories_{args.revision}'
    out.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False)
    snapshot=report/'source_snapshot';snapshot.mkdir()
    for f in paths[:-1]:(snapshot/f.name).write_bytes(f.read_bytes())
    write(report/'config.json',cfg)
    settings={k:cfg[k] for k in ('num_hands','min_hand_detection_confidence','min_hand_presence_confidence')}
    options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model)),running_mode=vision.RunningMode.IMAGE,**settings)
    detector=BoardDetector();records=[];errors=[];started=time.monotonic()
    with vision.HandLandmarker.create_from_options(options) as hand:
        for row in intake['recordings']:
            try:
                cal=read(ROOT/f"data/processed/{args.session}_calibration_v2/{row['episode_id']}.json")
                result=process(row,cal,cfg,out,report,hand,detector);records.append(result)
                print({k:result[k] for k in ('episode_id','frames','object_status_counts','hand_frames','reference_invalid_frames')},flush=True)
            except Exception as exc:
                errors.append({'episode_id':row['episode_id'],'exception':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
                print(errors[-1],flush=True)
    result={'status':'completed' if not errors else 'completed_with_errors','created_utc':now(),
        'session_id':args.session,'split':intake['split'],'config':cfg,'source_hashes':sources,'intake_sha256':sha(intake_path),
        'episodes':records,'errors':errors,'total_frames':sum(r['frames'] for r in records),
        'elapsed_s':time.monotonic()-started,'opencv':cv2.__version__,'mediapipe':importlib.metadata.version('mediapipe'),
        'strict_success_labels':0,'physical_action_labels':0,'metric_airborne_xyz_labels':0,'test_files_opened':False,
        'scope':'current RGB candidates and offline unreviewed event proposals; coverage is not accuracy'}
    write(report/'summary.json',result);write(out/'manifest.json',result)
    print({k:result[k] for k in ('status','total_frames','elapsed_s','errors')},flush=True)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
