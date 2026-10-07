"""Generate development-only object/hand candidates and annotated pilot videos."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from data.pilot_labels import make_reference,detect_object,check_reference,hand_object_overlap_qc
from inspect_pilot import contact_sheet,EDGES


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def annotate(frame,obj,hands,ref,marker_record,index,ms):
    display=frame.copy()
    cv2.polylines(display,[np.rint(ref.workspace_polygon).astype(np.int32)],True,(180,130,30),1)
    for poly in ref.boundary_polygons:cv2.polylines(display,[np.int32(poly)],True,(220,100,0),2)
    for key,poly in marker_record['marker_corners_px'].items():
        p=np.rint(poly).astype(np.int32)
        cv2.polylines(display,[p],True,(220,170,30),2)
        cv2.putText(display,f'ID {key}',tuple(p[0]),cv2.FONT_HERSHEY_SIMPLEX,.5,(220,170,30),1)
    if obj['bbox_xywh_px'] is not None:
        x,y,w,h=obj['bbox_xywh_px'];cv2.rectangle(display,(x,y),(x+w,y+h),(0,230,100),2)
        cv2.putText(display,'block candidate',(x,max(55,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,210,70),2)
    for hand in hands:
        px=[(round(p[0]*frame.shape[1]),round(p[1]*frame.shape[0])) for p in hand]
        for a,b in EDGES:cv2.line(display,px[a],px[b],(0,215,255),2,cv2.LINE_AA)
        for p in px:cv2.circle(display,p,3,(0,70,255),-1,cv2.LINE_AA)
    cv2.rectangle(display,(0,0),(frame.shape[1],44),(25,25,25),-1)
    text=f'{ms/1000:5.2f}s  frame {index}  object: {obj["status"]}  hands: {len(hands)}  | PILOT QA ONLY'
    cv2.putText(display,text,(12,29),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
    if not obj['auxiliary_mask_candidate']:
        cv2.putText(display,'No object target on this frame - retained RGB',(12,frame.shape[0]-15),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,170,255),2,cv2.LINE_AA)
    return display


def process(video,prior,cfg,geometry,out,details,hand_dir,detector,cache_key):
    path=ROOT/video['source_video']; expected_sha=video['sha256']
    if sha(path)!=expected_sha:raise ValueError(f'Raw source changed: {path}')
    marker_path=ROOT/prior['frame_details']
    marker_rows=[json.loads(line) for line in marker_path.read_text(encoding='utf-8').splitlines()]
    if len(marker_rows)!=prior['frames_decoded']:raise ValueError('Incomplete prior marker diagnostics')
    stem=path.stem; hand_path=hand_dir/f'{stem}.jsonl'; hand_meta_path=hand_dir/f'{stem}.meta.json'
    hands_cached=None
    if hand_path.exists() and hand_meta_path.exists():
        meta=json.loads(hand_meta_path.read_text(encoding='utf-8'))
        if meta.get('cache_key')==cache_key and meta.get('source_sha256')==expected_sha:
            rows=[json.loads(line) for line in hand_path.read_text(encoding='utf-8').splitlines()]
            if len(rows)==prior['frames_decoded'] and all(row['frame_index']==i for i,row in enumerate(rows)):
                hands_cached=rows
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError(f'Cannot open {path}')
    fps=cap.get(cv2.CAP_PROP_FPS);ok,first=cap.read()
    if not ok:raise ValueError('Empty video')
    ref=make_reference(first,marker_rows[0]['marker_corners_px'],geometry,cfg)
    save_json(details/f'{stem}_reference.json',{'reference_frame_index':0,'source_sha256':expected_sha,
        'marker_diagnostics_sha256':sha(marker_path),'board_to_image':ref.board_to_image.tolist(),
        'workspace_polygon_px':ref.workspace_polygon.tolist(),'boundary_polygons_px':ref.boundary_polygons,
        'coordinates':'pixel labels only; board transform is a provisional reference for gates, not a 3D measurement'})
    annotated_path=out/f'{stem}_annotated.mp4';height,width=first.shape[:2]
    writer=imageio_ffmpeg.write_frames(str(annotated_path),(width,height),fps=fps,codec='libx264',quality=7,
        macro_block_size=1,pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-movflags','+faststart'])
    writer.send(None)
    count=0;states=Counter();hand_frames=0;overlap_frames=0;reference_failures=0
    tiles=[];uncertain_tiles=[];sample_indices=set(np.linspace(0,prior['frames_decoded']-1,8).round().astype(int).tolist())
    audit_indices=[];hand_output=hand_path.open('w',encoding='utf-8') if hands_cached is None else None
    episode_id='pilot_'+expected_sha[:12]
    try:
        with (details/f'{stem}_labels.jsonl').open('w',encoding='utf-8') as labels:
            frame=first
            while True:
                ms=float(cap.get(cv2.CAP_PROP_POS_MSEC));markers=marker_rows[count]
                if markers['frame_index']!=count:raise ValueError('Marker frame order mismatch')
                reference_ok,shifts=check_reference(markers['marker_corners_px'],ref,cfg['max_reference_marker_center_shift_px'])
                # This object branch cannot see hand labels or previous/future object coordinates.
                obj=detect_object(frame,ref,cfg,reference_ok=reference_ok)
                if hands_cached is None:
                    detected=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))))
                    hands=[[[float(p.x),float(p.y),float(p.z)] for p in hand] for hand in detected.hand_landmarks]
                    hand_row={'frame_index':count,'landmarks_normalized':hands,
                              'handedness':[[(c.category_name,float(c.score)) for c in row] for row in detected.handedness]}
                    hand_output.write(json.dumps(hand_row,separators=(',',':'))+'\n')
                else:hand_row=hands_cached[count];hands=hand_row['landmarks_normalized']
                in_image=[[0<=p[0]<=1 and 0<=p[1]<=1 and bool(np.isfinite(p).all()) for p in hand] for hand in hands]
                overlap=hand_object_overlap_qc(hands,obj,frame.shape)
                record={'episode_id':episode_id,'source_sha256':expected_sha,'frame_index':count,'timestamp_ms':ms,
                    'partition':'development_only','training_ready':False,'rgb_retained':True,'object':obj,
                    'hand':{**hand_row,'inside_image_mask':in_image,'auxiliary_mask_candidate':bool(hands),
                            'per_joint_model_confidence_available':False},
                    'qc':{'expected_marker_ids_found':markers['marker_ids'],'first_frame_reference_valid':reference_ok,
                          'reference_motion_diagnostic':shifts,'hand_object_overlap':overlap,
                          'occlusion_status':'unknown','missing_object_is_not_proof_of_occlusion':True}}
                labels.write(json.dumps(record,separators=(',',':'))+'\n')
                states[obj['status']]+=1;hand_frames+=bool(hands);reference_failures+=not reference_ok
                overlap_frames+=bool(overlap and overlap['possible_occlusion_or_contact'])
                display=annotate(frame,obj,hands,ref,markers,count,ms)
                writer.send(np.ascontiguousarray(cv2.cvtColor(display,cv2.COLOR_BGR2RGB)))
                if count in sample_indices:
                    tiles.append((display.copy(),f'{stem[:8]}  {ms/1000:.2f}s  {obj["status"]}'))
                    audit_indices.append(count)
                if not obj['auxiliary_mask_candidate'] and (not uncertain_tiles or count-uncertain_tiles[-1][2]>=25) and len(uncertain_tiles)<8:
                    uncertain_tiles.append((display.copy(),f'{stem[:8]} {ms/1000:.2f}s {obj["reason"]}',count))
                count+=1
                ok,frame=cap.read()
                if not ok:break
    finally:
        cap.release();writer.close()
        if hand_output is not None:hand_output.close()
    if count!=prior['frames_decoded']:raise ValueError(f'Frame count changed: {count}')
    if sha(path)!=expected_sha:raise ValueError('Raw file changed during processing')
    if hands_cached is None:save_json(hand_meta_path,{'cache_key':cache_key,'source_sha256':expected_sha,'frames':count})
    contact_sheet(tiles,out/f'{stem}_review.jpg')
    if uncertain_tiles:contact_sheet([(a,b) for a,b,_ in uncertain_tiles],out/f'{stem}_uncertain.jpg')
    # Verify the derived MP4 can be decoded in full and was not truncated.
    check=cv2.VideoCapture(str(annotated_path));written=0
    while True:
        ok,decoded=check.read()
        if not ok:break
        if decoded.shape[:2]!=(height,width):raise ValueError('Output shape changed')
        written+=1
    check.release()
    if written!=count:raise ValueError(f'Annotated output has {written}/{count} frames')
    return {'episode_id':episode_id,'source_video':video['source_video'],'source_sha256':expected_sha,
        'frame_count':count,'object_status_counts':dict(states),'hand_frames':hand_frames,
        'possible_overlap_qc_frames':overlap_frames,'reference_invalid_frames':reference_failures,
        'annotated_video':annotated_path.relative_to(ROOT).as_posix(),'annotated_output_frames_verified':written,
        'labels':(details/f'{stem}_labels.jsonl').relative_to(ROOT).as_posix(),
        'contact_sheet':(out/f'{stem}_review.jpg').relative_to(ROOT).as_posix(),
        'review_frame_indices':audit_indices,'uncertain_review_indices':[i for _,_,i in uncertain_tiles],
        'hand_cache_reused':hands_cached is not None,'raw_source_unchanged':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    manifest=json.loads((ROOT/'data/manifests/pilot_received_20261006.json').read_text(encoding='utf-8'))
    videos=manifest['episodes']
    if len({v['sha256'] for v in videos})!=len(videos):raise ValueError('Duplicate source episodes')
    # Assign whole episodes to the development pool BEFORE any new label generation.
    # Unknown sessions are conservatively kept in one scene-family group, never split by frame.
    partition={'partition':'development_only','purpose':'pipeline development; excluded from final untouched test',
        'session_identity_known':False,'grouping_rule':'all current pilots in one conservative scene-family group',
        'episodes':[{'episode_id':'pilot_'+v['sha256'][:12],'source_video':v['source_video'],'sha256':v['sha256'],
                     'group_id':'pilot_shared_scene_sessions_unconfirmed','partition':'development_only'} for v in videos],
        'duplicate_copies_excluded':'data/raw/pilot_originals','formal_train_validation_test_split':'not created'}
    save_json(ROOT/'data/manifests/pilot_development_partition.json',partition)
    cfg_path=ROOT/'configs/pilot_labeling.json';cfg=json.loads(cfg_path.read_text(encoding='utf-8-sig'))
    geometry=json.loads((ROOT/'assets/calibration/board_geometry.json').read_text(encoding='utf-8-sig'))
    qc=json.loads((ROOT/'artifacts/diagnostics/pilot_20261006/quality_report.json').read_text(encoding='utf-8'))
    by_hash={v['sha256']:v for v in qc['videos']}
    out=ROOT/'artifacts/diagnostics/pilot_labels_v2';details=ROOT/'data/processed/pilot_labels_v2';hand_dir=ROOT/'data/processed/pilot_hands_full'
    for folder in (out,details,hand_dir):folder.mkdir(parents=True,exist_ok=True)
    model=ROOT/'artifacts/models/hand_landmarker.task'
    hand_settings={k:cfg[k] for k in ['num_hands','min_hand_detection_confidence','min_hand_presence_confidence']}
    hand_settings['min_tracking_confidence']=.5
    cache_key=hashlib.sha256(json.dumps({'model':sha(model),'settings':hand_settings,'mode':'IMAGE','mediapipe':importlib.metadata.version('mediapipe')},sort_keys=True).encode()).hexdigest()
    report={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'development_candidate_labels_only',
        'training_ready':False,'selected_video_count':len(videos[:args.limit] if args.limit else videos),
        'config':cfg,'config_sha256':sha(cfg_path),'detector_source_sha256':sha(ROOT/'src/data/pilot_labels.py'),
        'hand_model_sha256':sha(model),'hand_settings':hand_settings,'hand_mode':'IMAGE independent frame inference',
        'opencv':cv2.__version__,'mediapipe':importlib.metadata.version('mediapipe'),
        'limits':['Candidate coverage is not localization accuracy.','Object labels and masks never use hand outputs.',
                  'No interpolation, tracking fill, future endpoints or inferred 3D object positions.',
                  'RGB retained for every frame; invalid auxiliary labels are null.',
                  'First frame anchors exclusions; visible-marker displacement gates the object branch.',
                  'Hand-hull overlap is a separate QC hint, not ground-truth occlusion.',
                  'Annotated videos are derived visualizations without audio; raw source files are unchanged.'],
        'videos':[]}
    options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model)),running_mode=vision.RunningMode.IMAGE,**hand_settings)
    with vision.HandLandmarker.create_from_options(options) as detector:
        for v in (videos[:args.limit] if args.limit else videos):
            summary=process(v,by_hash[v['sha256']],cfg,geometry,out,details,hand_dir,detector,cache_key)
            report['videos'].append(summary);save_json(out/'label_report.json',report)
            print(json.dumps({k:summary[k] for k in ['source_video','frame_count','object_status_counts','hand_frames','reference_invalid_frames','annotated_output_frames_verified']},ensure_ascii=False),flush=True)
    total=sum(v['frame_count'] for v in report['videos']);counts=Counter()
    for v in report['videos']:counts.update(v['object_status_counts'])
    report['totals']={'frames':total,'object_status_counts':dict(counts),'hand_frames':sum(v['hand_frames'] for v in report['videos']),
        'candidate_fraction':counts['visible_candidate']/total if total else None,
        'reference_invalid_frames':sum(v['reference_invalid_frames'] for v in report['videos'])}
    save_json(out/'label_report.json',report)
    print(json.dumps(report['totals']),flush=True)

if __name__=='__main__':main()
