"""RGB object/hand candidates and offline event proposals for paired recordings.

Object branch reuses the historical conservative detector, without hand inputs,
temporal filling or endpoint-defined goals. This is a separately versioned
candidate-label derivative, NOT certified robot action or physical outcome data.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from data.pilot_labels import Reference, detect_object, dark_mask, hand_object_overlap_qc
from prepare_s01_calibration import project

EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def make_reference(first,cal,cfg):
    height,width=first.shape[:2]
    quad=np.array([[-15,-45],[115,-45],[115,165],[-15,165]],float)
    pts=np.concatenate([project(quad,np.array(b['grid_to_image_homography'])) for b in cal['boards'].values()])
    workspace_poly=cv2.convexHull(pts.astype(np.float32)).reshape(-1,2)
    workspace=np.zeros((height,width),np.uint8)
    cv2.fillPoly(workspace,[np.rint(workspace_poly).astype(np.int32)],255)
    markers=np.zeros_like(workspace)
    marker_polys=[]
    for board in cal['boards'].values():
        for corners in board['marker_corners_pixels'].values():
            poly=np.array(corners,float);marker_polys.append(poly)
            cv2.fillPoly(markers,[np.rint(poly).astype(np.int32)],255)
    markers=cv2.dilate(markers,np.ones((25,25),np.uint8))
    boundary=np.zeros_like(workspace);boundary_polys=[]
    contours,_=cv2.findContours(dark_mask(first,cfg['dark_max_channel'],3),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    left=max(p[0] for p in cal['boards']['source_left']['grid_quad_pixels'])
    right=min(p[0] for p in cal['boards']['target_right']['grid_quad_pixels'])
    for c in contours:
        rect=cv2.minAreaRect(c);a,b=rect[1]
        if min(a,b)>0 and max(a,b)/min(a,b)>4 and cv2.contourArea(c)>500 and left<rect[0][0]<right:
            cv2.drawContours(boundary,[c],-1,255,-1)
            boundary_polys.append(cv2.convexHull(c).reshape(-1,2).tolist())
    if not boundary_polys:raise ValueError('No first-frame separator found; review required')
    boundary=cv2.dilate(boundary,np.ones((13,13),np.uint8))
    # Size gate is an approximate planar support test, not an airborne scale/XYZ
    # estimator. The reference never uses a future object or hand observation.
    b=cal['boards']['source_left']
    ref=Reference(np.array(b['grid_to_image_homography']),np.array(b['image_to_grid_mm_homography']),
                  workspace,markers,boundary,{}, {},marker_polys,boundary_polys,workspace_poly,(height,width))
    return ref


def reference_qc(frame,cal,detector,max_shift):
    width=frame.shape[1];out={}
    for board,base in cal['boards'].items():
        x0,x1=(0,width//2) if board=='source_left' else (width//2,width)
        corners,ids,_=detector.detectMarkers(frame[:,x0:x1])
        found=[] if ids is None else ids.ravel().tolist()
        points={}
        for marker_id in (0,2):
            if found.count(marker_id)==1:points[marker_id]=corners[found.index(marker_id)].reshape(4,2)+[x0,0]
        if not points:
            out[board]={'valid':False,'visible_ids':[],'reason':'no_current_reference_marker'};continue
        original=np.concatenate([np.array(base['marker_corners_pixels'][str(i)],np.float32) for i in points])
        current=np.concatenate([np.array(points[i],np.float32) for i in points])
        transform,_=cv2.estimateAffinePartial2D(original,current,method=cv2.LMEDS)
        if transform is None:
            out[board]={'valid':False,'visible_ids':list(points),'reason':'reference_transform_failed'};continue
        grid=np.array(base['grid_quad_pixels'],np.float32)
        moved=cv2.transform(grid[None],transform)[0]
        grid_shift=float(np.max(np.linalg.norm(moved-grid,axis=1)))
        corner_shift=float(np.max(np.linalg.norm(current-original,axis=1)))
        out[board]={'valid':bool(max(grid_shift,corner_shift)<=max_shift),'visible_ids':list(points),
                    'grid_similarity_shift_pixels':grid_shift,'marker_corner_shift_pixels':corner_shift,
                    'method':'visible-corner similarity drift guard, not per-frame metric registration'}
    return out


def event_proposals(rows,cal,fps):
    """Offline proposals may inspect the episode; never feed them into RGB inputs."""
    usable=[r for r in rows if r['object']['auxiliary_mask_candidate']]
    if not usable:return {'events':[],'final_image_stationary_run_s':0.,'strict_success':None}
    anchor=np.array(usable[0]['object']['center_px'])
    contact=None;motion=None;arrival=None;separated=None
    for r in rows:
        qc=r['qc']['hand_object_overlap']
        if contact is None and qc and qc['possible_occlusion_or_contact']:contact=r['frame_index']
        o=r['object']
        if motion is None and o['center_px'] is not None and np.linalg.norm(np.array(o['center_px'])-anchor)>14:
            motion=r['frame_index']
    polygon=np.array(cal['boards']['target_right']['grid_quad_pixels'],np.float32)
    for end in range(11,len(rows)):
        window=rows[end-11:end+1]
        if not all(r['object']['auxiliary_mask_candidate'] for r in window):continue
        pts=np.array([r['object']['center_px'] for r in window])
        if cv2.pointPolygonTest(polygon,tuple(map(float,pts[-1])),False)>=0 and np.max(np.linalg.norm(pts-pts[-1],axis=1))<6:
            arrival=end-11;break
    if arrival is not None:
        for end in range(arrival+2,len(rows)):
            window=rows[end-2:end+1]
            if all(r['object']['auxiliary_mask_candidate'] and not (r['qc']['hand_object_overlap'] or {}).get('possible_occlusion_or_contact',False) for r in window):
                separated=end-2;break
    events=[]
    def add(name,lo,hi,method):
        if lo is None or hi is None:return
        lo=max(0,int(lo));hi=min(len(rows)-1,max(lo,int(hi)))
        events.append({'name':name,'frame_interval':[lo,hi],'time_interval_s':[lo/fps,hi/fps],
                       'annotation_source':'pseudolabel','visibility':'uncertain','review_status':'needs_visual_review',
                       'method':method,'not_robot_action':True})
    add('contact',None if contact is None else contact-2,None if contact is None else contact+2,'first 2D hand-hull/object-bbox overlap; not proof of touch')
    add('lift',contact,motion,'broad contact-to-first-image-motion search interval only; 3D lift onset/height not measured')
    add('transport',None if motion is None else motion-2,None if motion is None else motion+2,'first object image displacement above14px from pre-action visible center')
    add('placement',None if arrival is None else arrival-12,arrival,'target-board stationary 12-frame candidate; contact with table not measured')
    add('release',arrival,separated,'transition toward no detected 2D overlap; detector absence cannot establish physical release')
    add('withdraw',None if separated is None else separated-2,None if separated is None else separated+2,'first three-frame run without detected hand overlap; hand detection failure remains possible')
    final=rows[-1]['object']['center_px'];tail=0
    if final is not None:
        for r in reversed(rows):
            o=r['object']
            if not o['auxiliary_mask_candidate'] or np.linalg.norm(np.array(o['center_px'])-final)>6:break
            tail+=1
    return {'events':events,'final_image_stationary_run_s':tail/fps,'strict_success':None,
            'event_scope':'offline visual-search proposals; unreviewed contact/lift/release not certified',
            'final_image_stationary_run_note':'image-center stability only; not goal-footprint/contact/release stability scoring'}


def annotate(frame,row,cal,ref,trail):
    image=frame.copy();h,w=frame.shape[:2]
    goal=cal['goals'][cal['prescribed_goal_id']];x,y=goal['center_grid_mm']
    region=project([[x-20,y-20],[x+20,y-20],[x+20,y+20],[x-20,y+20]],np.array(cal['boards']['target_right']['grid_to_image_homography']))
    cv2.polylines(image,[np.rint(region).astype(np.int32)],True,(0,215,255),2)
    for poly in ref.boundary_polygons:cv2.polylines(image,[np.rint(poly).astype(np.int32)],True,(240,120,0),1)
    o=row['object']
    if o['bbox_xywh_px'] is not None:
        x,y,bw,bh=o['bbox_xywh_px'];cv2.rectangle(image,(x,y),(x+bw,y+bh),(0,230,90),2)
        cv2.polylines(image,[np.rint(o['polygon_px']).astype(np.int32)],True,(70,240,150),1)
    # Trail has explicit breaks; display never feeds back into labels.
    prev=None
    for point in trail:
        if point is not None and prev is not None:cv2.line(image,tuple(np.rint(prev).astype(int)),tuple(np.rint(point).astype(int)),(180,80,250),2)
        prev=point
    for hand in row['hand']['landmarks_normalized']:
        points=[(round(p[0]*w),round(p[1]*h)) for p in hand]
        for a,b in EDGES:cv2.line(image,points[a],points[b],(0,190,250),1)
        for p in points:cv2.circle(image,p,2,(0,80,250),-1)
    cv2.rectangle(image,(0,0),(w,45),(25,25,25),-1)
    cv2.putText(image,f'{row["episode_id"]}  f{row["frame_index"]} {row["timestamp_s"]:.2f}s  object:{o["status"]}  hands:{len(row["hand"]["landmarks_normalized"])}',
                (12,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
    cv2.putText(image,'VISIBLE RGB CANDIDATES | no gap filling / no physical XYZ / success unknown',(12,h-12),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,170,255),1)
    return image


def contact_sheet(tiles,path,columns=4):
    if not tiles:return
    while len(tiles)%columns:tiles.append(np.zeros_like(tiles[0]))
    image=cv2.vconcat([cv2.hconcat(tiles[i:i+columns]) for i in range(0,len(tiles),columns)])
    cv2.imwrite(str(path),image)


def process(intake,cal,cfg,out,report,hand_detector,aruco,config_hash,source_hashes):
    episode=intake['episode_id'];source=ROOT/intake['source']['relative_video_path']
    if digest(source)!=intake['source']['sha256']:raise ValueError('Raw hash mismatch')
    cap=cv2.VideoCapture(str(source));ok,frame=cap.read()
    if not ok:raise ValueError('Cannot decode first frame')
    fps=intake['source']['fps'];height,width=frame.shape[:2]
    ref=make_reference(frame,cal,cfg)
    video_path=report/f'{episode}_annotated.mp4'
    writer=imageio_ffmpeg.write_frames(str(video_path),(width,height),fps=fps,codec='libx264',quality=6,
                                      pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1)
    writer.send(None)
    rows=[];tiles=[];uncertain=[];trail=[]
    sample=set(np.linspace(0,intake['source']['frame_count']-1,24).astype(int).tolist())
    try:
        with (out/f'{episode}.frames.jsonl').open('w',encoding='utf-8') as labels:
            while ok:
                index=len(rows)
                qc=reference_qc(frame,cal,aruco,cfg['max_reference_marker_center_shift_px'])
                # Independent current RGB object detector. Hand inference happens
                # afterwards and cannot affect this target or its validity mask.
                obj=detect_object(frame,ref,cfg,reference_ok=all(b['valid'] for b in qc.values()))
                detected=hand_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))))
                hands=[[[float(p.x),float(p.y),float(p.z)] for p in h] for h in detected.hand_landmarks]
                inside=[[bool(np.isfinite(p).all() and 0<=p[0]<=1 and 0<=p[1]<=1) for p in h] for h in hands]
                overlap=hand_object_overlap_qc(hands,obj,frame.shape)
                row={'schema_version':'paired-two-board-trajectory-candidates-v1','episode_id':episode,'session_id':'S01','split':intake['split'],
                     'source_sha256':intake['source']['sha256'],'frame_index':index,'timestamp_s':index/fps,
                     'rgb_retained':True,'training_ready':False,'object':obj,
                     'hand':{'landmarks_normalized':hands,'inside_image_mask':inside,'auxiliary_mask_candidate':bool(hands),
                             'per_joint_model_confidence_available':False,'mask_semantics':'finite and inside image, NOT true per-joint visibility',
                             'z_semantics':'MediaPipe relative normalized z; NOT metric height'},
                     'object_on_table':None,'object_grid_uv':None,'object_robot_xyz':None,'robot_action':None,
                     'qc':{'reference':qc,'hand_object_overlap':overlap,'occlusion_status':'unknown','missing_object_is_not_proof_of_occlusion':True}}
                labels.write(json.dumps(row,separators=(',',':'))+'\n');rows.append(row)
                trail.append(obj['center_px'] if obj['auxiliary_mask_candidate'] else None)
                display=annotate(frame,row,cal,ref,trail[-75:])
                writer.send(np.ascontiguousarray(cv2.cvtColor(display,cv2.COLOR_BGR2RGB)))
                if index in sample:tiles.append(cv2.resize(display,(480,270)))
                if not obj['auxiliary_mask_candidate'] and (not uncertain or index-uncertain[-1][0]>=8) and len(uncertain)<12:
                    uncertain.append((index,cv2.resize(display,(480,270))))
                ok,frame=cap.read()
    finally:
        cap.release();writer.close()
    if len(rows)!=intake['source']['frame_count']:raise ValueError('Raw frame count differs from intake')
    proposals=event_proposals(rows,cal,fps)
    proposals.update(episode_id=episode,split=intake['split'],source_sha256=intake['source']['sha256'],calibration_id=cal['calibration_id'],
                     config_sha256=config_hash,code_sha256=source_hashes['extract_s01_trajectories.py'],hand_model_sha256=source_hashes['hand_landmarker.task'])
    write_json(out/f'{episode}.events.json',proposals)
    contact_sheet(tiles,report/f'{episode}_review.jpg')
    contact_sheet([t for _,t in uncertain],report/f'{episode}_uncertain.jpg')
    # Separate dense search sheet around event proposals for visual adjudication.
    event_indices=set()
    for event in proposals['events']:
        lo,hi=event['frame_interval'];event_indices.update(np.linspace(max(0,lo-5),min(len(rows)-1,hi+5),6).astype(int).tolist())
    event_tiles=[];cap=cv2.VideoCapture(str(source))
    for idx in sorted(event_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,frame=cap.read()
        if not ok:raise ValueError('Cannot decode event review frame')
        event_tiles.append(cv2.resize(annotate(frame,rows[idx],cal,ref,[]),(480,270)))
    cap.release();contact_sheet(event_tiles,report/f'{episode}_event_review.jpg')
    # Full decode of derivative video, not just container metadata.
    cap=cv2.VideoCapture(str(video_path));verified=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        if frame.shape[:2]!=(height,width):raise ValueError('Annotated resolution changed')
        verified+=1
    cap.release()
    if verified!=len(rows):raise ValueError('Annotated output truncated')
    counts=Counter(r['object']['status'] for r in rows)
    return {'episode_id':episode,'split':intake['split'],'source_sha256':intake['source']['sha256'],'frames':len(rows),
            'object_status_counts':dict(counts),'hand_frames':sum(bool(r['hand']['landmarks_normalized']) for r in rows),
            'reference_invalid_frames':sum(not all(q['valid'] for q in r['qc']['reference'].values()) for r in rows),
            'annotated_output_frames_verified':verified,'event_proposals':proposals['events'],
            'final_image_stationary_run_s':proposals['final_image_stationary_run_s'],
            'frame_labels':str((out/f'{episode}.frames.jsonl').relative_to(ROOT)).replace('\\','/'),
            'event_labels':str((out/f'{episode}.events.json').relative_to(ROOT)).replace('\\','/'),
            'annotated_video':str(video_path.relative_to(ROOT)).replace('\\','/'),
            'uncertain_review_indices':[i for i,_ in uncertain],'review_frame_indices':sorted(sample),'strict_success':None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',required=True);parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if not args.revision.replace('_','').isalnum():raise ValueError('Invalid revision')
    out=ROOT/'data/processed'/args.revision;report=ROOT/'artifacts/reports'/args.revision
    if out.exists() or report.exists():raise ValueError('Immutable revision already exists; use a new name')
    cfg=json.loads((PROJECT/'configs/pilot_labeling.json').read_text(encoding='utf-8'))
    cfg.update(version=args.revision,scope='S01 train candidate labels; separate from historical experiment',
               assumed_print_scale=1.,dark_max_channel=45,comparison_dark_max_channel=60,
               note='Object detector unchanged; updated darker thresholds selected on S01 train only. Each visible RGB component is an approximate candidate, never amodal/3D truth. Per-frame reference guard and no hand-dependent object targets.')
    cfg.pop('work_polygon_nominal_board_units',None)
    model=PROJECT/'artifacts/models/hand_landmarker.task'
    source_hashes={'extract_s01_trajectories.py':digest(__file__),'src/data/pilot_labels.py':digest(PROJECT/'src/data/pilot_labels.py'),
                   'prepare_s01_calibration.py':digest(ROOT/'prepare_s01_calibration.py'),'hand_landmarker.task':digest(model)}
    out.mkdir(parents=True);report.mkdir(parents=True)
    snapshot=report/'source_snapshot';snapshot.mkdir()
    for name,path in [('extract_s01_trajectories.py',Path(__file__)),('pilot_labels.py',PROJECT/'src/data/pilot_labels.py'),('prepare_s01_calibration.py',ROOT/'prepare_s01_calibration.py')]:
        (snapshot/name).write_bytes(path.read_bytes())
    write_json(report/'config.json',cfg);config_hash=digest(report/'config.json')
    settings={k:cfg[k] for k in ('num_hands','min_hand_detection_confidence','min_hand_presence_confidence')}
    options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model)),running_mode=vision.RunningMode.IMAGE,**settings)
    aruco=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    paths=sorted((ROOT/'data/processed/S01_intake_v1').glob('*.json'))
    if args.limit:paths=paths[:args.limit]
    episodes=[];start=time.monotonic()
    with vision.HandLandmarker.create_from_options(options) as hand_detector:
        for path in paths:
            intake=json.loads(path.read_text(encoding='utf-8'))
            cal=json.loads((ROOT/f'data/processed/S01_calibration_v2/{intake["episode_id"]}.json').read_text(encoding='utf-8'))
            result=process(intake,cal,cfg,out,report,hand_detector,aruco,config_hash,source_hashes)
            episodes.append(result)
            print(json.dumps({k:result[k] for k in ['episode_id','frames','object_status_counts','hand_frames','reference_invalid_frames']},ensure_ascii=False),flush=True)
    counts=Counter()
    for e in episodes:counts.update(e['object_status_counts'])
    summary={'revision':args.revision,'scope':'S01 train per-frame candidate labels and offline unreviewed event proposals',
             'episodes':episodes,'total_frames':sum(e['frames'] for e in episodes),'object_status_counts':dict(counts),
             'hand_frames':sum(e['hand_frames'] for e in episodes),'source_hashes':source_hashes,'config_sha256':config_hash,
             'opencv':cv2.__version__,'mediapipe':importlib.metadata.version('mediapipe'),'hand_mode':'IMAGE independent current frame',
             'hand_model_source':'existing project MediaPipe hand_landmarker.task; no new download',
             'raw_unchanged':True,'strict_success_labels':0,'robot_action_labels':0,'metric_airborne_xyz_labels':0,
             'elapsed_s':time.monotonic()-start,'training_ready':False,
             'limitations':['Candidate coverage is not accuracy.','Hand mask is in-image validity, not occlusion ground truth.','No gap filling or hand-dependent object labels.','Physical lift/contact/release and footprint containment remain unverified.']}
    write_json(report/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['revision','total_frames','object_status_counts','hand_frames','elapsed_s']},indent=2),flush=True)


if __name__=='__main__':main()
