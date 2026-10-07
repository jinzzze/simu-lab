"""Read-only raw-video QA. Outputs are diagnostics, never training splits or ground truth."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def contact_sheet(tiles, path, columns=4):
    width,height=480,302
    canvas=np.full((height*((len(tiles)+columns-1)//columns),width*columns,3),245,np.uint8)
    for j,(frame,label) in enumerate(tiles):
        scale=min(width/frame.shape[1],270/frame.shape[0])
        small=cv2.resize(frame,(round(frame.shape[1]*scale),round(frame.shape[0]*scale)))
        x=(j%columns)*width; y=(j//columns)*height
        canvas[y+32:y+32+small.shape[0],x:x+small.shape[1]]=small
        cv2.putText(canvas,label,(x+5,y+22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1,cv2.LINE_AA)
    cv2.imwrite(str(path),canvas)


def inspect(path, out, details, detector, aruco, geometry):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open {path}')
    fps=float(cap.get(cv2.CAP_PROP_FPS)); expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps<=0: raise RuntimeError(f'No valid nominal fps: {path}')
    board={int(m['id']):np.array(m['corners_mm'],np.float32) for m in geometry['markers']}
    chosen=set(np.linspace(0,max(0,expected-1),8).round().astype(int).tolist())
    stride=max(1,round(fps/5))
    records=[]; samples=[]; tiles=[]; pairs=0; any_marker=0; residuals=[]; marker_centers={k:[] for k in board}
    idx=0; previous_ms=None; nonmonotonic=0; intervals=[]; shape=None
    with (details/f'{path.stem}.jsonl').open('w',encoding='utf-8') as stream:
        while True:
            ok,frame=cap.read()
            if not ok: break
            shape=list(frame.shape)
            ms=float(cap.get(cv2.CAP_PROP_POS_MSEC))
            if previous_ms is not None:
                intervals.append(ms-previous_ms)
                nonmonotonic+=int(ms<=previous_ms)
            previous_ms=ms
            corners,ids,_=aruco.detectMarkers(frame)
            idlist=[] if ids is None else [int(i) for i in ids.ravel()]
            found={i:c.reshape(4,2) for i,c in zip(idlist,corners)}
            both=all(i in found for i in board)
            pairs+=int(both); any_marker+=int(any(i in found for i in board))
            record={'frame_index':idx,'timestamp_ms':ms,'marker_ids':idlist,'marker_corners_px':{str(i):c.tolist() for i,c in found.items()}}
            for i in board:
                if i in found: marker_centers[i].append(found[i].mean(axis=0))
            if both:
                src=np.concatenate([board[i] for i in board]); dst=np.concatenate([found[i] for i in board])
                h,_=cv2.findHomography(src,dst,0)
                if h is not None:
                    projected=cv2.perspectiveTransform(src[None],h)[0]
                    error=float(np.sqrt(np.mean(np.sum((projected-dst)**2,axis=1))))
                    residuals.append(error); record['board_fit_rmse_px']=error
            sampled=idx%stride==0
            if sampled or idx in chosen:
                rgb=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
                hands=[[[float(p.x),float(p.y),float(p.z)] for p in hand] for hand in result.hand_landmarks]
                record['hand_detection']={'regular_5hz_sample':sampled,'count':len(hands),'landmarks_normalized':hands,'handedness':[[(c.category_name,float(c.score)) for c in row] for row in result.handedness]}
                if sampled: samples.append({'frame_index':idx,'timestamp_ms':ms,'hand_count':len(hands)})
                if idx in chosen:
                    annotated=frame.copy()
                    if ids is not None: cv2.aruco.drawDetectedMarkers(annotated,corners,ids)
                    for hand in hands:
                        pixels=[(round(p[0]*frame.shape[1]),round(p[1]*frame.shape[0])) for p in hand]
                        for a,b in EDGES: cv2.line(annotated,pixels[a],pixels[b],(0,220,0),2,cv2.LINE_AA)
                        for x,y in pixels: cv2.circle(annotated,(x,y),3,(0,70,255),-1,cv2.LINE_AA)
                    tiles.append((annotated,f'{path.stem[:8]} {ms/1000:.2f}s hands={len(hands)} IDs={idlist}'))
            stream.write(json.dumps(record,separators=(',',':'))+'\n')
            idx+=1
    cap.release()
    contact_sheet(tiles,out/f'{path.stem}_annotated.jpg')
    marker_stats={}
    for i,values in marker_centers.items():
        points=np.array(values)
        if len(points):
            offset=np.linalg.norm(points-np.median(points,axis=0),axis=1)
            marker_stats[str(i)]={'frames_detected':len(values),'median_center_px':np.median(points,axis=0).tolist(),'p95_distance_from_median_px':float(np.percentile(offset,95)),'max_distance_from_median_px':float(offset.max())}
    return {'source_video':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'bytes':path.stat().st_size,'fps_reported':fps,'decoded_shape':shape,'frames_reported':expected,'frames_decoded':idx,'decoded_matches_reported':idx==expected,'duration_nominal_s':idx/fps,'last_timestamp_ms':previous_ms,'nonmonotonic_timestamps':nonmonotonic,'timestamp_step_ms_range':[float(min(intervals)),float(max(intervals))] if intervals else None,'frames_with_both_expected_markers':pairs,'both_markers_fraction':pairs/idx if idx else None,'frames_with_any_expected_marker':any_marker,'marker_stats':marker_stats,'board_fit_rmse_px_median':float(np.median(residuals)) if residuals else None,'board_fit_rmse_px_p95':float(np.percentile(residuals,95)) if residuals else None,'hand_sample_stride_frames':stride,'hand_sample_frames':len(samples),'samples_with_hand':sum(s['hand_count']>0 for s in samples),'hand_sample_fraction':sum(s['hand_count']>0 for s in samples)/len(samples) if samples else None,'hand_samples':samples,'annotated_contact_sheet':(out/f'{path.stem}_annotated.jpg').relative_to(ROOT).as_posix(),'frame_details':(details/f'{path.stem}.jsonl').relative_to(ROOT).as_posix()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='data/raw/pilot')
    parser.add_argument('--output',default='artifacts/diagnostics/pilot_20261006')
    parser.add_argument('--details',default='data/processed/pilot_qc_20261006')
    args=parser.parse_args()
    inp=ROOT/args.input; out=ROOT/args.output; details=ROOT/args.details
    for p in (out,details):
        p.resolve().relative_to(ROOT.resolve())
        p.mkdir(parents=True,exist_ok=True)
    files=sorted(p for p in inp.rglob('*') if p.is_file() and p.suffix.lower() in {'.mp4','.mov','.m4v','.avi'})
    geometry=json.loads((ROOT/'assets/calibration/board_geometry.json').read_text(encoding='utf-8-sig'))
    model=ROOT/'artifacts/models/hand_landmarker.task'
    settings={'num_hands':2,'min_hand_detection_confidence':.5,'min_hand_presence_confidence':.5,'min_tracking_confidence':.5}
    options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model)),running_mode=vision.RunningMode.IMAGE,**settings)
    aruco=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    report={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'Pilot QA only. No train/validation/test split, action labels or task success labels assigned. QC-index order is not capture order.','limits':['Hand output fraction includes frames with no hand; it is not accuracy or recall.','Marker reprojection error is in-sample fit residual, not independently validated metric accuracy.','Board geometry and flatness require physical verification; lifted object coordinates are not table-plane coordinates.','IMAGE-mode hand detection uses independent frames at nominal 5 Hz plus visual contact-sheet frames.','No flicker-free claim is inferred from nominal frame rate.'], 'opencv':cv2.__version__,'mediapipe':importlib.metadata.version('mediapipe'),'model_sha256':digest(model),'hand_settings':settings,'expected_marker_ids':geometry['expected_marker_ids'],'videos':[]}
    with vision.HandLandmarker.create_from_options(options) as detector:
        for path in files:
            try:
                record=inspect(path,out,details,detector,aruco,geometry)
                record['status']='decoded'
            except Exception as exc:
                record={'source_video':path.relative_to(ROOT).as_posix(),'status':'failed','error':str(exc)}
            report['videos'].append(record)
            (out/'quality_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in record.items() if k in ['source_video','status','error','frames_decoded','frames_with_both_expected_markers','hand_sample_frames','samples_with_hand','board_fit_rmse_px_median']}),flush=True)
    hashes={}
    for v in report['videos']:
        if 'sha256' in v: hashes.setdefault(v['sha256'],[]).append(v['source_video'])
    report['file_count']=len(files)
    report['duplicate_sha256_groups']=[v for v in hashes.values() if len(v)>1]
    report['all_files_decoded']=all(v.get('decoded_matches_reported',False) for v in report['videos']) and bool(files)
    (out/'quality_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if report['all_files_decoded'] else 1

if __name__=='__main__': raise SystemExit(main())
