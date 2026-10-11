"""New-video planar calibration: frozen board fit, X01-only blue detector."""
import argparse
import importlib.util
import sys
import traceback
import cv2
import numpy as np
from common import ROOT, OLD, PROJECT, read, write, sha, now

GOAL_RECIPE = {'fit_development_session': 'X01/train only', 'hue_bounds': [90,160],
    'saturation_thresholds': [20,25,30], 'value_bounds': [30,200],
    'close_kernel_pixels_at_5px_per_mm': 3, 'area_mm2_bounds': [35,200],
    'side_mm_bounds': [5,16], 'max_center_spread_mm': 1.5,
    'note': 'Candidate rule chosen on 16 X01 first frames. Fixed before X02; no X03 access.'}


def endpoints(path, expected):
    """Sequential decode: phone MP4 seeking can fail at a valid last frame."""
    cap=cv2.VideoCapture(str(path));first=last=None;count=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            if first is None:first=frame.copy()
            last=frame;count+=1
    finally:cap.release()
    if count!=expected or first is None:raise ValueError(f'Decode count {count} != intake {expected}')
    return first,last


def measure_blue_goals(image, calibration, module):
    scale=5.;matrix=np.diag([scale,scale,1.])@np.array(calibration['image_to_grid_mm_homography'])
    hsv=cv2.cvtColor(cv2.warpPerspective(image,matrix,(500,650)),cv2.COLOR_BGR2HSV)
    candidates={i:[] for i in range(1,5)}
    for threshold in GOAL_RECIPE['saturation_thresholds']:
        mask=((hsv[:,:,0]>=90)&(hsv[:,:,0]<=160)&(hsv[:,:,1]>=threshold)&
              (hsv[:,:,2]>=30)&(hsv[:,:,2]<=200)).astype(np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count,_,stats,centers=cv2.connectedComponentsWithStats(mask,8)
        for index,(x,y) in enumerate(module.PRIORS,1):
            options=[]
            for component in range(1,count):
                cx,cy=centers[component]/scale;sx,sy=stats[component,2:4]/scale;area=stats[component,4]/scale**2
                if abs(cx-x)<18 and abs(cy-y)<22 and 35<area<200 and 5<sx<16 and 5<sy<16:
                    options.append([cx,cy])
            if len(options)!=1:raise ValueError(f'G{index}: saturation {threshold}, candidates {options}')
            candidates[index].append(options[0])
    goals={}
    for index,points in candidates.items():
        center=np.median(points,axis=0);spread=float(np.max(np.linalg.norm(np.array(points)-center,axis=1)))
        if spread>1.5:raise ValueError(f'G{index}: unstable blue segmentation {spread}mm')
        goals[f'G{index}']={'center_grid_mm':center.tolist(),'center_grid_uv':(center/[100,130]).tolist(),
            'center_image_pixels':module.project([center],np.array(calibration['grid_to_image_homography']))[0].tolist(),
            'region_side_mm':40.,'region_axes':'board grid x right, y down',
            'definition':'virtual square centered on blue mark; not the blue mark footprint',
            'source_frame_index':0,'source':'blue component, three saturation thresholds; fixed on X01 train',
            'threshold_center_spread_mm':spread,'note':'Detector sensitivity, NOT total metric accuracy; paper/lens error unmeasured'}
    return goals


def legacy_module():
    spec=importlib.util.spec_from_file_location('frozen_s01_calibration',OLD/'prepare_s01_calibration.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class BoardDetector:
    """Ignore IDs absent from the printed template; retain duplicate valid IDs."""
    def __init__(self):
        self.inner=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

    def detectMarkers(self,image):
        corners,ids,rejected=self.inner.detectMarkers(image)
        if ids is None:return corners,ids,rejected
        keep=[i for i,x in enumerate(ids.ravel()) if x in (0,2)]
        return [corners[i] for i in keep],ids[keep] if keep else None,rejected


def main():
    p=argparse.ArgumentParser();p.add_argument('--session',choices=['X01','X02'],required=True)
    p.add_argument('--revision',default='v1');args=p.parse_args()
    assert args.revision.isalnum()
    module=legacy_module();intake_path=ROOT/f'data/processed/intake/{args.session.lower()}_intake_v1/manifest.json'
    intake=read(intake_path)
    assert intake['split']==('train' if args.session=='X01' else 'val')
    out=ROOT/f'data/processed/{args.session}_calibration_{args.revision}';out.mkdir(parents=True,exist_ok=False)
    preview=ROOT/f'artifacts/reports/{args.session}_calibration_{args.revision}';preview.mkdir(parents=True,exist_ok=False)
    geometry_path=PROJECT/'assets/calibration/board_geometry.json';geometry=read(geometry_path)
    snapshot=preview/'source_snapshot';snapshot.mkdir()
    for path in (ROOT/'calibrate_recordings.py',OLD/'prepare_s01_calibration.py',geometry_path):
        (snapshot/path.name).write_bytes(path.read_bytes())
    write(preview/'goal_recipe.json',GOAL_RECIPE)
    if args.session=='X02':
        frozen=ROOT/'artifacts/reports/X01_calibration_v2/source_snapshot/calibrate_recordings.py'
        assert sha(frozen)==sha(ROOT/'calibrate_recordings.py'),'X01 recipe code must be frozen before X02'
    detector=BoardDetector();records=[];errors=[];tiles=[]
    for row in intake['recordings']:
        try:
            path=ROOT/row['relative_video_path'];assert sha(path)==row['sha256'],'Raw hash mismatch'
            first,last=endpoints(path,row['qc']['decoded_frames'])
            boards={b:module.calibrate(first,b,geometry,detector) for b in ('source_left','target_right')}
            goals=measure_blue_goals(first,boards['target_right'],module)
            final={b:module.calibrate(last,b,geometry,detector) for b in boards}
            drift={b:float(np.max(np.linalg.norm(np.array(boards[b]['grid_quad_pixels'])-final[b]['grid_quad_pixels'],axis=1))) for b in boards}
            record={'calibration_id':row['episode_id']+'_board_local_'+args.revision,'episode_id':row['episode_id'],
                'session_id':args.session,'split':intake['split'],'raw_relative_path':row['relative_video_path'],
                'raw_sha256':row['sha256'],'first_frame_index':0,'boards':boards,'goals':goals,
                'prescribed_start_id':row['intended_start_id'],'prescribed_goal_id':row['intended_goal_id'],
                'end_frame_diagnostic':{'frame_index':row['qc']['decoded_frames']-1,'boards':final,
                    'max_grid_corner_displacement_pixels':drift,'scope':'endpoint geometry only'},
                'robot_registration':None,'object_trajectory':None,'strict_outcome':None,
                'geometry_sha256':sha(geometry_path),'tool_version':{'opencv':cv2.__version__,'numpy':np.__version__},
                'status':'board-local planar proxy; not airborne metric XYZ'}
            write(out/(row['episode_id']+'.json'),record);records.append(record)
            overlay=module.overlay(first,boards,goals,row['intended_goal_id'],
                f"{row['episode_id']} {row['intended_start_id']}->{row['intended_goal_id']} | goals from FIRST frame")
            assert cv2.imwrite(str(preview/(row['episode_id']+'_first.jpg')),overlay)
            tiles.append(cv2.resize(overlay,(640,360)))
        except Exception as exc:
            errors.append({'episode_id':row['episode_id'],'error':str(exc),'exception_type':type(exc).__name__,'traceback':traceback.format_exc()})
    for i in range(0,len(tiles),4):
        page=tiles[i:i+4]
        while len(page)<4:page.append(np.full_like(tiles[0],255))
        assert cv2.imwrite(str(preview/f'contact_{i//4+1:02d}.jpg'),cv2.vconcat([cv2.hconcat(page[:2]),cv2.hconcat(page[2:])]))
    result={'status':'completed' if not errors else 'completed_with_errors','session_id':args.session,'split':intake['split'],
        'created_utc':now(),'received':len(intake['recordings']),'calibrations_completed':len(records),'errors':errors,
        'intake_sha256':sha(intake_path),'geometry_sha256':sha(geometry_path),
        'frozen_method_sha256':sha(OLD/'prepare_s01_calibration.py'),'code_sha256':sha(ROOT/'calibrate_recordings.py'),
        'records':[{ 'episode_id':r['episode_id'],'file':str((out/(r['episode_id']+'.json')).relative_to(ROOT)),
                    'sha256':sha(out/(r['episode_id']+'.json'))} for r in records],
        'goal_centers_mean_mm':{g:np.mean([r['goals'][g]['center_grid_mm'] for r in records],axis=0).tolist() for g in ('G1','G2','G3','G4')} if records else {},
        'max_fit_rms_px':max((b['corner_fit_rms_pixels'] for r in records for b in r['boards'].values()),default=None),
        'max_endpoint_drift_px':max((v for r in records for v in r['end_frame_diagnostic']['max_grid_corner_displacement_pixels'].values()),default=None),
        'calibration_accuracy_manually_measured':False,'test_files_opened':False,'source_code_unchanged':True}
    write(out/'manifest.json',result);write(preview/'summary.json',result)
    print({k:v for k,v in result.items() if k not in ('records','goal_centers_mean_mm')},flush=True)
    if errors:sys.exit(1)


if __name__=='__main__':main()
