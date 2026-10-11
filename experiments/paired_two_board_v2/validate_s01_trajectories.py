"""Validate candidate provenance, masks, frame order and missing-data safety."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from data.pilot_labels import detect_object
from extract_s01_trajectories import make_reference


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--revision',default='S01_trajectories_v1');args=p.parse_args()
    report=ROOT/'artifacts/reports'/args.revision
    manifest=json.loads((report/'summary.json').read_text(encoding='utf-8'))
    cfg=json.loads((report/'config.json').read_text(encoding='utf-8'))
    assert sha(report/'config.json')==manifest['config_sha256']
    assert manifest['source_hashes']['extract_s01_trajectories.py']==sha(report/'source_snapshot/extract_s01_trajectories.py')
    assert manifest['source_hashes']['src/data/pilot_labels.py']==sha(report/'source_snapshot/pilot_labels.py')
    assert manifest['source_hashes']['prepare_s01_calibration.py']==sha(report/'source_snapshot/prepare_s01_calibration.py')
    assert manifest['strict_success_labels']==manifest['robot_action_labels']==manifest['metric_airborne_xyz_labels']==0
    assert len(manifest['episodes'])==16
    assert len({e['source_sha256'] for e in manifest['episodes']})==16
    total=0;counts=Counter();hand_frames=0
    for episode in manifest['episodes']:
        name=episode['episode_id']
        intake=json.loads((ROOT/f'data/processed/S01_intake_v1/{name}.json').read_text(encoding='utf-8'))
        assert sha(ROOT/intake['source']['relative_video_path'])==episode['source_sha256']
        records=[json.loads(line) for line in (ROOT/episode['frame_labels']).read_text(encoding='utf-8').splitlines()]
        assert len(records)==episode['frames']==intake['source']['frame_count']==episode['annotated_output_frames_verified']
        actual_counts=Counter()
        for index,row in enumerate(records):
            assert row['frame_index']==index and abs(row['timestamp_s']-index/intake['source']['fps'])<1e-9
            assert row['episode_id']==name and row['session_id']=='S01' and row['split']=='train'
            assert row['source_sha256']==episode['source_sha256'] and row['rgb_retained']
            assert row['object_on_table'] is None and row['object_grid_uv'] is None and row['object_robot_xyz'] is None and row['robot_action'] is None
            obj=row['object'];actual_counts[obj['status']]+=1
            assert not obj['depends_on_hand_labels'] and not obj['temporally_filled']
            if obj['auxiliary_mask_candidate']:
                assert obj['status']=='visible_candidate' and obj['bbox_xywh_px'] is not None
                x,y,w,h=obj['bbox_xywh_px'];assert 0<x<x+w<intake['source']['width'] and 0<y<y+h<intake['source']['height']
                assert np.isfinite(obj['center_px']).all() and len(obj['polygon_px'])==4
                assert len(obj['accepted_candidates'])==1
                assert all(q['valid'] for q in row['qc']['reference'].values())
            else:
                assert obj['bbox_xywh_px'] is None and obj['center_px'] is None and obj['polygon_px'] is None
            hand=row['hand'];assert len(hand['landmarks_normalized'])==len(hand['inside_image_mask'])
            assert not hand['per_joint_model_confidence_available']
            for landmarks,mask in zip(hand['landmarks_normalized'],hand['inside_image_mask']):
                assert len(landmarks)==len(mask)==21
                for point,valid in zip(landmarks,mask):
                    assert len(point)==3 and valid==bool(np.isfinite(point).all() and 0<=point[0]<=1 and 0<=point[1]<=1)
            hand_frames+=bool(hand['landmarks_normalized'])
        assert dict(actual_counts)==episode['object_status_counts']
        counts.update(actual_counts);total+=len(records)
        events=json.loads((ROOT/episode['event_labels']).read_text(encoding='utf-8'))
        assert events['strict_success'] is None and events['source_sha256']==episode['source_sha256']
        assert events['config_sha256']==manifest['config_sha256']
        for e in events['events']:
            lo,hi=e['frame_interval'];assert 0<=lo<=hi<len(records)
            assert np.allclose(e['time_interval_s'],[lo/intake['source']['fps'],hi/intake['source']['fps']])
            assert e['annotation_source']=='pseudolabel' and e['visibility']=='uncertain' and e['review_status']=='needs_visual_review'
        cap=cv2.VideoCapture(str(ROOT/episode['annotated_video']))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(records);cap.release()
    assert total==manifest['total_frames']==2604 and dict(counts)==manifest['object_status_counts'] and hand_frames==manifest['hand_frames']

    # Negative fixture using an actual first frame: erase only the object;
    # remaining small target centers, marker codes and black separator must not
    # become a replacement object. Also catches geometric/exclusion regressions.
    e=manifest['episodes'][0];intake=json.loads((ROOT/f'data/processed/S01_intake_v1/{e["episode_id"]}.json').read_text(encoding='utf-8'))
    cal=json.loads((ROOT/f'data/processed/S01_calibration_v2/{e["episode_id"]}.json').read_text(encoding='utf-8'))
    cap=cv2.VideoCapture(str(ROOT/intake['source']['relative_video_path']));ok,image=cap.read();cap.release();assert ok
    ref=make_reference(image,cal,cfg);obj=detect_object(image,ref,cfg);assert obj['auxiliary_mask_candidate']
    x,y,w,h=obj['bbox_xywh_px'];erased=image.copy();erased[max(0,y-8):y+h+8,max(0,x-8):x+w+8]=220
    assert not detect_object(erased,ref,cfg)['auxiliary_mask_candidate']
    missing=detect_object(image,ref,cfg,reference_ok=False)
    assert missing['bbox_xywh_px'] is None and not missing['auxiliary_mask_candidate']
    print(json.dumps({'passed':True,'episodes':16,'frames':total,'checks':['original hashes and immutable source snapshots','whole-session train inheritance','frame/timestamp continuity','independent object/hand masks','null unsupported geometry/actions/outcomes','bounded offline event proposals','negative real-frame object deletion and invalid reference'],'scope':'candidate preparation; not localization accuracy or physical success'},indent=2))


if __name__=='__main__':main()
