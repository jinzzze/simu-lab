import copy
import json
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from data.pilot_labels import make_reference,detect_object,hand_object_overlap_qc,check_reference
CFG=json.loads((ROOT/'configs/pilot_labeling.json').read_text(encoding='utf-8-sig'))
GEOM=json.loads((ROOT/'assets/calibration/board_geometry.json').read_text(encoding='utf-8-sig'))


def fixture():
    frame=np.full((650,750,3),220,np.uint8)
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners={}
    for m in GEOM['markers']:
        p=np.int32(m['corners_mm'])*2;x,y=p[0];side=60
        frame[y:y+side,x:x+side]=cv2.cvtColor(cv2.aruco.generateImageMarker(dictionary,m['id'],side),cv2.COLOR_GRAY2BGR)
        corners[str(m['id'])]=p.tolist()
    cv2.rectangle(frame,(430,140),(442,410),(5,5,5),-1)
    reference=make_reference(frame,corners,GEOM,CFG)
    return frame,reference


def test_codes_and_boundary_alone_are_not_object():
    f,r=fixture();result=detect_object(f,r,CFG)
    assert result['status']=='missing' and result['bbox_xywh_px'] is None


def test_one_object_and_two_objects_are_distinguished():
    f,r=fixture();cv2.rectangle(f,(180,220),(220,260),(5,5,5),-1)
    one=detect_object(f,r,CFG);assert one['status']=='visible_candidate'
    cv2.rectangle(f,(270,220),(310,260),(5,5,5),-1)
    two=detect_object(f,r,CFG);assert two['status']=='ambiguous' and two['bbox_xywh_px'] is None


def test_missing_or_occluded_object_does_not_reuse_previous_box():
    f,r=fixture();full=f.copy();cv2.rectangle(full,(180,220),(220,260),(5,5,5),-1)
    assert detect_object(full,r,CFG)['auxiliary_mask_candidate']
    partial=f.copy();cv2.rectangle(partial,(180,220),(195,260),(5,5,5),-1)
    for candidate in (f,partial):
        result=detect_object(candidate,r,CFG)
        assert result['bbox_xywh_px'] is None and not result['auxiliary_mask_candidate']


def test_merged_boundary_and_clipped_workspace_are_invalid():
    f,r=fixture();cv2.rectangle(f,(410,220),(450,260),(5,5,5),-1)
    assert not detect_object(f,r,CFG)['auxiliary_mask_candidate']
    f,r=fixture();cv2.rectangle(f,(60,220),(100,260),(5,5,5),-1)
    assert not detect_object(f,r,CFG)['auxiliary_mask_candidate']


def test_frame_order_and_hand_diagnostics_do_not_change_object_labels():
    f,r=fixture();cv2.rectangle(f,(180,220),(220,260),(5,5,5),-1)
    before=detect_object(f,r,CFG);copy_before=copy.deepcopy(before)
    hand_object_overlap_qc([[[.27,.36,0]]*21],before,f.shape)
    assert before==copy_before
    blank,_=fixture();detect_object(blank,r,CFG)
    assert detect_object(f,r,CFG)==before
    assert before['depends_on_hand_labels'] is False


def test_reference_failure_nulls_labels():
    f,r=fixture();cv2.rectangle(f,(180,220),(220,260),(5,5,5),-1)
    result=detect_object(f,r,CFG,reference_ok=False)
    assert result['bbox_xywh_px'] is None and not result['auxiliary_mask_candidate']


def test_regular_sixty_percent_fragment_is_rejected():
    f,r=fixture();cv2.rectangle(f,(180,220),(204,260),(5,5,5),-1)
    result=detect_object(f,r,CFG)
    assert result['bbox_xywh_px'] is None and not result['auxiliary_mask_candidate']


def test_rotation_or_scale_about_only_visible_marker_is_rejected():
    f,r=fixture();p=r.marker_corners[0];center=p.mean(axis=0)
    rotation=np.array([[np.cos(.12),-np.sin(.12)],[np.sin(.12),np.cos(.12)]],np.float32)
    for changed in ((p-center)@rotation.T+center,(p-center)*1.08+center):
        assert np.linalg.norm(changed.mean(axis=0)-center)<1e-3
        valid,diagnostic=check_reference({'0':changed.tolist()},r,12)
        assert not valid
    valid,_=check_reference({'0':p.tolist()},r,12)
    assert valid
