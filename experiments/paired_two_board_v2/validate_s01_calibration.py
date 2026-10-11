"""Provenance, goal timing, board separation and geometric quality checks."""
from pathlib import Path
import csv
import hashlib
import json
import cv2
import numpy as np
from prepare_s01_calibration import calibrate, project

root=Path(__file__).resolve().parent
project_root=root.parents[1]
measure=json.loads((root/'layout_measurements.json').read_text(encoding='utf-8'))
assert measure['grid_width_mm']==100 and measure['grid_height_mm']==130
assert measure['horizontal_grid_edge_gap_mm']==130
assert measure['derived_horizontal_grid_center_spacing_mm']==230
assert measure['common_frame']['robot_registration'] is None
with (root/'capture_checklist.csv').open(encoding='utf-8-sig',newline='') as f:
    rows={r['episode_id']:r for r in csv.DictReader(f) if r['session_id']=='S01'}
paths=sorted((root/'data/processed/S01_calibration_v2').glob('*.json'))
assert len(paths)==16
for path in paths:
    item=json.loads(path.read_text(encoding='utf-8'))
    row=rows[item['episode_id']]
    assert item['split']==row['split']=='train'
    assert item['raw_relative_path']==row['source_video_path']
    assert item['raw_sha256']==row['video_sha256']==hashlib.sha256((root/item['raw_relative_path']).read_bytes()).hexdigest()
    assert item['prescribed_goal_id']==row['goal_id']
    assert item['first_frame_index']==0
    assert item['robot_registration'] is None and item['strict_outcome'] is None
    assert item['code_sha256']==hashlib.sha256((root/'prepare_s01_calibration.py').read_bytes()).hexdigest()
    assert item['geometry_sha256']==hashlib.sha256((project_root/'assets/calibration/board_geometry.json').read_bytes()).hexdigest()
    for board,b in item['boards'].items():
        assert set(b['marker_corners_pixels'])=={'0','2'}
        assert b['board_id']==board
        assert b['corner_fit_rms_pixels']<=2 and b['corner_fit_max_pixels']<=4
        width=json.loads((root/'data/processed/S01_intake_v1'/path.name).read_text())['source']['width']
        assert b['roi_xyxy_pixels'][0:3:2]==([0,width//2] if board=='source_left' else [width//2,width])
        H=np.array(b['image_to_grid_mm_homography'])
        metric=np.array([[0.,0.],[100.,0.],[100.,130.],[0.,130.],[50.,65.]])
        recovered=project(project(metric,np.linalg.inv(H)),H)
        assert np.max(abs(recovered-metric))<1e-6
    for g in item['goals'].values():
        assert g['source_frame_index']==0 and g['region_side_mm']==40
        xy=np.array(g['center_grid_mm'])
        assert np.allclose(xy/[100,130],g['center_grid_uv'])
        assert np.all(xy-20>=0) and np.all(xy+20<=[100,130])
        assert g['threshold_center_spread_mm']<=1.5
    assert max(item['end_frame_diagnostic']['max_grid_corner_displacement_pixels'].values())<5

# Negative real-frame fixture: source board must not fall back to target's
# duplicate IDs when the source ROI is empty.
first=json.loads(paths[0].read_text())
cap=cv2.VideoCapture(str(root/first['raw_relative_path']))
ok,image=cap.read();cap.release();assert ok
masked=image.copy();masked[:,:image.shape[1]//2]=255
detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
geometry=json.loads((project_root/'assets/calibration/board_geometry.json').read_text())
calibrate(masked,'target_right',geometry,detector)
try:
    calibrate(masked,'source_left',geometry,detector)
except ValueError:
    pass
else:
    raise AssertionError('Source board silently reused target IDs')
print(json.dumps({'passed':True,'episodes':16,'checks':['raw hashes and train inheritance','pre-action-only target definition','local geometry and quality gates','no robot/strict-outcome claims','duplicate-ID negative fixture'],'scope':'calibration preparation only'},indent=2))
