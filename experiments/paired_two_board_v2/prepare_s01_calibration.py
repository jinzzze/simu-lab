"""Two board calibration previews, independent of policy and outcome labels.

Stage this script in experiments/paired_two_board_v2 before executing it.
Each goal is measured ONLY in frame zero, before transport. End frames are
diagnostics and never supply the goal. No 3D trajectory/action/success inferred.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
OUT = ROOT / 'data/processed/S01_calibration_v2'
REPORT = ROOT / 'artifacts/reports/S01_calibration_v2'
GEOMETRY = PROJECT / 'assets/calibration/board_geometry.json'
GRID = np.array([[55., 70.], [155., 70.], [155., 200.], [55., 200.]])
PRIORS = [(25., 32.5), (75., 32.5), (25., 97.5), (75., 97.5)]


def project(points, matrix):
    return cv2.perspectiveTransform(np.asarray(points, dtype=np.float64)[None], matrix)[0]


def calibrate(image, board, geometry, detector):
    height, width = image.shape[:2]
    x0, x1 = (0, width // 2) if board == 'source_left' else (width // 2, width)
    corners, ids, _ = detector.detectMarkers(image[:, x0:x1])
    found = [] if ids is None else ids.ravel().tolist()
    if sorted(found) != [0, 2]:
        raise ValueError(f'{board}: require exactly one ID0 and ID2 in locked ROI, got {found}')
    pixel, page, observations = [], [], {}
    for marker_id in (0, 2):
        observed = corners[found.index(marker_id)].reshape(4, 2) + [x0, 0]
        expected = next(m['corners_mm'] for m in geometry['markers'] if m['id'] == marker_id)
        pixel.extend(observed)
        page.extend(expected)
        observations[str(marker_id)] = observed.tolist()
    homography, _ = cv2.findHomography(np.array(pixel), np.array(page), 0)
    if homography is None or not np.isfinite(homography).all():
        raise ValueError('Invalid homography')
    error = np.linalg.norm(project(page, np.linalg.inv(homography)) - pixel, axis=1)
    if np.sqrt(np.mean(error**2)) > 2 or error.max() > 4:
        raise ValueError(f'{board}: corner-fit quality gate failed {error.tolist()}')
    page_to_grid = np.array([[1, 0, -55], [0, 1, -70], [0, 0, 1]], float)
    image_to_grid = page_to_grid @ homography
    return {
        'board_id': board, 'roi_xyxy_pixels': [x0, 0, x1, height],
        'marker_corners_pixels': observations,
        'image_to_grid_mm_homography': image_to_grid.tolist(),
        'grid_to_image_homography': np.linalg.inv(image_to_grid).tolist(),
        'corner_fit_rms_pixels': float(np.sqrt(np.mean(error**2))),
        'corner_fit_max_pixels': float(error.max()),
        'grid_quad_pixels': project(GRID, np.linalg.inv(homography)).tolist(),
        'status': 'board-local metric proxy, conditional on flat printed template; not robot registration',
    }


def measure_goals(image, calibration):
    scale = 5.
    matrix = np.diag([scale, scale, 1.]) @ np.array(calibration['image_to_grid_mm_homography'])
    gray = cv2.cvtColor(cv2.warpPerspective(image, matrix, (500, 650)), cv2.COLOR_BGR2GRAY)
    goals = {}
    for index, (x, y) in enumerate(PRIORS, 1):
        candidates = []
        # 120 merged grid/pencil lines into target marks on six S01 clips.
        # Fixed on TRAIN S01 only; three darker thresholds must all agree.
        for threshold in (60, 80, 100):
            mask = (gray < threshold).astype(np.uint8)
            count, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
            options = []
            for component in range(1, count):
                cx, cy = centers[component] / scale
                sx, sy = stats[component, cv2.CC_STAT_WIDTH:cv2.CC_STAT_HEIGHT+1] / scale
                area = stats[component, cv2.CC_STAT_AREA] / scale**2
                if abs(cx-x) < 18 and abs(cy-y) < 22 and 35 < area < 200 and 5 < sx < 16 and 5 < sy < 16:
                    options.append([cx, cy])
            if len(options) != 1:
                raise ValueError(f'G{index}: threshold {threshold}, ambiguous/absent center candidates {options}')
            candidates.append(options[0])
        center = np.median(candidates, axis=0)
        spread = float(np.max(np.linalg.norm(np.array(candidates) - center, axis=1)))
        if spread > 1.5:
            raise ValueError(f'G{index}: segmentation-threshold instability {spread} mm')
        goals[f'G{index}'] = {
            'center_grid_mm': center.tolist(), 'center_grid_uv': (center / [100, 130]).tolist(),
            'center_image_pixels': project([center], np.array(calibration['grid_to_image_homography']))[0].tolist(),
            'region_side_mm': 40., 'region_axes': 'board grid x right, y down',
            'definition': 'virtual square centered on blue mark; not the blue mark footprint',
            'source_frame_index': 0, 'source': 'dark square component in target grid quadrant; threshold 60/80/100 agreement',
            'threshold_center_spread_mm': spread,
            'note': 'threshold spread is detector sensitivity, NOT total metric accuracy; paper warp/lens error unmeasured',
        }
    return goals


def overlay(image, boards, goals, selected, title):
    image = image.copy()
    for board, calibration in boards.items():
        color = (60, 60, 240) if board == 'source_left' else (240, 150, 50)
        quad = np.rint(calibration['grid_quad_pixels']).astype(np.int32)
        cv2.polylines(image, [quad], True, color, 2)
        for marker_id, corners in calibration['marker_corners_pixels'].items():
            cv2.polylines(image, [np.rint(corners).astype(np.int32)], True, (0, 190, 0), 2)
        cv2.putText(image, f'{board}: fit RMS {calibration["corner_fit_rms_pixels"]:.2f}px',
                    (15 if board == 'source_left' else image.shape[1]//2 + 15, 50), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    target = boards['target_right']
    for goal_id, goal in goals.items():
        x, y = goal['center_grid_mm']
        region = [[x-20, y-20], [x+20, y-20], [x+20, y+20], [x-20, y+20]]
        polygon = np.rint(project(region, np.array(target['grid_to_image_homography']))).astype(np.int32)
        color = (0, 210, 250) if goal_id == selected else (220, 150, 30)
        cv2.polylines(image, [polygon], True, color, 3 if goal_id == selected else 1)
        center = np.rint(project([[x, y]], np.array(target['grid_to_image_homography']))[0]).astype(int)
        cv2.drawMarker(image, tuple(center), color, cv2.MARKER_CROSS, 12, 2)
        cv2.putText(image, goal_id, tuple(center + [12, -9]), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    cv2.rectangle(image, (0, image.shape[0]-32), (image.shape[1], image.shape[0]), (30, 30, 30), -1)
    cv2.putText(image, title, (12, image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    return image


def main():
    if OUT.exists() or REPORT.exists():
        raise RuntimeError('Immutable output revision exists; choose a new revision name rather than overwrite')
    OUT.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    geometry = json.loads(GEOMETRY.read_text(encoding='utf-8'))
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    with (ROOT/'capture_checklist.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['session_id']=='S01' and r['capture_status']=='received']
    records, errors, contact = [], [], []
    for row in rows:
        episode = row['episode_id']
        try:
            path = ROOT / row['source_video_path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['video_sha256']:
                raise ValueError('Raw hash mismatch')
            cap = cv2.VideoCapture(str(path))
            ok, first = cap.read()
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, count-1)
            end_ok, last = cap.read()
            cap.release()
            if not ok or not end_ok:
                raise ValueError('Frame decode failed')
            boards = {b: calibrate(first, b, geometry, detector) for b in ('source_left', 'target_right')}
            goals = measure_goals(first, boards['target_right'])
            final = {b: calibrate(last, b, geometry, detector) for b in ('source_left', 'target_right')}
            drift = {b: float(np.max(np.linalg.norm(np.array(boards[b]['grid_quad_pixels']) - final[b]['grid_quad_pixels'], axis=1))) for b in boards}
            record = {
                'calibration_id': f'{episode}_board_local_v1', 'episode_id': episode, 'session_id': 'S01', 'split': row['split'],
                'raw_relative_path': row['source_video_path'], 'raw_sha256': row['video_sha256'],
                'first_frame_index': 0, 'boards': boards, 'goals': goals, 'prescribed_goal_id': row['goal_id'],
                'end_frame_diagnostic': {'frame_index': count-1, 'boards': final, 'max_grid_corner_displacement_pixels': drift,
                                         'scope': 'endpoint camera/layout check only; not all-frame stability certification'},
                'robot_registration': None, 'object_trajectory': None, 'strict_outcome': None,
                'measurement_reference': 'layout_measurements.json',
                'geometry_sha256': hashlib.sha256(GEOMETRY.read_bytes()).hexdigest(),
                'tool_version': {'opencv': cv2.__version__, 'numpy': np.__version__},
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'status': 'calibration preview; grid-local coordinates only; object/event labels pending',
            }
            (OUT/f'{episode}.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
            start_overlay = overlay(first, boards, goals, row['goal_id'], f'{episode} {row["start_id"]}->{row["goal_id"]} | FIRST FRAME: goals fixed here | 40mm virtual regions')
            end_overlay = overlay(last, final, goals, row['goal_id'], f'{episode} | END FRAME diagnostic | no automatic strict success label')
            cv2.imwrite(str(REPORT/f'{episode}_start.jpg'), start_overlay)
            cv2.imwrite(str(REPORT/f'{episode}_end.jpg'), end_overlay)
            contact.append(cv2.hconcat([cv2.resize(start_overlay,(640,360)),cv2.resize(end_overlay,(640,360))]))
            records.append(record)
        except Exception as exc:
            errors.append({'episode_id': episode, 'error': str(exc)})
    for index in range(0,len(contact),4):
        cv2.imwrite(str(REPORT/f'review_{index//4+1}.jpg'), cv2.vconcat(contact[index:index+4]))
    summary = {'received':len(rows), 'calibrations_completed':len(records), 'errors':errors,
               'goal_centers_summary_grid_mm':{}, 'strict_success_labels':0,
               'scope':'real raw frame calibration previews, not a learned-policy benchmark or complete labels'}
    for goal in ('G1','G2','G3','G4'):
        pts = np.array([r['goals'][goal]['center_grid_mm'] for r in records])
        if len(pts):
            summary['goal_centers_summary_grid_mm'][goal] = {'mean':pts.mean(axis=0).tolist(), 'std':pts.std(axis=0).tolist(), 'n':len(pts)}
    if records:
        summary['max_corner_fit_rms_pixels'] = max(b['corner_fit_rms_pixels'] for r in records for b in r['boards'].values())
        summary['max_endpoint_grid_corner_displacement_pixels'] = max(v for r in records for v in r['end_frame_diagnostic']['max_grid_corner_displacement_pixels'].values())
    (REPORT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    if errors or len(records)!=16:
        sys.exit(1)


if __name__ == '__main__':
    main()
