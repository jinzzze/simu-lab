"""Create deterministic contact sheets for intake review; preserve raw recordings."""
from pathlib import Path
import hashlib
import cv2
import numpy as np
from common import ROOT, read, write, sha, now


def main():
    source = ROOT / 'data/processed/intake/x02_intake_v1/manifest.json'
    intake = read(source)
    assert intake['all_planned_decode_passed'] and intake['received_mapped'] == 12
    out = ROOT / 'data/processed/intake/x02_visual_review_v1'
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    for record in intake['recordings']:
        path = ROOT / record['relative_video_path']
        assert sha(path) == record['sha256']
        count = record['qc']['decoded_frames']
        selected = sorted(set(round((count-1)*f) for f in (0, .25, .5, .75, 1)))
        cap = cv2.VideoCapture(str(path))
        frames = {}; stream_hash = hashlib.sha256(); n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            stream_hash.update(frame.tobytes())
            if n in selected:
                frames[n] = frame.copy()
            n += 1
        cap.release()
        assert n == count and len(frames) == len(selected)
        folder = out / record['episode_id']; folder.mkdir()
        frame_records = []
        for tick, frame in frames.items():
            dest = folder / f'frame_{tick:04d}.jpg'
            assert cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            markers = []
            # Both paper boards use the same marker IDs; keep left/right ROIs separate.
            height, width = frame.shape[:2]
            for side, start, stop in (('left', 0, width//2), ('right', width//2, width)):
                corners, ids, _ = detector.detectMarkers(frame[:, start:stop])
                if ids is not None:
                    for c, marker_id in zip(corners, ids.ravel()):
                        markers.append({'board_roi': side, 'id': int(marker_id),
                            'corners_px': (c[0]+np.array([start, 0])).tolist()})
            frame_records.append({'frame': tick, 'nominal_time_s': tick/record['qc']['fps'],
                'path': str(dest.relative_to(ROOT)), 'aruco_candidates': markers})
        rows.append({'episode_id': record['episode_id'], 'filename': path.name,
            'frames': frame_records, 'raw_sha256': record['sha256'],
            'decoded_pixel_stream_sha256': stream_hash.hexdigest()})
    digests = [r['decoded_pixel_stream_sha256'] for r in rows]
    assert len(set(digests)) == len(rows), 'Duplicate decoded video frames'
    training_path = ROOT/'data/processed/intake/x01_visual_review_v1/manifest.json'
    training = read(training_path)
    training_intake_path = ROOT/'data/processed/intake/x01_intake_v1/manifest.json'
    training_intake = read(training_intake_path)
    assert sha(training_intake_path) == training['intake_sha256']
    for raw, prepared in zip(training_intake['recordings'], training['recordings']):
        assert raw['episode_id'] == prepared['episode_id']
        assert sha(ROOT/raw['relative_video_path']) == raw['sha256'] == prepared['raw_sha256']
    assert not set(digests).intersection(r['decoded_pixel_stream_sha256'] for r in training['recordings']), 'Decoded duplicate across X01/X02'
    pages = []
    for page in range(3):
        canvas = np.full((4*262, 5*400, 3), 255, np.uint8)
        for ri, row in enumerate(rows[page*4:page*4+4]):
            y = ri*262
            cv2.putText(canvas, row['filename'], (8,y+18), cv2.FONT_HERSHEY_SIMPLEX,.55,(25,25,25),1,cv2.LINE_AA)
            for ci, info in enumerate(row['frames']):
                frame = cv2.imread(str(ROOT/info['path']))
                canvas[y+30:y+255, ci*400:(ci+1)*400] = cv2.resize(frame,(400,225))
                cv2.putText(canvas, f"{info['nominal_time_s']:.2f}s", (ci*400+320,y+18),cv2.FONT_HERSHEY_SIMPLEX,.48,(40,40,40),1,cv2.LINE_AA)
        path = out / f'contact_{page+1:02d}.jpg'
        assert cv2.imwrite(str(path),canvas,[cv2.IMWRITE_JPEG_QUALITY,95])
        pages.append(str(path.relative_to(ROOT)))
    write(out/'manifest.json', {'status':'frames_prepared', 'created_utc':now(),
        'intake_sha256':sha(source), 'recordings':rows, 'contact_sheets':pages,
        'training_visual_manifest_sha256':sha(training_path), 'x01_x02_decoded_stream_overlap':0, 'all_decoded_streams_distinct':True, 'raw_modified':False, 'test_videos_opened':False,
        'marker_note':'4x4_50 detection candidates only; no camera calibration or physical success judgment.',
        'code_sha256':sha(ROOT/'inspect_x02_frames.py')})
    print({'out':str(out), 'videos':len(rows), 'decoded_pixel_streams_distinct':True})


if __name__ == '__main__':
    main()

