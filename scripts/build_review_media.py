"""Create a fixed first-scene comparison video and a local results page."""
from pathlib import Path
import html
import json
import cv2
import numpy as np
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/reports/visual_ablation_v1'


def decode(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f'Empty video: {path}')
    return frames, fps


def main():
    result = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    videos, fpss, captions = [], [], []
    for group in 'ABCD':
        folder = ROOT / 'artifacts/runs/visual_grasp_v1' / f'{group}_seed7'
        rows = [json.loads(x) for x in (folder / 'episodes.jsonl').read_text().splitlines()]
        first = rows[0]
        assert first['scene_seed'] == 30000
        frames, fps = decode(folder / 'first_scene_demo.mp4')
        videos.append(frames); fpss.append(fps)
        captions.append(f"{group} | {'SUCCESS' if first['success'] else 'FAIL'} | XY {first['initial_xy_error_mm']:.2f} mm")
    assert len(set(fpss)) == 1
    width, height = 640, 728
    count = max(map(len, videos))
    writer = imageio_ffmpeg.write_frames(str(OUT / 'first_scene_comparison.mp4'), (width, height), fps=fpss[0],
        codec='libx264', quality=8, pix_fmt_in='rgb24', pix_fmt_out='yuv420p', macro_block_size=1,
        output_params=['-movflags', '+faststart'])
    writer.send(None)
    selected = set(np.linspace(0, count - 1, 6).round().astype(int).tolist())
    contact = []
    try:
        for index in range(count):
            canvas = np.full((height, width, 3), 247, np.uint8)
            cv2.putText(canvas, 'Same scene 30000 | training seed 7 | first, not best', (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .52, (40,40,40), 1, cv2.LINE_AA)
            for gi, frames in enumerate(videos):
                x, y = (gi % 2) * 320, 32 + (gi // 2) * 348
                cv2.putText(canvas, captions[gi], (x+8, y+18), cv2.FONT_HERSHEY_SIMPLEX, .46, (30,30,30), 1, cv2.LINE_AA)
                frame = frames[min(index, len(frames)-1)]
                canvas[y+28:y+348, x:x+320] = cv2.resize(frame, (320,320))
            if index in selected:
                contact.append(canvas.copy())
            writer.send(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
    decoded, fps = decode(OUT / 'first_scene_comparison.mp4')
    assert len(decoded) == count and abs(fps - fpss[0]) < .01
    # Two compact chronological rows of the complete 2x2 comparison.
    thumbs = [cv2.resize(x, (320,364), interpolation=cv2.INTER_AREA) for x in contact]
    sheet = np.vstack([np.hstack(thumbs[:3]), np.hstack(thumbs[3:])])
    cv2.imwrite(str(OUT / 'first_scene_contact_sheet.jpg'), sheet)
    (OUT / 'media_validation.json').write_text(json.dumps({
        'selection': 'first preregistered scene 30000, fixed training seed 7, all A/B/C/D regardless of result',
        'frames': count, 'decoded_frames': len(decoded), 'fps': fps,
        'duration_s': count/fps, 'source_lengths': list(map(len,videos)),
        'alignment': 'same simulator time; shorter completed videos hold their final frame',
        'audio': False, 'purpose': 'qualitative illustration; aggregate metrics use all 768 episodes'}, indent=2), encoding='utf-8')
    from build_project_report import main as build_report
    build_report()
    print(json.dumps({'video_frames': count, 'report': str(OUT / 'index.html')}))


if __name__ == '__main__':
    main()
