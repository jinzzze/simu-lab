"""User-authorized test-file naming inspection; isolated from learning pipelines."""
import hashlib
from pathlib import Path
import cv2
import numpy as np
from common import ROOT, OLD, read, write, sha, now


def main():
    raw_root = ROOT/'data/raw/X03_test'
    paths = sorted(p for p in raw_root.iterdir() if p.is_file() and p.suffix.lower() in ('.mp4','.mov','.m4v'))
    assert len(paths) == 16, f'Expected 16 files, found {len(paths)}'
    out = ROOT/'data/processed/test_intake/x03_naming_v1'
    out.mkdir(parents=True,exist_ok=False)
    write(out/'access_record.json', {
        'created_utc':now(), 'session_id':'X03','split':'test',
        'authorization':'Applicant explicitly requested inspecting unnamed test videos and renaming them by content in the current chat.',
        'scope':'Technical decode and sampled frame inspection for naming only. No trajectories, metric success labels, training or tuning.',
        'test_content_exposed_for_administrative_review':True,
        'original_protocol_sha256':sha(ROOT/'protocol.json'),
        'prescribed_goal_cannot_be_proven_from_final_position':True,
        'repeat_number_rule':'Stable ordinal by original filename within each reviewed group; actual recording order remains unknown.',
        'script_sha256':sha(ROOT/'prepare_x03_name_review.py')
    })
    known_raw=set();known_pixels=set();sources={}
    for tag in ('x01_intake_v1','x02_intake_v1'):
        file=ROOT/'data/processed/intake'/tag/'manifest.json'
        data=read(file);sources[str(file.relative_to(ROOT))]=sha(file)
        known_raw.update(r['sha256'] for r in data['recordings'])
    for tag in ('x01_visual_review_v1','x02_visual_review_v1'):
        file=ROOT/'data/processed/intake'/tag/'manifest.json'
        data=read(file);sources[str(file.relative_to(ROOT))]=sha(file)
        known_pixels.update(r['decoded_pixel_stream_sha256'] for r in data['recordings'])
    import csv
    file=OLD/'capture_checklist.csv'
    with file.open(encoding='utf-8-sig',newline='') as f:
        known_raw.update(r['video_sha256'] for r in csv.DictReader(f) if r['video_sha256'])
    rows=[];pages=[]
    for i,path in enumerate(paths,1):
        digest=sha(path)
        cap=cv2.VideoCapture(str(path))
        fps=float(cap.get(cv2.CAP_PROP_FPS));declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices=sorted(set(round((declared-1)*f) for f in (0,.25,.5,.75,1)))
        frames={};n=0;shape=None;consistent=True;pixels=hashlib.sha256()
        while True:
            ok,frame=cap.read()
            if not ok:
                break
            if shape is None:shape=list(frame.shape)
            consistent=consistent and list(frame.shape)==shape
            pixels.update(frame.tobytes())
            if n in indices:frames[n]=frame.copy()
            n+=1
        cap.release()
        folder=out/f'T{i:02d}';folder.mkdir()
        samples=[]
        for tick,frame in frames.items():
            dest=folder/f'frame_{tick:04d}.jpg'
            assert cv2.imwrite(str(dest),frame,[cv2.IMWRITE_JPEG_QUALITY,95])
            samples.append({'frame':tick,'nominal_time_s':tick/fps if fps>0 else None,
                            'path':str(dest.relative_to(ROOT))})
        rows.append({'review_id':f'T{i:02d}','original_name':path.name,
            'original_relative_path':str(path.relative_to(ROOT)), 'sha256':digest,
            'bytes':path.stat().st_size,'fps':fps,'declared_frames':declared,'decoded_frames':n,
            'duration_s':n/fps if fps>0 else None,'shape':shape,'frames':samples,
            'technical_decode_pass':n>0 and n==declared and consistent and fps>0,
            'decoded_pixel_stream_sha256':pixels.hexdigest(),
            'matches_registered_train_val_file':digest in known_raw,
            'matches_x01_x02_decoded_stream':pixels.hexdigest() in known_pixels})
    for page in range((len(rows)+3)//4):
        canvas=np.full((4*262,2000,3),255,np.uint8)
        for ri,row in enumerate(rows[page*4:page*4+4]):
            y=ri*262
            label=f"{row['review_id']} {row['original_name'][:8]} ({row['duration_s']:.2f}s)"
            cv2.putText(canvas,label,(8,y+18),cv2.FONT_HERSHEY_SIMPLEX,.52,(25,25,25),1,cv2.LINE_AA)
            for ci,info in enumerate(row['frames']):
                frame=cv2.imread(str(ROOT/info['path']))
                canvas[y+30:y+255,ci*400:(ci+1)*400]=cv2.resize(frame,(400,225))
                cv2.putText(canvas,f"{info['nominal_time_s']:.2f}s",(ci*400+325,y+18),cv2.FONT_HERSHEY_SIMPLEX,.45,(40,40,40),1,cv2.LINE_AA)
        file=out/f'contact_{page+1:02d}.jpg'
        assert cv2.imwrite(str(file),canvas,[cv2.IMWRITE_JPEG_QUALITY,95])
        pages.append(str(file.relative_to(ROOT)))
    result={'status':'naming_review_frames_prepared','created_utc':now(),'session_id':'X03','split':'test',
        'recordings':rows,'contact_sheets':pages,'comparison_manifest_sha256':sources,
        'all_decode_passed':all(r['technical_decode_pass'] for r in rows),
        'unique_raw_files':len({r['sha256'] for r in rows}),
        'unique_decoded_streams':len({r['decoded_pixel_stream_sha256'] for r in rows}),
        'test_content_opened_for_authorized_naming':True,'training_allowed':False,
        'raw_names_changed':False,'raw_bytes_changed':False}
    write(out/'manifest.json',result)
    print({'out':str(out),'all_decode_passed':result['all_decode_passed'],
           'files':len(rows),'unique_raw_files':result['unique_raw_files'],
           'unique_decoded_streams':result['unique_decoded_streams'],
           'cross_split_raw_duplicates':sum(r['matches_registered_train_val_file'] for r in rows),
           'cross_split_pixel_duplicates':sum(r['matches_x01_x02_decoded_stream'] for r in rows),
           'durations':{r['review_id']:round(r['duration_s'],2) for r in rows}})


if __name__=='__main__':
    main()
