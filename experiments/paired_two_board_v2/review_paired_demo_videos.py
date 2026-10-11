"""Build indexed physical execution review sheets from recorded MP4s."""
from pathlib import Path
import argparse,json,hashlib
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',required=True);parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    report=ROOT/'artifacts/reports'/args.revision;manifest=read(report/'manifest.json')
    assert manifest['status']=='completed' or (args.allow_partial and manifest['status']=='collecting')
    partial=manifest['status']!='completed'
    receipt=report/('video_review_sheets_partial.json' if partial else 'video_review_sheets.json')
    if receipt.exists():raise ValueError('Preserve previous review receipt')
    records=[]
    for row in manifest['episodes']:
        if 'video' not in row:continue
        path=ROOT/row['video'];cap=cv2.VideoCapture(str(path));frames=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(frame)
        cap.release()
        assert len(frames)==row['video_frames_verified']
        meta=read((ROOT/row['file']).with_suffix('.metadata.json'))
        trace=read((ROOT/row['file']).with_suffix('.scoring.json'))
        indices=np.linspace(0,len(frames)-1,6).round().astype(int).tolist()
        canvas=np.full((790,960,3),255,np.uint8)
        title=f"{row['episode_id']} | intended {row['start_id']}->{row['goal_id']} | oracle success {row['success']}"
        cv2.putText(canvas,title,(10,28),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),1,cv2.LINE_AA)
        for tile,index in enumerate(indices):
            y=45+(tile//3)*360;x=(tile%3)*320
            image=cv2.resize(frames[index],(320,320));canvas[y:y+320,x:x+320]=image
            step=2*index;t=step*meta['control_dt_s'];stage='initial' if step==0 else trace[step-1]['stage']
            cv2.putText(canvas,f"frame {index}, t={t:.2f}s",(x+6,y+337),cv2.FONT_HERSHEY_SIMPLEX,.46,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(canvas,stage,(x+6,y+354),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(canvas,'Physical oracle demonstration; this is not a learned-policy evaluation.',(10,783),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1,cv2.LINE_AA)
        out=report/f"{row['episode_id']}_execution_review.jpg"
        ok,encoded=cv2.imencode('.jpg',canvas);assert ok
        if out.exists():assert out.read_bytes()==encoded.tobytes(),'Never replace a different prior review sheet'
        else:out.write_bytes(encoded.tobytes())
        records.append({'episode_id':row['episode_id'],'video':row['video'],'video_sha256':sha(path),
                        'full_decode_frames':len(frames),'sampled_indices':indices,
                        'review_sheet':str(out.relative_to(ROOT)).replace('\\','/'),'review_sheet_sha256':sha(out)})
    receipt.write_text(json.dumps({'revision':args.revision,'partial':partial,'sheets':records,'scope':'indexed samples and full decode of currently completed episode MP4s; separate visual review required'},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'revision':args.revision,'review_sheets':len(records),'all_recorded_videos_decoded':True},indent=2))


if __name__=='__main__':main()
