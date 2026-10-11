"""Decode fixed replay videos into review sheets; does not label them reviewed."""
import cv2
import numpy as np
from common import ROOT,read,write,sha,now


def main():
    out=ROOT/'artifacts/visual_review_v1';out.mkdir(parents=True,exist_ok=False);records=[]
    for tag in ('geometry_dev_replay_v1','timing_dev_replay_v1'):
        source=ROOT/'artifacts/replays'/tag;receipt=read(source/'manifest.json');video=source/'replay.mp4'
        assert sha(video)==receipt['video_sha256']
        cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(frame)
        cap.release();assert len(frames)==receipt['video_frames_verified']
        indices=np.linspace(0,len(frames)-1,12).astype(int).tolist();tiles=[]
        for i in indices:
            frame=cv2.resize(frames[i],(480,360));cv2.rectangle(frame,(0,0),(480,27),(255,255,255),-1)
            cv2.putText(frame,f'Frame {i} | {i/fps:.2f} s',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1,cv2.LINE_AA);tiles.append(frame)
        path=out/(tag+'.jpg');assert cv2.imwrite(str(path),cv2.vconcat([cv2.hconcat(tiles[i:i+4]) for i in range(0,12,4)]))
        records.append(dict(tag=tag,video_sha256=sha(video),frames_verified=len(frames),indices=indices,sheet_sha256=sha(path),
            sheet=str(path.relative_to(ROOT)).replace('\\','/'),success=receipt['outcome']['success'],lifted=receipt['outcome']['lifted']))
    write(out/'prepared.json',dict(created_utc=now(),records=records,code_sha256=sha(ROOT/'prepare_visual_qc.py'),
        scope='Frames prepared only; separate actual visual inspection required'))
    print(records)


if __name__=='__main__':main()
