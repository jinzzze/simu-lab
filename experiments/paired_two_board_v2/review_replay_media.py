"""Inspect an existing simulation replay by decoding evenly spaced video frames."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    folder=ROOT/f'artifacts/reports/policy_replays/{args.tag}';meta=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    path=folder/'replay.mp4';assert sha(path)==meta['outputs_sha256']['replay.mp4']
    capture=cv2.VideoCapture(str(path));fps=capture.get(cv2.CAP_PROP_FPS);frames=[]
    while True:
        ok,frame=capture.read()
        if not ok:break
        frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    capture.release();assert len(frames)==meta['outcome']['video_frames_verified'] and fps>0
    chosen=np.rint(np.linspace(0,len(frames)-1,6)).astype(int);out=folder/'visual_review';assert not out.exists();out.mkdir()
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for ax,index in zip(axes.flat,chosen):ax.imshow(frames[index]);ax.set_title(f'frame {index}, simulation t={index/fps:.2f}s');ax.axis('off')
    fig.suptitle(f'{meta["config"]["revision"]} / seed {meta["policy_seed"]} / scene {meta["scene_seed"]}\nSelected replay diagnostic, not a representative success estimate')
    fig.tight_layout();fig.savefig(out/'contact_sheet.png',dpi=140);plt.close(fig)
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'video_sha256':sha(path),'decoded_frames':len(frames),'fps':fps,
        'frame_shape':list(frames[0].shape),'selected_frame_indices':chosen.tolist(),'output_sha256':sha(out/'contact_sheet.png'),
        'visual_review':'pending visual inspection','source_sha256':sha(__file__)}
    (out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'image':str(out/'contact_sheet.png'),'receipt':receipt},ensure_ascii=False))
if __name__=='__main__':main()
