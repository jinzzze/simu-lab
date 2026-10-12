"""Plot frozen Task 2 video-derived inputs; no labels or experiments are recomputed."""
from pathlib import Path
import argparse, hashlib, json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence-root',type=Path,default=ROOT)
DATA=parser.parse_args().evidence_root
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
profiles_path=ROOT/'experiments/personal_video_timing_v4/data/processed/human_profiles_v1.json'
profile=next(p for p in read(profiles_path)['profiles'] if p['human_episode_id']=='X01_E01')
files=profile['human_derivation_file_sha256']
for name,digest in files.items(): assert sha(DATA/name)==digest,name
video=next(n for n in files if n.endswith('.mp4'))
calibration=read(DATA/next(n for n in files if 'calibration' in n))
rows=[json.loads(s) for s in (DATA/next(n for n in files if n.endswith('.jsonl'))).read_text().splitlines()]
valid=profile['valid_source_frame_indices']
progress=profile['source_progress_before_monotone_envelope']
sample=int(valid[int(np.argmin(np.abs(np.asarray(progress)-.5)))])
cap=cv2.VideoCapture(str(DATA/video))
frames=[]
for index in [0,sample]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read();assert ok
    frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
cap.release()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#172638','axes.titleweight':'bold'})
fig,axes=plt.subplots(1,3,figsize=(13,4.5),gridspec_kw={'width_ratios':[1,1,1]})
fig.subplots_adjust(left=.025,right=.98,top=.75,bottom=.24,wspace=.19)
fig.suptitle('Task 2: what the personal video contributes',x=.025,y=.98,ha='left',fontsize=18)
fig.text(.025,.865,'First X01 training clip: P1 → G3. Recorded images and the frozen training profile.',fontsize=12,color='#526875')
for ax,im in zip(axes[:2],frames): ax.imshow(im);ax.axis('off')
axes[0].set_title('Shared goal geometry',fontsize=13,pad=12)
g=calibration['goals']['G3']['center_image_pixels']
axes[0].scatter(*g,s=135,facecolors='none',edgecolors='#00b9eb',linewidths=2)
axes[0].annotate('G3',xy=g,xytext=(g[0]-220,g[1]+185),fontsize=13,color='#007cae',fontweight='bold',arrowprops={'arrowstyle':'->','color':'#007cae','lw':1.5},bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
axes[1].set_title('Visible object position',fontsize=13,pad=12)
box=rows[sample]['object']['bbox_xywh_px'];assert box is not None
x,y,w,h=box;axes[1].add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#52f54c',lw=2))
axes[1].text(.02,.02,f"Frame {sample} · {rows[sample]['timestamp_s']:.2f} s",transform=axes[1].transAxes,fontsize=10,color='white',bbox={'facecolor':'#172638','alpha':.8,'edgecolor':'none'})
t0=profile['coarse_time_interval_s'][0]
times=[rows[i]['timestamp_s']-t0 for i in valid]
knots=np.asarray(profile['progress_knots'])
ax=axes[2]
ax.plot(knots[:,0]*profile['duration_s'],knots[:,1],color='#007d96',lw=2,label='Stored guide')
ax.scatter(times,progress,s=9,color='#87949d',label='Visible samples',zorder=3)
ax.set_title('Coarse timing + progress',fontsize=13,pad=12)
ax.set_xlabel('Time within coarse window (s)',fontsize=10)
ax.set_ylabel('Image progress',fontsize=10)
ax.set_ylim(-.05,1.06);ax.set_xlim(-.05,profile['duration_s']+.05)
ax.set_yticks([0,.5,1]);ax.spines[['top','right']].set_visible(False)
ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=9,frameon=False)
fig.text(.025,.12,'Both groups use goal geometry. Only the video group adds the timing/progress guide to robot demonstrations.',fontsize=11)
fig.text(.025,.055,'Candidate image labels, not human 3D actions. Missing observations stay missing; the line is the stored guide.',fontsize=11,color='#526875')
out=ROOT/'media/task-2-video-example.png'
fig.savefig(out,dpi=180,facecolor='white');plt.close(fig)
receipt={'episode_id':profile['human_episode_id'],'selection':'First named X01 training clip; first frame and valid sample closest to 0.5 pre-envelope image progress.',
 'frame_indices':[0,sample],'source_video':video,'source_sha256':files,'profile_sha256':sha(profiles_path),
 'goal_center_image_pixels':g,'object_bbox_xywh_pixels':box,'coarse_time_interval_s':profile['coarse_time_interval_s'],
 'scope':'Visualization of frozen inputs, not a new annotation, training run or policy evaluation.'}
(ROOT/'media/task-2-figure-sources.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode('utf-8'))
print(json.dumps({'frames':[0,sample],'duration_s':profile['duration_s'],'output':out.name}))
