from pathlib import Path
import hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import argparse
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description='Plot frozen Task 1 labels and results; no training or evaluation.')
parser.add_argument('--data', type=Path, default=ROOT/'data/processed/pilot_pretraining_v1.npz')
DATA = parser.parse_args().data
MEDIA = ROOT / 'media'
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = read(ROOT / 'data/manifests/pilot_pretraining_v1.json')
assert digest(DATA) == manifest['npz_sha256']
data = np.load(DATA, allow_pickle=False)
# A deterministic illustration selection, independent of robot outcomes.
eligible = np.flatnonzero(data['object_mask'] & data['hand_mask'].all(axis=1))
i = int(eligible[0])
rgb = data['images'][i]
box = data['object_targets'][i]
points = data['hand_targets'][i].reshape(21, 2)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titleweight':'bold','text.color':'#172638','axes.labelcolor':'#172638'})
fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
fig.subplots_adjust(left=.015,right=.985,top=.79,bottom=.16,wspace=.04)
fig.suptitle('Task 1: what the auxiliary labels teach the visual model', fontsize=17, x=.025, ha='left', y=.98)
titles = ['Shared RGB input', 'Object-box target', 'Hand-keypoint target']
for ax, title in zip(axes, titles):
    ax.imshow(rgb, extent=(0,1,1,0), interpolation='nearest')
    ax.set_title(title, fontsize=13, pad=10)
    ax.axis('off')
cx,cy,w,h = box
axes[1].add_patch(Rectangle((cx-w/2,cy-h/2),w,h,fill=False,edgecolor='#51f54c',linewidth=2))
edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
for a,b in edges: axes[2].plot(points[[a,b],0],points[[a,b],1],color='#ffcf36',lw=1.4)
axes[2].scatter(points[:,0],points[:,1],s=10,color='#ffcf36',edgecolors='#8c5c00',linewidths=.3,zorder=3)
fig.text(.025,.105,'Same recorded frame in all panels. Green: visible-object box. Yellow: 21 hand landmarks.',fontsize=11)
fig.text(.025,.047,'Overlays illustrate stored training targets; the model receives the unannotated RGB image.',fontsize=11,color='#526875')
fig.savefig(MEDIA/'task-1-supervision.png',dpi=180,facecolor='white')
plt.close(fig)

report = ROOT/'artifacts/reports/visual_ablation_v1'
primary = read(report/'results.json')
low = read(report/'low32_results.json')
reference = read(report/'imagenet_reference.json')
rows = primary['groups']
if isinstance(rows, dict): rows=list(rows.values())
high = {r['group']:r for r in rows}
high['R']=reference['reference']
low = {r['group']:r for r in low['groups']}
groups=list('ABCDR'); colors=['#6f8190','#258b9b','#c6943c','#9c72ae','#999999']
fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
fig.subplots_adjust(left=.07,right=.985,top=.77,bottom=.24,wspace=.12)
fig.suptitle('Task 1: auxiliary supervision and transfer success',fontsize=17,x=.025,ha='left',y=.98)
numbers={}
for ax,(label,source) in zip(axes,[('256 adaptation images',high),('32 adaptation images',low)]):
    counts=[int(source[g]['successes']) for g in groups]
    totals=[int(source[g]['n']) for g in groups]
    assert totals == [192]*5
    numbers[label]=dict(zip(groups,counts))
    ax.bar(range(5),np.array(counts)/192*100,color=colors,width=.65,zorder=3)
    for x,n in enumerate(counts): ax.text(x,n/192*100+2,f'{n}/192',ha='center',fontsize=9.5)
    ax.set_title(label,fontsize=13,pad=15)
    ax.set_ylim(0,112);ax.set_yticks([0,25,50,75,100]);ax.set_xticks(range(5),['A\nRGB','B\n+ Box','C\n+ Hand','D\n+ Both','R\nNo video'])
    ax.grid(axis='y',color='#e3e8ec',zorder=0)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Simulation task success (%)')
fig.text(.025,.10,'Each bar: 3 model seeds × 64 shared scenes. All groups use the same fixed grasp controller.',fontsize=11)
fig.text(.025,.045,'The 32-image follow-up reuses test scenes. Small differences do not establish a reliable benefit.',fontsize=11,color='#526875')
fig.savefig(MEDIA/'task-1-results.png',dpi=180,facecolor='white')
plt.close(fig)
receipt={'description':'Plots of frozen Task 1 training arrays and recorded robot success counts; no new labels, training or evaluation.',
 'sample_selection':'First sample in NPZ order with a valid object box and all 21 hand landmarks inside the image.',
 'sample_index':i,'episode_id':str(data['episode_ids'][i]),'frame_index':int(data['frame_indices'][i]),
 'source_video':str(data['source_video'][i]),'dataset_sha256':digest(DATA),
 'object_target_cx_cy_w_h_normalized':box.tolist(),'hand_targets_xy_normalized':points.tolist(),
 'plot_counts':numbers,'result_sources':{p.name:digest(p) for p in [report/'results.json',report/'low32_results.json',report/'imagenet_reference.json']},
 'limits':'Candidate labels are not independently validated ground truth. Example selection illustrates available targets, not detection accuracy.'}
(MEDIA/'task-1-figure-sources.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode('utf-8'))
print(json.dumps({'sample_index':i,'episode':receipt['episode_id'],'frame':receipt['frame_index'],'results':numbers},indent=2))
