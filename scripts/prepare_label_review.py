"""Prepare blinded box review for existing development clips, never a new test split."""
from pathlib import Path
import json,hashlib
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/diagnostics/label_review_release_v1'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    items=[];predictions=[]
    for video in sorted((ROOT/'data/raw/pilot').glob('*.mp4')):
        rows=[json.loads(s) for s in (ROOT/f'data/processed/pilot_labels_v2/{video.stem}_labels.jsonl').read_text().splitlines()]
        rng=np.random.default_rng(int(hashlib.sha256(video.name.encode()).hexdigest()[:8],16))
        cap=cv2.VideoCapture(str(video))
        for quarter,segment in enumerate(np.array_split(np.arange(len(rows)),4)):
            index=int(rng.choice(segment));row=rows[index];sample=f's{len(items)+1:03d}'
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
            if not ok:raise RuntimeError(f'Cannot decode {video}:{index}')
            cv2.imwrite(str(OUT/f'{sample}.png'),frame)
            items.append({'id':sample,'image':f'{sample}.png','source_video':str(video.relative_to(ROOT)).replace('\\','/'),'source_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'frame_index':index,'quarter':quarter,'width':frame.shape[1],'height':frame.shape[0]})
            predictions.append({'id':sample,'object':row['object'],'hand':row['hand']})
        cap.release()
    (OUT/'sample_manifest.json').write_text(json.dumps({'scope':'Existing development data; not independent real validation','selection':'one SHA-seeded random frame per temporal quarter of every video; independent of predicted quality','items':items},indent=2))
    (OUT/'predictions_for_scoring.json').write_text(json.dumps(predictions,indent=2))
    page='''<!doctype html><html><meta charset="utf-8"><title>Blinded object annotation</title><style>body{font:16px system-ui;max-width:1280px;margin:25px auto;background:#eef3f3}canvas{width:100%;touch-action:none;background:white}button,select,input{padding:8px;margin:5px}p{line-height:1.5}</style><h1>Development label review — 48 frames</h1><p>Annotate the visible black block, excluding shadow and hidden parts. Drag a tight box. For an invisible or unjudgeable block select the appropriate status. Predictions are hidden. This is not an independent scene evaluation. Hand keypoint accuracy is not scored here.</p><label>Reviewer <input id="reviewer" placeholder="Your name"></label><select id="visibility"><option value="">Choose visibility</option><option>visible</option><option>partial</option><option>absent</option><option>unjudgeable</option></select><button id="clear">Clear box</button><button id="prev">Previous</button><button id="next">Save / next</button><button id="download">Download annotations</button><p id="status"></p><canvas id="canvas" width="1280" height="720"></canvas><script>
const items=ITEMS;let i=0,start=null,box=null;const answers={};const c=document.getElementById('canvas'),ctx=c.getContext('2d'),im=new Image(),vis=document.getElementById('visibility');
function draw(){ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(im,0,0);if(box){ctx.strokeStyle='#f03040';ctx.lineWidth=3;ctx.strokeRect(...box)}}
function show(){let s=items[i];c.width=s.width;c.height=s.height;box=answers[s.id]?.bbox_xywh_px||null;vis.value=answers[s.id]?.visibility||'';im.onload=draw;im.src=s.image;document.getElementById('status').textContent=`${i+1}/${items.length} — ${s.id}; ${Object.keys(answers).length} saved`}
function point(e){let r=c.getBoundingClientRect();return [(e.clientX-r.left)*c.width/r.width,(e.clientY-r.top)*c.height/r.height]}
c.onpointerdown=e=>{start=point(e);c.setPointerCapture(e.pointerId)};c.onpointerup=e=>{if(!start)return;let p=point(e);box=[Math.min(p[0],start[0]),Math.min(p[1],start[1]),Math.abs(p[0]-start[0]),Math.abs(p[1]-start[1])].map(Math.round);start=null;draw()};
function save(){if(!vis.value)return false;if(['visible','partial'].includes(vis.value)&&(!box||box[2]<1||box[3]<1)){alert('Draw a nonempty visible-component box.');return false}answers[items[i].id]={visibility:vis.value,bbox_xywh_px:['visible','partial'].includes(vis.value)?box:null};return true}
document.getElementById('next').onclick=()=>{if(save()){i=Math.min(i+1,items.length-1);show()}};document.getElementById('prev').onclick=()=>{save();i=Math.max(i-1,0);show()};document.getElementById('clear').onclick=()=>{box=null;draw()};document.getElementById('download').onclick=()=>{save();let reviewer=document.getElementById('reviewer').value.trim();if(!reviewer){alert('Enter reviewer name.');return}let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({reviewer,scope:'development_only',answers},null,2)],{type:'application/json'}));a.download='object_annotations.json';a.click();URL.revokeObjectURL(a.href)};show();</script></html>'''
    (OUT/'index.html').write_text(page.replace('ITEMS',json.dumps(items)),encoding='utf-8')
    print(f'Prepared {len(items)} blinded frames; annotations remain pending')
if __name__=='__main__':main()
