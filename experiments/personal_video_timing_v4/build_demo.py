"""92-second evidence walkthrough; recorded clips, all-seed results and limitations."""
import math
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from common import ROOT,PROJECT,read,write,sha,now

W,H,FPS=1280,720,25
FONT='C:/Windows/Fonts/arial.ttf';BOLD='C:/Windows/Fonts/arialbd.ttf'
FONTS={n:ImageFont.truetype(FONT,n) for n in (20,24,28,32,40,64)}
BFONTS={n:ImageFont.truetype(BOLD,n) for n in (24,28,32,40,64)}
INK='#153341';MUTED='#566c77';TEAL='#007d96'


def text(draw,xy,s,size=28,bold=False,fill=INK):
    draw.multiline_text(xy,s,font=(BFONTS if bold else FONTS)[size],fill=fill,spacing=12)


def canvas(title,sub):
    im=Image.new('RGB',(W,H),'#fbfcfd');d=ImageDraw.Draw(im)
    d.rectangle((0,0,W,10),fill=TEAL);text(d,(48,36),title,40,True);text(d,(48,100),sub,24,fill=MUTED)
    return im,d


def fit(im,box):
    x,y,w,h=box;copy=im.copy();copy.thumbnail((w,h),Image.Resampling.LANCZOS)
    return copy,(x+(w-copy.width)//2,y+(h-copy.height)//2)


def load_video(path):
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        frames.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
    cap.release();assert frames
    return frames,fps


def main():
    out=ROOT/'artifacts/demo_v1';out.mkdir(parents=True,exist_ok=False)
    results=read(ROOT/'artifacts/results_v1/results.json');conditions=results['study']['conditions']
    profiles=read(ROOT/'data/processed/human_profiles_v1.json')['profiles']
    profile=next(p for p in profiles if p['human_episode_id']=='X01_E01')
    raw_key=next(k for k in profile['human_derivation_file_sha256'] if k.endswith('.mp4'))
    raw=PROJECT/raw_key;assert sha(raw)==profile['human_derivation_file_sha256'][raw_key]
    human,human_fps=load_video(raw);robot=[];sources={raw_key:sha(raw)};outcomes=[]
    for c in conditions:
        folder=ROOT/f'artifacts/evaluation/test/{c}_seed7';rows=read(folder/'episodes.json');row=rows[0]
        assert row['case_id']==results['study']['test_cases'][0]['case_id']=='P1_G1_r1'
        video=folder/'first_predeclared_case.mp4';assert sha(video)==row['video_sha256']
        frames,fps=load_video(video);assert len(frames)==row['verified_video_frames']
        robot.append((frames,fps));sources[str(video.relative_to(PROJECT)).replace('\\','/')]=sha(video);outcomes.append(row)
    portable=read(ROOT/'receipts/portable_replay_v1/receipt.json');assert portable['status']=='passed'
    slides={}
    im,d=canvas('From personal video to goal-conditioned control','A controlled simulation study | simu-lab | local review draft')
    for x,color,prefix in [(80,'#c8524c','P'),(770,'#187cbe','G')]:
        d.rectangle((x,180,x+370,525),fill='#eef3f6',outline='#a4b1ba',width=2)
        for n,(dx,dy) in enumerate([(95,95),(270,95),(95,255),(270,255)],1):
            d.rectangle((x+dx-7,180+dy-7,x+dx+7,180+dy+7),fill=color)
            text(d,(x+dx-24,180+dy-56),f'{prefix}{n}',28,True,fill=color)
    text(d,(505,275),'Given goal\n     →',32,True)
    text(d,(90,555),'Grasp → lift → place → release → withdraw',32,True)
    text(d,(90,618),'Strict success: full block footprint inside a 40 mm target square.',24)
    slides[0]=im
    im,d=canvas('Personally recorded training video','First named X01 clip: P1 → G3 | original speed, then held final frame')
    text(d,(850,208),'24 training clips',32,True)
    text(d,(850,270),'S01: 8\nX01: 16\n\n12 P/G combinations\n2 examples per pair',24)
    text(d,(55,638),'X02 is development-only. X03 is reserved for a separate frozen video-label check.',24,fill=MUTED)
    slides[1]=im
    im,d=canvas('What the personal videos contribute','Shared geometry in both groups; transport timing/progress in one group')
    cards=[(55,'24 personal clips','First-frame goal centers\nCoarse transfer intervals\nVisible 2D progress'),
        (465,'Paired robot teachers','24 trajectories per group\nSame initial scenes\nExecuted robot commands'),
        (875,'CNN–GRU policies','3 seeds per group\n2,500 updates per seed\nAll final EMA models')]
    for x,title,body in cards:
        d.rounded_rectangle((x,205,x+350,470),radius=14,fill='#eaf2f5',outline='#bac9d0',width=2)
        text(d,(x+18,235),title,28,True);text(d,(x+18,310),body,24)
    text(d,(55,530),'At runtime: current RGB + robot state + prescribed goal + previous command.',24)
    text(d,(55,590),'Human metric actions are not recovered. Missing human labels remain missing.',24,fill=MUTED)
    slides[2]=im
    im,d=canvas('Fixed first test case: P1 → G1','Seed 7, first predeclared test case | one example, not the aggregate result')
    text(d,(62,167),'Shared geometry',28,True);text(d,(710,167),'+ Personal timing / progress',28,True,fill=TEAL)
    for j,r in enumerate(outcomes):text(d,(62+j*648,635),'Strict success: '+('YES' if r['success'] else 'NO'),28,True,fill=TEAL if r['success'] else '#9a4c39')
    slides[3]=im
    im,d=canvas('All frozen test results','Three seeds retained | 32 tasks per seed | eight shared physical starts')
    text(d,(620,182),'Geometry',28,True);text(d,(950,182),'+ Video',28,True,fill=TEAL)
    for y,key,label in [(270,'same_number','Same-number'),(340,'trained_cross','Seen cross-goal'),(410,'heldout','Held-out combinations'),(510,'all','All tasks')]:
        text(d,(60,y),label,28,True)
        for j,c in enumerate(conditions):
            r=results['totals']['test'][c][key];text(d,(640+j*330,y),f"{r['successes']}/{r['executions']}",32,True,fill=INK if j==0 else TEAL)
    text(d,(60,628),'+4.2 percentage points observed; no statistical-significance claim.',24,fill=MUTED)
    slides[4]=im
    im,d=canvas('Failures and limits remain visible','A small positive observed difference is not robust task mastery')
    text(d,(60,200),'Most failures occur before a qualified lift.',32,True)
    text(d,(60,265),'No qualified lift: 75/96 for geometry, 69/96 for + video.',28)
    text(d,(60,340),'Four initial test images repeat training/development pixels.',28)
    text(d,(60,398),'Post-hoc exclusion: 5/84 versus 8/84. Original results retained.',28)
    text(d,(60,478),'Same table/object family; coarse RGB labels; no real-robot evaluation.',24)
    text(d,(60,552),'Candidate coverage is not localization accuracy or physical success.',24,fill=MUTED)
    slides[5]=im
    im,d=canvas('Reproduction checked with actual executions','Source/configuration snapshots, data lineage and checkpoint SHA-256 hashes')
    text(d,(60,200),'Two fixed development cases replayed successfully as checks.',28,True)
    text(d,(60,270),'All input/action arrays and complete scoring traces matched exactly.',28)
    text(d,(60,350),'Repeated from a separate directory and an existing independent environment.',24)
    text(d,(60,410),'Same Windows host; not a new-machine installation or full retraining.',24,fill=MUTED)
    text(d,(60,500),'Public v1.0.0 contains the older study. This extension awaits a new release.',24)
    text(d,(60,570),'Personal recordings: applicant. Implementation/review assistance: Codex.',24)
    slides[6]=im
    segments=[(0,10,0),(10,22,1),(22,34,2),(34,50,3),(50,64,4),(64,78,5),(78,92,6)]
    output=out/'evidence_walkthrough.mp4';writer=imageio_ffmpeg.write_frames(str(output),(W,H),fps=FPS,
        codec='libx264',quality=7,pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1);writer.send(None)
    preview_times=[2,12,24,34,41,49,53,68,81,90];previews=[]
    try:
        for i in range(92*FPS):
            t=i/FPS;start,end,idx=next(s for s in segments if s[0]<=t<s[1]);im=slides[idx].copy()
            if idx==1:
                frame=human[min(int((t-start)*human_fps),len(human)-1)];tile,pos=fit(frame,(48,180,760,425));im.paste(tile,pos)
            elif idx==3:
                for j,(frames,fps) in enumerate(robot):
                    frame=frames[min(int((t-start)*fps),len(frames)-1)];tile,pos=fit(frame,(40+j*648,215,592,395));im.paste(tile,pos)
            writer.send(np.asarray(im))
            if i in [round(t*FPS) for t in preview_times]:previews.append(cv2.resize(cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR),(640,360)))
    finally:writer.close()
    cap=cv2.VideoCapture(str(output));verified=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape[:2]==(H,W);verified+=1
    cap.release();assert verified==2300
    sheet=cv2.vconcat([cv2.hconcat(previews[i:i+2]) for i in range(0,10,2)]);assert cv2.imwrite(str(out/'review.jpg'),sheet)
    write(out/'manifest.json',dict(created_utc=now(),source_sha256=sources,results_sha256=sha(ROOT/'artifacts/results_v1/results.json'),
        code_sha256=sha(ROOT/'build_demo.py'),video_sha256=sha(output),frames_verified=verified,fps=FPS,duration_s=92,
        robot_case_selection='First predeclared test case, seed7, both conditions; not best-case selection',
        human_clip_selection='First named X01 training clip',publicly_published=False,audio=False,
        segments=[dict(start_s=a,end_s=b,slide=c) for a,b,c in segments]))
    print(dict(video=str(output),verified_frames=verified,duration_s=92))


if __name__=='__main__':main()
