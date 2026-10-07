"""Build a captioned English evidence reel using existing footage at original speed."""
from pathlib import Path
import json,hashlib,textwrap
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reports/submission_v1'
FONT=Path('C:/Windows/Fonts/arial.ttf')
def font(n):return ImageFont.truetype(str(FONT),n)
def card(title,lines):
    im=Image.new('RGB',(1280,720),'#101e2c');d=ImageDraw.Draw(im)
    d.text((60,45),'SIMU LAB / jinzzze',font=font(22),fill='#6ce1cf');d.text((60,105),title,font=font(44),fill='white')
    y=205
    for line in lines:
        for part in textwrap.wrap(line,75):d.text((60,y),part,font=font(28),fill='#e2e9f0');y+=42
        y+=22
    d.text((60,669),'Recorded development evidence / independent real-scene validation deferred',font=font(19),fill='#92a9bb')
    return im
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    segments=[
      ('card','Can hand and object labels help?',7,['12 personal phone videos -> visual transfer -> simulated grasping.','Controlled auxiliary-supervision study, macro-outcome model, and a separate compact Diffusion Policy.']),
      ('video','Personally recorded development data',9,'data/raw/pilot/5d81c4e3767ab92cd6898978b3330c6e.mp4','First canonical clip; 720p / 25 fps. One development scene family.'),
      ('card','A shared experiment, four losses',9,['A: RGB consistency. B: A + object boxes.','C: A + hand landmarks. D: A + both targets.','Same frames, initialization, batches, budgets and grasp controller. Hand labels are training-only pseudo labels.']),
      ('video','Physical grasping in PyBullet',10,'artifacts/reports/visual_ablation_v1/first_scene_comparison.mp4','Seed 7 / first test scene 30000. Contact physics; no object attachment.'),
      ('card','The combined labels did not improve success',12,['256 sim images: A/B/C/D/R each 192/192.','32-image exploratory follow-up: A 184, B 183, C 184, D 181, R 182 / 192.','64 shared scenes x 3 seeds. No significant gain or real-world generalization is claimed.']),
      ('video','Action-conditioned macro-outcome model',10,'artifacts/reports/grasp_world_model_v1/integration/rgb_world_model_demo.mp4','One preselected RGB integration example, not a population success estimate.'),
      ('card','World-model prediction, not planning',9,['160 test transitions share 32 initial scenes.','28.41 mm mean endpoint error; 153/160 outcome labels correct.','Only 71/160 recorded test actions succeeded. The model does not select actions.']),
      ('video','Compact Diffusion Policy: learned commands',13,'artifacts/runs/diffusion_policy_grasp_v1/DP_seed7/first_scene_demo.mp4','Seed 7 / first scene 30000. Additional simulated expert action supervision.'),
      ('video','Failure evidence is retained',13,'artifacts/reports/diffusion_policy_v1/failure_review/DP_seed7_scene30001/failure_demo.mp4','First failed scene for seed 7. Replay of stored commands; no resampling.'),
      ('card','DP-D: 84/192 strict successes',10,['Seeds 7 / 17 / 27: 27/64, 25/64, 32/64.','90 failures without qualified lift; 18 after qualified lift.','Same frozen D7 initial vision; extra action data and different budgets. This is not a matched algorithm ablation.']),
      ('card','Scope and reproducibility',10,['Source, all-seed results, checksums, failure traces and checkpoints are prepared for review.','New independent real recordings are deferred. Human label accuracy and real-robot transfer remain unmeasured.','Repository: github.com/jinzzze/simu-lab / release v1.0.0.'])]
    writer=imageio_ffmpeg.write_frames(str(OUT/'submission_demo_en.mp4'),(1280,720),fps=20,codec='libx264',quality=7,macro_block_size=1,output_params=['-movflags','+faststart'])
    writer.send(None);timeline=[];t=0
    try:
        for idx,s in enumerate(segments):
            kind,title,duration,*rest=s;start=t
            if kind=='card':
                frame=np.asarray(card(title,rest[0]))
                for _ in range(duration*20):writer.send(frame)
            else:
                path=ROOT/rest[0];cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));last=None
                for j in range(duration*20):
                    index=min(int(j*fps/20),n-1);cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read()
                    if ok:last=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                    if last is None:raise RuntimeError(path)
                    im=Image.new('RGB',(1280,720),'#101e2c');d=ImageDraw.Draw(im);d.text((40,22),title,font=font(34),fill='white')
                    pic=Image.fromarray(last);scale=min(1200/pic.width,560/pic.height);pic=pic.resize((round(pic.width*scale),round(pic.height*scale)),Image.Resampling.LANCZOS);im.paste(pic,((1280-pic.width)//2,85+(560-pic.height)//2))
                    caption=rest[1]+(' Final frame held.' if j/20>=n/fps else '')
                    d.text((40,665),caption,font=font(19),fill='#d7e6ed');writer.send(np.asarray(im))
                cap.release()
            t+=duration;timeline.append({'start_seconds':start,'duration_seconds':duration,'kind':kind,'title':title,'source':rest[0] if kind=='video' else None})
            if kind=='card':card(title,rest[0]).save(OUT/f'slide_{idx:02d}.png')
    finally:writer.close()
    cap=cv2.VideoCapture(str(OUT/'submission_demo_en.mp4'));count=0
    while True:
        ok,f=cap.read()
        if not ok:break
        if count in [0,180,780,1400,2100]:cv2.imwrite(str(OUT/f'qa_frame_{count:04d}.jpg'),f)
        count+=1
    cap.release();assert count==t*20
    (OUT/'demo_manifest.json').write_text(json.dumps({'duration_seconds':t,'fps':20,'decoded_frames':count,'all_frames_decoded':True,'audio':'none; English captions','footage_speed':'original; final frame held when segment exceeds source length','selection':'first canonical real clip, preselected first simulation scenes and first seed7 failure; not best-of','sha256':hashlib.sha256((OUT/'submission_demo_en.mp4').read_bytes()).hexdigest(),'segments':timeline},indent=2))
    print(f'Validated {count} frames / {t} seconds')
if __name__=='__main__':main()
