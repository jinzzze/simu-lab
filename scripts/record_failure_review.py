"""Replay the first failure per low-data group; never select a flattering example."""
from pathlib import Path
import hashlib
import json
import sys
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.sim import GraspEnv
from src.predictor_process import PredictorProcess
from evaluate_visual_grasp import save_video


def main():
    report = ROOT/'artifacts/reports/visual_ablation_v1'
    records = [json.loads(x) for x in (report/'low32_episodes.jsonl').read_text().splitlines()]
    output = report/'failure_review'; output.mkdir(exist_ok=True)
    chosen=[]; env=GraspEnv(image_size=224)
    try:
        for group in 'ABCDR':
            failed=sorted((r for r in records if r['group']==group and not r['success']),key=lambda r:(r['training_seed'],r['scene_seed']))
            if not failed:continue
            row=failed[0];seed=row['training_seed'];scene=row['scene_seed']
            folder=output/f'{group}_seed{seed}_scene{scene}';folder.mkdir(exist_ok=True)
            checkpoint=ROOT/'artifacts/runs/visual_low32_state_v1'/f'{group}_seed{seed}'/'checkpoint.pt'
            obs=env.reset(scene);rgb=obs['rgb']
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['initial_rgb_sha256']
            with PredictorProcess(checkpoint,folder/'predictor_stderr.log') as predict:
                prediction=predict(rgb)
            assert np.allclose(prediction,row['prediction_xy_m'],atol=1e-6,rtol=0)
            # Projection is a figure diagnostic only; never supplied to control.
            view=np.asarray(env.view_matrix).reshape(4,4,order='F')
            projection=np.asarray(env.projection_matrix).reshape(4,4,order='F')
            clip=projection@view@np.r_[prediction,.005,1.]
            uv=np.array([(clip[0]/clip[3]+1)/2,(1-clip[1]/clip[3])/2])
            image=cv2.cvtColor(cv2.resize(rgb,(672,672),interpolation=cv2.INTER_NEAREST),cv2.COLOR_RGB2BGR)
            point=tuple(np.rint(uv*672).astype(int))
            cv2.drawMarker(image,point,(45,45,220),cv2.MARKER_CROSS,24,2)
            cv2.putText(image,f'{group} | seed {seed}, scene {scene} | error {row["initial_xy_error_mm"]:.1f} mm',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1,cv2.LINE_AA)
            cv2.putText(image,'Red cross: RGB-predicted grasp XY',(10,650),cv2.FONT_HERSHEY_SIMPLEX,.6,(30,30,190),1,cv2.LINE_AA)
            cv2.imwrite(str(folder/'initial_prediction.png'),image)
            outcome=env.execute_grasp(prediction,record=True)
            assert outcome['success']==row['success'], 'Replay changed the discrete task result'
            save_video(folder/'failure_demo.mp4',env.frames,1/(env.sim.dt*2))
            cap=cv2.VideoCapture(str(folder/'failure_demo.mp4'));decoded=0
            while True:
                ok,_=cap.read()
                if not ok:break
                decoded+=1
            cap.release();assert decoded==len(env.frames)
            thumb_indices=np.linspace(0,len(env.frames)-1,6).round().astype(int)
            thumbs=[cv2.cvtColor(env.frames[i],cv2.COLOR_RGB2BGR) for i in thumb_indices]
            cv2.imwrite(str(folder/'contact_sheet.jpg'),np.vstack([np.hstack(thumbs[:3]),np.hstack(thumbs[3:])]))
            review={'original':row,'replayed_prediction_xy_m':prediction.tolist(),'replayed_outcome':outcome,
                    'frames_verified':decoded,'selection':'first failed scene by training seed then scene seed; selected after evaluation for diagnosis',
                    'not_additional_independent_test_episode':True,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            (folder/'review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
            (folder/'trace.json').write_text(json.dumps(env.trace,indent=2),encoding='utf-8')
            chosen.append({'group':group,'seed':seed,'scene':scene,'path':str(folder.relative_to(report)),
                           'error_mm':row['initial_xy_error_mm'],'replay_matches_success':True})
            print(f'Recorded first failure: {group}, seed {seed}, scene {scene}',flush=True)
    finally:env.close()
    summary={'selection':chosen,'total_low_data_failures':sum(not r['success'] for r in records),
             'failed_without_qualified_lift':sum(not r['outcome'].get('lifted',False) for r in records if not r['success']),
             'scope':'Post-evaluation diagnosis; replays not counted as extra independent evidence'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':main()
