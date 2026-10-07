"""Replay the first failed recorded DP command stream per seed for diagnosis.

These are fixed recorded commands, not new sampled policy evaluations.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.policy.environment import PolicyGraspEnv,final_outcome
from evaluate_visual_grasp import save_video


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def main():
    cfg=read(ROOT/'configs/diffusion_policy.json')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seeds',nargs='+',type=int)
    args=parser.parse_args()
    out=ROOT/'artifacts/reports/diffusion_policy_v1/failure_review';out.mkdir(parents=True,exist_ok=True)
    selection=[]
    env=PolicyGraspEnv(image_size=224)
    try:
        for seed in args.seeds or cfg['training']['seeds']:
            assert seed in cfg['training']['seeds']
            source=ROOT/f'artifacts/runs/diffusion_policy_grasp_v1/DP_seed{seed}'
            rows=[json.loads(x) for x in (source/'episodes.jsonl').read_text().splitlines()]
            failures=[r for r in rows if not r['success']]
            if not failures:continue
            row=failures[0];scene=row['scene_seed'];folder=out/f'DP_seed{seed}_scene{scene}'
            if (folder/'review.json').exists():
                selection.append(read(folder/'review.json'));continue
            folder.mkdir(parents=True,exist_ok=True)
            rgb=env.reset(scene)['rgb']
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['initial_rgb_sha256']
            path=source/f'scene_{scene}_actions.npz'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['action_trace_sha256']
            with np.load(path,allow_pickle=False) as data:
                commands=data['predicted_chunks'][:,:cfg['execution_horizon']].reshape(-1,4)
            assert len(commands)==cfg['episode_steps']
            env.begin_policy(cfg,record=True)
            for command in commands:env.step_policy(command)
            measured=final_outcome(env)
            matches=measured['success']==row['success'] and measured['lifted']==row['lifted']
            displacement=float(np.linalg.norm(np.asarray(measured['final_object_xyz_m'])-row['final_object_xyz_m']))
            matches=bool(matches and displacement<1e-5)
            save_video(folder/'failure_demo.mp4',env.frames,1/(env.sim.dt*2))
            samples=[cv2.cvtColor(env.frames[i],cv2.COLOR_RGB2BGR) for i in np.linspace(0,len(env.frames)-1,6).round().astype(int)]
            cv2.imwrite(str(folder/'contact_sheet.jpg'),np.vstack([np.hstack(samples[:3]),np.hstack(samples[3:])]))
            write(folder/'trace.json',env.trace)
            entry={'policy_seed':seed,'scene_seed':scene,'selection':'first failure in ascending original scene order',
                   'replayed_original_commands':True,'resampled_diffusion':False,'replay_matches_original':matches,
                   'final_position_difference_m':displacement,'original':row,'replay':measured,
                   'relative_folder':str(folder.relative_to(out)).replace('\\','/'),
                   'scope':'diagnostic replay; not an additional independent evaluation sample'}
            write(folder/'review.json',entry);selection.append(entry)
            if not matches:raise AssertionError('Replay diverged; preserve evidence without describing it as the original run')
    finally:env.close()
    write(out/'summary.json',{'selection':selection,'all_replays_match':all(x['replay_matches_original'] for x in selection)})
    print(json.dumps([{'seed':r['policy_seed'],'scene':r['scene_seed'],'matches':r['replay_matches_original']} for r in selection]),flush=True)


if __name__=='__main__':main()
