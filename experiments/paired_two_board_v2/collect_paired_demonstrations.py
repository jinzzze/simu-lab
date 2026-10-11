"""Execute paired oracle teacher demos; preserve all attempts and raw provenance."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil
import cv2
import imageio_ffmpeg
import numpy as np
from paired_sim import PairedGoalEnv,video_transport_profile,ROOT,PROJECT


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def save_video(path,frames,dt):
    size=frames[0].shape[0]
    writer=imageio_ffmpeg.write_frames(str(path),(size,size),fps=1/(2*dt),codec='libx264',quality=7,pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1)
    writer.send(None)
    try:
        for frame in frames:writer.send(np.ascontiguousarray(frame))
    finally:writer.close()
    cap=cv2.VideoCapture(str(path));count=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        count+=1
    cap.release()
    if count!=len(frames):raise ValueError('Truncated demonstration MP4')
    return count


def save_episode(out,env,metadata):
    label=metadata['episode_id'];file=out/f'{label}.npz'
    np.savez_compressed(file,rgb=np.asarray(env.images,np.uint8),observations=np.asarray(env.observations,np.float32),
                        actions=np.asarray(env.actions,np.float32),stages=np.asarray(env.stages),
                        requested_commands=np.asarray(env.commands_requested,np.float32))
    write(out/f'{label}.metadata.json',metadata)
    # Post-step truth is separated from policy inputs; explicitly scoring/labels.
    write(out/f'{label}.scoring.json',env.trace)
    return {'episode_id':label,'split':metadata['split'],'condition':metadata['condition'],
            'scene_seed':env.seed,'goal_id':env.goal_id,'start_id':env.start_id,
            'steps':len(env.actions),'success':metadata['outcome']['success'],
            'initial_rgb_sha256':metadata['initial_rgb_sha256'],'file':str(file.relative_to(ROOT)).replace('\\','/'),
            'dataset_sha256':sha(file),'metadata_sha256':sha(out/f'{label}.metadata.json'),
            'scoring_sha256':sha(out/f'{label}.scoring.json')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','collect'],required=True);p.add_argument('--revision',required=True);args=p.parse_args()
    if not args.revision.replace('_','').isalnum():raise ValueError('Invalid output revision')
    out=ROOT/'data/processed'/args.revision;report=ROOT/'artifacts/reports'/args.revision
    if out.exists() or report.exists():raise ValueError('Never overwrite prior data')
    out.mkdir(parents=True);report.mkdir(parents=True);snap=report/'source_snapshot';snap.mkdir()
    sources={'paired_sim.py':ROOT/'paired_sim.py','collect_paired_demonstrations.py':Path(__file__),
             'goal_scoring.py':ROOT/'goal_scoring.py','sim_task_v1.json':ROOT/'configs/sim_task_v1.json',
             'sim_physics_v1.json':ROOT/'configs/sim_physics_v1.json','grasp_env.py':PROJECT/'src/sim/grasp_env.py'}
    for name,path in sources.items():shutil.copy2(path,snap/name)
    task=read(ROOT/'configs/sim_task_v1.json')
    manifest={'revision':args.revision,'mode':args.mode,'status':'collecting','started_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'privileged initial-state robot teacher demonstrations; not learned policy or human video success',
              'code_sha256':{name:sha(path) for name,path in sources.items()},'config':task,
              'retention':'all attempts retained; no outcome filtering','episodes':[],'raw_human_files_modified':False}
    write(report/'manifest.json',manifest)
    specs=[]
    if args.mode=='smoke':
        for i in range(1,5):specs.append({'scene_seed':59000+i,'start_id':f'P{i}','goal_id':f'G{i}','condition':'simulation_only','split':'development_smoke'})
        specs.append({'scene_seed':59005,'start_id':'P1','goal_id':'G1','condition':'wrong_goal_negative','split':'development_smoke','wrong_goal':'G2'})
    else:
        labels=sorted((ROOT/'data/processed/S01_intake_v1').glob('*.json'))
        assert len(labels)==16
        for condition in task['collection']['conditions']:
            for index,path in enumerate(labels):
                intake=read(path)
                specs.append({'scene_seed':task['collection']['train_scene_start']+index,'start_id':intake['intended_start']['start_id'],
                              'goal_id':intake['intended_goal']['goal_id'],'condition':condition,'split':'train',
                              'human_episode_id':intake['episode_id'] if condition=='video_guided' else None})
        # Simulator-only validation, no uncollected human session fabricated.
        for i in range(8):specs.append({'scene_seed':task['collection']['val_scene_start']+i,'start_id':f'P{i%4+1}',
                                     'goal_id':f'G{i%4+1}','condition':'shared_sim_validation','split':'val'})
    env=PairedGoalEnv()
    try:
        for index,spec in enumerate(specs):
            obs=env.reset_task(spec['scene_seed'],spec['start_id'],spec['goal_id'])
            initial_hash=hashlib.sha256(obs['rgb'].tobytes()).hexdigest()
            # Privileged INITIAL state is explicit teacher input, never online
            # object feedback. Final truth cannot repair the teacher's command.
            initial_truth=env.get_object_xy_for_labels()
            profile=video_transport_profile(ROOT,spec['human_episode_id']) if spec.get('human_episode_id') else None
            env.begin_demonstration()
            record=args.mode=='smoke' or (spec['condition']=='video_guided' and spec['scene_seed']<60004)
            wrong=np.array(task['goals'][spec['wrong_goal']]['xy_m']) if spec.get('wrong_goal') else None
            outcome=env.execute_expert(initial_truth,profile=profile,record=record,diagnostic_destination=wrong)
            episode=f'{spec["condition"]}_{spec["scene_seed"]}'
            meta=dict(spec,episode_id=episode,initial_rgb_sha256=initial_hash,initial_truth_xy_m=initial_truth.tolist(),
                      expert_initial_state_source='privileged simulator initial XY for teacher only',outcome=outcome,
                      human_profile=profile,goal_supplied_before_episode=True,
                      runtime_observation_truth=False,actions_recorded_before_physics=True,control_dt_s=env.sim.dt)
            # Failure is retained even if it violates the planning length gate.
            row=save_episode(out,env,meta)
            if record:
                row['video']=str((report/f'{episode}.mp4').relative_to(ROOT)).replace('\\','/')
                row['video_frames_verified']=save_video(report/f'{episode}.mp4',env.frames,env.sim.dt)
                cv2.imwrite(str(report/f'{episode}_initial.jpg'),cv2.cvtColor(obs['rgb'],cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(report/f'{episode}_final.jpg'),cv2.cvtColor(env.render(),cv2.COLOR_RGB2BGR))
            manifest['episodes'].append(row);write(report/'manifest.json',manifest)
            print(json.dumps({'episode':episode,'goal':env.goal_id,'steps':len(env.actions),'success':outcome['success'],
                              'lifted':outcome['lifted'],'inside_goal':outcome['final_flags']['inside_goal'],'stable_s':outcome['final_stable_duration_s']}),flush=True)
            if len(env.actions)>task['collection']['max_episode_steps']:raise ValueError('Exceeded predeclared length; retained partial data, revise explicitly')
    except Exception as exc:
        manifest['status']='failed';manifest['error']=str(exc);write(report/'manifest.json',manifest);raise
    finally:env.close()
    for name,path in sources.items():assert sha(path)==manifest['code_sha256'][name],'Source changed during collection'
    manifest['status']='completed';manifest['completed_utc']=datetime.now(timezone.utc).isoformat()
    manifest['summary']={condition:{'attempts':sum(e['condition']==condition for e in manifest['episodes']),
                                   'successes':sum(e['condition']==condition and e['success'] for e in manifest['episodes'])}
                         for condition in sorted({e['condition'] for e in manifest['episodes']})}
    write(report/'manifest.json',manifest);print(json.dumps(manifest['summary'],indent=2),flush=True)


if __name__=='__main__':main()
