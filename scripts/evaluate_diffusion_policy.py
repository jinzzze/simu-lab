"""Physical evaluation of DP-D, separate from the fixed-controller ablation."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.policy.environment import PolicyGraspEnv,observation,final_outcome
from src.policy.process import PolicyProcess
from evaluate_visual_grasp import save_video


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--limit',type=int,help='Debug-only prefix, distinct output path')
    args=parser.parse_args();cfg=read(ROOT/'configs/diffusion_policy.json');plan=cfg['evaluation']
    source_paths=[Path(__file__),ROOT/'src/policy/environment.py',ROOT/'src/policy/process.py',
                  ROOT/'src/policy/diffusion.py',ROOT/'scripts/diffusion_policy_worker.py',
                  ROOT/'src/sim/grasp_env.py',ROOT/'configs/grasp_sim.json',ROOT/'configs/diffusion_policy.json']
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    base=ROOT/('artifacts/diagnostics/diffusion_policy_smoke' if args.limit else 'artifacts/runs/diffusion_policy_grasp_v1')
    for seed in args.seeds or plan['seeds']:
        assert seed in plan['seeds']
        model_folder=ROOT/f'artifacts/runs/diffusion_policy_v1/DP_seed{seed}'
        manifest=read(model_folder/'manifest.json');checkpoint=model_folder/'checkpoint.pt'
        assert manifest['status']=='completed' and manifest['config']==cfg
        assert sha(checkpoint)==manifest['checkpoint_sha256']
        assert manifest['source_sha256']['src\\policy\\diffusion.py']==sha(ROOT/'src/policy/diffusion.py')
        folder=base/f'DP_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
        episode_path=folder/'episodes.jsonl';metadata_path=folder/'manifest.json'
        metadata={'status':'running','plan':cfg,'policy_seed':seed,
                  'checkpoint_sha256':sha(checkpoint),'visual_checkpoint_sha256':sha(ROOT/cfg['visual_checkpoint']),
                  'source_sha256':source_hashes,'kind':'debug_prefix' if args.limit else 'full_development_evaluation'}
        if metadata_path.exists():
            previous=read(metadata_path)
            for key in ['plan','policy_seed','checkpoint_sha256','visual_checkpoint_sha256','source_sha256','kind']:
                assert previous[key]==metadata[key],f'Resume mismatch: {key}'
        write(metadata_path,metadata)
        count=args.limit or plan['count'];scenes=list(range(plan['start'],plan['start']+count))
        rows=[json.loads(line) for line in episode_path.read_text().splitlines()] if episode_path.exists() else []
        assert [r['scene_seed'] for r in rows]==scenes[:len(rows)]
        env=PolicyGraspEnv(image_size=224)
        try:
            with PolicyProcess(checkpoint,folder/'predictor_stderr.log') as predict, episode_path.open('a',encoding='utf-8') as log:
                for scene in scenes[len(rows):]:
                    started=time.perf_counter();rgb=env.reset(scene)['rgb']
                    xy=predict.reset(rgb,plan['sampling_seed']+scene)
                    env.begin_policy(cfg,record=scene==scenes[0])
                    first=observation(env,xy,env.previous_command,cfg['episode_steps'])
                    history=[first.copy(),first.copy()];executed=[];chunks=[];inference_times=[]
                    for block in range(0,plan['limit_steps'],cfg['execution_horizon']):
                        before=time.perf_counter();chunk=predict.predict(history[-2:])
                        inference_times.append(time.perf_counter()-before);chunks.append(chunk)
                        for command in chunk[:min(cfg['execution_horizon'],plan['limit_steps']-block)]:
                            before_ee=env.robot.get_ee_position().copy()
                            applied=env.step_policy(command)
                            # Store the effective rate-limited setpoint, matching
                            # expert previous-command observations exactly.
                            delta=applied[:3]-before_ee
                            distance=np.linalg.norm(delta)
                            if distance>env.config['max_ee_step_m']:
                                delta*=env.config['max_ee_step_m']/distance
                            applied[:3]=before_ee+delta
                            env.previous_command=applied.copy()
                            executed.append(applied)
                            history.append(observation(env,xy,env.previous_command,cfg['episode_steps']))
                    # Privileged object state appears only in this terminal scorer.
                    outcome=final_outcome(env)
                    record={**outcome,'policy_seed':seed,'initial_visual_xy_m':xy.tolist(),
                            'initial_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                            'sampling_seed':plan['sampling_seed']+scene,'clipped_actions':env.clipped_actions,
                            'inference_calls':len(inference_times),'inference_mean_seconds':float(np.mean(inference_times)),
                            'wall_seconds':time.perf_counter()-started,'controller':'learned DDIM action chunks + shared low-level IK; no scripted stages'}
                    trajectory=folder/f'scene_{scene}_actions.npz'
                    np.savez_compressed(trajectory,observations=np.asarray(history[1:],np.float32),
                                        executed_commands=np.asarray(executed,np.float32),predicted_chunks=np.asarray(chunks,np.float32))
                    record['action_trace_sha256']=sha(trajectory)
                    if scene==scenes[0]:
                        save_video(folder/'first_scene_demo.mp4',env.frames,1/(env.sim.dt*2))
                        write(folder/'first_scene_trace.json',env.trace)
                    log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();rows.append(record)
                    print(f'DP seed={seed} scene={scene} success={record["success"]} completed={len(rows)}/{count}',flush=True)
        finally:env.close()
        for path in source_paths:
            assert sha(path)==source_hashes[str(path.relative_to(ROOT))], 'Evaluation source changed'
        metadata.update({'status':'completed','episodes':len(rows),'successes':sum(r['success'] for r in rows),
                         'lifted':sum(r['lifted'] for r in rows),'episodes_sha256':sha(episode_path)})
        write(metadata_path,metadata)
        print(json.dumps({'seed':seed,'successes':metadata['successes'],'n':len(rows)}),flush=True)


if __name__=='__main__':main()
