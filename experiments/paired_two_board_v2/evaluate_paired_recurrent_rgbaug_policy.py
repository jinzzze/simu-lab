"""Fixed-horizon physical rollout; learned commands only, causal RGB feedback."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil,time
import numpy as np
from paired_sim import goal_conditioned_vector,ROOT,PROJECT
from paired_sim import PairedGoalEnv
from paired_recurrent_goal_policy_process import GoalRelativeRecurrentPolicyProcess as ImagePolicyProcess
from collect_paired_demonstrations import save_video

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def suite_definition(config,name):
    evaluation=config['evaluation']
    if name=='validation':return dict(evaluation['validation'],jitter_m=.003)
    if name in evaluation['suites']:return evaluation['suites'][name]
    base=name.removesuffix('_control')
    if base in evaluation['diagnostics']:
        return dict(evaluation['diagnostics'][base],jitter_m=.003,diagnostic=True,
                    blank_rgb=base=='blank_rgb' and not name.endswith('_control'),
                    swapped_goal_vector=base=='swapped_goal_vector' and not name.endswith('_control'))
    raise ValueError('Unknown suite')


def policy_vectors(env,vectors,swap):
    vectors=np.asarray(vectors,np.float32).copy()
    if swap:
        replacement=f"G{int(env.goal_id[1])%4+1}";goal=np.asarray(env.task['goals'][replacement]['xy_m'])
        vectors[:,4:6]=goal;vectors[:,6:8]=goal-vectors[:,:2]
    return vectors


def run_episode(env,predict,cfg,definition,scene,record):
    pairing=(scene-definition['start'])%4+1;env.task['start_jitter_halfwidth_m']=definition['jitter_m']
    obs=env.reset_task(scene,f'P{pairing}',f'G{pairing}')
    if 'object_rgba' in definition:env.client.changeVisualShape(env.sim._bodies_idx['object'],-1,rgbaColor=definition['object_rgba'])
    if 'table_rgba' in definition:env.client.changeVisualShape(env.sim._bodies_idx['table'],-1,rgbaColor=definition['table_rgba'])
    if 'object_rgba' in definition or 'table_rgba' in definition:obs=env.policy_observation()
    env._reset_score();env._step_count=0;env.trace=[];env._record=record;env.capture_actions=False
    env.frames=[env.render_oblique()] if record else []
    first_sha=hashlib.sha256(obs['rgb'].tobytes()).hexdigest();sampling=cfg['evaluation']['sampling_seed']+scene
    predict.reset(sampling);history=[(0,obs)]
    pre_vectors=[];commands=[];requested=[];requested_delta=[];chunks=[];hist_indices=[];input_vectors=[];input_rgb=[];inference_times=[]
    limit=cfg['evaluation']['episode_steps'];started=time.perf_counter();clip_count=0
    for block in range(0,limit,cfg['execution_horizon']):
        last=history[-1:];images=np.stack([x[1]['rgb'] for x in last]);vectors=policy_vectors(env,[x[1]['vector'] for x in last],definition.get('swapped_goal_vector',False))
        before=time.perf_counter();chunk=predict.predict(images,vectors,blank_rgb=definition.get('blank_rgb',False))
        inference_times.append(time.perf_counter()-before);chunks.append(chunk);hist_indices.append([x[0] for x in last]);input_vectors.append(vectors);input_rgb.append(images)
        count=min(cfg['execution_horizon'],limit-block)
        for offset,command in enumerate(chunk[:count]):
            before_vector=goal_conditioned_vector(env.robot,env.goal_xy,env.previous_command)
            pre_vectors.append(before_vector);requested_delta.append(command.copy());requested.append(command.copy())
            applied=env.step_action(command);commands.append(applied.copy());clip_count+=int(not np.allclose(applied,requested[-1],atol=1e-6))
            # Only the final two consecutive post-step RGBs are needed at the
            # next 4-action prediction. No skipped/future state is substituted.
            if offset>=count-2:history.append((env._step_count,env.policy_observation()))
    outcome=env.outcome()
    record_row=dict(outcome,initial_rgb_sha256=first_sha,sampling_seed=sampling,
                    wall_s=time.perf_counter()-started,inference_calls=len(chunks),
                    inference_mean_s=float(np.mean(inference_times)),limited_commands=clip_count,
                    controller='learned causal recurrent goal-anchored absolute targets + shared IK/limits; no expert stages or object correction',
                    action_representation='goal_relative_xy_absolute_z_width; decoded absolute in predicted_chunks',runtime_object_truth=False,absolute_clock_input=False,feedback_history='current RGB every tick plus causal learned recurrent hidden state',
                    observation_interventions={k:definition.get(k,False) for k in ('blank_rgb','swapped_goal_vector')})
    arrays={'pre_step_vectors':np.asarray(pre_vectors,np.float32),'requested_commands':np.asarray(requested,np.float32),
            'decoded_absolute_actions':np.asarray(requested_delta,np.float32),'effective_commands':np.asarray(commands,np.float32),'predicted_chunks':np.asarray(chunks,np.float32),
            'inference_history_ticks':np.asarray(hist_indices,np.int32),'policy_input_vectors':np.asarray(input_vectors,np.float32),
            'policy_input_rgb':np.asarray(input_rgb,np.uint8)}
    return record_row,arrays


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/policy_recurrent_goal_rgbaug_v6.json');parser.add_argument('--suite',required=True)
    parser.add_argument('--seeds',nargs='+',type=int);parser.add_argument('--conditions',nargs='+');parser.add_argument('--debug-count',type=int)
    args=parser.parse_args();cfg=read(ROOT/args.config);definition=suite_definition(cfg,args.suite)
    names=['evaluate_paired_recurrent_rgbaug_policy.py','paired_recurrent_goal_policy_process.py','paired_recurrent_goal_policy_worker.py','paired_recurrent_goal_policy.py','paired_recurrent_policy.py','paired_delta_policy.py','paired_delta_env.py','paired_image_policy.py','paired_sim.py','goal_scoring.py','collect_paired_demonstrations.py']
    sources={name:ROOT/name for name in names};sources['policy_config.json']=ROOT/args.config
    sources['sim_task_v1.json']=ROOT/'configs/sim_task_v1.json';sources['sim_physics_v1.json']=ROOT/'configs/sim_physics_v1.json';sources['grasp_env.py']=PROJECT/'src/sim/grasp_env.py'
    sources['evaluation_addendum_rgbaug_v6.json']=ROOT/'configs/evaluation_addendum_rgbaug_v6.json'
    hashes={name:sha(path) for name,path in sources.items()}
    kind='debug_prefix' if args.debug_count else 'full_predeclared_development_suite'
    count=args.debug_count or definition['count'];assert 0<count<=definition['count']
    suite_path=args.suite+(f"_debug{count}" if args.debug_count else '')
    seeds=args.seeds or definition.get('seeds',cfg['training']['seeds'])
    for condition in args.conditions or cfg['conditions']:
        assert condition in cfg['conditions']
        for seed in seeds:
            assert seed in cfg['training']['seeds']
            training=ROOT/f"artifacts/runs/{cfg['revision']}/{condition}_seed{seed}";training_meta=read(training/'manifest.json');checkpoint=training/'checkpoint.pt'
            assert training_meta['status']=='completed' and training_meta['config']==cfg
            assert sha(checkpoint)==training_meta['checkpoint_sha256']
            assert training_meta['source_sha256']['paired_image_policy.py']==hashes['paired_image_policy.py']
            assert training_meta['source_sha256']['paired_delta_policy.py']==hashes['paired_delta_policy.py']
            assert training_meta['source_sha256']['paired_recurrent_policy.py']==hashes['paired_recurrent_policy.py']
            assert training_meta['source_sha256']['paired_recurrent_goal_policy.py']==hashes['paired_recurrent_goal_policy.py']
            folder=ROOT/f"artifacts/runs/{cfg['revision']}_evaluation/{suite_path}/{condition}_seed{seed}"
            folder.mkdir(parents=True,exist_ok=True);mp=folder/'manifest.json';episode_path=folder/'episodes.jsonl'
            meta={'status':'running','suite':args.suite,'suite_definition':definition,'kind':kind,'condition':condition,'seed':seed,
                  'config':cfg,'source_sha256':hashes,'checkpoint_sha256':sha(checkpoint),'count_planned':count,
                  'runtime_contract':'current RGB/robot/goal/history only; no clock, object truth, expert stage or scripted grip repair',
                  'started_utc':datetime.now(timezone.utc).isoformat()}
            if mp.exists():
                old=read(mp)
                for key in ('suite','suite_definition','kind','condition','seed','config','source_sha256','checkpoint_sha256','count_planned'):assert old[key]==meta[key],f'Resume mismatch {key}'
                meta['started_utc']=old['started_utc']
            else:
                snap=folder/'source_snapshot';snap.mkdir()
                for name,path in sources.items():shutil.copy2(path,snap/name)
            scenes=list(range(definition['start'],definition['start']+count))
            rows=[json.loads(line) for line in episode_path.read_text(encoding='utf-8').splitlines()] if episode_path.exists() else []
            assert [e['scene_seed'] for e in rows]==scenes[:len(rows)]
            if len(rows)==count and mp.exists() and read(mp)['status']=='completed':continue
            write(mp,meta);env=PairedGoalEnv()
            try:
                with ImagePolicyProcess(checkpoint,folder/'policy_stderr.log') as predict,episode_path.open('a',encoding='utf-8') as log:
                    for scene in scenes[len(rows):]:
                        # Only seed7 first scene is recorded per condition/suite.
                        record=seed==7 and scene==scenes[0]
                        row,arrays=run_episode(env,predict,cfg,definition,scene,record)
                        dataset=folder/f'scene_{scene}_inputs_actions.npz';np.savez_compressed(dataset,**arrays)
                        row.update(condition=condition,policy_seed=seed,dataset_sha256=sha(dataset))
                        trace_path=folder/f'scene_{scene}_scoring.json';write(trace_path,env.trace);row['scoring_sha256']=sha(trace_path)
                        if record:
                            video=folder/'first_scene_demo.mp4';row['video_frames_verified']=save_video(video,env.frames,env.sim.dt);row['video_sha256']=sha(video)
                        log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();rows.append(row)
                        print(json.dumps({'suite':args.suite,'condition':condition,'seed':seed,'scene':scene,'success':row['success'],'lifted':row['lifted'],'inside_goal':row['final_flags']['inside_goal'],'done':len(rows),'n':count}),flush=True)
            except Exception as exc:
                meta.update(status='failed',error=str(exc),episodes=len(rows));write(mp,meta);raise
            finally:env.close()
            for name,path in sources.items():assert sha(path)==hashes[name],'Evaluation source changed'
            meta.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat(),episodes=len(rows),
                        successes=sum(e['success'] for e in rows),lifted=sum(e['lifted'] for e in rows),episodes_sha256=sha(episode_path))
            write(mp,meta);print(json.dumps({'completed':folder.name,'suite':args.suite,'successes':meta['successes'],'episodes':len(rows)}),flush=True)


if __name__=='__main__':main()
