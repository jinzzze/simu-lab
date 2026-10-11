"""Recover actual pre-limit teacher requests by verified deterministic replay."""
import json
import numpy as np
from common import ROOT,read,write,sha,now,snapshot,source_paths,check_sources,verify_training_pairs
from paired_sim import PairedGoalEnv,goal_conditioned_vector


def main():
    plan=read(ROOT/'configs/waypoint_pilot_study_v1.json')
    index=read(ROOT/'artifacts/robot_demos_v1/training_index.json')
    assert sha(ROOT/'artifacts/robot_demos_v1/training_index.json')==plan['source_index_sha256']
    entries=index['train_groups']['coverage_24']+index['shared_validation']
    out=ROOT/'artifacts/waypoint_demos_v1';out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'data/processed/waypoint_demos_v1';data.mkdir(parents=True,exist_ok=False)
    hashes=snapshot(out,source_paths()+[ROOT/'collect_waypoint_labels.py',ROOT/'configs/waypoint_pilot_study_v1.json',ROOT/'configs/waypoint_pilot_v1.json'])
    manifest=dict(status='collecting',started_utc=now(),source_sha256=hashes,episodes=[],plan=plan)
    write(out/'manifest.json',manifest);env=PairedGoalEnv()
    try:
        for entry in entries:
            source=ROOT/entry['file'];assert sha(source)==entry['dataset_sha256']
            with np.load(source,allow_pickle=False) as d:
                rgb=d['rgb'].copy();observations=d['observations'].copy();actions=d['actions'].copy()
            trace_path=source.with_suffix('.scoring.json');assert sha(trace_path)==entry['scoring_sha256']
            env.reset_task(entry['scene_seed'],entry['start_id'],entry['goal_id'])
            initial=env.get_object_xy_for_labels();assert np.allclose(initial,entry['initial_truth_xy_m'],atol=1e-9)
            requested=[];actual=[];pre_vectors=[];tick=env._tick
            def capture(target,width,stage):
                request=np.r_[target,width]
                requested.append(request.astype(np.float32))
                actual.append(env.effective_command(request).astype(np.float32))
                pre_vectors.append(goal_conditioned_vector(env.robot,env.goal_xy,env.previous_command))
                tick(target,width,stage)
            env._tick=capture
            try:
                outcome=env.execute_expert(initial,profile=None,record=False)
            finally:
                env._tick=tick
            assert np.array_equal(np.array(actual,np.float32),actions),'Executed commands differ from original'
            assert np.array_equal(np.array(pre_vectors,np.float32),observations),'Input vector histories differ'
            assert env.trace==read(trace_path),'Physics scoring trace differs from original'
            assert outcome['success']==entry['success']
            name=('waypoint_24' if entry['split']=='train' else 'shared_validation')+'_'+str(entry['scene_seed'])
            path=data/(name+'.npz')
            np.savez_compressed(path,rgb=rgb,observations=observations,actions=np.array(requested,np.float32),executed_actions=actions)
            write(data/(name+'.scoring.json'),env.trace)
            row=dict(entry,episode_id=name,condition='waypoint_24' if entry['split']=='train' else 'shared_validation',
                file=str(path.relative_to(ROOT)).replace('\\','/'),dataset_sha256=sha(path),
                target_label_semantics='Teacher pre-limit requested waypoint; not the executed command',
                previous_command_semantics='Actual previous safety-limited executed command, unchanged from primary data',
                source_data_file=entry['file'],source_data_sha256=entry['dataset_sha256'],
                actual_commands_and_vectors_identical=True,full_scoring_trace_identical=True)
            manifest['episodes'].append(row);write(out/'manifest.json',manifest)
            print(json.dumps({'episode':name,'steps':len(requested),'success':row['success'],'verified_replay':True}),flush=True)
        check_sources(hashes)
        train=[r for r in manifest['episodes'] if r['split']=='train'];verify_training_pairs(train)
        manifest.update(status='completed',completed_utc=now(),training_gate_passed=True,
            label_semantics='Pre-limit teacher requests recovered during verified replay. RGB/state histories unchanged; all attempted outcomes retained.',
            summary={'waypoint_24':dict(episodes=len(train),successes=sum(r['success'] for r in train),lifts=sum(r['lifted'] for r in train))})
        write(out/'manifest.json',manifest)
        write(out/'training_index.json',dict(dataset_manifest_sha256=sha(out/'manifest.json'),
            train_groups={'waypoint_24':train},shared_validation=[r for r in manifest['episodes'] if r['split']=='val'],
            allowed_runtime_arrays=['rgb','observations'],training_target_array='actions',
            diagnostic_only_array='executed_actions',target_semantics=manifest['label_semantics']))
    except Exception as exc:
        manifest.update(status='failed',error=str(exc));write(out/'manifest.json',manifest);raise
    finally:
        env.close()


if __name__=='__main__':
    main()
