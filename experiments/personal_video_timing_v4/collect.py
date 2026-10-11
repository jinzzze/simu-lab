"""Paired geometry-only / personal timing teachers; keep every attempted rollout."""
import hashlib
import json
import numpy as np
from common import ROOT,PROJECT,read,write,sha,now,snapshot,source_paths,check_sources,verify_training_pairs,allowed_pair
from timing_env import VideoTimingEnv


def main():
    plan=read(ROOT/'configs/study_v1.json');profiles=read(ROOT/'data/processed/human_profiles_v1.json')
    assert sha(ROOT/'data/processed/human_profiles_v1.json')==plan['human_profiles_sha256']
    check_sources(profiles['source_sha256'])
    out=ROOT/'artifacts/robot_demos_v1';out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'data/processed/robot_demos_v1';data.mkdir(parents=True,exist_ok=False)
    hashes=snapshot(out,source_paths()+[ROOT/'collect.py',ROOT/'configs/study_v1.json',ROOT/'configs/policy_v1.json',
        ROOT/'data/processed/human_profiles_v1.json'])
    meta=dict(status='collecting',started_utc=now(),plan=plan,source_sha256=hashes,episodes=[])
    write(out/'manifest.json',meta);specs=[]
    for condition in plan['conditions']:
        for case in plan['train_cases']:specs.append(dict(case,condition=condition,split='train'))
    for case in plan['validation_cases']:specs.append(dict(case,condition='shared_validation',split='val'))
    assert all(allowed_pair(r['start_id'],r['goal_id']) for r in specs)
    mapping={r['human_episode_id']:r for r in profiles['profiles']}
    env=VideoTimingEnv()
    try:
        for spec in specs:
            obs=env.reset_task(spec['scene_seed'],spec['start_id'],spec['goal_id'])
            initial=env.get_object_xy_for_labels().copy();env.begin_demonstration()
            guided=spec['condition']=='video_timing_24'
            profile=mapping[spec['human_episode_id']] if guided else None
            if profile is not None:
                assert profile['human_split']=='train' and profile['goal_id']==spec['goal_id']
                assert profile['start_id']==spec['start_id']
            outcome=env.execute_expert(initial,profile=profile,record=False)
            eid=f"{spec['condition']}_{spec['scene_seed']}";path=data/(eid+'.npz')
            np.savez_compressed(path,rgb=np.array(env.images,np.uint8),observations=np.array(env.observations,np.float32),actions=np.array(env.actions,np.float32))
            trace=data/(eid+'.scoring.json');write(trace,env.trace)
            record=dict(spec,episode_id=eid,file=str(path.relative_to(ROOT)).replace('\\','/'),dataset_sha256=sha(path),
                scoring_sha256=sha(trace),steps=len(env.actions),success=outcome['success'],lifted=outcome['lifted'],outcome=outcome,
                initial_truth_xy_m=initial.tolist(),initial_rgb_sha256=hashlib.sha256(obs['rgb'].tobytes()).hexdigest(),
                teacher_truth_scope='initial XY for training teacher only; no online correction',human_timing_used=guided,
                human_profile=profile,human_geometry_used=True)
            write(data/(eid+'.metadata.json'),record);meta['episodes'].append(record);write(out/'manifest.json',meta)
            print(dict(episode=eid,pair=spec['start_id']+'_'+spec['goal_id'],success=outcome['success'],steps=len(env.actions)),flush=True)
        check_sources(hashes);check_sources(profiles['source_sha256'])
        groups={c:[r for r in meta['episodes'] if r['condition']==c] for c in plan['conditions']}
        for entries in groups.values():verify_training_pairs(entries)
        for a,b in zip(groups['geometry_only_24'],groups['video_timing_24']):
            assert a['scene_seed']==b['scene_seed'] and a['start_id']==b['start_id'] and a['goal_id']==b['goal_id']
            assert a['initial_rgb_sha256']==b['initial_rgb_sha256']
            assert np.allclose(a['initial_truth_xy_m'],b['initial_truth_xy_m'],atol=1e-9)
        summary={c:dict(episodes=len(v),successes=sum(r['success'] for r in v),lifts=sum(r['lifted'] for r in v),frames=sum(r['steps'] for r in v)) for c,v in groups.items()}
        gate=all(a['episodes']==24 and a['successes']>=20 and a['lifts']==24 for a in summary.values())
        meta.update(status='completed',completed_utc=now(),summary=summary,training_gate_passed=gate)
        write(out/'manifest.json',meta)
        write(out/'training_index.json',dict(dataset_manifest_sha256=sha(out/'manifest.json'),train_groups=groups,
            shared_validation=[r for r in meta['episodes'] if r['split']=='val'],all_attempts_retained=True))
        print(dict(summary=summary,training_gate_passed=gate),flush=True)
    except Exception as exc:
        meta.update(status='failed',error=str(exc));write(out/'manifest.json',meta);raise
    finally:env.close()


if __name__=='__main__':main()
