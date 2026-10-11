"""Verify source isolation, exact paired pre-transport behavior and causal labels."""
import json
from pathlib import Path
import numpy as np
from common import ROOT,read,write,sha,check_sources,now,verify_training_pairs


def main():
    out=ROOT/'artifacts/demonstration_audit_v1';out.mkdir(parents=True,exist_ok=False)
    path=ROOT/'artifacts/robot_demos_v1/manifest.json';m=read(path)
    assert m['status']=='completed';check_sources(m['source_sha256'])
    bundle=read(ROOT/'data/processed/human_profiles_v1.json');check_sources(bundle['source_sha256'])
    index=read(ROOT/'artifacts/robot_demos_v1/training_index.json')
    assert index['dataset_manifest_sha256']==sha(path)
    cache={};stats=[]
    for e in m['episodes']:
        file=ROOT/e['file'];trace_path=file.with_suffix('.scoring.json');trace=read(trace_path)
        assert sha(file)==e['dataset_sha256'] and sha(trace_path)==e['scoring_sha256']
        with np.load(file,allow_pickle=False) as data:
            arrays={k:data[k] for k in data.files}
        assert set(arrays)=={'rgb','observations','actions'}
        assert len(arrays['rgb'])==len(trace)==e['steps']
        assert np.array_equal(arrays['observations'][1:,8:12],arrays['actions'][:-1])
        assert np.allclose(arrays['observations'][:,4:6],e['outcome']['goal_xy_m'])
        prefix=next(i for i,t in enumerate(trace) if 'transport' in t['stage'])
        cache[e['episode_id']]=(e,arrays,prefix)
        if e['human_timing_used']:
            assert e['human_profile']['human_split']=='train'
            assert e['human_profile']['human_session'] in ('S01','X01')
            assert not e['human_profile']['runtime_observation']
        else:assert e['human_profile'] is None
    for a,b in zip(index['train_groups']['geometry_only_24'],index['train_groups']['video_timing_24']):
        ea,aa,na=cache[a['episode_id']];eb,ab,nb=cache[b['episode_id']]
        assert na==nb and ea['initial_rgb_sha256']==eb['initial_rgb_sha256']
        assert (ea['scene_seed'],ea['start_id'],ea['goal_id'])==(eb['scene_seed'],eb['start_id'],eb['goal_id'])
        for key in aa:assert np.array_equal(aa[key][:na],ab[key][:nb]),f'Pre-transport mismatch {a["episode_id"]} {key}'
        changed=aa['actions'].shape!=ab['actions'].shape or not np.array_equal(aa['actions'],ab['actions'])
        assert changed,'Video intervention had no effect on executed command labels'
        stats.append({'scene_seed':ea['scene_seed'],'personal_clip':eb['human_episode_id'],'pre_transport_steps_identical':na,
            'baseline_steps':ea['steps'],'video_steps':eb['steps'],'actions_changed':changed})
    for rows in index['train_groups'].values():verify_training_pairs(rows)
    result={'status':'passed','created_utc':now(),'teacher_quality_gate_passed':m['training_gate_passed'],
        'dataset_manifest_sha256':sha(path),'source_sha256':bundle['source_sha256'],'pairs':stats,
        'training_clips':24,'training_conditions':2,'same_geometry_and_pre_transport_behavior':True,
        'causal_previous_command':True,'all_attempts_retained':True,'scope':'Teacher demonstrations, not learned-policy success'}
    write(out/'receipt.json',result)
    print({k:v for k,v in result.items() if k not in ('source_sha256','pairs')},flush=True)


if __name__=='__main__':main()
