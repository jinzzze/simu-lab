"""Validate executed actions, task inputs, scoring timers and matched data."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);args=p.parse_args()
    report=ROOT/'artifacts/reports'/args.revision;manifest=read(report/'manifest.json')
    assert manifest['status']=='completed'
    for name,digest in manifest['code_sha256'].items():assert sha(report/'source_snapshot'/name)==digest
    data={};total=0
    for e in manifest['episodes']:
        path=ROOT/e['file'];assert sha(path)==e['dataset_sha256']
        meta=read(path.with_suffix('.metadata.json'));trace=read(path.with_suffix('.scoring.json'))
        assert sha(path.with_suffix('.metadata.json'))==e['metadata_sha256']
        assert sha(path.with_suffix('.scoring.json'))==e['scoring_sha256']
        with np.load(path,allow_pickle=False) as file:arrays={k:file[k] for k in file.files}
        assert set(arrays)=={'rgb','observations','actions','stages','requested_commands'}
        n=e['steps'];total+=n
        rgb=arrays['rgb'];obs=arrays['observations'];act=arrays['actions']
        assert rgb.shape==(n,224,224,3) and rgb.dtype==np.uint8
        assert obs.shape==(n,12) and act.shape==(n,4) and len(trace)==n
        assert n<=manifest['config']['collection']['max_episode_steps']
        assert np.isfinite(obs).all() and np.isfinite(act).all()
        goal=np.array(manifest['config']['goals'][e['goal_id']]['xy_m'])
        np.testing.assert_allclose(obs[:,4:6],np.repeat(goal[None],n,axis=0),atol=1e-7)
        np.testing.assert_allclose(obs[:,6:8],goal-obs[:,:2],atol=1e-7)
        np.testing.assert_allclose(obs[1:,8:],act[:-1],atol=1e-7)
        assert np.max(np.linalg.norm(act[:,:3]-obs[:,:3],axis=1))<=.008+1e-6
        limits=manifest['config']['safety_limits'];assert np.min(act[:,3])>=0 and np.max(act[:,3])<=.065+1e-7
        assert np.all(act[:,:3]>=np.array(limits['xyz_low'])-1e-7) and np.all(act[:,:3]<=np.array(limits['xyz_high'])+1e-7)
        assert hashlib.sha256(rgb[0].tobytes()).hexdigest()==e['initial_rgb_sha256']==meta['initial_rgb_sha256']
        assert meta['actions_recorded_before_physics'] and meta['goal_supplied_before_episode'] and not meta['runtime_observation_truth']
        assert 'privileged' in meta['expert_initial_state_source']
        dt=meta['control_dt_s'];stable=0.;lift=0.;max_lift=0.
        for index,t in enumerate(trace,1):
            assert t['step_after_action']==index and abs(t['t_s']-index*dt)<1e-8
            assert 'fully_right' not in t['flags']
            stable=stable+dt if all(t['flags'].values()) else 0.
            assert abs(t['goal_stable_duration_s']-stable)<1e-8
            lift=lift+dt if t['clearance_m']>=.015 and t['bilateral_grasp_contact'] else 0.
            max_lift=max(max_lift,lift)
        outcome=meta['outcome']
        assert abs(outcome['max_lift_duration_s']-max_lift)<1e-8 and abs(outcome['final_stable_duration_s']-stable)<1e-8
        assert outcome['stability_scope']=='designated_goal_containment' and not outcome['object_attached_or_teleported']
        assert outcome['success']==bool(max_lift>=.12 and stable>=.5 and all(trace[-1]['flags'].values()))
        if e['condition']=='video_guided':
            guide=meta['human_profile'];assert guide['human_split']=='train' and guide['human_session']=='S01'
            assert not guide['human_missing_labels_interpolated'] and guide['desired_robot_setpoints_interpolated']
            assert guide['goal_id']==e['goal_id'] and len(guide['valid_source_frame_indices'])>=3
            for relative,digest in guide['human_derivation_file_sha256'].items():assert sha(ROOT/relative)==digest
            assert 'video_guided_transport' in arrays['stages']
            knots=np.array(guide['progress_knots']);assert np.all(np.diff(knots[:,0])>0) and np.all(np.diff(knots[:,1])>=0)
        else:assert meta['human_profile'] is None
        data[e['episode_id']]=(e,meta,arrays)
    if manifest['mode']=='smoke':
        good=[e for e in manifest['episodes'] if e['condition']=='simulation_only']
        assert len(good)==4 and {e['goal_id'] for e in good}=={'G1','G2','G3','G4'} and all(e['success'] for e in good)
        negative=[e for e in manifest['episodes'] if e['condition']=='wrong_goal_negative']
        assert len(negative)==1 and not negative[0]['success']
        _,meta,_=data[negative[0]['episode_id']]
        assert not meta['outcome']['final_flags']['inside_goal']
        assert meta['outcome']['final_object_xyz_m'][0]>.045+.003/2 and meta['outcome']['lifted']
        assert meta['outcome']['final_stable_duration_s']==0
    else:
        assert len(manifest['episodes'])==40
        trains={condition:[e for e in manifest['episodes'] if e['condition']==condition] for condition in ['simulation_only','video_guided']}
        differences=[]
        for condition,es in trains.items():
            assert len(es)==16 and all(e['split']=='train' for e in es)
            for goal in ('G1','G2','G3','G4'):assert sum(e['goal_id']==goal for e in es)==4
        for seed in range(60000,60016):
            a,ma,aa=data[f'simulation_only_{seed}'];b,mb,bb=data[f'video_guided_{seed}']
            assert a['initial_rgb_sha256']==b['initial_rgb_sha256']
            assert a['goal_id']==b['goal_id'] and a['start_id']==b['start_id']
            np.testing.assert_allclose(ma['initial_truth_xy_m'],mb['initial_truth_xy_m'],atol=1e-6)
            different=aa['actions'].shape!=bb['actions'].shape or not np.allclose(aa['actions'],bb['actions'],atol=1e-6)
            differences.append(different)
        assert all(differences),'Video guides must produce distinct robot demonstration actions/timing'
        vals=[e for e in manifest['episodes'] if e['split']=='val']
        assert len(vals)==8 and all(e['condition']=='shared_sim_validation' for e in vals)
        assert not {e['scene_seed'] for e in vals}&{e['scene_seed'] for e in trains['simulation_only']}
    print(json.dumps({'passed':True,'revision':args.revision,'executed_episodes':len(data),'pre_step_action_records':total,
                      'checks':['preprovided constant goal and no clock/object state in policy vector','current RGB retained at every action','effective commands before physics','goal-specific stable and bilateral-lift timers','separate privileged scoring','matched initial scenes and equal demo counts' if manifest['mode']=='collect' else 'four physical targets plus crossed-line wrong-goal negative'],
                      'scope':'expert/oracle data preparation; no learned policy performance'},indent=2))


if __name__=='__main__':main()
