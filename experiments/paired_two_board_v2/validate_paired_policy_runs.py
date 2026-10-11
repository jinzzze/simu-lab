"""Validate paired training and actually executed causal policy rollouts."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--suite',default='validation')
    parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    cfg=read(ROOT/args.config);revision=cfg['revision'];runs=ROOT/f'artifacts/runs/{revision}'
    training={};checked_rollouts=0
    for condition in cfg['conditions']:
        for seed in cfg['training']['seeds']:
            folder=runs/f'{condition}_seed{seed}'
            if not (folder/'manifest.json').exists():
                assert not args.require_complete;continue
            meta=read(folder/'manifest.json')
            if meta['status']!='completed':assert not args.require_complete;continue
            assert meta['config']==cfg and meta['steps_completed']==cfg['training']['steps']
            assert sha(folder/'checkpoint.pt')==meta['checkpoint_sha256']
            for name,digest in meta['source_sha256'].items():assert sha(folder/'source_snapshot'/name)==digest
            assert meta['no_expert_stage_or_object_truth_input']
            training[(condition,seed)]=meta
    for seed in cfg['training']['seeds']:
        a=training.get(('simulation_only',seed));b=training.get(('video_guided',seed))
        if a and b:
            assert a['initial_weights_sha256']==b['initial_weights_sha256']
            assert a['episode_fraction_stream_sha256']==b['episode_fraction_stream_sha256']
            assert a['dataset_manifest_sha256']==b['dataset_manifest_sha256']
    diagnostic=cfg['evaluation']['diagnostics'].get(args.suite.removesuffix('_control'),{})
    evaluation_seeds=diagnostic.get('seeds',cfg['training']['seeds'])
    evaluation=ROOT/f'artifacts/runs/{revision}_evaluation/{args.suite}'
    matched={};complete_models=0;summary={}
    for condition in cfg['conditions']:
        for seed in evaluation_seeds:
            folder=evaluation/f'{condition}_seed{seed}'
            if not (folder/'manifest.json').exists():assert not args.require_complete;continue
            meta=read(folder/'manifest.json');rows=[json.loads(x) for x in (folder/'episodes.jsonl').read_text(encoding='utf-8').splitlines()] if (folder/'episodes.jsonl').exists() else []
            assert meta['config']==cfg and meta['condition']==condition and meta['seed']==seed
            if args.require_complete:assert meta['status']=='completed' and len(rows)==meta['count_planned']
            if meta['status']=='completed':
                complete_models+=1;assert sha(folder/'episodes.jsonl')==meta['episodes_sha256']
            assert meta['checkpoint_sha256']==training[(condition,seed)]['checkpoint_sha256']
            for name,digest in meta['source_sha256'].items():assert sha(folder/'source_snapshot'/name)==digest
            for row in rows:
                sid=row['scene_seed'];path=folder/f'scene_{sid}_inputs_actions.npz';trace_path=folder/f'scene_{sid}_scoring.json'
                assert sha(path)==row['dataset_sha256'] and sha(trace_path)==row['scoring_sha256']
                with np.load(path,allow_pickle=False) as file:data={key:file[key] for key in file.files}
                trace=read(trace_path);n=cfg['evaluation']['episode_steps'];h=cfg['execution_horizon'];calls=n//h
                assert row['steps']==len(trace)==n and row['inference_calls']==calls
                vector=data['pre_step_vectors'];action=data['effective_commands'];chunks=data['predicted_chunks']
                oh=cfg['observation_horizon'];ph=cfg['prediction_horizon']
                assert vector.shape==(n,12) and action.shape==(n,4) and chunks.shape==(calls,ph,4)
                assert data['policy_input_rgb'].shape==(calls,oh,224,224,3) and data['policy_input_rgb'].dtype==np.uint8
                assert hashlib.sha256(data['policy_input_rgb'][0,0].tobytes()).hexdigest()==row['initial_rgb_sha256']
                ticks=np.array([[max(0,i*h-j) for j in reversed(range(oh))] for i in range(calls)])
                assert np.array_equal(data['inference_history_ticks'],ticks)
                assert np.all(ticks<=np.arange(calls)[:,None]*h)
                expected=vector[ticks].copy()
                if row['observation_interventions']['swapped_goal_vector']:
                    goal=f"G{int(row['goal_id'][1])%4+1}";new=np.array(read(ROOT/'configs/sim_task_v1.json')['goals'][goal]['xy_m'])
                    expected[:,:,4:6]=new;expected[:,:,6:8]=new-expected[:,:,:2]
                np.testing.assert_allclose(data['policy_input_vectors'],expected,atol=1e-7)
                np.testing.assert_allclose(vector[1:,8:],action[:-1],atol=1e-7)
                goal=np.array(row['goal_xy_m']);np.testing.assert_allclose(vector[:,4:6],np.broadcast_to(goal,(n,2)),atol=1e-7)
                np.testing.assert_allclose(vector[:,6:8],goal-vector[:,:2],atol=1e-7)
                if cfg.get('action_representation')=='ee_delta_xyz_absolute_width':
                    delta=data['predicted_delta_actions'];np.testing.assert_allclose(delta,chunks[:,:h].reshape(n,4),atol=1e-7)
                    np.testing.assert_allclose(data['requested_commands'][:,:3],vector[:,:3]+delta[:,:3],atol=1e-7)
                    np.testing.assert_allclose(data['requested_commands'][:,3],delta[:,3],atol=1e-7)
                else:np.testing.assert_allclose(data['requested_commands'],chunks[:,:h].reshape(n,4),atol=1e-7)
                assert np.max(np.linalg.norm(action[:,:3]-vector[:,:3],axis=1))<=.008+1e-6
                limits=read(ROOT/'configs/sim_task_v1.json')['safety_limits']
                expected_command=np.clip(data['requested_commands'],limits['xyz_low']+[limits['width_low']],limits['xyz_high']+[limits['width_high']])
                delta=expected_command[:,:3]-vector[:,:3];length=np.linalg.norm(delta,axis=1,keepdims=True)
                expected_command[:,:3]=vector[:,:3]+delta*np.minimum(1.,limits['max_ee_step_m']/np.maximum(length,1e-12))
                np.testing.assert_allclose(action,expected_command,atol=2e-7,rtol=0)
                assert not row['runtime_object_truth'] and not row['absolute_clock_input'] and not row['object_attached_or_teleported']
                stable=0.;lift=0.;max_lift=0.;dt=.04
                for tick,t in enumerate(trace,1):
                    assert t['step_after_action']==tick and t['stage']=='learned_action'
                    assert 'fully_right' not in t['flags'] and abs(t['t_s']-tick*dt)<1e-7
                    stable=stable+dt if all(t['flags'].values()) else 0.
                    assert abs(t['goal_stable_duration_s']-stable)<1e-7
                    lift=lift+dt if t['clearance_m']>=.015 and t['bilateral_grasp_contact'] else 0.;max_lift=max(max_lift,lift)
                assert abs(max_lift-row['max_lift_duration_s'])<1e-7 and abs(stable-row['final_stable_duration_s'])<1e-7
                assert row['success']==bool(max_lift+1e-9>=.12 and stable+1e-9>=.5 and all(trace[-1]['flags'].values()))
                matched[(condition,seed,sid)]=row;checked_rollouts+=1
            summary[f'{condition}_seed{seed}']={'status':meta['status'],'episodes':len(rows),'successes':sum(r['success'] for r in rows),'lifted':sum(r['lifted'] for r in rows)}
    for (condition,seed,sid),row in matched.items():
        other=matched.get(('video_guided' if condition=='simulation_only' else 'simulation_only',seed,sid))
        if other:assert row['initial_rgb_sha256']==other['initial_rgb_sha256']
    transitive={}
    for name,relative in [('diffusion.py','src/policy/diffusion.py'),('training.py','src/perception/training.py')]:
        path=ROOT.parents[1]/relative;digest=sha(path)
        for meta in training.values():assert meta['source_sha256'][name]==digest,'Transitive helper differs from training snapshot'
        transitive[relative]=digest
    if args.require_complete:assert len(training)==6 and complete_models==len(cfg['conditions'])*len(evaluation_seeds)
    result={'passed':True,'revision':revision,'suite':args.suite,'complete_training_runs':len(training),'executed_rollouts_checked':checked_rollouts,
            'summary':summary,'validation_utc':datetime.now(timezone.utc).isoformat(),'evaluation_seeds':evaluation_seeds,
            'additional_transitive_helper_sha256':transitive,
            'checks':['paired initial weights and episode/fraction stream','checkpoint/source snapshots','actual consecutive causal RGB and vectors','absolute effective commands after optional delta conversion','shared rate limits','no clock/stage/object state in policy input','actual strict lift/goal stable scoring','matched initial images'],
            'scope':'learned simulation policy rollouts, separate from oracle teacher demos'}
    if args.require_complete:
        report=ROOT/f'artifacts/reports/{revision}';report.mkdir(parents=True,exist_ok=True)
        out=report/f'{args.suite}_validation.json';assert not out.exists(),'Preserve prior validation receipt'
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
