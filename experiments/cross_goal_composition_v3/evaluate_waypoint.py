"""Closed-loop fixed checkpoint evaluation; goal agreement and input provenance audited."""
import argparse
import json
import numpy as np
from common import ROOT,OLD,PROJECT,read,write,sha,now,snapshot,source_paths,check_sources
from paired_sim import PairedGoalEnv
from paired_recurrent_goal_policy_process import GoalRelativeRecurrentPolicyProcess
from paired_recombination_driver import run_episode
from collect_paired_demonstrations import save_video


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--condition',required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--suite',choices=['validation'],required=True);args=parser.parse_args()
    cfg=read(ROOT/'configs/waypoint_pilot_v1.json');study=read(ROOT/'configs/waypoint_pilot_study_v1.json')
    assert sha(ROOT/'configs/waypoint_pilot_study_v1.json')==cfg['study_sha256']
    assert args.condition in cfg['conditions'] and args.seed in cfg['training']['seeds']
    label=f'{args.condition}_seed{args.seed}';train=ROOT/'artifacts/models'/label;meta=read(train/'manifest.json')
    assert meta['status']=='completed' and meta['config']==cfg
    check_sources(meta['source_sha256']);checkpoint=train/'checkpoint.pt';assert sha(checkpoint)==meta['checkpoint_sha256']
    if args.suite=='test':
        for c in cfg['conditions']:
            for s in cfg['training']['seeds']:
                assert read(ROOT/f'artifacts/waypoint_pilot_evaluation/validation/{c}_seed{s}/manifest.json')['status']=='completed'
    cases=study[args.suite+'_cases'];out=ROOT/f'artifacts/waypoint_pilot_evaluation/{args.suite}/{label}'
    if out.exists():
        m=read(out/'manifest.json')
        assert m['status']=='completed' and m['checkpoint_sha256']==sha(checkpoint) and m['study_sha256']==sha(ROOT/'configs/waypoint_pilot_study_v1.json')
        check_sources(m['source_sha256']);return
    out.mkdir(parents=True)
    paths=source_paths()+[ROOT/'evaluate_waypoint.py',ROOT/'configs/waypoint_pilot_study_v1.json',ROOT/'configs/waypoint_pilot_v1.json']
    paths += [OLD/n for n in ('paired_recombination_driver.py','paired_recurrent_goal_policy_process.py','paired_recurrent_goal_policy_worker.py',
                'paired_recurrent_goal_policy.py','paired_recurrent_policy.py','paired_delta_policy.py','paired_image_policy.py','collect_paired_demonstrations.py')]
    paths += [PROJECT/'src/perception/training.py',PROJECT/'src/policy/diffusion.py']
    hashes=snapshot(out,paths)
    receipt=dict(status='running',suite=args.suite,condition=args.condition,seed=args.seed,started_utc=now(),
        source_sha256=hashes,checkpoint_sha256=sha(checkpoint),study_sha256=sha(ROOT/'configs/waypoint_pilot_study_v1.json'),
        planned_cases=len(cases),case_ids=[x['case_id'] for x in cases],
        scope='Current RGB/robot/goal/past-command only;384 fixed steps; no phase, clock or object correction')
    write(out/'manifest.json',receipt);env=PairedGoalEnv();rows=[]
    try:
        with GoalRelativeRecurrentPolicyProcess(checkpoint,out/'policy_stderr.log') as predict:
            for i,case in enumerate(cases):
                definition=dict(start=case['scene_seed'],jitter_m=study['jitter_m'],
                                prescribed_start_id=case['start_id'],prescribed_goal_id=case['goal_id'])
                record=args.suite=='test' and args.seed==7 and i==0
                row,arrays=run_episode(env,predict,cfg,definition,case['scene_seed'],record)
                assert np.allclose(arrays['policy_input_vectors'][:,:,4:6],row['goal_xy_m'])
                assert arrays['pre_step_vectors'].shape==(384,12)
                assert np.array_equal(arrays['inference_history_ticks'].ravel(),np.arange(384))
                assert row['start_id']==case['start_id'] and row['goal_id']==case['goal_id']
                assert not row['runtime_object_truth'] and not any(row['observation_interventions'].values())
                path=out/(case['case_id']+'_inputs_actions.npz');np.savez_compressed(path,**arrays)
                trace=out/(case['case_id']+'_scoring.json');write(trace,env.trace)
                row.update(case_id=case['case_id'],condition=args.condition,policy_seed=args.seed,
                    heldout_pair=case['heldout_pair'],same_number=case['start_id'][1]==case['goal_id'][1],
                    dataset_sha256=sha(path),scoring_sha256=sha(trace),rgb_numerical_and_scoring_goal_agree=True)
                if record:
                    video=out/'first_predeclared_case.mp4'
                    row['verified_video_frames']=save_video(video,env.frames,env.sim.dt);row['video_sha256']=sha(video)
                rows.append(row);write(out/'episodes.json',rows)
                print(json.dumps({'suite':args.suite,'run':label,'case':case['case_id'],'success':row['success'],'lifted':row['lifted']}),flush=True)
        check_sources(hashes)
        receipt.update(status='completed',completed_utc=now(),executions=len(rows),successes=sum(r['success'] for r in rows),
                       lifts=sum(r['lifted'] for r in rows),episodes_sha256=sha(out/'episodes.json'))
        write(out/'manifest.json',receipt)
    except Exception as exc:
        receipt.update(status='failed',error=str(exc));write(out/'manifest.json',receipt);raise
    finally:
        env.close()


if __name__=='__main__':
    main()
