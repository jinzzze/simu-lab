"""Replay the predeclared first case of a suite; do not add it to benchmark counts."""
import argparse
import re
import numpy as np
from common import ROOT,OLD,read,write,sha,now,check_sources
from paired_sim import PairedGoalEnv
from paired_recurrent_goal_policy_process import GoalRelativeRecurrentPolicyProcess
from paired_recombination_driver import run_episode
from collect_paired_demonstrations import save_video


def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=['paired_24','coverage_24'],default='paired_24')
    p.add_argument('--seed',type=int,default=7);p.add_argument('--tag',required=True)
    p.add_argument('--suite',choices=['validation','test'],default='test')
    args=p.parse_args();assert re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',args.tag)
    cfg=read(ROOT/'configs/policy_v1.json');study=read(ROOT/'configs/study_v1.json')
    assert cfg['study_sha256']==sha(ROOT/'configs/study_v1.json') and args.seed in study['seeds']
    train=ROOT/f'artifacts/models/{args.condition}_seed{args.seed}';m=read(train/'manifest.json')
    assert m['status']=='completed' and m['config']==cfg
    check_sources(m['source_sha256']);assert sha(train/'checkpoint.pt')==m['checkpoint_sha256']
    case=study[args.suite+'_cases'][0];ref=ROOT/f'artifacts/evaluation/{args.suite}/{args.condition}_seed{args.seed}'
    previous=read(ref/'manifest.json');assert previous['status']=='completed'
    check_sources(previous['source_sha256']);assert previous['episodes_sha256']==sha(ref/'episodes.json')
    expected=next(r for r in read(ref/'episodes.json') if r['case_id']==case['case_id'])
    out=ROOT/'artifacts/replays'/args.tag;out.mkdir(parents=True,exist_ok=False)
    env=PairedGoalEnv()
    try:
        with GoalRelativeRecurrentPolicyProcess(train/'checkpoint.pt',out/'policy_stderr.log') as predict:
            definition=dict(start=case['scene_seed'],jitter_m=study['jitter_m'],
                prescribed_start_id=case['start_id'],prescribed_goal_id=case['goal_id'])
            row,arrays=run_episode(env,predict,cfg,definition,case['scene_seed'],True)
        ap=ref/(case['case_id']+'_inputs_actions.npz');sp=ref/(case['case_id']+'_scoring.json')
        assert sha(ap)==expected['dataset_sha256'] and sha(sp)==expected['scoring_sha256']
        with np.load(ap,allow_pickle=False) as old:
            diffs={key:dict(exact=bool(np.array_equal(old[key],value)),
                maximum_absolute_difference=float(np.max(np.abs(old[key].astype(float)-value.astype(float))))) for key,value in arrays.items()}
        traces_identical=read(sp)==env.trace
        write(out/'scoring.json',env.trace);np.savez_compressed(out/'inputs_actions.npz',**arrays)
        video=out/'replay.mp4';frames=save_video(video,env.frames,env.sim.dt)
        receipt=dict(status='completed',created_utc=now(),condition=args.condition,seed=args.seed,case=case,suite=args.suite,
            checkpoint_sha256=m['checkpoint_sha256'],input_action_comparisons=diffs,full_scoring_trace_identical=traces_identical,
            all_arrays_identical=all(v['exact'] for v in diffs.values()),outcome=row,
            video_frames_verified=frames,video_sha256=sha(video),replay_script_sha256=sha(ROOT/'replay.py'),
            benchmark_count_addition=0,scope='Fixed first case in the named suite; same-host reproduction, not new test evidence.')
        write(out/'manifest.json',receipt)
        assert receipt['all_arrays_identical'] and traces_identical,'Replay differs; retained for diagnosis'
        print(__import__('json').dumps(receipt,ensure_ascii=False,indent=2),flush=True)
    finally:
        env.close()


if __name__=='__main__':
    main()
