"""Matched-scene RGB model evaluation with a shared physical grasp controller."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.sim.grasp_env import GraspEnv
from src.predictor_process import PredictorProcess


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save_video(path,frames,fps):
    if not frames:return
    height,width=frames[0].shape[:2]
    writer=imageio_ffmpeg.write_frames(str(path),(width,height),fps=fps,codec='libx264',quality=7,
        pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-movflags','+faststart'])
    writer.send(None)
    try:
        for frame in frames:writer.send(np.ascontiguousarray(frame))
    finally:writer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,default=ROOT/'artifacts/runs/visual_state_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/runs/visual_grasp_v1')
    parser.add_argument('--seeds',type=int,nargs='+',default=[7])
    parser.add_argument('--groups',nargs='+',default=['A','B','C','D'])
    parser.add_argument('--split',choices=['val','test'],default='test')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--record-first',action='store_true')
    parser.add_argument('--jobs',type=int,default=3,help='Independent training-seed processes; physics remains identical')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    data_manifest=json.loads((ROOT/'data/manifests/sim_adaptation_v1.json').read_text(encoding='utf-8'))
    scene_seeds=[x['scene_seed'] for x in data_manifest['scenes'] if x['split']==args.split]
    if args.limit:scene_seeds=scene_seeds[:args.limit]
    settings={'scope':'exploratory pilot-trained visual grasp evaluation; not final multi-session real-data evidence',
        'groups':args.groups,'training_seeds':args.seeds,'split':args.split,'scene_seeds':scene_seeds,
        'sim_config_sha256':sha(ROOT/'configs/grasp_sim.json'),'sim_code_sha256':sha(ROOT/'src/sim/grasp_env.py'),
        'dataset_manifest_sha256':sha(ROOT/'data/manifests/sim_adaptation_v1.json'),
        'recording_rule':'first preselected scene per group and seed, regardless of outcome',
        'runtime_inputs':'RGB to predictor; predicted XY, known goal and robot proprioception to fixed controller',
        'ground_truth_use':'separate label/error measurement and terminal scoring only',
        'prediction_process':'persistent RGB-only worker; avoids conflicting native OpenMP runtimes',
        'evaluator_sha256':sha(Path(__file__)),
        'predictor_worker_sha256':sha(ROOT/'scripts/rgb_predictor_worker.py'),
        'predictor_transport_sha256':sha(ROOT/'src/predictor_process.py')}
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.worker:
        (args.output/'evaluation_plan.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    if len(args.seeds)>1 and args.jobs>1 and not args.worker:
        def evaluate_seed(seed):
            command=[sys.executable,'-u',str(Path(__file__)), '--models',str(args.models),
                     '--output',str(args.output),'--seeds',str(seed),'--groups',*args.groups,
                     '--split',args.split,'--jobs','1','--worker']
            if args.limit:command.extend(['--limit',str(args.limit)])
            if args.record_first:command.append('--record-first')
            log=args.output/f'worker_seed{seed}.log'
            with log.open('w',encoding='utf-8') as stream:
                completed=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if completed.returncode:
                raise RuntimeError(f'Evaluation seed {seed} failed; see {log}')
            return seed
        with ThreadPoolExecutor(max_workers=min(args.jobs,len(args.seeds))) as pool:
            jobs=[pool.submit(evaluate_seed,seed) for seed in args.seeds]
            for job in as_completed(jobs):
                print(f'Completed physical evaluation for training seed {job.result()}',flush=True)
        results=[json.loads((args.output/f'{group}_seed{seed}'/'summary.json').read_text(encoding='utf-8'))
                 for seed in args.seeds for group in args.groups]
        (args.output/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(json.dumps(results),flush=True)
        return
    aggregate=[];env=GraspEnv(image_size=224);predict=None
    try:
        for train_seed in args.seeds:
            for group in args.groups:
                directory=args.output/f'{group}_seed{train_seed}';directory.mkdir(parents=True,exist_ok=True)
                checkpoint=args.models/f'{group}_seed{train_seed}'/'checkpoint.pt'
                model_hash=sha(checkpoint)
                completed=directory/'summary.json'
                if completed.exists():
                    previous=json.loads(completed.read_text(encoding='utf-8'))
                    if (previous['scene_seeds']==scene_seeds and previous['sim_code_sha256']==settings['sim_code_sha256']
                        and previous['sim_config_sha256']==settings['sim_config_sha256']
                        and previous['checkpoint_sha256']==model_hash):
                        aggregate.append(previous);print(f'Skipped complete matching evaluation {group} seed{train_seed}',flush=True);continue
                    raise FileExistsError(f'Incompatible existing evaluation: {directory}')
                predict=PredictorProcess(checkpoint,directory/'predictor_stderr.log')
                episodes=[]
                with (directory/'episodes.jsonl').open('w',encoding='utf-8') as stream:
                    for scene_seed in scene_seeds:
                        obs=env.reset(scene_seed)
                        rgb=obs['rgb']
                        rgb_sha=hashlib.sha256(rgb.tobytes()).hexdigest()
                        start=time.perf_counter();estimated=np.asarray(predict(rgb),np.float64);elapsed=time.perf_counter()-start
                        # Privileged accessor is for a diagnostic metric only; never supplied to the predictor/controller.
                        truth=env.get_object_xy_for_labels()
                        record=args.record_first and scene_seed==scene_seeds[0]
                        outcome=env.execute_grasp(estimated,record=record)
                        row={'group':group,'training_seed':train_seed,'scene_seed':scene_seed,
                            'initial_rgb_sha256':rgb_sha,'prediction_xy_m':estimated.tolist(),'label_xy_m':truth.tolist(),
                            'initial_xy_error_mm':float(np.linalg.norm(estimated-truth)*1000),
                            'prediction_wall_seconds':elapsed,'outcome':outcome,'success':bool(outcome['success'])}
                        stream.write(json.dumps(row)+'\n');stream.flush();episodes.append(row)
                        if record:
                            save_video(directory/'first_scene_demo.mp4',env.frames,1/(env.sim.dt*2))
                        print(f'{group} train_seed={train_seed} scene={scene_seed} success={row["success"]} xy_error={row["initial_xy_error_mm"]:.2f}mm',flush=True)
                errors=np.array([r['initial_xy_error_mm'] for r in episodes])
                result={'group':group,'training_seed':train_seed,'scene_seeds':scene_seeds,
                    'checkpoint_sha256':model_hash,'sim_config_sha256':settings['sim_config_sha256'],
                    'sim_code_sha256':settings['sim_code_sha256'],'n':len(episodes),'successes':sum(r['success'] for r in episodes),
                    'success_rate':float(np.mean([r['success'] for r in episodes])),
                    'xy_mean_error_mm':float(errors.mean()),'xy_median_error_mm':float(np.median(errors)),
                    'xy_p90_error_mm':float(np.percentile(errors,90)),
                    'scope':settings['scope'],'demo_is_first_scene_not_best_scene':True}
                completed.write_text(json.dumps(result,indent=2),encoding='utf-8');aggregate.append(result)
                summary_name=f'summary_seed{train_seed}.json' if args.worker else 'summary.json'
                (args.output/summary_name).write_text(json.dumps(aggregate,indent=2),encoding='utf-8')
                predict.close();predict=None
    finally:
        if predict is not None:predict.close()
        env.close()
    summary_name=f'summary_seed{args.seeds[0]}.json' if args.worker else 'summary.json'
    (args.output/summary_name).write_text(json.dumps(aggregate,indent=2),encoding='utf-8')
    print(json.dumps(aggregate),flush=True)

if __name__=='__main__':main()
