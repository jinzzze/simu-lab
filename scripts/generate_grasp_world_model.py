"""Collect predeclared grasp macro-actions; true state is for training labels only."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse, hashlib, json, subprocess, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.sim import GraspEnv
CFG = ROOT/'configs/grasp_world_model.json'
OUTPUT = ROOT/'data/processed/grasp_world_model_v1'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def collect(split,cfg):
    path=OUTPUT/f'{split}.jsonl'
    if path.exists(): raise FileExistsError(f'Preserve existing collection: {path}')
    spec=cfg['splits'][split]
    env=GraspEnv(image_size=224)
    try:
        with path.open('w',encoding='utf-8') as log:
            for scene in range(spec['start'],spec['start']+spec['count']):
                rng=np.random.default_rng(np.random.SeedSequence([cfg['offset_rng_seed'],scene]))
                offsets=np.vstack([np.zeros((1,2)),rng.uniform(-cfg['offset_bound_m'],cfg['offset_bound_m'],(4,2))])
                for index,offset in enumerate(offsets):
                    obs=env.reset(scene)
                    initial=env.get_object_xy_for_labels()
                    command=initial+offset
                    # This action is constructed before execution from privileged
                    # training state. Final motion never defines the command.
                    row={'split':split,'scene_seed':scene,'action_index':index,
                         'initial_xy_m':initial.tolist(),'command_xy_m':command.tolist(),
                         'prespecified_offset_m':offset.tolist(),
                         'initial_rgb_sha256':hashlib.sha256(obs['rgb'].tobytes()).hexdigest()}
                    outcome=env.execute_grasp(command)
                    final=env.get_object_xy_for_labels()
                    row.update({'final_xy_m':final.tolist(),'delta_xy_m':(final-initial).tolist(),
                                'lifted':bool(outcome.get('lifted',False)),'success':bool(outcome['success']),
                                'outcome':outcome})
                    log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                print(f'{split}: scene {scene}, collected {(scene-spec["start"]+1)*5}/{spec["count"]*5}',flush=True)
    finally: env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['train','val','test'])
    args=parser.parse_args();cfg=read(CFG)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if args.split:
        collect(args.split,cfg);return
    manifest_path=ROOT/'data/manifests/grasp_world_model_v1.json'
    if manifest_path.exists(): raise FileExistsError('Collection manifest already exists; preserve prior work')
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'collecting',
              'plan':cfg,'config_sha256':sha(CFG),'sim_source_sha256':sha(ROOT/'src/sim/grasp_env.py'),
              'sim_config_sha256':sha(ROOT/'configs/grasp_sim.json'),'collector_sha256':sha(__file__),
              'source':'simulator state/action/outcomes only; no phone-video physics labels'}
    write(manifest_path,manifest)
    def run(split):
        with (OUTPUT/f'{split}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--split',split],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:raise RuntimeError(f'{split} failed; inspect {OUTPUT/split}')
        return split
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(run,split) for split in cfg['splits']]
        for job in as_completed(jobs): print('Completed split '+job.result(),flush=True)
    records=[]
    for split,spec in cfg['splits'].items():
        rows=[json.loads(x) for x in (OUTPUT/f'{split}.jsonl').read_text().splitlines()]
        expected=[(split,s,i) for s in range(spec['start'],spec['start']+spec['count']) for i in range(5)]
        assert [(r['split'],r['scene_seed'],r['action_index']) for r in rows]==expected
        records.extend(rows)
    seen={}
    for row in records:
        s=row['scene_seed'];assert s not in seen or seen[s]==row['split'];seen[s]=row['split']
        assert np.allclose(np.asarray(row['initial_xy_m'])+row['prespecified_offset_m'],row['command_xy_m'])
    target=ROOT/'data/processed/grasp_world_model_v1.npz'
    np.savez_compressed(target,initial_xy_m=np.asarray([r['initial_xy_m'] for r in records],np.float32),
        command_xy_m=np.asarray([r['command_xy_m'] for r in records],np.float32),
        delta_xy_m=np.asarray([r['delta_xy_m'] for r in records],np.float32),
        binary_targets=np.asarray([[r['lifted'],r['success']] for r in records],np.float32),
        scene_seeds=np.asarray([r['scene_seed'] for r in records]),split=np.asarray([r['split'] for r in records]),
        action_index=np.asarray([r['action_index'] for r in records]))
    assert manifest['config_sha256']==sha(CFG) and manifest['sim_source_sha256']==sha(ROOT/'src/sim/grasp_env.py')
    manifest.update({'status':'completed','episodes':len(records),'independent_scene_count':len(seen),
        'dataset_sha256':sha(target),'split_counts':{s:sum(r['split']==s for r in records) for s in cfg['splits']},
        'split_successes':{s:sum(r['success'] for r in records if r['split']==s) for s in cfg['splits']},
        'jsonl_sha256':{s:sha(OUTPUT/f'{s}.jsonl') for s in cfg['splits']}})
    write(manifest_path,manifest);print(json.dumps(manifest['split_counts']),flush=True)


if __name__=='__main__':main()
