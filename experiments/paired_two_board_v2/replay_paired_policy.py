"""Re-run a fixed checkpoint/scene without overwriting or counting a new benchmark."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib,json,re,shutil
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
MODULES={'paired_goal_image_diffusion_v1':'evaluate_paired_policy',
    'paired_goal_image_diffusion_delta_v2':'evaluate_paired_delta_policy',
    'paired_goal_recurrent_bc_v3':'evaluate_paired_recurrent_policy',
    'paired_goal_relative_recurrent_bc_v4':'evaluate_paired_recurrent_goal_policy'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--tag',required=True)
    p.add_argument('--condition',choices=['simulation_only','video_guided'],default='simulation_only')
    p.add_argument('--seed',type=int,default=7);p.add_argument('--suite',default='validation');p.add_argument('--scene',type=int);args=p.parse_args()
    assert re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',args.tag),'Use a short safe output tag'
    cfg=read(ROOT/args.config);assert args.seed in cfg['training']['seeds']
    model=ROOT/f'artifacts/runs/{cfg["revision"]}/{args.condition}_seed{args.seed}'
    training=read(model/'manifest.json');assert training['status']=='completed' and training['config']==cfg
    checkpoint=model/'checkpoint.pt';assert sha(checkpoint)==training['checkpoint_sha256']
    # Validate every training dependency before loading the model in its worker.
    for name,digest in training['source_sha256'].items():
        path=ROOT/name
        if name=='policy_config.json':path=ROOT/args.config
        if name=='training.py':path=ROOT.parents[1]/'src/perception/training.py'
        if name=='diffusion.py':path=ROOT.parents[1]/'src/policy/diffusion.py'
        if name=='sim_task_v1.json':path=ROOT/'configs/sim_task_v1.json'
        assert sha(path)==digest,('Training source mismatch',name)
    driver=Path(cfg.get('evaluation_script',MODULES[cfg['kind']]+'.py')).stem
    module=importlib.import_module(driver);definition=module.suite_definition(cfg,args.suite);scene=definition['start'] if args.scene is None else args.scene
    assert definition['start']<=scene<definition['start']+definition['count']
    out=ROOT/f'artifacts/reports/policy_replays/{args.tag}';assert not out.exists();out.mkdir(parents=True)
    snapshot=out/'source_snapshot';snapshot.mkdir();sources={}
    # Include the complete local module surface and the shared native-env source.
    for path in ROOT.glob('*.py'):
        if path.name.startswith(('paired_','evaluate_paired','collect_paired','goal_scoring','replay_paired')):
            shutil.copy2(path,snapshot/path.name);sources[path.name]=sha(path)
    for name,path in [('policy_config.json',ROOT/args.config),('sim_task_v1.json',ROOT/'configs/sim_task_v1.json'),
        ('sim_physics_v1.json',ROOT/'configs/sim_physics_v1.json'),('grasp_env.py',ROOT.parents[1]/'src/sim/grasp_env.py'),
        ('diffusion.py',ROOT.parents[1]/'src/policy/diffusion.py'),('training.py',ROOT.parents[1]/'src/perception/training.py')]:
        shutil.copy2(path,snapshot/name);sources[name]=sha(path)
    env=module.PairedGoalEnv()
    try:
        with module.ImagePolicyProcess(checkpoint,out/'policy_stderr.log') as predictor:
            row,arrays=module.run_episode(env,predictor,cfg,definition,scene,True)
        np.savez_compressed(out/'inputs_actions.npz',**arrays);write(out/'scoring.json',env.trace)
        video=out/'replay.mp4';row['video_frames_verified']=module.save_video(video,env.frames,env.sim.dt)
    finally:env.close()
    reference=ROOT/f'artifacts/runs/{cfg["revision"]}_evaluation/{args.suite}/{args.condition}_seed{args.seed}'
    differences={};match=None
    if (reference/f'scene_{scene}_inputs_actions.npz').exists():
        with np.load(reference/f'scene_{scene}_inputs_actions.npz',allow_pickle=False) as original:
            assert set(original.files)==set(arrays)
            for key,current in arrays.items():
                previous=original[key];assert previous.shape==current.shape
                differences[key]={'exact':bool(np.array_equal(previous,current)),
                    'max_absolute_difference':float(np.max(np.abs(previous.astype(np.float64)-current.astype(np.float64))))}
        records=[json.loads(x) for x in (reference/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
        earlier=next(x for x in records if x['scene_seed']==scene)
        match={'all_input_action_arrays_identical':all(x['exact'] for x in differences.values()),
               'success_identical':row['success']==earlier['success'],'lifted_identical':row['lifted']==earlier['lifted'],
               'final_flags_identical':row['final_flags']==earlier['final_flags'],
               'reference_episode_log_sha256':sha(reference/'episodes.jsonl'),'array_differences':differences}
    manifest={'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),'config':cfg,'condition':args.condition,
        'policy_seed':args.seed,'scene_seed':scene,'checkpoint_sha256':sha(checkpoint),'source_sha256':sources,'evaluation_driver':driver,
        'outcome':row,'reference_comparison':match,'scope':'reproduction diagnostic; not added to benchmark counts',
        'scene_selection':'fixed first scene' if args.scene is None else 'explicit requested scene; may be selected after seeing development results; not representative performance',
        'outputs_sha256':{path.name:sha(path) for path in out.iterdir() if path.is_file()}}
    write(out/'manifest.json',manifest)
    text=f'# 指定场景复跑\n\n{cfg["revision"]}，{args.condition}，seed {args.seed}，场景 {scene}。严格成功 {row["success"]}，合格抬升 {row["lifted"]}。\n\n这是已有模型的场景复现，不计入新测试或总体成功数。\n\n'
    if args.scene is not None:text+='此场景在开发结果可见后明确指定，可能是成功案例。视频仅检视该回合，不能代替全部种子成功率。\n\n'
    if match:text+=f'与原执行输入/动作数组全部逐值一致：{match["all_input_action_arrays_identical"]}；成功/抬升/最终条件判定一致：{match["success_identical"] and match["lifted_identical"] and match["final_flags_identical"]}。\n\n'
    text+='[演示视频](replay.mp4) · [完整收据与差异](manifest.json)\n'
    (out/'REPORT_ZH.md').write_text(text,encoding='utf-8')
    print(json.dumps({'report':str(out/'REPORT_ZH.md'),'reference_comparison':match,'success':row['success']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
