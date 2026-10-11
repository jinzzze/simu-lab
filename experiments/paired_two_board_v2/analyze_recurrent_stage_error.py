"""Teacher-forced action errors grouped AFTER inference by diagnostic stage."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib,json,shutil,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from src.perception.training import seed_everything
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--tag',default='stage_error_v1');args=parser.parse_args()
    cfg=read(ROOT/args.config);module=importlib.import_module('paired_recurrent_goal_policy' if cfg['kind']=='paired_goal_relative_recurrent_bc_v4' else 'paired_recurrent_policy')
    assert cfg['kind'] in ('paired_goal_recurrent_bc_v3','paired_goal_relative_recurrent_bc_v4')
    index=read(ROOT/cfg['training_index']);entries=index['shared_validation'];seed_everything(0);torch.set_num_threads(4)
    data=module.load_episodes(ROOT,entries,128,'cuda');out=ROOT/f'artifacts/reports/{cfg["revision"]}/{args.tag}'
    assert not out.exists();out.mkdir(parents=True);records=[];sources={args.config:sha(ROOT/args.config),'analyze_recurrent_stage_error.py':sha(__file__)}
    for condition in cfg['conditions']:
        for seed in cfg['training']['seeds']:
            folder=ROOT/f'artifacts/runs/{cfg["revision"]}/{condition}_seed{seed}';meta=read(folder/'manifest.json')
            assert meta['status']=='completed' and meta['config']==cfg and sha(folder/'checkpoint.pt')==meta['checkpoint_sha256']
            payload=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
            model=module.GoalRecurrentModel(cfg).to('cuda').eval();model.load_state_dict(payload['ema_state'])
            normalizer={k:v.to('cuda') for k,v in payload['normalizer'].items()}
            sources[str((folder/'checkpoint.pt').relative_to(ROOT))]=sha(folder/'checkpoint.pt')
            for ep,entry in enumerate(entries):
                path=ROOT/entry['file'];assert sha(path)==entry['dataset_sha256'];sources[entry['file']]=entry['dataset_sha256']
                with torch.inference_mode():
                    rgb,obs,_,mask=module.episode_batch(data,torch.tensor([ep],device='cuda'),normalizer)
                    normalized,_=model(rgb,obs,None)
                    raw=normalized*normalizer['action_scale']+normalizer['action_center']
                    if cfg['action_representation']=='ee_delta_xyz_absolute_width':raw[:,:,:3]+=data[1][ep:ep+1,:,:3]
                    else:raw[:,:,:2]+=data[1][ep:ep+1,:,4:6]
                    n=int(data[3][ep]);difference=(raw[0,:n]-data[2][ep,:n]).cpu().numpy()*1000
                # Stage strings are read only after prediction. They never enter the model.
                with np.load(path,allow_pickle=False) as stored:stages=stored['stages']
                assert len(stages)==n
                for stage in dict.fromkeys(stages.tolist()):
                    d=difference[stages==stage]
                    records.append({'condition':condition,'policy_seed':seed,'teacher_episode':entry['episode_id'],'stage':stage,
                        'commands':len(d),'mean_xyz_error_mm':float(np.linalg.norm(d[:,:3],axis=1).mean()),
                        'mean_xy_error_mm':float(np.linalg.norm(d[:,:2],axis=1).mean()),'signed_xy_error_mm':d[:,:2].mean(0).tolist(),
                        'mean_width_error_mm':float(np.abs(d[:,3]).mean())})
            del model
    groups={}
    for condition in cfg['conditions']:
        subset=[r for r in records if r['condition']==condition];stage_rows={}
        for stage in dict.fromkeys(r['stage'] for r in subset):
            selected=[r for r in subset if r['stage']==stage];count=sum(r['commands'] for r in selected)
            stage_rows[stage]={'commands':count,**{key:sum(r[key]*r['commands'] for r in selected)/count for key in ('mean_xyz_error_mm','mean_xy_error_mm','mean_width_error_mm')}}
        groups[condition]=stage_rows
    result={'revision':cfg['revision'],'created_utc':datetime.now(timezone.utc).isoformat(),'groups':groups,'per_episode_model_stage':records,
        'source_sha256':sources,'scope':'teacher-forced current RGB/robot-state sequences; stage strings only group post-inference errors; not runtime stages, new physical rollouts or task success'}
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 循环策略：教师强制误差的阶段诊断','', '模型输入仍只有 RGB 与机器人/目标向量，整段因果循环前向。阶段标签在预测后才读取，只用于分组；没有把阶段交给策略。这里使用教师轨迹上的机器人状态，不能把小误差当作自由执行成功。',
        '', '| 条件 | 诊断阶段 | 样本命令数 | XYZ 均误差 mm | XY 均误差 mm | 宽度均误差 mm |','|---|---|---:|---:|---:|---:|']
    for condition,stages in groups.items():
        for stage,r in stages.items():lines.append(f'| {condition} | {stage} | {r["commands"]} | {r["mean_xyz_error_mm"]:.3f} | {r["mean_xy_error_mm"]:.3f} | {r["mean_width_error_mm"]:.3f} |')
    lines+=['', '计数复用了共同的八条教师验证轨迹和三个模型，按命令数加权，不是独立环境数量。v3 的微小位移偏差可能累积，v4 的绝对目标误差不会以同样方式积分；这是假设解释，实际自由执行仍以完整物理评分为准。数据中原有教师失败没有删除。',
        '', '[逐模型/轨迹/阶段、来源哈希与数字](results.json)']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copy2(__file__,out/Path(__file__).name)
    print(json.dumps({'report':str(out/'REPORT_ZH.md'),'groups':groups},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
