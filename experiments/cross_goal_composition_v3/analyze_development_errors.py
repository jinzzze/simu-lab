"""Teacher-forced development action errors with stages used only for reporting."""
from collections import defaultdict
import numpy as np
import torch
from common import ROOT,read,write,sha,now,check_sources
from paired_recurrent_goal_policy import GoalRecurrentModel,episode_batch,load_episodes


def main():
    out=ROOT/'artifacts/development_action_errors_v1';out.mkdir(parents=True,exist_ok=False)
    cfg=read(ROOT/'configs/policy_v1.json');index=read(ROOT/cfg['training_index'])
    torch.set_num_threads(4);data=load_episodes(ROOT,index['shared_validation'],128,'cuda')
    rows=[]
    for c in cfg['conditions']:
        for seed in cfg['training']['seeds']:
            folder=ROOT/f'artifacts/models/{c}_seed{seed}';meta=read(folder/'manifest.json')
            check_sources(meta['source_sha256']);assert sha(folder/'checkpoint.pt')==meta['checkpoint_sha256']
            payload=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
            model=GoalRecurrentModel(cfg).cuda().eval();model.load_state_dict(payload['ema_state'])
            normalizer={k:v.cuda() for k,v in payload['normalizer'].items()}
            with torch.inference_mode():
                for i,entry in enumerate(index['shared_validation']):
                    rgb,obs,act,mask=episode_batch(data,torch.tensor([i],device='cuda'),normalizer)
                    prediction,_=model(rgb,obs,None);n=entry['steps']
                    diff=((prediction-act)*normalizer['action_scale'])[0,:n].cpu().numpy()*1000
                    trace=read((ROOT/entry['file']).with_suffix('.scoring.json'))
                    assert len(trace)==n and sha((ROOT/entry['file']).with_suffix('.scoring.json'))==entry['scoring_sha256']
                    stages=np.array([r['stage'] for r in trace])
                    groups={s:np.flatnonzero(stages==s) for s in sorted(set(stages))}
                    groups['first10ticks']=np.arange(min(10,n));groups['all']=np.arange(n)
                    for stage,ids in groups.items():
                        rows.append(dict(condition=c,seed=seed,episode=entry['episode_id'],start_id=entry['start_id'],goal_id=entry['goal_id'],
                            same_number=entry['start_id'][1]==entry['goal_id'][1],stage=stage,steps=len(ids),
                            mean_xy_error_mm=float(np.linalg.norm(diff[ids,:2],axis=1).mean()),
                            mean_z_error_mm=float(np.abs(diff[ids,2]).mean()),mean_width_error_mm=float(np.abs(diff[ids,3]).mean())))
            del model
    groups=defaultdict(list)
    for row in rows:
        groups[(row['condition'],row['stage'])].append(row)
    summaries=[dict(condition=c,stage=stage,episode_model_means=len(values),
                    xy_mm=float(np.mean([v['mean_xy_error_mm'] for v in values])),
                    z_mm=float(np.mean([v['mean_z_error_mm'] for v in values])),
                    width_mm=float(np.mean([v['mean_width_error_mm'] for v in values]))) for (c,stage),values in sorted(groups.items())]
    result=dict(created_utc=now(),summary=summaries,episode_stage_errors=rows,
        scope='Post-hoc teacher-forced error on12 simulated developer demonstrations per model; each episode mean equally weighted. Expert stage is reporting metadata only, never a policy input. No new executions or model selection.',
        training_index_sha256=sha(ROOT/cfg['training_index']),analysis_sha256=sha(ROOT/'analyze_development_errors.py'))
    write(out/'results.json',result)
    text=['# 开发示范上的动作误差','',
        '模型使用真实当前/过去示范观测；它没有受到自己上一步预测误差造成的状态偏移。因此这是teacher-forced误差，不能替代闭环抓取成功率。',
        '教师阶段只用于分析分组，没有进入网络输入。先逐轨迹求平均，再对轨迹和三个模型求平均。','',
        '| 条件 | 阶段 | XY误差(mm) | Z误差(mm) | 夹爪宽度误差(mm) |','|---|---|---:|---:|---:|']
    for r in summaries:
        text.append(f"| {r['condition']} | {r['stage']} | {r['xy_mm']:.2f} | {r['z_mm']:.2f} | {r['width_mm']:.2f} |")
    text += ['', '[逐模型、逐轨迹阶段误差](results.json)']
    (out/'REPORT_ZH.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(__import__('json').dumps([r for r in summaries if r['stage'] in ('first10ticks','close','all')]))


if __name__=='__main__':
    main()
