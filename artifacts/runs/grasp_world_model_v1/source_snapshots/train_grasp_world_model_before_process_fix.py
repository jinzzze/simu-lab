"""Train fixed-budget macro-action outcome models and evaluate held-out scenes."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.perception.training import seed_everything,batch_stream
from src.world_model.numpy_model import features,MacroOutcomePredictor
OUT=ROOT/'artifacts/runs/grasp_world_model_v1'
REPORT=ROOT/'artifacts/reports/grasp_world_model_v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def metrics(pred_delta,probabilities,truth_delta,targets):
    errors=np.linalg.norm(pred_delta-truth_delta,axis=1)*1000
    result={'n_transitions':len(errors),'xy_mean_error_mm':float(errors.mean()),
            'xy_median_error_mm':float(np.median(errors)),'xy_p90_error_mm':float(np.percentile(errors,90)),
            'xy_rmse_mm':float(np.sqrt(np.mean(errors**2)))}
    for i,name in enumerate(['lift','success']):
        p=probabilities[:,i];y=targets[:,i];pos=p[y==1];neg=p[y==0]
        result[name]={'positives':int(y.sum()),'accuracy_at_0_5':float(np.mean((p>=.5)==y)),
                      'brier':float(np.mean((p-y)**2)),
                      'log_loss':float(-np.mean(y*np.log(np.clip(p,1e-7,1-1e-7))+(1-y)*np.log(np.clip(1-p,1e-7,1-1e-7)))),
                      'auroc':float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg))) if len(pos) and len(neg) else None}
    return result


def main():
    cfg=read(ROOT/'configs/grasp_world_model.json');data_manifest=read(ROOT/'data/manifests/grasp_world_model_v1.json')
    path=ROOT/'data/processed/grasp_world_model_v1.npz'
    assert data_manifest['status']=='completed' and data_manifest['dataset_sha256']==sha(path)
    assert data_manifest['config_sha256']==sha(ROOT/'configs/grasp_world_model.json')
    assert data_manifest['sim_source_sha256']==sha(ROOT/'src/sim/grasp_env.py')
    if OUT.exists():raise FileExistsError('Preserve existing world-model training run')
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
    data=np.load(path,allow_pickle=False)
    x=torch.from_numpy(features(data['initial_xy_m'],data['command_xy_m']))
    scale=cfg['model']['delta_scale_m']
    y=torch.from_numpy(data['delta_xy_m']/scale);binary=torch.from_numpy(data['binary_targets'])
    ids={split:np.flatnonzero(data['split']==split) for split in cfg['splits']}
    scene_sets={split:set(data['scene_seeds'][index].tolist()) for split,index in ids.items()}
    assert not scene_sets['train']&scene_sets['val'] and not scene_sets['train']&scene_sets['test'] and not scene_sets['val']&scene_sets['test']
    for split,index in ids.items():
        assert len(index)==cfg['splits'][split]['count']*5
    settings=cfg['training'];exports={'seeds':np.asarray(settings['seeds']), 'delta_scale_m':np.asarray(scale,np.float32)}
    ensemble_delta=[];ensemble_probs=[];per_seed=[];curves=[]
    metadata={'created_utc':datetime.now(timezone.utc).isoformat(),'plan':cfg,'dataset_sha256':sha(path),
        'dataset_manifest_sha256':sha(ROOT/'data/manifests/grasp_world_model_v1.json'),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/world_model/numpy_model.py']},
        'input_semantics':'true initial state for training/evaluation; externally estimated state at deployment',
        'scope':'Fixed-skill macro-action model; all transition labels are simulator generated',
        'torch':str(torch.__version__),'numpy':np.__version__}
    write(OUT/'manifest.json',metadata)
    for seed in settings['seeds']:
        seed_everything(seed);torch.set_num_threads(settings['torch_threads'])
        model=nn.Sequential(nn.Linear(4,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,4))
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
        directory=OUT/f'seed{seed}';directory.mkdir()
        start=time.perf_counter();logs=[];batch_hash=hashlib.sha256()
        for step,index in enumerate(batch_stream(ids['train'],settings['batch_size'],settings['steps'],seed+300),1):
            batch_hash.update(index.numpy().tobytes())
            output=model(x[index]);reg=F.mse_loss(output[:,:2],y[index]);cl=F.binary_cross_entropy_with_logits(output[:,2:],binary[index])
            loss=settings['regression_weight']*reg+settings['classification_weight']*cl
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            row={'step':step,'total':float(loss.detach()),'regression':float(reg.detach()),'classification':float(cl.detach())};logs.append(row)
            if step%500==0:print(f'WM seed={seed} step={step} loss={row["total"]:.5f}',flush=True)
        model.eval()
        with torch.inference_mode():raw=model(x);delta=raw[:,:2].numpy()*scale;prob=raw[:,2:].sigmoid().numpy()
        ensemble_delta.append(delta);ensemble_probs.append(prob);curves.append([r['total'] for r in logs])
        measured={split:metrics(delta[index],prob[index],data['delta_xy_m'][index],data['binary_targets'][index]) for split,index in ids.items()}
        entry={'seed':seed,'metrics':measured,'steps_completed':settings['steps'],'elapsed_seconds':time.perf_counter()-start,
               'batch_sequence_sha256':batch_hash.hexdigest(),'selection':'fixed last step, no validation/test checkpoint selection'}
        write(directory/'manifest.json',entry)
        (directory/'losses.jsonl').write_text('\n'.join(json.dumps(r) for r in logs)+'\n',encoding='utf-8')
        torch.save({'kind':'grasp_macro_world_model_v1','state_dict':model.state_dict(),'manifest':entry},directory/'checkpoint.pt')
        for layer in (0,2,4):
            exports[f's{seed}_w{layer}']=model[layer].weight.detach().numpy().copy()
            exports[f's{seed}_b{layer}']=model[layer].bias.detach().numpy().copy()
        per_seed.append(entry)
    np.savez_compressed(OUT/'ensemble.npz',**exports)
    delta=np.mean(ensemble_delta,axis=0);prob=np.mean(ensemble_probs,axis=0)
    numpy_predictor=MacroOutcomePredictor(OUT/'ensemble.npz')
    exported=numpy_predictor(data['initial_xy_m'],data['command_xy_m'])
    numpy_delta_difference=float(np.max(np.abs(exported['delta_xy_m']-delta)))
    numpy_prob_difference=float(np.max(np.abs(np.column_stack([exported['lift_probability'],exported['success_probability']])-prob)))
    assert numpy_delta_difference<1e-6 and numpy_prob_difference<2e-6,'NumPy export disagrees with Torch'
    means=data['delta_xy_m'][ids['train']].mean(axis=0);priors=data['binary_targets'][ids['train']].mean(axis=0)
    evaluations={}
    for split,index in ids.items():
        constant_probs=np.broadcast_to(priors,(len(index),2))
        evaluations[split]={'ensemble':metrics(delta[index],prob[index],data['delta_xy_m'][index],data['binary_targets'][index]),
          'persistence':metrics(np.zeros((len(index),2)),constant_probs,data['delta_xy_m'][index],data['binary_targets'][index]),
          'train_mean_delta':metrics(np.broadcast_to(means,(len(index),2)),constant_probs,data['delta_xy_m'][index],data['binary_targets'][index]),
          'independent_scenes':len(scene_sets[split])}
    result={**metadata,'per_seed':per_seed,'evaluations':evaluations,
      'export_max_delta_difference_m':numpy_delta_difference,'export_max_probability_difference':numpy_prob_difference,
      'ensemble_sha256':sha(OUT/'ensemble.npz'),'status':'completed',
      'limits':['Five actions share each initial scene; 160 test transitions are 32 scenes, not 160 independent environments.',
                'Training and prediction evaluation use simulator true initial XY. The separate RGB integration demo measures a different input condition.',
                'Only one fixed grasp skill, goal, block, camera/physics configuration. No long-horizon or pixel prediction claim.',
                'All transition targets come from simulation; personal video only contributes through the separately trained visual encoder.']}
    write(OUT/'results.json',result);write(REPORT/'results.json',result)
    np.savez_compressed(REPORT/'predictions.npz',predicted_delta_xy_m=delta,probabilities=prob,initial_xy_m=data['initial_xy_m'],
                        command_xy_m=data['command_xy_m'],true_delta_xy_m=data['delta_xy_m'],binary_targets=data['binary_targets'],
                        split=data['split'],scene_seeds=data['scene_seeds'])
    make_report(result,data,delta,prob,curves,ids['test'])
    print(json.dumps(evaluations['test'],indent=2),flush=True)


def make_report(result,data,delta,prob,curves,test):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=result['evaluations']['test']
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    names=['persistence','train_mean_delta','ensemble'];labels=['No motion','Train mean','Learned ensemble']
    axes[0].bar(labels,[table[name]['xy_mean_error_mm'] for name in names],color=['#9da9b0','#69879a','#087c83'])
    axes[0].set(ylabel='Final XY mean error (mm)',title='Unseen initial scenes')
    axes[1].bar(['Train prior','Learned ensemble'],[table['persistence']['success']['brier'],table['ensemble']['success']['brier']],color=['#9da9b0','#087c83'])
    axes[1].set(ylabel='Success probability Brier score',title='Lower is better')
    offsets=(data['command_xy_m'][test]-data['initial_xy_m'][test])*1000
    dots=axes[2].scatter(offsets[:,0],offsets[:,1],c=prob[test,1],vmin=0,vmax=1,cmap='viridis',s=30)
    positives=data['binary_targets'][test,1].astype(bool)
    axes[2].scatter(offsets[positives,0],offsets[positives,1],facecolors='none',edgecolors='red',s=60,label='Observed success')
    axes[2].set(xlabel='Command X offset (mm)',ylabel='Command Y offset (mm)',title='Action-conditioned predictions')
    axes[2].legend(fontsize=8);fig.colorbar(dots,ax=axes[2],label='Predicted success probability')
    fig.suptitle('Macro-action world model: 160 transitions from 32 held-out scenes',fontsize=12)
    fig.savefig(REPORT/'prediction_quality.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
    for seed,curve in zip(result['plan']['training']['seeds'],curves):
        ax.plot(np.arange(50,len(curve)+1),np.convolve(curve,np.ones(50)/50,mode='valid'),label=f'seed {seed}')
    ax.set(xlabel='Training step',ylabel='MSE + binary cross entropy',title='Fixed-budget training, trailing 50-step average');ax.legend()
    fig.savefig(REPORT/'training.png',dpi=180);plt.close(fig)
    lines=['# 抓取宏动作世界模型：独立扩展','','该模型学习一次完整抓取技能的状态转移和事件概率。输入是执行前的物体 XY 与下发抓取 XY，输出最终 XY 位移、合格抬升概率和严格任务成功概率。它不生成视频，也不替代物理引擎。','',
    '已冻结的数据为 64 个训练场景×5动作、16 个验证场景×5动作、32 个测试场景×5动作，共 560 次物理执行。每场景一个零偏移和四个预先随机确定的偏移；真实位移不作为动作标签。划分按完整初始场景，未根据结果调参。','',
    '训练和以下预测质量评估采用仿真真实初始 XY；部署接口只接收外部估计值。个人手机视频没有提供这些物理转移标签，它通过独立视觉模型参与后续 RGB 集成演示。','',
    '| 预测器 | 最终 XY 平均误差 / mm | 成功概率 Brier | 成功准确率（阈值0.5） |','|---|---:|---:|---:|']
    for key,label in zip(names,['不发生位移','训练集平均位移','三种子学习模型平均']):
        row=table[key];lines.append(f"| {label} | {row['xy_mean_error_mm']:.3f} | {row['success']['brier']:.4f} | {row['success']['accuracy_at_0_5']*100:.2f}% |")
    lines+=['','基线分类概率统一采用训练集事件频率；不使用测试标签估计常数。Brier 越小越好，分类准确率也可能受类别比例影响。','',
      '## 每个固定训练种子','','| 种子 | 最终 XY 平均误差 / mm | 成功概率 Brier |','|---|---:|---:|']
    for entry in result['per_seed']:
        row=entry['metrics']['test'];lines.append(f"| {entry['seed']} | {row['xy_mean_error_mm']:.3f} | {row['success']['brier']:.4f} |")
    lines+=['','![预测质量](prediction_quality.png)','','![训练损失](training.png)','','## 解释边界','',
      '- 三个种子全部使用最终固定步数检查点，ensemble 是预先约定的等权平均，没有挑选最好模型。',
      '- 160 条测试转移只来自 32 个初始场景；同场景五个动作相关，不按160个独立环境计算置信区间。',
      '- 场景物体、目标、技能、摩擦和质量固定。没有验证新物体、目标变化、长时序滚动或视频生成。',
      '- 使用真实状态的预测误差不能等同于 RGB 感知不确定条件下的预测误差；集成演示单独记录。',
      '- 这个扩展不修改前面 A/B/C/D/R 的1,920次模型控制结果，也不证明世界模型提高了策略成功率。',
      f"- NumPy 导出与 Torch 的最大位移差为 {result['export_max_delta_difference_m']:.3g} m，最大概率差 {result['export_max_probability_difference']:.3g}。",'',
      '[完整指标、配置与来源](results.json)','']
    (REPORT/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
