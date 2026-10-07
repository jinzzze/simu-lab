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
    np.savez_compressed(OUT/'torch_predictions.npz',delta=np.stack(ensemble_delta),probabilities=np.stack(ensemble_probs))
    write(OUT/'run_state.json',{'metadata':metadata,'per_seed':per_seed,'curves':curves})
    import subprocess
    subprocess.run([sys.executable,str(ROOT/'scripts/export_grasp_world_model_predictions.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/report_grasp_world_model.py')],cwd=ROOT,check=True)


if __name__=='__main__':main()
