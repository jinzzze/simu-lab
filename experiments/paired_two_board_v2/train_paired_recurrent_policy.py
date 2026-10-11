"""Paired fixed-budget recurrent BC on whole executed episodes."""
from datetime import datetime,timezone
from pathlib import Path
import argparse,copy,hashlib,json,shutil,time
import numpy as np
import torch
from paired_recurrent_policy import ROOT,PROJECT,GoalRecurrentModel,episode_batch,load_episodes,fixed_normalizer
from src.perception.training import seed_everything

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def weight_sha(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
ACTIVE=None


def main():
    global ACTIVE
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/policy_recurrent_v3.json');args=parser.parse_args()
    cfg=read(ROOT/args.config);index=read(ROOT/cfg['training_index']);task=read(ROOT/'configs/sim_task_v1.json')
    assert sha(ROOT/cfg['dataset_manifest'])==index['dataset_manifest_sha256'];assert torch.cuda.is_available()
    torch.set_num_threads(4);device='cuda';normalizer=fixed_normalizer(task,device)
    val=load_episodes(ROOT,index['shared_validation'],128,device)
    sources={name:ROOT/name for name in ('train_paired_recurrent_policy.py','paired_recurrent_policy.py','paired_delta_policy.py','paired_image_policy.py')}
    sources.update({'policy_config.json':ROOT/args.config,'sim_task_v1.json':ROOT/'configs/sim_task_v1.json','training.py':PROJECT/'src/perception/training.py','diffusion.py':PROJECT/'src/policy/diffusion.py'})
    hashes={name:sha(path) for name,path in sources.items()}
    for condition in cfg['conditions']:
        data=load_episodes(ROOT,index['train_groups'][condition],128,device)
        for seed in cfg['training']['seeds']:
            folder=ROOT/f"artifacts/runs/{cfg['revision']}/{condition}_seed{seed}";assert not folder.exists();folder.mkdir(parents=True);ACTIVE=folder
            snap=folder/'source_snapshot';snap.mkdir()
            for name,path in sources.items():shutil.copy2(path,snap/name)
            seed_everything(seed);model=GoalRecurrentModel(cfg).to(device);ema=copy.deepcopy(model).eval();settings=cfg['training']
            optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
            idx=torch.Generator(device=device).manual_seed(seed+1100);aug=torch.Generator(device=device).manual_seed(seed+3100)
            stream=hashlib.sha256();started=time.perf_counter();valid_targets=0
            meta={'status':'training','condition':condition,'seed':seed,'config':cfg,'initial_weights_sha256':weight_sha(model),
                  'source_sha256':hashes,'dataset_manifest_sha256':sha(ROOT/cfg['dataset_manifest']),
                  'started_utc':datetime.now(timezone.utc).isoformat(),'device':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
                  'selection':'fixed final EMA for all three seeds','hidden_state':'reset per whole episode; only causal learned recurrence',
                  'no_expert_stage_or_object_truth_input':True,'loss_weighting':'mean per valid episode, then mean across batch; longer episodes receive no extra episode weight'}
            write(folder/'manifest.json',meta)
            with (folder/'losses.jsonl').open('w',encoding='utf-8') as log:
                for step in range(1,settings['steps']+1):
                    episodes=torch.randint(16,(settings['batch_size'],),device=device,generator=idx);stream.update(episodes.cpu().numpy().tobytes())
                    rgb,obs,act,mask=episode_batch(data,episodes,normalizer)
                    brightness=(torch.rand((len(rgb),1,1,1,1),device=device,generator=aug)*2-1)*cfg['augmentation']['brightness']
                    contrast=1+(torch.rand((len(rgb),1,1,1,1),device=device,generator=aug)*2-1)*cfg['augmentation']['contrast']
                    rgb=((rgb-.5)*contrast+.5+brightness).clamp(0,1)
                    prediction,_=model(rgb,obs,None)
                    loss=(((prediction-act).square().mean(-1)*mask).sum(-1)/mask.sum(-1)).mean()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite recurrent loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),settings['grad_clip']);optimizer.step()
                    with torch.no_grad():
                        for average,current in zip(ema.parameters(),model.parameters()):average.lerp_(current,1-cfg['model']['ema_decay'])
                    valid_targets+=int(mask.sum())
                    row={'step':step,'normalized_bc_mse':float(loss.detach()),'wall_s':time.perf_counter()-started};log.write(json.dumps(row)+'\n')
                    if step%100==0:log.flush();print(json.dumps({'condition':condition,'seed':seed,**row}),flush=True)
            xyz=[];width=[];val_loss=[]
            with torch.inference_mode():
                for ep in range(len(index['shared_validation'])):
                    rgb,obs,act,mask=episode_batch(val,torch.tensor([ep],device=device),normalizer);prediction,_=ema(rgb,obs,None)
                    difference=(prediction-act)*normalizer['action_scale']
                    xyz.append(float((difference[:,:,:3].norm(dim=-1)*mask).sum()/mask.sum()*1000))
                    width.append(float((difference[:,:,3].abs()*mask).sum()/mask.sum()*1000))
                    val_loss.append(float(((prediction-act).square().mean(-1)*mask).sum()/mask.sum()))
            meta.update(status='completed',steps_completed=settings['steps'],parameters=sum(p.numel() for p in model.parameters()),
                        episode_fraction_stream_sha256=stream.hexdigest(),stream_semantics='whole episode IDs; no fractions',valid_action_targets_seen=valid_targets,
                        wall_s=time.perf_counter()-started,completed_utc=datetime.now(timezone.utc).isoformat(),
                        validation={'teacher_forced_episodes':8,'normalized_bc_mse':float(np.mean(val_loss)),
                                    'delta_xyz_error_mm':float(np.mean(xyz)),'width_error_mm':float(np.mean(width)),
                                    'scope':'teacher-forced recurrent action error, not closed-loop success'})
            for name,path in sources.items():assert sha(path)==hashes[name],'Source changed during recurrent training'
            payload={'kind':'paired_goal_recurrent_bc_v3','config':cfg,'condition':condition,'seed':seed,
                     'ema_state':{k:v.cpu() for k,v in ema.state_dict().items()},'model_state':{k:v.cpu() for k,v in model.state_dict().items()},
                     'normalizer':{k:v.cpu() for k,v in normalizer.items()},'manifest':meta}
            torch.save(payload,folder/'checkpoint.pt');meta['checkpoint_sha256']=sha(folder/'checkpoint.pt');write(folder/'manifest.json',meta)
            print(json.dumps({'completed_run':folder.name,'validation':meta['validation'],'wall_s':meta['wall_s']}),flush=True)
        del data


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if ACTIVE and (ACTIVE/'manifest.json').exists():
            meta=read(ACTIVE/'manifest.json');meta.update(status='failed',error=str(exc));write(ACTIVE/'manifest.json',meta)
        raise
