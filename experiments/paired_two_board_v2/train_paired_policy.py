"""Fixed-budget paired image diffusion training; all declared final EMA seeds."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,json,shutil,sys,time
import numpy as np
import torch
from paired_image_policy import ROOT,PROJECT,GoalImageDenoiser,fixed_normalizer,load_episodes,causal_batch,cosine_alphas,sample_actions
from src.policy.diffusion import diffusion_loss
from src.perception.training import seed_everything
ACTIVE_FOLDER=None

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def weight_sha(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    global ACTIVE_FOLDER
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/policy_v1.json')
    parser.add_argument('--seeds',nargs='+',type=int);parser.add_argument('--conditions',nargs='+');args=parser.parse_args()
    cfg=read(ROOT/args.config);index=read(ROOT/cfg['training_index']);task=read(ROOT/'configs/sim_task_v1.json')
    assert index['model_trained'] is False and index['allowed_npz_arrays']==['rgb','observations','actions']
    assert index['dataset_manifest_sha256']==sha(ROOT/cfg['dataset_manifest'])
    assert torch.cuda.is_available();device=cfg['training']['device'];torch.set_num_threads(4)
    normalizer=fixed_normalizer(task,device)
    validation=load_episodes(ROOT,index['shared_validation'],cfg['image_size'],device)
    alphas=cosine_alphas(cfg['model']['train_diffusion_steps'],device)
    sources={name:path for name,path in [('paired_image_policy.py',ROOT/'paired_image_policy.py'),('train_paired_policy.py',Path(__file__)),
                                        ('policy_config.json',ROOT/args.config),('sim_task_v1.json',ROOT/'configs/sim_task_v1.json'),
                                        ('diffusion.py',PROJECT/'src/policy/diffusion.py'),('training.py',PROJECT/'src/perception/training.py')]}
    source_hashes={name:sha(path) for name,path in sources.items()}
    for condition in args.conditions or cfg['conditions']:
        assert condition in cfg['conditions'];entries=index['train_groups'][condition];assert len(entries)==16
        dataset=load_episodes(ROOT,entries,cfg['image_size'],device)
        for seed in args.seeds or cfg['training']['seeds']:
            assert seed in cfg['training']['seeds']
            folder=ROOT/f"artifacts/runs/{cfg['revision']}/{condition}_seed{seed}"
            if folder.exists():raise FileExistsError(f'Never overwrite an existing run: {folder}')
            folder.mkdir(parents=True);snap=folder/'source_snapshot';snap.mkdir()
            ACTIVE_FOLDER=folder
            for name,path in sources.items():shutil.copy2(path,snap/name)
            seed_everything(seed);model=GoalImageDenoiser(cfg).to(device);initial_sha=weight_sha(model)
            ema=copy.deepcopy(model).eval();settings=cfg['training']
            optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
            idx=torch.Generator(device=device).manual_seed(seed+1100);noise=torch.Generator(device=device).manual_seed(seed+2100)
            aug=torch.Generator(device=device).manual_seed(seed+3100)
            stream=hashlib.sha256();start=time.perf_counter()
            meta={'status':'training','condition':condition,'seed':seed,'config':cfg,
                  'source_sha256':source_hashes,'initial_weights_sha256':initial_sha,
                  'dataset_manifest_sha256':sha(ROOT/cfg['dataset_manifest']),
                  'started_utc':datetime.now(timezone.utc).isoformat(),'normalization':'shared fixed engineering ranges, no val/test statistics',
                  'device':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
                  'selection':'all declared seeds; fixed final EMA; no best-step/checkpoint/seed selection',
                  'policy_inputs':['causal RGB history','robot proprioception','preprovided goal','past effective command'],
                  'no_expert_stage_or_object_truth_input':True}
            write(folder/'manifest.json',meta)
            with (folder/'losses.jsonl').open('w',encoding='utf-8') as log:
                for step in range(1,settings['steps']+1):
                    episodes=torch.randint(len(entries),(settings['batch_size'],),device=device,generator=idx)
                    fractions=torch.rand(settings['batch_size'],device=device,generator=idx)
                    stream.update(episodes.cpu().numpy().tobytes());stream.update(fractions.cpu().numpy().tobytes())
                    rgb,obs,act,mask,starts=causal_batch(dataset,episodes,fractions,normalizer,cfg)
                    brightness=(torch.rand((len(rgb),1,1,1,1),device=device,generator=aug)*2-1)*cfg['augmentation']['brightness']
                    contrast=1+(torch.rand((len(rgb),1,1,1,1),device=device,generator=aug)*2-1)*cfg['augmentation']['contrast']
                    rgb=((rgb-.5)*contrast+.5+brightness).clamp(0,1)
                    condition_vector=model.encode(rgb,obs)
                    loss=diffusion_loss(model,condition_vector,act,mask,alphas,noise)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),settings['grad_clip']);optimizer.step()
                    with torch.no_grad():
                        for average,current in zip(ema.parameters(),model.parameters()):average.lerp_(current,1-cfg['model']['ema_decay'])
                    row={'step':step,'noise_mse':float(loss.detach()),'wall_s':time.perf_counter()-start}
                    log.write(json.dumps(row)+'\n')
                    if step%250==0:log.flush();print(json.dumps({'condition':condition,'seed':seed,**row}),flush=True)
            vg=torch.Generator(device=device).manual_seed(77119)
            episodes=torch.randint(len(index['shared_validation']),(256,),device=device,generator=vg)
            fractions=torch.rand(256,device=device,generator=vg)
            rgb,obs,act,mask,_=causal_batch(validation,episodes,fractions,normalizer,cfg)
            with torch.inference_mode():
                encoded=ema.encode(rgb,obs);loss=float(diffusion_loss(ema,encoded,act,mask,alphas,vg))
                predicted=sample_actions(ema,encoded,alphas,vg,horizon=cfg['prediction_horizon'],steps=cfg['model']['inference_steps'])
                error=(predicted-act)*normalizer['action_scale']
                xyz=float((error[...,:3].norm(dim=-1)*mask).sum()/mask.sum()*1000)
                width=float((error[...,3].abs()*mask).sum()/mask.sum()*1000)
            meta.update(status='completed',steps_completed=settings['steps'],parameters=sum(p.numel() for p in model.parameters()),
                        episode_fraction_stream_sha256=stream.hexdigest(),wall_s=time.perf_counter()-start,
                        completed_utc=datetime.now(timezone.utc).isoformat(),
                        validation={'fixed_sequence_starts':256,'noise_mse':loss,'xyz_error_mm':xyz,'width_error_mm':width,'scope':'heldout teacher command error, not closed-loop task success'})
            for name,path in sources.items():assert sha(path)==source_hashes[name],'Source changed during training'
            payload={'kind':'paired_goal_image_diffusion_v1','config':cfg,'condition':condition,'seed':seed,
                     'ema_state':{k:v.cpu() for k,v in ema.state_dict().items()},
                     'model_state':{k:v.cpu() for k,v in model.state_dict().items()},
                     'normalizer':{k:v.cpu() for k,v in normalizer.items()},'manifest':meta}
            torch.save(payload,folder/'checkpoint.pt');meta['checkpoint_sha256']=sha(folder/'checkpoint.pt');write(folder/'manifest.json',meta)
            print(json.dumps({'completed_run':folder.name,'validation':meta['validation'],'wall_s':meta['wall_s']}),flush=True)
        del dataset


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if ACTIVE_FOLDER and (ACTIVE_FOLDER/'manifest.json').exists():
            failed=read(ACTIVE_FOLDER/'manifest.json');failed.update(status='failed',error=str(exc),failed_utc=datetime.now(timezone.utc).isoformat())
            write(ACTIVE_FOLDER/'manifest.json',failed)
        raise
