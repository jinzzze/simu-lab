"""Same v6 recurrent BC recipe, dynamic episode count and strict split guards."""
import argparse
import copy
import hashlib
import json
import time
import numpy as np
import torch
from common import ROOT,OLD,PROJECT,read,write,sha,now,snapshot,check_sources,verify_training_pairs
from paired_recurrent_goal_policy import GoalRecurrentModel,episode_batch,load_episodes,fixed_normalizer
from paired_rgb_augmentation import augment_rgb
from src.perception.training import seed_everything


def weight_sha(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--condition');parser.add_argument('--seed',type=int)
    args=parser.parse_args();cfg=read(ROOT/'configs/policy_v1.json')
    index=read(ROOT/cfg['training_index']);dataset=read(ROOT/cfg['dataset_manifest'])
    assert dataset['status']=='completed' and dataset['training_gate_passed']
    assert sha(ROOT/cfg['dataset_manifest'])==index['dataset_manifest_sha256']
    assert cfg['study_sha256']==sha(ROOT/'configs/study_v1.json')
    assert torch.cuda.is_available();torch.set_num_threads(4);device='cuda'
    normalizer=fixed_normalizer(read(OLD/'configs/sim_task_v1.json'),device)
    val=load_episodes(ROOT,index['shared_validation'],cfg['image_size'],device)
    conditions=[args.condition] if args.condition else cfg['conditions']
    seeds=[args.seed] if args.seed is not None else cfg['training']['seeds']
    assert set(conditions)<=set(cfg['conditions']) and set(seeds)<=set(cfg['training']['seeds'])
    paths=[ROOT/n for n in ('train.py','common.py','protocol.json','configs/study_v1.json','configs/policy_v1.json')]
    paths += [OLD/n for n in ('paired_recurrent_goal_policy.py','paired_recurrent_policy.py','paired_delta_policy.py',
                              'paired_image_policy.py','paired_rgb_augmentation.py','configs/sim_task_v1.json')]
    paths += [PROJECT/'src/perception/training.py',PROJECT/'src/policy/diffusion.py']
    for condition in conditions:
        entries=index['train_groups'][condition];verify_training_pairs(entries)
        assert len(entries)==cfg['training']['episodes_per_condition']==24
        data=load_episodes(ROOT,entries,cfg['image_size'],device)
        for seed in seeds:
            folder=ROOT/f'artifacts/models/{condition}_seed{seed}'
            if folder.exists():
                old=read(folder/'manifest.json')
                assert old['status']=='completed' and old['config']==cfg
                assert old['checkpoint_sha256']==sha(folder/'checkpoint.pt')
                check_sources(old['source_sha256']);continue
            folder.mkdir(parents=True)
            hashes=snapshot(folder,paths)
            seed_everything(seed);model=GoalRecurrentModel(cfg).to(device);ema=copy.deepcopy(model).eval()
            settings=cfg['training'];optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
            idx=torch.Generator(device=device).manual_seed(seed+1100);aug=torch.Generator(device=device).manual_seed(seed+3100)
            stream=hashlib.sha256();augstream=hashlib.sha256();valid_targets=0;started=time.perf_counter();counts=np.zeros(len(entries),np.int64)
            meta=dict(status='training',condition=condition,seed=seed,config=cfg,started_utc=now(),source_sha256=hashes,
                dataset_manifest_sha256=sha(ROOT/cfg['dataset_manifest']),training_index_sha256=sha(ROOT/cfg['training_index']),
                initial_weights_sha256=weight_sha(model),parameters=sum(p.numel() for p in model.parameters()),
                torch=str(torch.__version__),device=torch.cuda.get_device_name(0),
                selection='fixed final EMA; all seeds retained',personal_timing_used=condition=='video_timing_24',
                loss_weighting='mean over valid steps per episode, then mean over episodes')
            write(folder/'manifest.json',meta)
            try:
                with (folder/'losses.jsonl').open('w',encoding='utf-8') as log:
                    for step in range(1,settings['steps']+1):
                        episodes=torch.randint(len(entries),(settings['batch_size'],),device=device,generator=idx)
                        ids=episodes.cpu().numpy();stream.update(ids.tobytes());np.add.at(counts,ids,1)
                        rgb,obs,act,mask=episode_batch(data,episodes,normalizer)
                        rgb,parameters=augment_rgb(rgb,cfg['augmentation'],aug);augstream.update(parameters.detach().cpu().numpy().tobytes())
                        prediction,_=model(rgb,obs,None)
                        weights=torch.tensor(settings['coordinate_loss_weights'],device=device)
                        loss=((((prediction-act).square()*weights).mean(-1)*mask).sum(-1)/mask.sum(-1)).mean()
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite loss')
                        optimizer.zero_grad(set_to_none=True);loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),settings['grad_clip']);optimizer.step()
                        with torch.no_grad():
                            for a,b in zip(ema.parameters(),model.parameters()):
                                a.lerp_(b,1-cfg['model']['ema_decay'])
                        valid_targets+=int(mask.sum())
                        row=dict(step=step,weighted_bc_mse=float(loss.detach()),wall_s=time.perf_counter()-started)
                        log.write(json.dumps(row)+'\n')
                        if step%250==0:
                            log.flush();print(json.dumps(dict(condition=condition,seed=seed,**row)),flush=True)
                errors=[]
                with torch.inference_mode():
                    for i in range(len(index['shared_validation'])):
                        rgb,obs,act,mask=episode_batch(val,torch.tensor([i],device=device),normalizer)
                        pred,_=ema(rgb,obs,None);diff=(pred-act)*normalizer['action_scale']
                        errors.append(float((diff[:,:,:3].norm(dim=-1)*mask).sum()/mask.sum()*1000))
                check_sources(hashes)
                meta.update(status='completed',completed_utc=now(),steps_completed=settings['steps'],
                    valid_action_targets_seen=valid_targets,episode_sampling_counts=counts.tolist(),
                    episode_stream_sha256=stream.hexdigest(),augmentation_stream_sha256=augstream.hexdigest(),
                    validation_teacher_forced_xyz_mm=float(np.mean(errors)),wall_s=time.perf_counter()-started,
                    validation_scope='teacher-forced action error on12 developer trajectories; not closed-loop success')
                payload=dict(kind='paired_goal_relative_recurrent_bc_v4',config=cfg,condition=condition,seed=seed,
                    ema_state={k:v.cpu() for k,v in ema.state_dict().items()},model_state={k:v.cpu() for k,v in model.state_dict().items()},
                    normalizer={k:v.cpu() for k,v in normalizer.items()},manifest=meta)
                torch.save(payload,folder/'checkpoint.pt');meta['checkpoint_sha256']=sha(folder/'checkpoint.pt');write(folder/'manifest.json',meta)
                print(json.dumps({'completed':folder.name,'validation_xyz_mm':float(np.mean(errors)),'wall_s':meta['wall_s']}),flush=True)
            except Exception as exc:
                meta.update(status='failed',error=str(exc));write(folder/'manifest.json',meta);raise
        del data


if __name__=='__main__':
    main()
