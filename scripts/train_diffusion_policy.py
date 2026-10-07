"""Train all predeclared policy seeds; final EMA only, no checkpoint selection."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, hashlib, json, sys, time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.policy.diffusion import ActionDenoiser, cosine_alphas, sequence_batch, diffusion_loss, sample_actions
from src.perception.training import seed_everything


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_data(cfg, manifest, split, device):
    path = ROOT / f'data/processed/diffusion_policy_v1/{split}.npz'
    if sha(path) != manifest['splits'][split]['dataset_sha256']:
        raise ValueError('Demonstration dataset hash changed')
    with np.load(path, allow_pickle=False) as data:
        obs = torch.from_numpy(data['observations'].copy()).to(device)
        actions = torch.from_numpy(data['actions'].copy()).to(device)
        expected = np.arange(cfg['splits'][split]['start'], cfg['splits'][split]['start']+cfg['splits'][split]['count'])
        assert np.array_equal(data['scene_seeds'], expected)
    assert obs.shape == (len(expected), cfg['episode_steps'], 11)
    assert actions.shape == (len(expected), cfg['episode_steps'], 4)
    assert torch.isfinite(obs).all() and torch.isfinite(actions).all()
    return obs, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int)
    args = parser.parse_args()
    cfg = read(ROOT / 'configs/diffusion_policy.json')
    manifest = read(ROOT / 'data/manifests/diffusion_policy_v1.json')
    assert manifest['status'] == 'completed' and manifest['plan'] == cfg
    settings = cfg['training']; device = settings['device']
    assert torch.cuda.is_available(), 'The predeclared training device is CUDA'
    torch.set_num_threads(4)
    raw_obs, raw_actions = load_data(cfg, manifest, 'train', device)
    val_obs, val_actions = load_data(cfg, manifest, 'val', device)
    normalizer = {}
    for name, data in [('obs',raw_obs), ('action',raw_actions)]:
        flat = data.flatten(0,1); low, high = flat.min(dim=0).values, flat.max(dim=0).values
        normalizer[name+'_center'] = (low+high)/2
        normalizer[name+'_scale'] = ((high-low)/2).clamp_min(1e-6)
    obs = (raw_obs-normalizer['obs_center'])/normalizer['obs_scale']
    actions = (raw_actions-normalizer['action_center'])/normalizer['action_scale']
    vobs = (val_obs-normalizer['obs_center'])/normalizer['obs_scale']
    vactions = (val_actions-normalizer['action_center'])/normalizer['action_scale']
    alphas = cosine_alphas(cfg['model']['train_diffusion_steps'], device)
    out = ROOT / 'artifacts/runs/diffusion_policy_v1'; out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT/'src/policy/diffusion.py', ROOT/'configs/diffusion_policy.json', ROOT/'src/perception/training.py']
    seed_list = args.seeds or settings['seeds']
    assert all(seed in settings['seeds'] for seed in seed_list)
    for seed in seed_list:
        folder = out / f'DP_seed{seed}'
        if folder.exists():
            raise FileExistsError(f'Preserve existing training run: {folder}')
        folder.mkdir()
        metadata = {'status': 'training', 'seed': seed, 'config': cfg,
                    'started_utc': datetime.now(timezone.utc).isoformat(),
                    'source_sha256': {str(p.relative_to(ROOT)):sha(p) for p in sources},
                    'dataset_manifest_sha256':sha(ROOT/'data/manifests/diffusion_policy_v1.json'),
                    'dataset_sha256': {s:manifest['splits'][s]['dataset_sha256'] for s in cfg['splits']},
                    'selection':'fixed final EMA model, no best-step selection', 'torch':str(torch.__version__)}
        write(folder/'manifest.json', metadata)
        seed_everything(seed)
        model = ActionDenoiser(widths=tuple(cfg['model']['widths']), time_dim=cfg['model']['time_embedding']).to(device)
        ema = copy.deepcopy(model).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
        generator = torch.Generator(device=device).manual_seed(seed+2100)
        index_generator = torch.Generator(device=device).manual_seed(seed+1100)
        stream_hash = hashlib.sha256(); start=time.perf_counter()
        with (folder/'losses.jsonl').open('w', encoding='utf-8') as log:
            for step in range(1,settings['steps']+1):
                ep = torch.randint(len(obs), (settings['batch_size'],), device=device, generator=index_generator)
                t = torch.randint(cfg['episode_steps'], (settings['batch_size'],), device=device, generator=index_generator)
                stream_hash.update(torch.stack([ep,t],dim=1).cpu().numpy().tobytes())
                o,a,m = sequence_batch(obs, actions, ep,t,cfg['observation_horizon'],cfg['prediction_horizon'])
                loss = diffusion_loss(model,o,a,m,alphas,generator)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite diffusion training loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),settings['grad_clip']);optimizer.step()
                with torch.no_grad():
                    for average, current in zip(ema.parameters(),model.parameters()):
                        average.lerp_(current,1-cfg['model']['ema_decay'])
                row={'step':step,'noise_mse':float(loss.detach()),'elapsed_seconds':time.perf_counter()-start}
                log.write(json.dumps(row)+'\n')
                if step%250==0:
                    log.flush();print(f'DP seed={seed} step={step}/{settings["steps"]} loss={row["noise_mse"]:.5f} seconds={row["elapsed_seconds"]:.1f}',flush=True)
        # Validation is reported only after the final checkpoint; no adaptive selection.
        vg=torch.Generator(device=device).manual_seed(77119)
        ep=torch.randint(len(vobs),(256,),device=device,generator=vg)
        t=torch.randint(cfg['episode_steps'],(256,),device=device,generator=vg)
        vo,va,vm=sequence_batch(vobs,vactions,ep,t)
        with torch.inference_mode():
            validation_loss=float(diffusion_loss(ema,vo,va,vm,alphas,vg))
            predicted=sample_actions(ema,vo,alphas,vg)
            difference=(predicted-va)*normalizer['action_scale']
            position=(difference[...,:3].norm(dim=-1)*vm).sum()/vm.sum()*1000
            width=(difference[...,3].abs()*vm).sum()/vm.sum()*1000
        metadata.update({'status':'completed','steps_completed':settings['steps'],
                         'parameters':sum(p.numel() for p in model.parameters()),
                         'elapsed_seconds':time.perf_counter()-start,'batch_sequence_sha256':stream_hash.hexdigest(),
                         'validation':{'noise_mse':validation_loss,'action_xyz_mean_error_mm':float(position),
                                       'action_width_mean_absolute_error_mm':float(width),'sequence_starts':256}})
        for path in sources:
            assert sha(path)==metadata['source_sha256'][str(path.relative_to(ROOT))], 'Source changed during training'
        payload={'kind':'diffusion_policy_d_v1','config':cfg,'seed':seed,
                 'ema_state':{k:v.cpu() for k,v in ema.state_dict().items()},
                 'model_state':{k:v.cpu() for k,v in model.state_dict().items()},
                 'normalizer':{k:v.cpu() for k,v in normalizer.items()},'manifest':metadata}
        torch.save(payload,folder/'checkpoint.pt')
        metadata['checkpoint_sha256']=sha(folder/'checkpoint.pt')
        write(folder/'manifest.json',metadata)
        print(json.dumps({'seed':seed,'validation':metadata['validation']}),flush=True)


if __name__=='__main__':main()
