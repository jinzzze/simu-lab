"""Goal-conditioned image/action diffusion. Torch-only; no simulator import."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT))
from src.policy.diffusion import ActionDenoiser,cosine_alphas,sample_actions


class GoalImageDenoiser(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        channels=[3,16,32,64];blocks=[]
        for incoming,outgoing in zip(channels[:-1],channels[1:]):
            blocks.extend([nn.Conv2d(incoming,outgoing,5,2,2),nn.GroupNorm(8,outgoing),nn.SiLU()])
        if config['image_size']!=128:raise ValueError('Fixed deterministic 16-to-4 spatial pool requires 128px input')
        self.image_encoder=nn.Sequential(*blocks,nn.AvgPool2d(4,4),nn.Flatten(),nn.Linear(64*4*4,64),nn.SiLU())
        model=config['model']
        self.denoiser=ActionDenoiser(observation_dim=config['observation_horizon']*(64+12),
                                     widths=tuple(model['widths']),time_dim=model['time_embedding'])

    def encode(self,images,vectors):
        b,h=images.shape[:2]
        features=self.image_encoder(images.flatten(0,1)).reshape(b,h,64)
        return torch.cat([features,vectors],dim=-1)

    def forward(self,noisy,t,condition):
        return self.denoiser(noisy,t,condition)


def fixed_normalizer(task,device):
    limits=task['safety_limits'];xyz_lo=limits['xyz_low'];xyz_hi=limits['xyz_high']
    gx,gy=task['target_grid_top_left_xy_m'];w=task['board_grid_width_m'];height=task['board_grid_height_m']
    goal_lo=[gx,gy-height];goal_hi=[gx+w,gy]
    lo=xyz_lo+[0.]+goal_lo+[goal_lo[0]-xyz_hi[0],goal_lo[1]-xyz_hi[1]]+xyz_lo+[0.]
    hi=xyz_hi+[limits['width_high']]+goal_hi+[goal_hi[0]-xyz_lo[0],goal_hi[1]-xyz_lo[1]]+xyz_hi+[limits['width_high']]
    result={}
    for name,low,high in [('obs',lo,hi),('action',xyz_lo+[0.],xyz_hi+[limits['width_high']])]:
        low=torch.tensor(low,device=device,dtype=torch.float32);high=torch.tensor(high,device=device,dtype=torch.float32)
        result[name+'_center']=(low+high)/2;result[name+'_scale']=(high-low)/2
    return result


def load_episodes(root,entries,image_size,device):
    """Only allowed arrays; diagnostic metadata is never loaded into training."""
    raw=[]
    for item in entries:
        path=Path(root)/item['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['dataset_sha256']:raise ValueError('Dataset hash changed')
        with np.load(path,allow_pickle=False) as data:
            rgb=data['rgb'].copy();obs=data['observations'].copy();action=data['actions'].copy()
        n=len(action)
        if rgb.shape!=(n,224,224,3) or obs.shape!=(n,12) or action.shape!=(n,4):raise ValueError('Invalid input arrays')
        raw.append((rgb,obs,action))
    longest=max(len(a) for _,_,a in raw);count=len(raw)
    images=torch.empty((count,longest,3,image_size,image_size),device=device,dtype=torch.uint8)
    observations=torch.empty((count,longest,12),device=device);actions=torch.empty((count,longest,4),device=device)
    lengths=[]
    for index,(rgb,obs,act) in enumerate(raw):
        n=len(act);lengths.append(n)
        for start in range(0,n,128):
            part=torch.from_numpy(rgb[start:start+128]).to(device).permute(0,3,1,2).float()
            resized=F.interpolate(part,size=(image_size,image_size),mode='bilinear',align_corners=False)
            images[index,start:start+len(part)]=resized.round().clamp(0,255).to(torch.uint8)
        observations[index,:n]=torch.from_numpy(obs).to(device);actions[index,:n]=torch.from_numpy(act).to(device)
        images[index,n:]=images[index,n-1];observations[index,n:]=observations[index,n-1];actions[index,n:]=actions[index,n-1]
    return images,observations,actions,torch.tensor(lengths,device=device,dtype=torch.long)


def causal_batch(dataset,episodes,fractions,normalizer,config):
    images,observations,actions,lengths=dataset
    starts=(fractions*lengths[episodes]).long().minimum(lengths[episodes]-1)
    history=starts[:,None]+torch.arange(1-config['observation_horizon'],1,device=starts.device)[None]
    future=starts[:,None]+torch.arange(config['prediction_horizon'],device=starts.device)[None]
    history=history.clamp_min(0).minimum(lengths[episodes,None]-1)
    valid=future<lengths[episodes,None];future=future.minimum(lengths[episodes,None]-1)
    rgb=images[episodes[:,None],history].float()/255
    obs=(observations[episodes[:,None],history]-normalizer['obs_center'])/normalizer['obs_scale']
    act=(actions[episodes[:,None],future]-normalizer['action_center'])/normalizer['action_scale']
    return rgb,obs,act,valid.float(),starts


class GoalImagePolicy:
    def __init__(self,checkpoint,device='cuda'):
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if payload['kind']!='paired_goal_image_diffusion_v1':raise ValueError('Unexpected checkpoint')
        self.config=payload['config'];self.device=device
        self.model=GoalImageDenoiser(self.config).to(device).eval();self.model.load_state_dict(payload['ema_state'])
        self.normalizer={k:v.to(device) for k,v in payload['normalizer'].items()}
        self.alphas=cosine_alphas(self.config['model']['train_diffusion_steps'],device)
        self.generator=torch.Generator(device=device);self.reset(0)

    def reset(self,seed):self.generator.manual_seed(int(seed))

    @torch.inference_mode()
    def predict(self,images,vectors,blank_rgb=False):
        h=self.config['observation_horizon'];size=self.config['image_size']
        images=np.asarray(images);vectors=np.asarray(vectors,dtype=np.float32)
        if images.shape!=(h,224,224,3) or images.dtype!=np.uint8 or vectors.shape!=(h,12) or not np.isfinite(vectors).all():raise ValueError('Invalid causal RGB/vector history')
        rgb=torch.from_numpy(images.copy()).to(self.device).permute(0,3,1,2).float()
        rgb=F.interpolate(rgb,size=(size,size),mode='bilinear',align_corners=False).round().clamp(0,255)/255
        if blank_rgb:rgb=rgb*0+.5
        obs=torch.from_numpy(vectors.copy()).to(self.device)
        obs=(obs-self.normalizer['obs_center'])/self.normalizer['obs_scale']
        condition=self.model.encode(rgb[None],obs[None])
        action=sample_actions(self.model,condition,self.alphas,self.generator,
                              horizon=self.config['prediction_horizon'],steps=self.config['model']['inference_steps'])[0]
        return (action*self.normalizer['action_scale']+self.normalizer['action_center']).cpu().numpy()
