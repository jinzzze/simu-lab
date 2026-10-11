"""Stateful causal image imitation; no external phase or episode clock."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from paired_delta_policy import ROOT,PROJECT,fixed_normalizer,load_episodes
from paired_image_policy import GoalImageDenoiser


class GoalRecurrentModel(nn.Module):
    def __init__(self,config):
        super().__init__();self.image_encoder=GoalImageDenoiser(config).image_encoder
        self.rnn=nn.GRU(64+12,128,batch_first=True)
        self.head=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,4),nn.Tanh())

    def forward(self,images,vectors,hidden=None):
        b,t=images.shape[:2];feature=self.image_encoder(images.flatten(0,1)).reshape(b,t,64)
        output,hidden=self.rnn(torch.cat([feature,vectors],dim=-1),hidden)
        return self.head(output),hidden


def episode_batch(dataset,episodes,normalizer):
    images,vectors,actions,lengths=dataset
    rgb=images[episodes].float()/255;raw=vectors[episodes];act=actions[episodes].clone()
    obs=(raw-normalizer['obs_center'])/normalizer['obs_scale']
    act[:,:,:3]-=raw[:,:,:3];act=(act-normalizer['action_center'])/normalizer['action_scale']
    mask=(torch.arange(images.shape[1],device=images.device)[None]<lengths[episodes,None]).float()
    return rgb,obs,act,mask


class GoalRecurrentPolicy:
    def __init__(self,checkpoint,device='cuda'):
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if payload['kind']!='paired_goal_recurrent_bc_v3':raise ValueError('Unexpected recurrent checkpoint')
        self.config=payload['config'];self.device=device
        self.model=GoalRecurrentModel(self.config).to(device).eval();self.model.load_state_dict(payload['ema_state'])
        self.normalizer={k:v.to(device) for k,v in payload['normalizer'].items()};self.reset(0)

    def reset(self,seed):self.hidden=None

    @torch.inference_mode()
    def predict(self,images,vectors,blank_rgb=False):
        images=np.asarray(images);vectors=np.asarray(vectors,np.float32)
        if images.shape!=(1,224,224,3) or images.dtype!=np.uint8 or vectors.shape!=(1,12) or not np.isfinite(vectors).all():raise ValueError('Invalid current RGB/vector')
        rgb=torch.from_numpy(images.copy()).to(self.device).permute(0,3,1,2).float()
        rgb=F.interpolate(rgb,(128,128),mode='bilinear',align_corners=False).round().clamp(0,255)/255
        if blank_rgb:rgb=rgb*0+.5
        obs=torch.from_numpy(vectors.copy()).to(self.device);obs=(obs-self.normalizer['obs_center'])/self.normalizer['obs_scale']
        action,self.hidden=self.model(rgb[None],obs[None],self.hidden)
        return (action[0]*self.normalizer['action_scale']+self.normalizer['action_center']).cpu().numpy()
