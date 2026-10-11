"""Causal GRU with goal-relative absolute XY targets, not integrated deltas."""
import numpy as np
import torch
from torch.nn import functional as F
from paired_recurrent_policy import ROOT,PROJECT,GoalRecurrentModel,load_episodes
from paired_image_policy import fixed_normalizer as observation_normalizer


def fixed_normalizer(task,device):
    result=observation_normalizer(task,device)
    # Fixed engineering ranges, independent of all trajectory statistics.
    result['action_center']=torch.tensor([0.,0.,.083,.0325],device=device)
    result['action_scale']=torch.tensor([.45,.18,.077,.0325],device=device)
    return result


def episode_batch(dataset,episodes,normalizer):
    images,vectors,actions,lengths=dataset
    rgb=images[episodes].float()/255;raw=vectors[episodes];target=actions[episodes].clone()
    target[:,:,:2]-=raw[:,:,4:6]
    target=(target-normalizer['action_center'])/normalizer['action_scale']
    obs=(raw-normalizer['obs_center'])/normalizer['obs_scale']
    mask=(torch.arange(images.shape[1],device=images.device)[None]<lengths[episodes,None]).float()
    return rgb,obs,target,mask


def decode_target(normalized,vectors,normalizer):
    action=normalized*normalizer['action_scale']+normalizer['action_center']
    action=action.clone();action[...,:2]+=vectors[...,4:6]
    return action


class GoalRelativeRecurrentPolicy:
    def __init__(self,checkpoint,device='cuda'):
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if payload['kind']!='paired_goal_relative_recurrent_bc_v4':raise ValueError('Unexpected goal-relative checkpoint')
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
        raw=torch.from_numpy(vectors.copy()).to(self.device);obs=(raw-self.normalizer['obs_center'])/self.normalizer['obs_scale']
        prediction,self.hidden=self.model(rgb[None],obs[None],self.hidden)
        return decode_target(prediction[0],raw,self.normalizer).cpu().numpy()
