"""Same causal image model; relative EE actions derived from executed commands."""
import numpy as np
import torch
from paired_image_policy import (ROOT,PROJECT,GoalImageDenoiser,GoalImagePolicy,load_episodes,
                                  causal_batch as absolute_causal_batch,fixed_normalizer as absolute_normalizer,
                                  cosine_alphas,sample_actions)


def fixed_normalizer(task,device):
    result=absolute_normalizer(task,device);step=task['safety_limits']['max_ee_step_m'];width=task['safety_limits']['width_high']
    result['action_center']=torch.tensor([0.,0.,0.,width/2],device=device)
    result['action_scale']=torch.tensor([step,step,step,width/2],device=device)
    return result


def causal_batch(dataset,episodes,fractions,normalizer,config):
    rgb,obs,_,mask,starts=absolute_causal_batch(dataset,episodes,fractions,normalizer,config)
    _,vectors,actions,lengths=dataset
    future=starts[:,None]+torch.arange(config['prediction_horizon'],device=starts.device)[None]
    future=future.minimum(lengths[episodes,None]-1)
    targets=actions[episodes[:,None],future].clone()
    targets[:,:,:3]-=vectors[episodes[:,None],future,:3]
    targets=(targets-normalizer['action_center'])/normalizer['action_scale']
    return rgb,obs,targets,mask,starts


class GoalDeltaPolicy(GoalImagePolicy):
    def __init__(self,checkpoint,device='cuda'):
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if payload['kind']!='paired_goal_image_diffusion_delta_v2':raise ValueError('Expected delta checkpoint')
        self.config=payload['config'];assert self.config['action_representation']=='ee_delta_xyz_absolute_width'
        self.device=device;self.model=GoalImageDenoiser(self.config).to(device).eval();self.model.load_state_dict(payload['ema_state'])
        self.normalizer={k:v.to(device) for k,v in payload['normalizer'].items()}
        self.alphas=cosine_alphas(self.config['model']['train_diffusion_steps'],device)
        self.generator=torch.Generator(device=device);self.reset(0)
