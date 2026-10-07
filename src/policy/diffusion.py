"""Compact conditional temporal U-Net and DDPM training/DDIM inference.

Independent implementation of the action-diffusion formulation; not a copy of
published Diffusion Policy benchmark configurations. No simulator imports.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ResidualFiLM(nn.Module):
    def __init__(self, incoming, outgoing, condition_dim):
        super().__init__()
        self.first = nn.Sequential(nn.Conv1d(incoming, outgoing, 3, padding=1), nn.GroupNorm(8, outgoing), nn.SiLU())
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, outgoing * 2))
        self.second = nn.Sequential(nn.Conv1d(outgoing, outgoing, 3, padding=1), nn.GroupNorm(8, outgoing), nn.SiLU())
        self.skip = nn.Conv1d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, x, condition):
        scale, shift = self.condition(condition).unsqueeze(-1).chunk(2, dim=1)
        return self.second(self.first(x) * (1 + scale) + shift) + self.skip(x)


class ActionDenoiser(nn.Module):
    def __init__(self, observation_dim=22, action_dim=4, widths=(64,128), time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim*2), nn.SiLU(), nn.Linear(time_dim*2, time_dim))
        condition = time_dim + observation_dim
        a, b = widths
        self.encode = ResidualFiLM(action_dim, a, condition)
        self.down = nn.Conv1d(a, a, 4, stride=2, padding=1)
        self.bottleneck = nn.ModuleList([ResidualFiLM(a, b, condition), ResidualFiLM(b, b, condition)])
        self.up = nn.ConvTranspose1d(b, a, 4, stride=2, padding=1)
        self.decode = ResidualFiLM(a*2, a, condition)
        self.output = nn.Conv1d(a, action_dim, 1)

    def forward(self, noisy_actions, timestep, observations):
        half = self.time_dim // 2
        frequencies = torch.exp(torch.arange(half, device=noisy_actions.device) * (-math.log(10000)/(half-1)))
        angles = timestep.float()[:,None] * frequencies[None,:]
        t = self.time_mlp(torch.cat([angles.sin(), angles.cos()], dim=-1))
        condition = torch.cat([t, observations.flatten(1)], dim=-1)
        skip = self.encode(noisy_actions.transpose(1,2), condition)
        x = self.down(skip)
        for block in self.bottleneck:
            x = block(x, condition)
        x = self.decode(torch.cat([self.up(x), skip], dim=1), condition)
        return self.output(x).transpose(1,2)


def cosine_alphas(steps, device):
    time = torch.arange(steps+1, device=device, dtype=torch.float64) / steps
    alpha = torch.cos((time + .008) / 1.008 * math.pi / 2).square()
    beta = (1 - alpha[1:] / alpha[:-1]).clamp(max=.999)
    return torch.cumprod(1-beta, dim=0).float()


def sequence_batch(observations, actions, episodes, starts, obs_horizon=2, horizon=16):
    """Gather causal histories and future command targets within each episode."""
    length = observations.shape[1]
    history = starts[:,None] + torch.arange(1-obs_horizon, 1, device=starts.device)[None,:]
    future = starts[:,None] + torch.arange(horizon, device=starts.device)[None,:]
    obs = observations[episodes[:,None], history.clamp(0, length-1)]
    action = actions[episodes[:,None], future.clamp(0, length-1)]
    mask = (future < length).float()
    return obs, action, mask


def diffusion_loss(model, observations, actions, mask, alphas, generator=None):
    t = torch.randint(len(alphas), (len(actions),), device=actions.device, generator=generator)
    noise = torch.randn(actions.shape, device=actions.device, generator=generator)
    alpha = alphas[t,None,None]
    noisy = alpha.sqrt()*actions + (1-alpha).sqrt()*noise
    error = (model(noisy, t, observations) - noise).square()
    return (error*mask[...,None]).sum() / (mask.sum()*actions.shape[-1])


@torch.inference_mode()
def sample_actions(model, observations, alphas, generator, horizon=16, steps=20):
    if steps < 2 or steps > len(alphas):
        raise ValueError('Invalid number of denoising steps')
    x = torch.randn((len(observations), horizon, 4), device=observations.device, generator=generator)
    schedule = torch.linspace(len(alphas)-1, 0, steps, device=observations.device).round().long()
    for index, t in enumerate(schedule):
        epsilon = model(x, t.expand(len(x)), observations)
        alpha = alphas[t]
        clean = ((x-(1-alpha).sqrt()*epsilon)/alpha.sqrt()).clamp(-1,1)
        previous = alphas[schedule[index+1]] if index+1 < len(schedule) else x.new_tensor(1.)
        x = previous.sqrt()*clean + (1-previous).sqrt()*epsilon
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite denoised actions')
    return x


class DiffusionPolicy:
    def __init__(self, checkpoint, device='cuda'):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload.get('kind') != 'diffusion_policy_d_v1':
            raise ValueError('Unexpected policy checkpoint')
        self.config, self.device = payload['config'], torch.device(device)
        model_cfg = self.config['model']
        self.model = ActionDenoiser(widths=tuple(model_cfg['widths']), time_dim=model_cfg['time_embedding']).to(device)
        self.model.load_state_dict(payload['ema_state']); self.model.eval()
        self.normalizer = {key: value.to(device) for key, value in payload['normalizer'].items()}
        self.alphas = cosine_alphas(model_cfg['train_diffusion_steps'], device)
        self.generator = torch.Generator(device=device)
        self.reset(0)

    def reset(self, sampling_seed):
        self.generator.manual_seed(int(sampling_seed))

    @torch.inference_mode()
    def predict(self, history):
        obs = torch.as_tensor(history, device=self.device, dtype=torch.float32)
        if obs.shape != (2,11) or not torch.isfinite(obs).all():
            raise ValueError('Expected two causal 11D observations')
        normalized = (obs-self.normalizer['obs_center'])/self.normalizer['obs_scale']
        action = sample_actions(self.model, normalized[None], self.alphas, self.generator,
                                horizon=self.config['prediction_horizon'], steps=20)[0]
        return (action*self.normalizer['action_scale']+self.normalizer['action_center']).cpu().numpy()
