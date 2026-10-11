"""One label-preserving color transform per complete causal RGB episode."""
import torch

def augment_rgb(rgb, config, generator):
    if rgb.ndim != 5 or rgb.shape[2] != 3:
        raise ValueError('Expected batch/time/channel/height/width RGB')
    shape = (len(rgb), 1, 1, 1, 1)
    brightness = (torch.rand(shape, device=rgb.device, generator=generator) * 2 - 1) * config['brightness']
    contrast = 1 + (torch.rand(shape, device=rgb.device, generator=generator) * 2 - 1) * config['contrast']
    gain = 1 + (torch.rand((len(rgb), 1, 3, 1, 1), device=rgb.device, generator=generator) * 2 - 1) * config['channel_gain_halfwidth']
    transformed = (((rgb - .5) * contrast + .5 + brightness) * gain).clamp(0, 1)
    parameters = torch.cat((brightness.flatten(1), contrast.flatten(1), gain.flatten(1)), dim=1)
    return transformed, parameters
