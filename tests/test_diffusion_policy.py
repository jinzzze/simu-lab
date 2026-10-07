"""Critical sequence causality and diffusion inference properties (synthetic fixtures)."""
import torch
from src.policy.diffusion import ActionDenoiser, cosine_alphas, sequence_batch, sample_actions


def test_no_future_observations_or_episode_crossing():
    obs=torch.arange(2*5*11).reshape(2,5,11).float()
    act=torch.arange(2*5*4).reshape(2,5,4).float()
    o,a,mask=sequence_batch(obs,act,torch.tensor([0,1]),torch.tensor([0,4]))
    assert torch.equal(o[0,0],obs[0,0]) and torch.equal(o[0,1],obs[0,0])
    assert torch.equal(o[1,0],obs[1,3]) and torch.equal(o[1,1],obs[1,4])
    assert torch.equal(a[0,4],act[0,4]) and torch.equal(a[0,5],act[0,4])
    assert mask[0].sum()==5 and mask[1].sum()==1
    assert torch.equal(a[1,0],act[1,4])


def test_diffusion_schedule_and_seeded_sampling():
    torch.set_num_threads(2);torch.manual_seed(8)
    model=ActionDenoiser().eval();alpha=cosine_alphas(100,'cpu')
    assert (alpha>0).all() and (alpha<1).all() and (alpha[1:]<alpha[:-1]).all()
    obs=torch.zeros(2,2,11)
    x=sample_actions(model,obs,alpha,torch.Generator().manual_seed(99),steps=4)
    y=sample_actions(model,obs,alpha,torch.Generator().manual_seed(99),steps=4)
    assert x.shape==(2,16,4) and torch.isfinite(x).all()
    assert torch.equal(x,y) and x.abs().max()<=1.00001
    changed=sample_actions(model,obs+1,alpha,torch.Generator().manual_seed(99),steps=4)
    assert not torch.equal(x,changed), 'Denoising must depend on observations'
