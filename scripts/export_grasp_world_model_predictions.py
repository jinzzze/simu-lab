"""Recover predictions from saved final weights without any optimizer steps."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.perception.training import seed_everything
from src.world_model.numpy_model import features
from src.world_model.integrity import (load_training_data, check_ensemble_anchors, relative_hashes, validate_predictions)
OUT=ROOT/'artifacts/runs/grasp_world_model_v1'


def main():
    metadata=json.loads((OUT/'manifest.json').read_text())
    cfg=metadata['plan'];data,receipt=load_training_data(ROOT,metadata)
    check_ensemble_anchors(ROOT,OUT,metadata)
    with np.load(OUT/'ensemble.npz',allow_pickle=False) as packed:
        ensemble={key:packed[key].copy() for key in packed.files}
    checked=[OUT/'manifest.json',OUT/'ensemble.npz']
    seed_everything(0);x=torch.from_numpy(features(data['initial_xy_m'],data['command_xy_m']))
    deltas=[];probs=[];runs=[];curves=[]
    for seed in cfg['training']['seeds']:
        folder=OUT/f'seed{seed}'
        payload=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
        model=nn.Sequential(nn.Linear(4,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,4))
        model.load_state_dict(payload['state_dict']);model.eval()
        if ensemble['seeds'].tolist()!=cfg['training']['seeds'] or ensemble['delta_scale_m']!=np.asarray(cfg['model']['delta_scale_m'],dtype=ensemble['delta_scale_m'].dtype):
            raise ValueError('Ensemble seed order or scale changed')
        for layer in (0,2,4):
            for short,long in [('w','weight'),('b','bias')]:
                if not np.array_equal(payload['state_dict'][f'{layer}.{long}'].cpu().numpy(),ensemble[f's{seed}_{short}{layer}']):
                    raise ValueError('Checkpoint disagrees with anchored ensemble; outputs preserved')
        checked.extend([folder/'checkpoint.pt',folder/'manifest.json',folder/'losses.jsonl'])
        with torch.inference_mode():output=model(x)
        deltas.append(output[:,:2].numpy()*cfg['model']['delta_scale_m'])
        probs.append(output[:,2:].sigmoid().numpy())
        run=json.loads((folder/'manifest.json').read_text());assert run==payload['manifest']
        runs.append(run)
        curves.append([json.loads(line)['total'] for line in (folder/'losses.jsonl').read_text().splitlines()])
    deltas=np.stack(deltas);probs=np.stack(probs)
    validate_predictions(deltas,probs,(len(cfg['training']['seeds']),len(data['split']),2))
    receipt.update(relative_hashes(ROOT,checked))
    np.savez_compressed(OUT/'torch_predictions.npz',delta=deltas,probabilities=probs)
    receipt.update(relative_hashes(ROOT,[OUT/'torch_predictions.npz']))
    state={'verified_artifacts':receipt,'metadata':metadata,'per_seed':runs,'curves':curves,
           'recovery':'Saved final checkpoints re-evaluated with data and ensemble integrity checks; zero additional optimizer steps',
           'recovery_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'run_state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    print('Recovered predictions from all three preserved checkpoints; no retraining')


if __name__=='__main__':main()
