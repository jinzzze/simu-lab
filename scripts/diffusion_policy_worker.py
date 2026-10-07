"""Persistent Torch process for frozen RGB perception and learned action diffusion."""
import argparse, base64, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.policy.diffusion import DiffusionPolicy
from src.perception import load_state_predictor
from src.perception.training import seed_everything


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);args=parser.parse_args()
    seed_everything(0)
    policy=DiffusionPolicy(args.checkpoint)
    vision=load_state_predictor(ROOT/policy.config['visual_checkpoint'])
    print(json.dumps({'ready':True,'input_contract':'initial RGB; causal robot proprioception/history; sampling seed'}),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request['command']=='close':break
        if request['command']=='reset':
            shape=tuple(request['shape'])
            if len(shape)!=3 or shape[2]!=3 or min(shape)<=0 or max(shape)>4096:raise ValueError('Invalid image shape')
            image=np.frombuffer(base64.b64decode(request['rgb'],validate=True),dtype=np.uint8).reshape(shape)
            policy.reset(request['sampling_seed'])
            result={'initial_visual_xy_m':vision(image).tolist()}
        elif request['command']=='predict':
            result={'actions':policy.predict(request['history']).tolist()}
        else:raise ValueError('Unknown command')
        print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
