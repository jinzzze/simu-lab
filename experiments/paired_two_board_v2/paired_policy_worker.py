"""Persistent Torch-only worker receiving only causal RGB/robot/goal history."""
import argparse,base64,hashlib,json,sys
import numpy as np
import torch
from paired_image_policy import GoalImagePolicy
from src.perception.training import seed_everything


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);args=parser.parse_args()
    seed_everything(0);torch.set_num_threads(4);policy=GoalImagePolicy(args.checkpoint)
    print(json.dumps({'ready':True,'kind':'paired_goal_image_diffusion_v1','observation_horizon':2}),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request['command']=='close':break
        if request['command']=='reset':policy.reset(request['sampling_seed']);reply={'reset':True}
        elif request['command']=='predict':
            shape=tuple(request['shape']);assert shape==(2,224,224,3)
            blob=base64.b64decode(request['rgb'],validate=True);images=np.frombuffer(blob,dtype=np.uint8).reshape(shape)
            vectors=np.asarray(request['vectors'],np.float32)
            actions=policy.predict(images,vectors,blank_rgb=request.get('blank_rgb',False))
            reply={'actions':actions.tolist(),'received_rgb_sha256':hashlib.sha256(blob).hexdigest(),
                   'received_vector_sha256':hashlib.sha256(vectors.tobytes()).hexdigest()}
        else:raise ValueError('Unknown command')
        print(json.dumps(reply,allow_nan=False),flush=True)


if __name__=='__main__':main()
