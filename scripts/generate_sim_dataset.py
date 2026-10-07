"""Generate independently seeded simulator RGB/XY pairs; privileged labels only here."""
from pathlib import Path
import sys,json,hashlib,time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sim.grasp_env import GraspEnv


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg_path=ROOT/'configs/sim_dataset.json';cfg=json.loads(cfg_path.read_text(encoding='utf-8-sig'))
    scene_config=ROOT/'configs/grasp_sim.json';scene_hash=digest(scene_config)
    manifest={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'config':cfg,'config_sha256':digest(cfg_path),
        'sim_config_sha256':scene_hash,'sim_source_sha256':digest(ROOT/'src/sim/grasp_env.py'),
        'scope':'privileged labels for supervised adaptation, not policy observations','scenes':[]}
    for split,(low,high) in cfg['seed_ranges'].items():
        for seed in range(low,high):manifest['scenes'].append({'scene_seed':seed,'split':split})
    assert len({x['scene_seed'] for x in manifest['scenes']})==len(manifest['scenes'])
    manifest_path=ROOT/'data/manifests/sim_adaptation_v1.json'
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    images=[];targets=[];uvs=[];seeds=[];splits=[]
    diag=ROOT/'artifacts/diagnostics/sim_dataset';diag.mkdir(parents=True,exist_ok=True)
    env=GraspEnv(image_size=cfg['image_size'])
    try:
        for idx,record in enumerate(manifest['scenes']):
            obs=env.reset(record['scene_seed'])
            label=env.get_object_label()
            images.append(obs['rgb']);targets.append(label['xy_m']);uvs.append(label['uv_normalized'])
            seeds.append(record['scene_seed']);splits.append(record['split'])
            if idx==0:
                preview=cv2.cvtColor(obs['rgb'],cv2.COLOR_RGB2BGR)
                point=np.rint(np.asarray(label['uv_normalized'])*cfg['image_size']).astype(int)
                cv2.circle(preview,tuple(point),5,(0,0,255),1)
                cv2.imwrite(str(diag/'train_scene_10000_label.jpg'),preview)
            if (idx+1)%64==0:print(f'Generated {idx+1}/{len(manifest["scenes"])} scenes',flush=True)
    finally:env.close()
    assert digest(scene_config)==scene_hash,'Simulator configuration changed mid-generation'
    out=ROOT/'data/processed/sim_adaptation_v1.npz'
    np.savez_compressed(out,images=np.asarray(images,np.uint8),target_xy_m=np.asarray(targets,np.float32),
                        object_uv_normalized=np.asarray(uvs,np.float32),scene_seeds=np.asarray(seeds,np.int64),split=np.asarray(splits))
    manifest['dataset_sha256']=digest(out);manifest['count']=len(images)
    manifest['image_shape']=list(np.asarray(images).shape)
    manifest['split_counts']={key:splits.count(key) for key in cfg['seed_ranges']}
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'count':len(images),'split_counts':manifest['split_counts'],'dataset_sha256':manifest['dataset_sha256']}))

if __name__=='__main__':main()
