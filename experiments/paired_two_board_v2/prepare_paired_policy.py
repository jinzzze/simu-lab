"""Freeze image-policy training/evaluation before any learned-policy run."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    path=ROOT/'configs/policy_v1.json';assert not path.exists(),'Preserve frozen plan'
    manifest=ROOT/'artifacts/reports/paired_robot_demos_v1/manifest.json'
    assert read(manifest)['status']=='completed'
    config={'revision':'paired_image_policy_v1','kind':'paired_goal_image_diffusion_v1',
            'scope':'exploratory matched simulation-only versus offline video-guided robot imitation; not human 3D recovery or original A/B/C/D ablation',
            'training_index':'artifacts/reports/paired_robot_demos_v1/training_index.json',
            'dataset_manifest':'artifacts/reports/paired_robot_demos_v1/manifest.json','dataset_manifest_sha256':sha(manifest),
            'conditions':['simulation_only','video_guided'],'image_size':128,'observation_horizon':2,
            'prediction_horizon':16,'execution_horizon':4,'vector_dim':12,'action_dim':4,
            'normalization':'fixed shared physical ranges from sim_task_v1; no heldout data statistics',
            'model':{'rgb_encoder':'three stride-2 5x5 CNNs, GroupNorm/SiLU, spatial 4x4 pooling, 64D feature',
                     'widths':[64,128],'time_embedding':64,'train_diffusion_steps':100,'inference_steps':20,
                     'ema_decay':.995,'pretrained_weights':None,'initialization':'same full model initial weights for paired seed'},
            'augmentation':{'brightness':.04,'contrast':.04,'geometry':False,'paired_history_consistent':True},
            'training':{'seeds':[7,17,27],'steps':6000,'batch_size':64,'device':'cuda','learning_rate':.0001,
                        'weight_decay':.000001,'grad_clip':1.,'optimizer':'AdamW',
                        'sampling':'same episode/fraction stream across conditions, each episode uniformly weighted',
                        'selection':'all fixed final EMA checkpoints; validation reporting only, no best seed or checkpoint'},
            'evaluation':{'episode_steps':384,'sampling_seed':8821,
                          'validation':{'start':61000,'count':8,'purpose':'closed-loop development check on shared validation scene IDs'},
                          'suites':{'nominal':{'start':62000,'count':16,'jitter_m':.003},
                                    'appearance':{'start':62100,'count':8,'jitter_m':.003,'object_rgba':[.08,.08,.08,1.],'table_rgba':[.78,.83,.87,1.]},
                                    'wider_start':{'start':62200,'count':8,'jitter_m':.012}},
                          'diagnostics':{'blank_rgb':{'start':62300,'count':4,'seeds':[7]},
                                         'swapped_goal_vector':{'start':62400,'count':4,'seeds':[7]}},
                          'scoring':'same strict designated-goal physics; fixed horizon, scorer does not terminate/repair policy',
                          'goals':'pairs cycled P1G1 through P4G4; not arbitrary/unseen goal claims'},
            'collaboration_skipped':['S02/S03/S04 applicant recordings','human strict footprint outcome adjudication','real board-to-robot registration','actual capture timestamps','external publication and application submission'],
            'created_utc':datetime.now(timezone.utc).isoformat(),'status':'frozen before policy training or evaluation'}
    write(path,config)
    receipt=ROOT/'receipts/paired_policy_v1_plan';receipt.mkdir()
    shutil.copy2(path,receipt/'policy_v1.json')
    write(receipt/'manifest.json',{'frozen_utc':config['created_utc'],'policy_config_sha256':sha(path),
                                 'dataset_manifest_sha256':sha(manifest),'previous_data_scope':'40 executed teacher demos, no new policy result at freeze'})
    print('Frozen six-run paired image policy plan',flush=True)


if __name__=='__main__':main()
