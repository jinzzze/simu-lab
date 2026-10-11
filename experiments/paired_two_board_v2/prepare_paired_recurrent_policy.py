"""Exploratory architecture revision; new independent simulation suites."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
cfg=read(ROOT/'configs/policy_delta_v2.json')
cfg.update(revision='paired_recurrent_policy_v3',kind='paired_goal_recurrent_bc_v3',
           observation_horizon=1,prediction_horizon=1,execution_horizon=1,
           revision_reason='short-history diffusion development rollouts showed missed grasp/phase transitions; evaluate deterministic causal recurrent imitation as a separate architecture',
           created_utc=datetime.now(timezone.utc).isoformat())
cfg['model'].update(type='128D GRU over current 64D RGB encoding and 12D robot/goal input; tanh 4D delta/width head',hidden_dim=128)
cfg['training'].update(steps=2500,batch_size=2,learning_rate=.0003,
                       sampling='uniform whole episodes; all valid timestep targets; zero hidden state per episode',
                       objective='masked MSE on normalized current EE delta and desired gripper width')
cfg['training']['budget_comparison']='equal within the two v3 conditions; full-episode recurrent BC differs in targets/compute from diffusion revisions, no equal-budget cross-architecture claim'
cfg['memory']='learned recurrent hidden state is updated from past/current inputs; it can encode temporal information, but no absolute clock or expert phase is supplied'
for name,start in [('nominal',64000),('appearance',64100),('wider_start',64200)]:cfg['evaluation']['suites'][name]['start']=start
for name,start in [('blank_rgb',64300),('swapped_goal_vector',64400)]:cfg['evaluation']['diagnostics'][name]['start']=start
path=ROOT/'configs/policy_recurrent_v3.json';assert not path.exists();write(path,cfg)
receipt=ROOT/'receipts/paired_recurrent_policy_v3_plan';receipt.mkdir();shutil.copy2(path,receipt/'policy_recurrent_v3.json')
write(receipt/'manifest.json',{'frozen_utc':cfg['created_utc'],'config_sha256':sha(path),'same_40_executed_teacher_episodes':True,
                             'exploratory_architecture_revision':True,'old_models_results_preserved':True,'no_v3_independent_suite_seen':True})
write(ROOT/'configs/evaluation_addendum_recurrent_v3.json',{'frozen_utc':cfg['created_utc'],'config_sha256':sha(path),
      'diagnostic_controls':'same scenes and initial images with unmodified RGB/goal inputs; seed7 per condition',
      'goal_corruption_scope':'conflicting numeric input and original image/scoring target, not an arbitrary-goal benchmark',
      'cooperation_skipped':cfg['collaboration_skipped']})
print('Frozen causal recurrent BC architecture; all earlier runs retained')
