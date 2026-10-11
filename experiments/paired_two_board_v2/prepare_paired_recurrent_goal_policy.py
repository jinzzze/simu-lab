"""Freeze an independent action-coordinate refinement before training."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
cfg=read(ROOT/'configs/policy_recurrent_v3.json')
cfg.update(revision='paired_recurrent_goal_policy_v4',kind='paired_goal_relative_recurrent_bc_v4',
    action_representation='goal_relative_xy_absolute_z_width',
    action_semantics='teacher effective absolute command XY minus preprovided goal XY; Z and width absolute; runtime adds numeric goal XY to model XY offset, never current EE or object truth',
    normalization='same engineering observation ranges; action XY goal offsets scale450/180mm, absolute Z center83mm scale77mm, width center/scale32.5mm; no heldout statistics',
    revision_reason='v3 fixed-seed7 first-four development smoke qualified two lifts but zero successes, with first placement center52.4mm off goal; test whether absolute goal-anchored targets reduce cumulative delta drift',
    hypothesis_status='unverified; v3 full validation still running when this refinement is frozen',
    created_utc=datetime.now(timezone.utc).isoformat())
cfg['model']['type']='same 128D causal GRU and 4D tanh head; output goal-relative absolute XY, absolute Z and width'
cfg['training']['objective']='masked MSE on normalized goal-relative absolute XY, absolute Z and desired width'
cfg['training']['budget_comparison']='same architecture, whole episode stream, training steps and data as v3; paired equal budgets within v4; changed coordinate scales/loss weights make this exploratory action-parameterization refinement'
for name,start in [('nominal',65000),('appearance',65100),('wider_start',65200)]:cfg['evaluation']['suites'][name]['start']=start
for name,start in [('blank_rgb',65300),('swapped_goal_vector',65400)]:cfg['evaluation']['diagnostics'][name]['start']=start
path=ROOT/'configs/policy_recurrent_goal_v4.json';assert not path.exists();write(path,cfg)
receipt=ROOT/'receipts/paired_recurrent_goal_policy_v4_plan';receipt.mkdir();shutil.copy2(path,receipt/path.name)
write(receipt/'manifest.json',{'frozen_utc':cfg['created_utc'],'config_sha256':sha(path),'same_40_executed_teacher_episodes':True,
    'initialization_architecture_unchanged_from_v3':True,'old_models_results_preserved':True,'independent_suites_not_seen':True})
write(ROOT/'configs/evaluation_addendum_recurrent_goal_v4.json',{'frozen_utc':cfg['created_utc'],'config_sha256':sha(path),
    'diagnostic_controls':'same scenes and initial images, unmodified RGB/goal, seed7 per condition',
    'goal_corruption_scope':'conflicting numeric input and original image/scoring target, not arbitrary-goal benchmark',
    'cooperation_skipped':cfg['collaboration_skipped']})
print('Frozen goal-relative absolute recurrent action refinement')
