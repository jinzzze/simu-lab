"""One fixed precision-weighted refinement, retaining all prior results."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(ROOT/'configs/policy_recurrent_goal_v4.json')
cfg.update(revision='paired_recurrent_goal_precision_v5',evaluation_script='evaluate_paired_recurrent_precision_policy.py',
    revision_reason='v4 complete development success5/24 versus2/24 and teacher-forced approach XY error6.13/6.41mm; test physical-coordinate precision weighting without changing targets/model/data/controller',
    created_utc=datetime.now(timezone.utc).isoformat(),hypothesis_status='fixed exploratory weighting, unverified; not selected by new-suite outcomes')
cfg['training']['coordinate_loss_weights']=[(.45/.077)**2,(.18/.077)**2,1.,2.]
cfg['training']['objective']='per-episode masked coordinate-weighted MSE; XYZ share77mm physical error unit, width retains twice its original normalized weight'
cfg['training']['budget_comparison']='same architecture, targets, normalizer, optimizer, data, episode stream and2500updates as v4; only coordinate loss weights change; paired equal update/episode budget, actual valid-frame target counts differ between video-guided/simulation conditions'
cfg['model']['type']='same v4 causal128D GRU and goal-relative absolute target decoder; checkpoint architecture kind unchanged'
for name,start in [('nominal',66000),('appearance',66100),('wider_start',66200)]:cfg['evaluation']['suites'][name]['start']=start
for name,start in [('blank_rgb',66300),('swapped_goal_vector',66400)]:cfg['evaluation']['diagnostics'][name]['start']=start
path=ROOT/'configs/policy_recurrent_goal_precision_v5.json';assert not path.exists();write(path,cfg)
out=ROOT/'receipts/paired_recurrent_goal_precision_v5_plan';out.mkdir();shutil.copy2(path,out/path.name)
write(out/'manifest.json',{'frozen_utc':cfg['created_utc'],'config_sha256':sha(path),'old_models_and_all_results_preserved':True,
    'same_architecture_and_target_coordinates_as_v4':True,'only_loss_weights_changed':True,'new_independent_suite_ids_not_seen':True,
    'v4_development_successes':{'simulation_only':5,'video_guided':2},'v4_independent_success_outcomes_not_inspected_before_this_freeze':True})
write(ROOT/'configs/evaluation_addendum_precision_v5.json',{'frozen_utc':cfg['created_utc'],'config_sha256':sha(path),
    'diagnostic_controls':'same scenes and initial images, unmodified RGB/goal, seed7 per condition',
    'goal_corruption_scope':'conflicting numeric input and original image/scoring target, not arbitrary-goal benchmark',
    'cooperation_skipped':cfg['collaboration_skipped']})
text=(ROOT/'train_paired_recurrent_goal_policy.py').read_text(encoding='utf-8')
text=text.replace('train_paired_recurrent_goal_policy.py','train_paired_recurrent_precision_policy.py').replace('configs/policy_recurrent_goal_v4.json','configs/policy_recurrent_goal_precision_v5.json')
old='loss=(((prediction-act).square().mean(-1)*mask).sum(-1)/mask.sum(-1)).mean()';assert old in text
text=text.replace(old,"weights=torch.tensor(settings['coordinate_loss_weights'],device=device)\n                    loss=((((prediction-act).square()*weights).mean(-1)*mask).sum(-1)/mask.sum(-1)).mean()")
text=text.replace("row={'step':step,'normalized_bc_mse':", "row={'step':step,'training_weighted_bc_mse':")
target=ROOT/'train_paired_recurrent_precision_policy.py';assert not target.exists();target.write_text(text,encoding='utf-8')
text=(ROOT/'evaluate_paired_recurrent_goal_policy.py').read_text(encoding='utf-8')
text=text.replace('evaluate_paired_recurrent_goal_policy.py','evaluate_paired_recurrent_precision_policy.py').replace('configs/policy_recurrent_goal_v4.json','configs/policy_recurrent_goal_precision_v5.json').replace('evaluation_addendum_recurrent_goal_v4.json','evaluation_addendum_precision_v5.json')
target=ROOT/'evaluate_paired_recurrent_precision_policy.py';assert not target.exists();target.write_text(text,encoding='utf-8')
text=(ROOT/'run_paired_policy_followup.py').read_text(encoding='utf-8')
text=text.replace("ROOT/EXECUTABLES[cfg['kind']]","ROOT/cfg.get('evaluation_script',EXECUTABLES[cfg['kind']])")
target=ROOT/'run_paired_precision_followup.py';assert not target.exists();target.write_text(text,encoding='utf-8')
print('Frozen precision-only loss refinement; earlier running source files unchanged')
