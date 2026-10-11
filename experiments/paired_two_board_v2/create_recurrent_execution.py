"""Independent per-tick recurrent worker/client/evaluator from audited delta IO."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def transform(source,target,replacements):
    text=(ROOT/source).read_text(encoding='utf-8')
    for old,new in replacements:
        assert old in text,(source,old);text=text.replace(old,new)
    path=ROOT/target;assert not path.exists();path.write_text(text,encoding='utf-8')
transform('paired_policy_worker.py','paired_recurrent_policy_worker.py',[
    ('from paired_image_policy import GoalImagePolicy','from paired_recurrent_policy import GoalRecurrentPolicy'),
    ('policy=GoalImagePolicy(args.checkpoint)','policy=GoalRecurrentPolicy(args.checkpoint)'),
    ("'kind':'paired_goal_image_diffusion_v1'","'kind':'paired_goal_recurrent_bc_v3'"),
    ("'observation_horizon':2","'observation_horizon':1"),('(2,224,224,3)','(1,224,224,3)')])
transform('paired_policy_process.py','paired_recurrent_policy_process.py',[
    ('class ImagePolicyProcess:','class RecurrentPolicyProcess:'),
    ("ROOT/'paired_policy_worker.py'","ROOT/'paired_recurrent_policy_worker.py'"),
    ('(2,224,224,3)','(1,224,224,3)'),('(2,12)','(1,12)'),('(16,4)','(1,4)')])
transform('evaluate_paired_delta_policy.py','evaluate_paired_recurrent_policy.py',[
    ('from paired_delta_policy_process import DeltaPolicyProcess as ImagePolicyProcess','from paired_recurrent_policy_process import RecurrentPolicyProcess as ImagePolicyProcess'),
    ("default='configs/policy_delta_v2.json'","default='configs/policy_recurrent_v3.json'"),
    ('history=[(0,obs),(0,obs)]','history=[(0,obs)]'),('last=history[-2:]','last=history[-1:]'),
    ('learned image diffusion chunks + shared IK/limits; no expert stages or object correction','learned recurrent BC delta + shared IK/limits; no expert stages or object correction'),
    ('two consecutive current/past RGBs at each prediction','current RGB every tick plus causal learned recurrent hidden state'),
    ("names=['evaluate_paired_delta_policy.py','paired_delta_policy_process.py','paired_delta_policy_worker.py','paired_delta_policy.py','paired_delta_env.py','paired_image_policy.py','paired_sim.py','goal_scoring.py','collect_paired_demonstrations.py']",
     "names=['evaluate_paired_recurrent_policy.py','paired_recurrent_policy_process.py','paired_recurrent_policy_worker.py','paired_recurrent_policy.py','paired_delta_policy.py','paired_delta_env.py','paired_image_policy.py','paired_sim.py','goal_scoring.py','collect_paired_demonstrations.py']"),
    ("sources['evaluation_addendum_delta_v2.json']=ROOT/'configs/evaluation_addendum_delta_v2.json'","sources['evaluation_addendum_recurrent_v3.json']=ROOT/'configs/evaluation_addendum_recurrent_v3.json'"),
    ("assert training_meta['source_sha256']['paired_delta_policy.py']==hashes['paired_delta_policy.py']","assert training_meta['source_sha256']['paired_delta_policy.py']==hashes['paired_delta_policy.py']\n            assert training_meta['source_sha256']['paired_recurrent_policy.py']==hashes['paired_recurrent_policy.py']")])
print('Recurrent current-image IO created; no earlier source modified')
