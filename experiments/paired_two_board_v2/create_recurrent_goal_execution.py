"""Create separate executable copies, preserving all earlier frozen sources."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def generate(source,target,replacements):
    text=(ROOT/source).read_text(encoding='utf-8')
    for old,new in replacements:
        assert old in text,(source,old);text=text.replace(old,new)
    path=ROOT/target;assert not path.exists();path.write_text(text,encoding='utf-8')

generate('train_paired_recurrent_policy.py','train_paired_recurrent_goal_policy.py',[
    ('from paired_recurrent_policy import','from paired_recurrent_goal_policy import'),
    ('configs/policy_recurrent_v3.json','configs/policy_recurrent_goal_v4.json'),
    ("('train_paired_recurrent_policy.py','paired_recurrent_policy.py','paired_delta_policy.py','paired_image_policy.py')","('train_paired_recurrent_goal_policy.py','paired_recurrent_goal_policy.py','paired_recurrent_policy.py','paired_delta_policy.py','paired_image_policy.py')"),
    ("'delta_xyz_error_mm'","'absolute_xyz_error_mm'"),
    ('paired_goal_recurrent_bc_v3','paired_goal_relative_recurrent_bc_v4')])
generate('paired_recurrent_policy_worker.py','paired_recurrent_goal_policy_worker.py',[
    ('from paired_recurrent_policy import GoalRecurrentPolicy','from paired_recurrent_goal_policy import GoalRelativeRecurrentPolicy'),
    ('GoalRecurrentPolicy(args.checkpoint)','GoalRelativeRecurrentPolicy(args.checkpoint)'),
    ('paired_goal_recurrent_bc_v3','paired_goal_relative_recurrent_bc_v4')])
generate('paired_recurrent_policy_process.py','paired_recurrent_goal_policy_process.py',[
    ('RecurrentPolicyProcess','GoalRelativeRecurrentPolicyProcess'),
    ('paired_recurrent_policy_worker.py','paired_recurrent_goal_policy_worker.py')])
generate('evaluate_paired_recurrent_policy.py','evaluate_paired_recurrent_goal_policy.py',[
    ('from paired_delta_env import DeltaGoalEnv as PairedGoalEnv','from paired_sim import PairedGoalEnv'),
    ('from paired_recurrent_policy_process import RecurrentPolicyProcess as ImagePolicyProcess','from paired_recurrent_goal_policy_process import GoalRelativeRecurrentPolicyProcess as ImagePolicyProcess'),
    ('configs/policy_recurrent_v3.json','configs/policy_recurrent_goal_v4.json'),
    ("['evaluate_paired_recurrent_policy.py','paired_recurrent_policy_process.py','paired_recurrent_policy_worker.py','paired_recurrent_policy.py'","['evaluate_paired_recurrent_goal_policy.py','paired_recurrent_goal_policy_process.py','paired_recurrent_goal_policy_worker.py','paired_recurrent_goal_policy.py','paired_recurrent_policy.py'"),
    ('evaluation_addendum_recurrent_v3.json','evaluation_addendum_recurrent_goal_v4.json'),
    ("            assert training_meta['source_sha256']['paired_recurrent_policy.py']==hashes['paired_recurrent_policy.py']", "            assert training_meta['source_sha256']['paired_recurrent_policy.py']==hashes['paired_recurrent_policy.py']\n            assert training_meta['source_sha256']['paired_recurrent_goal_policy.py']==hashes['paired_recurrent_goal_policy.py']"),
    ("requested_delta.append(command.copy());requested.append(np.r_[before_vector[:3]+command[:3],command[3]])","requested_delta.append(command.copy());requested.append(command.copy())"),
    ("'predicted_delta_actions':np.asarray(requested_delta,np.float32),","'decoded_absolute_actions':np.asarray(requested_delta,np.float32),"),
    ('learned recurrent BC delta + shared IK/limits','learned causal recurrent goal-anchored absolute targets + shared IK/limits'),
    ("action_representation='ee_delta_xyz_absolute_width'","action_representation='goal_relative_xy_absolute_z_width; decoded absolute in predicted_chunks'")])
print('Created independent v4 trainer/worker/evaluator without editing v1-v3')
