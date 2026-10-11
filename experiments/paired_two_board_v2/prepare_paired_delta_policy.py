"""Freeze a development revision following failed absolute-action smoke."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
config=read(ROOT/'configs/policy_v1a.json')
config.update(revision='paired_delta_image_policy_v2',kind='paired_goal_image_diffusion_delta_v2',
              action_representation='ee_delta_xyz_absolute_width',
              action_semantics='teacher effective XYZ command minus same pre-action robot EE XYZ; future labels use their corresponding future teacher robot state; deployment adds each predicted delta to current measured EE',
              revision_reason='v1a seed7 four-goal development smoke had zero qualified lifts; hypothesis: relative small-step action coordinates reduce stochastic absolute-command drift',
              hypothesis_status='unverified; not a promised improvement',
              created_utc=datetime.now(timezone.utc).isoformat())
config.pop('implementation_repair',None);config.pop('supersedes_failed_execution',None)
config['evaluation']['suites']['nominal']['start']=63000
config['evaluation']['suites']['appearance']['start']=63100
config['evaluation']['suites']['wider_start']['start']=63200
config['evaluation']['diagnostics']['blank_rgb']['start']=63300
config['evaluation']['diagnostics']['swapped_goal_vector']['start']=63400
config['evaluation']['main_suite_gate']='run shared 8-scene validation for all six models first; if either condition has zero successes across all three seeds, retain failure and leave independent suites unexecuted while diagnosing a new revision'
config['normalization']='same shared engineering observation ranges; delta action XYZ scale 8mm per axis, width scale32.5mm; no heldout statistics'
path=ROOT/'configs/policy_delta_v2.json';assert not path.exists();write(path,config)
receipt=ROOT/'receipts/paired_delta_policy_v2_plan';receipt.mkdir();shutil.copy2(path,receipt/'policy_delta_v2.json')
write(receipt/'manifest.json',{'frozen_utc':config['created_utc'],'config_sha256':sha(path),'only_research_change':'absolute versus relative XYZ action parameterization','same_RGB_and_robot_demos':True,
                             'old_development_smoke':'artifacts/runs/paired_image_policy_v1a_evaluation/validation_debug4/simulation_only_seed7',
                             'old_final_models_preserved':True,'no_new_independent_suite_seen':True})
addendum={'frozen_utc':config['created_utc'],'policy_config_sha256':sha(path),
          'diagnostic_controls':'unmodified controls at the same scene/sampling seeds as RGB/goal corruptions; seed7 per condition only',
          'goal_corruption_scope':'numeric goal rotates while original image/score remain commanded Gi; intentional input conflict, not an arbitrary-goal test',
          'collaboration_skipped':config['collaboration_skipped']}
write(ROOT/'configs/evaluation_addendum_delta_v2.json',addendum)
print('Frozen relative-action revision; no extra demonstrations, labels or training budget')
